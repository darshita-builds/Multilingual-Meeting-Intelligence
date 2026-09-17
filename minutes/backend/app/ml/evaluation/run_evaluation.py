"""Reproducible evaluation entry point.

    python -m backend.app.ml.evaluation.run_evaluation
    MOM_EXTRACTOR_BACKEND=transformer python -m backend.app.ml.evaluation.run_evaluation
    python -m backend.app.ml.evaluation.run_evaluation \
        --segmenter-backend embedding --extractor-backend transformer

Backend selection follows the platform's own convention (`ml/registry.py`):
read from `MOM_SEGMENTER_BACKEND`/`MOM_EXTRACTOR_BACKEND`, optionally
overridden by CLI flags for convenience when comparing arms in one sitting.
Running with no flags and no env override scores the `baseline` arm -- the
required control -- with zero extra installs.

Writes `evaluation/results.json` (machine-readable) and `evaluation/report.md`
(human-readable) at the repository root, and logs summary metrics to MLflow if
it is installed (`mlflow_tracking.py`; a no-op otherwise).

Data provenance -- see `docs/ml-evaluation.md` for the full picture:

  * `fixtures/meetings/*.txt`             SYNTHETIC (authored, not real meetings)
  * `evaluation/data/gold_annotations.json`   MANUALLY ANNOTATED by Person A
    (single annotator) over that synthetic text -- span-level decisions/
    actions, not just the expected *counts* `fixtures/meetings/DATASET.md` carries.
  * `evaluation/data/synthetic_wer_pairs.json`  SYNTHETIC (hand-written ASR
    error simulation, not real ASR output -- no real speech exists in this
    repository to transcribe).

No REAL (recorded-meeting) data is used anywhere in this evaluation. Numbers
here demonstrate the metrics are implemented correctly and give a first,
honest read on the extractor's behaviour on the corpus that was used to
calibrate it -- they are not a generalisation estimate. See the report's own
"Limitations" section.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.ml import registry as ml_registry
from backend.app.ml.evaluation.action_metrics import (
    ActionItemGold,
    ActionItemPrediction,
    evaluate_actions,
)
from backend.app.ml.evaluation.decision_metrics import (
    DecisionGold,
    DecisionPrediction,
    evaluate_decisions,
)
from backend.app.ml.evaluation.error_analysis import analyse_actions, analyse_decisions
from backend.app.ml.evaluation.language_breakdown import aggregate_span_metrics_by_language
from backend.app.ml.evaluation.wer import corpus_word_error_rate, normalize_for_wer, word_error_rate
from backend.app.ml.mlflow_tracking import tracking_run

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = Path(__file__).resolve().parent / "data"
FIXTURES_DIR = ROOT / "fixtures" / "meetings"
OUTPUT_DIR = ROOT / "evaluation"


def _bind_backends(segmenter_backend: str | None, extractor_backend: str | None) -> None:
    if segmenter_backend:
        os.environ["MOM_SEGMENTER_BACKEND"] = segmenter_backend
    if extractor_backend:
        os.environ["MOM_EXTRACTOR_BACKEND"] = extractor_backend
    ml_registry.reset_cache()


def _micro_average(rows: list[dict]) -> dict:
    tp = sum(r["true_positives"] for r in rows)
    fp = sum(r["false_positives"] for r in rows)
    fn = sum(r["false_negatives"] for r in rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_extraction(segmenter: Any, extractor: Any, gold_data: dict) -> dict:
    decision_rows, action_rows = [], []
    all_decision_errors, all_action_errors = [], []
    per_scenario_detail = []

    for slug, scenario_gold in gold_data["scenarios"].items():
        text = (FIXTURES_DIR / f"{slug}.txt").read_text(encoding="utf-8")
        seg_result = segmenter.segment(text)
        ext_result = extractor.extract(seg_result.blocks)

        gold_decisions = [DecisionGold(text=d["text"]) for d in scenario_gold["decisions"]]
        gold_actions = [
            ActionItemGold(
                text=a["text"], owner_name=a.get("owner_name"), deadline=a.get("deadline")
            )
            for a in scenario_gold["actions"]
        ]
        pred_decisions = [DecisionPrediction(text=d.text) for d in ext_result.decisions]
        pred_actions = [
            ActionItemPrediction(text=a.text, owner_name=a.owner_name, deadline=a.deadline)
            for a in ext_result.actions
        ]

        d_match = evaluate_decisions(pred_decisions, gold_decisions)
        a_metrics = evaluate_actions(pred_actions, gold_actions)

        language = scenario_gold["language"]
        decision_rows.append({"language": language, **d_match.as_dict()})
        action_rows.append({"language": language, **a_metrics.span.as_dict()})

        all_decision_errors += analyse_decisions(slug, pred_decisions, gold_decisions, d_match)
        all_action_errors += analyse_actions(slug, pred_actions, gold_actions, a_metrics.span)

        confidences = [b.confidence for b in seg_result.blocks]
        per_scenario_detail.append(
            {
                "scenario": slug,
                "language": language,
                "segmentation": {
                    "blocks": len(seg_result.blocks),
                    "mean_confidence": (
                        round(statistics.fmean(confidences), 3) if confidences else None
                    ),
                },
                "decisions": d_match.as_dict(),
                "actions": a_metrics.as_dict(),
            }
        )

    return {
        "segmenter": getattr(segmenter, "name", "?"),
        "extractor": getattr(extractor, "name", "?"),
        "per_scenario": per_scenario_detail,
        "decisions": {
            "overall": _micro_average(decision_rows),
            "by_language": aggregate_span_metrics_by_language(decision_rows),
        },
        "actions": {
            "overall": _micro_average(action_rows),
            "by_language": aggregate_span_metrics_by_language(action_rows),
        },
        "errors": {
            "decisions": [e.as_dict() for e in all_decision_errors],
            "actions": [e.as_dict() for e in all_action_errors],
        },
    }


def evaluate_wer() -> dict:
    pairs_data = json.loads((DATA_DIR / "synthetic_wer_pairs.json").read_text(encoding="utf-8"))
    per_pair, normalised_pairs = [], []
    for pair in pairs_data["pairs"]:
        ref_n = normalize_for_wer(pair["reference"])
        hyp_n = normalize_for_wer(pair["hypothesis"])
        result = word_error_rate(ref_n, hyp_n)
        per_pair.append({"id": pair["id"], "error_type": pair["error_type"], **result.as_dict()})
        normalised_pairs.append((ref_n, hyp_n))
    corpus = corpus_word_error_rate(normalised_pairs)
    return {
        "provenance": pairs_data["provenance"],
        "per_pair": per_pair,
        "corpus": corpus.as_dict(),
    }


def _comparison_section(results: dict) -> list[str]:
    baseline = results.get("baseline_comparison")
    if baseline is None:
        return []
    x = results["extraction"]
    lines = [
        "## Baseline comparison",
        "",
        "The rule-based baseline (`baseline.py`) scored on the same gold data, as the "
        "required control arm for this comparison study:",
        "",
        "| Arm | Segmenter | Extractor | Decision F1 | Action F1 |",
        "|---|---|---|---|---|",
        f"| Configured | `{x['segmenter']}` | `{x['extractor']}` | "
        f"{x['decisions']['overall']['f1']} | {x['actions']['overall']['f1']} |",
        f"| Baseline | `{baseline['segmenter']}` | `{baseline['extractor']}` | "
        f"{baseline['decisions']['overall']['f1']} | {baseline['actions']['overall']['f1']} |",
        "",
        "Both arms are scored against the same manually-annotated gold set on the same "
        "8-scenario corpus (the calibration-set caveat in Limitations applies to both).",
        "",
    ]
    return lines


def _render_report(results: dict) -> str:
    x = results["extraction"]
    w = results["wer"]
    lines = [
        "# ML evaluation report",
        "",
        f"Generated {results['generated_at']} by "
        "`python -m backend.app.ml.evaluation.run_evaluation`. Do not hand-edit -- regenerate.",
        "",
        "## Data provenance",
        "",
        "| Source | Kind |",
        "|---|---|",
        "| `fixtures/meetings/*.txt` | SYNTHETIC -- authored corpus, "
        "see `fixtures/meetings/DATASET.md` |",
        "| `evaluation/data/gold_annotations.json` | MANUALLY ANNOTATED "
        "(single annotator: Person A) over the synthetic corpus |",
        "| `evaluation/data/synthetic_wer_pairs.json` | SYNTHETIC -- hand-written "
        "ASR error simulation, not real ASR output |",
        "",
        "No REAL (recorded-meeting) data is used anywhere in this report. "
        "See **Limitations** below before citing any number here as an accuracy claim.",
        "",
        "## Backends under evaluation",
        "",
        f"- Segmenter: `{x['segmenter']}`",
        f"- Extractor: `{x['extractor']}`",
        "",
    ]
    lines += _comparison_section(results)
    lines += [
        "## Decision extraction",
        "",
        f"Overall (micro-averaged over {len(x['per_scenario'])} scenarios): "
        f"P={x['decisions']['overall']['precision']} "
        f"R={x['decisions']['overall']['recall']} "
        f"F1={x['decisions']['overall']['f1']} "
        f"(tp={x['decisions']['overall']['true_positives']}, "
        f"fp={x['decisions']['overall']['false_positives']}, "
        f"fn={x['decisions']['overall']['false_negatives']})",
        "",
        "By language:",
        "",
        "| Language | Scenarios | P | R | F1 |",
        "|---|---|---|---|---|",
    ]
    for lang, row in x["decisions"]["by_language"].items():
        lines.append(
            f"| {lang} | {row['scenarios']} | {row['precision']} | {row['recall']} | {row['f1']} |"
        )

    lines += [
        "",
        "## Action extraction",
        "",
        f"Overall: P={x['actions']['overall']['precision']} "
        f"R={x['actions']['overall']['recall']} "
        f"F1={x['actions']['overall']['f1']} "
        f"(tp={x['actions']['overall']['true_positives']}, "
        f"fp={x['actions']['overall']['false_positives']}, "
        f"fn={x['actions']['overall']['false_negatives']})",
        "",
        "By language:",
        "",
        "| Language | Scenarios | P | R | F1 |",
        "|---|---|---|---|---|",
    ]
    for lang, row in x["actions"]["by_language"].items():
        lines.append(
            f"| {lang} | {row['scenarios']} | {row['precision']} | {row['recall']} | {row['f1']} |"
        )

    lines += [
        "",
        "Owner/deadline slot accuracy, restricted to correctly-matched action items "
        "(see each scenario in `results.json` for the per-scenario breakdown; "
        "`null` means no gold item in that scenario had that slot to check):",
        "",
        "| Scenario | Owner accuracy | Deadline accuracy |",
        "|---|---|---|",
    ]
    for row in x["per_scenario"]:
        oa = row["actions"].get("owner_accuracy")
        da = row["actions"].get("deadline_accuracy")
        oa_display = oa if oa is not None else "—"
        da_display = da if da is not None else "—"
        lines.append(f"| {row['scenario']} | {oa_display} | {da_display} |")

    lines += [
        "",
        "## Agenda segmentation (descriptive)",
        "",
        "No boundary-level gold annotations exist yet (`segmentation_metrics.py` is ready for "
        "them once they do -- see Limitations). Reporting block counts and mean confidence per "
        "scenario as a sanity check instead of inventing reference boundaries.",
        "",
        "| Scenario | Blocks | Mean confidence |",
        "|---|---|---|",
    ]
    for row in x["per_scenario"]:
        seg = row["segmentation"]
        lines.append(f"| {row['scenario']} | {seg['blocks']} | {seg['mean_confidence']} |")

    lines += [
        "",
        "## Word Error Rate (SYNTHETIC pairs -- not a WhisperSTT accuracy measurement)",
        "",
        f"Corpus WER (micro-averaged): **{w['corpus']['wer']}** "
        f"({w['corpus']['substitutions']} sub, {w['corpus']['deletions']} del, "
        f"{w['corpus']['insertions']} ins over {w['corpus']['reference_words']} reference words)",
        "",
        "| Pair | Error type | WER |",
        "|---|---|---|",
    ]
    for pair in w["per_pair"]:
        lines.append(f"| {pair['id']} | {pair['error_type']} | {pair['wer']} |")

    lines += ["", "## Error analysis", ""]
    error_sections = (
        ("Decisions", x["errors"]["decisions"]),
        ("Actions", x["errors"]["actions"]),
    )
    for item_type, errors in error_sections:
        lines.append(f"### {item_type}")
        lines.append("")
        if not errors:
            lines.append("None.")
        else:
            for e in errors:
                detail = f" -- {e['detail']}" if e["detail"] else ""
                lines.append(f'- **{e["kind"]}** ({e["scenario"]}): "{e["text"]}"{detail}')
        lines.append("")

    lines += [
        "## Limitations",
        "",
        "- **Thresholds were calibrated against this same 8-scenario corpus** (there is no "
        "separate dev/test split -- the corpus is too small to hold one out meaningfully). "
        "Numbers above show the method works and give an honest first read, not a "
        "generalisation estimate. A held-out, independently-annotated set is future work.",
        "- **Single annotator.** `gold_annotations.json` has no inter-annotator agreement "
        "measure. A second annotator pass is listed as future work in `docs/ml-evaluation.md`.",
        "- **No real speech or real meeting transcripts anywhere in this evaluation.** WER is "
        "demonstrated on synthetic pairs only; STT accuracy on real audio is unmeasured.",
        "- **English-dominant corpus.** Only one scenario is meaningfully code-mixed; the "
        "language-wise breakdown above should not be read as parity across five languages.",
        "- Marathi, Bengali and Gujarati have cue dictionaries (`multilingual_cues.py`) but no "
        "fixture, gold annotation or measured recall at all -- they are structurally supported, "
        "not evaluated. No accuracy claim is made for them anywhere in this project.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Person A's ML evaluation suite.")
    parser.add_argument("--segmenter-backend", choices=["baseline", "embedding"], default=None)
    parser.add_argument("--extractor-backend", choices=["baseline", "transformer"], default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to write results.json/report.md (default: evaluation/ at the repo root). "
        "Tests point this at a tmp_path so a test run never overwrites the committed report.",
    )
    args = parser.parse_args(argv)

    _bind_backends(args.segmenter_backend, args.extractor_backend)

    gold_data = json.loads((DATA_DIR / "gold_annotations.json").read_text(encoding="utf-8"))
    segmenter = ml_registry.get_segmenter()
    extractor = ml_registry.get_extractor()
    extraction_results = evaluate_extraction(segmenter, extractor, gold_data)

    # The blueprint requires a baseline comparison. If the configured extractor
    # is not itself the baseline, always also score the rule-based baseline on
    # the same gold data, so every report includes the comparison rather than
    # requiring a second manual run.
    baseline_comparison = None
    if not extractor.name.startswith("baseline"):
        from backend.app.ml.baseline import BaselineAgendaSegmenter, BaselineExtractor

        baseline_comparison = evaluate_extraction(
            BaselineAgendaSegmenter(), BaselineExtractor(), gold_data
        )

    results = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "backends": ml_registry.active_backends(),
        "extraction": extraction_results,
        "baseline_comparison": baseline_comparison,
        "wer": evaluate_wer(),
    }

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_render_report(results), encoding="utf-8")

    decision_f1 = results["extraction"]["decisions"]["overall"]["f1"]
    action_f1 = results["extraction"]["actions"]["overall"]["f1"]
    wer_value = results["wer"]["corpus"]["wer"]

    with tracking_run(
        "evaluation", params={"segmenter": segmenter.name, "extractor": extractor.name}
    ) as tracked:
        tracked.metrics(
            {"decision_f1": decision_f1, "action_f1": action_f1, "synthetic_wer": wer_value}
        )
        tracked.artifact(str(output_dir / "results.json"))
        tracked.artifact(str(output_dir / "report.md"))

    print(f"Wrote {output_dir / 'results.json'} and {output_dir / 'report.md'}")
    print(f"Decision F1: {decision_f1}   Action F1: {action_f1}   Synthetic WER: {wer_value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
