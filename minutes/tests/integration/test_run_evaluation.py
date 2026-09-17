"""End-to-end test of the evaluation CLI. Runs the baseline arm only --
no ML dependency install, no model download -- and writes to a tmp_path so a
test run never overwrites the committed evaluation/results.json and
evaluation/report.md.
"""

from __future__ import annotations

import json

from backend.app.ml import registry as ml_registry
from backend.app.ml.evaluation.run_evaluation import main


def test_run_evaluation_baseline_arm_produces_valid_output(tmp_path, monkeypatch):
    monkeypatch.setenv("MOM_SEGMENTER_BACKEND", "baseline")
    monkeypatch.setenv("MOM_EXTRACTOR_BACKEND", "baseline")
    ml_registry.reset_cache()

    exit_code = main(
        [
            "--segmenter-backend",
            "baseline",
            "--extractor-backend",
            "baseline",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0

    results_path = tmp_path / "results.json"
    report_path = tmp_path / "report.md"
    assert results_path.exists()
    assert report_path.exists()

    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["extraction"]["segmenter"] == "baseline-agenda-v1"
    assert results["extraction"]["extractor"] == "baseline-extractor-v1"
    # both arms are the baseline here, so no separate comparison is computed
    assert results["baseline_comparison"] is None

    # sanity: every scenario in the gold set was actually scored
    per_scenario = results["extraction"]["per_scenario"]
    assert len(per_scenario) == 8
    assert {row["scenario"] for row in per_scenario} == {
        "01_clean_english_standup",
        "02_code_mixed_hindi_english",
        "03_pii_heavy_onboarding",
        "04_prompt_injection_planning",
        "05_no_decisions_social",
        "06_hedged_ambiguous",
        "07_disfluent_noisy",
        "08_long_quarterly_review",
    }

    # metrics are well-formed probabilities, not placeholders
    overall = results["extraction"]["decisions"]["overall"]
    assert 0.0 <= overall["precision"] <= 1.0
    assert 0.0 <= overall["recall"] <= 1.0
    assert 0.0 <= overall["f1"] <= 1.0

    # the empty-meeting scenario must not be scored as containing anything
    empty_scenario = next(r for r in per_scenario if r["scenario"] == "05_no_decisions_social")
    assert empty_scenario["decisions"]["true_positives"] == 0
    assert empty_scenario["actions"]["true_positives"] == 0

    # WER section ran against the synthetic pairs and produced a real number
    assert 0.0 <= results["wer"]["corpus"]["wer"] <= 5.0

    report_text = report_path.read_text(encoding="utf-8")
    assert "SYNTHETIC" in report_text
    assert "MANUALLY ANNOTATED" in report_text
    assert "Limitations" in report_text

    ml_registry.reset_cache()
