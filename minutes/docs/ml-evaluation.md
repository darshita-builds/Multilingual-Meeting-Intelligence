# ML evaluation

How Person A's models (`backend/app/ml/whisper_stt.py`, `pyannote_diarizer.py`,
`embedding_segmenter.py`, `transformer_extractor.py`) are evaluated, what data
backs the numbers, and what those numbers do and do not prove. Read this
alongside [`fixtures/meetings/DATASET.md`](../fixtures/meetings/DATASET.md)
(provenance of the transcripts themselves) and
[`docs/architecture.md`](architecture.md) (the ML seam / contract).

## Reproduce it

```powershell
pip install -r requirements-ml.txt      # once; installs sentence-transformers etc.
python scripts/make_fixtures.py         # generates fixtures/meetings/ if not already present
python -m backend.app.ml.evaluation.run_evaluation --segmenter-backend embedding --extractor-backend transformer
```

Writes `evaluation/results.json` (machine-readable) and `evaluation/report.md`
(human-readable, includes a baseline comparison and an error-analysis
section) at the repository root. Running with no flags scores whatever
`MOM_SEGMENTER_BACKEND`/`MOM_EXTRACTOR_BACKEND` are set to in the environment
(default `baseline`, which needs no extra installs at all).

## Data inventory -- REAL vs SYNTHETIC vs MANUALLY ANNOTATED

| Data | Kind | Where |
|---|---|---|
| Meeting transcripts (8 scenarios) | **SYNTHETIC** -- authored for this project, no real meeting | `fixtures/meetings/*.txt` |
| `.wav` files | **SYNTHETIC, non-representative** -- a 2-second placeholder tone, not speech | `fixtures/meetings/*.wav` |
| Expected decision/action *counts* | Author's own count when each scenario was written, not independently annotated | `fixtures/meetings/manifest.json` |
| Span-level decision/action gold | **MANUALLY ANNOTATED** by Person A (single annotator), over the synthetic transcripts above | `backend/app/ml/evaluation/data/gold_annotations.json` |
| WER reference/hypothesis pairs | **SYNTHETIC** -- hand-written to simulate documented ASR error patterns, not real ASR output | `backend/app/ml/evaluation/data/synthetic_wer_pairs.json` |

**No REAL (recorded-meeting) data exists anywhere in this repository or its
evaluation**, and none is used. See `fixtures/meetings/DATASET.md` for why
that is a deliberate choice (consent, redistribution, reproducibility), not an
oversight.

## Annotation protocol

Span-level gold (`gold_annotations.json`) extends the count-only protocol
already documented in `fixtures/meetings/DATASET.md`:

* a **decision** is annotated when a sentence states a settled outcome (a
  commitment, not a proposal or a hedge -- "maybe we might" is never a
  decision);
* an **action** is annotated when a sentence assigns work to be done; `owner`
  and `deadline` are recorded only when the sentence states them
  unambiguously as a concrete name or calendar reference. A vague trigger
  phrase ("before the cutover") is deliberately left `null` even though the
  sentence mentions *something* -- it is not a date, and inventing one to fill
  the field would defeat the point of the anti-hallucination contract this
  whole project enforces (`evidence_quote` / "never invent owners or
  deadlines", see `backend/app/ml/base.py` and `docs/architecture.md`);
* the four adversarial sentences in `04_prompt_injection_planning` are never
  annotated as content, matching the platform's own injection-defence stance.

**This is single-annotator data.** No inter-annotator agreement measure
exists. For a solo capstone track this is a known, accepted limitation of the
"Data package" deliverable rather than a gap that was missed -- closing it
needs a second annotator, which is out of scope for one person. It is listed
under Limitations below and in every generated report so it cannot be missed
or silently relied on as if it were stronger evidence than it is.

## Metrics

| Metric | Module | Notes |
|---|---|---|
| Word Error Rate | `evaluation/wer.py` | Word-level Levenshtein alignment, implemented from scratch (no `jiwer` dependency). Demonstrated on **synthetic** pairs only -- see below. |
| Decision extraction | `evaluation/decision_metrics.py` | Precision/recall/F1 over Jaccard-matched spans (`extraction_matching.py`). Accuracy is not meaningful here -- see the module docstring for why. |
| Action extraction | `evaluation/action_metrics.py` | Same span matching, plus owner/deadline slot accuracy restricted to matched true positives. |
| Segmentation | `evaluation/segmentation_metrics.py` | Boundary precision/recall/F1 with a character-offset tolerance window -- a deliberate simplification of Pk/WindowDiff, not full TextTiling evaluation, because no boundary-level gold exists yet (see Limitations). The module is ready to use the moment that gold exists. |
| Language breakdown | `evaluation/language_breakdown.py` | Micro-averaged (counts summed before dividing) precision/recall/F1 per detected language, applied to both decisions and actions. |
| Error analysis | `evaluation/error_analysis.py` | Typed, per-item error records (false positive / false negative / wrong owner / wrong deadline) with the offending text -- not just aggregate numbers. |

### On WER specifically

`fixtures/meetings/` has no real speech (the corpus card says so explicitly).
`run_evaluation.py` therefore computes WER on
`evaluation/data/synthetic_wer_pairs.json` -- six hand-written
(reference, simulated-hypothesis) pairs, each labelled with the ASR error
pattern it simulates (substitution, deletion, insertion, code-mixed script
confusion, digit drop). This proves the WER implementation is correct. **It is
not, and must never be cited as, a measurement of WhisperSTT's real-world
accuracy.** That measurement needs real recordings and real Whisper output,
which do not exist in this project yet.

## Calibration methodology -- and its limitation

`EmbeddingAgendaSegmenter`'s boundary threshold and `TransformerExtractor`'s
two similarity thresholds (`MOM_EXTRACTOR_THRESHOLD_HIGH` /
`_LOW`) were tuned by running both models against the 8-scenario fixture
corpus and adjusting until the extracted decisions/actions matched the
manually-annotated gold set. Two consequences worth being explicit about:

1. **There is no held-out test set.** Eight scenarios is too small to split
   meaningfully into calibration and evaluation subsets, so the numbers in
   `evaluation/report.md` are measured on the same data the thresholds were
   tuned against. Treat them as *"the method works and behaves as designed"*,
   not as a generalisation estimate. A larger, independently-annotated corpus
   (ideally with real recordings) would be needed to report a genuine
   held-out number, and is listed as future work below.
2. **The calibration surfaced and fixed two real bugs**, which is exactly
   what calibration against real fixtures is for: a tie-break rule that let a
   sentence with an explicit decision cue ("we have decided...") get
   classified as an action because its action-prototype similarity happened
   to be marginally higher, and an English-only action-cue list that missed
   verbs outside the original narrow whitelist ("run" in "will run a quick
   spike"). Both are described in `transformer_extractor.py`'s docstring.

## Multilingual coverage

| Language | Classification (embeddings) | Owner/deadline cues | Evaluated? |
|---|---|---|---|
| English | Yes (model covers ~50 languages) | Yes | Yes -- 7/8 fixture scenarios |
| Hindi | Yes | Yes (Devanagari + romanised) | Partially -- 1/8 fixture scenarios (`02_code_mixed_hindi_english`) |
| Marathi | Yes (same multilingual model) | No cue patterns yet | **No** -- no fixture, no gold, no measured recall |
| Bengali | Yes | No cue patterns yet | **No** |
| Gujarati | Yes | No cue patterns yet | **No** |

`backend/app/ml/multilingual_cues.py` marks Marathi/Bengali/Gujarati
`evaluated=False` in code, not just in this document, specifically so a
future contributor cannot accidentally report an accuracy number for them
without first building the fixture and gold data that number would need.
**No performance claim is made for these three languages anywhere in this
project.**

## MLflow

`backend/app/ml/mlflow_tracking.py` logs `decision_f1`, `action_f1` and
`synthetic_wer` plus the active backend names as params, to a local
`./mlruns/` directory by default (already gitignored) or a real tracking
server if `MLFLOW_TRACKING_URI` is set. It is entirely optional: if `mlflow`
is not installed, every call becomes a no-op with a log line --
`run_evaluation.py` produces identical `results.json`/`report.md` either way.

## Known limitations (summary)

* Single annotator, no inter-annotator agreement.
* Thresholds calibrated and evaluated on the same 8-scenario corpus -- no
  held-out split.
* No real audio or real ASR output anywhere; WER is a synthetic
  demonstration.
* English-dominant corpus (7/8 scenarios); only Hindi has any code-mixed
  coverage, and only in one scenario.
* Marathi, Bengali, Gujarati are structurally supported (cue dictionaries
  exist, the embedding classifier is language-agnostic) but wholly
  unevaluated.
* No boundary-level segmentation gold, so segmentation is reported
  descriptively (block count, mean confidence) rather than as P/R/F1 against
  a reference segmentation.

## Future work

* A second annotator over `gold_annotations.json` plus an agreement measure
  (Cohen's kappa or similar) -- the single biggest gap for a genuine
  evaluation dossier.
* Real recordings (with consent) for at least English and Hindi, to measure
  actual WER and to check the fixture-derived cue/threshold calibration
  transfers to real disfluent speech.
* Fixture scenarios and gold annotations in Marathi, Bengali and Gujarati,
  before making any accuracy claim for them.
* Boundary-level segmentation gold, to exercise `segmentation_metrics.py`
  for real instead of only reporting descriptive statistics.
