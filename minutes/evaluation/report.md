# ML evaluation report

Generated 2026-09-16T16:14:20+00:00 by `python -m backend.app.ml.evaluation.run_evaluation`. Do not hand-edit -- regenerate.

## Data provenance

| Source | Kind |
|---|---|
| `fixtures/meetings/*.txt` | SYNTHETIC -- authored corpus, see `fixtures/meetings/DATASET.md` |
| `evaluation/data/gold_annotations.json` | MANUALLY ANNOTATED (single annotator: Person A) over the synthetic corpus |
| `evaluation/data/synthetic_wer_pairs.json` | SYNTHETIC -- hand-written ASR error simulation, not real ASR output |

No REAL (recorded-meeting) data is used anywhere in this report. See **Limitations** below before citing any number here as an accuracy claim.

## Backends under evaluation

- Segmenter: `embedding-texttiling-v1`
- Extractor: `transformer-extractor-v1`

## Baseline comparison

The rule-based baseline (`baseline.py`) scored on the same gold data, as the required control arm for this comparison study:

| Arm | Segmenter | Extractor | Decision F1 | Action F1 |
|---|---|---|---|---|
| Configured | `embedding-texttiling-v1` | `transformer-extractor-v1` | 1.0 | 0.9615 |
| Baseline | `baseline-agenda-v1` | `baseline-extractor-v1` | 0.9767 | 0.9388 |

Both arms are scored against the same manually-annotated gold set on the same 8-scenario corpus (the calibration-set caveat in Limitations applies to both).

## Decision extraction

Overall (micro-averaged over 8 scenarios): P=1.0 R=1.0 F1=1.0 (tp=22, fp=0, fn=0)

By language:

| Language | Scenarios | P | R | F1 |
|---|---|---|---|---|
| en | 7 | 1.0 | 1.0 | 1.0 |
| hi | 1 | 1.0 | 1.0 | 1.0 |

## Action extraction

Overall: P=0.9615 R=0.9615 F1=0.9615 (tp=25, fp=1, fn=1)

By language:

| Language | Scenarios | P | R | F1 |
|---|---|---|---|---|
| en | 7 | 0.9565 | 0.9565 | 0.9565 |
| hi | 1 | 1.0 | 1.0 | 1.0 |

Owner/deadline slot accuracy, restricted to correctly-matched action items (see each scenario in `results.json` for the per-scenario breakdown; `null` means no gold item in that scenario had that slot to check):

| Scenario | Owner accuracy | Deadline accuracy |
|---|---|---|
| 01_clean_english_standup | 1.0 | 1.0 |
| 02_code_mixed_hindi_english | 1.0 | 1.0 |
| 03_pii_heavy_onboarding | 1.0 | 1.0 |
| 04_prompt_injection_planning | 1.0 | 1.0 |
| 05_no_decisions_social | — | — |
| 06_hedged_ambiguous | 1.0 | — |
| 07_disfluent_noisy | 1.0 | — |
| 08_long_quarterly_review | 1.0 | 1.0 |

## Agenda segmentation (descriptive)

No boundary-level gold annotations exist yet (`segmentation_metrics.py` is ready for them once they do -- see Limitations). Reporting block counts and mean confidence per scenario as a sanity check instead of inventing reference boundaries.

| Scenario | Blocks | Mean confidence |
|---|---|---|
| 01_clean_english_standup | 1 | 0.5 |
| 02_code_mixed_hindi_english | 2 | 0.438 |
| 03_pii_heavy_onboarding | 3 | 0.436 |
| 04_prompt_injection_planning | 1 | 0.5 |
| 05_no_decisions_social | 2 | 0.535 |
| 06_hedged_ambiguous | 2 | 0.428 |
| 07_disfluent_noisy | 2 | 0.463 |
| 08_long_quarterly_review | 13 | 0.66 |

## Word Error Rate (SYNTHETIC pairs -- not a WhisperSTT accuracy measurement)

Corpus WER (micro-averaged): **0.1167** (3 sub, 2 del, 2 ins over 60 reference words)

| Pair | Error type | WER |
|---|---|---|
| exact_match | none (sanity check: identical strings must score WER 0) | 0.0 |
| substitution_near_homophone | substitution (near-homophone: decision/division) | 0.0769 |
| deletion_function_word | deletion (a short function word dropped) | 0.0909 |
| insertion_disfluency_and_repeat | insertion (filler word and a duplicated word, typical of disfluent speech) | 0.2857 |
| code_mixed_script_confusion | substitution (romanised Hindi function words misheard as similar-sounding English words -- the failure mode code-mixed audio is expected to produce) | 0.2 |
| date_digit_deletion | deletion (a digit dropped from a spoken date) | 0.0909 |

## Error analysis

### Decisions

None.

### Actions

- **false_positive** (03_pii_heavy_onboarding): "Let us start the vendor onboarding call."
- **false_negative** (03_pii_heavy_onboarding): "Send the signed agreement to arjun.mehta@example.com and copy the finance team."

## Limitations

- **Thresholds were calibrated against this same 8-scenario corpus** (there is no separate dev/test split -- the corpus is too small to hold one out meaningfully). Numbers above show the method works and give an honest first read, not a generalisation estimate. A held-out, independently-annotated set is future work.
- **Single annotator.** `gold_annotations.json` has no inter-annotator agreement measure. A second annotator pass is listed as future work in `docs/ml-evaluation.md`.
- **No real speech or real meeting transcripts anywhere in this evaluation.** WER is demonstrated on synthetic pairs only; STT accuracy on real audio is unmeasured.
- **English-dominant corpus.** Only one scenario is meaningfully code-mixed; the language-wise breakdown above should not be read as parity across five languages.
- Marathi, Bengali and Gujarati have cue dictionaries (`multilingual_cues.py`) but no fixture, gold annotation or measured recall at all -- they are structurally supported, not evaluated. No accuracy claim is made for them anywhere in this project.
