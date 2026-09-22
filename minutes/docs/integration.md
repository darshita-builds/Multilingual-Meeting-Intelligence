# Integration: Person A × Person B

This document records how Person A's ML models and Person B's governed
platform fit together, and — the point of this file — **what was actually
verified end to end**, not just what the contract says should work. For the
contract itself and the platform's own design rationale, see
[`architecture.md`](architecture.md) (Person B) and
[`ml-evaluation.md`](ml-evaluation.md) (Person A); this file does not repeat
either, it documents the join between them.

## The short version

Nothing needed to be built to connect Person A and Person B — the connection
already existed, on purpose. Person B built the platform against four
`Protocol`s (`backend/app/ml/base.py`) before Person A's implementations
existed, specifically so that landing a real model would mean *implementing a
class*, not *wiring anything*. That prediction held: filling in the four stub
files in a prior session required zero changes to `agent/tools.py`,
`orchestrator.py`, `models.py` or `schemas.py`. This session's job was to
**prove** that claim by actually running the joined system, not just trust
the contract — and to fix the one real gap that surfaced from doing so.

## Person A architecture (brief)

| Component | File | Produces |
|---|---|---|
| STT | `backend/app/ml/whisper_stt.py` | `TranscriptionResult` |
| Diarization (optional) | `backend/app/ml/pyannote_diarizer.py` | `DiarizationResult` |
| Agenda segmentation | `backend/app/ml/embedding_segmenter.py` | `SegmentationResult` |
| Decision/action extraction | `backend/app/ml/transformer_extractor.py` | `ExtractionResult` |

Each also has a `baseline` counterpart in `backend/app/ml/baseline.py`
(Person B's, kept as the required control arm — never modified by this
integration work). See `docs/ml-evaluation.md` for how the real backends were
built and calibrated.

## Person B architecture (brief)

FastAPI + PostgreSQL/SQLite, a governed agent orchestrator with a six-tool
allow-list, two human approval gates, an append-only audit trail, and a React
reviewer editor. See `docs/architecture.md` for the full picture.

## The integration seam

```
backend/app/ml/base.py         four Protocols: SpeechToText, Diarizer,
                                AgendaSegmenter, DecisionActionExtractor

backend/app/ml/registry.py     binds ONE implementation per Protocol by env
                                var, lazily (baseline never imports torch)

backend/app/agent/tools.py     the four read tools (transcribe_audio,
                                diarize_speakers, segment_agenda,
                                extract_decisions_actions) call
                                ml.get_stt() / get_diarizer() / get_segmenter()
                                / get_extractor() and reshape the Protocol's
                                dataclasses into the tool's dict return

backend/app/agent/orchestrator.py   calls those four tools in sequence, then
                                     feeds extract_decisions_actions's output
                                     straight into the GATED persist_minutes
                                     tool -- no ML output reaches the database
                                     without a human approving that exact
                                     payload
```

Nothing here changed. This is Person B's design, and it is why "integration"
was a filling-in exercise rather than a wiring exercise.

## Tool allow-list (as registered — not renamed)

The blueprint's suggested tool names (`transcribe_meeting`, `diarize_meeting`,
`extract_decisions`, `extract_actions`) do **not** match what is actually
registered, and this document uses the real names rather than the suggested
ones — renaming a governed, audited tool is a breaking change with no
benefit:

| Registered name | Side effect | Gated |
|---|---|---|
| `transcribe_audio` | read | no |
| `diarize_speakers` | read | no |
| `segment_agenda` | read | no |
| `extract_decisions_actions` | read | no |
| `persist_minutes` | write | **yes** |
| `export_actions` | external | **yes** |

`ToolRegistry.register` (in `agent/registry.py`) structurally refuses to
register a `write`/`external` tool without `requires_approval=True`, so this
table cannot silently drift from what the code enforces.

## Data contract (as it actually exists — not the blueprint's generic shape)

The blueprint's suggested JSON schema (`meeting_id` / `transcript` /
`speakers` / `agenda_segments` / `decisions` / `actions` at the top level)
was **not** created — Person B already has an equivalent, more specific
schema, and duplicating it would violate the platform's actual data model.
The real mapping, dataclass → ORM row, happens in `agent/tools.py::_persist_minutes`:

| Person A dataclass field | Person B `Decision`/`ActionItem` column |
|---|---|
| `ExtractedDecision.text` / `ExtractedAction.text` | `text` |
| `.confidence` | `confidence` |
| `.evidence_quote` | `evidence_quote` (must be a verbatim transcript substring — enforced by tests) |
| `.agenda_position` | resolved to `agenda_block_id` via `_block_ids_by_position` |
| `ExtractedAction.owner_name` | `owner_name` (`None`, never invented) |
| `ExtractedAction.deadline` | `deadline` (`None`, never invented) |
| — | `run_id`, `source_tool` (= `ExtractionResult.model_name`), `status=PROPOSED` are added by the tool, not by Person A |

`owner_name`/`deadline` being `None` when unstated is enforced in
`transformer_extractor.py` (never guessed) and checked by
`tests/unit/test_transformer_extractor.py::test_owner_and_deadline_are_none_when_not_stated`.

## Database

No schema changes were made or needed. `Transcript → AgendaBlock →
Decision/ActionItem` (see `models.py`) already exactly fits what the four
Protocols return; `_persist_blocks` and `_persist_minutes` in
`orchestrator.py`/`tools.py` do the mapping. Verified via the manual
walkthrough below: real rows were written, read back through
`GET /jobs/{id}/minutes`, edited, approved and exported.

## Configuration

**`MOM_BACKEND` does not exist** — grepped for it across the repo, zero
matches. The real mechanism is four independent variables (each `baseline` by
default):

```bash
MOM_STT_BACKEND=whisper           # baseline | whisper
MOM_DIARIZER_BACKEND=pyannote     # baseline | pyannote
MOM_SEGMENTER_BACKEND=embedding   # baseline | embedding
MOM_EXTRACTOR_BACKEND=transformer # baseline | transformer
```

plus model-specific knobs (`MOM_WHISPER_MODEL`, `MOM_PYANNOTE_MODEL`,
`MOM_EMBEDDING_MODEL`, `MOM_EXTRACTOR_MODEL`, `MOM_EXTRACTOR_THRESHOLD_HIGH`/
`_LOW`, `HF_TOKEN`, ...) documented in `.env.example`.

### A real bug this integration pass found and fixed

Running the actual application (not just `TestClient`) surfaced a genuine
gap: `Settings` (`backend/app/config.py`) reads `.env` via pydantic-settings
for its own declared fields, but pydantic-settings **never writes `.env`
values into the real process environment** — it only populates the one
`Settings` instance. Every `MOM_*_BACKEND` switch (and `HF_TOKEN`, and every
per-model env var in `backend/app/ml/*.py`) is read with a raw `os.getenv()`
call, not through `Settings`. Result: setting `MOM_EXTRACTOR_BACKEND=transformer`
in `.env` had **no effect** when running `python -m uvicorn` directly — it
silently kept using `baseline`. It worked under Docker Compose only because
Compose has its own, separate `.env` auto-load for `${VAR}` substitution in
`docker-compose.yml`, which masked the gap there.

Fix: `backend/app/config.py` now calls `load_dotenv(PROJECT_ROOT / ".env")`
before anything else, so `os.environ` is populated the same way in both the
local and the Docker path. `python-dotenv` was already a transitive
dependency of `pydantic-settings`, so this adds nothing to `requirements.txt`.
`tests/conftest.py` was extended to explicitly pin all four
`MOM_*_BACKEND` variables (it already pinned two of the four), so a
developer's local `.env` can never leak a non-baseline backend into the test
suite — matching the protective pattern the file already used for
`DATABASE_URL`, `JWT_SECRET`, etc.

## What was verified, and how

1. **`tests/integration/test_pipeline_with_person_a_models.py`** (new) — the
   full governed pipeline (upload → run → gate → approve → review → export →
   audit) with `MOM_SEGMENTER_BACKEND=embedding` and
   `MOM_EXTRACTOR_BACKEND=transformer`, run through `TestClient`. Asserts
   every persisted item is traced to `transformer-extractor-v1` (not
   `baseline-extractor-v1`), every `evidence_quote` is a verbatim substring of
   the stored transcript, and a second test asserts the baseline and real
   arms concretely disagree on at least one item (the comparison is not
   theatre). **2 passed.**
2. **Full suite**, `pytest tests -q` — **258 passed**, 0 failed, both before
   and after the `.env`-loading fix.
3. **Live server**, genuinely started (`python -m uvicorn backend.app.main:app`,
   real port, real SQLite file, real HTTP requests via `requests`, not
   `TestClient`) with `MOM_SEGMENTER_BACKEND=embedding` and
   `MOM_EXTRACTOR_BACKEND=transformer` set in `.env`: login → upload the
   code-mixed Hindi/English fixture → start a governed run → inspect the
   gate (2 decisions, 3 actions, confidences 0.54–0.95, one Hindi decision
   surfaced correctly) → approve → confirm every persisted item traced to
   `transformer-extractor-v1` → edit one action item as a human reviewer →
   approve the rest → export (second gate) → approve → download the CSV →
   confirm the audit trail contains every required action. All 12 steps
   passed against the real, running process.

   This run also caught and fixed a real, unrelated problem: a stale orphaned
   `python.exe` process from an earlier, unrelated session was already
   holding port 8000, so the first server-start attempt silently failed to
   bind while `curl /health` kept answering from the *old* process (which
   was still running the baseline backend) — which is exactly why a
   "download and see if it breaks" pass catches things a mocked test cannot.
   Identified the PID, confirmed it was an unrelated stale process (wrong
   Python interpreter, ~20 hours old), stopped it, and re-ran clean.

## Running it yourself

```powershell
pip install -r requirements-ml.txt
copy .env.example .env
# edit .env: MOM_SEGMENTER_BACKEND=embedding, MOM_EXTRACTOR_BACKEND=transformer
python -m uvicorn backend.app.main:app --reload --port 8000
```

Then use the reviewer UI (`cd frontend; npm install; npm run dev`) or the
API directly — upload any `.txt` in `fixtures/meetings/`, start a run, watch
the gate, approve, review, export.

## Testing

```powershell
pytest tests -q                                                    # 258, mocked, no model download
pytest tests/integration/test_pipeline_with_person_a_models.py -v  # the 2 real-backend integration tests
python scripts/run_meeting_pipeline.py --audio fixtures/meetings/01_clean_english_standup.txt \
    --segmenter embedding --extractor transformer                 # standalone demo
```

## Known limitations

* `PyannoteDiarizer`/`WhisperSTT` were **not** exercised through the live
  server in this pass (no `HF_TOKEN`, and the fixture `.wav` files are
  placeholder tones — see `DATASET.md`). Their unit tests (mocked) pass; a
  real end-to-end run with real audio and a real HF token is unverified.
* The manual walkthrough used a `.txt` transcript upload (skips STT), the
  same path the platform recommends for demos generally — real-audio-through-Whisper
  is validated at the unit level (`test_whisper_stt.py`) but not through a
  live server in this pass.
* Frontend (React reviewer editor) was not driven in this pass — only the API
  it calls was verified directly.
* See `docs/ml-evaluation.md`'s own Limitations section for everything about
  the ML numbers themselves (single annotator, no held-out split, etc.) —
  unaffected by this integration work.

## 2026-09-20 follow-up: the gap above, closed, plus one real bug found

This session did what the previous pass's Limitations section flagged as
unverified: installed `requirements-ml.txt` in a fresh venv (Python 3.13,
Windows) and ran real audio through `WhisperSTT`, not a `.txt` transcript.

* **Real Whisper on real speech, proven distinct from the baseline.** The
  fixture `.wav` files are still placeholder tones (`DATASET.md`'s point
  stands), so a real-speech WAV was synthesized from
  `01_clean_english_standup.txt` via Windows SAPI TTS (`Microsoft Zira`, real
  synthesized speech, not a recorded meeting — documented as such, not
  claimed as more). `WhisperSTT.transcribe()` produced a transcript with
  genuine ASR artifacts ("Priya" → "PREA", "Aarti" → "ARTI" — SAPI
  mispronounces the names, and Whisper transcribes what it heard), proving
  this is real inference, not `baseline.py`'s sidecar-`.txt` shortcut.
* **The recorder's exact output format, proven end to end.** No system
  `ffmpeg` was available, so the same WAV was transcoded to genuine WebM/Opus
  with PyAV (`av`, already a `pyannote.audio` transitive dependency) —
  correct Matroska magic bytes, same container `MeetingRecorder.jsx`
  produces. That file was pushed through `ingestion.store_upload` (the exact
  validation `POST /jobs` runs) and then through real `WhisperSTT` directly.
  Both stages passed. This is the strongest evidence yet that the recorder's
  output is compatible with the real pipeline, not just structurally
  (extension/magic-byte allow-lists) but by actually decoding and
  transcribing one.
* **A real torch/torchaudio incompatibility, found and pinned.** Installing
  `requirements-ml.txt` with no other pins resolves `torch==2.14.0` +
  `torchaudio==2.11.0` (the newest torchaudio wheel that exists at all for
  cp313) on a fresh Windows/Python 3.13 install. `PyannoteDiarizer` fails
  immediately with `AttributeError: module 'torchaudio' has no attribute
  'AudioMetaData'` — torchaudio 2.11.0 has dropped the I/O API
  `pyannote.audio` 3.3.2 still calls. This is not a missing-dependency error
  (both packages import fine), so `ml/registry.py`'s fallback-to-baseline
  never triggers; it would have looked like real diarization was available
  and then failed at first real use. `requirements-ml.txt` now pins
  `torch==2.6.0` + `torchaudio==2.6.0` (a matched pair, verified working —
  torchaudio's compiled extension is ABI-locked to one exact torch build) as
  the newest confirmed-working combination. With that pin, `PyannoteDiarizer`
  imports and runs correctly up to the point of needing `HF_TOKEN` (not
  available in this session — see below), which is the correct, documented,
  graceful-degradation failure mode, not a crash.
* **`HF_TOKEN` still not available**, so `PyannoteDiarizer` was verified up to
  (and including) its "no token" error path, not against a real gated model.
  The orchestrator's degrade-on-`MLServiceError` behaviour
  (`agent/orchestrator.py::_run_governed`) was read and matches what the
  fixed dependency stack now actually raises. Note the asymmetry: the
  ungoverned control arm (`_run_ungoverned`) calls the diarizer directly with
  no try/except, so a diarization failure fails that arm's run outright
  rather than degrading — plausibly intentional (the control arm is "no
  governance, no safety nets" by design for the comparison study), but worth
  a deliberate decision rather than an assumption if diarization is ever
  turned on for that arm in a real deployment.
* **A real anti-hallucination bug found and fixed**: `multilingual_cues.py`'s
  English deadline pattern `\bbefore\s+(\w+day|the\s+\w+)\b` captured "the
  cutover" and "the board review" themselves as deadline strings for
  "Rahul needs to check the staging database backups before the cutover" and
  the equivalent board-review sentence — both scenarios where
  `gold_annotations.json` explicitly records `deadline: null`, and where this
  doc's own evaluation counterpart (`docs/ml-evaluation.md`) already
  documents "before the cutover" as the canonical example of a vague trigger
  phrase that must never become a deadline. Fixed by narrowing the pattern to
  `\bbefore\s+(\w+day)\b` (real calendar references only — "before Friday" —
  still match); regression test added
  (`test_vague_before_phrase_does_not_invent_a_deadline` in
  `tests/unit/test_transformer_extractor.py`). **This bug was invisible to
  `evaluation/report.md`'s reported F1 and slot-accuracy numbers**, both
  before and after the fix: `action_metrics.py::evaluate_actions` only checks
  deadline accuracy `if g.deadline is not None` (line 64), so a matched
  action whose gold deadline is `null` is never scored on that slot even if
  the prediction invented one. This is a real measurement blind spot for
  exactly the failure mode the project's anti-hallucination contract cares
  about most — worth fixing in `action_metrics.py` (e.g. score a non-null
  prediction against a null gold as wrong) before citing deadline-accuracy
  numbers as evidence the contract holds; not fixed in this pass because
  changing evaluation semantics is a bigger, separate decision than fixing
  the extractor bug it would have caught.
* **CAPTCHA added** (`backend/app/security/captcha.py`): a self-hosted
  arithmetic challenge on `/auth/register`, off by default
  (`CAPTCHA_ENABLED=false`), gated the same way `registration_mode` and
  `pii_scrubbing_enabled` already are. No external key or secret. See
  `tests/security/test_captcha.py`.
* Full suite: 293 passed (280 baseline + 13 new CAPTCHA tests), 0 failed,
  `ruff check`/`ruff format --check` clean on every file this session
  touched, `bandit -r backend scripts -ll` and `pip-audit --requirement
  requirements.txt --strict` both clean.

## 2026-09-22 follow-up: CAPTCHA frontend, and a real finding from real audio

* **CAPTCHA wired into `Login.jsx`.** The backend has existed since
  2026-09-20 (above) but nothing called it. Added: `api.getCaptcha()`
  (`frontend/src/api.js`), and in `Login.jsx` -- fetch
  `registration-policy` on mount to learn `captcha_required`; when true and
  the register form is shown, fetch a challenge, render its question with a
  plain text input, and send `captcha_id`/`captcha_answer` on submit. A
  failed attempt fetches a fresh challenge (the old one is consumed either
  way -- `security/captcha.py` is single-use). Verified against a real,
  separately-started `uvicorn` process with `CAPTCHA_ENABLED=true`,
  reproducing exactly the request sequence the browser makes
  (`GET registration-policy` → `GET captcha` → `POST register` →
  `POST login`): a wrong answer was refused, a correct one was accepted
  (201), and login then succeeded (200) with a real JWT. `CAPTCHA_ENABLED`
  stays off by default; `npm run build` is clean with the change; no new
  frontend test runner exists in this project (only `build`/`lint`, and
  `lint` was already broken pre-existing -- `eslint` is invoked by
  `package.json` but was never added to `devDependencies`), so the register
  flow's enabled/disabled behaviour is covered at the API-contract level by
  `tests/security/test_captcha.py`, the layer the frontend actually talks to.

* **Diarization, re-confirmed.** Same result as 2026-09-20: `HF_TOKEN` is
  still not set in this environment, so real gated-model diarization could
  not be executed. Re-verified instead, live, in this session: the pinned
  torch 2.6.0/torchaudio 2.6.0 pair still imports cleanly
  (`torchaudio.AudioMetaData` present), `PyannoteDiarizer` still resolves
  its configured model (`pyannote/speaker-diarization-3.1`, overridable via
  `MOM_PYANNOTE_MODEL`) and still raises the correct, actionable
  `MLServiceError` naming `HF_TOKEN` when it is absent -- not a crash. Do
  not read this as "diarization works" -- the actual gated pyannote model
  was never called.

* **Real Hindi audio, provided by the user
  (`hindi_meeting_noisy_20dB.wav`, ~5.6 MB, 16 kHz mono PCM, 3:03,
  genuinely noisy business-meeting audio) -- run through the real,
  unmodified pipeline end to end**: `ingestion.store_upload` →
  `WhisperSTT.transcribe()` → `EmbeddingAgendaSegmenter.segment()` →
  `TransformerExtractor.extract()`. This is the first time in this
  project's history that Person A's downstream models ran on real speech
  audio rather than a fixture `.txt` or a synthetic tone. It found a real,
  previously-invisible failure mode:

  1. **STT**: succeeded. `detected_languages=['hi']`, `is_code_mixed=False`,
     `duration_seconds=183.02` (matches the file), 41 timestamped segments,
     295.5s to transcribe (CPU, `small`, int8 -- ~1.6x realtime, plausible
     for noisy 20 dB-SNR audio). The transcript is real, degraded ASR
     output, not a placeholder: recognisable Hindi business-meeting content
     (deliveries, warehouse stock, customer payments, a rupee figure, a
     Tuesday follow-up) mixed with clear noise-induced errors ("बजट" →
     "बजजद"/"बजज़", "तय" → "ताए", "गोदाम" split into "गो दाम"), and one
     genuine repetition artefact -- the "गो दाम का स्टोक" segment is
     transcribed twice, near-verbatim, at two different timestamp ranges
     (62.21–75.21 and 76.21–90.21) -- a known Whisper failure mode on
     noisy/long audio, not a bug in this codebase.
  2. **`is_code_mixed=False` is plausible, not obviously wrong**: the
     transcript contains zero Latin-script characters (verified by regex
     scan). But it is genuinely code-mixed at the *vocabulary* level --
     "स्टोक" (stock), "डिलीवरी" (delivery), "पेमेंट" (payment), "एदवास"
     (advance), "अप्रेशन्स" (operations) are English loanwords, just
     transliterated into Devanagari rather than kept in Latin script. The
     existing code-mixing heuristic (`whisper_stt.py::_detect_languages`,
     also `multilingual_cues.py`) is a **script**-mixing detector
     (Devanagari + Latin both present) -- it cannot see this, because there
     is no competing script to detect. This is a real, previously
     undocumented gap: code-mixing that surfaces as loanwords-in-Devanagari
     rather than script-switching is structurally invisible to the current
     detector. Not fixed in this pass (surfaced by testing, not asked-for
     scope this round).
  3. **Segmentation and extraction both collapsed to one block/item for
     the entire 3-minute, 3-agenda-item meeting** (block confidence 0.50,
     1 action with `owner=None deadline='कल तक'`, **0 decisions** despite
     the transcript containing what reads as at least one real decision
     -- "ताए करते है, की तेल का स्टोक कम से कम दो हवते का रखा जाएगा").
     Root cause, verified directly against the real transcript text, not
     inferred: the real transcript contains **zero** sentence-terminating
     characters at all -- no `.`, no Devanagari danda `।`, no `!`/`?`
     (confirmed by character count: 0/0/0/0 across 2142 characters; only
     26 commas). Both `EmbeddingAgendaSegmenter._sentences()` and
     `TransformerExtractor._sentences()` share the same regex,
     `[^.!?।\n]+[.!?।]?` (`embedding_segmenter.py`, `transformer_extractor.py`),
     which -- given no terminator anywhere in the input -- returns the
     **entire transcript as one "sentence."** The segmenter's own
     `len(sentences) <= 1` guard (`embedding_segmenter.py`) then correctly,
     deliberately, non-crashingly returns exactly one block (this is
     working as designed for degenerate input, just not usefully); the
     extractor receives that one giant block, embeds it as a single
     ~2100-character vector, and classifies the entire meeting once.
     `deadline='कल तक'` is not invented -- "कल तक" is a real, listed Hindi
     deadline cue (`multilingual_cues.py::HINDI.deadline_patterns`) and is a
     verbatim substring of the transcript -- but it is attributed to the
     whole meeting instead of the one sentence that actually says it, which
     is uninformative to a reviewer even though it is not a hallucination
     by this project's own definition (real substring, real cue match).
     `owner=None` is also correct-by-omission rather than a fix: `HINDI`'s
     only owner pattern (`multilingual_cues.py`) matches a **Latin-script**
     name before `ko`/`ne` ("Rahul ko"/"Priya ne"); there is no
     Devanagari-script owner pattern at all, so a name spoken and
     transcribed in Devanagari (e.g. "अमित", mentioned twice in this
     transcript with clear task assignments -- "अमित तुम कल तक सपलायर
     को...") can never be extracted as an owner today, independent of the
     sentence-splitting issue above.

     **This is the single most significant finding of this pass**: every
     multilingual/segmentation/extraction number in `evaluation/report.md`
     (F1 1.0 decisions, F1 0.9615 actions) was measured on clean,
     already-punctuated fixture text, and on exactly one code-mixed
     scenario that Whisper never touched. This real-audio run shows the
     downstream pipeline's sentence-boundary assumption does not hold for
     real Hindi ASR output, and that gap was invisible until real audio was
     used. **Not fixed in this pass** -- the instruction for this round was
     to run and document a real audio sample, not to modify the pipeline;
     flagged here as the highest-value next fix, with the exact regex and
     both call sites named above.

  Raw pipeline output (full transcript, all 41 timestamped segments, and
  the segmenter/extractor output) is preserved outside the repository (not
  committed -- it is a real recording of unknown provenance/consent status,
  and this project's own fixture-corpus policy in
  `fixtures/meetings/DATASET.md` is explicit that only synthetic,
  authored-for-this-project audio belongs in the repo).

## 2026-09-22 follow-up: the segmentation bug, fixed -- and what real audio still exposes

The single most significant finding of the 2026-09-20 pass (above) was fixed
in this session: `backend/app/ml/multilingual_cues.py::split_into_sentences()`
is a new, shared, tiered sentence splitter used by both
`embedding_segmenter.py` and `transformer_extractor.py` in place of each
file's own copy of the same regex. Full design rationale and the "why this
never invents text" argument are in that function's own docstring; summary:

1. **Real STT segment boundaries**, if the caller has them and each one can
   be found as a literal, sequential substring of the transcript (built and
   tested, not yet wired into `agent/orchestrator.py` -- see "Deliberately
   not done" below).
2. **Punctuation** (`.`/`!`/`?`/`।`) -- byte-for-byte the original behaviour,
   engaged whenever terminators are not sparse. This is why every one of the
   280 pre-existing tests, and the clean-fixture evaluation numbers below,
   are unchanged.
3. **Comma-delimited clauses**, only when terminators are absent or occur
   less than once per 40 words (and the input is at least 12 words -- a
   short, genuinely single-sentence input is left alone). A comma is a token
   Whisper itself emitted; splitting on it reveals structure already in the
   transcript rather than inventing any.

**Before vs after, the same real audio file
(`hindi_meeting_noisy_20dB.wav`), full pipeline, no mocks:**

| | Before (2026-09-20) | After (2026-09-22) |
|---|---|---|
| Agenda blocks | 1 (the entire 3-minute transcript) | 9 |
| Decisions | 0 | 1, evidence-scoped to its own clause |
| Actions | 1, evidence = the **entire transcript** | 0 (see below -- not a regression) |

The decision found in the "after" run -- `इस बार हमें पहले से अडर देन चाही है
सही बात है,` ("this time we should place the order in advance -- that's
right") -- has `evidence_quote` equal to its own 12-word clause, not the
whole recording. This is the direct, concrete result of the fix: correctly
*scoped* extraction, exactly the goal stated for this work (not maximising
extracted-item counts).

### The action count going to zero is not a regression -- and not a segmentation problem

Re-running the *same* audio file a second time (this session, for the
"after" comparison) produced a **different** transcript from the first run
(2026-09-20): 56 STT segments this time vs. 41 before, and different
transcription errors in places -- `faster-whisper`'s CPU inference is not
perfectly deterministic run-to-run on the same input. Concretely: the
sentence that clearly states an action with a deadline --
"अमित तुम [कल|खल] तक सपलायर को तो सो दिभबों का अडर दे दो" ("Amit, place an
order of 100 boxes with the supplier by tomorrow") -- transcribed "कल"
("tomorrow") correctly in the first run, and as "खल" (not a real word) in
this run. `multilingual_cues.py::HINDI.deadline_patterns`'s "कल तक" cue is an
exact string match; "खल तक" does not match it, and neither the action-cue
list nor the embedding similarity alone cleared the classification
threshold for that sentence in this run. **Zero actions is the honest,
correct output given what Whisper actually transcribed this time** -- not a
segmentation failure (the sentence is on its own, correctly, in block #5),
and not something to paper over by loosening exact-match cue regexes to
"help" one run, which would trade a real precision loss for one run's
recall. Documented rather than fixed: this is a real-audio ASR-noise
sensitivity in the *extraction* layer's cue matching, a materially different
problem from the segmentation bug this session fixed, and worth a dedicated
look (e.g. fuzzy/edit-distance cue matching, or falling back to the
action-prototype embedding similarity alone when a near-miss cue is
detected) rather than a quick patch here.

### Hindi owner extraction: what was added, what was deliberately not

`HINDI.owner_patterns` gained a Devanagari-script को/ने pattern (the
existing pattern was romanised-Latin-only -- "Aarti ko", not "अमित ने" --
so a name spoken and transcribed entirely in Devanagari could never be
found). Verified against your three example sentences:

| Sentence | Owner found |
|---|---|
| "पूजा को रिपोर्ट तैयार करनी है" | `पूजा` (correct) |
| "अमित ने यह काम पूरा करना है" | `अमित` (correct) |
| "राहुल कल तक रिपोर्ट भेजेगा" | `None` (known gap, see below) |

A `\b`-based version of this pattern was tried first and found broken by
direct testing, not inspection: Python's `\b` is defined via `\w`, and `\w`
excludes Unicode category Mc (spacing combining marks) -- which is what a
Devanagari dependent vowel sign *is*. `को\b` silently fails to match "को" at
a word's end before whitespace. Fixed with `(?![ऀ-ॿ])` (not
immediately followed by another Devanagari character) instead, which also
correctly rejects "कोई" (a different word starting with "को").

A second pattern, matching a name immediately before a future-tense verb
(covering "राहुल ... भेजेगा"), was designed, tested, and **deliberately
dropped**: Hindi is SOV (subject-object-verb), so in a real sentence the
word immediately before the verb is usually the *object*
("रिपोर्ट कल तक ... भेजेगा" matches "रिपोर्ट", not "राहुल"). A
clause-initial-word variant was tried next and also rejected: tested against
"अब हम रिपोर्ट भेजेंगे" ("now we will send the report", no person named at
all), it extracts "अब" ("now") as the owner. Both are documented, not
shipped -- inventing a wrong owner is a worse outcome than correctly
returning `None`, and the task's own instruction is explicit on this point.

### Deliberately not done: wiring segments into the governed pipeline

`split_into_sentences()`'s segment-based tier is real, tested
(`tests/unit/test_multilingual_cues.py`, `test_embedding_segmenter.py`), and
reachable today via a direct `.segment(text, segments)` call -- but
`agent/orchestrator.py`'s two call sites (`_run_governed` line ~309,
`_run_ungoverned` line ~418) still call `.segment(transcript.text)` with no
`segments` argument, exactly as before this session. Real STT segments (with
real timestamps) sit unused in scope at both call sites. Wiring them through
would mean: adding an optional `segments` field to
`agent/tools.py::SegmentAgendaArgs` (backward compatible -- existing callers
omit it and behave identically) and passing the already-in-scope `segments`
variable through both orchestrator call sites. This is Person B's file, and
this session's brief was explicit about not touching Person B's files unless
strictly necessary; the text-based fallback alone already fixes the
diagnosed bug (proven above, using this exact real audio file), so this
wiring was left undone rather than made without being asked for. It is a
small, additive, low-risk follow-up if wanted.

### Multilingual/code-mixed detection (`is_code_mixed`) -- assessed, not changed

Investigated whether the script-based code-mixing detector
(`whisper_stt.py::_detect_languages`) could be improved to catch English
loanwords transliterated into Devanagari (the 2026-09-20 finding -- "स्टॉक",
"पेमेंट", "एडवांस" etc. are English loanwords, invisible to a detector that
only looks for competing scripts). Concluded this should not be changed in
this pass: any loanword-list approach has no gold data to validate
precision/recall against, and this project's own stated principle --
applied everywhere else in this codebase (see `evaluated=False` on
Marathi/Bengali/Gujarati/Awadhi) -- is to never claim a capability without
evaluation backing it. Shipping an unevaluated heuristic risks exactly the
overclaiming the task instructions explicitly warn against ("do not claim a
language is present merely because a keyword appears"). `is_code_mixed`'s
existing script-based definition is unchanged.

### Awadhi

Added as a fifth `LanguageCues` entry, `evaluated=False`, matching the
existing Marathi/Bengali/Gujarati pattern. Unlike those three, no
independently-verified Awadhi-specific decision/action vocabulary was
available to write against -- Awadhi is closely related to Hindi and
formal/business Awadhi speech commonly draws on shared Hindi vocabulary for
these concepts, so the entry reuses `HINDI`'s cue patterns verbatim rather
than asserting dialect-specific forms with no way to check them.

### Diarization -- re-confirmed, still gated

Same result as both prior sessions: `HF_TOKEN` is not set in this
environment. Re-verified live: `PyannoteDiarizer` still resolves its
configured model and still raises the correct, actionable `MLServiceError`
naming `HF_TOKEN` when absent. **Real gated-model diarization was not
executed.** No diarization code was changed this session.

### Domain Minutes / four-domain integration

Not modified. The fix operates entirely upstream of `Decision`/`ActionItem`
persistence -- `minutes_templates.py`, `generated_minutes.py`, the approval
workflow and all four domain renderers consume whatever the (now better-
scoped) extraction produces exactly as before; nothing about that data flow
changed. Verified by the full test suite, which includes
`tests/integration/test_generated_minutes.py` and
`tests/unit/test_minutes_templates.py`, both passing unchanged.

### Tests and final regression

```powershell
$env:JWT_SECRET="ci-only-secret-not-used-anywhere-real"
.venv\Scripts\python -m pytest tests -o addopts="" -q --deselect tests/unit/test_pyannote_diarizer.py::test_missing_dependency_raises_clear_ml_service_error --deselect tests/unit/test_whisper_stt.py::test_missing_dependency_raises_clear_ml_service_error
.venv\Scripts\python -m ruff check backend tests
.venv\Scripts\python -m ruff format --check backend tests
.venv\Scripts\python -m bandit -r backend scripts -ll
.venv\Scripts\python -m pip_audit --requirement requirements.txt --strict
cd frontend; npm run build
```

**314 passed, 0 failed, 2 deselected** (same 2 as every prior session --
their whole purpose is asserting the "library not installed" path, correctly
inapplicable once `requirements-ml.txt` is installed). Up from 293: 21 new
tests (`split_into_sentences` tiers, Hindi owner extraction, the real-shaped
decision/action scoping regression, Awadhi's structural presence). Ruff,
bandit, pip-audit and the frontend build all clean, matching every prior
session in this project.

Clean-fixture evaluation (`evaluation/report.md`) re-run after the fix:
**Decision F1 = 1.0, Action F1 = 0.9615 -- identical to before**, confirming
the fix is inert on already-well-punctuated text, as designed.
