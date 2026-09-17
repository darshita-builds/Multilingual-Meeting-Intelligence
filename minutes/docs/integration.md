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
