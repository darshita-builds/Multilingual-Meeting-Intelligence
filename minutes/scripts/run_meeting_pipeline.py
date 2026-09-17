"""Standalone demo of Person A's ML pipeline.

Independent of the FastAPI app, the database, and the governed orchestrator
(`backend/app/agent/orchestrator.py` already composes these same four
services behind human approval gates; this script composes them directly,
read-only, so Person A's models can be proven to work on their own, with
nothing else running).

    python scripts/run_meeting_pipeline.py --audio fixtures/meetings/01_clean_english_standup.txt
    python scripts/run_meeting_pipeline.py --audio recording.wav --diarize
    python scripts/run_meeting_pipeline.py --audio recording.wav \
        --stt whisper --segmenter embedding --extractor transformer

Backend selection matches the platform's own environment variables
(`MOM_STT_BACKEND` etc., see `.env.example`) -- the CLI flags here just set
them before the registry binds, so this script exercises exactly the same
binding code the API uses. With no flags it runs entirely on the rule-based
baseline, so it works with zero extra installs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows terminals commonly default stdout to a legacy codepage (cp1252) that
# cannot encode Devanagari -- and this script exists specifically to print
# multilingual/code-mixed output. Force UTF-8 rather than let a Hindi agenda
# title crash the demo.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _bind_backends(args: argparse.Namespace) -> None:
    if args.stt:
        os.environ["MOM_STT_BACKEND"] = args.stt
    if args.diarizer:
        os.environ["MOM_DIARIZER_BACKEND"] = args.diarizer
    if args.segmenter:
        os.environ["MOM_SEGMENTER_BACKEND"] = args.segmenter
    if args.extractor:
        os.environ["MOM_EXTRACTOR_BACKEND"] = args.extractor


def _print_report(
    audio_path: Path, transcription, segmentation, extraction, speaker_count: int
) -> None:
    duration = transcription.duration_seconds or 0.0
    minutes, seconds = int(duration // 60), int(duration % 60)

    print()
    print("Meeting processed successfully.")
    print()
    print(f"Source: {audio_path.name}")
    languages = ", ".join(transcription.detected_languages) or "unknown"
    print(f"Language(s): {languages}" + (" (code-mixed)" if transcription.is_code_mixed else ""))
    print(f"Duration: {minutes:02d}:{seconds:02d}")
    if speaker_count:
        print(f"Speakers detected: {speaker_count}")
    print(f"Transcription model: {transcription.model_name}")
    print()

    print(f"Agenda ({segmentation.model_name}):")
    if not segmentation.blocks:
        print("  (no agenda blocks -- this should not happen; see embedding_segmenter.py)")
    for block in segmentation.blocks:
        print(f"  {block.position + 1}. {block.title}  [confidence {block.confidence:.2f}]")
    print()

    print(f"Decisions ({extraction.model_name}):")
    if not extraction.decisions:
        print("  (none)")
    for decision in extraction.decisions:
        print(f"  - {decision.text}  [confidence {decision.confidence:.2f}]")
    print()

    print("Actions:")
    if not extraction.actions:
        print("  (none)")
    for action in extraction.actions:
        owner = action.owner_name or "(unassigned)"
        deadline = f" -> {action.deadline}" if action.deadline else ""
        print(f"  - {owner}: {action.text}{deadline}  [confidence {action.confidence:.2f}]")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Person A's ML pipeline standalone.")
    parser.add_argument(
        "--audio", required=True, type=Path, help="Audio file, or a .txt/.vtt/.md transcript."
    )
    parser.add_argument("--language-hint", default=None)
    parser.add_argument("--diarize", action="store_true", help="Also run speaker diarization.")
    parser.add_argument("--stt", choices=["baseline", "whisper"], default=None)
    parser.add_argument("--diarizer", choices=["baseline", "pyannote"], default=None)
    parser.add_argument("--segmenter", choices=["baseline", "embedding"], default=None)
    parser.add_argument("--extractor", choices=["baseline", "transformer"], default=None)
    args = parser.parse_args(argv)

    if not args.audio.exists():
        print(f"error: {args.audio} does not exist", file=sys.stderr)
        return 1

    _bind_backends(args)

    from backend.app.ml import registry as ml
    from backend.app.ml.base import MLServiceError

    print(f"Backends: {ml.active_backends()}")

    stt = ml.get_stt()
    print(f"Transcribing {args.audio.name} with {stt.name}...")
    try:
        transcription = stt.transcribe(args.audio, language_hint=args.language_hint)
    except MLServiceError as exc:
        print(f"error: transcription failed: {exc}", file=sys.stderr)
        return 1

    segments = transcription.segments
    speaker_count = 0
    if args.diarize:
        diarizer = ml.get_diarizer()
        print(f"Diarizing with {diarizer.name}...")
        try:
            diarization = diarizer.diarize(args.audio, segments)
            segments = diarization.segments
            speaker_count = diarization.speaker_count
        except MLServiceError as exc:
            print(f"  diarization unavailable ({exc}); continuing without speaker labels.")

    segmenter = ml.get_segmenter()
    segmentation = segmenter.segment(transcription.text, segments)

    extractor = ml.get_extractor()
    extraction = extractor.extract(segmentation.blocks)

    _print_report(args.audio, transcription, segmentation, extraction, speaker_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
