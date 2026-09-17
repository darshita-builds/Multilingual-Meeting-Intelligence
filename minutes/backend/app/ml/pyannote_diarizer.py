"""Speaker diarization with pyannote.audio.

Diarization is the blueprint's *optional* module, and the orchestrator treats
it that way (`agent/orchestrator.py::_run_governed`): if `diarize()` raises,
the run records `stage.skipped` with `outcome="degraded"` and continues
without speaker labels. Every failure path here therefore raises
`MLServiceError` rather than letting the pipeline fail outright -- a missing
`HF_TOKEN`, an unaccepted model licence, or a network error should degrade the
run, not break it.

What this returns
------------------
A `DiarizationResult` whose `segments` are exactly the STT segments the caller
passed in, each with `speaker` populated. pyannote's pipeline produces its own
turn boundaries, independent of Whisper's segment boundaries, so this module
*maps* pyannote turns onto the existing STT segments by picking, for each
segment, the speaker turn with the greatest temporal overlap -- re-segmenting
here would desynchronise the reviewer editor, which renders STT segment
boundaries.

Privacy
-------
Speaker labels are pseudonymous and renumbered by first appearance
(`SPEAKER_00`, `SPEAKER_01`, ...), independent of whatever raw label pyannote
assigned internally. Nothing in this module maps a label to a real name --
doing so would reintroduce exactly the personal data `security/pii.py` exists
to remove before storage.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from backend.app.ml.base import DiarizationResult, MLServiceError, Segment
from backend.app.observability.logging import get_logger

log = get_logger("ml.pyannote_diarizer")

DEFAULT_MODEL = os.getenv("MOM_PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")

_Turn = tuple[float, float, str]  # (start, end, raw_label)


class PyannoteDiarizer:
    """Speaker segmentation and labelling."""

    name = "pyannote-3.1"

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        self.model_id = model_id
        self.name = f"pyannote-{model_id.rsplit('-', 1)[-1]}" if "-" in model_id else "pyannote"
        self._pipeline: Any = None  # lazy

    def diarize(self, audio_path: Path, segments: list[Segment]) -> DiarizationResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise MLServiceError(f"audio file does not exist: {audio_path}")
        if not segments:
            return DiarizationResult(segments=[], speaker_count=0, model_name=self._model_name())

        pipeline = self._load_pipeline()

        try:
            annotation = pipeline(str(audio_path))
        except Exception as exc:
            raise MLServiceError(f"pyannote diarization failed: {exc}") from exc

        turns: list[_Turn] = [
            (turn.start, turn.end, label)
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]

        if not turns:
            # The diarizer found no speech turns at all. Report what actually
            # happened (one inferred speaker) rather than crashing -- the
            # contract requires every segment to get a `speaker`.
            labelled = [
                Segment(start=s.start, end=s.end, text=s.text, speaker="SPEAKER_00")
                for s in segments
            ]
            return DiarizationResult(
                segments=labelled, speaker_count=1, model_name=self._model_name()
            )

        raw_to_normalised = self._normalise_labels(turns)
        labelled = [
            Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker=raw_to_normalised[self._best_speaker(seg, turns)],
            )
            for seg in segments
        ]
        speaker_count = len({s.speaker for s in labelled})
        return DiarizationResult(
            segments=labelled, speaker_count=speaker_count, model_name=self._model_name()
        )

    # ------------------------------------------------------------------ #

    def _model_name(self) -> str:
        return f"{self.name} ({self.model_id})"

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise MLServiceError(
                "pyannote.audio is not installed; install requirements-ml.txt "
                "or set MOM_DIARIZER_BACKEND=baseline"
            ) from exc

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise MLServiceError(
                "HF_TOKEN is not set. pyannote's speaker-diarization models are gated on "
                "Hugging Face: accept the model licence at the model page, then set HF_TOKEN "
                "in the environment (never commit it). Diarization is optional -- the "
                "orchestrator degrades gracefully without it."
            )
        try:
            pipeline = Pipeline.from_pretrained(self.model_id, use_auth_token=token)
        except Exception as exc:
            raise MLServiceError(
                f"failed to load pyannote pipeline {self.model_id!r}: {exc}"
            ) from exc
        if pipeline is None:
            raise MLServiceError(
                f"pyannote returned no pipeline for {self.model_id!r}; "
                "check the model licence has been accepted for this HF_TOKEN"
            )
        self._pipeline = pipeline
        log.info("pyannote_pipeline_loaded", model_id=self.model_id)
        return self._pipeline

    @staticmethod
    def _best_speaker(seg: Segment, turns: list[_Turn]) -> str:
        best_label, best_overlap = None, -1.0
        for t_start, t_end, label in turns:
            overlap = min(seg.end, t_end) - max(seg.start, t_start)
            if overlap > best_overlap:
                best_overlap, best_label = overlap, label
        if best_overlap > 0 and best_label is not None:
            return best_label
        # No temporal overlap at all (the segment falls in a gap pyannote
        # marked as non-speech) -- fall back to the nearest turn by centre
        # distance, so every segment still gets a speaker.
        mid = (seg.start + seg.end) / 2
        nearest = min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])))
        return nearest[2]

    @staticmethod
    def _normalise_labels(turns: list[_Turn]) -> dict[str, str]:
        order: list[str] = []
        for _, _, label in sorted(turns, key=lambda t: t[0]):
            if label not in order:
                order.append(label)
        return {label: f"SPEAKER_{i:02d}" for i, label in enumerate(order)}
