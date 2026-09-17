"""Whisper / faster-whisper speech-to-text.

Multilingual, code-mixed-aware transcription. `faster-whisper` (CTranslate2) is
used rather than `openai-whisper` per the stub's guidance -- roughly 4x faster
at equal accuracy, and it ships its own CPU-friendly int8 quantised inference
path, which matters because CPU is this project's default target (blueprint
rule: no GPU required).

Long-audio handling
--------------------
Whole files under `MOM_WHISPER_CHUNK_SECONDS` (default 600s / 10 minutes) are
transcribed in one pass. Longer files are decoded once into a 16 kHz waveform
and processed in overlapping windows (`MOM_WHISPER_CHUNK_OVERLAP_SECONDS`,
default 2s), each transcribed independently and stitched back together with
timestamps offset to the chunk's position in the original file. The overlap
exists so a word spoken across a chunk boundary is not silently dropped; the
half-overlap deduplication in `_transcribe_chunks` keeps it from appearing
twice. This bounds peak memory and keeps any one tool call (which runs under a
per-run time budget -- see `agent/registry.py`) working on a fixed-size window
regardless of total recording length.

Language detection / code-mixing
---------------------------------
Whisper reports one dominant language per pass (`info.language`). Code-mixed
speech (the core difficulty this project targets) does not fit that model
well, so `_detect_languages` adds a second pass: a script-detection sweep over
the *merged transcript text* for Devanagari and Latin characters, the same
technique `baseline.py` uses. `is_code_mixed=True` whenever more than one
language is present in the result, whether from Whisper's own per-chunk
detections disagreeing (a genuinely multilingual recording) or from the
script sweep (Devanagari and Latin both appearing in Whisper's own output,
which is what a code-mixed transcription tends to produce).

Contract notes
--------------
* Raises `MLServiceError` on any failure, including a missing dependency --
  the registry only auto-falls-back to baseline when the *module import*
  fails (see `ml/registry.py`), and this class can be instantiated even
  without faster-whisper installed (dependencies are loaded lazily, on first
  `transcribe()` call), so a call-time `MLServiceError` is how a genuinely
  missing dependency is surfaced without hiding the reason.
* Does not mutate anything on disk -- pure read, as the tool contract requires.
* Model is loaded once per process (`self._model` cached on the instance) --
  `ml/registry.py` caches one `WhisperSTT` instance per process, so this is
  paid at most once per deployment, not per request.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.app.ml.base import MLServiceError, Segment, TranscriptionResult
from backend.app.observability.logging import get_logger

log = get_logger("ml.whisper_stt")

DEFAULT_MODEL_SIZE = os.getenv("MOM_WHISPER_MODEL", "small")
DEFAULT_COMPUTE_TYPE = os.getenv("MOM_WHISPER_COMPUTE", "int8")
DEFAULT_DEVICE = os.getenv("MOM_WHISPER_DEVICE", "cpu")
#: Files longer than this are processed in overlapping windows instead of one
#: pass. <= 0 disables chunking (always transcribe as a single pass).
DEFAULT_CHUNK_SECONDS = float(os.getenv("MOM_WHISPER_CHUNK_SECONDS", "600"))
DEFAULT_CHUNK_OVERLAP_SECONDS = float(os.getenv("MOM_WHISPER_CHUNK_OVERLAP_SECONDS", "2"))

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
_SAMPLE_RATE = 16_000


@dataclass
class _Piece:
    """One transcribed window, before merging."""

    segments: list[Segment] = field(default_factory=list)
    language: str | None = None


class WhisperSTT:
    """Multilingual, code-mixed-aware transcription via faster-whisper."""

    name = f"faster-whisper-{DEFAULT_MODEL_SIZE}"

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        device: str = DEFAULT_DEVICE,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        chunk_overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
    ) -> None:
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.chunk_seconds = chunk_seconds
        self.chunk_overlap_seconds = max(0.0, chunk_overlap_seconds)
        # Instance-level, not just the class default: two instances configured
        # with different model sizes must report their own model_name.
        self.name = f"faster-whisper-{model_size}"
        self._model: Any = None  # lazy: do not pay model load at import time

    def transcribe(self, audio_path: Path, language_hint: str | None = None) -> TranscriptionResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise MLServiceError(f"audio file does not exist: {audio_path}")

        model = self._load_model()

        try:
            audio = self._decode(audio_path)
        except MLServiceError:
            raise
        except Exception as exc:
            raise MLServiceError(f"failed to decode {audio_path.name}: {exc}") from exc

        duration = len(audio) / _SAMPLE_RATE

        try:
            if self.chunk_seconds > 0 and duration > self.chunk_seconds:
                pieces = self._transcribe_chunks(model, audio, duration, language_hint)
            else:
                pieces = [self._transcribe_array(model, audio, 0.0, language_hint)]
        except Exception as exc:
            raise MLServiceError(f"faster-whisper transcription failed: {exc}") from exc

        return self._merge(pieces, duration)

    # ------------------------------------------------------------------ #
    # Model / audio loading -- both lazy, both raise MLServiceError
    # ------------------------------------------------------------------ #

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise MLServiceError(
                "faster-whisper is not installed; install requirements-ml.txt "
                "or set MOM_STT_BACKEND=baseline"
            ) from exc
        log.info(
            "whisper_model_loading",
            model_size=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )
        return self._model

    @staticmethod
    def _decode(audio_path: Path) -> Any:
        try:
            from faster_whisper.audio import decode_audio
        except ImportError as exc:
            raise MLServiceError(
                "faster-whisper is not installed; install requirements-ml.txt "
                "or set MOM_STT_BACKEND=baseline"
            ) from exc
        return decode_audio(str(audio_path))

    # ------------------------------------------------------------------ #
    # Transcription
    # ------------------------------------------------------------------ #

    def _transcribe_array(
        self, model: Any, audio_array: Any, offset_seconds: float, language_hint: str | None
    ) -> _Piece:
        segments_gen, info = model.transcribe(
            audio_array, language=language_hint, vad_filter=True, word_timestamps=False
        )
        segments = [
            Segment(
                start=round(offset_seconds + s.start, 2),
                end=round(offset_seconds + s.end, 2),
                text=s.text.strip(),
            )
            for s in segments_gen
            if s.text and s.text.strip()
        ]
        return _Piece(segments=segments, language=info.language)

    def _transcribe_chunks(
        self, model: Any, audio: Any, duration: float, language_hint: str | None
    ) -> list[_Piece]:
        step = max(1.0, self.chunk_seconds - self.chunk_overlap_seconds)
        pieces: list[_Piece] = []
        start, index = 0.0, 0
        while start < duration:
            end = min(duration, start + self.chunk_seconds)
            s_idx, e_idx = int(start * _SAMPLE_RATE), int(end * _SAMPLE_RATE)
            piece = self._transcribe_array(model, audio[s_idx:e_idx], start, language_hint)
            if index > 0:
                # Drop the half of the overlap already covered by the previous
                # chunk, so a segment spanning the boundary is not duplicated.
                floor = start + self.chunk_overlap_seconds / 2
                piece.segments = [s for s in piece.segments if s.start >= floor]
            pieces.append(piece)
            log.info(
                "whisper_chunk_transcribed",
                chunk_index=index,
                start_s=round(start, 1),
                end_s=round(end, 1),
            )
            index += 1
            start += step
        return pieces

    # ------------------------------------------------------------------ #
    # Merge + language detection
    # ------------------------------------------------------------------ #

    def _merge(self, pieces: list[_Piece], duration: float) -> TranscriptionResult:
        segments: list[Segment] = [s for piece in pieces for s in piece.segments]
        text = " ".join(s.text for s in segments).strip()

        primary = next((p.language for p in pieces if p.language), None)
        languages = self._detect_languages(text, primary)

        return TranscriptionResult(
            text=text,
            segments=segments,
            detected_languages=languages,
            is_code_mixed=len(languages) > 1,
            duration_seconds=round(duration, 2),
            model_name=f"{self.name} ({self.compute_type})",
        )

    @staticmethod
    def _detect_languages(text: str, primary: str | None) -> list[str]:
        found: list[str] = []
        if primary:
            found.append(primary)
        if _DEVANAGARI.search(text) and "hi" not in found:
            found.append("hi")
        if _LATIN.search(text) and "en" not in found:
            found.append("en")
        return found or ["unknown"]
