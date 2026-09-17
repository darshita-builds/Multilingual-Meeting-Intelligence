"""WhisperSTT, with `faster_whisper` faked via `sys.modules` injection -- the
real package is not installed in this test environment (by design: no model
download in CI), so this proves the transcription/merge/chunking logic is
correct without it, and confirms the "not installed" path degrades cleanly
when it genuinely is absent.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from backend.app.ml.base import MLServiceError
from backend.app.ml.whisper_stt import WhisperSTT


class _FakeSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start, self.end, self.text = start, end, text


class _FakeInfo:
    def __init__(self, language: str | None) -> None:
        self.language = language


def _fake_model_factory(scripted_results: list[tuple[list[_FakeSegment], str | None]]):
    """Returns (model_class, calls) -- `calls` records every `audio` argument
    `transcribe()` was called with, in order, so chunking can be asserted on."""
    calls: list = []

    class _Model:
        def __init__(self, model_size, device, compute_type):
            self.model_size, self.device, self.compute_type = model_size, device, compute_type

        def transcribe(self, audio, language=None, vad_filter=True, word_timestamps=False):
            calls.append(audio)
            segments, language_out = scripted_results[len(calls) - 1]
            return iter(segments), _FakeInfo(language_out)

    return _Model, calls


def _install_fake_faster_whisper(monkeypatch, model_cls, decode_audio_fn) -> None:
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = model_cls
    fw_audio = types.ModuleType("faster_whisper.audio")
    fw_audio.decode_audio = decode_audio_fn
    fw.audio = fw_audio
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", fw_audio)


def test_missing_audio_file_raises_before_touching_any_model(tmp_path):
    stt = WhisperSTT()
    with pytest.raises(MLServiceError, match="does not exist"):
        stt.transcribe(tmp_path / "nope.wav")


def test_missing_dependency_raises_clear_ml_service_error(tmp_path):
    """faster-whisper genuinely is not installed in this environment -- this
    exercises the real (not mocked) not-installed path."""
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    stt = WhisperSTT()
    with pytest.raises(MLServiceError, match="faster-whisper is not installed"):
        stt.transcribe(audio_path)


def test_short_audio_is_transcribed_in_a_single_pass(monkeypatch, tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")

    scripted = [
        ([_FakeSegment(0.0, 2.0, " Hello there. "), _FakeSegment(2.0, 4.0, " How are you? ")], "en")
    ]
    model_cls, calls = _fake_model_factory(scripted)
    _install_fake_faster_whisper(monkeypatch, model_cls, lambda path: np.zeros(16_000 * 10))

    stt = WhisperSTT(chunk_seconds=600)
    result = stt.transcribe(audio_path)

    assert len(calls) == 1  # no chunking -- well under the threshold
    assert result.text == "Hello there. How are you?"
    assert [(s.start, s.end, s.text) for s in result.segments] == [
        (0.0, 2.0, "Hello there."),
        (2.0, 4.0, "How are you?"),
    ]
    assert result.detected_languages == ["en"]
    assert result.is_code_mixed is False
    assert result.duration_seconds == 10.0
    assert result.model_name == f"faster-whisper-{stt.model_size} ({stt.compute_type})"


def test_long_audio_is_chunked_with_offsets_and_overlap_deduplication(monkeypatch, tmp_path):
    audio_path = tmp_path / "long.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")

    # chunk_seconds=100, overlap=10 -> step=90. Chunk offsets: 0, 90, 180.
    # duration=250s -> three chunks: [0,100) [90,190) [180,250).
    scripted = [
        ([_FakeSegment(10.0, 12.0, "kept: first chunk.")], "en"),
        (
            [
                _FakeSegment(0.0, 2.0, "dropped: inside the overlap region."),
                _FakeSegment(8.0, 10.0, "kept: past the overlap region."),
            ],
            "en",
        ),
        ([_FakeSegment(6.0, 8.0, "kept: third chunk.")], "en"),
    ]
    model_cls, calls = _fake_model_factory(scripted)
    _install_fake_faster_whisper(monkeypatch, model_cls, lambda path: np.zeros(16_000 * 250))

    stt = WhisperSTT(chunk_seconds=100, chunk_overlap_seconds=10)
    result = stt.transcribe(audio_path)

    assert len(calls) == 3
    texts = [s.text for s in result.segments]
    assert texts == ["kept: first chunk.", "kept: past the overlap region.", "kept: third chunk."]
    # offsets were actually applied, not just left at each chunk's local time
    starts = [s.start for s in result.segments]
    assert starts == [10.0, 98.0, 186.0]
    assert result.duration_seconds == 250.0


def test_code_mixing_is_detected_from_the_merged_text_script(monkeypatch, tmp_path):
    """Whisper reports one dominant language per pass; a script sweep over the
    merged text is what actually flags code-mixing (see the module docstring)."""
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")

    scripted = [([_FakeSegment(0.0, 2.0, "Hello नमस्ते")], "en")]
    model_cls, _calls = _fake_model_factory(scripted)
    _install_fake_faster_whisper(monkeypatch, model_cls, lambda path: np.zeros(16_000 * 5))

    stt = WhisperSTT(chunk_seconds=600)
    result = stt.transcribe(audio_path)

    assert set(result.detected_languages) == {"en", "hi"}
    assert result.is_code_mixed is True


def test_empty_segments_are_dropped(monkeypatch, tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")

    scripted = [([_FakeSegment(0.0, 1.0, "   "), _FakeSegment(1.0, 2.0, "Real text.")], "en")]
    model_cls, _calls = _fake_model_factory(scripted)
    _install_fake_faster_whisper(monkeypatch, model_cls, lambda path: np.zeros(16_000 * 5))

    stt = WhisperSTT(chunk_seconds=600)
    result = stt.transcribe(audio_path)

    assert len(result.segments) == 1
    assert result.segments[0].text == "Real text."
