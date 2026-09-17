"""PyannoteDiarizer, with `pyannote.audio` faked via `sys.modules` injection --
the real package is not installed in this test environment (heavy torch-based
dependency chain, no model download in CI)."""

from __future__ import annotations

import sys
import types

import pytest

from backend.app.ml.base import MLServiceError, Segment
from backend.app.ml.pyannote_diarizer import PyannoteDiarizer


class _FakeTurn:
    def __init__(self, start: float, end: float) -> None:
        self.start, self.end = start, end


class _FakeAnnotation:
    def __init__(self, tracks: list[tuple[float, float, str]]) -> None:
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False):
        for start, end, label in self._tracks:
            yield _FakeTurn(start, end), "_", label


def _fake_pipeline_factory(tracks: list[tuple[float, float, str]]):
    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model_id, use_auth_token=None):
            instance = cls()
            instance.model_id = model_id
            instance.token = use_auth_token
            return instance

        def __call__(self, audio_path):
            return _FakeAnnotation(tracks)

    return _Pipeline


def _install_fake_pyannote(monkeypatch, pipeline_cls) -> None:
    pyannote_pkg = types.ModuleType("pyannote")
    audio_mod = types.ModuleType("pyannote.audio")
    audio_mod.Pipeline = pipeline_cls
    pyannote_pkg.audio = audio_mod
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio_mod)


def test_missing_audio_file_raises_before_touching_any_model(tmp_path):
    diarizer = PyannoteDiarizer()
    with pytest.raises(MLServiceError, match="does not exist"):
        diarizer.diarize(tmp_path / "nope.wav", [Segment(start=0.0, end=1.0, text="hi")])


def test_empty_segments_short_circuits_with_no_model_call(tmp_path, monkeypatch):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF")
    # No fake pyannote installed at all -- if diarize() tried to load a
    # pipeline for an empty segment list, this would raise MLServiceError.
    diarizer = PyannoteDiarizer()
    result = diarizer.diarize(audio_path, [])
    assert result.segments == []
    assert result.speaker_count == 0


def test_missing_dependency_raises_clear_ml_service_error(tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF")
    diarizer = PyannoteDiarizer()
    with pytest.raises(MLServiceError, match=r"pyannote\.audio is not installed"):
        diarizer.diarize(audio_path, [Segment(start=0.0, end=1.0, text="hi")])


def test_missing_hf_token_raises_clear_ml_service_error(monkeypatch, tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF")
    _install_fake_pyannote(monkeypatch, _fake_pipeline_factory([]))
    monkeypatch.delenv("HF_TOKEN", raising=False)

    diarizer = PyannoteDiarizer()
    with pytest.raises(MLServiceError, match="HF_TOKEN"):
        diarizer.diarize(audio_path, [Segment(start=0.0, end=1.0, text="hi")])


def test_segments_are_labelled_by_greatest_temporal_overlap(monkeypatch, tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    tracks = [(0.0, 5.0, "SPEAKER_A"), (5.0, 10.0, "SPEAKER_B")]
    _install_fake_pyannote(monkeypatch, _fake_pipeline_factory(tracks))

    diarizer = PyannoteDiarizer()
    segments = [
        Segment(start=0.5, end=4.5, text="mostly in speaker A's turn"),
        Segment(start=6.0, end=9.0, text="entirely in speaker B's turn"),
    ]
    result = diarizer.diarize(audio_path, segments)

    # Pseudonymous, renumbered by first chronological appearance, not the
    # raw pyannote label.
    assert result.segments[0].speaker == "SPEAKER_00"
    assert result.segments[1].speaker == "SPEAKER_01"
    assert result.speaker_count == 2
    # original segment content (start/end/text) is preserved, not re-segmented
    assert result.segments[0].start == 0.5
    assert result.segments[0].text == "mostly in speaker A's turn"


def test_segment_with_no_overlap_falls_back_to_nearest_turn(monkeypatch, tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    tracks = [(0.0, 2.0, "SPEAKER_A"), (20.0, 22.0, "SPEAKER_B")]
    _install_fake_pyannote(monkeypatch, _fake_pipeline_factory(tracks))

    diarizer = PyannoteDiarizer()
    # falls entirely in the silence gap between the two turns, but closer to A
    segments = [Segment(start=3.0, end=4.0, text="in the gap")]
    result = diarizer.diarize(audio_path, segments)

    assert result.segments[0].speaker == "SPEAKER_00"  # nearest = SPEAKER_A


def test_no_turns_at_all_degrades_to_one_inferred_speaker(monkeypatch, tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    _install_fake_pyannote(monkeypatch, _fake_pipeline_factory([]))

    diarizer = PyannoteDiarizer()
    segments = [Segment(start=0.0, end=1.0, text="hello")]
    result = diarizer.diarize(audio_path, segments)

    assert result.speaker_count == 1
    assert result.segments[0].speaker == "SPEAKER_00"
