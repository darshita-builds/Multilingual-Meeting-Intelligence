"""EmbeddingAgendaSegmenter, with the embedding call mocked -- no model
download, but real array math (numpy) exercising the actual boundary-
detection logic against a synthetic 4-topic transcript."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.ml.base import MLServiceError
from backend.app.ml.embedding_segmenter import EmbeddingAgendaSegmenter

# One-hot "topic" vectors: within a topic, every sentence gets the identical
# vector (similarity 1.0); across topics, vectors are orthogonal (similarity
# 0.0). This produces an unambiguous, hand-verifiable boundary signal without
# needing a real embedding model.
_TOPIC_VECTORS = [np.eye(4)[i] for i in range(4)]

SYNTHETIC_TRANSCRIPT = (
    "The budget for this quarter needs review. "
    "We spent more than expected on marketing. "
    "Finance will send updated numbers next week. "
    "The project deadline is approaching fast. "
    "We need to finish testing by Friday. "
    "The client expects delivery next month. "
    "Team responsibilities were discussed today. "
    "Everyone has a clear role now. "
    "New hires start next week. "
    "The next meeting is scheduled for Monday. "
    "Please prepare your status updates. "
    "We will review progress then."
)


def _four_topic_encode(model_id, sentences):
    return np.array([_TOPIC_VECTORS[i // 3] for i in range(len(sentences))])


def _flat_encode(model_id, sentences):
    """Every sentence gets the same vector -- no topic shift anywhere."""
    return np.array([[1.0, 0.0]] * len(sentences))


def test_finds_four_topic_boundaries_in_a_synthetic_transcript(monkeypatch):
    monkeypatch.setattr("backend.app.ml.embedding_segmenter.embed_sentences", _four_topic_encode)
    segmenter = EmbeddingAgendaSegmenter()

    result = segmenter.segment(SYNTHETIC_TRANSCRIPT)

    assert len(result.blocks) == 4
    assert [b.position for b in result.blocks] == [0, 1, 2, 3]
    # contiguous, non-overlapping spans covering the transcript in order
    for prev, nxt in zip(result.blocks, result.blocks[1:], strict=False):
        assert prev.end_char <= nxt.start_char
    # the three real boundaries (deep valleys) should be reported confidently
    for block in result.blocks[:3]:
        assert block.confidence > 0.8


def test_flat_similarity_returns_exactly_one_block_never_zero(monkeypatch):
    """No real topic shift anywhere -- the contract requires >=1 block, since
    the platform treats zero blocks as a failed run."""
    monkeypatch.setattr("backend.app.ml.embedding_segmenter.embed_sentences", _flat_encode)
    segmenter = EmbeddingAgendaSegmenter()

    result = segmenter.segment("One. Two. Three. Four. Five. Six.")

    assert len(result.blocks) == 1
    assert result.blocks[0].start_char == 0


def test_empty_text_returns_one_block_not_zero():
    segmenter = EmbeddingAgendaSegmenter()
    result = segmenter.segment("   ")
    assert len(result.blocks) == 1
    assert result.blocks[0].confidence == 0.0


def test_single_sentence_returns_one_block():
    segmenter = EmbeddingAgendaSegmenter()
    result = segmenter.segment("Just one sentence here.")
    assert len(result.blocks) == 1


def test_missing_sentence_transformers_raises_ml_service_error(monkeypatch):
    def _explode(model_id, sentences):
        raise ImportError("No module named 'sentence_transformers'")

    monkeypatch.setattr("backend.app.ml.embedding_segmenter.embed_sentences", _explode)
    segmenter = EmbeddingAgendaSegmenter()

    with pytest.raises(MLServiceError, match="sentence-transformers"):
        segmenter.segment("First sentence. Second sentence. Third sentence.")


def test_generic_embedding_failure_is_wrapped_as_ml_service_error(monkeypatch):
    def _explode(model_id, sentences):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("backend.app.ml.embedding_segmenter.embed_sentences", _explode)
    segmenter = EmbeddingAgendaSegmenter()

    with pytest.raises(MLServiceError):
        segmenter.segment("First sentence. Second sentence. Third sentence.")


def test_confidence_floor_when_there_is_no_valley():
    assert EmbeddingAgendaSegmenter._confidence([0.5, 0.5, 0.5], 1) == 0.15


def test_confidence_ceiling_on_a_very_deep_valley():
    assert EmbeddingAgendaSegmenter._confidence([1.0, -2.0, 1.0], 1) == 0.95
