"""EmbeddingAgendaSegmenter, with the embedding call mocked -- no model
download, but real array math (numpy) exercising the actual boundary-
detection logic against a synthetic 4-topic transcript."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.ml.base import MLServiceError, Segment
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


# --------------------------------------------------------------------------- #
# Real-world regression: punctuation-free Hindi ASR output no longer
# collapses into one block. See docs/integration.md's 2026-09-22 entry --
# a real noisy Hindi recording produced a Whisper transcript with zero
# sentence-terminating punctuation, which this segmenter's `len(sentences)
# <= 1` guard turned into exactly one agenda block for a 3-topic, 3-minute
# meeting.
# --------------------------------------------------------------------------- #


def test_punctuation_free_hindi_transcript_no_longer_collapses_to_one_block(monkeypatch):
    monkeypatch.setattr("backend.app.ml.embedding_segmenter.embed_sentences", _four_topic_encode)
    text = (
        "पहला मुद्दा डिलीवरी में देरी की बात है, दूसरी बार भी यही समस्या हुई, "
        "इस पर तुरंत ध्यान देना जरूरी है, "
        "दूसरा मुद्दा गोदाम का स्टोक है, तेल का स्टोक बहुत कम है, "
        "अगले हफ्ते तक अडर देना चाहिए, "
        "तीसरा मुद्दा ग्राहकों का पेमेंट है, तीन ग्राहकों का पेमेंट बकाया है, "
        "इस हफ्ते फोन करके बात करनी है, "
        "आखिरी मुद्दा अगली बैठक की तारीख है, सोमवार को अगली बैठक होगी, "
        "सब लोग समय पर आएं"
    )
    segmenter = EmbeddingAgendaSegmenter()

    result = segmenter.segment(text)

    assert len(result.blocks) > 1
    # contiguous, non-overlapping, and every block's text is real transcript
    # content -- nothing invented, matching the same contract the punctuated
    # path already guarantees.
    for prev, nxt in zip(result.blocks, result.blocks[1:], strict=False):
        assert prev.end_char <= nxt.start_char
    for block in result.blocks:
        assert block.text.strip()


def test_segments_are_used_as_the_top_priority_split_signal(monkeypatch):
    """When real STT segment boundaries are supplied and align with `text`,
    they are used directly rather than re-deriving boundaries from
    punctuation or commas -- reusing the same 4-topic synthetic transcript as
    the punctuation-based test above, but as unpunctuated STT segments (no
    periods at all) instead of sentences split on periods."""
    monkeypatch.setattr("backend.app.ml.embedding_segmenter.embed_sentences", _four_topic_encode)

    clauses = [c.rstrip(". ") for c in SYNTHETIC_TRANSCRIPT.split(". ") if c.strip()]
    text = " ".join(clauses)  # the same content, but with no terminating punctuation
    segments = []
    cursor = 0.0
    for c in clauses:
        segments.append(Segment(start=cursor, end=cursor + 1.0, text=c))
        cursor += 1.0
    segmenter = EmbeddingAgendaSegmenter()

    # Without segments, this text has zero punctuation and 66 words (well
    # above the sparsity floor) -- it would fall through to the comma
    # fallback, which finds no commas either here, so it would degrade to a
    # single block. Confirms segments really are doing the work below.
    assert len(segmenter.segment(text).blocks) == 1

    result = segmenter.segment(text, segments)

    assert len(result.blocks) == 4
    for prev, nxt in zip(result.blocks, result.blocks[1:], strict=False):
        assert prev.end_char <= nxt.start_char
