"""TransformerExtractor, with the embedding call mocked -- deterministic
control over prototype similarity so the gating/tie-break/confidence logic is
tested directly rather than depending on a real model's numbers."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.ml.base import AgendaBlockResult, MLServiceError
from backend.app.ml.transformer_extractor import (
    ACTION_PROTOTYPES,
    DECISION_PROTOTYPES,
    TransformerExtractor,
)

D = np.array([1.0, 0.0, 0.0])  # "close to the decision prototypes"
A = np.array([0.0, 1.0, 0.0])  # "close to the action prototypes"


def _block(text: str, position: int = 0) -> AgendaBlockResult:
    return AgendaBlockResult(
        position=position, title="t", text=text, start_char=0, end_char=len(text)
    )


def make_fake_encode(sentence_vectors: dict[str, np.ndarray]):
    """Prototype lists get a constant vector each; test sentences are looked
    up explicitly so a typo/unmapped sentence fails loudly instead of
    silently getting a zero vector."""

    def _fake(model_id, sentences):
        if list(sentences) == DECISION_PROTOTYPES:
            return np.array([D] * len(DECISION_PROTOTYPES))
        if list(sentences) == ACTION_PROTOTYPES:
            return np.array([A] * len(ACTION_PROTOTYPES))
        vectors = []
        for s in sentences:
            assert s in sentence_vectors, f"unmapped test sentence: {s!r}"
            vectors.append(sentence_vectors[s])
        return np.array(vectors)

    return _fake


def _patch(monkeypatch, sentence_vectors: dict[str, np.ndarray]) -> None:
    monkeypatch.setattr(
        "backend.app.ml.transformer_extractor.embed_sentences", make_fake_encode(sentence_vectors)
    )


def test_high_similarity_sentence_is_classified_as_decision(monkeypatch):
    text = "We have decided to proceed with the plan."
    _patch(monkeypatch, {text: np.array([0.9, 0.05, 0.05])})
    extractor = TransformerExtractor()

    result = extractor.extract([_block(text)])

    assert len(result.decisions) == 1
    assert result.actions == []
    assert result.decisions[0].evidence_quote in text
    assert result.decisions[0].confidence > 0.7


def test_high_similarity_sentence_is_classified_as_action_with_owner_and_deadline(monkeypatch):
    text = "Priya will send the report by Friday."
    _patch(monkeypatch, {text: np.array([0.05, 0.9, 0.05])})
    extractor = TransformerExtractor()

    result = extractor.extract([_block(text)])

    assert len(result.actions) == 1
    action = result.actions[0]
    assert action.owner_name == "Priya"
    assert action.deadline == "Friday"
    assert action.evidence_quote in text


def test_low_similarity_and_no_cue_is_not_extracted(monkeypatch):
    text = "The weather has been quite nice this week and everyone enjoyed lunch outside."
    _patch(monkeypatch, {text: np.array([0.05, 0.05, 0.9])})
    extractor = TransformerExtractor()

    result = extractor.extract([_block(text)])

    assert result.decisions == []
    assert result.actions == []


def test_empty_meeting_produces_no_invented_minutes(monkeypatch):
    """Guards against over-extraction: a social catch-up with several
    unrelated sentences must yield nothing, not a plausible-looking guess."""
    sentences = {
        "Hello everyone, thanks for joining the catch-up.": np.array([0.05, 0.05, 0.9]),
        "The weather has been quite pleasant this week.": np.array([0.05, 0.05, 0.9]),
        "Nice to see everyone, have a good weekend.": np.array([0.05, 0.05, 0.9]),
    }
    _patch(monkeypatch, sentences)
    extractor = TransformerExtractor()
    block = _block(" ".join(sentences))

    result = extractor.extract([block])

    assert result.decisions == []
    assert result.actions == []


def test_low_similarity_is_rescued_by_a_corroborating_cue(monkeypatch):
    """The exact failure mode calibration surfaced: 'we have decided to
    postpone...' scores well below the high-confidence threshold on
    embedding similarity alone, but its own text contains an explicit
    decision cue -- the two-tier gate must accept it."""
    text = "We have decided to postpone the release until next quarter."
    _patch(monkeypatch, {text: np.array([0.3, 0.05, 0.65])})  # d_sim=0.3: below high, above low
    extractor = TransformerExtractor()

    result = extractor.extract([_block(text)])

    assert len(result.decisions) == 1
    # rescued by a cue, not a strong embedding signal -- confidence should be
    # real but modest, not the same as a high-similarity item.
    assert 0.05 <= result.decisions[0].confidence < 0.7


def test_similarity_alone_below_low_threshold_is_not_rescued_by_absence_of_cue(monkeypatch):
    text = "Payment will go to the account as previously discussed."
    _patch(monkeypatch, {text: np.array([0.0, 0.1, 0.9])})  # a_sim=0.1: below even the low tier
    extractor = TransformerExtractor()

    result = extractor.extract([_block(text)])

    assert result.actions == []


def test_tie_break_prefers_the_class_with_an_unambiguous_cue(monkeypatch):
    """Both prototype sets clear the high threshold, but only the decision
    cue is present -- the calibration bug this regression-tests against."""
    text = "We have decided to revisit this in two weeks."
    _patch(monkeypatch, {text: np.array([0.6, 0.65, 0.0])})  # a_sim slightly > d_sim
    extractor = TransformerExtractor()

    result = extractor.extract([_block(text)])

    assert len(result.decisions) == 1
    assert result.actions == []


def test_hedged_statement_scores_lower_confidence_than_a_firm_one(monkeypatch):
    firm = "We have decided to proceed with the launch."
    hedged = "Maybe we have decided to reconsider, not sure yet."
    vec = np.array([0.7, 0.0, 0.3])
    _patch(monkeypatch, {firm: vec, hedged: vec})
    extractor = TransformerExtractor()

    result = extractor.extract([_block(f"{firm} {hedged}")])

    by_text = {d.text: d.confidence for d in result.decisions}
    assert by_text[firm] > by_text[hedged]


def test_owner_and_deadline_are_none_when_not_stated(monkeypatch):
    text = "Please review the proposal."
    _patch(monkeypatch, {text: np.array([0.05, 0.6, 0.35])})
    extractor = TransformerExtractor()

    result = extractor.extract([_block(text)])

    assert len(result.actions) == 1
    assert result.actions[0].owner_name is None
    assert result.actions[0].deadline is None


def test_vague_before_phrase_does_not_invent_a_deadline(monkeypatch):
    """'before the cutover' is a trigger phrase, not a calendar reference --
    gold_annotations.json marks this exact sentence deadline=None (see
    docs/ml-evaluation.md's annotation protocol). Regression test for a bug
    where the deadline regex's `before the \\w+` branch captured "the cutover"
    itself as the deadline string."""
    text = "Rahul needs to check the staging database backups before the cutover."
    _patch(monkeypatch, {text: np.array([0.05, 0.9, 0.05])})
    extractor = TransformerExtractor()

    result = extractor.extract([_block(text)])

    assert len(result.actions) == 1
    assert result.actions[0].owner_name == "Rahul"
    assert result.actions[0].deadline is None


def test_hindi_decision_is_extracted_with_verbatim_evidence(monkeypatch):
    """Real regression case, docs/integration.md 2026-09-22: before the
    segmentation fix, this kind of sentence was never seen by the extractor
    on its own -- it was buried inside one giant multi-minute block."""
    text = "तय हुआ कि अगली मीटिंग सोमवार को होगी"
    _patch(monkeypatch, {text: np.array([0.92, 0.05, 0.03])})
    extractor = TransformerExtractor()

    result = extractor.extract([_block(text)])

    assert len(result.decisions) == 1
    assert result.decisions[0].evidence_quote == text
    assert result.decisions[0].confidence > 0.5
    assert result.actions == []


def test_real_transcript_shaped_block_scopes_action_to_its_own_clause(monkeypatch):
    """The exact failure mode this whole fix targets: a block containing
    several comma-delimited clauses (what a punctuation-free real Hindi
    transcript looks like after the 2026-09-22 segmentation fix) must not
    attribute one clause's action to the whole block's text -- the deadline
    'कल तक' the extractor finds must be evidenced by, and scoped to, the one
    clause that actually says it."""
    decision_clause = "तय हुआ कि अगली मीटिंग सोमवार को होगी,"  # trailing comma preserved verbatim
    action_clause = "पूजा को रिपोर्ट कल तक तैयार करनी है"
    block_text = f"{decision_clause} {action_clause}"

    _patch(
        monkeypatch,
        {
            decision_clause: np.array([0.9, 0.05, 0.05]),
            action_clause: np.array([0.05, 0.88, 0.07]),
        },
    )
    extractor = TransformerExtractor()

    result = extractor.extract([_block(block_text)])

    assert len(result.decisions) == 1
    assert result.decisions[0].evidence_quote == decision_clause
    assert result.decisions[0].evidence_quote != block_text  # not the whole block

    assert len(result.actions) == 1
    action = result.actions[0]
    assert action.evidence_quote == action_clause
    assert action.evidence_quote != block_text  # the real bug: was the whole transcript
    assert action.owner_name == "पूजा"
    assert action.deadline == "कल तक"


def test_short_sentences_are_filtered_before_embedding(monkeypatch):
    """All sentences are below min_words, so the embedding call should never
    happen at all -- an empty vector map means the fake encoder would raise
    on any lookup, proving no embedding call was made."""
    _patch(monkeypatch, {})
    extractor = TransformerExtractor()

    result = extractor.extract([_block("Yes. No. Ok.")])

    assert result.decisions == []
    assert result.actions == []


def test_extract_with_no_blocks_returns_empty_result():
    extractor = TransformerExtractor()
    result = extractor.extract([])
    assert result.decisions == []
    assert result.actions == []


def test_romanised_hindi_owner_is_extracted_via_the_ko_case_marker(monkeypatch):
    text = "Aarti ko monthly report prepare karna hai."
    _patch(monkeypatch, {text: np.array([0.05, 0.6, 0.35])})
    extractor = TransformerExtractor()

    result = extractor.extract([_block(text)])

    assert len(result.actions) == 1
    assert result.actions[0].owner_name == "Aarti"


def test_missing_sentence_transformers_raises_ml_service_error(monkeypatch):
    def _explode(model_id, sentences):
        raise ImportError("No module named 'sentence_transformers'")

    monkeypatch.setattr("backend.app.ml.transformer_extractor.embed_sentences", _explode)
    extractor = TransformerExtractor()

    with pytest.raises(MLServiceError, match="sentence-transformers"):
        extractor.extract([_block("We have decided to proceed with the plan and ship it.")])
