"""Cue-dictionary coverage: English/Hindi are expected to fire; the other
languages are structurally present but must stay marked unevaluated."""

from __future__ import annotations

from backend.app.ml.base import Segment
from backend.app.ml.multilingual_cues import (
    HINDI,
    cues_for,
    evaluated_languages,
    split_into_sentences,
)


def test_english_is_always_included_even_for_unknown_language_hints():
    langs = cues_for(["xx-not-a-real-code"])
    assert {c.code for c in langs} == {"en"}


def test_hindi_cues_are_included_when_detected():
    langs = cues_for(["hi"])
    assert {c.code for c in langs} == {"en", "hi"}


def test_no_duplicate_cue_sets_when_language_repeats():
    langs = cues_for(["hi", "hi-IN"])
    codes = [c.code for c in langs]
    assert codes.count("hi") == 1


def test_english_decision_and_action_cues_fire():
    from backend.app.ml.multilingual_cues import ENGLISH

    assert ENGLISH.decision_cues.search("We have decided to proceed.")
    assert ENGLISH.action_cues.search("Priya will send the report.")
    assert not ENGLISH.decision_cues.search("The weather has been pleasant.")


def test_hindi_devanagari_decision_cue_fires():
    from backend.app.ml.multilingual_cues import HINDI

    assert HINDI.decision_cues.search("हमने तय किया है कि आगे बढ़ेंगे।")


def test_hindi_romanised_obligation_cue_fires():
    from backend.app.ml.multilingual_cues import HINDI

    assert HINDI.action_cues.search("Aarti ko report prepare karna hai.")


def test_hedge_cues_catch_common_hedging_language():
    from backend.app.ml.multilingual_cues import ENGLISH

    assert ENGLISH.hedge_cues.search("Maybe we might move to Redis.")
    assert not ENGLISH.hedge_cues.search("We have decided to proceed.")


def test_only_english_and_hindi_are_marked_evaluated():
    """Marathi/Bengali/Gujarati/Awadhi exist so the architecture is not
    English/Hindi-only by construction, but must not be silently promoted to
    "evaluated" -- that would misrepresent untested coverage as measured
    accuracy."""
    assert evaluated_languages() == {"en", "hi"}


def test_awadhi_is_structurally_present_but_unevaluated():
    langs = cues_for(["awa"])
    assert {c.code for c in langs} == {"en", "awa"}
    assert "awa" not in evaluated_languages()


# --------------------------------------------------------------------------- #
# split_into_sentences -- the 2026-09-22 fix for real, punctuation-free Hindi
# ASR output collapsing an entire transcript into one "sentence". See the
# function's own docstring in multilingual_cues.py for the tiered strategy,
# and docs/integration.md for how this was found (a real noisy Hindi
# recording, not a synthetic case).
# --------------------------------------------------------------------------- #


def test_english_transcript_with_punctuation_is_unaffected():
    """Tier 1 (punctuation) must behave byte-for-byte as before -- this is
    the regression guard for every existing clean-fixture evaluation number."""
    text = "We reviewed the budget. Priya will send the report by Friday. The meeting ended."
    sentences = split_into_sentences(text)
    assert [s for s, _, _ in sentences] == [
        "We reviewed the budget.",
        "Priya will send the report by Friday.",
        "The meeting ended.",
    ]


def test_hindi_transcript_with_no_punctuation_no_longer_collapses():
    """The exact bug: real Whisper output on noisy Hindi audio with zero
    sentence terminators. Before the fix this returned a single span covering
    the whole 2000+ character transcript."""
    text = (
        "नमस्ते सब को आज की मेटिंग शुरू करते हैं, आज हमारे एजिन्दा में तीन मुदे हैं, "
        "पहला डिलीवरी में हो रही देरी, दुसरा गोदाम का स्टोक और तीसरा ग्राहकों का पेमेंट, "
        "नमस्ते प्रिया जी, नमस्ते राहुल, पहले डिलीवरी से शुरू करो"
    )
    sentences = split_into_sentences(text)
    assert len(sentences) > 1
    # Every span must be a literal, verbatim substring of the input -- the
    # anti-hallucination contract this whole project enforces, applied to
    # segmentation too.
    for s, start, end in sentences:
        assert text[start:end].strip() == s


def test_english_transcript_with_sparse_punctuation_falls_back():
    """One period across a long transcript is 'extremely sparse', not absent
    -- must still engage the comma fallback rather than trust the single
    40+-word 'sentence' the lone period implies."""
    text = (
        "we opened with the budget review, then moved to delivery delays, "
        "then discussed warehouse stock levels, then covered customer payments, "
        "then discussed the hiring plan for next quarter, then reviewed the "
        "marketing calendar, then closed with next steps and a reminder about "
        "the holiday schedule and the postponed client visit."
    )
    assert len(text.split()) > 40, "test text must exceed the sparsity threshold"
    sentences = split_into_sentences(text)
    assert len(sentences) > 1


def test_mixed_language_transcript_splits_on_available_terminators():
    text = "Today we discuss project status. Kal tak report submit karni hai."
    sentences = split_into_sentences(text)
    assert [s for s, _, _ in sentences] == [
        "Today we discuss project status.",
        "Kal tak report submit karni hai.",
    ]


def test_timestamp_based_segmentation_is_preferred_when_segments_align():
    """Real STT segment boundaries take priority over both punctuation and the
    comma fallback when the caller has them and they align with `text`."""
    text = "पहला मुद्दा डिलीवरी दूसरा मुद्दा स्टोक तीसरा मुद्दा पेमेंट"
    segments = [
        Segment(start=0.0, end=2.0, text="पहला मुद्दा डिलीवरी"),
        Segment(start=2.0, end=4.0, text="दूसरा मुद्दा स्टोक"),
        Segment(start=4.0, end=6.0, text="तीसरा मुद्दा पेमेंट"),
    ]
    sentences = split_into_sentences(text, segments)
    assert [s for s, _, _ in sentences] == [
        "पहला मुद्दा डिलीवरी",
        "दूसरा मुद्दा स्टोक",
        "तीसरा मुद्दा पेमेंट",
    ]
    # offsets must be real, contiguous positions into `text`
    for s, start, end in sentences:
        assert text[start:end] == s


def test_segment_alignment_failure_falls_back_to_text_based_tiers():
    """If a segment's text cannot be located in `text` (e.g. PII scrubbing
    diverged between the two independent scrub passes -- see
    agent/orchestrator.py::_persist_transcript), segment-based splitting is
    abandoned for the whole call rather than silently misattributing offsets."""
    text = "one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve"
    bogus_segments = [Segment(start=0.0, end=1.0, text="this text is not in the transcript")]
    sentences = split_into_sentences(text, bogus_segments)
    # falls through to the comma fallback (12 words, no terminators) rather
    # than raising or returning something misaligned
    assert len(sentences) > 1
    for s, start, end in sentences:
        assert text[start:end].strip() == s


def test_no_timestamp_fallback_uses_punctuation_then_commas():
    """No `segments` argument at all -- every existing caller's current
    behaviour -- must still work exactly as the text-only tiers describe."""
    text = "बजट पर चर्चा हुई, डिलीवरी में देरी हुई, अगली बैठक सोमवार को होगी"
    sentences = split_into_sentences(text, segments=None)
    assert len(sentences) == 3


def test_very_short_transcript_is_not_treated_as_sparse():
    """Below the word-count floor, one sentence is a plausible real answer,
    not a sign the punctuation split degenerated -- must not engage the
    comma fallback for genuinely short input."""
    text = "ठीक है धन्यवाद"  # "okay, thank you" -- 2 words, no punctuation
    sentences = split_into_sentences(text)
    assert len(sentences) == 1
    assert sentences[0][0] == text


def test_transcript_that_should_remain_one_segment_does():
    """A short, single-topic, well-punctuated transcript must not be
    artificially split just because the fallback machinery now exists."""
    text = "The meeting is postponed."
    sentences = split_into_sentences(text)
    assert len(sentences) == 1
    assert sentences[0][0] == text


def test_empty_text_returns_no_sentences():
    assert split_into_sentences("") == []


# --------------------------------------------------------------------------- #
# Hindi owner extraction (Devanagari script) -- 2026-09-22.
# --------------------------------------------------------------------------- #


def _owner(text: str) -> str | None:
    for pattern in HINDI.owner_patterns:
        m = pattern.search(text)
        if m:
            return m.group(1)
    return None


def test_hindi_owner_with_ko_marker():
    assert _owner("पूजा को रिपोर्ट तैयार करनी है") == "पूजा"


def test_hindi_owner_with_ne_marker():
    assert _owner("अमित ने यह काम पूरा करना है") == "अमित"


def test_english_owner_pattern_unaffected():
    from backend.app.ml.multilingual_cues import ENGLISH

    for pattern in ENGLISH.owner_patterns:
        m = pattern.search("Priya will send the report by Friday.")
        if m:
            assert m.group(1) == "Priya"
            return
    raise AssertionError("no English owner pattern matched")


def test_mixed_language_owner_romanised_ko():
    """Code-mixed speech: a Latin-script name with a romanised Hindi case
    marker -- the pattern this project already had before 2026-09-22,
    unaffected by adding the Devanagari-script patterns alongside it."""
    assert _owner("Aarti ko report prepare karna hai") == "Aarti"


def test_no_owner_stated_in_hindi_remains_none():
    assert _owner("रिपोर्ट अगले सप्ताह तैयार होगी") is None


def test_ambiguous_hindi_owner_is_not_invented():
    """A subject named at the very start of a sentence, with the verb far
    away (Hindi's normal SOV order), is a known, documented gap -- extracting
    nothing is correct here, not a bug: inventing a guess would be worse."""
    assert _owner("राहुल कल तक रिपोर्ट भेजेगा") is None
