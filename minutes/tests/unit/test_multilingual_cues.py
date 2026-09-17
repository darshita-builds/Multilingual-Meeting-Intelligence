"""Cue-dictionary coverage: English/Hindi are expected to fire; the other
three languages are structurally present but must stay marked unevaluated."""

from __future__ import annotations

from backend.app.ml.multilingual_cues import cues_for, evaluated_languages


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
    """Marathi/Bengali/Gujarati exist so the architecture is not English/Hindi
    -only by construction, but must not be silently promoted to "evaluated"
    -- that would misrepresent untested coverage as measured accuracy."""
    assert evaluated_languages() == {"en", "hi"}
