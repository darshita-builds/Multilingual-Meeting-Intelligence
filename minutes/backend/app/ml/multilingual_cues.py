"""Multilingual keyword cues for decision/action extraction.

Shared by `transformer_extractor.py` (and available to any future extractor).
`baseline.py` has its own English-first regexes and is left alone -- it is the
fixed control arm for the comparison study, not a place to add languages.

Coverage is honest, not aspirational:

  * **English, Hindi** -- cue lists were written against the fixture corpus
    (`fixtures/meetings/02_code_mixed_hindi_english.txt`) and checked to fire on
    it. Treat these as evaluated, in the limited sense that one code-mixed
    scenario exists to check them against.
  * **Marathi, Bengali, Gujarati** -- a small starter list of common decision/
    action verbs, added so the architecture does not silently exclude them, but
    with `evaluated=False`: no fixture, no annotation, no measured recall. Do not
    report accuracy numbers for these languages without first building evaluation
    data (see `docs/ml-evaluation.md`).

Extending this file for a new language means adding one `LanguageCues` entry --
nothing in `transformer_extractor.py` needs to change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LanguageCues:
    code: str
    name: str
    decision_cues: re.Pattern[str]
    action_cues: re.Pattern[str]
    hedge_cues: re.Pattern[str]
    owner_patterns: tuple[re.Pattern[str], ...] = field(default_factory=tuple)
    deadline_patterns: tuple[re.Pattern[str], ...] = field(default_factory=tuple)
    #: Whether this language's cues have been checked against any annotated
    #: fixture. False means "present so the pipeline does not ignore the
    #: language entirely", not "measured".
    evaluated: bool = False


def _p(pattern: str, flags: int = re.I) -> re.Pattern[str]:
    return re.compile(pattern, flags)


_EMPTY_PATTERN = _p(r"(?!x)x")  # matches nothing; placeholder for un-populated fields

# --------------------------------------------------------------------------- #
# English
# --------------------------------------------------------------------------- #

_EN_DEADLINE = (
    _p(r"\bby\s+(\d{1,2}(?:st|nd|rd|th)?\s+\w+(?:\s+\d{4})?)"),
    _p(r"\bby\s+(eod|eow|tomorrow|today|tonight|next week|this week|month end)\b"),
    _p(r"\bby\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"),
    _p(r"\b(\d{4}-\d{2}-\d{2})\b"),
    _p(r"\bbefore\s+(\w+day|the\s+\w+)\b"),
)

_EN_OWNER = (
    _p(r"\b([A-Z][a-z]{2,15})\s+(?:will|to|should|needs? to|has to|is going to)\b", re.UNICODE),
    _p(r"\bassigned to\s+([A-Z][a-z]{2,15})"),
    _p(r"\bowner\s*(?:is|:)\s*([A-Z][a-z]{2,15})"),
)

ENGLISH = LanguageCues(
    code="en",
    name="English",
    decision_cues=_p(
        r"\b(we (?:have )?(?:decided|agreed|concluded|settled on)|"
        r"(?:it (?:was|is)|final) (?:decided|decision|agreed)|"
        r"the decision is|we(?:'| a)re going (?:to|with)|"
        r"approved|sign(?:ed)? off|we will go with|let'?s go with)\b"
    ),
    action_cues=_p(
        r"\b(will (?:"
        r"send|share|prepare|update|review|draft|check|fix|deploy|call|follow up|"
        r"run|start|submit|complete|schedule|handle|coordinate|finalise|finalize|"
        r"organise|organize|arrange|test|investigate|escalate|circulate|present|"
        r"distribute|own|lead)|"
        r"needs? to|has to|should|action item|take(?:s)? (?:this|that) up|"
        r"assign(?:ed)? to|owner is|responsible for|please (?:"
        r"send|share|prepare|update|review|draft|check|fix|deploy|call|submit)|"
        r"I'?ll |we'?ll |follow[- ]up|"
        r"by (?:eod|eow|tomorrow|next week|friday|monday))\b"
    ),
    hedge_cues=_p(r"\b(maybe|might|perhaps|possibly|not sure|tentatively|if (?:we|it))\b"),
    owner_patterns=_EN_OWNER,
    deadline_patterns=_EN_DEADLINE,
    evaluated=True,
)

# --------------------------------------------------------------------------- #
# Hindi (Devanagari + common code-mixed romanisation left to the English cues,
# since code-mixed speech interleaves English verbs -- "prepare karna hai" is
# caught by the English action cue on "prepare").
# --------------------------------------------------------------------------- #

HINDI = LanguageCues(
    code="hi",
    name="Hindi",
    decision_cues=_p(r"(तय हुआ|तय किया|फाइनल है|सहमति बनी|निर्णय लिया)"),
    # Devanagari cues, plus the same light-verb obligation construction
    # ("X karna hai" = "X needs to be done") written in Latin script -- code-
    # mixed meetings are routinely romanised end-to-end, not just word-mixed
    # with Devanagari (see transformer_extractor.py's "Multilingual coverage"
    # note on `_ROMANIZED_HINDI`).
    action_cues=_p(
        r"(करना है|करेंगे|ज़िम्मेदारी|कल तक|अगले हफ्ते तक|"
        r"\bkarna hai\b|\bkarna tha\b|\bkarni hai\b|\bkarne hai\b|"
        r"\bkarenge\b|\bkarega\b|\bkaregi\b)"
    ),
    hedge_cues=_p(r"(शायद|हो सकता है|अभी तय नहीं)"),
    owner_patterns=(
        # "Aarti ko ... karna hai" / "Rahul ne" -- Hindi case markers on a Latin
        # name in code-mixed speech.
        _p(r"\b([A-Z][a-z]{2,15})\s+(?:ko|ne)\b"),
    ),
    deadline_patterns=(_p(r"(कल तक|अगले हफ्ते तक|इस हफ्ते तक|महीने के अंत तक)"),),
    evaluated=True,
)

# --------------------------------------------------------------------------- #
# Marathi, Bengali, Gujarati -- starter coverage, NOT evaluated.
#
# These lists give the extension architecture a real example for three more
# languages so it is not English/Hindi-only by construction, but no fixture,
# annotation or measured recall backs them yet. `evaluated=False` is read by
# the evaluation language-breakdown report to keep these languages out of any
# accuracy claim until real data exists.
# --------------------------------------------------------------------------- #

MARATHI = LanguageCues(
    code="mr",
    name="Marathi",
    decision_cues=_p(r"(ठरले|ठरवले|मान्य झाले|निर्णय घेतला)"),
    action_cues=_p(r"(करायचे आहे|जबाबदारी|पुढच्या आठवड्यापर्यंत)"),
    hedge_cues=_p(r"(कदाचित|अजून ठरलेले नाही)"),
    evaluated=False,
)

BENGALI = LanguageCues(
    code="bn",
    name="Bengali",
    decision_cues=_p(r"(সিদ্ধান্ত হয়েছে|সিদ্ধান্ত নেওয়া হয়েছে|চূড়ান্ত)"),
    action_cues=_p(r"(করতে হবে|দায়িত্ব|আগামী সপ্তাহের মধ্যে)"),
    hedge_cues=_p(r"(হয়তো|এখনো ঠিক হয়নি)"),
    evaluated=False,
)

GUJARATI = LanguageCues(
    code="gu",
    name="Gujarati",
    decision_cues=_p(r"(નક્કી થયું|નિર્ણય લેવાયો|મંજૂર)"),
    action_cues=_p(r"(કરવાનું છે|જવાબદારી|આવતા અઠવાડિયા સુધીમાં)"),
    hedge_cues=_p(r"(કદાચ|હજુ નક્કી નથી)"),
    evaluated=False,
)

REGISTRY: dict[str, LanguageCues] = {
    c.code: c for c in (ENGLISH, HINDI, MARATHI, BENGALI, GUJARATI)
}

#: Always included: code-mixed meetings routinely carry English verbs
#: ("prepare", "will send") regardless of the matrix language.
_DEFAULT_LANGS = ("en",)


def cues_for(languages: list[str] | None) -> list[LanguageCues]:
    """Cue sets to check for a sentence, given the transcript's detected languages.

    English is always included (code-mixing means an English verb can carry the
    action/decision cue even in an otherwise non-English sentence). Unknown
    language codes are ignored rather than raised, since `detected_languages`
    comes from an STT model that may report codes this dictionary does not
    (yet) cover.
    """
    codes = list(_DEFAULT_LANGS)
    for lang in languages or []:
        code = (lang or "").split("-")[0].lower()
        if code in REGISTRY and code not in codes:
            codes.append(code)
    return [REGISTRY[c] for c in codes]


def evaluated_languages() -> set[str]:
    return {c.code for c in REGISTRY.values() if c.evaluated}
