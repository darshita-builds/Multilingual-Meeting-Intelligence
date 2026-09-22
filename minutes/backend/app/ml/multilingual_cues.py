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

Also home to `split_into_sentences()` (see below), shared by
`embedding_segmenter.py` and `transformer_extractor.py` -- the fix for a real
bug found 2026-09-22 running real noisy Hindi audio through the pipeline (see
`docs/integration.md`): Whisper's Hindi output can contain zero sentence-
terminating punctuation, which collapsed a 3-minute, multi-topic meeting into
one giant "sentence" and, from there, one agenda block and one mis-scoped
action. `baseline.py` keeps its own, separate, unmodified sentence-splitting
regex -- it is the fixed control arm for the comparison study.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


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
    # "before Friday"/"before Monday" is a real calendar reference; "before the
    # cutover"/"before the board review" is a vague trigger phrase, not a date,
    # and must stay null (see docs/ml-evaluation.md's annotation protocol and
    # gold_annotations.json) -- a broader "before the <anything>" branch used to
    # capture "the cutover" itself as the deadline, which is exactly the
    # invented-deadline failure mode this module exists to avoid.
    _p(r"\bbefore\s+(\w+day)\b"),
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
        # Devanagari-script equivalent of the pattern above -- "पूजा को रिपोर्ट
        # तैयार करनी है" / "अमित ने यह काम पूरा करना है". Added 2026-09-22 after
        # real noisy Hindi audio showed owner extraction had no Devanagari
        # pattern at all, only the romanised one -- a name spoken and
        # transcribed entirely in Devanagari could never be found (see
        # docs/integration.md). Same precision tradeoff the English/romanised
        # patterns already accept: a common noun immediately before को/ने
        # ("टीम को", "the team") also matches, exactly as "Friday will..."
        # would match the English `will` pattern -- this is a keyword
        # heuristic, not named-entity recognition, and was already an
        # accepted limitation before this addition.
        #
        # Deliberately NOT `\b`-terminated: Python's `\b` is defined via `\w`,
        # and `\w` does not include Unicode category Mc (spacing combining
        # marks) -- which is what a Devanagari dependent vowel sign (matra,
        # e.g. the ो in को) *is*. `को\b` silently fails to match "को" at the
        # end of a word before whitespace, because position is Mc-then-space,
        # which `\b` does not see as a word/non-word transition. Verified by
        # direct testing (`को\b` against "पूजा को रिपोर्ट" returns no match at
        # all) before landing on `(?![ऀ-ॿ])` -- "not immediately
        # followed by another Devanagari character" -- as the Unicode-correct
        # equivalent, which also correctly rejects "कोई" (a different word
        # that happens to start with "को") tested against the same case.
        _p(r"([ऀ-ॣ०-ॿ]{2,20}?)\s+(?:को|ने)(?![ऀ-ॿ])"),
        # A verb-adjacency pattern for the future tense ("राहुल ... भेजेगा")
        # was tried and deliberately dropped: Hindi is SOV (subject-object-
        # verb), so in a real sentence like "राहुल कल तक रिपोर्ट भेजेगा" the
        # word immediately before the verb is the *object* ("रिपोर्ट", report),
        # not the subject ("राहुल") -- adjacency-based matching gets the wrong
        # word. A clause-initial heuristic (first word of the sentence, if it
        # ends in a future-tense verb) was tried next and also rejected: it
        # matches discourse markers with equal confidence -- "अब हम रिपोर्ट
        # भेजेंगे" ("now we will send the report", no person named at all)
        # extracts "अब" ("now") as the owner, tested and confirmed. Shipping
        # either would risk a false owner more often than it would find a
        # real one, which is a worse failure than returning `None`. The
        # "राहुल कल तक रिपोर्ट भेजेगा"-style construction is therefore a known,
        # deliberate gap -- see docs/integration.md and docs/ml-evaluation.md.
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

# Awadhi is written in Devanagari and closely related to Hindi -- close enough
# that no independently-verified Awadhi-specific decision/action vocabulary
# was available to write this against (unlike Marathi/Bengali/Gujarati, which
# are more lexically distinct from Hindi). Rather than invent dialect-specific
# forms that cannot be checked, this entry deliberately reuses the Hindi
# decision/action/hedge cues verbatim -- both because formal/business Awadhi
# speech commonly draws on shared Hindi vocabulary for exactly these concepts,
# and because doing otherwise would mean asserting linguistic forms with no
# way to verify them, which is a worse failure than acknowledging the overlap.
# `evaluated=False`, same as the other three -- no fixture, no annotation, no
# measured recall for Awadhi specifically.
AWADHI = LanguageCues(
    code="awa",
    name="Awadhi",
    decision_cues=HINDI.decision_cues,
    action_cues=HINDI.action_cues,
    hedge_cues=HINDI.hedge_cues,
    evaluated=False,
)

REGISTRY: dict[str, LanguageCues] = {
    c.code: c for c in (ENGLISH, HINDI, MARATHI, BENGALI, GUJARATI, AWADHI)
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


# --------------------------------------------------------------------------- #
# Robust sentence splitting -- shared by embedding_segmenter.py and
# transformer_extractor.py.
#
# Real bug, found 2026-09-22 running real noisy Hindi audio through the
# pipeline (see docs/integration.md): faster-whisper's Hindi output can
# contain zero sentence-terminating punctuation at all -- no `.`, no
# Devanagari danda `।`, no `!`/`?` -- across an entire multi-minute
# transcript. The original regex (`[^.!?।\n]+[.!?।]?`) then returns the whole
# transcript as one match, which collapsed a 3-minute, 3-agenda-item meeting
# into one agenda block and one action item scoped to the entire recording.
#
# Fix, in priority order, and why each tier cannot invent text:
#   1. Real STT segment boundaries, if the caller has them and each segment's
#      text can be found as a literal, sequential substring of `text`. These
#      are the ASR model's own detected utterance/pause boundaries, with real
#      timestamps -- the strongest available signal, not a text heuristic at
#      all. If alignment fails for even one segment (should not happen when
#      `text` really is `" ".join(s.text for s in segments)`, but PII
#      scrubbing runs on the flat text and the segment list through two
#      separate code paths -- see `agent/orchestrator.py::_persist_transcript`
#      -- and is not guaranteed to produce byte-identical results), this tier
#      is abandoned entirely for that call and tier 2 runs on the full text,
#      rather than silently misattributing a span.
#   2. The existing punctuation regex -- unchanged, byte-for-byte, whenever
#      terminators are not sparse. This is why every already-passing test and
#      every clean-fixture evaluation number is unaffected by this change.
#   3. Comma-delimited clauses, only when terminators are absent or very
#      sparse relative to word count. A comma is a token the ASR model itself
#      chose to emit; splitting on it reveals structure already present in
#      the transcript rather than inventing any.
#
# No tier ever fabricates text: every returned span is a literal substring of
# `text`, located by character offset, and the segmenter/extractor's own
# "never fewer than one block/never guess a field" contracts are unchanged.
# --------------------------------------------------------------------------- #

_SENTENCE_SPLIT = re.compile(r"[^.!?।\n]+[.!?।]?")
_CLAUSE_SPLIT = re.compile(r"[^,.!?।\n]+[,.!?।]?")
_TERMINATORS = re.compile(r"[.!?।]")

#: Below this word count, returning one span is plausible on its own merits,
#: not a sign the punctuation-based split degenerated -- do not engage the
#: comma fallback for a genuinely short input.
SPARSE_MIN_WORDS = 12
#: If the punctuation-implied average "sentence" is longer than this many
#: words, treat punctuation as too sparse to trust.
SPARSE_WORDS_PER_TERMINATOR = 40

Span = tuple[str, int, int]


def split_into_sentences(text: str, segments: Any = None) -> list[Span]:
    """Return `(sentence_text, start_char, end_char)` spans covering `text`.

    `segments` is optional and duck-typed: an iterable of objects or dicts
    each exposing `.text`/`["text"]` (matches both `ml.base.Segment` and the
    dict shape tool calls pass around). Passing `None` (the default, and
    still what every current caller in this codebase does) is exactly the
    previous behaviour's punctuation-then-nothing path, tier 3 included.
    """
    if segments:
        from_segments = _split_from_segments(text, segments)
        if from_segments is not None:
            return from_segments

    sentences = _split(text, _SENTENCE_SPLIT)
    if not sentences or not _is_sparse(text, sentences):
        return sentences

    clauses = _split(text, _CLAUSE_SPLIT)
    # Only adopt the comma-based split if it actually found more structure --
    # a transcript with neither terminators nor commas gains nothing from a
    # second identical pass, and the original (already-computed) result is
    # returned rather than silently no-op-ing through redundant work.
    return clauses if len(clauses) > len(sentences) else sentences


def _split_from_segments(text: str, segments: Any) -> list[Span] | None:
    out: list[Span] = []
    cursor = 0
    for seg in segments:
        raw = seg.text if hasattr(seg, "text") else seg.get("text", "")
        seg_text = (raw or "").strip()
        if not seg_text:
            continue
        idx = text.find(seg_text, cursor)
        if idx == -1:
            return None  # cannot reliably align -- fall back rather than guess
        out.append((seg_text, idx, idx + len(seg_text)))
        cursor = idx + len(seg_text)
    return out or None


def _is_sparse(text: str, sentences: list[Span]) -> bool:
    words = text.split()
    if len(words) < SPARSE_MIN_WORDS:
        return False
    terminators = len(_TERMINATORS.findall(text))
    if terminators == 0:
        return True
    return len(words) / terminators > SPARSE_WORDS_PER_TERMINATOR


def _split(text: str, pattern: re.Pattern[str]) -> list[Span]:
    out = []
    for m in pattern.finditer(text):
        s = m.group(0).strip()
        if s:
            out.append((s, m.start(), m.end()))
    return out
