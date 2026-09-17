"""Decision and action extraction via multilingual sentence embeddings.

Route 1 from the module's original docstring ("fine-tuned classifier... cheap,
fast, fully offline, probabilities are naturally calibrated confidences"), with
one substitution: instead of fine-tuning DistilBERT on labelled sentences (no
labelled corpus exists yet -- see `docs/ml-evaluation.md`), this classifies by
**embedding similarity to a small set of hand-written decision/action
exemplars** ("prototypes"). It needs no training data, runs on CPU, and stays
genuinely distinct from `baseline.py`'s regex matching: a sentence that means
"the launch date is locked in" without using any of the baseline's cue words
still scores high here, because the embedding captures meaning, not surface
form.

Two-stage design
-----------------
1. **Classification** -- cosine similarity between a sentence's embedding and
   the decision/action prototype sets (multilingual, see below) decides
   *whether* a sentence is a decision, an action, or neither, via a two-tier
   gate calibrated against the fixture corpus (`fixtures/meetings/`, see
   `docs/ml-evaluation.md` for the numbers):

     * similarity >= `threshold_high` -- accepted outright. The embedding
       alone is confident enough ("It was agreed that we will go with the
       managed Postgres option." scores 0.69 against the decision prototypes).
     * `threshold_low` <= similarity < `threshold_high` -- accepted only if a
       literal cue word from `multilingual_cues.py` also matches. This
       recovers true decisions the embedding alone under-scores ("We have
       decided to postpone the Hindi localisation release" scores only 0.26 --
       "postpone" pulls it away from the prototypes' proceed/approve framing --
       but its own text contains the literal cue "we have decided"). A single
       weak signal (embedding OR cue) is not enough on its own; the two
       corroborate each other. This is what keeps transitional phrases like
       "Moving on to the next item" (0.40 similarity, no cue) from being
       accepted -- one signal alone, of either kind, is exactly the failure
       mode that produced false positives during calibration.

   `confidence` is the raw similarity passed through a sigmoid centred below
   `threshold_high`, nudged up when a cue also matched (corroborating
   evidence) and down when a hedge word is present (calibration requirement:
   hedged statements must score lower than firm ones). A cue-only-rescued item
   therefore reports a real but modest confidence, not the same confidence as
   a high-similarity item -- which is the honest signal, not a bug.
2. **Slot filling** -- owner and deadline are *not* inferred from the
   embedding. They come from `multilingual_cues.py`'s regex patterns, exactly
   as the stub's suggested implementation describes ("owner and deadline then
   come from a separate extraction pass"). If no pattern matches, the field is
   `None` -- never guessed.

Anti-hallucination
-------------------
`evidence_quote` is always the literal sentence span matched out of
`block.text` by the same offset-preserving splitter the segmenter uses, so it
is a verbatim substring by construction -- there is no path that fabricates or
paraphrases a quote.

Multilingual coverage
----------------------
Classification (the embedding step) works in any of the ~50 languages the
embedding model covers, prototypes included for English and Hindi. Owner/
deadline slot-filling only has patterns for English and Hindi (see
`multilingual_cues.py`); Marathi/Bengali/Gujarati sentences can still be
classified as decisions/actions, just without an owner or deadline extracted --
that degrades to `None`, which is the contract-correct behaviour, not a bug.

Code-mixed meetings are routinely written in Latin script even when the
matrix language is Hindi ("Aarti ko ... prepare karna hai" -- no Devanagari
character in sight). `_languages_for` therefore checks for a short list of
common romanised Hindi function words (`hai`, `karna`, `ko`, `ki`, ...) in
addition to the Devanagari script sniff, so Hindi cues (including the "ko"/
"ne" owner pattern) still apply to fully-romanised code-mixed sentences. This
list is intentionally small and un-evaluated beyond the one fixture scenario
that exercises it (`02_code_mixed_hindi_english`) -- see `docs/ml-evaluation.md`.

Contract notes
--------------
* Raises `MLServiceError`, never a `sentence-transformers`/torch exception.
* Receives PII-scrubbed text; a redacted owner never surfaces as
  `"[REDACTED:EMAIL]"` -- the owner regex cannot match an all-caps bracketed
  token, and `_owner` discards an all-caps candidate defensively anyway.
* A meeting with nothing decided returns empty lists: sentences below
  `threshold` on both prototype sets, or shorter than `min_words`, are simply
  not emitted.
"""

from __future__ import annotations

import math
import os
import re

from backend.app.ml.base import (
    AgendaBlockResult,
    ExtractedAction,
    ExtractedDecision,
    ExtractionResult,
    MLServiceError,
)
from backend.app.ml.multilingual_cues import LanguageCues, cues_for
from backend.app.ml.sentence_embeddings import encode as embed_sentences

# The stub's original placeholder (`distilbert-base-multilingual-cased`) named a
# fine-tuned-classifier backbone; this implementation takes the embedding-
# similarity route instead (see module docstring), so the default model is the
# same multilingual sentence encoder the segmenter uses -- sharing it means the
# two Person-A model-based components load one set of weights between them.
DEFAULT_MODEL = os.getenv("MOM_EXTRACTOR_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
#: Similarity at/above which a sentence is accepted on the embedding alone.
DEFAULT_THRESHOLD_HIGH = float(os.getenv("MOM_EXTRACTOR_THRESHOLD_HIGH", "0.55"))
#: Similarity at/above which a sentence is accepted IF a cue word also
#: matches. Below this, neither signal is trusted alone. See the module
#: docstring's "Two-stage design" section for why both tiers exist.
DEFAULT_THRESHOLD_LOW = float(os.getenv("MOM_EXTRACTOR_THRESHOLD_LOW", "0.20"))
DEFAULT_MIN_WORDS = int(os.getenv("MOM_EXTRACTOR_MIN_WORDS", "4"))

_SENTENCE_SPLIT = re.compile(r"[^.!?।\n]+[.!?।]?")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_ALL_CAPS = re.compile(r"^[A-Z0-9_:\[\]]+$")
# Common romanised Hindi function words -- catches code-mixed sentences with
# no Devanagari character at all. Small and unevaluated beyond one fixture;
# see the module docstring's "Multilingual coverage" section.
_ROMANIZED_HINDI = re.compile(
    r"\b(hai|hain|nahi|karna|karo|kar|karenge|karega|karegi|rahega|rahegi|"
    r"tay|baat|agenda|aaj|hum|humne|ka|ki|ko|ne|se|tak)\b",
    re.I,
)

# Best-effort, hand-written exemplars -- not sourced from a corpus. English
# exemplars are checked against the fixture corpus; Hindi exemplars are
# included so code-mixed meetings are not English-only, but (like the Hindi
# entry in multilingual_cues.py) have not been separately evaluated at scale.
DECISION_PROTOTYPES = [
    "We have decided to proceed with the plan.",
    "It was agreed that the team will use the new design.",
    "The decision is final.",
    "We are going with option two.",
    "The proposal was approved by the group.",
    "Everyone signed off on the budget.",
    "हमने तय किया है कि हम आगे बढ़ेंगे।",
    "यह निर्णय अंतिम है।",
    "प्रस्ताव को मंजूरी दी गई।",
]

ACTION_PROTOTYPES = [
    "Priya will send the report by Friday.",
    "Rahul needs to review the document.",
    "Someone should prepare the presentation.",
    "I will follow up with the vendor.",
    "Please update the tracker before the meeting.",
    "The team is responsible for fixing the bug.",
    "राहुल को रिपोर्ट भेजनी है।",
    "हमें अगले हफ्ते तक यह पूरा करना है।",
]


class TransformerExtractor:
    """Embedding-similarity decision/action classifier with rule-based slot filling."""

    name = "transformer-extractor-v1"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        threshold_high: float = DEFAULT_THRESHOLD_HIGH,
        threshold_low: float = DEFAULT_THRESHOLD_LOW,
        min_words: int = DEFAULT_MIN_WORDS,
    ) -> None:
        self.model_id = model_id
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.min_words = min_words

    def extract(self, blocks: list[AgendaBlockResult]) -> ExtractionResult:
        flat: list[tuple[AgendaBlockResult, str]] = [
            (block, sentence) for block in blocks for sentence in self._sentences(block.text)
        ]
        if not flat:
            return ExtractionResult(decisions=[], actions=[], model_name=self._model_name())

        candidate_idx = [i for i, (_, s) in enumerate(flat) if len(s.split()) >= self.min_words]
        if not candidate_idx:
            return ExtractionResult(decisions=[], actions=[], model_name=self._model_name())

        try:
            proto_d = embed_sentences(self.model_id, DECISION_PROTOTYPES)
            proto_a = embed_sentences(self.model_id, ACTION_PROTOTYPES)
            sentence_vecs = embed_sentences(self.model_id, [flat[i][1] for i in candidate_idx])
        except ImportError as exc:
            raise MLServiceError(
                "sentence-transformers is not installed; install requirements-ml.txt "
                "or set MOM_EXTRACTOR_BACKEND=baseline"
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive, model-library-specific
            raise MLServiceError(f"extraction failed: {exc}") from exc

        decisions: list[ExtractedDecision] = []
        actions: list[ExtractedAction] = []

        for row, i in enumerate(candidate_idx):
            block, sentence = flat[i]
            vec = sentence_vecs[row]
            d_sim = float((proto_d @ vec).max())
            a_sim = float((proto_a @ vec).max())

            cue_sets = cues_for(self._languages_for(sentence))
            hedge_hit = any(c.hedge_cues.search(sentence) for c in cue_sets)
            decision_cue_hit = any(c.decision_cues.search(sentence) for c in cue_sets)
            action_cue_hit = any(c.action_cues.search(sentence) for c in cue_sets)

            passes_decision = self._passes(d_sim, decision_cue_hit)
            passes_action = self._passes(a_sim, action_cue_hit)
            if not passes_decision and not passes_action:
                continue

            if passes_decision and passes_action:
                # Both classes cleared the gate: an unambiguous cue on one side
                # only settles it ("We have decided to revisit this in two
                # weeks" has an explicit decision cue and no action cue, even
                # though its similarity to the action prototypes is close --
                # "revisit ... in two weeks" reads task-shaped). Otherwise the
                # higher-similarity class wins.
                if decision_cue_hit and not action_cue_hit:
                    classify_as_decision = True
                elif action_cue_hit and not decision_cue_hit:
                    classify_as_decision = False
                else:
                    classify_as_decision = d_sim >= a_sim
            else:
                classify_as_decision = passes_decision

            if classify_as_decision:
                decisions.append(
                    ExtractedDecision(
                        text=sentence,
                        confidence=self._confidence(d_sim, hedge_hit, decision_cue_hit),
                        evidence_quote=sentence,
                        agenda_position=block.position,
                    )
                )
            else:
                actions.append(
                    ExtractedAction(
                        text=sentence,
                        owner_name=self._owner(sentence, cue_sets),
                        deadline=self._deadline(sentence, cue_sets),
                        confidence=self._confidence(a_sim, hedge_hit, action_cue_hit),
                        evidence_quote=sentence,
                        agenda_position=block.position,
                    )
                )

        return ExtractionResult(decisions=decisions, actions=actions, model_name=self._model_name())

    # ------------------------------------------------------------------ #

    def _model_name(self) -> str:
        return f"{self.name} ({self.model_id})"

    def _passes(self, similarity: float, cue_hit: bool) -> bool:
        if similarity >= self.threshold_high:
            return True
        return similarity >= self.threshold_low and cue_hit

    def _confidence(self, similarity: float, hedge_hit: bool, cue_hit: bool) -> float:
        """Sigmoid centred below `threshold_high`, nudged by corroborating cues --
        see the module docstring's "Two-stage design" section."""
        centre = self.threshold_high - 0.15
        scaled = 1 / (1 + math.exp(-8 * (similarity - centre)))
        if cue_hit:
            scaled = min(0.95, scaled + 0.08)
        if hedge_hit:
            scaled = max(0.05, scaled - 0.25)
        return round(min(0.95, max(0.05, scaled)), 3)

    @staticmethod
    def _languages_for(sentence: str) -> list[str]:
        if _DEVANAGARI.search(sentence) or _ROMANIZED_HINDI.search(sentence):
            return ["hi"]
        return ["en"]

    @staticmethod
    def _owner(sentence: str, cue_sets: list[LanguageCues]) -> str | None:
        for cues in cue_sets:
            for pattern in cues.owner_patterns:
                m = pattern.search(sentence)
                if m:
                    candidate = m.group(1).strip()
                    if _ALL_CAPS.match(candidate):
                        continue  # a redaction placeholder or acronym, not a name
                    return candidate
        return None

    @staticmethod
    def _deadline(sentence: str, cue_sets: list[LanguageCues]) -> str | None:
        for cues in cue_sets:
            for pattern in cues.deadline_patterns:
                m = pattern.search(sentence)
                if m:
                    return (m.group(1) if m.groups() else m.group(0)).strip()
        return None

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [m.group(0).strip() for m in _SENTENCE_SPLIT.finditer(text) if m.group(0).strip()]
