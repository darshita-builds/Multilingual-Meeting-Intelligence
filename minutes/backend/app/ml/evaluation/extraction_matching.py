"""Shared span-matching logic for action/decision evaluation.

A predicted item's `text` rarely reproduces a gold annotation's exact string
(different whitespace normalisation between the baseline's `_normalise` and
the transformer extractor's raw sentence, for instance). Matching by word-set
overlap (Jaccard similarity) is standard practice for information-extraction
evaluation and survives that noise while still requiring real agreement --
two sentences about different things will not share enough words to match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "")}


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class MatchResult:
    true_positives: int
    false_positives: int
    false_negatives: int
    #: (predicted_index, gold_index) for every matched pair -- used by
    #: error_analysis.py to look up which items were and were not matched.
    matched_pairs: list[tuple[int, int]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return round(self.true_positives / denom, 4) if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return round(self.true_positives / denom, 4) if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return round(2 * p * r / (p + r), 4) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def match_spans(
    predicted_texts: list[str], gold_texts: list[str], threshold: float = 0.5
) -> MatchResult:
    """Greedy best-match bipartite matching by Jaccard word overlap.

    Greedy rather than optimal (Hungarian algorithm) is a deliberate
    simplification: gold sets here are a handful of items per meeting, where
    greedy and optimal matching essentially never disagree, and greedy is far
    simpler to read and audit.
    """
    remaining_gold = list(range(len(gold_texts)))
    matched_pairs: list[tuple[int, int]] = []
    for p_idx, p_text in enumerate(predicted_texts):
        best_idx, best_score = None, 0.0
        for g_idx in remaining_gold:
            score = jaccard(p_text, gold_texts[g_idx])
            if score > best_score:
                best_score, best_idx = score, g_idx
        if best_idx is not None and best_score >= threshold:
            matched_pairs.append((p_idx, best_idx))
            remaining_gold.remove(best_idx)

    tp = len(matched_pairs)
    return MatchResult(
        true_positives=tp,
        false_positives=len(predicted_texts) - tp,
        false_negatives=len(gold_texts) - tp,
        matched_pairs=matched_pairs,
    )
