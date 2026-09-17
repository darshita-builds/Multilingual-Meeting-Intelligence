"""Precision/recall/F1 for decision extraction against annotated gold data.

The blueprint asks for "accuracy and/or precision/recall/F1". Accuracy is not
meaningful here: decision extraction is not classification over a fixed set of
candidates, it is variable-count information extraction, so there is no
natural denominator for "correct out of how many". Precision/recall/F1 over
matched spans is used instead -- the standard choice for this task, and the
same approach `action_metrics.py` takes.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.ml.evaluation.extraction_matching import MatchResult, match_spans


@dataclass
class DecisionPrediction:
    text: str


@dataclass
class DecisionGold:
    text: str


def evaluate_decisions(
    predicted: list[DecisionPrediction], gold: list[DecisionGold], threshold: float = 0.5
) -> MatchResult:
    return match_spans([p.text for p in predicted], [g.text for g in gold], threshold=threshold)
