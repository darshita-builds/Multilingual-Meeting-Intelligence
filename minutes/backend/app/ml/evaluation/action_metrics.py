"""Precision/recall/F1 for action-item extraction against annotated gold data.

Matching is text-span based (`extraction_matching.match_spans`), not exact
string equality, because an extractor's `text` field is not guaranteed to
reproduce a gold annotation's exact tokenisation.

Beyond the span-level P/R/F1 the acceptance checklist asks for, this also
reports slot-level accuracy (owner, deadline) restricted to matched pairs --
whether the extractor found the right *sentence* is a different question from
whether it filled in the right owner, and folding them into one number would
hide a real failure mode: a correctly identified action with a wrong or
missing owner.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.ml.evaluation.extraction_matching import MatchResult, match_spans


@dataclass
class ActionItemPrediction:
    text: str
    owner_name: str | None = None
    deadline: str | None = None


@dataclass
class ActionItemGold:
    text: str
    owner_name: str | None = None
    deadline: str | None = None


@dataclass
class ActionMetrics:
    span: MatchResult
    #: None means "no matched item had a gold owner/deadline to check" --
    #: distinct from 0.0, which would mean "checked, and always wrong".
    owner_accuracy: float | None
    deadline_accuracy: float | None

    def as_dict(self) -> dict:
        return {
            **self.span.as_dict(),
            "owner_accuracy": self.owner_accuracy,
            "deadline_accuracy": self.deadline_accuracy,
        }


def evaluate_actions(
    predicted: list[ActionItemPrediction], gold: list[ActionItemGold], threshold: float = 0.5
) -> ActionMetrics:
    span = match_spans([p.text for p in predicted], [g.text for g in gold], threshold=threshold)

    owner_checks = owner_correct = 0
    deadline_checks = deadline_correct = 0
    for p_idx, g_idx in span.matched_pairs:
        p, g = predicted[p_idx], gold[g_idx]
        if g.owner_name is not None:
            owner_checks += 1
            owner_correct += _normalise(p.owner_name) == _normalise(g.owner_name)
        if g.deadline is not None:
            deadline_checks += 1
            deadline_correct += _normalise(p.deadline) == _normalise(g.deadline)

    return ActionMetrics(
        span=span,
        owner_accuracy=round(owner_correct / owner_checks, 4) if owner_checks else None,
        deadline_accuracy=round(deadline_correct / deadline_checks, 4) if deadline_checks else None,
    )


def _normalise(value: str | None) -> str | None:
    return value.strip().lower() if value else None
