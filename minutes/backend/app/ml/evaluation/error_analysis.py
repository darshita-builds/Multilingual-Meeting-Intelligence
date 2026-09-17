"""Structured error analysis over matched/unmatched extraction results.

Produces a flat list of typed error records -- one per false positive, false
negative, and slot-filling miss among true positives -- so `report.md` shows
concrete failure examples, not only aggregate numbers. This is what "error
analysis" means as a deliverable: a reviewable list of what went wrong and
why, not a metric.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.ml.evaluation.action_metrics import ActionItemGold, ActionItemPrediction
from backend.app.ml.evaluation.decision_metrics import DecisionGold, DecisionPrediction
from backend.app.ml.evaluation.extraction_matching import MatchResult

Kind = str  # "false_positive" | "false_negative" | "wrong_owner" | "wrong_deadline"


@dataclass
class ErrorRecord:
    scenario: str
    kind: Kind
    item_type: str  # "decision" | "action"
    text: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "kind": self.kind,
            "item_type": self.item_type,
            "text": self.text,
            "detail": self.detail,
        }


def analyse_decisions(
    scenario: str,
    predicted: list[DecisionPrediction],
    gold: list[DecisionGold],
    match: MatchResult,
) -> list[ErrorRecord]:
    matched_pred = {p for p, _ in match.matched_pairs}
    matched_gold = {g for _, g in match.matched_pairs}
    errors = [
        ErrorRecord(scenario, "false_positive", "decision", p.text)
        for i, p in enumerate(predicted)
        if i not in matched_pred
    ]
    errors += [
        ErrorRecord(scenario, "false_negative", "decision", g.text)
        for i, g in enumerate(gold)
        if i not in matched_gold
    ]
    return errors


def analyse_actions(
    scenario: str,
    predicted: list[ActionItemPrediction],
    gold: list[ActionItemGold],
    match: MatchResult,
) -> list[ErrorRecord]:
    matched_pred = {p for p, _ in match.matched_pairs}
    matched_gold = {g for _, g in match.matched_pairs}
    errors = [
        ErrorRecord(scenario, "false_positive", "action", p.text)
        for i, p in enumerate(predicted)
        if i not in matched_pred
    ]
    errors += [
        ErrorRecord(scenario, "false_negative", "action", g.text)
        for i, g in enumerate(gold)
        if i not in matched_gold
    ]
    for p_idx, g_idx in match.matched_pairs:
        p, g = predicted[p_idx], gold[g_idx]
        if g.owner_name and (p.owner_name or "").strip().lower() != g.owner_name.strip().lower():
            errors.append(
                ErrorRecord(
                    scenario,
                    "wrong_owner",
                    "action",
                    p.text,
                    f"got {p.owner_name!r}, expected {g.owner_name!r}",
                )
            )
        if g.deadline and (p.deadline or "").strip().lower() != g.deadline.strip().lower():
            errors.append(
                ErrorRecord(
                    scenario,
                    "wrong_deadline",
                    "action",
                    p.text,
                    f"got {p.deadline!r}, expected {g.deadline!r}",
                )
            )
    return errors
