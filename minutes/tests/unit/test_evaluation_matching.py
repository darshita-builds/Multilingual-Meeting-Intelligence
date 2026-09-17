"""Span matching (Jaccard) and the decision/action metrics built on it."""

from __future__ import annotations

from backend.app.ml.evaluation.action_metrics import (
    ActionItemGold,
    ActionItemPrediction,
    evaluate_actions,
)
from backend.app.ml.evaluation.decision_metrics import (
    DecisionGold,
    DecisionPrediction,
    evaluate_decisions,
)
from backend.app.ml.evaluation.extraction_matching import jaccard, match_spans


def test_jaccard_identical_text_is_one():
    assert jaccard("Priya will send the report", "Priya will send the report") == 1.0


def test_jaccard_disjoint_text_is_zero():
    assert jaccard("Priya will send a report", "cats enjoy sunny afternoons outside") == 0.0


def test_jaccard_empty_text_is_zero():
    assert jaccard("", "something") == 0.0
    assert jaccard("something", "") == 0.0


def test_match_spans_perfect_match():
    result = match_spans(
        ["Priya will send the report by Friday"], ["Priya will send the report by Friday"]
    )
    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.precision == result.recall == result.f1 == 1.0


def test_match_spans_extra_prediction_is_false_positive():
    result = match_spans(
        ["Priya will send the report", "totally unrelated sentence about weather"],
        ["Priya will send the report"],
    )
    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 0
    assert result.precision == 0.5
    assert result.recall == 1.0


def test_match_spans_missed_gold_is_false_negative():
    result = match_spans([], ["Priya will send the report"])
    assert result.true_positives == 0
    assert result.false_negatives == 1
    assert result.recall == 0.0
    assert result.precision == 0.0  # no predictions -> precision is defined as 0, not undefined


def test_match_spans_no_gold_no_predictions_does_not_divide_by_zero():
    """The 05_no_decisions_social case: empty meeting, empty gold, empty
    predictions. Precision/recall are conventionally undefined at 0/0; this
    implementation defines them as 0.0 rather than raising. That looks like a
    "failure" read in isolation, but it is harmless in the aggregate report:
    run_evaluation.py micro-averages by summing tp/fp/fn across scenarios
    first, and this scenario contributes (0, 0, 0) -- zero to every term, so
    it cannot pull an aggregate F1 down."""
    result = match_spans([], [])
    assert result.true_positives == 0
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_match_spans_below_threshold_does_not_match():
    result = match_spans(
        ["completely different content here"], ["Priya will send the report"], threshold=0.5
    )
    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 1


def test_evaluate_decisions_wraps_match_spans():
    predicted = [DecisionPrediction(text="We have decided to proceed with the plan")]
    gold = [DecisionGold(text="We have decided to proceed with the plan")]
    result = evaluate_decisions(predicted, gold)
    assert result.f1 == 1.0


def test_evaluate_actions_reports_owner_and_deadline_accuracy_on_true_positives_only():
    predicted = [
        ActionItemPrediction(
            text="Priya will send the report by Friday", owner_name="Priya", deadline="Friday"
        ),
        ActionItemPrediction(
            text="unrelated false positive sentence", owner_name=None, deadline=None
        ),
    ]
    gold = [
        ActionItemGold(
            text="Priya will send the report by Friday", owner_name="Priya", deadline="Friday"
        ),
    ]
    metrics = evaluate_actions(predicted, gold)
    assert metrics.span.true_positives == 1
    assert metrics.span.false_positives == 1
    assert metrics.owner_accuracy == 1.0
    assert metrics.deadline_accuracy == 1.0


def test_evaluate_actions_wrong_owner_is_scored_even_though_the_span_matched():
    predicted = [
        ActionItemPrediction(text="Rahul will check the backups", owner_name="Priya", deadline=None)
    ]
    gold = [ActionItemGold(text="Rahul will check the backups", owner_name="Rahul", deadline=None)]
    metrics = evaluate_actions(predicted, gold)
    assert metrics.span.true_positives == 1  # the sentence itself matched
    assert metrics.owner_accuracy == 0.0  # but the owner is wrong


def test_evaluate_actions_owner_accuracy_is_none_when_nothing_to_check():
    """Distinguish "no gold owner to check" (None) from "checked, always wrong" (0.0)."""
    predicted = [
        ActionItemPrediction(text="Send the signed agreement to the vendor", owner_name=None)
    ]
    gold = [ActionItemGold(text="Send the signed agreement to the vendor", owner_name=None)]
    metrics = evaluate_actions(predicted, gold)
    assert metrics.owner_accuracy is None
    assert metrics.deadline_accuracy is None


def test_owner_accuracy_is_case_and_whitespace_insensitive():
    predicted = [ActionItemPrediction(text="Priya will send the report", owner_name="  priya  ")]
    gold = [ActionItemGold(text="Priya will send the report", owner_name="Priya")]
    metrics = evaluate_actions(predicted, gold)
    assert metrics.owner_accuracy == 1.0
