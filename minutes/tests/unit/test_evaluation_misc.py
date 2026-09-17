"""language_breakdown.py, error_analysis.py, segmentation_metrics.py and
mlflow_tracking.py -- grouped since each is small and self-contained."""

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
from backend.app.ml.evaluation.error_analysis import analyse_actions, analyse_decisions
from backend.app.ml.evaluation.language_breakdown import aggregate_span_metrics_by_language
from backend.app.ml.evaluation.segmentation_metrics import evaluate_boundaries
from backend.app.ml.mlflow_tracking import tracking_run

# --------------------------------------------------------------------------- #
# language_breakdown
# --------------------------------------------------------------------------- #


def test_language_breakdown_is_micro_averaged_not_a_mean_of_per_scenario_ratios():
    rows = [
        {"language": "en", "true_positives": 10, "false_positives": 0, "false_negatives": 0},
        {"language": "en", "true_positives": 0, "false_positives": 0, "false_negatives": 10},
        {"language": "hi", "true_positives": 1, "false_positives": 0, "false_negatives": 0},
    ]
    out = aggregate_span_metrics_by_language(rows)
    assert set(out) == {"en", "hi"}
    assert out["en"]["scenarios"] == 2
    # micro: 10 tp / (10 tp + 10 fn) = 0.5 recall -- NOT the mean of 1.0 and 0.0
    assert out["en"]["recall"] == 0.5
    assert out["hi"]["f1"] == 1.0


def test_language_breakdown_handles_a_language_with_only_true_negatives():
    rows = [{"language": "en", "true_positives": 0, "false_positives": 0, "false_negatives": 0}]
    out = aggregate_span_metrics_by_language(rows)
    assert out["en"]["precision"] == 0.0
    assert out["en"]["recall"] == 0.0


# --------------------------------------------------------------------------- #
# error_analysis
# --------------------------------------------------------------------------- #


def test_analyse_decisions_reports_false_positive_and_false_negative():
    predicted = [DecisionPrediction(text="totally unrelated content")]
    gold = [DecisionGold(text="We have decided to proceed with the plan")]
    match = evaluate_decisions(predicted, gold)
    errors = analyse_decisions("scenario-x", predicted, gold, match)
    kinds = {e.kind for e in errors}
    assert kinds == {"false_positive", "false_negative"}


def test_analyse_decisions_reports_nothing_on_a_perfect_match():
    predicted = [DecisionPrediction(text="We have decided to proceed with the plan")]
    gold = [DecisionGold(text="We have decided to proceed with the plan")]
    match = evaluate_decisions(predicted, gold)
    assert analyse_decisions("scenario-x", predicted, gold, match) == []


def test_analyse_actions_reports_wrong_owner_and_wrong_deadline_on_matched_spans():
    predicted = [
        ActionItemPrediction(
            text="Rahul will check the backups by Friday", owner_name="Priya", deadline="Monday"
        )
    ]
    gold = [
        ActionItemGold(
            text="Rahul will check the backups by Friday", owner_name="Rahul", deadline="Friday"
        )
    ]
    match = evaluate_actions(predicted, gold).span
    errors = analyse_actions("scenario-x", predicted, gold, match)
    kinds = {e.kind for e in errors}
    assert kinds == {"wrong_owner", "wrong_deadline"}
    assert all(e.scenario == "scenario-x" for e in errors)


def test_analyse_actions_does_not_flag_a_correctly_unset_owner():
    predicted = [
        ActionItemPrediction(text="Send the signed agreement to the vendor", owner_name=None)
    ]
    gold = [ActionItemGold(text="Send the signed agreement to the vendor", owner_name=None)]
    match = evaluate_actions(predicted, gold).span
    assert analyse_actions("scenario-x", predicted, gold, match) == []


# --------------------------------------------------------------------------- #
# segmentation_metrics
# --------------------------------------------------------------------------- #


def test_evaluate_boundaries_exact_match():
    metrics = evaluate_boundaries([100, 250], [100, 250], tolerance_chars=10)
    assert metrics.true_positives == 2
    assert metrics.precision == metrics.recall == 1.0


def test_evaluate_boundaries_within_tolerance_still_counts():
    metrics = evaluate_boundaries([105], [100], tolerance_chars=10)
    assert metrics.true_positives == 1


def test_evaluate_boundaries_outside_tolerance_does_not_count():
    metrics = evaluate_boundaries([200], [100], tolerance_chars=10)
    assert metrics.true_positives == 0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0


def test_evaluate_boundaries_no_reference_boundaries():
    metrics = evaluate_boundaries([], [], tolerance_chars=10)
    assert metrics.true_positives == 0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0


# --------------------------------------------------------------------------- #
# mlflow_tracking -- must never fail evaluation when mlflow is not installed
# --------------------------------------------------------------------------- #


def test_tracking_run_degrades_to_a_noop_without_raising():
    with tracking_run("test-run", params={"a": 1}) as tracked:
        # Whether or not mlflow happens to be installed in this environment,
        # these calls must never raise.
        tracked.params({"b": 2})
        tracked.metrics({"f1": 0.9})
        tracked.artifact(__file__)
