"""Thin, optional MLflow wrapper.

MLflow is not a platform dependency -- `requirements.txt` (Person B's base
install) does not include it, and the app must run without it. This module is
imported only by `evaluation/run_evaluation.py`, and every call degrades to a
no-op (with a log line) if `mlflow` is not installed or tracking cannot be
reached, so a missing/unreachable MLflow server never breaks an evaluation
run -- consistent with the "do not make MLflow mandatory" rule.

Usage
-----
    with tracking_run("extractor-eval", params={...}) as log:
        log.metrics({"decision_f1": 0.83, "action_f1": 0.79})

Set `MLFLOW_TRACKING_URI` to point at a real tracking server (or a local
`./mlruns` directory, MLflow's default); `mlruns/` is already gitignored.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from backend.app.observability.logging import get_logger

log = get_logger("ml.mlflow_tracking")

DEFAULT_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "meeting-intelligence-evaluation")


class _NullLogger:
    """No-op stand-in used when MLflow is unavailable, so callers never need
    an `if mlflow_available` branch."""

    def params(self, values: dict[str, Any]) -> None:
        pass

    def metrics(self, values: dict[str, float]) -> None:
        pass

    def artifact(self, path: str) -> None:
        pass


class _MlflowLogger:
    def __init__(self, mlflow_module: Any) -> None:
        self._mlflow = mlflow_module

    def params(self, values: dict[str, Any]) -> None:
        self._mlflow.log_params(values)

    def metrics(self, values: dict[str, float]) -> None:
        self._mlflow.log_metrics(values)

    def artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)


@contextmanager
def tracking_run(
    run_name: str,
    params: dict[str, Any] | None = None,
    experiment: str = DEFAULT_EXPERIMENT,
):
    """Context manager yielding a logger. Falls back to a no-op logger --
    never raises -- if MLflow is not installed or the tracking store cannot be
    opened (e.g. no write access, unreachable remote URI)."""
    try:
        import mlflow
    except ImportError:
        log.info("mlflow_not_installed", run_name=run_name, note="metrics were not tracked")
        yield _NullLogger()
        return

    try:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            logger = _MlflowLogger(mlflow)
            if params:
                logger.params(params)
            yield logger
    except Exception as exc:  # pragma: no cover - depends on external tracking store
        log.error("mlflow_tracking_failed", run_name=run_name, error=str(exc))
        yield _NullLogger()
