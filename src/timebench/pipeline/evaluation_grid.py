"""Resolve the selected shared Seasonal Naive evaluation grid."""

from __future__ import annotations

import os
from pathlib import Path

from timebench.evaluation.grid import EVALUATION_GRID_DEFINITION, EVALUATION_GRID_FILE
from timebench.pipeline.runs import ManifestError, select_completed_runs


def resolve_shared_evaluation_grid(
    dataset: str, term: str, target_mode: str
) -> Path:
    """Return one selected completed grid produced in the shared Seasonal root."""

    evaluations_root = os.environ.get("TIME_SEASONAL_EVALUATIONS_ROOT")
    if not evaluations_root:
        raise ManifestError(
            "TIME_SEASONAL_EVALUATIONS_ROOT must point to Seasonal Naive evaluations"
        )
    if target_mode not in {"univariate", "multivariate"}:
        raise ManifestError(f"Unknown target mode {target_mode!r}")
    identity_root = (
        Path(evaluations_root).expanduser().resolve()
        / "univariate"
        / dataset
        / term
    )
    selected = select_completed_runs(
        identity_root,
        models={"seasonal_naive"},
        target_modes={"univariate"},
        config_filters={
            "pipeline_config.evaluation_grid": EVALUATION_GRID_DEFINITION,
        },
        config_policy="error",
        repeat_policy="latest",
    )
    if len(selected) != 1:
        raise ManifestError(
            f"Expected one selected Seasonal Naive run below {identity_root}, found {len(selected)}"
        )
    path = selected[0][0] / EVALUATION_GRID_FILE
    if not path.is_file() or path.stat().st_size == 0:
        raise ManifestError(f"Selected Seasonal Naive run has no evaluation grid: {path}")
    return path
