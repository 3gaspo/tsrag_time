"""Evaluation-scoped fallback reporting shared by TIME experiments."""

from collections import Counter
from collections.abc import Mapping

import numpy as np


def summarize_fallbacks(
    fallback_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    reasons_by_row: Mapping[int, str] | None = None,
) -> dict:
    """Summarize fallbacks on rows that contribute to scientific evaluation."""

    fallback = np.asarray(fallback_mask, dtype=bool)
    evaluated = np.asarray(evaluation_mask, dtype=bool)
    if fallback.ndim != 1 or evaluated.ndim != 1 or fallback.shape != evaluated.shape:
        raise ValueError("fallback and evaluation masks must be aligned one-dimensional arrays")

    reported = fallback & evaluated
    summary = {
        "fallback_count": int(reported.sum()),
        "evaluated_rows": int(evaluated.sum()),
        "all_fallback_rows": int(fallback.sum()),
        "total_rows": int(fallback.size),
        "fallback_reasons": {},
    }
    if reasons_by_row is None:
        return summary

    reasons = {int(row): str(reason) for row, reason in reasons_by_row.items()}
    if any(row < 0 or row >= fallback.size for row in reasons):
        raise ValueError("fallback reason row is outside the fallback mask")
    missing = [int(row) for row in np.flatnonzero(reported) if int(row) not in reasons]
    if missing:
        raise ValueError(f"evaluated fallback rows have no reason: {missing[:10]}")
    summary["fallback_reasons"] = dict(sorted(Counter(
        reasons[int(row)] for row in np.flatnonzero(reported)
    ).items()))
    return summary
