"""Shared missingness rules for optional validation windows."""

import numpy as np


def finite_row_mask(rows) -> np.ndarray:
    """Return one flag per row indicating at least one finite observation."""

    return np.asarray(
        [np.isfinite(np.asarray(row)).any() for row in rows], dtype=bool
    )


def validation_window_mask(contexts, futures) -> np.ndarray:
    """Keep validation rows with finite support in both context and future."""

    context_mask = finite_row_mask(contexts)
    future_mask = finite_row_mask(futures)
    if context_mask.shape != future_mask.shape:
        raise ValueError("Validation contexts and futures must have the same rows")
    return context_mask & future_mask
