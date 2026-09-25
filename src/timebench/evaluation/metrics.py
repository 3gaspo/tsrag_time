"""
Per-window metrics computation for time series forecasting evaluation.
Aligned with GluonTS implementation.

Supported metrics:
- MSE: Mean Squared Error (using median forecast, aligned with GluonTS MSE[0.5])
- MAE: Mean Absolute Error (using median forecast)
- RMSE: Root Mean Squared Error (using median forecast)
- MAPE: Mean Absolute Percentage Error (using median forecast, returns fraction)
- sMAPE: Symmetric Mean Absolute Percentage Error (using median forecast, range [0, 2])
- MASE: Mean Absolute Scaled Error (using median forecast)
- ND: Normalized Deviation (using median forecast)
- CRPS: Continuous Ranked Probability Score (MeanWeightedSumQuantileLoss)

"""

import numpy as np


def summarize_metric_values(values: np.ndarray, evaluation_values: int) -> dict:
    """Aggregate finite metric cells; dispersion is population (ddof=0)."""
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return {
        "mean": float(np.nanmean(finite)) if finite.size else None,
        "std": float(np.nanstd(finite, dtype=np.float64, ddof=0)) if finite.size else None,
        "variance": float(np.nanvar(finite, dtype=np.float64, ddof=0)) if finite.size else None,
        "dispersion_ddof": 0,
        "finite_values": int(finite.size),
        "evaluation_values": int(evaluation_values),
        "total_values": int(values.size),
    }


def fill_missing_history(context: np.ndarray) -> np.ndarray:
    """Apply the maintained TIME forward-fill policy along the time axis."""

    values = np.asarray(context).copy()
    for history in values.reshape(-1, values.shape[-1]):
        if not np.isnan(history).any():
            continue
        missing = np.isnan(history)
        indexes = np.where(~missing, np.arange(len(missing)), 0)
        np.maximum.accumulate(indexes, out=indexes)
        history[:] = history[indexes]
        if np.isnan(history).any():
            observed = history[~np.isnan(history)]
            first_observed = observed[0] if len(observed) else 0
            history[:] = np.nan_to_num(history, nan=first_observed)
    return values


def seasonal_naive_point_forecast(
    context: np.ndarray,
    prediction_length: int,
    seasonality: int,
) -> np.ndarray:
    """Repeat the last season after Improved's missing-history preparation."""

    values = fill_missing_history(context)
    period = min(int(seasonality), values.shape[-1])
    if period <= 0:
        raise ValueError("seasonality and context length must be positive")
    repeats = int(np.ceil(int(prediction_length) / period))
    return np.tile(values[..., -period:], (*([1] * (values.ndim - 1)), repeats))[
        ..., : int(prediction_length)
    ]


def seasonal_naive_scale(
    context: np.ndarray,
    seasonality: int,
    *,
    squared: bool = False,
) -> np.ndarray:
    """Return a finite-pair seasonal scale without collapsing missing dates.

    The final axis is time. A seasonal difference contributes only when both
    observations exist at their original timestamps. ``squared=False`` returns
    the MASE denominator; ``squared=True`` returns its RMS analogue for MSSE.
    """

    values = np.asarray(context, dtype=np.float64)
    period = int(seasonality)
    if period <= 0:
        raise ValueError("seasonality must be positive")
    result_shape = values.shape[:-1]
    if values.shape[-1] <= period:
        return np.full(result_shape, np.nan, dtype=np.float64)
    left = values[..., :-period]
    right = values[..., period:]
    valid = np.isfinite(left) & np.isfinite(right)
    differences = np.where(valid, right - left, 0.0)
    contributions = np.square(differences) if squared else np.abs(differences)
    counts = valid.sum(axis=-1)
    mean = np.divide(
        contributions.sum(axis=-1, dtype=np.float64),
        counts,
        out=np.full(result_shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )
    scale = np.sqrt(mean) if squared else mean
    return np.where(scale > 0, scale, np.nan)


# Default quantile levels for CRPS computation (aligned with GluonTS)
DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def compute_per_window_metrics_from_quantiles(
    predictions_quantiles: np.ndarray,
    ground_truth: np.ndarray,
    context: np.ndarray,
    seasonality: int = 1,
    quantile_levels: list[float] = None,
    target_mask: np.ndarray | None = None,
    evaluation_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute evaluation metrics for each prediction window from quantile forecasts.

    Args:
        predictions_quantiles: Quantile forecasts with shape
            (num_series, num_windows, num_quantiles, num_variates, pred_len)
        ground_truth: Ground truth values with shape
            (num_series, num_windows, num_variates, pred_len)
        context: Historical context with shape
            (num_series, num_windows, num_variates, max_ctx_len)
            Note: Shorter contexts are NaN-padded
        seasonality: Seasonal period length for MASE computation
        quantile_levels: Quantile levels corresponding to predictions_quantiles.
            Defaults to [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    Returns:
        Dictionary of metric arrays, each with shape (num_series, num_windows, num_variates)
        Keys: MSE, MAE, RMSE, MAPE, sMAPE, MASE, ND, CRPS
    """
    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILE_LEVELS

    (
        num_series,
        num_windows,
        num_quantiles,
        num_variates,
        pred_len,
    ) = predictions_quantiles.shape

    if len(quantile_levels) != num_quantiles:
        raise ValueError(
            "quantile_levels length must match predictions_quantiles' quantile dimension"
        )
    if 0.5 not in quantile_levels:
        raise ValueError("quantile_levels must include 0.5 for median-based metrics")
    median_idx = quantile_levels.index(0.5)

    if np.isinf(ground_truth).any() or np.isinf(context).any():
        raise ValueError("Ground truth and context may contain NaNs, never infinities")

    target_mask = (
        np.isfinite(ground_truth)
        if target_mask is None
        else np.asarray(target_mask, dtype=bool)
    )
    if target_mask.shape != ground_truth.shape:
        raise ValueError("target_mask must have the same shape as ground_truth")
    if np.any(target_mask & ~np.isfinite(ground_truth)):
        raise ValueError("target_mask includes non-finite ground-truth values")
    metric_shape = (num_series, num_windows, num_variates)
    evaluation_mask = (
        target_mask.any(axis=-1)
        if evaluation_mask is None
        else np.asarray(evaluation_mask, dtype=bool)
    )
    if evaluation_mask.shape != metric_shape:
        raise ValueError("evaluation_mask has the wrong series/window/variate shape")
    if np.any(evaluation_mask & ~target_mask.any(axis=-1)):
        raise ValueError("evaluation_mask includes cells without finite ground truth")

    # Cells outside the shared grid remain absent for every method and metric.
    mse = np.full(metric_shape, np.nan)
    mae = np.full(metric_shape, np.nan)
    rmse = np.full(metric_shape, np.nan)
    mape = np.full(metric_shape, np.nan)
    smape = np.full(metric_shape, np.nan)
    mase = np.full(metric_shape, np.nan)
    nd = np.full(metric_shape, np.nan)

    # CRPS (Continuous Ranked Probability Score) - using weighted quantile loss
    crps = np.full(metric_shape, np.nan)

    for s in range(num_series):
        for w in range(num_windows):
            for v in range(num_variates):
                if not evaluation_mask[s, w, v]:
                    continue
                q_preds = predictions_quantiles[s, w, :, v]  # (num_quantiles, pred_len)
                gt = ground_truth[s, w, v]  # (pred_len,)
                ctx = context[s, w, v]  # (max_ctx_len,)

                # Compute median (0.5 quantile) forecast - used for most metrics
                median_pred = q_preds[median_idx]  # (pred_len,)

                valid_mask = target_mask[s, w, v]

                if np.isinf(q_preds[:, valid_mask]).any():
                    raise ValueError(
                        "infinite forecast on shared evaluation grid at "
                        f"series={s}, window={w}, variate={v}"
                    )

                # Median metrics average over finite produced values only.
                median_valid = valid_mask & np.isfinite(median_pred)
                if np.any(median_valid):
                    median_gt = np.asarray(gt[median_valid], dtype=np.float64)
                    median_values = np.asarray(median_pred[median_valid], dtype=np.float64)
                    error = median_gt - median_values
                    abs_error = np.abs(error)
                    mse[s, w, v] = np.nanmean(error ** 2)
                    mae[s, w, v] = np.nanmean(abs_error)
                    rmse[s, w, v] = np.sqrt(mse[s, w, v])

                    nonzero_target = np.abs(median_gt) > 0
                    if np.any(nonzero_target):
                        mape[s, w, v] = np.nanmean(
                            abs_error[nonzero_target] / np.abs(median_gt[nonzero_target])
                        )

                    smape_denominator = np.abs(median_gt) + np.abs(median_values)
                    smape_vals = np.divide(
                        2 * abs_error,
                        smape_denominator,
                        out=np.zeros_like(abs_error),
                        where=smape_denominator > 0,
                    )
                    smape[s, w, v] = np.nanmean(smape_vals)
                    seasonal_error = seasonal_naive_scale(ctx, seasonality)
                    mase[s, w, v] = mae[s, w, v] / seasonal_error
                    abs_label_sum = np.nansum(np.abs(median_gt))
                    if abs_label_sum > 0:
                        nd[s, w, v] = np.nansum(abs_error) / abs_label_sum

                # CRPS (MeanWeightedSumQuantileLoss) from provided quantiles
                weighted_quantile_losses = []
                for q, q_pred in zip(quantile_levels, q_preds):
                    quantile_valid = valid_mask & np.isfinite(q_pred)
                    if not np.any(quantile_valid):
                        continue
                    quantile_gt = np.asarray(gt[quantile_valid], dtype=np.float64)
                    quantile_values = np.asarray(q_pred[quantile_valid], dtype=np.float64)
                    abs_label_sum = np.nansum(np.abs(quantile_gt))
                    if abs_label_sum <= 0:
                        continue
                    q_error = quantile_gt - quantile_values
                    indicator = (quantile_values >= quantile_gt).astype(float)
                    q_loss = 2 * np.abs(q_error * (indicator - q))
                    weighted_quantile_losses.append(np.nansum(q_loss) / abs_label_sum)
                if weighted_quantile_losses:
                    crps[s, w, v] = np.nanmean(weighted_quantile_losses)

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "MASE": mase,
        "ND": nd,
        "CRPS": crps,
    }
