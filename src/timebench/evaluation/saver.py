"""
Prediction saving utilities for per-window evaluation.

Output structure:
    output_dir/
        {dataset_config}/
            predictions.npz  # Contains quantile predictions and levels
            metrics.npz      # Contains per-window metrics
            metrics_summary.json  # Contains lightweight aggregate metrics
            config.json     # Contains dataset config
"""

import json
import os

import numpy as np

from timebench.evaluation.metrics import (
    compute_per_window_metrics_from_quantiles,
    summarize_metric_values,
)
from timebench.evaluation.grid import (
    EVALUATION_GRID_DEFINITION,
    EVALUATION_GRID_FILE,
    build_evaluation_grid,
    load_evaluation_grid,
    save_evaluation_grid,
)


def save_window_predictions(
    dataset,
    fc_quantiles: np.ndarray,
    ds_config: str,
    output_base_dir: str,
    seasonality: int = 1,
    model_hyperparams: dict = None,
    quantile_levels: list[float] = None,
    inference_seconds: float | None = None,
    task_output_dir: str | None = None,
    evaluation_grid_path: str | None = None,
    create_evaluation_grid: bool = False,
) -> dict:
    """
    Save predictions and metrics for each test window.

    Args:
        dataset: Dataset object with attributes:
            - test_data: Test data iterator
            - windows: Number of test windows
            - target_dim: Number of variates
            - prediction_length: Forecast horizon
            - freq: Data frequency string
        fc_quantiles: Concatenated quantile prediction array.
            Shape:
                - (num_total_instances, num_quantiles, prediction_length) for univariate
                - (num_total_instances, num_quantiles, num_variates, prediction_length) for multivariate
            where num_total_instances = num_series_exp * num_windows
        ds_config: Dataset configuration string, e.g., "m4_weekly/W/short"
        output_base_dir: Base directory for output files
        seasonality: Seasonal period length for MASE computation
        model_hyperparams: Dictionary of model hyperparameters to save in config
        quantile_levels: Quantile levels for output (default: [0.1, 0.2, ..., 0.9])
        inference_seconds: Accelerator-synchronized wall time for the complete
            test forecasting loop. Model loading, dataset construction, metric
            computation, and result saving must be excluded.
        task_output_dir: Optional exact task leaf. When omitted, preserve the
            upstream TIME ``output_base_dir/ds_config`` behavior.

    Output files:
        predictions.npz:
            - predictions_quantiles: (num_series, num_windows, num_quantiles, num_variates, prediction_length)
            - quantile_levels

        metrics.npz:
            - Each metric array with shape (num_series, num_windows, num_variates)

        metrics_summary.json:
            - Finite mean and coverage counts for each metric

        config.json:
            - Dataset, forecast-shape, metric, and model configuration

    Returns:
        config: Dictionary containing dataset config
    """
    # Setup quantile levels
    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    quantile_levels_array = np.array(quantile_levels, dtype=float)
    quantile_levels_list = [float(q) for q in quantile_levels_array.tolist()]
    num_quantiles = len(quantile_levels_list)

    # Create output directory for this dataset config
    ds_output_dir = task_output_dir or os.path.join(output_base_dir, ds_config)
    os.makedirs(ds_output_dir, exist_ok=True)

    test_data = dataset.test_data
    num_windows = dataset.windows
    prediction_length = dataset.prediction_length
    original_num_variates = dataset.target_dim  # Original variates before to_univariate

    # Count number of series during experiment (test_data contains num_series_exp * num_windows instances)
    num_total_instances = fc_quantiles.shape[0]
    if num_total_instances % num_windows != 0:
        raise ValueError(
            f"Forecast instance count ({num_total_instances}) must be divisible "
            f"by the configured number of windows ({num_windows})."
        )
    num_series_exp = num_total_instances // num_windows  # != num_series in original dataset if to_univariate is True

    print(f"    Total instances: {num_total_instances}, Series during experiment: {num_series_exp}, Windows: {num_windows}")

    # Collect ground truth labels and contexts for metrics computation
    print("    Collecting ground truth and context...")
    ground_truths = []
    contexts = []
    for inp, label in test_data:
        ground_truths.append(label["target"])
        contexts.append(inp["target"])

    # Determine if forecasts are univariate by checking the shape
    forecast_is_univariate = False
    if fc_quantiles.ndim == 3:
        # Shape: (num_total_instances, num_quantiles, prediction_length) -> univariate
        assert fc_quantiles.shape[1] == num_quantiles, \
            f"Mismatch in quantiles (dim 1). Expected {num_quantiles}, but got {fc_quantiles.shape[1]}. Full shape: {fc_quantiles.shape}"
        assert fc_quantiles.shape[2] == prediction_length, \
            f"Mismatch in prediction_length (dim 2). Expected {prediction_length}, but got {fc_quantiles.shape[2]}. Full shape: {fc_quantiles.shape}"
        forecast_is_univariate = True

    elif fc_quantiles.ndim == 4:
        # Shape: (num_total_instances, num_quantiles, num_variates, prediction_length)
        assert fc_quantiles.shape[1] == num_quantiles, \
            f"Mismatch in quantiles (dim 1). Expected {num_quantiles}, but got {fc_quantiles.shape[1]}. Full shape: {fc_quantiles.shape}"
        assert fc_quantiles.shape[2] == original_num_variates, \
            f"Mismatch in num_variates (dim 2). Expected {original_num_variates}, but got {fc_quantiles.shape[2]}. Full shape: {fc_quantiles.shape}"
        assert fc_quantiles.shape[3] == prediction_length, \
            f"Mismatch in prediction_length (dim 3). Expected {prediction_length}, but got {fc_quantiles.shape[3]}. Full shape: {fc_quantiles.shape}"
    else:
        raise ValueError(f"Expected fc_quantiles to have 3 or 4 dimensions, but got {fc_quantiles.ndim}. Shape: {fc_quantiles.shape}")


    # Detect to_univariate case: forecasts are univariate but original dataset has multiple variates
    # In this case, num_series_exp = original_num_series * original_num_variates
    is_to_univariate = forecast_is_univariate and original_num_variates > 1 and \
                       (num_series_exp % original_num_variates == 0)

    if is_to_univariate:
        # Reverse the to_univariate transformation
        # MultivariateToUnivariate expands as: series_0_dim0, series_0_dim1, ..., series_1_dim0, ...
        num_series = num_series_exp // original_num_variates
        num_variates = original_num_variates
        print(f"    Detected to_univariate case: {num_series_exp} flat series -> {num_series} series x {num_variates} variates")
    else:
        num_series = num_series_exp
        num_variates = original_num_variates if not forecast_is_univariate else 1

    # Initialize arrays
    predictions_quantiles = np.zeros((num_series, num_windows, num_quantiles, num_variates, prediction_length))
    ground_truth = np.zeros((num_series, num_windows, num_variates, prediction_length))

    # Find max context length for metrics computation
    context_len = max(ctx.shape[-1] for ctx in contexts)
    context_array = np.full((num_series, num_windows, num_variates, context_len), np.nan)

    print("    Organizing data into arrays...")
    for idx, (gt, ctx) in enumerate(zip(ground_truths, contexts)):
        # Extract fc_q from the concatenated array
        fc_q = fc_quantiles[idx]

        # Normalize fc_q to shape: (num_quantiles, num_variates, prediction_length)
        if fc_q.ndim == 2:
            # (num_quantiles, prediction_length) -> (num_quantiles, 1, prediction_length)
            fc_q = fc_q[:, np.newaxis, :]

        # Normalize ground truth
        if gt.ndim == 1:
            gt = gt[np.newaxis, :]

        # Normalize context
        if ctx.ndim == 1:
            ctx = ctx[np.newaxis, :]

        if is_to_univariate:
            # In to_univariate case:
            # idx iterates as: series_0_dim0_win0, series_0_dim0_win1, ..., series_0_dim1_win0, ...
            # We need to map flat_series_idx to (series_idx, variate_idx)
            flat_series_idx = idx // num_windows
            window_idx = idx % num_windows
            series_idx = flat_series_idx // num_variates
            variate_idx = flat_series_idx % num_variates

            predictions_quantiles[series_idx, window_idx, :, variate_idx, :] = fc_q[:, 0, :]
            ground_truth[series_idx, window_idx, variate_idx, :] = gt[0, :]

            ctx_len_actual = ctx.shape[-1]
            context_array[series_idx, window_idx, variate_idx, :ctx_len_actual] = ctx[0, :]
        else:
            series_idx = idx // num_windows
            window_idx = idx % num_windows

            predictions_quantiles[series_idx, window_idx] = fc_q
            ground_truth[series_idx, window_idx] = gt

            # Handle context shape mismatch
            if ctx.shape[0] != num_variates and ctx.shape[-1] == num_variates:
                ctx = ctx.T
            ctx_len_actual = ctx.shape[-1]
            context_array[series_idx, window_idx, :, :ctx_len_actual] = ctx

    if create_evaluation_grid:
        if evaluation_grid_path is not None:
            raise ValueError("cannot both create and consume an evaluation grid")
        target_mask, evaluation_mask = build_evaluation_grid(
            predictions_quantiles,
            ground_truth,
            context_array,
            seasonality,
            quantile_levels_list,
        )
        grid_path = save_evaluation_grid(
            os.path.join(ds_output_dir, EVALUATION_GRID_FILE),
            target_mask,
            evaluation_mask,
        )
    else:
        if evaluation_grid_path is None:
            raise ValueError("evaluation_grid_path is required for non-Seasonal evaluation")
        grid_path = os.path.abspath(os.path.expanduser(evaluation_grid_path))
        target_mask, evaluation_mask = load_evaluation_grid(
            grid_path, ground_truth=ground_truth
        )

    if np.isinf(predictions_quantiles).any():
        raise ValueError("Infinite forecast values cannot be evaluated")
    prediction_nan = np.isnan(predictions_quantiles)
    evaluated_positions = (
        target_mask[:, :, None, :, :]
        & evaluation_mask[:, :, None, :, None]
    )
    evaluated_positions = np.broadcast_to(
        evaluated_positions, predictions_quantiles.shape
    )
    prediction_outputs = {
        "total_values": int(predictions_quantiles.size),
        "nan_values": int(prediction_nan.sum()),
        "nan_rate": float(prediction_nan.mean()),
        "cells_with_nan": int(prediction_nan.any(axis=(2, 4)).sum()),
        "total_cells": int(num_series * num_windows * num_variates),
        "evaluation_values": int(evaluated_positions.sum()),
        "evaluation_nan_values": int((prediction_nan & evaluated_positions).sum()),
    }
    prediction_outputs["evaluation_nan_rate"] = (
        prediction_outputs["evaluation_nan_values"] / prediction_outputs["evaluation_values"]
        if prediction_outputs["evaluation_values"] else None
    )

    # Save quantiles to npz file
    # Use float16 to reduce storage (sufficient for visualization purposes)
    # Apply dynamic scaling to prevent float16 overflow (max ~65504)
    FLOAT16_SAFE_MAX = 60000.0  # Leave margin below 65504
    finite_predictions = np.abs(predictions_quantiles[np.isfinite(predictions_quantiles)])
    max_abs_val = float(finite_predictions.max()) if finite_predictions.size else 0.0

    prediction_scale_factor = 1.0
    if max_abs_val > FLOAT16_SAFE_MAX:
        # Calculate scale factor as power of 10 for cleaner scaling
        prediction_scale_factor = float(10 ** np.ceil(np.log10(max_abs_val / FLOAT16_SAFE_MAX)))
        print(f"    ⚠️  Predictions max value ({max_abs_val:.2f}) exceeds float16 safe range.")
        print(f"    ⚠️  Applying scale factor: {prediction_scale_factor:.0f} (values will be divided)")
        predictions_quantiles_scaled = predictions_quantiles / prediction_scale_factor
    else:
        predictions_quantiles_scaled = predictions_quantiles

    npz_path = os.path.join(ds_output_dir, "predictions.npz")
    np.savez_compressed(
        npz_path,
        predictions_quantiles=predictions_quantiles_scaled.astype(np.float16),
        quantile_levels=quantile_levels_array.astype(np.float16),
    )
    print(f"    Saved predictions (quantiles) to {npz_path}")

    # Compute per-window metrics
    print("    Computing per-window metrics...")
    metrics = compute_per_window_metrics_from_quantiles(
        predictions_quantiles=predictions_quantiles,
        ground_truth=ground_truth,
        context=context_array,
        seasonality=seasonality,
        quantile_levels=quantile_levels_list,
        target_mask=target_mask,
        evaluation_mask=evaluation_mask,
    )

    # Save metrics to npz file
    metrics_path = os.path.join(ds_output_dir, "metrics.npz")
    np.savez_compressed(metrics_path, **metrics)
    print(f"    Saved metrics to {metrics_path}")

    # Save config
    config = {
        "dataset_config": ds_config,
        "num_series": num_series,
        "num_windows": num_windows,
        "num_variates": num_variates,
        "num_model_inputs": num_total_instances,
        "prediction_length": prediction_length,
        "num_quantiles": num_quantiles,
        "quantile_levels": quantile_levels_list,
        "freq": dataset.freq,
        "seasonality": seasonality,
        "context_length": context_len,
        "metric_names": list(metrics.keys()),
        "prediction_scale_factor": prediction_scale_factor,  # For float16 overflow prevention
        "prediction_outputs": prediction_outputs,
        "metrics_summary_file": "metrics_summary.json",
        "evaluation_grid": {
            "definition": EVALUATION_GRID_DEFINITION,
            "source": str(grid_path),
            "valid_values": int(np.count_nonzero(evaluation_mask)),
            "total_values": int(evaluation_mask.size),
        },
    }

    launch_id = os.environ.get("TIME_LAUNCH_ID")
    if launch_id:
        config["launch_id"] = launch_id

    if inference_seconds is not None:
        inference_seconds = float(inference_seconds)
        if not np.isfinite(inference_seconds) or inference_seconds < 0:
            raise ValueError("inference_seconds must be finite and non-negative")
        config["inference_seconds"] = inference_seconds

    # Add model hyperparameters if provided
    if model_hyperparams:
        config.update(model_hyperparams)

    config_path = os.path.join(ds_output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"    Saved config to {config_path}")

    metric_summaries = {}
    for metric_name, metric_values in metrics.items():
        metric_summaries[metric_name] = summarize_metric_values(
            metric_values, np.count_nonzero(evaluation_mask)
        )
    metrics_summary = {
        "dataset_config": ds_config,
        "aggregation": "mean over the shared Seasonal Naive MASE evaluation grid",
        "evaluation_grid": config["evaluation_grid"],
        "prediction_outputs": prediction_outputs,
        "metrics": metric_summaries,
    }
    if launch_id:
        metrics_summary["launch_id"] = launch_id
    if model_hyperparams and "model" in model_hyperparams:
        metrics_summary["model"] = model_hyperparams["model"]
    if inference_seconds is not None:
        metrics_summary["inference_seconds"] = inference_seconds

    metrics_summary_path = os.path.join(ds_output_dir, "metrics_summary.json")
    with open(metrics_summary_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"    Saved aggregate metrics to {metrics_summary_path}")

    # Print average metrics summary
    print("    Metrics summary (averaged over all series/windows/variates):")
    for metric_name, metric_values in metrics.items():
        mean_val = metric_summaries[metric_name]["mean"]
        print(f"      {metric_name}: {'undefined' if mean_val is None else f'{mean_val:.4f}'}")

    return config
