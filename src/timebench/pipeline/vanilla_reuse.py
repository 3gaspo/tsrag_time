"""Read exact canonical vanilla test forecasts produced by Evaluating TSFMs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _run_number(path: Path) -> int:
    try:
        return int(path.parent.name.rpartition("_")[2])
    except ValueError:
        return -1


def _task_identity(dataset: str, term: str, backbone: str, target_mode: str) -> dict:
    name, _, frequency = dataset.rpartition("/")
    return {
        "model": backbone,
        "target_mode": target_mode,
        "dataset": name or dataset,
        "frequency": frequency,
        "term": term,
    }


def _expected_windows(references: np.ndarray) -> tuple[list[int], int, int]:
    items = sorted({int(value) for value in references[:, 0]})
    channels = sorted({int(value) for value in references[:, 1]})
    if items != list(range(len(items))) or channels != list(range(len(channels))):
        raise ValueError("Prepared references do not use contiguous item/channel indices")
    counts = {
        len(np.unique(references[
            (references[:, 0] == item) & (references[:, 1] == channel), 2
        ]))
        for item in items for channel in channels
    }
    if len(counts) != 1:
        raise ValueError("Prepared references do not have one common test-window count")
    return items, len(channels), counts.pop()


def _compatible(manifest: dict, *, identity: dict, context_length: int,
                prediction_length: int, test_length: int, windows: int) -> bool:
    model = manifest.get("model_config", {})
    pipeline = manifest.get("pipeline_config", {})
    experiment = manifest.get("experiment_config", {})
    return (
        manifest.get("status") == "completed"
        and manifest.get("experiment") == "foundation_models_raw_inference"
        and manifest.get("identity") == identity
        and model.get("model_size") == {
            "chronos2": "chronos2", "chronos_bolt": "base",
            "ts_icl": "tsicl-v1", "timesfm3": "timesfm3",
        }.get(identity["model"])
        and model.get("context_length") == int(context_length)
        and pipeline.get("prediction_length") == int(prediction_length)
        and pipeline.get("test_length") == int(test_length)
        and pipeline.get("windows") == int(windows)
        and pipeline.get("target_mode") == identity["target_mode"]
        and pipeline.get("covariate_mode", "none") == "none"
        and experiment.get("covariate_mode", "none") == "none"
        and experiment.get("instance_normalization", "none") == "none"
    )


def _median_rows(forecasts: np.ndarray, levels: np.ndarray, target_mode: str,
                 references: np.ndarray, prediction_length: int) -> np.ndarray:
    median = np.flatnonzero(np.isclose(levels, 0.5, rtol=0.0, atol=1e-12))
    if len(median) != 1:
        raise ValueError("Canonical vanilla cache must contain exactly one 0.5 quantile")
    values = np.asarray(forecasts[:, median[0]], dtype=np.float32)
    items, channels, windows = _expected_windows(references)
    if target_mode == "univariate":
        if values.ndim == 3 and values.shape[1] == 1:
            values = values[:, 0]
        expected = (len(references), prediction_length)
        if values.shape != expected:
            raise ValueError(f"Expected univariate vanilla shape {expected}, got {values.shape}")
        return values
    expected = (len(items) * windows, channels, prediction_length)
    if values.shape != expected:
        raise ValueError(f"Expected multivariate vanilla shape {expected}, got {values.shape}")
    rows = np.empty((len(references), prediction_length), dtype=np.float32)
    origins = {
        item: sorted({int(value) for value in references[references[:, 0] == item, 2]})
        for item in items
    }
    positions = {
        tuple(map(int, reference)): row
        for row, reference in enumerate(references)
    }
    for item_index, item in enumerate(items):
        for window_index, origin in enumerate(origins[item]):
            forecast = values[item_index * windows + window_index]
            for channel in range(channels):
                rows[positions[(item, channel, origin)]] = forecast[channel]
    return rows


def load_vanilla_test_rows(root: str | Path | None, *, backbone: str,
                           target_mode: str, dataset: str, term: str,
                           context_length: int, prediction_length: int,
                           test_length: int, references: np.ndarray):
    """Return all compatible median rows and provenance, or None plus a miss reason."""
    metadata = {
        "configured_root": None if root is None else str(Path(root).expanduser()),
        "status": "not_configured",
    }
    if root is None:
        return None, metadata
    root = Path(root).expanduser().resolve()
    identity = _task_identity(dataset, term, backbone, target_mode)
    _, _, windows = _expected_windows(np.asarray(references))
    task_root = root / backbone / target_mode / dataset / term
    manifests = sorted(task_root.glob("run_*/manifest.json"), key=_run_number, reverse=True)
    metadata.update({"configured_root": str(root), "status": "no_compatible_run",
                     "checked_runs": len(manifests)})
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not _compatible(
            manifest,
            identity=identity,
            context_length=context_length,
            prediction_length=prediction_length,
            test_length=test_length,
            windows=windows,
        ):
            continue
        payload_path = manifest_path.parent / "raw_predictions.npz"
        if not payload_path.is_file():
            continue
        try:
            with np.load(payload_path, allow_pickle=False) as payload:
                rows = _median_rows(
                    payload["forecasts"], payload["quantile_levels"], target_mode,
                    np.asarray(references), prediction_length,
                )
        except (KeyError, OSError, ValueError) as exception:
            metadata.update({"status": "incompatible_payload", "reason": str(exception),
                             "manifest": str(manifest_path)})
            continue
        if np.isinf(rows).any():
            metadata.update({"status": "incompatible_payload",
                             "reason": "infinite forecast values",
                             "manifest": str(manifest_path)})
            continue
        metadata.update({"status": "reused", "manifest": str(manifest_path),
                         "rows": int(len(rows))})
        return rows, metadata
    return None, metadata
