"""Runtime path contract shared by TIME commands and cluster wrappers."""

import os
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _configured_path(variable: str, fallback: Path) -> Path:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    value = os.getenv(variable)
    return Path(value).expanduser() if value else fallback


def data_root() -> Path:
    """Prepared CSV, summary, and Arrow dataset workspace."""
    return _configured_path("TIME_DATA_ROOT", PROJECT_ROOT / "datasets")


def dataset_storage_root() -> Path:
    """HF Arrow datasets consumed by :class:`timebench.evaluation.Dataset`."""
    return _configured_path("TIME_DATASET", data_root() / "hf_dataset")


def dataset_metadata_root() -> Path:
    """Shared dataset-derived quality reports and feature artifacts."""
    return _configured_path("TIME_METADATA", data_root() / "time_metadata")


def weights_root() -> Path:
    """Model checkpoints and package caches."""
    return _configured_path("TIME_WEIGHTS", PROJECT_ROOT / "weights")


def foundation_weight_path(
    relative: str | Path,
    *,
    explicit: str | Path | None = None,
    directory: bool,
) -> Path:
    """Resolve one required local foundation-model checkpoint."""
    path = Path(explicit).expanduser() if explicit else weights_root() / relative
    path = path.resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        expected = "directory" if directory else "file"
        raise FileNotFoundError(f"Foundation-model weight {expected} not found: {path}")
    return path


def outputs_root() -> Path:
    """Generated predictions, metrics, and experiment reports."""
    return _configured_path("TIME_OUTPUTS", PROJECT_ROOT / "outputs")


def foundation_experiment_name(experiment: str | None = None) -> str:
    """Resolve the independently launched experiment owning foundation tasks."""
    value = experiment or os.getenv("TIME_EXPERIMENT", "foundation_models")
    if value not in {"foundation_models", "channels_comparison"}:
        raise ValueError(f"Unknown foundation experiment {value!r}")
    return value


def foundation_experiment_root(experiment: str | None = None) -> Path:
    """Task root for one maintained foundation experiment."""
    return outputs_root() / foundation_experiment_name(experiment) / "tasks"


def foundation_identity_root(
    experiment_root: str | Path,
    model: str,
    target_mode: str,
    dataset: str,
    term: str,
) -> Path:
    """Identity directory whose non-path configurations live in ``run_n``."""
    if target_mode not in {"univariate", "multivariate"}:
        raise ValueError(f"Unknown target mode {target_mode!r}")
    return Path(experiment_root) / model / target_mode / dataset / term


def logs_root() -> Path:
    """Runtime streams and scheduler logs."""
    return _configured_path("TIME_LOGS", PROJECT_ROOT / "logs")
