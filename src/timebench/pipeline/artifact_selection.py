"""One file-selection contract for cluster transfer and Git publication."""

import argparse
from pathlib import Path
import sys


LIGHT_NAMES = {
    "foundation_model_summary.csv", "foundation_model_summary.md",
    "foundation_model_report_manifest.json", "mase_vs_features.svg",
    "mase_vs_features_data.csv", "mase_vs_features_correlations.csv",
    "SELECTED_RUNS.json", "manifest.json", "model_manifest.json",
    "result_manifest.json", "prediction_manifest.json", "selection.json",
    "comparison_summary.json", "time_summary_manifest.json", "time_summary.json",
    "time_tasks.csv", "audit_manifest.json", "config.json", "metrics_summary.json",
    "summary.json", "report_manifest.json", "comparison.csv", "task_summary.csv",
    "dataset_summary.csv", "full_dataset.csv", "dataset_features_full.csv",
    "prepared.json", "retrieval.json", "prediction.json", "extraction.json",
    "weight.json", "selections.csv", "selection_summary.csv", "adaptation_summary.csv",
    "bayes_mixture.json", "bayes_past_targets_mixture.json",
}
DETAIL_NAMES = {
    "metrics.npz", "window_events.csv", "nonfinite_positions.csv", "full.csv",
}
BINARY_SUFFIXES = {".pt", ".npy", ".npz", ".cbm", ".pkl", ".pickle"}


def selected(path, size):
    """Report bundles include PNG/PDF figures; raw recovery arrays stay separate."""
    path = Path(path)
    if size == "full":
        return True
    if path.suffix in BINARY_SUFFIXES:
        return size == "detailed" and path.name == "metrics.npz"
    if {"reports", "performance"}.intersection(path.parts):
        return True
    if "manifest_history" in path.parts or "time_inference" in path.parts:
        return path.suffix == ".json"
    return path.name in LIGHT_NAMES or (
        size == "detailed" and (
            path.name in DETAIL_NAMES or path.name.endswith("fallback_reasons.json")
        )
    )


def filters(size):
    if size == "full":
        return []
    result = []
    if size == "detailed":
        result.append("--include=metrics.npz")
    result.extend(f"--exclude=*{suffix}" for suffix in sorted(BINARY_SUFFIXES))
    result.extend([
        "--include=*/", "--include=/reports/***", "--include=**/reports/***",
        "--include=**/performance/***", "--include=**/manifest_history/*.json",
        "--include=**/time_inference/**.json",
    ])
    result.extend(f"--include={name}" for name in sorted(LIGHT_NAMES))
    if size == "detailed":
        result.extend(f"--include={name}" for name in sorted(DETAIL_NAMES))
        result.append("--include=*fallback_reasons.json")
    return [*result, "--exclude=*"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("filters", "paths"))
    parser.add_argument("roots", nargs="*", type=Path)
    parser.add_argument("--size", choices=("lightweight", "detailed", "full"), required=True)
    parser.add_argument("--max-bytes", type=int, default=100000000)
    args = parser.parse_args()
    if args.command == "filters":
        print("\n".join(filters(args.size)))
        return
    for root in args.roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or not selected(path.relative_to(root), args.size):
                continue
            if args.size == "lightweight" and path.stat().st_size > args.max_bytes:
                continue
            sys.stdout.buffer.write(str(path).encode() + b"\0")


if __name__ == "__main__":
    main()
