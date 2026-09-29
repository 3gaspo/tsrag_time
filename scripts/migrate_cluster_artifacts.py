#!/usr/bin/env python3
"""One-time, idempotent migration to the surface/experiment artifact layout.

Run this in the Selena checkout after pulling the matching code. By default it
uses the project's scratch outputs and logs. Pass --dry-run to print the plan.
Delete this script after every retained cluster tree has been migrated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT = PROJECT_ROOT.name
RUN = re.compile(r"run_(\d+)$")


class Migration:
    def __init__(self, outputs: Path, logs: Path, dry_run: bool):
        self.outputs = outputs.resolve()
        self.logs = logs.resolve()
        self.dry_run = dry_run
        self.moves: list[tuple[str, str]] = []
        self.actions = 0

    def note(self, message: str) -> None:
        print(("PLAN" if self.dry_run else "APPLY") + f": {message}")
        self.actions += 1

    def mkdir(self, path: Path) -> None:
        if path.exists():
            return
        self.note(f"create {path}")
        if not self.dry_run:
            path.mkdir(parents=True, exist_ok=True)

    def move(self, source: Path, target: Path) -> None:
        if not source.exists():
            return
        source = source.resolve()
        target = target.resolve()
        if source == target:
            return
        self.moves.append((str(source), str(target)))
        self.note(f"move {source} -> {target}")
        if self.dry_run:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            source.replace(target)
            return
        if source.is_file() or target.is_file():
            if source.is_file() and target.is_file() and source.read_bytes() == target.read_bytes():
                source.unlink()
                return
            raise FileExistsError(f"Refusing to overwrite different artifact: {target}")
        self.merge_directory(source, target)

    def merge_directory(self, source: Path, target: Path) -> None:
        for child in sorted(source.iterdir(), key=lambda path: path.name):
            self.move(child, target / child.name)
        source.rmdir()

    def move_contents(self, source: Path, target: Path) -> None:
        if not source.is_dir():
            return
        self.moves.append((str(source.resolve()), str(target.resolve())))
        self.mkdir(target)
        for child in sorted(source.iterdir(), key=lambda path: path.name):
            self.move(child, target / child.name)
        if not self.dry_run and source.exists() and not any(source.iterdir()):
            source.rmdir()

    def remove_empty(self, root: Path) -> None:
        if self.dry_run or not root.exists():
            return
        for path in sorted((item for item in root.rglob("*") if item.is_dir()),
                           key=lambda item: len(item.parts), reverse=True):
            if path.exists() and not any(path.iterdir()):
                path.rmdir()
        if root.exists() and root.is_dir() and not any(root.iterdir()):
            root.rmdir()

    def rewrite_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self.rewrite_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.rewrite_value(item) for item in value]
        if isinstance(value, str):
            previous = None
            while value != previous:
                previous = value
                for source, target in sorted(
                    self.moves, key=lambda item: len(item[0]), reverse=True
                ):
                    value = value.replace(source, target)
                    value = value.replace(
                        source.replace("\\", "/"), target.replace("\\", "/")
                    )
        return value

    def migrate_run_configs(self) -> None:
        for config_path in sorted(self.outputs.rglob("config.json")):
            manifest_path = config_path.parent / "manifest.json"
            if RUN.fullmatch(config_path.parent.name) is None or not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config.pop("launch_id", None)
            metadata = manifest.setdefault("artifact_metadata", {})
            existing = metadata.get("evaluation")
            if existing is not None and existing != config:
                raise ValueError(f"Conflicting evaluation metadata in {manifest_path}")
            metadata["evaluation"] = config
            manifest["required_artifacts"] = [
                name for name in manifest.get("required_artifacts", [])
                if Path(name).name != "config.json"
            ]
            self.note(f"move run configuration into {manifest_path}")
            if not self.dry_run:
                self.write_json(manifest_path, manifest)
                config_path.unlink()

    def rewrite_json(self) -> None:
        for path in sorted(self.outputs.rglob("*.json")):
            if "manifest_history" in path.parts or path.name == "config.json":
                continue
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            updated = self.rewrite_value(current)
            if path.name == "metrics_summary.json" and isinstance(updated, dict):
                updated.pop("launch_id", None)
            if isinstance(updated, dict) and PROJECT == "selectime" and updated.get("experiment") == "selectime":
                updated["experiment"] = "scope_selection"
            if updated != current:
                self.note(f"rewrite paths/configuration in {path}")
                if not self.dry_run:
                    self.write_json(path, updated)

    @staticmethod
    def write_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)


def scratch_root(kind: str) -> Path:
    nni_file = Path(os.environ.get("TIME_NNI_FILE", Path.home() / "codes/.secrets/nni"))
    nni = nni_file.read_text(encoding="utf-8").splitlines()[0].strip().lower()
    if re.fullmatch(r"[a-z][a-z0-9_-]*", nni) is None:
        raise ValueError(f"Invalid NNI in {nni_file}")
    return Path("/scratch/users") / nni / "codes" / PROJECT / kind


def configured_root(explicit: Path | None, environment: str, kind: str) -> Path:
    if explicit is not None:
        return explicit
    configured = os.environ.get(environment)
    return Path(configured) if configured else scratch_root(kind)


def register_shared_seasonal_rewrites(migration: Migration) -> None:
    configured = os.environ.get("TIME_SEASONAL_ROOT")
    if configured:
        project_root = Path(configured).expanduser().resolve()
    else:
        nni_file = Path(os.environ.get("TIME_NNI_FILE", Path.home() / "codes/.secrets/nni"))
        nni = nni_file.read_text(encoding="utf-8").splitlines()[0].strip().lower()
        if re.fullmatch(r"[a-z][a-z0-9_-]*", nni) is None:
            raise ValueError(f"Invalid NNI in {nni_file}")
        project_root = Path("/scratch/users") / nni / "codes/seasonal"
    legacy = project_root / "foundation_models"
    current = project_root / "outputs/seasonal_naive"
    migration.moves.extend([
        (
            str((legacy / "inference/seasonal_naive").resolve()),
            str((current / "inference").resolve()),
        ),
        (
            str((legacy / "tasks/seasonal_naive").resolve()),
            str((current / "evaluations").resolve()),
        ),
    ])


def report_time(path: Path) -> tuple[str, int]:
    manifests = list(path.rglob("*report_manifest.json"))
    values: list[str] = []
    for manifest_path in manifests:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("completed_at", "created_at", "generated_at", "updated_at"):
            if payload.get(key):
                values.append(str(payload[key]))
    return (max(values, default=""), path.stat().st_mtime_ns)


def flatten_reports(migration: Migration, source: Path, target: Path) -> None:
    if not source.is_dir():
        return
    candidates = sorted((path for path in source.iterdir() if path.is_dir()), key=report_time)
    if not candidates:
        return
    direct_exists = target.is_dir() and any(path.name != "history" for path in target.iterdir())
    current = None if direct_exists else candidates.pop()
    history = target / "history"
    next_history = 0
    if history.is_dir():
        indices = [int(match.group(1)) for path in history.iterdir()
                   if path.is_dir() and (match := re.fullmatch(r"report_(\d+)", path.name))]
        next_history = max(indices, default=-1) + 1
    for candidate in candidates:
        migration.move(candidate, history / f"report_{next_history}")
        next_history += 1
    if current is not None:
        migration.move_contents(current, target)
    migration.remove_empty(source)


def migrate_evaluating_outputs(migration: Migration) -> None:
    old_reports = migration.outputs / "reports"
    if old_reports.is_dir():
        for experiment_root in sorted(old_reports.iterdir(), key=lambda path: path.name):
            if experiment_root.is_dir():
                flatten_reports(
                    migration,
                    experiment_root,
                    migration.outputs / experiment_root.name / "reports",
                )
    migration.remove_empty(old_reports)


def manifest_completed_at(run: Path) -> tuple[str, int]:
    try:
        value = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        return (str(value.get("completed_at") or value.get("updated_at") or ""),
                run.stat().st_mtime_ns)
    except (OSError, json.JSONDecodeError):
        return ("", run.stat().st_mtime_ns)


def migrate_selectime_outputs(migration: Migration) -> None:
    old_root = migration.outputs / "selectime"
    migration.move(old_root, migration.outputs / "scope_selection")
    sources: dict[str, list[Path]] = {}
    internal = old_root if migration.dry_run and old_root.is_dir() else migration.outputs / "scope_selection"
    if internal.is_dir():
        for model_dir in internal.iterdir():
            old = model_dir / "reports"
            if old.is_dir():
                sources.setdefault(model_dir.name, []).extend(
                    path for path in old.rglob("run_*") if path.is_dir() and RUN.fullmatch(path.name)
                )
    external = migration.outputs / "reports" / "selectime"
    if external.is_dir():
        for model_dir in external.iterdir():
            if model_dir.is_dir():
                sources.setdefault(model_dir.name, []).extend(
                    path for path in model_dir.rglob("run_*") if path.is_dir() and RUN.fullmatch(path.name)
                )
    for model, runs in sources.items():
        target = migration.outputs / "scope_selection" / "reports" / model / "comparison"
        existing = [int(match.group(1)) for path in target.glob("run_*")
                    if (match := RUN.fullmatch(path.name))]
        next_run = max(existing, default=-1) + 1
        for run in sorted(set(runs), key=manifest_completed_at):
            if target in run.parents:
                continue
            migration.move(run, target / f"run_{next_run}")
            next_run += 1
        for old_report_root in (internal / model / "reports", external / model):
            if not old_report_root.is_dir():
                continue
            for old_selector in old_report_root.rglob("SELECTED_RUNS.json"):
                migration.note(f"remove stale report selection {old_selector}")
                if not migration.dry_run:
                    old_selector.unlink()
        if migration.dry_run:
            migration.note(f"rebuild automatic report selections in {target}")
        elif target.is_dir():
            selections: dict[str, tuple[Path, dict[str, Any]]] = {}
            for run in target.glob("run_*"):
                manifest_path = run / "manifest.json"
                if not manifest_path.is_file():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                scientific = {key: manifest.get(key, {}) for key in
                              ("model_config", "pipeline_config", "experiment_config")}
                key = json.dumps(scientific, sort_keys=True)
                current = selections.get(key)
                if current is None or manifest_completed_at(run) > manifest_completed_at(current[0]):
                    selections[key] = (run, scientific)
            Migration.write_json(target / "SELECTED_RUNS.json", {
                "schema_version": 1,
                "selections": [
                    {"scientific_config": scientific, "mode": "auto", "run": run.name}
                    for run, scientific in selections.values()
                ],
            })
    migration.remove_empty(migration.outputs / "reports")
    if internal.is_dir():
        for model_dir in internal.iterdir():
            migration.remove_empty(model_dir / "reports")


def migrate_tsrag_outputs(migration: Migration) -> None:
    reports = migration.outputs / "tsrag" / "reports"
    if not reports.is_dir():
        return
    candidates = [path for path in reports.iterdir()
                  if path.is_dir() and (path / "report_manifest.json").is_file()]
    if not candidates:
        return
    candidates.sort(key=report_time)
    current = None if (reports / "report_manifest.json").is_file() else candidates.pop()
    history = reports / "history"
    indices = [int(match.group(1)) for path in history.glob("report_*")
               if path.is_dir() and (match := re.fullmatch(r"report_(\d+)", path.name))]
    next_history = max(indices, default=-1) + 1
    for candidate in candidates:
        migration.move(candidate, history / f"report_{next_history}")
        next_history += 1
    if current is not None:
        migration.move_contents(current, reports)


def status_experiment(workflow: str) -> str:
    if workflow.startswith("channels"):
        return "channels_comparison"
    if workflow.startswith("context"):
        return "context_size"
    if workflow.startswith("instance"):
        return "instance_normalization"
    if workflow.startswith("foundation"):
        return "foundation_models"
    if workflow.startswith("dataset"):
        return "dataset_diagnostics"
    raise ValueError(f"Cannot map workflow to an experiment: {workflow}")


def flatten_workflow_status(migration: Migration, source: Path, target: Path) -> None:
    if not source.is_dir():
        return
    migration.mkdir(target)
    for child in sorted(source.iterdir(), key=lambda path: path.name):
        if child.is_file():
            migration.move(child, target / child.name)
            continue
        launch = child.name
        for status in sorted((path for path in child.rglob("*") if path.is_file())):
            suffix = "__".join(status.relative_to(child).parts)
            migration.move(status, target / f"{launch}__{suffix}")
    migration.remove_empty(source)


def flatten_hydra(migration: Migration, experiment: str) -> None:
    target = migration.logs / experiment / "hydra"
    roots = [migration.logs / "hydra"]
    if target.is_dir():
        roots.append(target)
    for root in roots:
        if not root.is_dir():
            continue
        for run_dir in sorted(list(root.iterdir()), key=lambda path: path.name):
            match = re.fullmatch(r"\d{8}_\d{6}_(.+)", run_dir.name)
            if not run_dir.is_dir() or match is None:
                continue
            stage = match.group(1)
            destination = target / stage
            if PROJECT == "selectime":
                config_path = run_dir / ".hydra" / "config.yaml"
                content = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
                values = {}
                for key in ("model", "prediction_group"):
                    value_match = re.search(rf"(?m)^{key}:\s*['\"]?([^'\"\s#]+)", content)
                    if value_match is None:
                        raise ValueError(f"Cannot identify {key} for Hydra run {run_dir}")
                    value = value_match.group(1)
                    if not value or value in {".", ".."} or Path(value).name != value:
                        raise ValueError(f"Invalid {key} in Hydra run {run_dir}: {value!r}")
                    values[key] = value
                destination /= values["model"] / values["prediction_group"]
            for artifact in sorted(path for path in run_dir.rglob("*") if path.is_file()):
                suffix = "__".join(part.lstrip(".") for part in artifact.relative_to(run_dir).parts)
                migration.move(artifact, destination / f"{run_dir.name}__{suffix}")
        migration.remove_empty(root)


def flatten_dataset_metadata(migration: Migration) -> None:
    root = migration.logs / "dataset_diagnostics" / "dataset_metadata"
    if not root.is_dir():
        return
    for job_dir in sorted(list(root.iterdir()), key=lambda path: path.name):
        if not job_dir.is_dir():
            continue
        for artifact in sorted(path for path in job_dir.rglob("*") if path.is_file()):
            suffix = "__".join(artifact.relative_to(job_dir).parts)
            migration.move(artifact, root / f"{job_dir.name}__{suffix}")
    migration.remove_empty(root)


def evaluating_stream_plan(
    logs: Path,
) -> tuple[dict[Path, tuple[str, str]], list[tuple[Path, Path]]]:
    job_experiments: dict[str, str] = {}
    workflow_moves: list[tuple[Path, Path]] = []
    status_root = logs / "workflow_status"
    if status_root.is_dir():
        for workflow in status_root.iterdir():
            if not workflow.is_dir():
                continue
            experiment = status_experiment(workflow.name)
            workflow_moves.append((workflow, logs / experiment / "workflow_status" / workflow.name))
            for status in workflow.rglob("*.status"):
                for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("slurm_job_id="):
                        job_experiments[line.partition("=")[2].strip()] = experiment
    stream_plan: dict[Path, tuple[str, str]] = {}
    unknown: list[Path] = []
    for stream in [*logs.glob("*.out"), *logs.glob("*.err"), *logs.glob("*.log")]:
        matches = re.findall(r"(?<!\d)(\d{4,})(?!\d)", stream.name)
        experiment = next((job_experiments[value] for value in matches if value in job_experiments), None)
        if experiment is None:
            sample = stream.read_text(encoding="utf-8", errors="replace")[:200000]
            experiment = next((name for name in ("channels_comparison", "context_size",
                               "instance_normalization", "foundation_models") if name in sample), None)
        if experiment is None and stream.name.startswith(("sc2covar", "sc2multi", "sc2uni")):
            experiment = "channels_comparison"
        if experiment is None and stream.name.startswith("sdiagnostics"):
            experiment = "dataset_diagnostics"
        if experiment is None and stream.name.startswith(
            ("schronos", "sseasonal", "stsicl", "stimesfm", "ssummary")
        ):
            experiment = "foundation_models"
        if experiment is None:
            unknown.append(stream)
        else:
            kind = "stage_logs" if stream.suffix == ".log" else "slurm"
            stream_plan[stream] = (experiment, kind)
    if unknown:
        joined = "\n".join(str(path) for path in unknown)
        raise ValueError(f"Cannot identify the experiment for these Slurm streams:\n{joined}")
    return stream_plan, workflow_moves


def migrate_logs(migration: Migration, experiment: str | None = None) -> None:
    if PROJECT == "evaluating_tsfms":
        streams, workflows = evaluating_stream_plan(migration.logs)
        for source, target in workflows:
            flatten_workflow_status(migration, source, target)
        for source, (owner, kind) in streams.items():
            migration.move(source, migration.logs / owner / kind / source.name)
        migration.remove_empty(migration.logs / "workflow_status")
    else:
        assert experiment is not None
        flatten_hydra(migration, experiment)
        old_status = migration.logs / "workflow_status"
        if old_status.is_dir():
            for workflow in list(old_status.iterdir()):
                if workflow.is_dir():
                    flatten_workflow_status(
                        migration, workflow,
                        migration.logs / experiment / "workflow_status" / workflow.name,
                    )
            migration.remove_empty(old_status)
        for stream in [*migration.logs.glob("*.out"), *migration.logs.glob("*.err")]:
            migration.move(stream, migration.logs / experiment / "slurm" / stream.name)
        for stage_log in migration.logs.glob("*.log"):
            migration.move(stage_log, migration.logs / experiment / "stage_logs" / stage_log.name)
    migration.move(migration.logs / "dataset_metadata",
                   migration.logs / "dataset_diagnostics" / "dataset_metadata")
    flatten_dataset_metadata(migration)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path)
    parser.add_argument("--logs-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    outputs_root = configured_root(args.outputs_root, "TIME_OUTPUTS", "outputs")
    logs_root = configured_root(args.logs_root, "TIME_LOGS", "logs")
    migration = Migration(outputs_root, logs_root, args.dry_run)
    register_shared_seasonal_rewrites(migration)
    if PROJECT == "evaluating_tsfms":
        evaluating_stream_plan(migration.logs)  # Validate every root stream before moving anything.
        migrate_evaluating_outputs(migration)
        migrate_logs(migration)
    elif PROJECT == "selectime":
        migrate_selectime_outputs(migration)
        migrate_logs(migration, "scope_selection")
    elif PROJECT == "tsrag_time":
        migrate_tsrag_outputs(migration)
        migrate_logs(migration, "tsrag")
    else:
        raise ValueError(f"Unsupported project checkout: {PROJECT}")
    migration.migrate_run_configs()
    migration.rewrite_json()
    print(f"{'Planned' if args.dry_run else 'Completed'} {migration.actions} migration actions")


if __name__ == "__main__":
    main()
