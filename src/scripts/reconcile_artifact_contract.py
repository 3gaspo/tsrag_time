#!/usr/bin/env python3
"""Shared implementation for artifact-config migration and report recovery.

Use ``migrate_artifact_configs.py`` to rewrite existing run manifests and
``regenerate_stale_reports.py`` to plan or submit report-only recovery.  This
module deliberately has no combined destructive command.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

MANIFEST = "manifest.json"
RUN_PREFIX = "run_"
SELECTION = "SELECTED_RUNS.json"
CONFIG_FIELDS = ("model_config", "pipeline_config", "experiment_config")
OPERATIONAL_FIELDS = {
    "action", "artifact_revision", "attempts", "completed_at", "computed_at",
    "error", "launch_id", "launched_at", "manifest_path", "project",
    "required_artifacts", "run", "runtime_config", "slurm_array_task_id",
    "slurm_job_id", "started_at", "status", "updated_at",
}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def manifest_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.rglob(MANIFEST)
        if "manifest_history" not in path.parts
        and "reports" not in path.parts
        and path.parent.name.startswith(RUN_PREFIX)
        and path.parent.name[len(RUN_PREFIX):].isdigit()
    ]


def old_reference(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and {"identity", "run", "completed_at"}.issubset(value)
        and set(value).issubset({"identity", "run", "completed_at"})
    )


def scientific_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = {
        "schema_version": payload.get("schema_version"),
        "experiment": payload.get("experiment"),
        "identity": deepcopy(payload.get("identity", {})),
        **{field: deepcopy(payload.get(field, {})) for field in CONFIG_FIELDS},
    }
    for name in ("seed", "seeds"):
        if name in payload:
            snapshot[name] = deepcopy(payload[name])
    return snapshot


class Migration:
    def __init__(self, output_root: Path, dependency_roots: list[Path]):
        self.output_root = output_root.resolve()
        roots = [self.output_root, *[path.resolve() for path in dependency_roots]]
        paths = {path.resolve() for root in roots for path in manifest_paths(root)}
        self.paths = sorted(paths)
        self.raw = {path: read_json(path) for path in self.paths}
        self.planned: dict[Path, dict[str, Any]] = {}
        self.visiting: set[Path] = set()
        self.exact_index: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
        for path, payload in self.raw.items():
            identity = canonical(payload.get("identity", {}))
            run = path.parent.name
            completed = str(payload.get("completed_at"))
            self.exact_index[(identity, run, completed)].append(path)

    def resolve(self, reference: Mapping[str, Any], consumer: Path) -> Path:
        identity = canonical(reference["identity"])
        run = str(reference["run"])
        completed = str(reference.get("completed_at"))
        candidates = self.exact_index.get((identity, run, completed), [])
        completed_candidates = [
            path for path in candidates if self.raw[path].get("status") == "completed"
        ]
        if len(completed_candidates) != 1:
            names = [str(path) for path in completed_candidates]
            raise ValueError(
                f"{consumer}: legacy dependency {reference} resolves to "
                f"{len(completed_candidates)} completed producers: {names}. "
                "Add the producer artifact tree with --dependency-root if it "
                "is outside this project's output root."
            )
        return completed_candidates[0]

    def normalize(
        self,
        value: Any,
        dotted_path: str,
        consumer: Path,
        dependencies: dict[str, Any],
    ) -> Any:
        if old_reference(value):
            producer_path = self.resolve(value, consumer)
            producer = self.migrate_manifest(producer_path)
            snapshot = scientific_snapshot(producer)
            launch = producer.get("launch", {})
            dependencies[dotted_path] = {
                "scientific": snapshot,
                "artifact": {
                    "run": producer_path.parent.name,
                    "revision": int(producer.get("artifact_revision", 1)),
                },
                "provenance": {
                    "project": producer.get("project"),
                    "manifest_path": str(producer_path),
                    "completed_at": producer.get("completed_at"),
                    "launch_id": launch.get("launch_id"),
                    "launched_at": launch.get("launched_at"),
                },
            }
            return snapshot
        if isinstance(value, Mapping):
            forbidden = sorted(set(map(str, value)) & OPERATIONAL_FIELDS)
            if forbidden:
                raise ValueError(
                    f"{consumer}: operational fields remain at {dotted_path}: {forbidden}"
                )
            return {
                str(key): self.normalize(
                    item, f"{dotted_path}.{key}", consumer, dependencies
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self.normalize(item, f"{dotted_path}[{index}]", consumer, dependencies)
                for index, item in enumerate(value)
            ]
        return value

    def migrate_manifest(self, path: Path) -> dict[str, Any]:
        path = path.resolve()
        if path in self.planned:
            return self.planned[path]
        if path in self.visiting:
            raise ValueError(f"Dependency cycle involving {path}")
        self.visiting.add(path)
        payload = deepcopy(self.raw[path])
        dependencies = deepcopy(payload.get("dependencies", {}))
        for field in CONFIG_FIELDS:
            payload[field] = self.normalize(
                payload.get(field, {}), field, path, dependencies
            )
        payload["dependencies"] = dependencies
        payload["artifact_revision"] = int(payload.get("artifact_revision", 1))
        self.visiting.remove(path)
        self.planned[path] = payload
        return payload

    def plan(self) -> None:
        for path in self.paths:
            if path.is_relative_to(self.output_root):
                self.migrate_manifest(path)


def scientific_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {field: payload.get(field, {}) for field in CONFIG_FIELDS}


def run_number(path: Path) -> int:
    return int(path.parent.name[len(RUN_PREFIX):])


def selection_payloads(
    output_root: Path,
    manifests: Mapping[Path, Mapping[str, Any]],
) -> dict[Path, dict[str, Any]]:
    planned: dict[Path, dict[str, Any]] = {}
    by_root: dict[Path, list[tuple[Path, Mapping[str, Any]]]] = defaultdict(list)
    for path, payload in manifests.items():
        by_root[path.parent.parent].append((path, payload))
    for root, root_manifests in by_root.items():
        completed = [
            (path, payload)
            for path, payload in root_manifests
            if payload.get("status") == "completed"
        ]
        if not completed:
            continue
        old_path = root / SELECTION
        old_entries = (
            read_json(old_path).get("selections", []) if old_path.is_file() else []
        )
        pinned_runs = {
            str(entry.get("run"))
            for entry in old_entries
            if entry.get("mode") == "pinned"
        }
        groups: dict[str, list[tuple[Path, Mapping[str, Any]]]] = defaultdict(list)
        for item in completed:
            groups[canonical(scientific_config(item[1]))].append(item)
        entries = []
        for key in sorted(groups):
            items = groups[key]
            pinned = [item for item in items if item[0].parent.name in pinned_runs]
            if len(pinned) > 1:
                raise ValueError(f"{root}: multiple pinned repeats survive migration")
            selected = pinned[0] if pinned else max(
                items,
                key=lambda item: (
                    str(item[1].get("completed_at") or item[1].get("updated_at") or ""),
                    run_number(item[0]),
                ),
            )
            entries.append(
                {
                    "scientific_config": scientific_config(selected[1]),
                    "mode": "pinned" if pinned else "auto",
                    "run": selected[0].parent.name,
                }
            )
        planned[old_path] = {"schema_version": 1, "selections": entries}
    return planned


def dependency_state(value: Any) -> tuple[bool, bool]:
    if isinstance(value, Mapping):
        if old_reference(value):
            return False, True
        if isinstance(value.get("scientific"), Mapping) and isinstance(
            value.get("artifact"), Mapping
        ) and "revision" in value["artifact"]:
            return True, False
        states = [dependency_state(item) for item in value.values()]
        return any(state[0] for state in states), any(state[1] for state in states)
    if isinstance(value, list):
        states = [dependency_state(item) for item in value]
        return any(state[0] for state in states), any(state[1] for state in states)
    return False, False


def dependencies_are_current(value: Any) -> bool:
    found_current, found_legacy = dependency_state(value)
    return found_current and not found_legacy


def stale_report_roots(output_root: Path) -> list[Path]:
    roots = sorted(
        {
            path
            for path in output_root.rglob("reports")
            if path.is_dir() and path.name == "reports"
        },
        key=lambda path: len(path.parts),
    )
    top_level = [
        root
        for root in roots
        if not any(parent in root.parents for parent in roots)
    ]
    stale = []
    for root in top_level:
        manifests = sorted(root.rglob("*report_manifest.json"))
        payload_files = [
            path for path in root.rglob("*") if path.is_file() and path.name != ".gitkeep"
        ]
        if payload_files and (
            not manifests
            or any(not dependencies_are_current(read_json(path)) for path in manifests)
        ):
            stale.append(root)
    return stale


def archive_and_write(path: Path, payload: Mapping[str, Any]) -> None:
    original = read_json(path)
    if original == payload:
        return
    history = path.parent / "manifest_history"
    history.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(history / f"{stamp}_dependency_contract.json", original)
    write_json(path, payload)


def launch_overrides(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        experiment, separator, launch_id = value.partition("=")
        if not separator or not experiment or not launch_id:
            raise ValueError(f"Expected EXPERIMENT=LAUNCH_ID, got {value!r}")
        result[experiment] = launch_id
    return result


def infer_report_launches(stale_roots: list[Path]) -> dict[str, str]:
    """Recover unambiguous input launch IDs from the current report bundle."""
    launches: dict[str, str] = {}
    requiring_launch_id = {
        "channels_comparison",
        "coefficient_aggregation",
        "context_size",
        "foundation_models",
        "instance_normalization",
        "joint_panel",
        "self_augmentation",
        "user_generalization",
        "variate_modes",
    }
    for root in stale_roots:
        values = set()
        for path in root.rglob("*report_manifest.json"):
            if "history" in path.relative_to(root).parts:
                continue
            selection = read_json(path).get("selection", {})
            launch_id = selection.get("launch_id")
            if launch_id:
                values.add(str(launch_id))
        if not values and root.parent.name in requiring_launch_id:
            for path in root.parent.rglob(MANIFEST):
                if "reports" in path.parts or "manifest_history" in path.parts:
                    continue
                payload = read_json(path)
                launch_id = payload.get("launch", {}).get("launch_id")
                launch_id = launch_id or payload.get("launch_id")
                if launch_id:
                    values.add(str(launch_id))
        if len(values) == 1:
            launches[root.parent.name] = values.pop()
    return launches


def report_commands(
    project: str,
    stale_roots: list[Path],
    cluster: str,
    launches: Mapping[str, str],
) -> list[tuple[list[str], dict[str, str]]]:
    families = {root.parent.name for root in stale_roots}
    commands: list[tuple[list[str], dict[str, str]]] = []
    if project == "evaluating_tsfms":
        front = f"slurm/{cluster}/foundation_summary"
        front += "_selena.slurm" if cluster == "selena" else ".slurm"
        for family in sorted(families):
            launch_id = launches.get(family)
            if family in {"context_size", "instance_normalization"}:
                launch_id = launch_id or f"<{family}-original-launch-id>"
                commands.append(
                    (
                        [
                            "sbatch",
                            "--export="
                            f"ALL,TIME_EXPERIMENT={family},TIME_LAUNCH_ID={launch_id},"
                            "TIME_SUMMARY_SCRIPT=summarize_foundation_ablation.sh",
                            front,
                        ],
                        {},
                    )
                )
            elif family == "foundation_models":
                launch_id = launch_id or "<foundation_models-original-launch-id>"
                export = f"ALL,TIME_LAUNCH_ID={launch_id}"
                commands.append((["sbatch", f"--export={export}", front], {}))
            elif family == "channels_comparison":
                launch_id = launch_id or "<channels_comparison-original-launch-id>"
                commands.append(
                    (["bash", "scripts/channels_comparison.sh", cluster],
                     {"TIME_REPORT_ONLY": "1", "TIME_LAUNCH_ID": launch_id})
                )
    elif project == "fine_time":
        front = f"slurm/{cluster}/foundation_summary"
        front += "_selena.slurm" if cluster == "selena" else ".slurm"
        for family in sorted(families & {"context_size", "instance_normalization"}):
            launch_id = launches.get(family, f"<{family}-original-launch-id>")
            export = (
                f"ALL,TIME_EXPERIMENT={family},TIME_LAUNCH_ID={launch_id},"
                "TIME_SUMMARY_SCRIPT=summarize_foundation_ablation.sh"
            )
            commands.append((["sbatch", f"--export={export}", front], {}))
        if "foundation_models" in families:
            launch_id = launches.get("foundation_models", "<foundation_models-original-launch-id>")
            commands.append((["sbatch", f"--export=ALL,TIME_LAUNCH_ID={launch_id}", front], {}))
        if "channels_comparison" in families:
            launch_id = launches.get(
                "channels_comparison", "<channels_comparison-original-launch-id>"
            )
            commands.append((
                ["bash", "scripts/channels_comparison.sh", cluster],
                {"TIME_REPORT_ONLY": "1", "TIME_LAUNCH_ID": launch_id},
            ))
        if "task_finetuning" in families:
            front = (
                "slurm/selena/fine_time_summary_selena.slurm"
                if cluster == "selena"
                else "slurm/dgx/fine_time_summary.slurm"
            )
            commands.append((["sbatch", front], {}))
        if "task_finetuning_lora" in families:
            front = (
                "slurm/selena/fine_time_lora_summary_selena.slurm"
                if cluster == "selena"
                else "slurm/dgx/fine_time_lora_summary.slurm"
            )
            commands.append((["sbatch", front], {}))
    elif project == "selectime" and "scope_selection" in families:
        models = set()
        for root in stale_roots:
            if root.parent.name != "scope_selection":
                continue
            for manifest in root.rglob("*report_manifest.json"):
                relative = manifest.relative_to(root)
                if relative.parts:
                    models.add(relative.parts[0])
        if not models:
            raise ValueError(
                "Cannot infer Selectime report models from the stale report tree"
            )
        for model in sorted(models):
            commands.append((
                ["bash", "scripts/submit_experiment.sh", cluster, f"model={model}"],
                {"STAGES": "report"},
            ))
    elif project == "tsrag_time" and "tsrag" in families:
        commands.append((["bash", "scripts/submit_experiment.sh", cluster], {"STAGES": "report"}))
    elif project == "xtsf_time" and "shapley" in families:
        front = "xtsf_time_selena.slurm" if cluster == "selena" else "xtsf_time.slurm"
        commands.append((["sbatch", front], {"STAGES": "aggregate"}))
    elif project == "self_time" and "self_augmentation" in families:
        front = (
            "slurm/selena/self_time/summary_selena.slurm"
            if cluster == "selena"
            else "slurm/dgx/self_time/summary.slurm"
        )
        launch_id = launches.get(
            "self_augmentation", "<self_augmentation-original-launch-id>"
        )
        commands.append(
            (["sbatch", f"--export=ALL,TIME_LAUNCH_ID={launch_id}", front], {})
        )
    elif project == "classic_tsf" and "foundation_models" in families:
        front = (
            "slurm/selena/foundation_summary_selena.slurm"
            if cluster == "selena"
            else "slurm/dgx/foundation_summary.slurm"
        )
        launch_id = launches.get(
            "foundation_models", "<foundation_models-original-launch-id>"
        )
        commands.append(
            (["sbatch", f"--export=ALL,TIME_LAUNCH_ID={launch_id}", front], {})
        )
    elif project == "fl_time" and "coefficient_aggregation" in families:
        launch_id = launches.get(
            "coefficient_aggregation", "<coefficient_aggregation-original-launch-id>"
        )
        commands.append(
            (
                ["bash", "scripts/submit_experiment.sh"],
                {"STAGES": "report", "TIME_LAUNCH_ID": launch_id},
            )
        )
    elif project == "linear_time":
        for family in sorted(families & {"joint_panel", "user_generalization", "variate_modes"}):
            launch_id = launches.get(family, f"<{family}-original-launch-id>")
            commands.append(
                (
                    ["bash", "scripts/submit_study.sh", family],
                    {"STAGES": "report", "TIME_LAUNCH_ID": launch_id},
                )
            )
    supported = {
        "evaluating_tsfms": {"context_size", "instance_normalization", "foundation_models", "channels_comparison"},
        "fine_time": {"context_size", "instance_normalization", "foundation_models", "channels_comparison", "task_finetuning", "task_finetuning_lora"},
        "selectime": {"scope_selection"}, "tsrag_time": {"tsrag"},
        "xtsf_time": {"shapley"}, "self_time": {"self_augmentation"},
        "classic_tsf": {"foundation_models"},
        "fl_time": {"coefficient_aggregation"},
        "linear_time": {"joint_panel", "user_generalization", "variate_modes"},
    }.get(project, set())
    unsupported = sorted(families - supported)
    if unsupported:
        raise ValueError(
            "No report-only recovery command is defined for stale report families: "
            f"{unsupported}"
        )
    return commands


def dependency_roots_from_args(values: list[Path]) -> list[Path]:
    dependency_roots = list(values)
    seasonal = os.getenv("TIME_SEASONAL_EVALUATIONS_ROOT")
    if seasonal:
        seasonal_root = Path(seasonal).expanduser().resolve()
        dependency_roots.append(
            seasonal_root.parent
            if seasonal_root.name == "evaluations"
            else seasonal_root
        )
    else:
        seasonal_outputs = os.getenv("TIME_SEASONAL_OUTPUTS_ROOT")
        if seasonal_outputs:
            dependency_roots.append(Path(seasonal_outputs).expanduser().resolve())
        else:
            default = default_cluster_root("seasonal", "outputs", required=False)
            if default is not None and default.exists():
                dependency_roots.append(default)
    return dependency_roots


def default_cluster_root(
    project: str, kind: str, *, required: bool = True
) -> Path | None:
    nni_file = Path(
        os.environ.get("TIME_NNI_FILE", Path.home() / "codes/.secrets/nni")
    )
    if not nni_file.is_file():
        if required:
            raise ValueError(
                "Pass --outputs-root, export TIME_OUTPUTS, or provide the Selena "
                f"account file at {nni_file}"
            )
        return None
    nni = nni_file.read_text(encoding="utf-8").splitlines()[0].strip().lower()
    if re.fullmatch(r"[a-z][a-z0-9_-]*", nni) is None:
        raise ValueError(f"Invalid NNI in {nni_file}")
    return Path("/scratch/users") / nni / "codes" / project / kind


def configured_outputs_root(explicit: Path | None, project: str) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    configured = os.getenv("TIME_OUTPUTS") or os.getenv("OUTPUTS_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    default = default_cluster_root(project, "outputs")
    assert default is not None
    return default.resolve()


def migrate_configs_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        help="artifact root; defaults to TIME_OUTPUTS, OUTPUTS_ROOT, or Selena scratch",
    )
    parser.add_argument(
        "--dependency-root",
        type=Path,
        action="append",
        default=[],
        help="additional producer artifact root; may be repeated",
    )
    parser.add_argument(
        "--apply", action="store_true", help="write the reviewed migration plan"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    project = project_root.name
    output_root = configured_outputs_root(args.outputs_root, project)
    migration = Migration(output_root, dependency_roots_from_args(args.dependency_root))
    migration.plan()
    local_plans = {
        path: payload
        for path, payload in migration.planned.items()
        if path.is_relative_to(output_root)
    }
    selections = selection_payloads(output_root, local_plans)
    changed = sum(read_json(path) != payload for path, payload in local_plans.items())
    changed_selections = sum(
        not path.is_file() or read_json(path) != payload
        for path, payload in selections.items()
    )
    print(
        f"preflight project={project} manifests={len(local_plans)} "
        f"manifest_updates={changed} selection_updates={changed_selections}"
    )

    if not args.apply:
        print("dry_run=true; pass --apply after reviewing the preflight")
        return

    for path, payload in local_plans.items():
        archive_and_write(path, payload)
    for path, payload in selections.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, payload)
    print(
        f"applied project={project} manifest_updates={changed} "
        f"selection_updates={changed_selections}"
    )


def regenerate_reports_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        help="artifact root; defaults to TIME_OUTPUTS, OUTPUTS_ROOT, or Selena scratch",
    )
    parser.add_argument("--cluster", choices=("dgx", "selena"), default="selena")
    parser.add_argument(
        "--launch-id",
        action="append",
        default=[],
        metavar="FAMILY=ID",
        help="original launch identifier required by some summary fronts",
    )
    parser.add_argument(
        "--submit", action="store_true", help="submit the reviewed report-only plan"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    project = project_root.name
    output_root = configured_outputs_root(args.outputs_root, project)
    stale = stale_report_roots(output_root)
    launches = infer_report_launches(stale)
    launches.update(launch_overrides(args.launch_id))
    commands = report_commands(project, stale, args.cluster, launches)
    unresolved = [
        value
        for command, environment in commands
        for value in [*command, *environment.values()]
        if "<" in value and ">" in value
    ]
    print(
        f"preflight project={project} stale_report_roots={len(stale)} "
        f"report_commands={len(commands)}"
    )
    for root in stale:
        print(f"stale_report={root}")
    for command, environment in commands:
        prefix = " ".join(f"{key}={value}" for key, value in environment.items())
        rendered = " ".join(command)
        if prefix:
            rendered = f"{prefix} {rendered}"
        print(f"report_command={rendered}")
    if not args.submit:
        print("dry_run=true; pass --submit after reviewing the report plan")
        return
    if unresolved:
        raise ValueError(
            "Resolve report launch placeholders with --launch-id before submission: "
            f"{unresolved}"
        )
    for command, extra_environment in commands:
        environment = os.environ.copy()
        environment.update(extra_environment)
        subprocess.run(command, cwd=project_root, env=environment, check=True)


def main() -> None:
    raise SystemExit(
        "Use src/scripts/migrate_artifact_configs.py or "
        "src/scripts/regenerate_stale_reports.py"
    )


if __name__ == "__main__":
    main()
