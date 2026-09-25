"""Plain-config manifests for TIME experiment runs."""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
VALID_STATUSES = {"running", "interrupted", "computed", "completed"}
CONFLICT_POLICIES = ("overwrite_exact", "overwrite_path", "new")
CONFIG_POLICIES = ("error", "distinct", "latest", "average")
REPEAT_POLICIES = ("selected", "latest", "distinct", "average")
RUN_PATTERN = re.compile(r"run_(\d+)")
SELECTION_NAME = "SELECTED_RUNS.json"
PROJECT_NAME = os.environ.get(
    "TIME_PROJECT_NAME", Path(__file__).resolve().parents[3].name
)


class ManifestError(ValueError):
    """Raised when run identity or selection is ambiguous or invalid."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_interrupted_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Retain task recovery state when quota prevents an atomic replacement."""

    try:
        _write_manifest(path, manifest)
    except OSError as error:
        if error.errno not in {errno.EDQUOT, errno.ENOSPC}:
            raise
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
        path.write_text(
            json.dumps(dict(manifest), sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _run_index(run_dir: Path) -> int:
    match = RUN_PATTERN.fullmatch(run_dir.name)
    if match is None:
        raise ManifestError(f"Run directory must be named run_<n>: {run_dir}")
    return int(match.group(1))


def load_manifest(path_or_run: str | Path) -> dict[str, Any]:
    """Load and validate one current manifest."""
    path = Path(path_or_run)
    if path.is_dir():
        path = path / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"{path} uses schema_version={manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if manifest.get("status") not in VALID_STATUSES:
        raise ManifestError(f"{path} has invalid status {manifest.get('status')!r}")
    _run_index(path.parent)
    return manifest


def manifest_reference(path_or_run: str | Path) -> dict[str, Any]:
    """Compact dependency identity without recursively embedding its config."""
    path = Path(path_or_run)
    run_dir = path if path.is_dir() else path.parent
    manifest = load_manifest(run_dir)
    return {"identity": manifest["identity"], "run": run_dir.name,
        "completed_at": manifest.get("completed_at")}


@dataclass
class RunHandle:
    """One allocated run that becomes completed only after artifact validation."""

    run_dir: Path
    manifest: dict[str, Any]
    action: str
    _completed: bool = False

    @property
    def should_run(self) -> bool:
        return self.action not in {"skip", "finalize"}

    def __enter__(self) -> "RunHandle":
        return self

    def _artifacts(self, required_artifacts: Sequence[str] | None) -> list[str]:
        artifacts = [
            str(value)
            for value in (
                required_artifacts
                if required_artifacts is not None
                else self.manifest.get("required_artifacts", [])
            )
        ]
        if not artifacts:
            raise ManifestError(f"Run has no required artifacts: {self.run_dir}")
        missing = [
            name
            for name in artifacts
            if not (self.run_dir / name).is_file()
            or (self.run_dir / name).stat().st_size == 0
        ]
        if missing:
            raise ManifestError(
                f"Run has missing or empty required artifacts: {missing}"
            )
        return artifacts

    def compute(self, required_artifacts: Sequence[str]) -> None:
        """Preserve finished computation before a separate final check."""
        if self.action in {"skip", "finalize"}:
            self._completed = True
            return
        artifacts = self._artifacts(required_artifacts)
        self.manifest["required_artifacts"] = artifacts
        self.manifest["status"] = "computed"
        self.manifest.pop("error", None)
        self.manifest["computed_at"] = _now()
        self.manifest["updated_at"] = self.manifest["computed_at"]
        _write_manifest(self.run_dir / MANIFEST_NAME, self.manifest)
        self._completed = True

    def complete(self, required_artifacts: Sequence[str] | None = None) -> None:
        if self.action == "skip":
            self._completed = True
            return
        artifacts = self._artifacts(required_artifacts)
        self.manifest["required_artifacts"] = artifacts
        self.manifest["status"] = "completed"
        self.manifest.pop("error", None)
        self.manifest["completed_at"] = _now()
        self.manifest["updated_at"] = self.manifest["completed_at"]
        _write_manifest(self.run_dir / MANIFEST_NAME, self.manifest)
        self._completed = True
        _update_auto_selection(self.run_dir, self.manifest)

    def __exit__(self, error_type, error, traceback) -> bool:
        if self.action == "skip":
            return False
        if (
            self.manifest.get("status") not in {"computed", "completed"}
            and (error_type is not None or not self._completed)
        ):
            self.manifest["status"] = "interrupted"
            self.manifest["updated_at"] = _now()
            if error_type is not None:
                self.manifest["error"] = {
                    "type": error_type.__name__,
                    "message": str(error),
                }
            elif "error" not in self.manifest:
                self.manifest["error"] = {
                    "type": "IncompleteRun",
                    "message": "run exited without declaring complete artifacts",
                }
            _write_manifest(self.run_dir / MANIFEST_NAME, self.manifest)
        return False


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    lowered = value.casefold()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ManifestError(f"{name} must be true or false, got {value!r}")


def _run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and RUN_PATTERN.fullmatch(path.name)
            and (path / MANIFEST_NAME).is_file()
        ),
        key=_run_index,
    )


def _clear_run_artifacts(run_dir: Path) -> None:
    for path in run_dir.iterdir():
        if path.name in {MANIFEST_NAME, "manifest_history"}:
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown")).strip("_")


def _archive_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    history = run_dir / "manifest_history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = re.sub(r"[^0-9]", "", _now())[:20]
    launch_id = _safe_name(manifest.get("launch", {}).get("launch_id"))
    path = history / f"{stamp}_{launch_id}.json"
    suffix = 1
    while path.exists():
        path = history / f"{stamp}_{launch_id}_{suffix}.json"
        suffix += 1
    _write_manifest(path, manifest)


def _scientific_config_from_values(
    model_config: Mapping[str, Any],
    pipeline_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_config": dict(model_config),
        "pipeline_config": dict(pipeline_config),
        "experiment_config": dict(experiment_config),
    }


def _attempt(action: str, launched_at: str) -> dict[str, Any]:
    return {
        "action": action,
        "launch_id": os.environ.get("TIME_LAUNCH_ID"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "launched_at": launched_at,
    }


def _launch_matches(manifest: Mapping[str, Any], launch_id: str) -> bool:
    launch = manifest.get("launch", {})
    if launch.get("launch_id") == launch_id:
        return True
    return any(
        attempt.get("launch_id") == launch_id
        for attempt in launch.get("attempts", [])
    )


def interrupt_launch(root: str | Path, launch_id: str) -> list[Path]:
    """Mark task manifests still owned by a failed launch as interrupted."""
    base = Path(root).expanduser().resolve()
    if not base.exists():
        return []
    changed = []
    for manifest_path in sorted(base.rglob(MANIFEST_NAME)):
        if (
            "manifest_history" in manifest_path.relative_to(base).parts
            or RUN_PATTERN.fullmatch(manifest_path.parent.name) is None
        ):
            continue
        manifest = load_manifest(manifest_path)
        if (
            manifest["status"] == "running"
            and str(manifest.get("launch", {}).get("launch_id")) == str(launch_id)
        ):
            manifest["status"] = "interrupted"
            manifest["updated_at"] = _now()
            manifest["error"] = {
                "type": "InterruptedLaunch",
                "message": f"launch {launch_id} ended before task completion",
            }
            _write_interrupted_manifest(manifest_path, manifest)
            changed.append(manifest_path.parent)
    return changed


def _selection_entries(identity_root: Path) -> list[dict[str, Any]]:
    path = identity_root / SELECTION_NAME
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"{path} uses an unsupported selection schema")
    entries = list(payload.get("selections", []))
    if any(entry.get("mode") not in {"auto", "pinned"} for entry in entries):
        raise ManifestError(f"{path} contains an invalid selection mode")
    return entries


def _write_selection_entries(identity_root: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    _write_manifest(
        identity_root / SELECTION_NAME,
        {"schema_version": SCHEMA_VERSION, "selections": list(entries)},
    )


def _update_auto_selection(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    scientific = _scientific_config(manifest)
    entries = _selection_entries(run_dir.parent)
    for entry in entries:
        if entry.get("scientific_config") == scientific:
            if entry.get("mode") == "auto":
                entry["run"] = run_dir.name
            break
    else:
        entries.append(
            {"scientific_config": scientific, "mode": "auto", "run": run_dir.name}
        )
    _write_selection_entries(run_dir.parent, entries)


def _select_reuse_source(
    source: str | Path,
    identity: Mapping[str, Any],
    scientific: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    source_root = Path(source).expanduser().resolve()
    if not source_root.exists():
        raise ManifestError(f"Reuse source does not exist: {source_root}")
    if source_root.is_file():
        source_paths = [source_root]
    elif RUN_PATTERN.fullmatch(source_root.name) and (source_root / MANIFEST_NAME).is_file():
        source_paths = [source_root / MANIFEST_NAME]
    else:
        source_paths = [
            path
            for path in source_root.rglob(MANIFEST_NAME)
            if "manifest_history" not in path.relative_to(source_root).parts
            and RUN_PATTERN.fullmatch(path.parent.name)
        ]
    reusable = []
    for source_path in source_paths:
        source_manifest = load_manifest(source_path)
        if (
            source_manifest["status"] == "completed"
            and source_manifest.get("identity") == dict(identity)
            and _scientific_config(source_manifest) == dict(scientific)
        ):
            reusable.append((source_path.parent, source_manifest))
    if not reusable:
        raise ManifestError(
            "No completed reuse source has the expected identity and scientific "
            f"configuration below {source_root}"
        )
    if len(reusable) > 1:
        selected = []
        for source_dir, source_manifest in reusable:
            for entry in _selection_entries(source_dir.parent):
                if (
                    entry.get("scientific_config") == dict(scientific)
                    and entry.get("run") == source_dir.name
                ):
                    selected.append((source_dir, source_manifest))
                    break
        reusable = selected
    if len(reusable) != 1:
        raise ManifestError(
            f"Reuse source is ambiguous for {dict(identity)}; select one completed run"
        )
    source_dir, source_manifest = reusable[0]
    compact_artifacts = ("config.json", "metrics_summary.json")
    missing = [
        name
        for name in compact_artifacts
        if not (source_dir / name).is_file()
        or (source_dir / name).stat().st_size == 0
    ]
    if missing:
        raise ManifestError(
            f"Reusable run lacks compact summary artifacts {missing}: {source_dir}"
        )
    return source_dir, source_manifest


def set_selected_run(run_dir: str | Path, *, pinned: bool = True) -> None:
    """Pin or automatically track one completed exact configuration repeat."""
    selected_dir = Path(run_dir).expanduser().resolve()
    manifest = load_manifest(selected_dir)
    if manifest["status"] != "completed":
        raise ManifestError(f"Cannot select an incomplete run: {selected_dir}")
    scientific = _scientific_config(manifest)
    entries = _selection_entries(selected_dir.parent)
    replacement = {
        "scientific_config": scientific,
        "mode": "pinned" if pinned else "auto",
        "run": selected_dir.name,
    }
    for index, entry in enumerate(entries):
        if entry.get("scientific_config") == scientific:
            entries[index] = replacement
            break
    else:
        entries.append(replacement)
    _write_selection_entries(selected_dir.parent, entries)


def allocate_run(
    identity_root: str | Path,
    *,
    experiment: str,
    identity: Mapping[str, Any],
    model_config: Mapping[str, Any],
    pipeline_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    policy: str | None = None,
    skip_completed: bool | None = None,
    force: bool | None = None,
    run_index: int | None = None,
    reuse_from: str | Path | None = None,
) -> RunHandle:
    """Skip, resume, overwrite, or allocate one exact task configuration."""
    policy = policy or os.environ.get("TIME_RUN_CONFLICT_POLICY", "overwrite_exact")
    if policy not in CONFLICT_POLICIES:
        raise ManifestError(f"Run conflict policy must be one of {CONFLICT_POLICIES}")
    if skip_completed is None:
        skip_completed = _bool_env("TIME_SKIP_COMPLETED", True)
    if force is None:
        force = _bool_env("TIME_FORCE_RERUN", False)
    if run_index is None and os.environ.get("TIME_RUN_INDEX"):
        run_index = int(os.environ["TIME_RUN_INDEX"])
    if run_index is not None and run_index < 0:
        raise ManifestError("run_index must be non-negative")

    root = Path(identity_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dirs = _run_dirs(root)
    scientific = _scientific_config_from_values(
        model_config, pipeline_config, experiment_config
    )
    existing = [(path, load_manifest(path)) for path in run_dirs]
    exact = [
        (path, manifest)
        for path, manifest in existing
        if manifest.get("experiment") == str(experiment)
        and manifest.get("identity") == dict(identity)
        and _scientific_config(manifest) == scientific
    ]
    computed = [
        (path, manifest)
        for path, manifest in exact
        if manifest["status"] == "computed"
        and (run_index is None or _run_index(path) == run_index)
    ]
    if computed and policy == "overwrite_exact" and not force:
        target, manifest = max(computed, key=lambda item: _run_index(item[0]))
        launched_at = _now()
        attempt = _attempt("finalize", launched_at)
        previous_launch = dict(manifest.get("launch", {}))
        attempt["computed_by"] = {
            "launch_id": previous_launch.get("launch_id"),
            "slurm_job_id": previous_launch.get("slurm_job_id"),
            "computed_at": manifest.get("computed_at"),
        }
        manifest["launch"] = {
            "launch_id": attempt["launch_id"],
            "slurm_job_id": attempt["slurm_job_id"],
            "launched_at": launched_at,
            "action": "finalize",
            "attempts": [*previous_launch.get("attempts", []), attempt],
        }
        manifest["updated_at"] = launched_at
        _write_manifest(target / MANIFEST_NAME, manifest)
        print(
            "TIME run allocation "
            f"action=finalize run={target} launch_id={attempt['launch_id']} "
            f"slurm_job_id={attempt['slurm_job_id']} launched_at={launched_at}",
            flush=True,
        )
        return RunHandle(target, manifest, "finalize", _completed=True)
    strict_reuse = reuse_from or os.environ.get("TIME_REUSE_FROM") or None
    optional_reuse = (
        None
        if strict_reuse is not None
        else os.environ.get("TIME_REUSE_IF_AVAILABLE_FROM") or None
    )
    selected_reuse = strict_reuse or optional_reuse
    reusable = None
    if selected_reuse is not None:
        try:
            reusable = _select_reuse_source(selected_reuse, identity, scientific)
        except ManifestError as error:
            if strict_reuse is not None:
                raise
            print(
                "TIME run allocation "
                f"action=recompute reuse_source={Path(selected_reuse).expanduser()} "
                f"reason={error}",
                flush=True,
            )
    if reusable is not None:
        source_dir, source_manifest = reusable
        compact_artifacts = ("config.json", "metrics_summary.json")
        completed_local = [
            (path, manifest)
            for path, manifest in exact
            if manifest["status"] == "completed"
        ]
        if completed_local and skip_completed and not force:
            target, local_manifest = max(
                completed_local, key=lambda item: _run_index(item[0])
            )
            reused_at = _now()
            attempt = _attempt("reuse", reused_at)
            launch = dict(local_manifest.get("launch", {}))
            launch["attempts"] = [*launch.get("attempts", []), attempt]
            local_manifest["launch"] = launch
            local_manifest["updated_at"] = reused_at
            _write_manifest(target / MANIFEST_NAME, local_manifest)
            print(
                "TIME run allocation "
                f"action=skip run={target} source_launch={launch.get('launch_id')} "
                f"launch_id={attempt['launch_id']} slurm_job_id={attempt['slurm_job_id']} "
                f"launched_at={reused_at}",
                flush=True,
            )
            return RunHandle(target, local_manifest, "skip", _completed=True)
        running_local = [
            path for path, manifest in exact if manifest["status"] == "running"
        ]
        if running_local and not force:
            raise ManifestError(f"Matching run is already running: {running_local[-1]}")

        if run_index is not None:
            target = root / f"run_{run_index}"
        elif exact:
            target = max(exact, key=lambda item: _run_index(item[0]))[0]
        else:
            next_index = 0 if not run_dirs else max(map(_run_index, run_dirs)) + 1
            target = root / f"run_{next_index}"
        old_manifest = load_manifest(target) if target.exists() else None
        if old_manifest is not None:
            same = (
                old_manifest.get("experiment") == str(experiment)
                and old_manifest.get("identity") == dict(identity)
                and _scientific_config(old_manifest) == scientific
            )
            if not same and policy != "overwrite_path":
                raise ManifestError(
                    f"{target} contains a different configuration; use overwrite_path"
                )
            _archive_manifest(target, old_manifest)
            _clear_run_artifacts(target)
        else:
            target.mkdir()
        for name in compact_artifacts:
            shutil.copy2(source_dir / name, target / name)
        reused_at = _now()
        source_manifest_path = source_dir / MANIFEST_NAME
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "project": PROJECT_NAME,
            "experiment": str(experiment),
            "identity": dict(identity),
            "model_config": dict(model_config),
            "pipeline_config": dict(pipeline_config),
            "runtime_config": dict(runtime_config),
            "experiment_config": dict(experiment_config),
            "provenance": {
                **dict(provenance or {}),
                "reused_from_manifest": str(source_manifest_path),
                "reused_from_experiment": source_manifest.get("experiment"),
            },
            "launch": {
                "launch_id": os.environ.get("TIME_LAUNCH_ID"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "launched_at": reused_at,
                "action": "reuse",
                "attempts": [_attempt("reuse", reused_at)],
            },
            "status": "completed",
            "required_artifacts": list(compact_artifacts),
            "started_at": reused_at,
            "completed_at": reused_at,
            "updated_at": reused_at,
        }
        _write_manifest(target / MANIFEST_NAME, manifest)
        _update_auto_selection(target, manifest)
        print(
            "TIME run allocation "
            f"action=reuse run={target} source={source_manifest_path} "
            f"launch_id={manifest['launch']['launch_id']} "
            f"slurm_job_id={manifest['launch']['slurm_job_id']} reused_at={reused_at}",
            flush=True,
        )
        return RunHandle(target, manifest, "skip", _completed=True)

    target = root / f"run_{run_index}" if run_index is not None else None
    old_manifest = load_manifest(target) if target is not None and target.exists() else None
    action = "new"

    if target is not None and target.exists() and policy == "new":
        raise ManifestError(f"Requested new run already exists: {target}")
    if old_manifest is not None and policy == "overwrite_path":
        action = "overwrite"
    if target is None and policy == "new":
        next_index = 0 if not run_dirs else max(map(_run_index, run_dirs)) + 1
        target = root / f"run_{next_index}"
    elif target is None and policy == "overwrite_path" and run_dirs:
        target = run_dirs[-1]
        old_manifest = load_manifest(target)
        action = "overwrite"
    elif target is None and exact:
        target, old_manifest = max(exact, key=lambda item: _run_index(item[0]))
    elif target is None:
        next_index = 0 if not run_dirs else max(map(_run_index, run_dirs)) + 1
        target = root / f"run_{next_index}"

    if old_manifest is not None and action != "overwrite":
        same = (
            old_manifest.get("experiment") == str(experiment)
            and old_manifest.get("identity") == dict(identity)
            and _scientific_config(old_manifest) == scientific
        )
        if not same and policy != "overwrite_path":
            raise ManifestError(
                f"{target} contains a different configuration; use overwrite_path"
            )
        if same:
            status = old_manifest["status"]
            if status == "completed" and skip_completed and not force:
                launched_at = _now()
                attempt = _attempt("reuse", launched_at)
                launch = dict(old_manifest.get("launch", {}))
                launch["attempts"] = [*launch.get("attempts", []), attempt]
                old_manifest["launch"] = launch
                old_manifest["updated_at"] = launched_at
                _write_manifest(target / MANIFEST_NAME, old_manifest)
                print(
                    "TIME run allocation "
                    f"action=skip run={target} source_launch={launch.get('launch_id')} "
                    f"launch_id={attempt['launch_id']} slurm_job_id={attempt['slurm_job_id']} "
                    f"launched_at={launched_at}",
                    flush=True,
                )
                return RunHandle(target, old_manifest, "skip", _completed=True)
            if status == "running" and not force:
                raise ManifestError(f"Matching run is already running: {target}")
            action = "resume" if status == "interrupted" and not force else "overwrite"
        else:
            action = "overwrite"

    assert target is not None
    launched_at = _now()
    previous_launch = dict((old_manifest or {}).get("launch", {}))
    if old_manifest is not None:
        if _scientific_config(old_manifest) != scientific:
            entries = [
                entry
                for entry in _selection_entries(target.parent)
                if entry.get("run") != target.name
            ]
            _write_selection_entries(target.parent, entries)
        _archive_manifest(target, old_manifest)
        _clear_run_artifacts(target)
    else:
        target.mkdir()
    attempts = (
        list(previous_launch.get("attempts", [])) if action == "resume" else []
    )
    attempt = _attempt(action, launched_at)
    if action == "resume":
        attempt["resumed_from"] = {
            "launch_id": previous_launch.get("launch_id"),
            "slurm_job_id": previous_launch.get("slurm_job_id"),
            "launched_at": previous_launch.get("launched_at")
            or old_manifest.get("started_at"),
        }
    attempts.append(attempt)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "project": PROJECT_NAME,
        "experiment": str(experiment),
        "identity": dict(identity),
        "model_config": dict(model_config),
        "pipeline_config": dict(pipeline_config),
        "runtime_config": dict(runtime_config),
        "experiment_config": dict(experiment_config),
        "provenance": dict(provenance or {}),
        "launch": {
            "launch_id": attempt["launch_id"],
            "slurm_job_id": attempt["slurm_job_id"],
            "launched_at": launched_at,
            "action": action,
            "attempts": attempts,
        },
        "status": "running",
        "required_artifacts": [],
        "started_at": launched_at,
        "updated_at": launched_at,
    }
    _write_manifest(target / MANIFEST_NAME, manifest)
    print(
        "TIME run allocation "
        f"action={action} run={target} launch_id={attempt['launch_id']} "
        f"slurm_job_id={attempt['slurm_job_id']} launched_at={launched_at}"
        + (
            " "
            f"resumed_from_launch={previous_launch.get('launch_id')} "
            f"resumed_from_job={previous_launch.get('slurm_job_id')}"
            if action == "resume"
            else ""
        ),
        flush=True,
    )
    return RunHandle(target, manifest, action)


def _nested_value(manifest: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = manifest
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def parse_config_filters(values: Sequence[str] | None) -> dict[str, Any]:
    """Parse repeatable ``manifest.path=JSON`` aggregate filters."""
    filters: dict[str, Any] = {}
    for value in values or []:
        key, separator, raw = value.partition("=")
        if not separator or not key:
            raise ManifestError(f"Run-config filter must be FIELD=JSON, got {value!r}")
        filters[key] = json.loads(raw)
    return filters


def resolve_target_mode(
    requested: str,
    *,
    target_dim: int,
    supports_multivariate: bool,
) -> str:
    """Resolve ``auto`` to the representation actually passed to the model."""
    if requested not in {"auto", "univariate", "multivariate"}:
        raise ValueError(
            "target_mode must be auto, univariate, or multivariate"
        )
    if requested == "multivariate" and not supports_multivariate:
        raise ValueError("This model does not support multivariate target inputs")
    if requested == "multivariate" and target_dim < 2:
        raise ValueError("A one-channel dataset cannot run in multivariate mode")
    if requested == "auto":
        if supports_multivariate and target_dim > 1:
            return "multivariate"
        return "univariate"
    return requested


def _scientific_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_config": manifest.get("model_config", {}),
        "pipeline_config": manifest.get("pipeline_config", {}),
        "experiment_config": manifest.get("experiment_config", {}),
    }


def _same_config_groups(
    manifests: Sequence[tuple[Path, dict[str, Any]]],
) -> list[list[tuple[Path, dict[str, Any]]]]:
    groups: list[list[tuple[Path, dict[str, Any]]]] = []
    for item in manifests:
        config = _scientific_config(item[1])
        for group in groups:
            if _scientific_config(group[0][1]) == config:
                group.append(item)
                break
        else:
            groups.append([item])
    return groups


def _latest_key(item: tuple[Path, Mapping[str, Any]]) -> tuple[str, int]:
    manifest = item[1]
    timestamp = str(
        manifest.get("completed_at")
        or manifest.get("updated_at")
        or manifest.get("started_at")
        or ""
    )
    return timestamp, _run_index(item[0])


def _flatten_config(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten_config(item, path))
        else:
            flattened[path] = item
    return flattened


def _label_value(value: Any) -> str:
    if isinstance(value, list):
        value = "-".join(map(str, value))
    return _safe_name(value)


def _selected_repeat(
    identity_root: Path,
    scientific: Mapping[str, Any],
    items: Sequence[tuple[Path, dict[str, Any]]],
) -> tuple[Path, dict[str, Any]]:
    for entry in _selection_entries(identity_root):
        if entry.get("scientific_config") == dict(scientific):
            run_name = entry.get("run")
            for item in items:
                if item[0].name == run_name:
                    return item
            raise ManifestError(f"{identity_root}/{run_name} is selected but unavailable")
    return max(items, key=_latest_key)


def select_completed_runs(
    root: str | Path,
    *,
    models: set[str] | None = None,
    target_modes: set[str] | None = None,
    launch_id: str | None = None,
    config_filters: Mapping[str, Any] | None = None,
    config_policy: str = "error",
    repeat_policy: str = "selected",
) -> list[tuple[Path, dict[str, Any]]]:
    """Select completed task runs under explicit config and repeat policies."""
    if config_policy not in CONFIG_POLICIES:
        raise ManifestError(f"config_policy must be one of {CONFIG_POLICIES}")
    if repeat_policy not in REPEAT_POLICIES:
        raise ManifestError(f"repeat_policy must be one of {REPEAT_POLICIES}")
    filters = dict(config_filters or {})
    candidates: list[tuple[Path, dict[str, Any]]] = []
    root = Path(root).expanduser().resolve()
    if not root.exists():
        return []
    for path in sorted(root.rglob(MANIFEST_NAME)):
        if RUN_PATTERN.fullmatch(path.parent.name) is None:
            continue
        manifest = load_manifest(path)
        if manifest["status"] != "completed":
            continue
        identity = manifest.get("identity", {})
        if models is not None and identity.get("model") not in models:
            continue
        if target_modes is not None and identity.get("target_mode") not in target_modes:
            continue
        if launch_id is not None and not _launch_matches(manifest, launch_id):
            continue
        if any(_nested_value(manifest, key) != value for key, value in filters.items()):
            continue
        candidates.append((path.parent, manifest))

    by_identity: list[list[tuple[Path, dict[str, Any]]]] = []
    for item in candidates:
        identity = item[1].get("identity", {})
        for group in by_identity:
            if group[0][1].get("identity", {}) == identity:
                group.append(item)
                break
        else:
            by_identity.append([item])

    selected: list[tuple[Path, dict[str, Any]]] = []
    for identity_group in by_identity:
        config_groups = _same_config_groups(identity_group)
        if len(config_groups) > 1 and config_policy == "error":
            identity = identity_group[0][1]["identity"]
            raise ManifestError(
                "Multiple scientific run configurations match "
                f"{identity}; use --run-config or an explicit distinct, latest, "
                "or average config policy"
            )
        if config_policy == "latest" and len(config_groups) > 1:
            latest_group = max(
                config_groups,
                key=lambda group: _latest_key(max(group, key=_latest_key)),
            )
            config_groups = [latest_group]

        flattened = [
            _flatten_config(_scientific_config(group[0][1]))
            for group in config_groups
        ]
        differing_keys = {
            key
            for key in set().union(*(values.keys() for values in flattened))
            if len(
                {
                    json.dumps(values.get(key), sort_keys=True)
                    for values in flattened
                }
            )
            > 1
        }
        for group, flat_config in zip(config_groups, flattened):
            group = sorted(group, key=_latest_key)
            scientific = _scientific_config(group[0][1])
            if repeat_policy == "latest":
                chosen = [group[-1]]
            elif repeat_policy == "selected":
                chosen = [_selected_repeat(group[0][0].parent, scientific, group)]
            else:
                chosen = group

            config_suffix = ""
            if config_policy == "distinct" and differing_keys:
                config_suffix = "__" + "__".join(
                    f"{_safe_name(key)}-{_label_value(flat_config.get(key))}"
                    for key in sorted(differing_keys)
                )
            for path, manifest in chosen:
                selected_manifest = dict(manifest)
                model_label = f"{manifest['identity']['model']}{config_suffix}"
                if repeat_policy == "distinct" and len(group) > 1:
                    model_label = f"{model_label}_{path.name}"
                selected_manifest["selection"] = {
                    "config_policy": config_policy,
                    "repeat_policy": repeat_policy,
                    "scientific_config": scientific,
                    "model_label": model_label,
                    "run": path.name,
                }
                selected.append((path, selected_manifest))

    experiments = {item[1].get("experiment") for item in selected}
    if len(experiments) > 1:
        raise ManifestError(
            f"Selected runs cross experiment roots: {sorted(experiments)}; "
            "point the result reader at one experiment root"
        )

    by_task: list[list[tuple[Path, dict[str, Any]]]] = []
    for item in selected:
        identity = item[1]["identity"]
        key = (
            identity["model"],
            identity["dataset"],
            identity["frequency"],
            identity["term"],
        )
        for group in by_task:
            sample = group[0][1]["identity"]
            sample_key = (
                sample["model"],
                sample["dataset"],
                sample["frequency"],
                sample["term"],
            )
            if key == sample_key:
                group.append(item)
                break
        else:
            by_task.append([item])
    for group in by_task:
        modes = {item[1]["identity"]["target_mode"] for item in group}
        if len(modes) > 1:
            identity = group[0][1]["identity"]
            raise ManifestError(
                "Multiple target modes match "
                f"{identity['model']} {identity['dataset']}/{identity['frequency']} "
                f"{identity['term']}: {sorted(modes)}; use --target-mode"
            )

    if config_policy == "error":
        by_model: list[list[tuple[Path, dict[str, Any]]]] = []
        for item in selected:
            identity = item[1]["identity"]
            key = identity["model"]
            for group in by_model:
                sample_identity = group[0][1]["identity"]
                if key == sample_identity["model"]:
                    group.append(item)
                    break
            else:
                by_model.append([item])
        for group in by_model:
            global_configs = [
                {
                    "model_config": item[1].get("model_config", {}),
                    "covariate_mode": item[1]
                    .get("experiment_config", {})
                    .get("covariate_mode"),
                }
                for item in group
            ]
            if any(config != global_configs[0] for config in global_configs[1:]):
                identity = group[0][1]["identity"]
                raise ManifestError(
                    "Selected tasks mix model or experiment configurations for "
                    f"{identity['model']}; use "
                    "--run-config to select one configuration"
                )
    return sorted(
        selected,
        key=lambda item: tuple(str(value) for value in item[1]["identity"].values()),
    )
