"""Log cluster-visible devices and available memory without changing results."""

import json
import os
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def log(event, **values):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] resources {event} " + json.dumps(values, sort_keys=True), flush=True)


def memory_snapshot():
    """Distinguish host RAM availability from a job's cgroup memory headroom."""
    fields = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        fields = {
            line.split(":")[0]: int(line.split()[1]) * 1024
            for line in meminfo.read_text().splitlines()
            if line.startswith(("MemTotal:", "MemAvailable:"))
        }
    result = {key + "_MiB": value / 2**20 for key, value in fields.items()}
    membership = Path("/proc/self/cgroup")
    if membership.exists():
        for line in membership.read_text().splitlines():
            _, controllers, group = line.split(":", 2)
            if controllers == "":
                root = Path("/sys/fs/cgroup") / group.lstrip("/")
                limit_path, usage_path = root / "memory.max", root / "memory.current"
            elif "memory" in controllers.split(","):
                root = Path("/sys/fs/cgroup/memory") / group.lstrip("/")
                limit_path = root / "memory.limit_in_bytes"
                usage_path = root / "memory.usage_in_bytes"
            else:
                continue
            if not limit_path.exists() or not usage_path.exists():
                continue
            limit = limit_path.read_text().strip()
            if limit != "max":
                limit, usage = int(limit), int(usage_path.read_text())
                # Huge v1 sentinel limits do not represent a finite job budget.
                if limit < 2**60:
                    result.update(
                        cgroup_limit_MiB=limit / 2**20,
                        cgroup_used_MiB=usage / 2**20,
                        cgroup_headroom_MiB=max(0, limit - usage) / 2**20,
                    )
    return result


def main():
    log(
        "allocation",
        host=socket.gethostname(),
        **{name: os.environ.get(name, "unset") for name in (
            "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
            "CUDA_VISIBLE_DEVICES", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
            "SLURM_MEM_PER_CPU",
        )},
        cpu_count=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
    )
    try:
        log("RAM", **memory_snapshot())
    except (OSError, ValueError) as error:
        log("RAM_unavailable", reason=str(error))

    # Inventory is labelled separately: nvidia-smi may list more GPUs than CUDA.
    if shutil.which("nvidia-smi"):
        try:
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.free,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15, check=True,
            )
            log("nvidia_inventory", columns="index,uuid,name,driver,total_MiB,free_MiB,used_MiB",
                devices=probe.stdout.strip().splitlines())
        except (OSError, subprocess.SubprocessError) as error:
            log("nvidia_inventory_unavailable", reason=str(error))
    else:
        log("nvidia_inventory_unavailable", reason="nvidia-smi is not installed")

    # Probe CUDA inside the scheduler step, not on the login/submission host.
    try:
        import torch
        available = torch.cuda.is_available()
        log("torch", version=torch.__version__, cuda_available=available,
            visible_gpu_count=torch.cuda.device_count() if available else 0,
            note="visibility probe; CPU-only stages need not use an allocated GPU")
        if available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                free, total = torch.cuda.mem_get_info(index)
                log("cuda_device", logical_index=index, name=properties.name,
                    total_MiB=total / 2**20, free_MiB=free / 2**20,
                    capability=f"{properties.major}.{properties.minor}")
    except (ImportError, OSError, RuntimeError) as error:
        log("torch_probe_unavailable", reason=str(error))


if __name__ == "__main__":
    main()
