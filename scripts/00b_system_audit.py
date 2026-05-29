#!/usr/bin/env python3
"""Collect reproducibility-critical environment metadata for Stage 0."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import (
    command_exists,
    json_dump,
    load_yaml,
    project_root,
    run_command,
    software_version,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def meminfo() -> dict[str, int | None]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return {"mem_total_kib": None, "mem_available_kib": None}
    values: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            number = int(value.strip().split()[0])
            values[key] = number
    return {
        "mem_total_kib": values.get("MemTotal"),
        "mem_available_kib": values.get("MemAvailable"),
    }


def nvidia_info() -> list[dict[str, str]]:
    if not command_exists("nvidia-smi"):
        return []
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.strip().splitlines():
        index, name, memory_total, driver_version = [part.strip() for part in line.split(",", 3)]
        gpus.append(
            {
                "index": index,
                "name": name,
                "memory_total_mib": memory_total,
                "driver_version": driver_version,
            }
        )
    return gpus


def conda_info() -> dict[str, object]:
    if not command_exists("conda"):
        return {"available": False}
    result = run_command(["conda", "info", "--json"])
    if result.returncode != 0:
        return {"available": True, "error": result.stderr.strip()}
    payload = json.loads(result.stdout)
    return {
        "available": True,
        "version": software_version("conda"),
        "active_prefix": payload.get("active_prefix"),
        "root_prefix": payload.get("root_prefix"),
        "active_environment_name": payload.get("active_prefix_name"),
        "solver": payload.get("solver"),
    }


def env_snapshot() -> dict[str, str]:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env and command_exists("conda"):
        result = run_command(["conda", "env", "export", "-n", conda_env, "--no-builds"])
        if result.returncode == 0:
            return {"kind": "conda_export", "content": result.stdout}
    result = run_command(["python3", "-m", "pip", "freeze"])
    return {"kind": "pip_freeze", "content": result.stdout if result.returncode == 0 else ""}


def tool_versions() -> dict[str, str | None]:
    def package_version(package: str) -> str | None:
        try:
            return importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            return None

    versions = {
        "python": platform.python_version(),
        "snakemake": package_version("snakemake") or software_version("snakemake"),
        "vina": software_version("vina"),
        "gnina": software_version("gnina", ["--version"]),
        "obabel": software_version("obabel", ["-V"]),
        "plip": package_version("plip") or software_version("plip", ["-h"]),
        "fpocket": software_version("fpocket", ["-h"]),
        "openmm": package_version("openmm"),
        "rdkit": package_version("rdkit"),
        "openai": package_version("openai"),
        "langgraph": package_version("langgraph"),
        "tar": software_version("tar"),
        "unzip": software_version("unzip", ["-v"]),
        "sqlite3": software_version("sqlite3"),
    }
    return versions


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage0 = config["stage0"]

    disk_targets = [root, Path("/"), Path("/data")]
    disk = {}
    for target in disk_targets:
        if target.exists():
            usage = shutil.disk_usage(target)
            disk[str(target)] = {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            }

    payload = {
        "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "hostname": platform.node(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu_count": os.cpu_count(),
        "memory": meminfo(),
        "disk": disk,
        "python_executable": shutil.which("python3"),
        "conda": conda_info(),
        "docker": {
            "available": command_exists("docker"),
            "version": software_version("docker"),
        },
        "gpus": nvidia_info(),
        "tool_versions": tool_versions(),
        "env_snapshot": env_snapshot(),
        "cwd": str(root),
    }
    json_dump(root / stage0["system_audit"], payload)


if __name__ == "__main__":
    main()
