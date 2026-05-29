#!/usr/bin/env python3
"""Initialize Stage 0 project directories, manifest, and state."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import (
    command_exists,
    detect_git_commit,
    ensure_dir,
    iso_now,
    json_dump,
    load_yaml,
    project_root,
    run_command,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def environment_snapshot() -> dict[str, str]:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env and command_exists("conda"):
        result = run_command(["conda", "env", "export", "-n", conda_env, "--no-builds"])
        if result.returncode == 0:
            return {"kind": "conda_export", "content": result.stdout}
    result = run_command(["python3", "-m", "pip", "freeze"])
    return {"kind": "pip_freeze", "content": result.stdout if result.returncode == 0 else ""}


def software_versions() -> dict[str, str | None]:
    def package_version(package: str) -> str | None:
        try:
            return importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            return None

    versions = {"python": platform.python_version()}
    for binary, args, key in [
        ("python3", ["--version"], "python3"),
        ("snakemake", ["--version"], "snakemake_cli"),
        ("conda", ["--version"], "conda"),
        ("vina", ["--version"], "vina"),
        ("gnina", ["--version"], "gnina"),
        ("obabel", ["-V"], "obabel"),
        ("plip", ["-h"], "plip_cli"),
        ("fpocket", ["-h"], "fpocket"),
    ]:
        if not command_exists(binary):
            versions[key] = None
            continue
        result = run_command([binary] + args)
        if result.returncode == 0:
            versions[key] = (result.stdout or result.stderr).strip().splitlines()[0]
        else:
            versions[key] = None
    versions["snakemake"] = package_version("snakemake") or versions.get("snakemake_cli")
    versions["plip"] = package_version("plip") or versions.get("plip_cli")
    versions["openmm"] = package_version("openmm")
    versions["rdkit"] = package_version("rdkit")
    versions["openai"] = package_version("openai")
    versions["langgraph"] = package_version("langgraph")
    return versions


def hashed_inputs(paths: list[Path], root: Path) -> dict[str, str]:
    hashes = {}
    for path in paths:
        if path.exists():
            hashes[str(path.relative_to(root))] = sha256_file(path)
    return hashes


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage0 = config["stage0"]
    outputs_root = root / stage0["project_state_root"]
    stage_dirs = config["outputs"]["stage_dirs"]

    for stage_name in stage_dirs:
        ensure_dir(outputs_root / stage_name / "artifacts")
        ensure_dir(outputs_root / stage_name / "qc")
    ensure_dir(root / stage0["structure_cache_root"])
    ensure_dir(root / stage0["structure_raw_root"])

    manifest_path = outputs_root / "run_manifest.json"
    state_path = outputs_root / "state.json"

    git_commit, git_status = detect_git_commit(root)
    started_at = iso_now()
    input_paths = [
        root / "configs/base.yaml",
        root / "configs/cases.yaml",
        root / "schemas/run_manifest.schema.json",
        root / "schemas/state.schema.json",
        root / "workflows/Snakefile",
        root / stage0["data_audit"],
        root / stage0["system_audit"],
        root / stage0["alias_map"],
        root / stage0["archive_magic_report"],
        root / stage0["archive_index"],
    ]
    commands = [
        "bash scripts/00_data_audit.sh",
        "bash scripts/00b_system_audit.sh",
        "python3 scripts/03a_build_archive_index.py",
        "python3 scripts/00c_initialize_stage0.py",
        "python3 scripts/00d_validate_stage0.py",
    ]

    manifest = {
        "project_id": config["project"]["id"],
        "stage": "stage0",
        "git_commit": git_commit,
        "git_status": git_status,
        "software_versions": software_versions(),
        "env_snapshot": environment_snapshot(),
        "random_seeds": {"python": 0},
        "input_hashes": hashed_inputs(input_paths, root),
        "commands": commands,
        "started_at": started_at,
        "finished_at": iso_now(),
    }
    state = {
        "project_id": config["project"]["id"],
        "stage": "stage0",
        "inputs": {
            "config": "configs/base.yaml",
            "cases": "configs/cases.yaml",
            "data_root": stage0["data_root"],
        },
        "artifacts": {
            "file_manifest": stage0["file_manifest"],
            "data_audit": stage0["data_audit"],
            "system_audit": stage0["system_audit"],
            "alias_map": stage0["alias_map"],
            "archive_magic_report": stage0["archive_magic_report"],
            "archive_index": stage0["archive_index"],
            "structure_cache_root": stage0["structure_cache_root"],
            "structure_raw_root": stage0["structure_raw_root"],
        },
        "qc": {
            "validation_report": stage0["validation_report"],
        },
        "software_versions": manifest["software_versions"],
        "seeds": {"python": 0},
        "commands": commands,
        "llm_decisions": [],
    }
    json_dump(manifest_path, manifest)
    json_dump(state_path, state)


if __name__ == "__main__":
    main()
