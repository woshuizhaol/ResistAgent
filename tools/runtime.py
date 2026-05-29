#!/usr/bin/env python3
"""Shared runtime helpers for Stage 0 and later workflows."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

MAX_WORKERS = min(12, max(1, (os.cpu_count() or 1) // 2))


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """PyYAML loader that rejects duplicate mapping keys."""


def _construct_mapping_no_duplicates(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            line_number = int(getattr(key_node.start_mark, "line", 0)) + 1
            raise ValueError(f"Duplicate YAML key {key!r} at line {line_number}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_no_duplicates,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=_UniqueKeySafeLoader) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
    )


def command_exists(binary: str) -> bool:
    return shutil.which(binary) is not None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path | str, payload: Any) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def text_dump(path: Path | str, text: str) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def software_version(binary: str, args: list[str] | None = None) -> str | None:
    if not command_exists(binary):
        return None
    command = [binary] + (args or ["--version"])
    result = run_command(command)
    if result.returncode != 0:
        return None
    return (result.stdout or result.stderr).strip().splitlines()[0]


def detect_git_commit(root: Path) -> tuple[str | None, str]:
    result = run_command(["git", "rev-parse", "HEAD"], cwd=root)
    if result.returncode != 0:
        return None, "not_a_git_repo"
    return result.stdout.strip(), "clean_or_dirty_unknown"
