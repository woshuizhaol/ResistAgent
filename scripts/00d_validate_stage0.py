#!/usr/bin/env python3
"""Validate Stage 0 manifest/state JSON against schema and basic output presence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jsonschema import validate

from tools.runtime import json_dump, load_yaml, project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage0 = config["stage0"]
    project_state_root = root / stage0["project_state_root"]

    run_manifest = load_json(project_state_root / "run_manifest.json")
    state = load_json(project_state_root / "state.json")
    run_manifest_schema = load_json(root / "schemas/run_manifest.schema.json")
    state_schema = load_json(root / "schemas/state.schema.json")
    validate(instance=run_manifest, schema=run_manifest_schema)
    validate(instance=state, schema=state_schema)

    expected_outputs = {
        "file_manifest": root / stage0["file_manifest"],
        "data_audit": root / stage0["data_audit"],
        "system_audit": root / stage0["system_audit"],
        "alias_map": root / stage0["alias_map"],
        "archive_magic_report": root / stage0["archive_magic_report"],
        "archive_index": root / stage0["archive_index"],
        "structure_cache_root": root / stage0["structure_cache_root"],
        "structure_raw_root": root / stage0["structure_raw_root"],
        "run_manifest": project_state_root / "run_manifest.json",
        "state": project_state_root / "state.json",
    }
    payload = {
        "project_id": config["project"]["id"],
        "stage": "stage0",
        "schemas_valid": True,
        "checks": {name: path.exists() for name, path in expected_outputs.items()},
    }
    json_dump(root / stage0["validation_report"], payload)


if __name__ == "__main__":
    main()
