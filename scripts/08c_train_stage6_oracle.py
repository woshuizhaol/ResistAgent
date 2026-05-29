#!/usr/bin/env python3
"""Train counter-design step oracle v2 artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import load_yaml, project_root
from tools.stage6_oracle_v2 import fit_stage6_oracle_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--stage5-subdir", default="stage5")
    parser.add_argument("--fallback-stage5-subdir", action="append", default=[])
    return parser.parse_args()


def selected_cases(cases_config: dict[str, object], case_id: str | None) -> list[dict[str, object]]:
    cases = list(cases_config.get("set_d", []))
    if case_id is None:
        return cases
    return [case for case in cases if str(case.get("case_id")) == str(case_id)]


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage6 = dict(config.get("stage6", {}))
    cases_config = load_yaml(root / config["stage2"]["cases_frozen_config"])
    cases = selected_cases(cases_config, args.case_id)
    if not cases:
        raise SystemExit(f"No case matched --case-id={args.case_id}")
    for case_entry in cases:
        metadata = fit_stage6_oracle_v2(
            root=root,
            case_entries=list(cases_config.get("set_d", [])),
            focus_case_id=str(case_entry["case_id"]),
            stage6=stage6,
            preferred_stage5_subdir=str(args.stage5_subdir),
            fallback_stage5_subdirs=list(args.fallback_stage5_subdir or []),
        )
        print(metadata)


if __name__ == "__main__":
    main()
