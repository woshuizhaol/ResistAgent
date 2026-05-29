#!/usr/bin/env python3
"""Build counter-design step receptor-ensemble manifests for configured cases."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import load_yaml, project_root
from tools.stage5_utils import build_stage5_target_panel, stage5_for_case
from tools.stage6_receptor_ensemble import read_csv_optional, write_receptor_ensemble_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--case-id", default=None)
    return parser.parse_args()


def merge_nested_dicts(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_nested_dicts(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def stage6_for_case(stage6: dict, case_id: str) -> dict:
    overrides = dict(stage6.get("case_overrides", {}))
    case_override = dict(overrides.get(case_id, {}))
    if not case_override:
        merged = copy.deepcopy(stage6)
        merged.pop("case_overrides", None)
        return merged
    merged = merge_nested_dicts(stage6, case_override)
    merged.pop("case_overrides", None)
    return merged


def selected_cases(cases_config: dict[str, object], case_id: str | None) -> list[dict[str, object]]:
    cases = list(cases_config.get("set_d", []))
    if case_id is None:
        return cases
    return [case for case in cases if str(case.get("case_id")) == str(case_id)]


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage5 = dict(config["stage5"])
    stage6 = dict(config["stage6"])
    cases_config = load_yaml(root / config["stage2"]["cases_frozen_config"])

    cases = selected_cases(cases_config, args.case_id)
    if not cases:
        raise SystemExit(f"No case matched --case-id={args.case_id}")

    for case_entry in cases:
        case_id = str(case_entry["case_id"])
        case_stage6 = stage6_for_case(stage6, case_id)
        if not bool(case_stage6.get("receptor_ensemble_enabled", False)):
            continue
        case_root = root / "outputs" / case_id
        target_panel = read_csv_optional(case_root / "stage5" / "target_panel.csv")
        if target_panel.empty:
            case_stage5 = stage5_for_case(stage5, case_id)
            site_rank = read_csv_optional(case_root / "stage4" / "mutation_rank.csv")
            combo_rank = read_csv_optional(case_root / "stage4" / "combo_rank.csv")
            mutation_status = read_csv_optional(case_root / "stage3_2" / "mutation_site_status.csv")
            target_panel = build_stage5_target_panel(
                root=root,
                case_id=case_id,
                site_rank=site_rank,
                combo_rank=combo_rank,
                mutation_status=mutation_status,
                stage5=case_stage5,
            )
        if not target_panel.empty and "stage5_ready" in target_panel.columns:
            target_panel = target_panel[target_panel["stage5_ready"].fillna(False).astype(bool)].copy()
        else:
            target_panel = target_panel.iloc[0:0].copy()
        summary = write_receptor_ensemble_artifacts(
            root=root,
            case_entry=case_entry,
            panel_frame=target_panel,
            stage6=case_stage6,
        )
        print(summary)


if __name__ == "__main__":
    main()
