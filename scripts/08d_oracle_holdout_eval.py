#!/usr/bin/env python3
"""Evaluate counter-design step oracle v2 with grouped holdout splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import ensure_dir, json_dump, load_yaml, project_root
from tools.stage6_oracle_v2 import (
    build_stage6_oracle_training_frame,
    conservative_ensemble_holdout,
    evaluate_group_holdout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--stage5-subdir", default="stage5")
    parser.add_argument("--fallback-stage5-subdir", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage6 = dict(config.get("stage6", {}))
    cases_config = load_yaml(root / config["stage2"]["cases_frozen_config"])
    case_entries = list(cases_config.get("set_d", []))
    focus_case = next((case for case in case_entries if str(case.get("case_id")) == str(args.case_id)), None)
    if focus_case is None:
        raise SystemExit(f"No case matched --case-id={args.case_id}")

    frame = build_stage6_oracle_training_frame(
        root=root,
        case_entries=case_entries,
        preferred_stage5_subdir=str(args.stage5_subdir),
        fallback_stage5_subdirs=list(args.fallback_stage5_subdir or []),
    )
    case_frame = frame[frame["case_id"].eq(str(args.case_id))].copy()
    domain_frame = frame[frame["target_domain"].eq(str(focus_case.get("target_domain") or "unknown").lower())].copy()

    case_metrics = evaluate_group_holdout(
        frame=case_frame,
        stage6=stage6,
        group_column="target_key",
        label="case_group_holdout",
    )
    domain_frame = domain_frame.copy()
    if not domain_frame.empty:
        domain_frame["domain_group"] = domain_frame["case_id"].astype(str) + "::" + domain_frame["target_key"].astype(str)
    domain_metrics = evaluate_group_holdout(
        frame=domain_frame,
        stage6=stage6,
        group_column="domain_group",
        label="domain_group_holdout",
    )

    output_root = ensure_dir(root / "outputs" / str(args.case_id) / str(stage6.get("oracle_v2_output_dirname", "stage6_oracle_v2")))
    payload = {
        "case_id": str(args.case_id),
        "case_metrics": case_metrics,
        "domain_metrics": domain_metrics,
        "ensemble_metrics": conservative_ensemble_holdout(
            case_metrics=case_metrics,
            domain_metrics=domain_metrics,
            residual_weight=float(stage6.get("oracle_v2_conservative_residual_weight", 1.0)),
        ),
    }
    json_dump(output_root / "holdout_eval.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
