#!/usr/bin/env python3
"""counter-design step postmortem reporting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import ensure_dir, iso_now, json_dump, load_yaml, project_root
from tools.stage5_utils import relative_path
from tools.stage6_postmortem import (
    action_family_yield_frame,
    constraint_funnel_frame,
    diversity_collapse_report_frame,
    ensure_stage6_health_columns,
    invalid_reason_breakdown_frame,
    llm_failure_context_audit_frame,
    load_calibrator_metadata,
    oracle_dependency_report_frame,
    panel_failure_matrix_frame,
    stage6_executability_metrics,
)
from tools.stage6_utils import read_csv_optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--stage6-subdir", default="stage6")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--update-qc", action="store_true")
    return parser.parse_args()


def selected_cases(cases_config: dict[str, object], case_id: str | None) -> list[dict[str, object]]:
    cases = list(cases_config.get("set_d", []))
    if case_id is None:
        return cases
    return [case for case in cases if str(case.get("case_id")) == str(case_id)]


def write_table(frame, path: Path) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    cases_config = load_yaml(root / config["stage2"]["cases_frozen_config"])
    cases = selected_cases(cases_config, args.case_id)
    if not cases:
        raise SystemExit(f"No counter-design step case matched --case-id={args.case_id}")

    missing_cases: list[str] = []
    for case_entry in cases:
        case_id = str(case_entry["case_id"])
        stage6_root = root / "outputs" / case_id / str(args.stage6_subdir)
        leaderboard_path = stage6_root / "leaderboard.csv"
        prefilter_path = stage6_root / "design_prefilter_audit.csv"
        if not leaderboard_path.exists():
            if args.skip_missing:
                missing_cases.append(case_id)
                continue
            raise SystemExit(f"Missing leaderboard for {case_id}: {leaderboard_path}")

        leaderboard = ensure_stage6_health_columns(read_csv_optional(leaderboard_path))
        prefilter_audit = read_csv_optional(prefilter_path)
        calibrator_metadata = load_calibrator_metadata(stage6_root / "calibrator" / "stage6_calibrator.json")

        postmortem_root = ensure_dir(stage6_root / "postmortem")
        outputs = {
            "constraint_funnel": postmortem_root / "constraint_funnel.csv",
            "invalid_reason_breakdown": postmortem_root / "invalid_reason_breakdown.csv",
            "panel_failure_matrix": postmortem_root / "panel_failure_matrix.csv",
            "action_family_yield": postmortem_root / "action_family_yield.csv",
            "diversity_collapse_report": postmortem_root / "diversity_collapse_report.csv",
            "oracle_dependency_report": postmortem_root / "oracle_dependency_report.csv",
            "llm_failure_context_audit": postmortem_root / "llm_failure_context_audit.csv",
        }

        write_table(constraint_funnel_frame(leaderboard), outputs["constraint_funnel"])
        write_table(invalid_reason_breakdown_frame(leaderboard), outputs["invalid_reason_breakdown"])
        write_table(panel_failure_matrix_frame(leaderboard), outputs["panel_failure_matrix"])
        write_table(action_family_yield_frame(leaderboard), outputs["action_family_yield"])
        write_table(diversity_collapse_report_frame(leaderboard), outputs["diversity_collapse_report"])
        write_table(oracle_dependency_report_frame(leaderboard, calibrator_metadata), outputs["oracle_dependency_report"])
        write_table(llm_failure_context_audit_frame(stage6_root), outputs["llm_failure_context_audit"])

        if args.update_qc:
            qc_path = stage6_root / "stage6_qc.json"
            qc_payload: dict[str, Any] = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {"case_id": case_id}
            qc_payload.update(stage6_executability_metrics(leaderboard))
            qc_payload["postmortem_generated_at"] = iso_now()
            qc_payload["postmortem_stage6_subdir"] = str(args.stage6_subdir)
            qc_payload["postmortem_artifacts"] = {
                key: relative_path(path, root)
                for key, path in outputs.items()
            }
            if not prefilter_audit.empty:
                qc_payload["prefilter_audit_row_count"] = int(len(prefilter_audit))
            if calibrator_metadata:
                qc_payload["calibrator_available"] = bool(calibrator_metadata.get("available", False))
                qc_payload["calibrator_reason"] = str(calibrator_metadata.get("reason") or "")
            json_dump(qc_path, qc_payload)

    if missing_cases:
        print(json.dumps({"skipped_missing_cases": missing_cases}, ensure_ascii=True))


if __name__ == "__main__":
    main()
