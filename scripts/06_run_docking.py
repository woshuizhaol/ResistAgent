#!/usr/bin/env python3
"""mutation-effect step paired WT/MT docking for top mutation-proposal step candidates."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import ensure_dir, load_yaml, project_root
from tools.stage5_utils import (
    build_hiv_reference,
    build_stage5_target_panel,
    run_stage5_pair_docking,
    stage5_for_case,
    write_json,
    write_table,
)

DOCKING_COLUMNS = [
    "case_id",
    "effect_scope",
    "target_key",
    "target_slug",
    "stage5_run_root",
    "stage4_rank",
    "risk_score",
    "impact_evidence_tier",
    "proxy_status",
    "representative_sample_id",
    "sample_source",
    "sample_root",
    "stage5_ready",
    "stage5_skip_reason",
    "stage5_selection_bucket",
    "used_synthetic_combo_model",
    "used_stage5_modeled_sample",
    "stage5_model_kind",
    "component_positions",
    "component_count",
    "stage4_delta_dock_proxy",
    "stage4_delta_ifp_proxy",
    "stage4_anchor_loss_fraction",
    "stage4_local_rmsd_a",
    "stage4_ddg_fold_surrogate",
    "stage5_status",
    "stage5_error",
    "stage5_attempt_count",
    "stage5_attempt_history_json",
    "wt_best_affinity_kcal_mol",
    "mt_best_affinity_kcal_mol",
    "delta_dock_kcal_mol",
    "wt_pose_count",
    "mt_pose_count",
    "wt_pose_sdf",
    "mt_pose_sdf",
    "wt_receptor_pdb",
    "mt_receptor_pdb",
    "wt_complex_docked_pdb",
    "mt_complex_docked_pdb",
    "docking_box_source",
    "started_at",
    "finished_at",
    "run_summary_json",
]

HIV_POSE_COLUMNS = [
    "case_id",
    "target_key",
    "target_slug",
    "effect_scope",
    "sample_id",
    "pose_set",
    "attempt_index",
    "seed",
    "mode_rank",
    "affinity_kcal_mol",
    "selected_for_stage5",
    "pose_label",
    "nnrti_min_distance_a",
    "nnrti_coverage_count",
    "nnrti_coverage_fraction",
    "active_site_min_distance_a",
    "active_site_coverage_count",
    "active_site_coverage_fraction",
    "pose_sdf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--case-id", default=None)
    return parser.parse_args()


def read_csv_optional(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns or [])
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns or [])


def selected_cases(cases_config: dict[str, object], case_id: str | None) -> list[dict[str, object]]:
    cases = list(cases_config.get("set_d", []))
    if case_id is None:
        return cases
    return [case for case in cases if str(case.get("case_id")) == str(case_id)]


def stage5_case_root(root: Path, case_id: str) -> Path:
    return ensure_dir(root / "outputs" / case_id / "stage5")


def run_docking_job(job: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
    return run_stage5_pair_docking(**job)


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage2 = config["stage2"]
    stage3_5 = config["stage3_5"]
    stage5 = dict(config["stage5"])
    cases_config = load_yaml(root / config["stage2"]["cases_frozen_config"])

    cases = selected_cases(cases_config, args.case_id)
    if not cases:
        raise SystemExit(f"No mutation-effect step case matched --case-id={args.case_id}")

    for case_entry in cases:
        case_id = str(case_entry["case_id"])
        case_stage5 = stage5_for_case(stage5, case_id)
        case_root = stage5_case_root(root, case_id)
        site_rank = read_csv_optional(case_root.parent / "stage4" / "mutation_rank.csv")
        combo_rank = read_csv_optional(case_root.parent / "stage4" / "combo_rank.csv")
        mutation_status = read_csv_optional(case_root.parent / "stage3_2" / "mutation_site_status.csv")

        target_panel = build_stage5_target_panel(
            root=root,
            case_id=case_id,
            site_rank=site_rank,
            combo_rank=combo_rank,
            mutation_status=mutation_status,
            stage5=case_stage5,
        )
        write_table(target_panel, case_root / "target_panel.csv")

        hiv_reference = build_hiv_reference(
            root=root,
            case_entry=case_entry,
            stage2=stage2,
            stage3_5=stage3_5,
        )

        docking_rows: list[dict[str, object]] = []
        pose_rows: list[dict[str, object]] = []
        docking_runs_root = ensure_dir(case_root / "docking_runs")
        jobs = [
            {
                "root": root,
                "case_id": case_id,
                "target_row": target_row,
                "stage5": case_stage5,
                "hiv_reference": hiv_reference,
                "output_root": docking_runs_root,
            }
            for target_row in target_panel.to_dict(orient="records")
        ]
        max_workers = max(1, min(int(case_stage5["max_parallel_jobs"]), len(jobs)))
        if max_workers == 1:
            for job in jobs:
                docking_row, run_pose_rows = run_docking_job(job)
                docking_rows.append(docking_row)
                pose_rows.extend(run_pose_rows)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(run_docking_job, job): job for job in jobs}
                for future in as_completed(future_map):
                    docking_row, run_pose_rows = future.result()
                    docking_rows.append(docking_row)
                    pose_rows.extend(run_pose_rows)

        docking_frame = pd.DataFrame.from_records(docking_rows, columns=DOCKING_COLUMNS).sort_values(
            ["effect_scope", "stage4_rank", "target_key"],
            ascending=[True, True, True],
            na_position="last",
        )
        pose_frame = pd.DataFrame.from_records(pose_rows, columns=HIV_POSE_COLUMNS).sort_values(
            ["effect_scope", "target_key", "pose_set", "attempt_index", "seed", "mode_rank"],
            ascending=[True, True, True, True, True, True],
            na_position="last",
        )
        write_table(docking_frame, case_root / "docking_scores.csv")
        write_table(pose_frame, case_root / "pose_ensemble.csv")
        write_table(pose_frame[pose_frame["pose_label"].fillna("").astype(str).ne("")], case_root / "hiv_nnrti_pose_qc.csv")

        summary = {
            "case_id": case_id,
            "selected_site_targets": int(target_panel["effect_scope"].eq("site").sum()) if not target_panel.empty else 0,
            "selected_combo_targets": int(target_panel["effect_scope"].eq("combo").sum()) if not target_panel.empty else 0,
            "stage5_ready_target_count": int(target_panel["stage5_ready"].fillna(False).astype(bool).sum()) if not target_panel.empty else 0,
            "docking_success_count": int(docking_frame["stage5_status"].eq("ok").sum()) if not docking_frame.empty else 0,
            "docking_failure_count": int(docking_frame["stage5_status"].eq("failed").sum()) if not docking_frame.empty else 0,
            "skipped_unready_count": int(docking_frame["stage5_status"].eq("skipped").sum()) if not docking_frame.empty else 0,
        }
        write_json(case_root / "docking_summary.json", summary)


if __name__ == "__main__":
    main()
