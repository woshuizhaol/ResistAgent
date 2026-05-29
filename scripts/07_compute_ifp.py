#!/usr/bin/env python3
"""mutation-effect step PLIP IFP diffs and deterministic mechanism features."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import load_yaml, project_root
from tools.stage5_utils import (
    compute_ifp_effect_row,
    load_anchor_labels,
    stage5_for_case,
    write_json,
    write_table,
)

IFP_COLUMNS = [
    "case_id",
    "effect_scope",
    "target_key",
    "target_slug",
    "representative_sample_id",
    "stage5_status",
    "impact_evidence_tier",
    "sample_source",
    "used_synthetic_combo_model",
    "used_stage5_modeled_sample",
    "stage5_model_kind",
    "stage4_rank",
    "risk_score",
    "delta_dock_kcal_mol",
    "stage4_local_rmsd_a",
    "ifp_status",
    "ifp_error",
    "refinement_status",
    "refinement_error",
    "local_sampling_applied",
    "ifp_jaccard_loss",
    "anchor_loss_fraction",
    "wt_residue_count",
    "mt_residue_count",
    "lost_residue_labels_json",
    "gained_residue_labels_json",
    "lost_anchor_labels_json",
    "hydrogen_bond_delta_count",
    "salt_bridge_delta_count",
    "hydrophobic_delta_count",
    "pi_stacking_delta_count",
    "pi_cation_delta_count",
    "metal_complex_delta_count",
    "wt_pocket_volume_proxy_a3",
    "mt_pocket_volume_proxy_a3",
    "wt_fpocket_volume_a3",
    "mt_fpocket_volume_a3",
    "fpocket_error",
    "pocket_volume_change_fraction",
    "wt_contact_density",
    "mt_contact_density",
    "contact_density_change",
    "wt_polar_exposed_fraction",
    "mt_polar_exposed_fraction",
    "solvent_proxy_shift",
    "wt_top_seed_count",
    "mt_top_seed_count",
    "ifp_occupancy_shift_mean_abs",
    "ifp_occupancy_anchor_loss",
    "wt_ifp_occupancy_json",
    "mt_ifp_occupancy_json",
    "mechanism_labels_json",
    "mechanism_signature",
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


def configured_gpu_ids(stage5: dict[str, object]) -> list[int]:
    raw_value = os.environ.get("RESISTGPT_STAGE5_GPU_IDS")
    if raw_value in {None, ""}:
        raw_value = stage5.get("local_sampling_gpu_ids", [])
    if raw_value is None or raw_value == "":
        return []
    if isinstance(raw_value, str):
        tokens = [token.strip() for token in raw_value.split(",")]
    else:
        tokens = list(raw_value)
    gpu_ids: list[int] = []
    for token in tokens:
        if token in {None, ""}:
            continue
        gpu_ids.append(int(token))
    return gpu_ids


def run_ifp_job(job: dict[str, object]) -> dict[str, object]:
    return compute_ifp_effect_row(**job)


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage5 = dict(config["stage5"])
    cases_config = load_yaml(root / config["stage2"]["cases_frozen_config"])

    cases = selected_cases(cases_config, args.case_id)
    if not cases:
        raise SystemExit(f"No mutation-effect step case matched --case-id={args.case_id}")

    for case_entry in cases:
        case_id = str(case_entry["case_id"])
        case_stage5 = stage5_for_case(stage5, case_id)
        case_root = root / "outputs" / case_id / "stage5"
        docking_frame = read_csv_optional(case_root / "docking_scores.csv")
        pose_ensemble = read_csv_optional(case_root / "pose_ensemble.csv")
        anchor_labels = load_anchor_labels(root / "outputs" / case_id / "stage3_5" / "wt_anchor_residues.txt")

        jobs = [
            {
                "root": root,
                "case_id": case_id,
                "docking_row": row,
                "anchor_labels": anchor_labels,
                "stage5": case_stage5,
                "pose_ensemble": pose_ensemble,
                "gpu_id": None,
            }
            for row in docking_frame.to_dict(orient="records")
        ]
        effect_rows: list[dict[str, object]] = []
        gpu_ids = configured_gpu_ids(case_stage5)
        if gpu_ids:
            for index, job in enumerate(jobs):
                job["gpu_id"] = gpu_ids[index % len(gpu_ids)]
        worker_cap = len(gpu_ids) if gpu_ids else int(case_stage5.get("max_parallel_jobs", 4))
        max_workers = max(1, min(int(case_stage5.get("max_parallel_jobs", 4)), worker_cap, len(jobs)))
        if max_workers == 1:
            for job in jobs:
                effect_rows.append(run_ifp_job(job))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(run_ifp_job, job): job for job in jobs}
                for future in as_completed(future_map):
                    effect_rows.append(future.result())
        effect_frame = pd.DataFrame.from_records(effect_rows, columns=IFP_COLUMNS)
        write_table(effect_frame, case_root / "ifp_diff.csv")

        summary = {
            "case_id": case_id,
            "ifp_success_count": int(effect_frame["ifp_status"].eq("ok").sum()) if not effect_frame.empty else 0,
            "ifp_skipped_count": int(effect_frame["ifp_status"].eq("skipped").sum()) if not effect_frame.empty else 0,
            "mechanism_signature_counts": (
                effect_frame["mechanism_signature"].fillna("none").value_counts().to_dict() if not effect_frame.empty else {}
            ),
        }
        write_json(case_root / "ifp_summary.json", summary)


if __name__ == "__main__":
    main()
