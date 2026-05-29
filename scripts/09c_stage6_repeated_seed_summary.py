#!/usr/bin/env python3
"""Summarize counter-design step same-budget repeated-seed robust-vs-naive runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import ensure_dir, iso_now, json_dump, project_root


CASE_IDS = ("egfr_erlotinib", "hiv_rt_rilpivirine")
OBJECTIVES = ("naive", "robust")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20260517")
    parser.add_argument("--seeds", type=int, nargs="*", default=[101, 202, 303])
    parser.add_argument("--case-id", nargs="*", default=list(CASE_IDS))
    parser.add_argument("--output-dir", default="reports/paper_md_strict_20260517/data")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def _median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return math.nan if values.empty else float(values.median())


def _mean_bool(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    return float(frame[column].fillna(False).astype(bool).mean())


def _nunique_top(frame: pd.DataFrame, column: str, n: int) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame.head(int(n))[column].fillna("").replace("", pd.NA).dropna().nunique())


def _stage6_subdir(run_id: str, seed: int) -> str:
    return f"stage6_repeated_seed_{run_id}_seed{int(seed)}"


def _extract_qc_budget(qc_path: Path) -> dict[str, Any]:
    if not qc_path.exists():
        return {}
    try:
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        "max_rounds": qc.get("max_rounds"),
        "proposal_count": qc.get("proposal_count"),
        "beam_width": qc.get("beam_width"),
        "max_parallel_candidates": qc.get("max_parallel_candidates"),
        "target_parallel_workers": qc.get("target_parallel_workers"),
        "search_seed": qc.get("search_seed"),
        "search_rank_jitter": qc.get("search_rank_jitter"),
        "disable_llm": qc.get("disable_llm"),
        "objective_execution_order": "|".join(str(v) for v in list(qc.get("objective_execution_order") or [])),
    }


def _top20_summary(case_id: str, seed: int, stage6_subdir: str, root: Path) -> list[dict[str, Any]]:
    stage6_root = root / "outputs" / case_id / stage6_subdir
    leaderboard_path = stage6_root / "leaderboard.csv"
    if not leaderboard_path.exists():
        return []
    frame = pd.read_csv(leaderboard_path)
    if frame.empty or "objective_name" not in frame.columns:
        return []
    budget = _extract_qc_budget(stage6_root / "stage6_qc.json")
    rows: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        sub = frame[frame["objective_name"].astype(str).eq(objective)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("objective_reward", ascending=False).reset_index(drop=True)
        top20 = sub.head(20)
        top50 = sub.head(50)
        best = sub.iloc[0]
        rows.append(
            {
                "case_id": case_id,
                "seed": int(seed),
                "stage6_subdir": stage6_subdir,
                "objective": objective,
                **budget,
                "candidate_count": int(len(sub)),
                "unique_candidate_count": int(sub["candidate_id"].nunique()) if "candidate_id" in sub.columns else int(len(sub)),
                "valid_candidate_rate": _mean_bool(sub, "candidate_valid"),
                "chemical_valid_rate": _mean_bool(sub, "chemical_valid"),
                "wt_pass_rate": _mean_bool(sub, "wt_hard_constraint_pass"),
                "panel_passing_rate": _mean_bool(sub, "panel_passing"),
                "panel_coverage_pass_rate": _mean_bool(sub, "panel_coverage_pass"),
                "top20_valid_candidate_rate": _mean_bool(top20, "candidate_valid"),
                "top20_wt_constraint_pass_rate": _mean_bool(top20, "wt_hard_constraint_pass"),
                "top20_panel_passing_rate": _mean_bool(top20, "panel_passing"),
                "top20_robust_score_median": _median(top20, "robust_score"),
                "top20_robust_core_median": _median(top20, "robust_core"),
                "top20_robust_site_core_median": _median(top20, "robust_site_core"),
                "top20_combo_robust_core_median": _median(top20, "combo_robust_core"),
                "top20_combo_naive_mean_affinity_median": _median(top20, "combo_naive_mean_affinity"),
                "top20_dep_median": _median(top20, "dep"),
                "top20_hotspot_fraction_median": _median(top20, "hotspot_fraction"),
                "top20_keep_ifp_median": _median(top20, "keep_ifp"),
                "top20_keep_ifp_nonhotspot_median": _median(top20, "keep_ifp_nonhotspot"),
                "top20_effective_compensation_gain_median": _median(top20, "effective_compensation_gain"),
                "top20_new_nonhotspot_residue_count_median": _median(top20, "new_nonhotspot_residue_count"),
                "top50_scaffold_unique": _nunique_top(top50, "scaffold_smiles", 50),
                "best_candidate_id": str(best.get("candidate_id") or ""),
                "best_objective_reward": _safe_float(best.get("objective_reward")),
                "best_robust_score": _safe_float(best.get("robust_score")),
                "best_dep": _safe_float(best.get("dep")),
                "best_hotspot_fraction": _safe_float(best.get("hotspot_fraction")),
                "best_keep_ifp_nonhotspot": _safe_float(best.get("keep_ifp_nonhotspot")),
                "summary_generated_at": iso_now(),
            }
        )
    return rows


def _paired_rows(long_frame: pd.DataFrame) -> pd.DataFrame:
    if long_frame.empty:
        return pd.DataFrame()
    metric_cols = [
        column
        for column in long_frame.columns
        if column
        not in {
            "case_id",
            "seed",
            "stage6_subdir",
            "objective",
            "objective_execution_order",
            "best_candidate_id",
            "summary_generated_at",
        }
        and pd.api.types.is_numeric_dtype(long_frame[column])
    ]
    rows: list[dict[str, Any]] = []
    for (case_id, seed), group in long_frame.groupby(["case_id", "seed"], dropna=False):
        robust = group[group["objective"].eq("robust")]
        naive = group[group["objective"].eq("naive")]
        if robust.empty or naive.empty:
            continue
        r = robust.iloc[0]
        n = naive.iloc[0]
        row: dict[str, Any] = {
            "case_id": str(case_id),
            "seed": int(seed),
            "stage6_subdir": str(r.get("stage6_subdir") or n.get("stage6_subdir") or ""),
            "paired_complete": True,
        }
        for metric in metric_cols:
            rv = _safe_float(r.get(metric))
            nv = _safe_float(n.get(metric))
            row[f"robust_{metric}"] = rv
            row[f"naive_{metric}"] = nv
            row[f"delta_{metric}"] = rv - nv if math.isfinite(rv) and math.isfinite(nv) else math.nan
        if str(case_id) == "hiv_rt_rilpivirine":
            row["hiv_delta_top20_site_core"] = row.get("delta_top20_robust_site_core_median", math.nan)
            row["hiv_delta_top20_combo_core"] = row.get("delta_top20_combo_robust_core_median", math.nan)
            row["hiv_delta_combo_minus_site_sensitivity"] = (
                row["hiv_delta_top20_combo_core"] - row["hiv_delta_top20_site_core"]
                if math.isfinite(row.get("hiv_delta_top20_combo_core", math.nan))
                and math.isfinite(row.get("hiv_delta_top20_site_core", math.nan))
                else math.nan
            )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _aggregate_pairs(pair_frame: pd.DataFrame) -> pd.DataFrame:
    if pair_frame.empty:
        return pd.DataFrame()
    metric_cols = [
        column
        for column in pair_frame.columns
        if column not in {"case_id", "seed", "stage6_subdir", "paired_complete"}
        and pd.api.types.is_numeric_dtype(pair_frame[column])
    ]
    rows: list[dict[str, Any]] = []
    for case_id, group in pair_frame.groupby("case_id", dropna=False):
        row: dict[str, Any] = {
            "case_id": str(case_id),
            "seed_count": int(group["seed"].nunique()),
            "summary_generated_at": iso_now(),
        }
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                row[f"{metric}_mean"] = math.nan
                row[f"{metric}_median"] = math.nan
                row[f"{metric}_min"] = math.nan
                row[f"{metric}_max"] = math.nan
            else:
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_median"] = float(values.median())
                row[f"{metric}_min"] = float(values.min())
                row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def main() -> None:
    args = parse_args()
    root = project_root()
    output_dir = ensure_dir(root / str(args.output_dir))
    long_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for case_id in [str(case) for case in args.case_id]:
        for seed in [int(seed) for seed in args.seeds]:
            subdir = _stage6_subdir(str(args.run_id), seed)
            rows = _top20_summary(case_id, seed, subdir, root)
            if not rows:
                missing.append(f"{case_id}:{subdir}")
                continue
            long_rows.extend(rows)

    if args.require_complete and missing:
        raise SystemExit("Missing repeated-seed outputs: " + ", ".join(missing))

    long_frame = pd.DataFrame.from_records(long_rows)
    pair_frame = _paired_rows(long_frame)
    aggregate_frame = _aggregate_pairs(pair_frame)

    long_path = output_dir / "stage6_repeated_seed_objective_summary.csv"
    pair_path = output_dir / "stage6_repeated_seed_paired_summary.csv"
    aggregate_path = output_dir / "stage6_repeated_seed_paired_aggregate.csv"
    manifest_path = output_dir / "stage6_repeated_seed_summary_manifest.json"
    long_frame.to_csv(long_path, index=False)
    pair_frame.to_csv(pair_path, index=False)
    aggregate_frame.to_csv(aggregate_path, index=False)
    payload = {
        "generated_at": iso_now(),
        "run_id": str(args.run_id),
        "seeds": [int(seed) for seed in args.seeds],
        "case_ids": [str(case) for case in args.case_id],
        "missing": missing,
        "objective_summary": str(long_path.relative_to(root)),
        "paired_summary": str(pair_path.relative_to(root)),
        "paired_aggregate": str(aggregate_path.relative_to(root)),
        "paired_row_count": int(len(pair_frame)),
        "objective_row_count": int(len(long_frame)),
        "required_complete": bool(args.require_complete),
    }
    json_dump(manifest_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
