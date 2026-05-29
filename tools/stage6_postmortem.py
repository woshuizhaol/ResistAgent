#!/usr/bin/env python3
"""Postmortem utilities for counter-design step search diagnostics."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


HEALTH_COLUMNS = (
    "chemical_valid",
    "prefilter_pass",
    "wt_pass",
    "panel_coverage_pass",
    "panel_passing",
)


def _parse_json_payload(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text or text in {"nan", "NaN", "None"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    return series.fillna(default).astype(bool)


def ensure_stage6_health_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "chemical_valid" not in enriched.columns:
        enriched["chemical_valid"] = True
    if "prefilter_pass" not in enriched.columns:
        if "docking_skipped" in enriched.columns:
            enriched["prefilter_pass"] = ~_bool_series(enriched, "docking_skipped", default=False)
        else:
            enriched["prefilter_pass"] = True
    if "wt_pass" not in enriched.columns:
        enriched["wt_pass"] = _bool_series(enriched, "wt_hard_constraint_pass", default=False)
    if "panel_coverage_pass" not in enriched.columns:
        enriched["panel_coverage_pass"] = _bool_series(enriched, "coverage_pass", default=False)
    if "panel_passing" not in enriched.columns:
        enriched["panel_passing"] = _bool_series(enriched, "candidate_valid", default=False)
    if "panel_passing_rate" not in enriched.columns:
        enriched["panel_passing_rate"] = pd.NA
    if "valid_candidate_rate" not in enriched.columns:
        enriched["valid_candidate_rate"] = pd.NA
    return enriched


def stage6_executability_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "chemical_valid_rate": 0.0,
            "prefilter_pass_rate": 0.0,
            "wt_pass_rate": 0.0,
            "panel_coverage_pass_rate": 0.0,
            "panel_passing_rate": 0.0,
            "valid_candidate_rate": 0.0,
        }
    enriched = ensure_stage6_health_columns(frame)
    chemical_valid = _bool_series(enriched, "chemical_valid", default=True)
    prefilter_pass = _bool_series(enriched, "prefilter_pass", default=True)
    wt_pass = _bool_series(enriched, "wt_pass", default=False)
    panel_coverage_pass = _bool_series(enriched, "panel_coverage_pass", default=False)
    panel_passing = _bool_series(enriched, "panel_passing", default=False)
    panel_passing_rate = float(panel_passing.mean()) if len(panel_passing) else 0.0
    return {
        "chemical_valid_rate": float(chemical_valid.mean()) if len(chemical_valid) else 0.0,
        "prefilter_pass_rate": float(prefilter_pass.mean()) if len(prefilter_pass) else 0.0,
        "wt_pass_rate": float(wt_pass.mean()) if len(wt_pass) else 0.0,
        "panel_coverage_pass_rate": float(panel_coverage_pass.mean()) if len(panel_coverage_pass) else 0.0,
        "panel_passing_rate": panel_passing_rate,
        "valid_candidate_rate": panel_passing_rate,
    }


def constraint_funnel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["objective_name", "stage", "candidate_count", "pass_count", "overall_pass_rate", "conditional_pass_rate"]
        )
    enriched = ensure_stage6_health_columns(frame)
    rows: list[dict[str, Any]] = []
    for objective_name in ["all", *sorted(str(value) for value in enriched.get("objective_name", pd.Series(dtype=str)).dropna().unique())]:
        subset = enriched if objective_name == "all" else enriched[enriched["objective_name"].eq(objective_name)].copy()
        total = int(len(subset))
        if total == 0:
            continue
        masks = [
            ("chemical_valid", _bool_series(subset, "chemical_valid", default=True)),
            ("prefilter_pass", _bool_series(subset, "prefilter_pass", default=True)),
            ("wt_pass", _bool_series(subset, "wt_pass", default=False)),
            ("panel_coverage_pass", _bool_series(subset, "panel_coverage_pass", default=False)),
            ("panel_passing", _bool_series(subset, "panel_passing", default=False)),
        ]
        sequential_mask = pd.Series(True, index=subset.index, dtype=bool)
        previous_pass_count = total
        for stage_name, stage_mask in masks:
            sequential_mask = sequential_mask & stage_mask
            pass_count = int(sequential_mask.sum())
            rows.append(
                {
                    "objective_name": objective_name,
                    "stage": stage_name,
                    "candidate_count": total,
                    "pass_count": pass_count,
                    "overall_pass_rate": float(pass_count / total) if total else 0.0,
                    "conditional_pass_rate": float(pass_count / previous_pass_count) if previous_pass_count else 0.0,
                }
            )
            previous_pass_count = pass_count
    return pd.DataFrame.from_records(rows)


def invalid_reason_breakdown_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["reason_type", "reason", "count", "fraction_of_candidates", "fraction_of_invalid_candidates"]
        )
    enriched = ensure_stage6_health_columns(frame)
    total = int(len(enriched))
    panel_passing = _bool_series(enriched, "panel_passing", default=False)
    invalid = enriched.loc[~panel_passing].copy()
    invalid_total = int(len(invalid))
    primary_counter: Counter[str] = Counter()
    for _, row in invalid.iterrows():
        if not bool(row.get("chemical_valid", True)):
            primary_counter["chemical_invalid"] += 1
        elif not bool(row.get("prefilter_pass", False)):
            primary_counter["prefilter_failed"] += 1
        elif not bool(row.get("wt_pass", row.get("wt_hard_constraint_pass", False))):
            primary_counter["wt_failed"] += 1
        elif not bool(row.get("panel_coverage_pass", row.get("coverage_pass", False))):
            primary_counter["coverage_failed"] += 1
        else:
            primary_counter["panel_failed_other"] += 1
    rows = [
        {
            "reason_type": "primary",
            "reason": reason,
            "count": int(count),
            "fraction_of_candidates": float(count / total) if total else 0.0,
            "fraction_of_invalid_candidates": float(count / invalid_total) if invalid_total else 0.0,
        }
        for reason, count in primary_counter.items()
    ]
    flag_masks = {
        "keep_ifp_soft_failed": ~_bool_series(enriched, "keep_ifp_constraint_pass", default=True),
        "compensation_soft_failed": ~_bool_series(enriched, "compensation_constraint_pass", default=True),
        "high_uncertainty": _bool_series(enriched, "high_uncertainty", default=False),
        "hotspot_dependent": pd.to_numeric(enriched.get("dep", pd.Series(dtype=float)), errors="coerce").fillna(0.0).ge(0.7),
    }
    for reason, mask in flag_masks.items():
        count = int(mask.sum())
        rows.append(
            {
                "reason_type": "flag",
                "reason": reason,
                "count": count,
                "fraction_of_candidates": float(count / total) if total else 0.0,
                "fraction_of_invalid_candidates": float(count / invalid_total) if invalid_total else 0.0,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(["reason_type", "count", "reason"], ascending=[True, False, True])


def panel_failure_matrix_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "target_scores_json" not in frame.columns:
        return pd.DataFrame(
            columns=[
                "objective_name",
                "target_key",
                "effect_scope",
                "evaluation_count",
                "success_count",
                "uncertain_count",
                "failure_rate",
                "mean_score",
                "mean_keep_ifp",
                "partner_chain_sensitive_rate",
            ]
        )
    rows: list[dict[str, Any]] = []
    enriched = ensure_stage6_health_columns(frame)
    for _, candidate in enriched.iterrows():
        target_scores = _parse_json_payload(candidate.get("target_scores_json")) or {}
        if not isinstance(target_scores, dict):
            continue
        for target_key, payload in target_scores.items():
            if not isinstance(payload, dict):
                continue
            score_value = pd.to_numeric(pd.Series([payload.get("score")]), errors="coerce").iloc[0]
            keep_ifp_value = pd.to_numeric(pd.Series([payload.get("keep_ifp")]), errors="coerce").iloc[0]
            rows.append(
                {
                    "objective_name": str(candidate.get("objective_name") or ""),
                    "candidate_id": str(candidate.get("candidate_id") or ""),
                    "target_key": str(target_key),
                    "effect_scope": str(payload.get("effect_scope") or ""),
                    "docking_status": str(payload.get("docking_status") or ""),
                    "target_uncertain": bool(payload.get("target_uncertain", False)),
                    "score": score_value,
                    "keep_ifp": keep_ifp_value,
                    "is_partner_chain_sensitive": bool(payload.get("is_partner_chain_sensitive", False)),
                    "lost_anchor_count": int(len(payload.get("lost_anchor_labels") or [])),
                }
            )
    target_frame = pd.DataFrame.from_records(rows)
    if target_frame.empty:
        return target_frame
    grouped_rows: list[dict[str, Any]] = []
    for group_keys, group in target_frame.groupby(["objective_name", "target_key", "effect_scope"], dropna=False):
        objective_name, target_key, effect_scope = group_keys
        evaluation_count = int(len(group))
        success_mask = group["docking_status"].eq("ok")
        uncertain_mask = group["target_uncertain"].fillna(False).astype(bool)
        grouped_rows.append(
            {
                "objective_name": str(objective_name),
                "target_key": str(target_key),
                "effect_scope": str(effect_scope),
                "evaluation_count": evaluation_count,
                "success_count": int(success_mask.sum()),
                "uncertain_count": int(uncertain_mask.sum()),
                "failure_rate": float((~success_mask).mean()) if evaluation_count else 0.0,
                "mean_score": None if group["score"].dropna().empty else float(group["score"].dropna().mean()),
                "mean_keep_ifp": None if group["keep_ifp"].dropna().empty else float(group["keep_ifp"].dropna().mean()),
                "partner_chain_sensitive_rate": float(group["is_partner_chain_sensitive"].fillna(False).astype(bool).mean()),
                "mean_lost_anchor_count": float(group["lost_anchor_count"].mean()) if evaluation_count else 0.0,
            }
        )
    return pd.DataFrame.from_records(grouped_rows).sort_values(
        ["objective_name", "failure_rate", "target_key"],
        ascending=[True, False, True],
    )


def _last_action_family(action_sequence_json: Any) -> str:
    payload = _parse_json_payload(action_sequence_json) or []
    if not isinstance(payload, list) or not payload:
        return "LEAD"
    last = payload[-1]
    if not isinstance(last, dict):
        return "UNKNOWN"
    return str(last.get("edit_family") or last.get("action_label") or "UNKNOWN")


def action_family_yield_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "objective_name",
                "action_family",
                "candidate_count",
                "panel_passing_rate",
                "wt_pass_rate",
                "panel_coverage_pass_rate",
                "mean_objective_reward",
                "best_objective_reward",
                "mean_new_nonhotspot_contact_score",
            ]
        )
    enriched = ensure_stage6_health_columns(frame)
    enriched["action_family"] = enriched.get("action_sequence_json", pd.Series(dtype=object)).apply(_last_action_family)
    rows: list[dict[str, Any]] = []
    for group_keys, group in enriched.groupby(["objective_name", "action_family"], dropna=False):
        objective_name, action_family = group_keys
        metrics = stage6_executability_metrics(group)
        new_nonhotspot = pd.to_numeric(group.get("new_nonhotspot_contact_score", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "objective_name": str(objective_name),
                "action_family": str(action_family),
                "candidate_count": int(len(group)),
                "chemical_valid_rate": float(metrics["chemical_valid_rate"]),
                "prefilter_pass_rate": float(metrics["prefilter_pass_rate"]),
                "wt_pass_rate": float(metrics["wt_pass_rate"]),
                "panel_coverage_pass_rate": float(metrics["panel_coverage_pass_rate"]),
                "panel_passing_rate": float(metrics["panel_passing_rate"]),
                "mean_objective_reward": float(pd.to_numeric(group["objective_reward"], errors="coerce").dropna().mean()),
                "best_objective_reward": float(pd.to_numeric(group["objective_reward"], errors="coerce").dropna().max()),
                "mean_new_nonhotspot_contact_score": None
                if new_nonhotspot.dropna().empty
                else float(new_nonhotspot.dropna().mean()),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["objective_name", "panel_passing_rate", "mean_objective_reward", "action_family"],
        ascending=[True, False, False, True],
    )


def diversity_collapse_report_frame(frame: pd.DataFrame, top_ns: tuple[int, ...] = (10, 20, 50)) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "objective_name",
                "top_n",
                "candidate_count",
                "unique_scaffold_count",
                "unique_scaffold_fraction",
                "dominant_scaffold_share",
                "scaffold_hhi",
            ]
        )
    enriched = ensure_stage6_health_columns(frame)
    rows: list[dict[str, Any]] = []
    for objective_name, group in enriched.groupby("objective_name", dropna=False):
        ranked = group.sort_values("objective_reward", ascending=False).reset_index(drop=True)
        for top_n in top_ns:
            subset = ranked.head(int(top_n)).copy()
            if subset.empty:
                continue
            scaffolds = subset.get("scaffold_smiles", pd.Series(dtype=str)).fillna("NA").astype(str)
            counts = scaffolds.value_counts()
            shares = counts / max(1, len(subset))
            rows.append(
                {
                    "objective_name": str(objective_name),
                    "top_n": int(top_n),
                    "candidate_count": int(len(subset)),
                    "unique_scaffold_count": int(counts.size),
                    "unique_scaffold_fraction": float(counts.size / max(1, len(subset))),
                    "dominant_scaffold": str(counts.index[0]) if not counts.empty else "",
                    "dominant_scaffold_share": float(shares.iloc[0]) if not shares.empty else 0.0,
                    "scaffold_hhi": float((shares.pow(2)).sum()) if not shares.empty else 0.0,
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(["objective_name", "top_n"], ascending=[True, True])


def oracle_dependency_report_frame(frame: pd.DataFrame, calibrator_metadata: dict[str, Any] | None = None) -> pd.DataFrame:
    calibrator_metadata = dict(calibrator_metadata or {})
    rows: list[dict[str, Any]] = [
        {
            "row_type": "calibrator_metadata",
            "objective_name": "all",
            "candidate_count": int(len(frame)),
            "calibrator_available": bool(calibrator_metadata.get("available", False)),
            "training_row_count": int(calibrator_metadata.get("training_row_count", 0) or 0),
            "training_target_count": int(calibrator_metadata.get("training_target_count", 0) or 0),
            "dock_vs_mmgbsa_pearson_r": calibrator_metadata.get("dock_vs_mmgbsa_pearson_r"),
            "calibrator_reason": str(calibrator_metadata.get("reason") or ""),
            "calibrated_target_score_fraction": None,
            "raw_target_score_fraction": None,
            "mean_target_score": None,
            "mean_calibrated_score": None,
            "mean_raw_docking_score": None,
            "panel_passing_rate": None,
            "high_uncertainty_rate": None,
            "panel_coverage_mean": None,
            "target_success_count_mean": None,
            "objective_reward_vs_calibrated_score_pearson_r": None,
        }
    ]
    if frame.empty or "target_scores_json" not in frame.columns:
        return pd.DataFrame.from_records(rows)
    enriched = ensure_stage6_health_columns(frame)
    candidate_rows: list[dict[str, Any]] = []
    for _, candidate in enriched.iterrows():
        target_scores = _parse_json_payload(candidate.get("target_scores_json")) or {}
        if not isinstance(target_scores, dict):
            target_scores = {}
        successful_payloads = [
            payload
            for payload in target_scores.values()
            if isinstance(payload, dict) and str(payload.get("docking_status") or "") == "ok"
        ]
        score_values = [payload.get("score") for payload in successful_payloads if payload.get("score") is not None]
        calibrated_values = [
            payload.get("calibrated_score") for payload in successful_payloads if payload.get("calibrated_score") is not None
        ]
        raw_values = [
            payload.get("raw_docking_score") for payload in successful_payloads if payload.get("raw_docking_score") is not None
        ]
        candidate_rows.append(
            {
                "objective_name": str(candidate.get("objective_name") or ""),
                "objective_reward": pd.to_numeric(pd.Series([candidate.get("objective_reward")]), errors="coerce").iloc[0],
                "panel_passing": bool(candidate.get("panel_passing", candidate.get("candidate_valid", False))),
                "high_uncertainty": bool(candidate.get("high_uncertainty", False)),
                "panel_coverage": pd.to_numeric(pd.Series([candidate.get("panel_coverage")]), errors="coerce").iloc[0],
                "target_success_count": int(len(successful_payloads)),
                "calibrated_target_score_fraction": float(len(calibrated_values) / len(successful_payloads))
                if successful_payloads
                else 0.0,
                "raw_target_score_fraction": float(len(raw_values) / len(successful_payloads)) if successful_payloads else 0.0,
                "mean_target_score": None if not score_values else float(pd.to_numeric(pd.Series(score_values), errors="coerce").dropna().mean()),
                "mean_calibrated_score": None
                if not calibrated_values
                else float(pd.to_numeric(pd.Series(calibrated_values), errors="coerce").dropna().mean()),
                "mean_raw_docking_score": None
                if not raw_values
                else float(pd.to_numeric(pd.Series(raw_values), errors="coerce").dropna().mean()),
            }
        )
    candidate_frame = pd.DataFrame.from_records(candidate_rows)
    if candidate_frame.empty:
        return pd.DataFrame.from_records(rows)
    for objective_name in ["all", *sorted(str(value) for value in candidate_frame["objective_name"].dropna().unique())]:
        subset = candidate_frame if objective_name == "all" else candidate_frame[candidate_frame["objective_name"].eq(objective_name)].copy()
        if subset.empty:
            continue
        corr = None
        calibrated = pd.to_numeric(subset["mean_calibrated_score"], errors="coerce")
        rewards = pd.to_numeric(subset["objective_reward"], errors="coerce")
        valid_corr = calibrated.notna() & rewards.notna()
        if int(valid_corr.sum()) >= 3 and calibrated[valid_corr].nunique() > 1 and rewards[valid_corr].nunique() > 1:
            corr = float(calibrated[valid_corr].corr(rewards[valid_corr], method="pearson"))
        rows.append(
            {
                "row_type": "objective_summary",
                "objective_name": objective_name,
                "candidate_count": int(len(subset)),
                "calibrator_available": bool(calibrator_metadata.get("available", False)),
                "training_row_count": int(calibrator_metadata.get("training_row_count", 0) or 0),
                "training_target_count": int(calibrator_metadata.get("training_target_count", 0) or 0),
                "dock_vs_mmgbsa_pearson_r": calibrator_metadata.get("dock_vs_mmgbsa_pearson_r"),
                "calibrator_reason": str(calibrator_metadata.get("reason") or ""),
                "calibrated_target_score_fraction": float(pd.to_numeric(subset["calibrated_target_score_fraction"], errors="coerce").mean()),
                "raw_target_score_fraction": float(pd.to_numeric(subset["raw_target_score_fraction"], errors="coerce").mean()),
                "mean_target_score": None
                if pd.to_numeric(subset["mean_target_score"], errors="coerce").dropna().empty
                else float(pd.to_numeric(subset["mean_target_score"], errors="coerce").dropna().mean()),
                "mean_calibrated_score": None
                if calibrated.dropna().empty
                else float(calibrated.dropna().mean()),
                "mean_raw_docking_score": None
                if pd.to_numeric(subset["mean_raw_docking_score"], errors="coerce").dropna().empty
                else float(pd.to_numeric(subset["mean_raw_docking_score"], errors="coerce").dropna().mean()),
                "panel_passing_rate": float(subset["panel_passing"].fillna(False).astype(bool).mean()),
                "high_uncertainty_rate": float(subset["high_uncertainty"].fillna(False).astype(bool).mean()),
                "panel_coverage_mean": None
                if pd.to_numeric(subset["panel_coverage"], errors="coerce").dropna().empty
                else float(pd.to_numeric(subset["panel_coverage"], errors="coerce").dropna().mean()),
                "target_success_count_mean": float(pd.to_numeric(subset["target_success_count"], errors="coerce").mean()),
                "objective_reward_vs_calibrated_score_pearson_r": corr,
            }
        )
    return pd.DataFrame.from_records(rows)


def llm_failure_context_audit_frame(stage6_root: Path) -> pd.DataFrame:
    llm_root = stage6_root / "llm"
    if not llm_root.exists():
        return pd.DataFrame(
            columns=[
                "objective_name",
                "round_index",
                "beam_size",
                "worst_target_count",
                "worst_site_target_count",
                "worst_combo_target_count",
                "partner_chain_sensitive_worst_target_count",
                "uncertain_worst_target_count",
                "anchor_loss_worst_target_count",
                "steric_clash_worst_target_count",
                "electrostatic_shift_worst_target_count",
                "coverage_fail_count",
                "wt_fail_count",
                "keep_ifp_fail_count",
                "compensation_fail_count",
                "case_specific_hint_count",
                "partner_chain_residue_count",
            ]
        )
    rows: list[dict[str, Any]] = []
    for objective_dir in sorted(path for path in llm_root.iterdir() if path.is_dir()):
        objective_name = str(objective_dir.name)
        for input_path in sorted(objective_dir.glob("round_*_input.json")):
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            round_index = int(re.search(r"(\d+)", input_path.stem).group(1)) if re.search(r"(\d+)", input_path.stem) else 0
            failure_context = dict(payload.get("failure_context") or {})
            pocket_context = dict(payload.get("pocket_context") or {})
            worst_targets = [dict(item) for item in list(failure_context.get("worst_targets") or payload.get("worst_targets") or []) if isinstance(item, dict)]
            mechanism_counter: Counter[str] = Counter()
            for item in worst_targets:
                for label in list(item.get("mechanism_labels") or []):
                    mechanism_counter[str(label)] += 1
            constraint_breakdown = dict(payload.get("constraint_breakdown") or {})
            rows.append(
                {
                    "objective_name": objective_name,
                    "round_index": round_index,
                    "beam_size": int(dict(payload.get("search_state") or {}).get("beam_size") or 0),
                    "worst_target_count": int(len(worst_targets)),
                    "worst_site_target_count": int(sum(str(item.get("effect_scope") or "") == "site" for item in worst_targets)),
                    "worst_combo_target_count": int(sum(str(item.get("effect_scope") or "") == "combo" for item in worst_targets)),
                    "partner_chain_sensitive_worst_target_count": int(
                        sum(bool(item.get("is_partner_chain_sensitive")) for item in worst_targets)
                    ),
                    "uncertain_worst_target_count": int(sum(bool(item.get("target_uncertain")) for item in worst_targets)),
                    "anchor_loss_worst_target_count": int(mechanism_counter.get("anchor_loss", 0)),
                    "steric_clash_worst_target_count": int(mechanism_counter.get("steric_clash", 0)),
                    "electrostatic_shift_worst_target_count": int(mechanism_counter.get("electrostatic_shift", 0)),
                    "coverage_fail_count": int(constraint_breakdown.get("coverage_fail_count") or 0),
                    "wt_fail_count": int(constraint_breakdown.get("wt_fail_count") or 0),
                    "keep_ifp_fail_count": int(constraint_breakdown.get("keep_ifp_fail_count") or 0),
                    "compensation_fail_count": int(constraint_breakdown.get("compensation_fail_count") or 0),
                    "case_specific_hint_count": int(len(list(pocket_context.get("case_specific_action_hints") or []))),
                    "partner_chain_residue_count": int(len(list(pocket_context.get("partner_chain_residues") or []))),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "objective_name",
                "round_index",
                "beam_size",
                "worst_target_count",
                "worst_site_target_count",
                "worst_combo_target_count",
                "partner_chain_sensitive_worst_target_count",
                "uncertain_worst_target_count",
                "anchor_loss_worst_target_count",
                "steric_clash_worst_target_count",
                "electrostatic_shift_worst_target_count",
                "coverage_fail_count",
                "wt_fail_count",
                "keep_ifp_fail_count",
                "compensation_fail_count",
                "case_specific_hint_count",
                "partner_chain_residue_count",
            ]
        )
    return pd.DataFrame.from_records(rows).sort_values(["objective_name", "round_index"], ascending=[True, True])


def load_calibrator_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
