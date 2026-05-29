#!/usr/bin/env python3
"""mutation-effect step calibration, mechanism aggregation, and agent interpretation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jsonschema import validate as jsonschema_validate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.mutation_effect_agent import MutationEffectAgent, MutationEffectAgentConfig
from agents.orchestrator import DecisionRecord, StageOrchestrator
from tools.runtime import detect_git_commit, ensure_dir, iso_now, json_dump, load_yaml, project_root
from tools.stage5_physics import multi_score_consensus
from tools.stage5_utils import (
    calibration_frame,
    calibration_metrics,
    hashed_inputs,
    mechanism_cluster_frame,
    native_value,
    parse_json_list,
    relative_path,
    stage5_software_versions,
    write_json,
    write_table,
    load_empirical_ddg_lookup,
    assign_epistasis_flag,
    ensure_stage5_scoring_payload,
    stage5_for_case,
)


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


def load_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def selected_cases(cases_config: dict[str, object], case_id: str | None) -> list[dict[str, object]]:
    cases = list(cases_config.get("set_d", []))
    if case_id is None:
        return cases
    return [case for case in cases if str(case.get("case_id")) == str(case_id)]


def mechanism_label_counts(frame: pd.DataFrame) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for labels in frame.get("mechanism_labels_json", pd.Series(dtype=str)).tolist():
        for label in parse_json_list(labels):
            counter[str(label)] += 1
    return dict(sorted(counter.items()))


def configured_gpu_ids(stage5: dict[str, Any]) -> list[int]:
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


def configured_max_workers(stage5: dict[str, Any], *, scoring_job_count: int, gpu_ids: list[int]) -> int:
    default_cap = int(stage5.get("mmgbsa_max_parallel_jobs", 4))
    raw_value = os.environ.get("RESISTGPT_STAGE5_MAX_WORKERS")
    if raw_value in {None, ""}:
        worker_cap = len(gpu_ids) if gpu_ids else default_cap
    else:
        worker_cap = int(raw_value)
    return max(1, min(default_cap, int(scoring_job_count), int(worker_cap)))


def scoring_result_columns() -> list[str]:
    return [
        "scoring_status",
        "scoring_error",
        "wt_mmgbsa_binding_kcal_mol",
        "mt_mmgbsa_binding_kcal_mol",
        "delta_mmgbsa_binding_kcal_mol",
        "wt_gnina_affinity_kcal_mol",
        "mt_gnina_affinity_kcal_mol",
        "delta_gnina_affinity_kcal_mol",
        "available_score_count",
        "nonzero_direction_count",
        "consensus_fraction",
        "consensus_direction",
        "high_uncertainty",
    ]


def drop_stale_scoring_columns(frame: pd.DataFrame) -> pd.DataFrame:
    base_columns = set(scoring_result_columns())
    stale_columns = [
        column
        for column in frame.columns
        if column in base_columns or (column.endswith(("_x", "_y")) and column[:-2] in base_columns)
    ]
    if not stale_columns:
        return frame
    return frame.drop(columns=stale_columns)


def run_scoring_job(job: dict[str, Any]) -> dict[str, Any]:
    identifiers = {
        "case_id": str(job["case_id"]),
        "effect_scope": str(job["effect_scope"]),
        "target_key": str(job["target_key"]),
    }
    try:
        payload = ensure_stage5_scoring_payload(
            root=Path(str(job["root"])),
            docking_row=dict(job["docking_row"]),
            stage5=dict(job["stage5"]),
            gpu_id=job.get("gpu_id"),
        )
        return {
            **identifiers,
            "scoring_status": "ok",
            "scoring_error": None,
            **payload,
        }
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        if "MMPBSA.py" in error_text or "DELTA TOTAL" in error_text:
            consensus = multi_score_consensus(
                {
                    "vina": native_value(dict(job["docking_row"]).get("delta_dock_kcal_mol")),
                    "mmgbsa": None,
                    "gnina": None,
                },
                neutral_threshold=float(dict(job["stage5"]).get("multi_score_neutral_threshold_kcal_mol", 0.25)),
                consensus_threshold=float(dict(job["stage5"]).get("multi_score_consensus_threshold", 0.70)),
            )
            consensus["high_uncertainty"] = True
            return {
                **identifiers,
                "scoring_status": "ok",
                "scoring_error": error_text,
                "wt_mmgbsa_binding_kcal_mol": None,
                "mt_mmgbsa_binding_kcal_mol": None,
                "delta_mmgbsa_binding_kcal_mol": None,
                "wt_gnina_affinity_kcal_mol": None,
                "mt_gnina_affinity_kcal_mol": None,
                "delta_gnina_affinity_kcal_mol": None,
                **consensus,
            }
        return {
            **identifiers,
            "scoring_status": "failed",
            "scoring_error": error_text,
            "wt_mmgbsa_binding_kcal_mol": None,
            "mt_mmgbsa_binding_kcal_mol": None,
            "delta_mmgbsa_binding_kcal_mol": None,
            "wt_gnina_affinity_kcal_mol": None,
            "mt_gnina_affinity_kcal_mol": None,
            "delta_gnina_affinity_kcal_mol": None,
            "available_score_count": 0,
            "nonzero_direction_count": 0,
            "consensus_fraction": None,
            "consensus_direction": "neutral",
            "high_uncertainty": True,
        }


def merge_agent_site_effects(frame: pd.DataFrame, payload: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = {str(row["mutation_key"]): row for row in payload.get("site_effects", [])}
    rows: list[dict[str, Any]] = []
    for record in frame.sort_values("stage4_rank").to_dict(orient="records"):
        mutation_key = str(record["target_key"])
        agent_row = lookup.get(mutation_key, {})
        rows.append(
            {
                "mutation_key": mutation_key,
                "effect_status": str(record.get("ifp_status") or "skipped"),
                "impact_evidence_tier": str(record.get("impact_evidence_tier") or ""),
                "sample_source": str(record.get("sample_source") or ""),
                "representative_sample_id": str(record.get("representative_sample_id") or ""),
                "risk_score": native_value(record.get("risk_score")),
                "stage4_rank": native_value(record.get("stage4_rank")),
                "delta_dock_kcal_mol": native_value(record.get("delta_dock_kcal_mol")),
                "ifp_jaccard_loss": native_value(record.get("ifp_jaccard_loss")),
                "anchor_loss_fraction": native_value(record.get("anchor_loss_fraction")),
                "local_rmsd_a": native_value(record.get("stage4_local_rmsd_a")),
                "pocket_volume_change_fraction": native_value(record.get("pocket_volume_change_fraction")),
                "solvent_proxy_shift": native_value(record.get("solvent_proxy_shift")),
                "delta_mmgbsa_binding_kcal_mol": native_value(record.get("delta_mmgbsa_binding_kcal_mol")),
                "delta_gnina_affinity_kcal_mol": native_value(record.get("delta_gnina_affinity_kcal_mol")),
                "consensus_fraction": native_value(record.get("consensus_fraction")),
                "consensus_direction": str(record.get("consensus_direction") or "neutral"),
                "high_uncertainty": bool(record.get("high_uncertainty", True)),
                "ifp_occupancy_shift_mean_abs": native_value(record.get("ifp_occupancy_shift_mean_abs")),
                "ifp_occupancy_anchor_loss": native_value(record.get("ifp_occupancy_anchor_loss")),
                "mechanism_labels": parse_json_list(record.get("mechanism_labels_json")),
                "lost_anchor_labels": parse_json_list(record.get("lost_anchor_labels_json")),
                "reasoning": agent_row.get("reasoning"),
                "supporting_signals": agent_row.get("supporting_signals", []),
                "confidence": agent_row.get("confidence"),
                "caveat": agent_row.get("caveat"),
            }
        )
    return rows


def merge_agent_combo_effects(frame: pd.DataFrame, payload: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = {str(row["combination_key"]): row for row in payload.get("combo_effects", [])}
    rows: list[dict[str, Any]] = []
    for record in frame.sort_values("stage4_rank").to_dict(orient="records"):
        combo_key = str(record["target_key"])
        agent_row = lookup.get(combo_key, {})
        rows.append(
            {
                "combination_key": combo_key,
                "effect_status": str(record.get("ifp_status") or "skipped"),
                "impact_evidence_tier": str(record.get("impact_evidence_tier") or ""),
                "sample_source": str(record.get("sample_source") or ""),
                "representative_sample_id": str(record.get("representative_sample_id") or ""),
                "used_synthetic_combo_model": bool(record.get("used_synthetic_combo_model", False)),
                "risk_score": native_value(record.get("risk_score")),
                "stage4_rank": native_value(record.get("stage4_rank")),
                "delta_dock_kcal_mol": native_value(record.get("delta_dock_kcal_mol")),
                "ifp_jaccard_loss": native_value(record.get("ifp_jaccard_loss")),
                "anchor_loss_fraction": native_value(record.get("anchor_loss_fraction")),
                "local_rmsd_a": native_value(record.get("stage4_local_rmsd_a")),
                "pocket_volume_change_fraction": native_value(record.get("pocket_volume_change_fraction")),
                "solvent_proxy_shift": native_value(record.get("solvent_proxy_shift")),
                "delta_mmgbsa_binding_kcal_mol": native_value(record.get("delta_mmgbsa_binding_kcal_mol")),
                "delta_gnina_affinity_kcal_mol": native_value(record.get("delta_gnina_affinity_kcal_mol")),
                "consensus_fraction": native_value(record.get("consensus_fraction")),
                "consensus_direction": str(record.get("consensus_direction") or "neutral"),
                "high_uncertainty": bool(record.get("high_uncertainty", True)),
                "ifp_occupancy_shift_mean_abs": native_value(record.get("ifp_occupancy_shift_mean_abs")),
                "ifp_occupancy_anchor_loss": native_value(record.get("ifp_occupancy_anchor_loss")),
                "mechanism_labels": parse_json_list(record.get("mechanism_labels_json")),
                "lost_anchor_labels": parse_json_list(record.get("lost_anchor_labels_json")),
                "epistasis_flag": str(record.get("epistasis_flag") or "unresolved"),
                "reasoning": agent_row.get("reasoning"),
                "supporting_signals": agent_row.get("supporting_signals", []),
                "confidence": agent_row.get("confidence"),
                "caveat": agent_row.get("caveat"),
            }
        )
    return rows


def build_case_story_lines(
    *,
    case_entry: dict[str, Any],
    qc_payload: dict[str, Any],
    payload: dict[str, Any],
    site_effects: list[dict[str, Any]],
    combo_effects: list[dict[str, Any]],
) -> list[str]:
    lines = [
        f"# {case_entry['case_id']} mutation-effect step Mutation Effect",
        "",
        f"- 目标：`{case_entry['target_name']}`",
        f"- 药物：`{case_entry['drug_name']}`",
        f"- 选中 site 目标：`{qc_payload['selected_site_targets']}`",
        f"- 选中 combo 目标：`{qc_payload['selected_combo_targets']}`",
        f"- docking 成功数：`{qc_payload['docking_success_count']}`",
        f"- docking 失败数：`{qc_payload['docking_failure_count']}`",
        f"- IFP 成功数：`{qc_payload['ifp_success_count']}`",
        f"- calibration 样本数：`{qc_payload['calibration_sample_count']}`",
        f"- 多打分高不确定条目：`{qc_payload.get('high_uncertainty_count', 0)}`",
    ]
    if qc_payload.get("mutation_effect_model"):
        lines.append(
            f"- GLM 审计：`model={qc_payload['mutation_effect_model']}`，`tokens={qc_payload['mutation_effect_tokens']}`，"
            f" `latency={qc_payload['mutation_effect_latency_seconds']:.2f}s`，"
            f" `prompt_hash={qc_payload['mutation_effect_prompt_hash'][:12]}`，"
            f" `thinking={qc_payload['mutation_effect_thinking']['type']}`"
        )
    lines.extend(
        [
            "",
            "## Executive Summary",
            payload["executive_summary"]["overall_effect_pattern"],
            "",
            f"- Calibration：{payload['executive_summary']['calibration_note']}",
            f"- Principal uncertainty：{payload['executive_summary']['principal_uncertainty']}",
            "",
            "## Site Effects",
        ]
    )
    if not site_effects:
        lines.append("无 site effect 条目。")
    for row in site_effects:
        if row["effect_status"] != "ok":
            lines.append(f"- `{row['mutation_key']}`: `{row['effect_status']}`，原因：{row.get('caveat') or row.get('reasoning') or 'not available'}")
            continue
        lines.append(
            f"- `{row['mutation_key']}`: 标签=`{'|'.join(row['mechanism_labels'])}`；"
            f" delta_dock=`{row['delta_dock_kcal_mol']}`；ifp_loss=`{row['ifp_jaccard_loss']}`；"
            f" anchor_loss=`{row['anchor_loss_fraction']}`；mmgbsa=`{row.get('delta_mmgbsa_binding_kcal_mol')}`；"
            f" consensus=`{row.get('consensus_fraction')}`。{row.get('reasoning') or ''}"
        )
    lines.extend(["", "## Combo Effects"])
    if not combo_effects:
        lines.append("无 combo effect 条目。")
    for row in combo_effects:
        if row["effect_status"] != "ok":
            lines.append(f"- `{row['combination_key']}`: `{row['effect_status']}`，原因：{row.get('caveat') or row.get('reasoning') or 'not available'}")
            continue
        lines.append(
            f"- `{row['combination_key']}`: 标签=`{'|'.join(row['mechanism_labels'])}`；"
            f" epistasis=`{row['epistasis_flag']}`；delta_dock=`{row['delta_dock_kcal_mol']}`；"
            f" ifp_loss=`{row['ifp_jaccard_loss']}`；mmgbsa=`{row.get('delta_mmgbsa_binding_kcal_mol')}`；"
            f" consensus=`{row.get('consensus_fraction')}`。{row.get('reasoning') or ''}"
        )
    lines.extend(["", "## Global Caveats"])
    caveats = payload.get("global_caveats", [])
    if not caveats:
        lines.append("- 无额外 caveat。")
    else:
        lines.extend([f"- {line}" for line in caveats])
    return lines


def write_story(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def plot_calibration(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    if frame.empty or len(frame) < 2:
        ax.text(0.5, 0.5, "Insufficient calibration data", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.scatter(frame["delta_dock_kcal_mol"], frame["delta_mmgbsa_binding_kcal_mol"], alpha=0.8)
        x = frame["delta_dock_kcal_mol"]
        y = frame["delta_mmgbsa_binding_kcal_mol"]
        if len(frame) >= 2 and x.nunique() > 1:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = pd.Series(sorted(x.unique()), dtype=float)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="black", linewidth=1.2)
        ax.set_xlabel("Docking delta (kcal/mol)")
        ax.set_ylabel("MM/GBSA delta (kcal/mol)")
        ax.set_title("mutation-effect step Docking vs MM/GBSA")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_case_summary_artifacts(*, case_root: Path, effect_frame: pd.DataFrame, calibration_rows: pd.DataFrame) -> None:
    cluster_frame = mechanism_cluster_frame(effect_frame) if not effect_frame.empty else mechanism_cluster_frame(pd.DataFrame())
    write_table(cluster_frame, case_root / "mechanism_clusters.csv")
    write_table(calibration_rows, case_root / "scoring_calibration.csv")
    plot_calibration(calibration_rows, case_root / "scoring_calibration_plot.png")


def empty_payload(calibration_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "executive_summary": {
            "overall_effect_pattern": "No structure-backed mutation-effect step target passed the deterministic gate for interpretation.",
            "calibration_note": (
                f"Calibration sample count={int(calibration_summary.get('calibration_sample_count', 0))}; "
                "MM/GBSA alignment was not interpretable for this case."
            ),
            "principal_uncertainty": "The current case lacks enough mutation-effect step-ready structural evidence for a reliable mechanism narrative.",
        },
        "site_effects": [],
        "combo_effects": [],
        "global_caveats": [
            "mutation-effect step only interprets structure-backed WT/MT pairs; unresolved targets remain in skipped state.",
        ],
    }


def run_mutation_effect_agent(
    *,
    effect_agent: MutationEffectAgent,
    case_entry: dict[str, Any],
    site_effect_frame: pd.DataFrame,
    combo_effect_frame: pd.DataFrame,
    qc_payload: dict[str, Any],
    calibration_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], DecisionRecord | None, str, str | None]:
    site_ok = site_effect_frame[site_effect_frame["ifp_status"].eq("ok")].sort_values("stage4_rank")
    combo_ok = combo_effect_frame[combo_effect_frame["ifp_status"].eq("ok")].sort_values("stage4_rank")
    if site_ok.empty and combo_ok.empty:
        agent_input = effect_agent.build_prompt_input(
            case_entry=case_entry,
            site_effects=site_ok,
            combo_effects=combo_ok,
            qc_payload=qc_payload,
            calibration_summary=calibration_summary,
        )
        return agent_input, empty_payload(calibration_summary), None, "empty_payload", None
    try:
        agent_input, llm_payload, llm_record = effect_agent.run(
            case_entry=case_entry,
            site_effects=site_ok,
            combo_effects=combo_ok,
            qc_payload=qc_payload,
            calibration_summary=calibration_summary,
        )
        return agent_input, llm_payload, llm_record, "ok", None
    except Exception as exc:
        agent_input = effect_agent.build_prompt_input(
            case_entry=case_entry,
            site_effects=site_ok,
            combo_effects=combo_ok,
            qc_payload=qc_payload,
            calibration_summary=calibration_summary,
        )
        llm_payload = empty_payload(calibration_summary)
        llm_payload["global_caveats"].append(f"MutationEffectAgent unavailable: {exc}")
        return agent_input, llm_payload, None, "failed", str(exc)


def load_env_snapshot(existing_manifest: dict[str, Any]) -> dict[str, Any]:
    snapshot = existing_manifest.get("env_snapshot")
    if isinstance(snapshot, dict) and "kind" in snapshot and "content" in snapshot:
        return snapshot
    return {
        "kind": "inline",
        "content": "\n".join(
            [
                f"CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV', '')}",
                f"GLM_MODEL={os.environ.get('GLM_MODEL', '')}",
                f"GLM_BASE_URL={os.environ.get('GLM_BASE_URL', '')}",
            ]
        ),
    }


def write_case_state(
    *,
    root: Path,
    case_entry: dict[str, Any],
    args_config: str,
    stage1: dict[str, Any],
    stage5: dict[str, Any],
    site_effect_frame: pd.DataFrame,
    combo_effect_frame: pd.DataFrame,
    qc_payload: dict[str, Any],
    llm_payload: dict[str, Any],
    llm_record: Any,
    llm_input_path: Path,
    llm_payload_path: Path,
    story_path: Path,
    software_versions: dict[str, Any],
    commands: list[str],
) -> None:
    case_id = str(case_entry["case_id"])
    state_path = root / f"outputs/{case_id}/state.json"
    state = load_json_or_empty(state_path)
    inputs = state.get("inputs") if isinstance(state.get("inputs"), dict) else {}
    artifacts = state.get("artifacts") if isinstance(state.get("artifacts"), dict) else {}
    qc_block = state.get("qc") if isinstance(state.get("qc"), dict) else {}
    llm_decisions = state.get("llm_decisions") if isinstance(state.get("llm_decisions"), list) else []

    case_stage5_root = root / "outputs" / case_id / "stage5"
    inputs.update(
        {
            "config": args_config,
            "master_table": stage1["master_table"],
            "cases": load_yaml(root / args_config)["stage2"]["cases_frozen_config"],
        }
    )
    artifacts["stage5"] = {
        "docking_scores_csv": relative_path(case_stage5_root / "docking_scores.csv", root),
        "ifp_diff_csv": relative_path(case_stage5_root / "ifp_diff.csv", root),
        "mutation_effects_json": relative_path(case_stage5_root / "mutation_effects.json", root),
        "combination_effects_json": relative_path(case_stage5_root / "combination_effects.json", root),
        "mechanism_clusters_csv": relative_path(case_stage5_root / "mechanism_clusters.csv", root),
        "scoring_calibration_csv": relative_path(case_stage5_root / "scoring_calibration.csv", root),
        "scoring_calibration_plot_png": relative_path(case_stage5_root / "scoring_calibration_plot.png", root),
        "effect_story_input_json": relative_path(llm_input_path, root),
        "effect_story_payload_json": relative_path(llm_payload_path, root),
        "effect_story_md": relative_path(story_path, root),
        "stage5_qc_json": relative_path(case_stage5_root / "stage5_qc.json", root),
    }
    qc_block["stage5"] = qc_payload
    state.update(
        {
            "project_id": case_id,
            "stage": "stage5",
            "inputs": inputs,
            "artifacts": artifacts,
            "qc": qc_block,
            "software_versions": software_versions,
            "seeds": {
                "python": 42,
                "vina_seeds": list(stage5["vina_seeds"]),
            },
            "commands": commands,
            "llm_decisions": llm_decisions,
        }
    )
    json_dump(state_path, state)
    if llm_record is None:
        return
    orchestrator = StageOrchestrator(state_path=state_path)
    orchestrator.append_decision(
        DecisionRecord(
            agent_name="MutationEffectAgent",
            decision_type="mechanism_labeling",
            input_artifacts=[
                relative_path(case_stage5_root / "docking_scores.csv", root),
                relative_path(case_stage5_root / "ifp_diff.csv", root),
                relative_path(case_stage5_root / "stage5_qc.json", root),
                relative_path(llm_input_path, root),
            ],
            tool_calls=[
                {
                    "tool_name": "GLMClient.chat_json",
                    "model": llm_record.model,
                    "base_url": os.environ.get("GLM_BASE_URL"),
                    "temperature": float(stage5.get("agent_temperature", 0.2)),
                    "max_tokens": int(stage5["agent_max_tokens"]),
                    "thinking": {"type": str(stage5["agent_thinking"])},
                    "prompt_hash": llm_record.prompt_hash,
                    "tokens": llm_record.tokens,
                    "latency_seconds": llm_record.latency_seconds,
                    "retry_count": llm_record.retry_count,
                }
            ],
            decision_rationale=str(llm_payload["executive_summary"]["overall_effect_pattern"]),
            output_artifacts=[
                relative_path(case_stage5_root / "mutation_effects.json", root),
                relative_path(case_stage5_root / "combination_effects.json", root),
                relative_path(story_path, root),
            ],
        )
    )


def write_case_manifest(
    *,
    root: Path,
    case_entry: dict[str, Any],
    args_config: str,
    stage1: dict[str, Any],
    stage5: dict[str, Any],
    started_at: str,
    software_versions: dict[str, Any],
    commands: list[str],
    llm_record: Any,
    llm_input_path: Path,
    llm_payload_path: Path,
    story_path: Path,
) -> None:
    case_id = str(case_entry["case_id"])
    manifest_path = root / f"outputs/{case_id}/run_manifest.json"
    manifest = load_json_or_empty(manifest_path)
    stage_runs = manifest.get("stage_runs") if isinstance(manifest.get("stage_runs"), dict) else {}
    case_stage5_root = root / "outputs" / case_id / "stage5"
    input_paths = [
        root / args_config,
        root / stage1["mdrdb_main"],
        case_stage5_root / "docking_scores.csv",
        case_stage5_root / "ifp_diff.csv",
        llm_input_path,
    ]
    input_hashes = manifest.get("input_hashes") if isinstance(manifest.get("input_hashes"), dict) else {}
    input_hashes.update(hashed_inputs(input_paths, root))
    stage_run = {
        "started_at": started_at,
        "finished_at": iso_now(),
        "outputs": [
            relative_path(case_stage5_root / "mutation_effects.json", root),
            relative_path(case_stage5_root / "combination_effects.json", root),
            relative_path(case_stage5_root / "mechanism_clusters.csv", root),
            relative_path(case_stage5_root / "scoring_calibration.csv", root),
            relative_path(case_stage5_root / "scoring_calibration_plot.png", root),
            relative_path(case_stage5_root / "stage5_qc.json", root),
            relative_path(llm_input_path, root),
            relative_path(llm_payload_path, root),
            relative_path(story_path, root),
        ],
    }
    if llm_record is not None:
        stage_run["llm"] = {
            "provider": "zhipu_glm",
            "model": llm_record.model,
            "base_url": os.environ.get("GLM_BASE_URL"),
            "prompt_hash": llm_record.prompt_hash,
            "temperature": float(stage5.get("agent_temperature", 0.2)),
            "max_tokens": int(stage5["agent_max_tokens"]),
            "thinking": {"type": str(stage5["agent_thinking"])},
            "tokens": llm_record.tokens,
            "latency_seconds": llm_record.latency_seconds,
            "retry_count": llm_record.retry_count,
        }
    stage_runs["stage5"] = stage_run
    git_commit, git_status = detect_git_commit(root)
    manifest.update(
        {
            "project_id": case_id,
            "stage": "stage5",
            "git_commit": git_commit,
            "git_status": git_status,
            "software_versions": software_versions,
            "env_snapshot": load_env_snapshot(manifest),
            "random_seeds": {
                "python": 42,
                "vina_seeds": list(stage5["vina_seeds"]),
            },
            "input_hashes": input_hashes,
            "commands": commands,
            "started_at": manifest.get("started_at", started_at),
            "finished_at": iso_now(),
            "stage_runs": stage_runs,
        }
    )
    json_dump(manifest_path, manifest)


def validate_case_artifacts(root: Path, case_id: str) -> None:
    state_path = root / f"outputs/{case_id}/state.json"
    manifest_path = root / f"outputs/{case_id}/run_manifest.json"
    state = load_json_or_empty(state_path)
    manifest = load_json_or_empty(manifest_path)
    state_schema = json.loads((root / "schemas/state.schema.json").read_text(encoding="utf-8"))
    manifest_schema = json.loads((root / "schemas/run_manifest.schema.json").read_text(encoding="utf-8"))
    jsonschema_validate(instance=state, schema=state_schema)
    jsonschema_validate(instance=manifest, schema=manifest_schema)


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage1 = config["stage1"]
    stage5 = dict(config["stage5"])
    cases_config = load_yaml(root / config["stage2"]["cases_frozen_config"])
    software_versions = stage5_software_versions()
    cases = selected_cases(cases_config, args.case_id)
    if not cases:
        raise SystemExit(f"No mutation-effect step case matched --case-id={args.case_id}")

    all_effect_frames: list[pd.DataFrame] = []
    all_calibration_frames: list[pd.DataFrame] = []
    case_qc_rows: list[dict[str, Any]] = []

    commands = [
        f"python scripts/06_run_docking.py --config {args.config}" + (f" --case-id {args.case_id}" if args.case_id else ""),
        f"python scripts/07_compute_ifp.py --config {args.config}" + (f" --case-id {args.case_id}" if args.case_id else ""),
        f"python scripts/08b_calibrate_scoring.py --config {args.config}" + (f" --case-id {args.case_id}" if args.case_id else ""),
    ]

    for case_entry in cases:
        started_at = iso_now()
        case_id = str(case_entry["case_id"])
        case_stage5 = stage5_for_case(stage5, case_id)
        effect_agent = MutationEffectAgent(
            config=MutationEffectAgentConfig(
                temperature=float(case_stage5.get("agent_temperature", 0.2)),
                max_tokens=int(case_stage5["agent_max_tokens"]),
                thinking={"type": str(case_stage5["agent_thinking"])},
                top_site_n=int(case_stage5["agent_top_site_n"]),
                top_combo_n=int(case_stage5["agent_top_combo_n"]),
            )
        )
        case_root = ensure_dir(root / "outputs" / case_id / "stage5")
        docking_frame = read_csv_optional(case_root / "docking_scores.csv")
        ifp_frame = read_csv_optional(case_root / "ifp_diff.csv")
        ifp_frame = drop_stale_scoring_columns(ifp_frame)
        docking_lookup = {
            (str(row["effect_scope"]), str(row["target_key"])): row
            for row in docking_frame.to_dict(orient="records")
        }
        gpu_ids = configured_gpu_ids(stage5)
        scoring_jobs = [
            {
                "root": str(root),
                "case_id": case_id,
                "effect_scope": str(row["effect_scope"]),
                "target_key": str(row["target_key"]),
                "docking_row": docking_lookup[(str(row["effect_scope"]), str(row["target_key"]))],
                "stage5": case_stage5,
                "gpu_id": gpu_ids[index % len(gpu_ids)] if gpu_ids else None,
            }
            for index, row in enumerate(ifp_frame.to_dict(orient="records"))
            if str(row.get("ifp_status") or "") == "ok" and (str(row["effect_scope"]), str(row["target_key"])) in docking_lookup
        ]
        scoring_rows: list[dict[str, Any]] = []
        max_workers = configured_max_workers(
            case_stage5,
            scoring_job_count=len(scoring_jobs),
            gpu_ids=gpu_ids,
        )
        if max_workers == 1:
            for job in scoring_jobs:
                scoring_rows.append(run_scoring_job(job))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(run_scoring_job, job): job for job in scoring_jobs}
                for future in as_completed(future_map):
                    scoring_rows.append(future.result())
        scoring_frame = pd.DataFrame.from_records(scoring_rows)
        if not scoring_frame.empty:
            ifp_frame = ifp_frame.merge(scoring_frame, on=["case_id", "effect_scope", "target_key"], how="left")
        else:
            for column, default in [
                ("scoring_status", "skipped"),
                ("scoring_error", None),
                ("wt_mmgbsa_binding_kcal_mol", None),
                ("mt_mmgbsa_binding_kcal_mol", None),
                ("delta_mmgbsa_binding_kcal_mol", None),
                ("wt_gnina_affinity_kcal_mol", None),
                ("mt_gnina_affinity_kcal_mol", None),
                ("delta_gnina_affinity_kcal_mol", None),
                ("available_score_count", 0),
                ("nonzero_direction_count", 0),
                ("consensus_fraction", None),
                ("consensus_direction", "neutral"),
                ("high_uncertainty", True),
            ]:
                ifp_frame[column] = default
        write_table(ifp_frame, case_root / "ifp_diff.csv")

        site_effect_frame = ifp_frame[ifp_frame["effect_scope"].eq("site")].copy() if not ifp_frame.empty else pd.DataFrame()
        combo_effect_frame = ifp_frame[ifp_frame["effect_scope"].eq("combo")].copy() if not ifp_frame.empty else pd.DataFrame()

        site_lookup = {
            str(row["target_key"]): row
            for _, row in site_effect_frame[site_effect_frame["ifp_status"].eq("ok")].iterrows()
        }
        if not combo_effect_frame.empty:
            combo_effect_frame["epistasis_flag"] = combo_effect_frame.apply(assign_epistasis_flag, axis=1, args=(site_lookup,))
        else:
            combo_effect_frame["epistasis_flag"] = pd.Series(dtype=str)

        ok_effects = pd.concat(
            [
                site_effect_frame[site_effect_frame["ifp_status"].eq("ok")],
                combo_effect_frame[combo_effect_frame["ifp_status"].eq("ok")],
            ],
            ignore_index=True,
        )
        ddg_lookup = load_empirical_ddg_lookup(
            root,
            stage1,
            docking_frame.get("representative_sample_id", pd.Series(dtype=str)).fillna("").astype(str).tolist(),
        )
        case_calibration = calibration_frame(effect_frame=ok_effects, ddg_lookup=ddg_lookup)
        calibration_summary = calibration_metrics(case_calibration)
        write_case_summary_artifacts(
            case_root=case_root,
            effect_frame=ok_effects,
            calibration_rows=case_calibration,
        )

        qc_payload = {
            "case_id": case_id,
            "selected_site_targets": int(site_effect_frame.shape[0]),
            "selected_combo_targets": int(combo_effect_frame.shape[0]),
            "docking_success_count": int(docking_frame["stage5_status"].eq("ok").sum()) if not docking_frame.empty else 0,
            "docking_failure_count": int(docking_frame["stage5_status"].eq("failed").sum()) if not docking_frame.empty else 0,
            "skipped_unready_count": int(docking_frame["stage5_status"].eq("skipped").sum()) if not docking_frame.empty else 0,
            "ifp_success_count": int(ifp_frame["ifp_status"].eq("ok").sum()) if not ifp_frame.empty else 0,
            "ifp_skipped_count": int(ifp_frame["ifp_status"].eq("skipped").sum()) if not ifp_frame.empty else 0,
            "scoring_success_count": int(ifp_frame["scoring_status"].fillna("").eq("ok").sum()) if not ifp_frame.empty else 0,
            "scoring_failure_count": int(ifp_frame["scoring_status"].fillna("").eq("failed").sum()) if not ifp_frame.empty else 0,
            "local_sampling_count": int(ifp_frame["local_sampling_applied"].fillna(False).astype(bool).sum()) if not ifp_frame.empty else 0,
            "high_uncertainty_count": int(ifp_frame["high_uncertainty"].fillna(True).astype(bool).sum()) if not ifp_frame.empty else 0,
            "mechanism_label_counts": mechanism_label_counts(ok_effects),
            "epistasis_flag_counts": combo_effect_frame["epistasis_flag"].fillna("unresolved").value_counts().to_dict()
            if not combo_effect_frame.empty
            else {},
            **calibration_summary,
        }

        llm_input_path = case_root / "effect_story_input.json"
        llm_payload_path = case_root / "effect_story_payload.json"
        story_path = case_root / "mutation_effect_story.md"
        agent_input, llm_payload, llm_record, agent_status, agent_error = run_mutation_effect_agent(
            effect_agent=effect_agent,
            case_entry=case_entry,
            site_effect_frame=site_effect_frame,
            combo_effect_frame=combo_effect_frame,
            qc_payload=qc_payload,
            calibration_summary=calibration_summary,
        )

        write_json(llm_input_path, agent_input)
        write_json(llm_payload_path, llm_payload)

        if llm_record is not None:
            qc_payload.update(
                {
                    "mutation_effect_model": llm_record.model,
                    "mutation_effect_prompt_hash": llm_record.prompt_hash,
                    "mutation_effect_tokens": llm_record.tokens,
                    "mutation_effect_latency_seconds": float(llm_record.latency_seconds),
                    "mutation_effect_retry_count": int(llm_record.retry_count),
                    "mutation_effect_thinking": {"type": str(case_stage5["agent_thinking"])},
                }
            )
        else:
            qc_payload.update(
                {
                    "mutation_effect_model": None,
                    "mutation_effect_prompt_hash": None,
                    "mutation_effect_tokens": 0,
                    "mutation_effect_latency_seconds": 0.0,
                    "mutation_effect_retry_count": 0,
                    "mutation_effect_thinking": {"type": str(case_stage5["agent_thinking"])},
                }
            )
        qc_payload.update(
            {
                "mutation_effect_agent_status": str(agent_status),
                "mutation_effect_agent_error": agent_error,
            }
        )

        mutation_effects = merge_agent_site_effects(site_effect_frame, llm_payload)
        combination_effects = merge_agent_combo_effects(combo_effect_frame, llm_payload)
        write_json(case_root / "mutation_effects.json", mutation_effects)
        write_json(case_root / "combination_effects.json", combination_effects)
        write_story(
            story_path,
            build_case_story_lines(
                case_entry=case_entry,
                qc_payload=qc_payload,
                payload=llm_payload,
                site_effects=mutation_effects,
                combo_effects=combination_effects,
            ),
        )
        write_json(case_root / "stage5_qc.json", qc_payload)

        write_case_state(
            root=root,
            case_entry=case_entry,
            args_config=args.config,
            stage1=stage1,
            stage5=case_stage5,
            site_effect_frame=site_effect_frame,
            combo_effect_frame=combo_effect_frame,
            qc_payload=qc_payload,
            llm_payload=llm_payload,
            llm_record=llm_record,
            llm_input_path=llm_input_path,
            llm_payload_path=llm_payload_path,
            story_path=story_path,
            software_versions=software_versions,
            commands=commands,
        )
        write_case_manifest(
            root=root,
            case_entry=case_entry,
            args_config=args.config,
            stage1=stage1,
            stage5=case_stage5,
            started_at=started_at,
            software_versions=software_versions,
            commands=commands,
            llm_record=llm_record,
            llm_input_path=llm_input_path,
            llm_payload_path=llm_payload_path,
            story_path=story_path,
        )
        validate_case_artifacts(root, case_id)

        case_qc_rows.append(qc_payload)
        if not ok_effects.empty:
            all_effect_frames.append(ok_effects.assign(case_id=case_id))
        if not case_calibration.empty:
            all_calibration_frames.append(case_calibration)

    combined_effects = pd.concat(all_effect_frames, ignore_index=True) if all_effect_frames else pd.DataFrame()
    combined_calibration = pd.concat(all_calibration_frames, ignore_index=True) if all_calibration_frames else pd.DataFrame()
    if args.case_id is None:
        cluster_frame = mechanism_cluster_frame(combined_effects) if not combined_effects.empty else mechanism_cluster_frame(pd.DataFrame())
        project_root_stage5 = ensure_dir(root / "outputs" / "stage5")
        write_table(cluster_frame, project_root_stage5 / "mechanism_clusters.csv")
        write_table(combined_calibration, project_root_stage5 / "scoring_calibration.csv")
        plot_calibration(combined_calibration, project_root_stage5 / "scoring_calibration_plot.png")

    total_tokens = int(sum(int(row.get("mutation_effect_tokens") or 0) for row in case_qc_rows))
    mean_latency = (
        0.0
        if not case_qc_rows
        else float(sum(float(row.get("mutation_effect_latency_seconds") or 0.0) for row in case_qc_rows) / len(case_qc_rows))
    )
    project_qc = {
        "case_count": int(len(case_qc_rows)),
        "selected_site_targets_total": int(sum(int(row.get("selected_site_targets") or 0) for row in case_qc_rows)),
        "selected_combo_targets_total": int(sum(int(row.get("selected_combo_targets") or 0) for row in case_qc_rows)),
        "docking_success_total": int(sum(int(row.get("docking_success_count") or 0) for row in case_qc_rows)),
        "docking_failure_total": int(sum(int(row.get("docking_failure_count") or 0) for row in case_qc_rows)),
        "skipped_unready_total": int(sum(int(row.get("skipped_unready_count") or 0) for row in case_qc_rows)),
        "ifp_success_total": int(sum(int(row.get("ifp_success_count") or 0) for row in case_qc_rows)),
        "scoring_success_total": int(sum(int(row.get("scoring_success_count") or 0) for row in case_qc_rows)),
        "scoring_failure_total": int(sum(int(row.get("scoring_failure_count") or 0) for row in case_qc_rows)),
        "local_sampling_total": int(sum(int(row.get("local_sampling_count") or 0) for row in case_qc_rows)),
        "high_uncertainty_total": int(sum(int(row.get("high_uncertainty_count") or 0) for row in case_qc_rows)),
        "calibration_sample_count": int(len(combined_calibration)),
        **calibration_metrics(combined_calibration),
        "mechanism_label_counts": mechanism_label_counts(combined_effects) if not combined_effects.empty else {},
        "mutation_effect_model_set": sorted(
            {
                str(row.get("mutation_effect_model"))
                for row in case_qc_rows
                if row.get("mutation_effect_model") not in {None, ""}
            }
        ),
        "mutation_effect_total_tokens": total_tokens,
        "mutation_effect_mean_latency_seconds": mean_latency,
    }
    if args.case_id is None:
        write_json(project_root_stage5 / "stage5_qc.json", project_qc)


if __name__ == "__main__":
    main()
