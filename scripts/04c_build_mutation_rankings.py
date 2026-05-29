#!/usr/bin/env python3
"""mutation-proposal step mutation proposal ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import validate as jsonschema_validate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FITNESS_FORMULA_VERSION = "resistagent_default_v1"

from agents.mutation_proposal_agent import MutationProposalAgent, MutationProposalAgentConfig
from agents.orchestrator import DecisionRecord, StageOrchestrator
from tools.runtime import (
    command_exists,
    detect_git_commit,
    ensure_dir,
    iso_now,
    json_dump,
    load_yaml,
    project_root,
    run_command,
    sha256_file,
    text_dump,
)
from tools.stage4_utils import (
    build_case_context,
    build_hiv_pose_reference,
    conservation_values_for_positions,
    empirical_position_entropy_lookup,
    estimate_ddg_surrogate,
    fitness_weight_from_ddg,
    heuristic_impact_from_context,
    local_backbone_rmsd,
    load_alphafold_position_entropy,
    materialize_synthetic_combo_sample,
    ndcg_at_k,
    parse_mutation_components,
    rank_normalize,
    summarize_conservation_source,
    sample_proxy_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mutation-proposal step mutation proposal rankings.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to base config.")
    parser.add_argument("--case-id", dest="case_ids", action="append", default=None, help="Optional case_id filter; may be provided multiple times.")
    return parser.parse_args()


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    if pd.isna(value):
        return default
    return float(value)


def _scope_label(
    structure_layers: list[str],
    p_drug_selected: float,
    *,
    proxy_eligible: bool = True,
) -> tuple[str, bool, float]:
    if "anchor" in structure_layers:
        return "anchor", bool(proxy_eligible), 1.0
    if "pocket" in structure_layers:
        return "pocket", bool(proxy_eligible), 1.0
    if "second_shell" in structure_layers:
        return "second_shell", bool(proxy_eligible), 1.0
    if p_drug_selected > 0.0:
        return "prior_hotspot", False, 0.15
    return "other", False, 0.35


def _rule_based_mode(row: pd.Series) -> tuple[bool, str]:
    evidence = "|".join(
        str(row.get(column) or "")
        for column in [
            "background_evidence_tier",
            "drug_selected_evidence_tier",
            "background_source_list",
            "drug_selected_source_list",
        ]
    ).lower()
    if "rule_based_fallback" in evidence:
        return True, "rule_based_fallback"
    return False, "data_driven"


def _list_string(values: list[Any]) -> str:
    return "|".join(str(value) for value in values if value not in (None, ""))


def _sample_status_map(case_id: str, root: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    status_path = root / f"outputs/{case_id}/stage3_2/mutation_site_status.csv"
    residue_map_path = root / f"outputs/{case_id}/stage3_2/residue_map.json"
    status = pd.read_csv(status_path)
    residue_payload = json.loads(residue_map_path.read_text(encoding="utf-8"))
    sample_records = {str(row["sample_id"]): row for row in residue_payload.get("samples", [])}
    wt_template = residue_payload["wt_template"]
    return status, sample_records, wt_template


def _contains_forbidden_viral_sources(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    columns = [
        "background_source_list",
        "drug_selected_source_list",
        "source_db",
    ]
    tokens: list[str] = []
    for column in columns:
        tokens.extend(frame.get(column, pd.Series(dtype=str)).fillna("").astype(str).tolist())
    text = "|".join(tokens)
    return "cosmic" in text.lower() or "depmap" in text.lower()


def _component_positions(components: list[Any]) -> list[int]:
    return [int(component.position) for component in components if component.position is not None]


def _case_rank_qc(frame: pd.DataFrame, top_k: int, relevance_col: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "candidate_count": 0,
            "top20_hotspot_hits": 0,
            "top20_hotspot_recall": 0.0,
            "ndcg_risk": 0.0,
            "ndcg_frequency": 0.0,
            "ndcg_background_only": 0.0,
            "ndcg_random_mean": 0.0,
        }
    hotspot_total = int(frame[relevance_col].fillna(False).astype(bool).sum())
    top = frame.head(top_k)
    hotspot_hits = int(top[relevance_col].fillna(False).astype(bool).sum())
    denominator = min(int(top_k), hotspot_total) if hotspot_total > 0 else 0
    random_scores: list[float] = []
    relevance = frame[relevance_col].fillna(False).astype(int).tolist()
    rng = random.Random(42)
    for _ in range(200):
        shuffled = list(range(len(frame)))
        rng.shuffle(shuffled)
        shuffled_frame = frame.copy()
        shuffled_frame["__random_score__"] = shuffled
        random_scores.append(ndcg_at_k(shuffled_frame, "__random_score__", relevance_col, top_k))
    return {
        "candidate_count": int(len(frame)),
        "top20_hotspot_hits": hotspot_hits,
        "top20_hotspot_recall": 0.0 if denominator == 0 else float(hotspot_hits / denominator),
        "ndcg_risk": ndcg_at_k(frame, "risk_score", relevance_col, top_k),
        "ndcg_frequency": ndcg_at_k(frame, "frequency_only_score", relevance_col, top_k),
        "ndcg_background_only": ndcg_at_k(frame, "background_only_score", relevance_col, top_k),
        "ndcg_random_mean": float(sum(random_scores) / len(random_scores)),
    }


def _residue_risk_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "gene_symbol",
                "target_position",
                "best_mutation_key",
                "max_risk_score",
                "mean_risk_score",
                "candidate_count",
                "structure_layer",
                "known_hotspot_any",
                "functional_site_flag",
                "min_ligand_distance_a",
                "mean_position_entropy",
                "conservation_source",
                "solvent_accessibility",
                "secondary_structure_context",
            ]
        )
    records: list[dict[str, Any]] = []
    for (_, position), group in frame.groupby(["case_id", "target_position"], dropna=False):
        ordered = group.sort_values("risk_score", ascending=False)
        best = ordered.iloc[0]
        records.append(
            {
                "case_id": best["case_id"],
                "gene_symbol": best["gene_symbol"],
                "target_position": position,
                "best_mutation_key": best["mutation_key"],
                "max_risk_score": float(best["risk_score"]),
                "mean_risk_score": float(group["risk_score"].mean()),
                "candidate_count": int(len(group)),
                "structure_layer": best["structure_layer"],
                "known_hotspot_any": bool(group["known_hotspot"].fillna(False).astype(bool).any()),
                "functional_site_flag": bool(group["functional_site_flag"].fillna(False).astype(bool).any()),
                "min_ligand_distance_a": _safe_float(best["min_ligand_distance_a"]),
                "mean_position_entropy": _safe_float(group["position_entropy_mean"].mean()) if "position_entropy_mean" in group else None,
                "conservation_source": best.get("conservation_source"),
                "solvent_accessibility": best["solvent_accessibility"],
                "secondary_structure_context": best["secondary_structure_context"],
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["case_id", "max_risk_score"], ascending=[True, False])


def _markdown_table(frame: pd.DataFrame, columns: list[str], rename: dict[str, str] | None = None, top_n: int = 5) -> list[str]:
    if frame.empty:
        return ["无可报告条目。"]
    rename = rename or {}
    subset = frame[columns].head(top_n).copy()
    subset = subset.rename(columns=rename)
    header = "| " + " | ".join(subset.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(subset.columns)) + " |"
    rows = [
        "| " + " | ".join(
            f"{value:.4f}" if isinstance(value, float) and math.isfinite(value) else str(value)
            for value in row
        )
        + " |"
        for row in subset.itertuples(index=False, name=None)
    ]
    return [header, divider, *rows]


def _native_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else float(value)
    if hasattr(value, "item"):
        return _native_value(value.item())
    return value


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _load_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return [value]


def _sample_candidate_maps(case_master: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    site_candidates: dict[str, list[tuple[int, int, str]]] = {}
    combo_candidates: dict[str, list[str]] = {}
    for row in case_master.itertuples(index=False):
        sample_id = str(row.SAMPLE_ID)
        combination_size = int(row.combination_size)
        for mutation_key in _as_list(getattr(row, "component_mutation_keys", [])):
            key = str(mutation_key)
            site_candidates.setdefault(key, []).append((0 if combination_size == 1 else 1, combination_size, sample_id))
        combo_key = str(getattr(row, "combination_key", "") or "")
        if combination_size > 1 and combo_key:
            combo_candidates.setdefault(combo_key, []).append(sample_id)
    ordered_sites = {
        key: [sample_id for _, _, sample_id in sorted(rows, key=lambda item: (item[0], item[1], item[2]))]
        for key, rows in site_candidates.items()
    }
    ordered_combos = {
        key: sorted(dict.fromkeys(rows))
        for key, rows in combo_candidates.items()
    }
    return ordered_sites, ordered_combos


def _resolve_eligible_sample_id(
    candidates: list[str],
    *,
    status_lookup: dict[str, dict[str, Any]],
    sample_records: dict[str, dict[str, Any]],
    extraction_lookup: dict[str, dict[str, Any]],
) -> str:
    for sample_id in candidates:
        status_row = status_lookup.get(sample_id)
        sample_record = sample_records.get(sample_id)
        extraction_row = extraction_lookup.get(sample_id)
        if sample_record is None or status_row is None or extraction_row is None:
            continue
        if not bool(status_row.get("eligible_for_stage5", False)):
            continue
        if not bool(extraction_row.get("core_roles_complete", False)):
            continue
        return sample_id
    return ""


def _environment_snapshot() -> dict[str, str]:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env and command_exists("conda"):
        result = run_command(["conda", "env", "export", "-n", conda_env, "--no-builds"])
        if result.returncode == 0:
            return {"kind": "conda_export", "content": result.stdout}
    result = run_command([sys.executable, "-m", "pip", "freeze"])
    return {"kind": "pip_freeze", "content": result.stdout if result.returncode == 0 else ""}


def _package_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def _software_versions() -> dict[str, str | None]:
    versions = {"python": platform.python_version()}
    for binary, args, key in [
        ("python3", ["--version"], "python3"),
        ("conda", ["--version"], "conda"),
        ("snakemake", ["--version"], "snakemake_cli"),
        ("vina", ["--version"], "vina"),
        ("obabel", ["-V"], "obabel"),
        ("plip", ["-h"], "plip_cli"),
    ]:
        if not command_exists(binary):
            versions[key] = None
            continue
        result = run_command([binary] + args)
        versions[key] = (result.stdout or result.stderr).strip().splitlines()[0] if result.returncode == 0 else None
    versions["openai"] = _package_version("openai")
    versions["pandas"] = _package_version("pandas")
    versions["pyyaml"] = _package_version("PyYAML")
    versions["biopython"] = _package_version("biopython")
    return versions


def _hashed_inputs(paths: list[Path], root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        if path.exists():
            hashes[_relative_path(path, root)] = sha256_file(path)
    return hashes


def _frame_records(frame: pd.DataFrame, columns: list[str], top_n: int) -> list[dict[str, Any]]:
    if frame.empty or top_n <= 0:
        return []
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].head(top_n).copy()
    return [
        {column: _native_value(value) for column, value in row.items()}
        for row in subset.to_dict(orient="records")
    ]


def _structural_context_summary(context: dict[str, Any], context_frame: pd.DataFrame) -> dict[str, Any]:
    layer_counts = {
        str(layer): int(count)
        for layer, count in context_frame.get("structure_layer", pd.Series(dtype=str)).fillna("unknown").value_counts().to_dict().items()
    }
    return {
        "anchor_positions": [int(position) for position in sorted(context.get("anchor_positions", []))],
        "layer_counts": layer_counts,
        "anchor_residue_count": int(context_frame.get("anchor_flag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "baseline_ifp_residue_count": int(context_frame.get("baseline_ifp_flag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
    }


def _source_tokens(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {token.strip() for token in str(value).split("|") if token.strip()}


def _combo_projection_leakage_mutations(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    mask = (
        frame.get("single_sample_support", pd.Series(dtype=float)).fillna(0).astype(int).eq(0)
        & frame.get("drug_selected_evidence_tier", pd.Series(dtype=str)).fillna("").str.contains("observed_same_drug", case=False)
    )
    return sorted(frame.loc[mask, "mutation_key"].astype(str).tolist())


def _site_priors_unchanged(site_rank: pd.DataFrame, case_site: pd.DataFrame) -> bool:
    if site_rank.empty or case_site.empty:
        return True
    baseline = case_site[["mutation_key", "P_background", "P_drug_selected"]].drop_duplicates("mutation_key")
    current = site_rank[["mutation_key", "P_background", "P_drug_selected"]].drop_duplicates("mutation_key")
    merged = baseline.merge(current, on="mutation_key", how="outer", suffixes=("_baseline", "_current"))
    for _, row in merged.iterrows():
        for field in ["P_background", "P_drug_selected"]:
            left = _safe_float(row.get(f"{field}_baseline"), 0.0) or 0.0
            right = _safe_float(row.get(f"{field}_current"), 0.0) or 0.0
            if not math.isclose(float(left), float(right), rel_tol=1.0e-9, abs_tol=1.0e-12):
                return False
    return True


def _site_component_audit(site_rank: pd.DataFrame) -> bool:
    if site_rank.empty:
        return True
    if site_rank.get("component_count", pd.Series(dtype=float)).fillna(0).astype(int).gt(1).any():
        return False
    return not site_rank.get("mutation_key", pd.Series(dtype=str)).fillna("").str.contains(r"\+").any()


def _combo_component_audit(combo_rank: pd.DataFrame) -> bool:
    if combo_rank.empty:
        return True
    return combo_rank.get("combination_key", pd.Series(dtype=str)).fillna("").str.contains(r"\+").all()


def _viral_allowed_source_audit(site_rank: pd.DataFrame, combo_rank: pd.DataFrame) -> bool:
    if site_rank.empty or not site_rank.get("domain_type", pd.Series(dtype=str)).fillna("").eq("viral").any():
        return True
    allowed_site_sources = {"HIVDB_GenoRx", "HIVDB_rule_fallback", "MdrDB", "epsilon_background", "none"}
    for column in ["background_source_list", "drug_selected_source_list"]:
        for value in site_rank.get(column, pd.Series(dtype=str)).fillna("").tolist():
            if not _source_tokens(value).issubset(allowed_site_sources):
                return False
    for value in combo_rank.get("source_db", pd.Series(dtype=str)).fillna("").tolist():
        if not _source_tokens(value).issubset({"MdrDB"}):
            return False
    return True


def _validate_stage4_case_artifacts(root: Path, state_path: Path, manifest_path: Path) -> None:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state_schema = json.loads((root / "schemas/state.schema.json").read_text(encoding="utf-8"))
    manifest_schema = json.loads((root / "schemas/run_manifest.schema.json").read_text(encoding="utf-8"))
    jsonschema_validate(instance=state, schema=state_schema)
    jsonschema_validate(instance=manifest, schema=manifest_schema)


def _story_site_lines(site_frame: pd.DataFrame, payload: dict[str, Any]) -> list[str]:
    if not payload.get("site_interpretations"):
        return ["无 agent 解释条目。"]
    lookup = {str(row["mutation_key"]): row for row in site_frame.to_dict(orient="records")}
    lines: list[str] = []
    for row in payload["site_interpretations"]:
        mutation_key = str(row["mutation_key"])
        table_row = lookup.get(mutation_key, {})
        evidence = []
        if table_row:
            evidence.append(f"risk={float(table_row['risk_score']):.4f}")
            evidence.append(f"P={float(table_row['P_appearance']):.4f}")
            evidence.append(f"layer={table_row['structure_layer']}")
            evidence.append(f"tier={table_row['impact_evidence_tier']}")
        lines.append(
            f"- `{mutation_key}`: {row['reasoning']} 机制假设：{row['mechanism_hypothesis']}。"
            f" 支撑：`{' ; '.join(evidence)}`。置信度：`{row['confidence']}`。局限：{row['caveat']}"
        )
    return lines


def _story_combo_lines(combo_frame: pd.DataFrame, payload: dict[str, Any]) -> list[str]:
    if not payload.get("combo_interpretations"):
        return ["无可报告组合条目。"]
    lookup = {str(row["combination_key"]): row for row in combo_frame.to_dict(orient="records")}
    lines: list[str] = []
    for row in payload["combo_interpretations"]:
        combo_key = str(row["combination_key"])
        table_row = lookup.get(combo_key, {})
        evidence = []
        if table_row:
            evidence.append(f"risk={float(table_row['risk_score']):.4f}")
            evidence.append(f"P_combo={float(table_row['P_appearance_combo']):.4f}")
            evidence.append(f"tier={table_row['impact_evidence_tier']}")
        lines.append(
            f"- `{combo_key}`: {row['reasoning']} 机制假设：{row['mechanism_hypothesis']}。"
            f" 支撑：`{' ; '.join(evidence)}`。置信度：`{row['confidence']}`。局限：{row['caveat']}"
        )
    return lines


def _story_residue_lines(payload: dict[str, Any]) -> list[str]:
    if not payload.get("residue_watchlist"):
        return ["无额外残基 watchlist。"]
    return [
        f"- `{int(row['target_position'])}`: 最优突变为 `{row['best_mutation_key']}`。{row['why_it_matters']}"
        for row in payload["residue_watchlist"]
    ]


def _write_case_state(
    *,
    root: Path,
    case_entry: dict[str, Any],
    args_config: str,
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    stage3: dict[str, Any],
    stage4: dict[str, Any],
    case_output_root: Path,
    qc_payload: dict[str, Any],
    site_rank: pd.DataFrame,
    combo_rank: pd.DataFrame,
    llm_payload: dict[str, Any],
    llm_record: Any,
    llm_input_path: Path,
    llm_payload_path: Path,
    conservation_profile_path: Path,
    story_path: Path,
    software_versions: dict[str, Any],
    commands: list[str],
) -> None:
    case_id = str(case_entry["case_id"])
    state_path = root / f"outputs/{case_id}/state.json"
    state = _load_json_or_empty(state_path)
    inputs = state.get("inputs") if isinstance(state.get("inputs"), dict) else {}
    artifacts = state.get("artifacts") if isinstance(state.get("artifacts"), dict) else {}
    qc_block = state.get("qc") if isinstance(state.get("qc"), dict) else {}
    llm_decisions = state.get("llm_decisions") if isinstance(state.get("llm_decisions"), list) else []

    inputs.update(
        {
            "config": args_config,
            "master_table": stage1["master_table"],
            "cases": stage2["cases_frozen_config"],
            "case_manifest": str(case_entry.get("manifest_path") or ""),
            "site_pool_report": stage2["site_pool_report"],
            "combo_panel": stage2["combo_panel"],
            "extraction_source_report": stage3["extraction_source_report"],
        }
    )
    artifacts["wt"] = {
        "complex_pdb": _relative_path(root / f"outputs/{case_id}/stage3_5/wt_complex.pdb", root),
        "ifp": _relative_path(root / f"outputs/{case_id}/stage3_5/wt_ifp.json", root),
        "anchor_residues": _relative_path(root / f"outputs/{case_id}/stage3_5/wt_anchor_residues.txt", root),
    }
    artifacts["priors"] = {
        "site_pool": stage2["site_pool_report"],
        "combo_panel": stage2["combo_panel"],
        "fitness_formula_version": FITNESS_FORMULA_VERSION,
    }
    artifacts["mutations"] = {
        "topk": _frame_records(site_rank, ["mutation_rank", "mutation_key", "risk_score"], int(stage4["top_k"])),
        "rank_csv": _relative_path(case_output_root / "mutation_rank.csv", root),
        "residue_risk_csv": _relative_path(case_output_root / "residue_risk.csv", root),
        "combo_rank_csv": _relative_path(case_output_root / "combo_rank.csv", root),
    }
    artifacts["stage4"] = {
        "structural_context_csv": _relative_path(case_output_root / "structural_context.csv", root),
        "sample_proxy_features_csv": _relative_path(case_output_root / "sample_proxy_features.csv", root),
        "conservation_profile_json": _relative_path(conservation_profile_path, root),
        "mutation_story_input_json": _relative_path(llm_input_path, root),
        "mutation_story_payload_json": _relative_path(llm_payload_path, root),
        "mutation_story_md": _relative_path(story_path, root),
        "stage4_qc_json": _relative_path(case_output_root / "stage4_qc.json", root),
    }
    qc_block["stage4"] = qc_payload
    qc_block["docking_fail_rate"] = qc_payload["site_docking_failure_rate"]

    state.update(
        {
            "project_id": case_id,
            "stage": "stage4",
            "inputs": inputs,
            "artifacts": artifacts,
            "qc": qc_block,
            "software_versions": software_versions,
            "seeds": {
                "python": 42,
                "quick_docking_seeds": list(stage4["quick_docking_seeds"]),
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
            agent_name="MutationProposalAgent",
            decision_type="mutation_story_structuring",
            input_artifacts=[
                _relative_path(case_output_root / "mutation_rank.csv", root),
                _relative_path(case_output_root / "residue_risk.csv", root),
                _relative_path(case_output_root / "combo_rank.csv", root),
                _relative_path(case_output_root / "structural_context.csv", root),
                _relative_path(conservation_profile_path, root),
                _relative_path(case_output_root / "stage4_qc.json", root),
                _relative_path(llm_input_path, root),
            ],
            tool_calls=[
                {
                    "tool_name": "GLMClient.chat_json",
                    "model": llm_record.model,
                    "base_url": os.environ.get("GLM_BASE_URL"),
                    "temperature": float(stage4.get("agent_temperature", 0.2)),
                    "max_tokens": int(stage4["agent_max_tokens"]),
                    "thinking": {"type": str(stage4["agent_thinking"])},
                    "prompt_hash": llm_record.prompt_hash,
                    "tokens": llm_record.tokens,
                    "latency_seconds": llm_record.latency_seconds,
                    "retry_count": llm_record.retry_count,
                }
            ],
            decision_rationale=str(llm_payload["executive_summary"]["overall_risk_pattern"]),
            output_artifacts=[
                _relative_path(llm_payload_path, root),
                _relative_path(story_path, root),
            ],
        )
    )


def _write_case_manifest(
    *,
    root: Path,
    case_entry: dict[str, Any],
    args_config: str,
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    stage3: dict[str, Any],
    stage4: dict[str, Any],
    case_output_root: Path,
    started_at: str,
    software_versions: dict[str, Any],
    env_snapshot: dict[str, str],
    git_commit: str | None,
    git_status: str,
    commands: list[str],
    llm_record: Any,
    llm_input_path: Path,
    llm_payload_path: Path,
    conservation_profile_path: Path,
    story_path: Path,
) -> None:
    case_id = str(case_entry["case_id"])
    manifest_path = root / f"outputs/{case_id}/run_manifest.json"
    manifest = _load_json_or_empty(manifest_path)
    stage_runs = manifest.get("stage_runs") if isinstance(manifest.get("stage_runs"), dict) else {}
    input_paths = [
        root / args_config,
        root / stage1["master_table"],
        root / stage2["cases_frozen_config"],
        root / stage2["site_pool_report"],
        root / stage2["combo_panel"],
        root / stage3["extraction_source_report"],
        root / f"outputs/{case_id}/stage3_2/residue_map.json",
        root / f"outputs/{case_id}/stage3_2/mutation_site_status.csv",
        root / f"outputs/{case_id}/stage3_5/wt_complex.pdb",
        root / f"outputs/{case_id}/stage3_5/wt_ifp.json",
        root / f"outputs/{case_id}/stage3_5/wt_anchor_residues.txt",
        case_output_root / "structural_context.csv",
        conservation_profile_path,
        llm_input_path,
    ]
    input_hashes = manifest.get("input_hashes") if isinstance(manifest.get("input_hashes"), dict) else {}
    input_hashes.update(_hashed_inputs(input_paths, root))
    stage_runs["stage4"] = {
        "started_at": started_at,
        "finished_at": iso_now(),
        "outputs": [
            _relative_path(case_output_root / "mutation_rank.csv", root),
            _relative_path(case_output_root / "residue_risk.csv", root),
            _relative_path(case_output_root / "combo_rank.csv", root),
            _relative_path(case_output_root / "stage4_qc.json", root),
            _relative_path(conservation_profile_path, root),
            _relative_path(llm_input_path, root),
            _relative_path(llm_payload_path, root),
            _relative_path(story_path, root),
        ],
    }
    if llm_record is not None:
        stage_runs["stage4"]["llm"] = {
            "provider": "zhipu_glm",
            "model": llm_record.model,
            "base_url": os.environ.get("GLM_BASE_URL"),
            "prompt_hash": llm_record.prompt_hash,
            "temperature": float(stage4.get("agent_temperature", 0.2)),
            "max_tokens": int(stage4["agent_max_tokens"]),
            "thinking": {"type": str(stage4["agent_thinking"])},
            "tokens": llm_record.tokens,
            "latency_seconds": llm_record.latency_seconds,
            "retry_count": llm_record.retry_count,
        }
    manifest.update(
        {
            "project_id": case_id,
            "stage": "stage4",
            "git_commit": git_commit,
            "git_status": git_status,
            "software_versions": software_versions,
            "env_snapshot": env_snapshot,
            "random_seeds": {
                "python": 42,
                "quick_docking_seeds": list(stage4["quick_docking_seeds"]),
            },
            "input_hashes": input_hashes,
            "commands": commands,
            "started_at": manifest.get("started_at", started_at),
            "finished_at": iso_now(),
            "stage_runs": stage_runs,
        }
    )
    json_dump(manifest_path, manifest)


def _story_lines(
    case_entry: dict[str, Any],
    site_frame: pd.DataFrame,
    combo_frame: pd.DataFrame,
    qc: dict[str, Any],
    llm_payload: dict[str, Any],
) -> list[str]:
    title = f"# {case_entry['case_id']} mutation-proposal step Mutation Proposal"
    lines = [
        title,
        "",
        f"- 目标：`{case_entry['target_name']}`",
        f"- 药物：`{case_entry['drug_name']}`",
        f"- 站点候选数：`{qc['site_candidate_count']}`",
        f"- 结构支撑候选数：`{qc['site_structure_backed_count']}`",
        f"- fallback 候选数：`{qc['site_fallback_count']}`",
        f"- Top20 热点召回：`{qc['site_top20_hotspot_recall']:.3f}`",
        f"- NDCG@20：`risk={qc['site_ndcg_risk']:.3f}`，`freq={qc['site_ndcg_frequency']:.3f}`，`background-only={qc['site_ndcg_background_only']:.3f}`，`random_mean={qc['site_ndcg_random_mean']:.3f}`",
        f"- 规则字典占比：`{qc['site_rule_based_fraction']:.3f}`",
        f"- proxy 重试总数：`{qc['site_proxy_total_retries']}`",
        f"- conservation 主来源：`{qc['fitness_conservation_primary_source']}`",
        f"- GLM 审计：`model={qc['mutation_story_model']}`，`tokens={qc['mutation_story_tokens']}`，`latency={qc['mutation_story_latency_seconds']:.2f}s`，`prompt_hash={qc['mutation_story_prompt_hash'][:12]}`，`thinking={qc['mutation_story_thinking']['type']}`",
        "",
        "## Agent Summary",
        f"- 总结：{llm_payload['executive_summary']['overall_risk_pattern']}",
        f"- 排序逻辑：{llm_payload['executive_summary']['ranking_logic']}",
        f"- 覆盖说明：{llm_payload['executive_summary']['coverage_note']}",
        f"- 主局限：{llm_payload['executive_summary']['principal_caveat']}",
        "",
        "## Site Interpretations",
        *_story_site_lines(site_frame, llm_payload),
        "",
        "## Top Site Mutations",
        *_markdown_table(
            site_frame,
            [
                "mutation_rank",
                "mutation_key",
                "risk_score",
                "P_appearance",
                "delta_dock_proxy",
                "delta_ifp_proxy",
                "FitnessWeight",
                "structure_layer",
                "impact_evidence_tier",
            ],
            rename={"mutation_key": "mutation", "risk_score": "risk"},
            top_n=10,
        ),
    ]
    if not combo_frame.empty:
        lines.extend(
            [
                "",
                "## Combo Interpretations",
                *_story_combo_lines(combo_frame, llm_payload),
                "",
                "## Top Observed Combos",
                *_markdown_table(
                    combo_frame,
                    [
                        "combo_rank",
                        "combination_key",
                        "risk_score",
                        "P_appearance_combo",
                        "delta_dock_proxy",
                        "delta_ifp_proxy",
                        "FitnessWeight",
                        "impact_evidence_tier",
                    ],
                    rename={"combination_key": "combo", "risk_score": "risk"},
                    top_n=10,
                ),
            ]
        )
    lines.extend(["", "## Residue Watchlist", *_story_residue_lines(llm_payload)])
    if llm_payload.get("global_caveats"):
        lines.extend(["", "## Global Caveats", *[f"- {item}" for item in llm_payload["global_caveats"]]])
    lines.extend(
        [
            "",
            "## Audit Notes",
            f"- `site_risk` / `combo_rank` 深度审计：`site_single_component_only={qc['site_single_component_only']}`，`combo_multi_component_only={qc['combo_multi_component_only']}`，`site_prior_unchanged_from_stage2={qc['site_prior_unchanged_from_stage2']}`，`combo_projection_leakage_count={qc['combo_projection_leakage_count']}`。",
            f"- `viral` 来源审计：`viral_allowed_prior_sources_only={qc['viral_allowed_prior_sources_only']}`，`viral_no_cosmic_depmap_leakage={qc['viral_no_cosmic_depmap_leakage']}`；若无结构支撑则显式标记 `structure_unavailable_fallback`。",
            f"- `FitnessWeight` 采用 `resistagent_default_v1`，并加入 conservation-aware surrogate；当前主来源：`{qc['fitness_conservation_primary_source']}`。",
        ]
    )
    return lines


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    llm_config = config.get("llm", {})
    stage1 = config["stage1"]
    stage1_5 = config["stage1_5"]
    stage2 = config["stage2"]
    stage3 = config["stage3"]
    stage3_5 = config["stage3_5"]
    stage4 = config["stage4"]
    cases_config = load_yaml(root / stage2["cases_frozen_config"])
    cases = cases_config.get("set_d", [])
    selected_case_ids = {str(case_id) for case_id in (args.case_ids or [])}
    if selected_case_ids:
        cases = [case for case in cases if str(case.get("case_id")) in selected_case_ids]
        if not cases:
            raise SystemExit(f"No cases matched --case-id filter: {sorted(selected_case_ids)}")

    site_pool = pd.read_csv(root / stage2["site_pool_report"])
    combo_panel = pd.read_csv(root / stage2["combo_panel"])
    extraction = pd.read_csv(root / stage3["extraction_source_report"])
    master_table = pd.read_parquet(
        root / stage1["master_table"],
        columns=[
            "SAMPLE_ID",
            "UNIPROT_ID",
            "drug_name",
            "combination_size",
            "combination_key",
            "component_mutation_keys",
        ],
    )

    summary_cases: list[dict[str, Any]] = []
    outputs_root = ensure_dir(root / "outputs/stage4")
    command_text = f"{sys.executable} scripts/04c_build_mutation_rankings.py --config {args.config}"
    if selected_case_ids:
        command_text += "".join(f" --case-id {case_id}" for case_id in sorted(selected_case_ids))
    commands = [command_text]
    git_commit, git_status = detect_git_commit(root)
    software_versions = _software_versions()
    env_snapshot = _environment_snapshot()
    proposal_agent = MutationProposalAgent(
        config=MutationProposalAgentConfig(
            temperature=float(stage4.get("agent_temperature", llm_config.get("default_temperature", 0.2))),
            max_tokens=int(stage4["agent_max_tokens"]),
            thinking={"type": str(stage4["agent_thinking"])},
            top_site_n=int(stage4["agent_top_site_n"]),
            top_combo_n=int(stage4["agent_top_combo_n"]),
            top_residue_n=int(stage4["agent_top_residue_n"]),
        )
    )
    for case_entry in cases:
        case_started_at = iso_now()
        case_id = str(case_entry["case_id"])
        case_site = site_pool[site_pool["case_id"] == case_id].copy()
        case_combo = combo_panel[combo_panel["case_id"] == case_id].copy()
        case_output_root = ensure_dir(root / f"outputs/{case_id}/stage4")
        residue_map_path = root / f"outputs/{case_id}/stage3_2/residue_map.json"
        wt_complex_path = root / f"outputs/{case_id}/stage3_5/wt_complex.pdb"
        wt_ifp_path = root / f"outputs/{case_id}/stage3_5/wt_ifp.json"
        anchor_path = root / f"outputs/{case_id}/stage3_5/wt_anchor_residues.txt"
        context = build_case_context(
            case_id=case_id,
            wt_complex_path=wt_complex_path,
            residue_map_path=residue_map_path,
            anchor_path=anchor_path,
            wt_ifp_path=wt_ifp_path,
            pocket_cutoff_a=float(stage4["pocket_distance_a"]),
            second_shell_cutoff_a=float(stage4["second_shell_distance_a"]),
            root=root if str(case_entry.get("target_domain") or "").lower() == "rt" else None,
            hiv_reference_holo_pdb=str(stage3_5["hiv_reference_holo_pdb"]) if str(case_entry.get("target_domain") or "").lower() == "rt" else None,
            hiv_reference_holo_chain=str(stage3_5["hiv_reference_holo_chain"]) if str(case_entry.get("target_domain") or "").lower() == "rt" else None,
        )
        context_frame = context["context_frame"]
        context_frame.to_csv(case_output_root / "structural_context.csv", index=False)
        structural_context_summary = _structural_context_summary(context, context_frame)
        case_master = master_table[
            master_table["UNIPROT_ID"].astype(str).eq(str(case_entry["uniprot_id"]))
            & master_table["drug_name"].astype(str).eq(str(case_entry["drug_name"]))
        ].copy()
        site_sample_candidates, combo_sample_candidates = _sample_candidate_maps(case_master)
        conservation_profile = load_alphafold_position_entropy(
            uniprot_id=str(case_entry.get("uniprot_id") or ""),
            cache_root=root / stage1_5["alphafold_cache_root"],
            timeout_sec=int(stage2["request_timeout_sec"]),
        )
        empirical_entropy_lookup = empirical_position_entropy_lookup(case_master)
        conservation_profile["empirical_position_entropy"] = {
            str(position): round(value, 6) for position, value in empirical_entropy_lookup.items()
        }
        conservation_profile["numbering_system"] = context["numbering_system"]
        conservation_profile["rt_offset"] = context["rt_offset"]
        json_dump(case_output_root / "conservation_profile.json", conservation_profile)
        hiv_pose_reference = None
        if str(case_entry.get("target_domain") or "").lower() == "rt":
            hiv_pose_reference = build_hiv_pose_reference(
                root=root,
                case_entry=case_entry,
                pocket_positions=[int(value) for value in stage2["hiv_pocket_positions"]],
                reference_holo_pdb=str(stage3_5["hiv_reference_holo_pdb"]),
                reference_holo_chain=str(stage3_5["hiv_reference_holo_chain"]),
                contact_cutoff_a=float(stage3_5["hiv_pose_contact_cutoff_a"]),
            )

        mutation_status, sample_records, wt_template = _sample_status_map(case_id, root)
        extraction_case = extraction[extraction["case_id"] == case_id].copy()
        extraction_lookup = {
            str(row["sample_id"]): row
            for row in extraction_case.to_dict(orient="records")
        }
        status_lookup = {
            str(row["sample_id"]): row
            for row in mutation_status.to_dict(orient="records")
        }

        site_records = case_site.to_dict(orient="records")
        precomputed_site: list[dict[str, Any]] = []
        for row in site_records:
            components = parse_mutation_components(str(row["mutation_key"]))
            positions = _component_positions(components)
            component_context = context_frame[context_frame["target_position"].isin(positions)].copy()
            structure_layers = component_context["structure_layer"].tolist() or ["other"]
            proxy_eligible = bool(component_context["sample_proxy_eligible"].fillna(True).astype(bool).all()) if not component_context.empty else True
            p_drug_selected = _safe_float(row.get("P_drug_selected"), 0.0) or 0.0
            scope_label, candidate_in_scope, scope_multiplier = _scope_label(
                structure_layers,
                p_drug_selected,
                proxy_eligible=proxy_eligible,
            )
            resolved_sample_id = _resolve_eligible_sample_id(
                [str(row.get("representative_sample_id") or ""), *site_sample_candidates.get(str(row["mutation_key"]), [])],
                status_lookup=status_lookup,
                sample_records=sample_records,
                extraction_lookup=extraction_lookup,
            )
            precomputed_site.append(
                {
                    "components": components,
                    "positions": positions,
                    "component_context": component_context,
                    "structure_layers": structure_layers,
                    "proxy_eligible": proxy_eligible,
                    "scope_label": scope_label,
                    "candidate_in_scope": candidate_in_scope,
                    "scope_multiplier": scope_multiplier,
                    "resolved_sample_id": resolved_sample_id,
                }
            )

        combo_records = case_combo.to_dict(orient="records")
        precomputed_combo: list[dict[str, Any]] = []
        for row in combo_records:
            combo_key = str(row["combination_key"])
            components = parse_mutation_components(combo_key)
            positions = _component_positions(components)
            component_context = context_frame[context_frame["target_position"].isin(positions)].copy()
            structure_layers = component_context["structure_layer"].tolist() or ["other"]
            anchor_flags = component_context["anchor_flag"].fillna(False).astype(bool).tolist()
            relative_sasa_values = component_context["relative_sasa"].tolist()
            resolved_sample_id = _resolve_eligible_sample_id(
                [str(row.get("representative_sample_id") or ""), *combo_sample_candidates.get(combo_key, [])],
                status_lookup=status_lookup,
                sample_records=sample_records,
                extraction_lookup=extraction_lookup,
            )
            synthetic_sample_id = None
            synthetic_sample_root = None
            if len(components) > 1 and str(case_entry.get("target_domain") or "").lower() == "rt":
                synthetic_sample_id = f"combo_model_{hashlib.sha1(f'{case_id}:{combo_key}'.encode('utf-8')).hexdigest()[:12]}"
                synthetic_sample_root = case_output_root / "synthetic_combo_samples" / synthetic_sample_id
                materialize_synthetic_combo_sample(
                    combo_key=combo_key,
                    components=components,
                    sample_root=synthetic_sample_root,
                    wt_complex_path=wt_complex_path,
                    ligand_input_path=root / f"outputs/{case_id}/stage1_5/raw/ligand.sdf",
                    chain_id=str(context["chain_id"]),
                    residue_rows=list(context["residue_rows"]),
                    numbering_system=str(context["numbering_system"]),
                    rt_offset=context.get("rt_offset"),
                )
            precomputed_combo.append(
                {
                    "row": row,
                    "combo_key": combo_key,
                    "components": components,
                    "positions": positions,
                    "component_context": component_context,
                    "structure_layers": structure_layers,
                    "anchor_flags": anchor_flags,
                    "relative_sasa_values": relative_sasa_values,
                    "resolved_sample_id": resolved_sample_id,
                    "synthetic_sample_id": synthetic_sample_id,
                    "synthetic_sample_root": None if synthetic_sample_root is None else str(synthetic_sample_root),
                }
            )

        sample_jobs: dict[str, dict[str, Any]] = {}
        site_sample_job_ids = {
            str(pre.get("resolved_sample_id") or "")
            for row, pre in zip(site_records, precomputed_site)
            if pre["candidate_in_scope"] and pre.get("resolved_sample_id")
        }
        combo_observed_job_ids = {
            str(pre.get("resolved_sample_id") or "")
            for pre in precomputed_combo
            if pre.get("resolved_sample_id")
        }
        combo_synthetic_job_ids = {
            str(pre.get("synthetic_sample_id") or "")
            for pre in precomputed_combo
            if pre.get("synthetic_sample_id")
        }
        for sample_id in sorted((site_sample_job_ids | combo_observed_job_ids) - {""}):
                status_row = status_lookup.get(sample_id)
                sample_record = sample_records.get(sample_id)
                extraction_row = extraction_lookup.get(sample_id)
                if sample_record is None or status_row is None or extraction_row is None:
                    continue
                if not bool(status_row.get("eligible_for_stage5", False)):
                    continue
                if not bool(extraction_row.get("core_roles_complete", False)):
                    continue
                sample_jobs[sample_id] = {
                    "sample_id": sample_id,
                    "sample_root": str(root / stage3["structures_root"] / sample_id),
                    "cache_path": str(case_output_root / "sample_proxies" / f"{sample_id}.json"),
                    "work_root": str(case_output_root / "sample_proxies" / sample_id),
                    "sample_residue_rows": sample_record["residues"],
                    "target_numbering_system": sample_record["target_numbering_system"],
                    "rt_offset": sample_record.get("rt_numbering_offset"),
                    "case_anchor_positions": context["anchor_positions"],
                    "default_box_size_a": stage4["default_box_size_a"],
                    "ligand_box_padding_a": stage4["ligand_box_padding_a"],
                    "use_pdbfixer": stage4["use_pdbfixer"],
                    "protein_prep_ph": stage4["protein_prep_ph"],
                    "quick_docking_seeds": stage4["quick_docking_seeds"],
                    "vina_exhaustiveness": stage4["vina_exhaustiveness"],
                    "vina_num_modes": stage4["vina_num_modes"],
                    "vina_energy_range": stage4["vina_energy_range"],
                    "vina_cpu_threads": stage4["vina_cpu_threads"],
                    "proxy_retry_limit": stage4["proxy_retry_limit"],
                    "proxy_box_expansion_step_a": stage4["proxy_box_expansion_step_a"],
                    "hiv_mode": bool(hiv_pose_reference),
                    "hiv_required_pose_label": None if hiv_pose_reference is None else stage3_5["hiv_required_pose_label"],
                    "hiv_pose_contact_cutoff_a": None if hiv_pose_reference is None else stage3_5["hiv_pose_contact_cutoff_a"],
                    "nnrti_residue_coords": {} if hiv_pose_reference is None else hiv_pose_reference["nnrti_residue_coords"],
                    "active_site_residue_coords": {} if hiv_pose_reference is None else hiv_pose_reference["active_site_residue_coords"],
                    "proxy_evidence_tier_success": "paired_observed_structure",
                    "allow_complex_ifp_only_fallback": bool(hiv_pose_reference),
                    "complex_ifp_only_evidence_tier": "paired_observed_complex_ifp_only",
                }
        for pre in precomputed_combo:
            synthetic_sample_id = str(pre.get("synthetic_sample_id") or "")
            synthetic_sample_root = str(pre.get("synthetic_sample_root") or "")
            if not synthetic_sample_id or not synthetic_sample_root:
                continue
            sample_jobs[synthetic_sample_id] = {
                "sample_id": synthetic_sample_id,
                "sample_root": synthetic_sample_root,
                "cache_path": str(case_output_root / "sample_proxies" / f"{synthetic_sample_id}.json"),
                "work_root": str(case_output_root / "sample_proxies" / synthetic_sample_id),
                "sample_residue_rows": list(context["residue_rows"]),
                "target_numbering_system": str(context["numbering_system"]),
                "rt_offset": context.get("rt_offset"),
                "case_anchor_positions": context["anchor_positions"],
                "default_box_size_a": stage4["default_box_size_a"],
                "ligand_box_padding_a": stage4["ligand_box_padding_a"],
                "use_pdbfixer": stage4["use_pdbfixer"],
                "protein_prep_ph": stage4["protein_prep_ph"],
                "quick_docking_seeds": stage4["quick_docking_seeds"],
                "vina_exhaustiveness": stage4["vina_exhaustiveness"],
                "vina_num_modes": stage4["vina_num_modes"],
                "vina_energy_range": stage4["vina_energy_range"],
                "vina_cpu_threads": stage4["vina_cpu_threads"],
                "proxy_retry_limit": stage4["proxy_retry_limit"],
                "proxy_box_expansion_step_a": stage4["proxy_box_expansion_step_a"],
                "hiv_mode": bool(hiv_pose_reference),
                "hiv_required_pose_label": None if hiv_pose_reference is None else stage3_5["hiv_required_pose_label"],
                "hiv_pose_contact_cutoff_a": None if hiv_pose_reference is None else stage3_5["hiv_pose_contact_cutoff_a"],
                "nnrti_residue_coords": {} if hiv_pose_reference is None else hiv_pose_reference["nnrti_residue_coords"],
                "active_site_residue_coords": {} if hiv_pose_reference is None else hiv_pose_reference["active_site_residue_coords"],
                "proxy_evidence_tier_success": "modeled_combo_structure",
                "allow_complex_ifp_only_fallback": True,
                "complex_ifp_only_evidence_tier": "modeled_combo_structure_ifp_only",
            }

        sample_proxy_results: dict[str, dict[str, Any]] = {}
        if sample_jobs:
            with ProcessPoolExecutor(max_workers=int(stage4["max_parallel_jobs"])) as pool:
                future_map = {pool.submit(sample_proxy_worker, job): sample_id for sample_id, job in sample_jobs.items()}
                for future in as_completed(future_map):
                    sample_id = future_map[future]
                    sample_proxy_results[sample_id] = future.result()

        proxy_rows = [
            {"sample_id": sample_id, **payload}
            for sample_id, payload in sorted(sample_proxy_results.items())
        ]
        pd.DataFrame.from_records(proxy_rows).to_csv(case_output_root / "sample_proxy_features.csv", index=False)

        case_site_rows: list[dict[str, Any]] = []
        for row, pre in zip(site_records, precomputed_site):
            mutation_key = str(row["mutation_key"])
            components = pre["components"]
            positions = pre["positions"]
            component_context = pre["component_context"]
            structure_layers = pre["structure_layers"]
            anchor_flags = component_context["anchor_flag"].fillna(False).astype(bool).tolist()
            relative_sasa_values = component_context["relative_sasa"].tolist()
            known_hotspot = bool(row.get("known_hotspot", False))
            evaluation_hotspot = bool(row.get("evaluation_hotspot_label", row.get("known_hotspot", False)))
            p_background = _safe_float(row.get("P_background"), 0.0) or 0.0
            p_drug_selected = _safe_float(row.get("P_drug_selected"), 0.0) or 0.0
            if p_background <= 0.0 and str(row.get("domain_type")) == "viral":
                p_background = float(stage4["epsilon_background"])
            p_appearance = 1.0 - (1.0 - p_background) * (1.0 - p_drug_selected)
            scope_label = pre["scope_label"]
            candidate_in_scope = pre["candidate_in_scope"]
            scope_multiplier = pre["scope_multiplier"]
            rep_sample_id = str(pre.get("resolved_sample_id") or row.get("representative_sample_id") or "")
            sample_record = sample_records.get(rep_sample_id)
            sample_proxy = sample_proxy_results.get(rep_sample_id)
            local_rmsd = None
            if candidate_in_scope and sample_record is not None and sample_proxy and sample_proxy.get("proxy_status") == "ok":
                local_rmsd = local_backbone_rmsd(
                    wt_pdb=root / stage3["structures_root"] / rep_sample_id / "WT.pdb",
                    mt_pdb=root / stage3["structures_root"] / rep_sample_id / "MT.pdb",
                    sample_rows=sample_record["residues"],
                    numbering_system=str(sample_record["target_numbering_system"]),
                    rt_offset=sample_record.get("rt_numbering_offset"),
                    mutated_positions=positions,
                    radius_a=float(stage4["local_rmsd_radius_a"]),
                )
            position_entropy_values, conservation_source_counts = conservation_values_for_positions(
                positions,
                numbering_system=str(context["numbering_system"]),
                rt_offset=context.get("rt_offset"),
                alphafold_payload=conservation_profile,
                empirical_lookup=empirical_entropy_lookup,
            )
            ddg_fold = estimate_ddg_surrogate(
                components,
                relative_sasa_values,
                local_rmsd,
                position_entropy_values=position_entropy_values,
            )
            fitness_weight = fitness_weight_from_ddg(ddg_fold)
            if sample_proxy and sample_proxy.get("proxy_status") == "ok":
                delta_dock = _safe_float(sample_proxy.get("delta_dock_proxy"))
                delta_ifp = _safe_float(sample_proxy.get("ifp_jaccard_loss"))
                anchor_loss = _safe_float(sample_proxy.get("anchor_loss_fraction"))
                if delta_ifp is None:
                    delta_ifp = anchor_loss
                impact_evidence_tier = str(sample_proxy.get("proxy_evidence_tier"))
                proxy_status = str(sample_proxy.get("proxy_status"))
            else:
                delta_dock, delta_ifp = heuristic_impact_from_context(
                    components=components,
                    structure_layers=structure_layers,
                    anchor_flags=anchor_flags,
                    p_drug_selected=p_drug_selected,
                )
                anchor_loss = None
                impact_evidence_tier = "structure_unavailable_fallback"
                proxy_status = "fallback"
            if delta_ifp is None:
                _, delta_ifp = heuristic_impact_from_context(
                    components=components,
                    structure_layers=structure_layers,
                    anchor_flags=anchor_flags,
                    p_drug_selected=p_drug_selected,
                )
            uses_rule_based, prior_evidence_mode = _rule_based_mode(pd.Series(row))
            case_site_rows.append(
                {
                    **row,
                    "target_position": positions[0] if positions else None,
                    "component_count": len(components),
                    "component_positions": json.dumps(positions),
                    "structure_layer": scope_label,
                    "candidate_in_scope": bool(candidate_in_scope),
                    "scope_multiplier": float(scope_multiplier),
                    "functional_site_flag": bool(known_hotspot or any(anchor_flags) or component_context["baseline_ifp_flag"].fillna(False).astype(bool).any()),
                    "min_ligand_distance_a": _safe_float(component_context["min_ligand_distance_a"].min()) if not component_context.empty else None,
                    "solvent_accessibility": component_context["solvent_accessibility"].mode().iloc[0] if not component_context.empty else "unknown",
                    "secondary_structure_context": component_context["secondary_structure_context"].mode().iloc[0] if not component_context.empty else "loop",
                    "delta_dock_proxy": _safe_float(delta_dock, 0.0) or 0.0,
                    "delta_ifp_proxy": _safe_float(delta_ifp, 0.0) or 0.0,
                    "anchor_loss_fraction": anchor_loss,
                    "local_backbone_rmsd_a": local_rmsd,
                    "ddg_fold_surrogate": float(ddg_fold),
                    "ddg_fold_source": "surrogate_local_rmsd_physchem_sasa",
                    "fitness_formula_version": str(row.get("fitness_formula_version") or FITNESS_FORMULA_VERSION),
                    "FitnessWeight": float(fitness_weight),
                    "position_entropy_mean": _safe_float(
                        (sum(value for value in position_entropy_values if value is not None) / len([value for value in position_entropy_values if value is not None]))
                        if any(value is not None for value in position_entropy_values)
                        else None
                    ),
                    "conservation_source": summarize_conservation_source(conservation_source_counts),
                    "P_background": float(p_background),
                    "P_drug_selected": float(p_drug_selected),
                    "P_appearance": float(p_appearance),
                    "impact_evidence_tier": impact_evidence_tier,
                    "proxy_status": proxy_status,
                    "prior_evidence_mode": prior_evidence_mode,
                    "uses_rule_based_fallback": bool(uses_rule_based),
                    "evidence_sources": _list_string(
                        [
                            row.get("background_source_list"),
                            row.get("drug_selected_source_list"),
                            impact_evidence_tier,
                        ]
                    ),
                    "representative_sample_available": bool(str(row.get("representative_sample_id") or "") in sample_records),
                    "resolved_sample_id": rep_sample_id or None,
                    "resolved_sample_is_alternate": bool(rep_sample_id and rep_sample_id != str(row.get("representative_sample_id") or "")),
                    "evaluation_hotspot_label": bool(evaluation_hotspot),
                    "known_hotspot": bool(known_hotspot),
                }
            )

        site_rank = pd.DataFrame.from_records(case_site_rows)
        if not site_rank.empty:
            site_rank["dock_ranknorm"] = rank_normalize(site_rank["delta_dock_proxy"])
            site_rank["ifp_ranknorm"] = rank_normalize(site_rank["delta_ifp_proxy"])
            site_rank["impact_score"] = (site_rank["dock_ranknorm"] + 0.5 * site_rank["ifp_ranknorm"]) * site_rank["scope_multiplier"]
            site_rank["risk_score"] = site_rank["P_appearance"] * site_rank["impact_score"] * site_rank["FitnessWeight"]
            site_rank["risk_calibrated"] = (
                0.50 * rank_normalize(site_rank["risk_score"])
                + 0.25 * rank_normalize(site_rank["P_appearance"])
                + 0.15 * site_rank["dock_ranknorm"]
                + 0.10 * site_rank["ifp_ranknorm"]
            )
            site_rank["background_only_score"] = site_rank["P_background"] * site_rank["impact_score"] * site_rank["FitnessWeight"]
            site_rank["frequency_only_score"] = site_rank["P_appearance"]
            site_rank = site_rank.sort_values(
                ["risk_calibrated", "risk_score", "P_appearance", "delta_dock_proxy"],
                ascending=[False, False, False, False],
            ).reset_index(drop=True)
            site_rank["mutation_rank"] = range(1, len(site_rank) + 1)
        site_rank.to_csv(case_output_root / "mutation_rank.csv", index=False)

        combo_rows: list[dict[str, Any]] = []
        for pre in precomputed_combo:
            row = pre["row"]
            combo_key = pre["combo_key"]
            components = pre["components"]
            positions = pre["positions"]
            component_context = pre["component_context"]
            structure_layers = pre["structure_layers"]
            anchor_flags = pre["anchor_flags"]
            relative_sasa_values = pre["relative_sasa_values"]
            rep_sample_id = str(pre.get("resolved_sample_id") or "")
            synthetic_sample_id = str(pre.get("synthetic_sample_id") or "")
            observed_sample_record = sample_records.get(rep_sample_id)
            observed_sample_proxy = sample_proxy_results.get(rep_sample_id)
            synthetic_sample_proxy = sample_proxy_results.get(synthetic_sample_id)
            sample_record = observed_sample_record
            sample_proxy = observed_sample_proxy
            proxy_input_id = rep_sample_id
            used_synthetic_combo_model = False
            if not (sample_proxy and sample_proxy.get("proxy_status") == "ok") and synthetic_sample_proxy and synthetic_sample_proxy.get("proxy_status") == "ok":
                sample_record = {
                    "residues": list(context["residue_rows"]),
                    "target_numbering_system": str(context["numbering_system"]),
                    "rt_numbering_offset": context.get("rt_offset"),
                }
                sample_proxy = synthetic_sample_proxy
                proxy_input_id = synthetic_sample_id
                used_synthetic_combo_model = True
            local_rmsd = None
            if sample_record is not None and sample_proxy and sample_proxy.get("proxy_status") == "ok":
                sample_root = (
                    Path(str(pre["synthetic_sample_root"]))
                    if used_synthetic_combo_model
                    else root / stage3["structures_root"] / rep_sample_id
                )
                local_rmsd = local_backbone_rmsd(
                    wt_pdb=sample_root / "WT.pdb",
                    mt_pdb=sample_root / "MT.pdb",
                    sample_rows=sample_record["residues"],
                    numbering_system=str(sample_record["target_numbering_system"]),
                    rt_offset=sample_record.get("rt_numbering_offset"),
                    mutated_positions=positions,
                    radius_a=float(stage4["local_rmsd_radius_a"]),
                )
            position_entropy_values, conservation_source_counts = conservation_values_for_positions(
                positions,
                numbering_system=str(context["numbering_system"]),
                rt_offset=context.get("rt_offset"),
                alphafold_payload=conservation_profile,
                empirical_lookup=empirical_entropy_lookup,
            )
            ddg_fold = estimate_ddg_surrogate(
                components,
                relative_sasa_values,
                local_rmsd,
                position_entropy_values=position_entropy_values,
            )
            fitness_weight = fitness_weight_from_ddg(ddg_fold)
            if sample_proxy and sample_proxy.get("proxy_status") == "ok":
                delta_dock = _safe_float(sample_proxy.get("delta_dock_proxy"), 0.0) or 0.0
                delta_ifp = _safe_float(sample_proxy.get("ifp_jaccard_loss"))
                anchor_loss = _safe_float(sample_proxy.get("anchor_loss_fraction"))
                if delta_ifp is None:
                    delta_ifp = anchor_loss
                impact_evidence_tier = str(sample_proxy.get("proxy_evidence_tier"))
                proxy_status = str(sample_proxy.get("proxy_status"))
            else:
                delta_dock, delta_ifp = heuristic_impact_from_context(
                    components=components,
                    structure_layers=structure_layers,
                    anchor_flags=anchor_flags,
                    p_drug_selected=float(row.get("freq") or 0.0),
                )
                anchor_loss = None
                impact_evidence_tier = "structure_unavailable_fallback"
                proxy_status = "fallback"
            combo_rows.append(
                {
                    **row,
                    "component_positions": json.dumps(positions),
                    "structure_layer": _scope_label(structure_layers, float(row.get("freq") or 0.0))[0],
                    "delta_dock_proxy": float(delta_dock),
                    "delta_ifp_proxy": float(delta_ifp or 0.0),
                    "anchor_loss_fraction": anchor_loss,
                    "local_backbone_rmsd_a": local_rmsd,
                    "ddg_fold_surrogate": float(ddg_fold),
                    "ddg_fold_source": "surrogate_local_rmsd_physchem_sasa",
                    "fitness_formula_version": FITNESS_FORMULA_VERSION,
                    "FitnessWeight": float(fitness_weight),
                    "position_entropy_mean": _safe_float(
                        (sum(value for value in position_entropy_values if value is not None) / len([value for value in position_entropy_values if value is not None]))
                        if any(value is not None for value in position_entropy_values)
                        else None
                    ),
                    "conservation_source": summarize_conservation_source(conservation_source_counts),
                    "P_appearance_combo": float(row.get("freq") or 0.0),
                    "impact_evidence_tier": impact_evidence_tier,
                    "proxy_status": proxy_status,
                    "evidence_sources": _list_string([row.get("source_db"), impact_evidence_tier]),
                    "resolved_sample_id": rep_sample_id or None,
                    "resolved_sample_is_alternate": bool(rep_sample_id and rep_sample_id != str(row.get("representative_sample_id") or "")),
                    "proxy_input_id": proxy_input_id or None,
                    "used_synthetic_combo_model": bool(used_synthetic_combo_model),
                }
            )

        combo_rank = pd.DataFrame.from_records(combo_rows)
        if not combo_rank.empty:
            combo_rank["dock_ranknorm"] = rank_normalize(combo_rank["delta_dock_proxy"])
            combo_rank["ifp_ranknorm"] = rank_normalize(combo_rank["delta_ifp_proxy"])
            combo_rank["impact_score"] = combo_rank["dock_ranknorm"] + 0.5 * combo_rank["ifp_ranknorm"]
            combo_rank["risk_score"] = combo_rank["P_appearance_combo"] * combo_rank["impact_score"] * combo_rank["FitnessWeight"]
            combo_rank["risk_calibrated"] = (
                0.55 * rank_normalize(combo_rank["risk_score"])
                + 0.25 * rank_normalize(combo_rank["P_appearance_combo"])
                + 0.20 * combo_rank["ifp_ranknorm"]
            )
            combo_rank = combo_rank.sort_values(
                ["risk_calibrated", "risk_score", "P_appearance_combo", "delta_dock_proxy"],
                ascending=[False, False, False, False],
            ).reset_index(drop=True)
            combo_rank["combo_rank"] = range(1, len(combo_rank) + 1)
        combo_rank.to_csv(case_output_root / "combo_rank.csv", index=False)

        residue_risk = _residue_risk_table(site_rank)
        residue_risk.to_csv(case_output_root / "residue_risk.csv", index=False)

        site_qc = _case_rank_qc(site_rank, int(stage4["top_k"]), "evaluation_hotspot_label")
        rule_based_fraction = (
            0.0
            if site_rank.empty
            else float(site_rank["uses_rule_based_fallback"].fillna(False).astype(bool).mean())
        )
        rule_based_count = int(site_rank["uses_rule_based_fallback"].fillna(False).astype(bool).sum()) if not site_rank.empty else 0
        structure_backed_count = (
            int(site_rank["impact_evidence_tier"].fillna("").ne("structure_unavailable_fallback").sum())
            if not site_rank.empty
            else 0
        )
        fallback_count = (
            int(site_rank["impact_evidence_tier"].fillna("").eq("structure_unavailable_fallback").sum())
            if not site_rank.empty
            else 0
        )
        site_proxy_results = {
            sample_id: payload
            for sample_id, payload in sample_proxy_results.items()
            if sample_id in site_sample_job_ids
        }
        combo_proxy_results = {
            sample_id: payload
            for sample_id, payload in sample_proxy_results.items()
            if sample_id in combo_observed_job_ids or sample_id in combo_synthetic_job_ids
        }
        proxy_job_count = int(len(site_proxy_results))
        combo_proxy_job_count = int(len(combo_proxy_results))
        synthetic_combo_proxy_job_count = int(len(combo_synthetic_job_ids))
        proxy_total_retries = int(sum(int(payload.get("proxy_retry_count") or 0) for payload in site_proxy_results.values()))
        docking_failures = int(
            sum(1 for payload in site_proxy_results.values() if str(payload.get("proxy_status")) == "failed")
        )
        combo_projection_leakage_mutations = _combo_projection_leakage_mutations(site_rank)
        site_prior_unchanged = _site_priors_unchanged(site_rank, case_site)
        site_component_audit_ok = _site_component_audit(site_rank)
        combo_component_audit_ok = _combo_component_audit(combo_rank)
        viral_allowed_sources_ok = _viral_allowed_source_audit(site_rank, combo_rank)
        conservation_primary_source = (
            str(site_rank["conservation_source"].fillna("missing").value_counts().idxmax())
            if not site_rank.empty and "conservation_source" in site_rank.columns and not site_rank["conservation_source"].dropna().empty
            else str(conservation_profile.get("source") or "missing")
        )
        qc_payload = {
            "case_id": case_id,
            "generated_at": iso_now(),
            "site_candidate_count": int(len(site_rank)),
            "site_structure_backed_count": structure_backed_count,
            "site_fallback_count": fallback_count,
            "site_in_scope_candidate_count": int(site_rank["candidate_in_scope"].fillna(False).astype(bool).sum()) if not site_rank.empty else 0,
            "site_rule_based_fraction": rule_based_fraction,
            "site_rule_based_count": rule_based_count,
            "site_proxy_job_count": proxy_job_count,
            "site_proxy_total_retries": proxy_total_retries,
            "site_docking_failure_count": docking_failures,
            "site_docking_failure_rate": 0.0 if proxy_job_count == 0 else float(docking_failures / proxy_job_count),
            "site_top20_hotspot_recall": site_qc["top20_hotspot_recall"],
            "site_top20_hotspot_hits": site_qc["top20_hotspot_hits"],
            "site_ndcg_risk": site_qc["ndcg_risk"],
            "site_ndcg_frequency": site_qc["ndcg_frequency"],
            "site_ndcg_background_only": site_qc["ndcg_background_only"],
            "site_ndcg_random_mean": site_qc["ndcg_random_mean"],
            "combo_candidate_count": int(len(combo_rank)),
            "combo_structure_backed_count": int(
                combo_rank.get("impact_evidence_tier", pd.Series(dtype=str)).fillna("").ne("structure_unavailable_fallback").sum()
            ) if not combo_rank.empty else 0,
            "combo_proxy_job_count": combo_proxy_job_count,
            "combo_synthetic_proxy_job_count": synthetic_combo_proxy_job_count,
            "site_combo_split_ok": bool(
                site_component_audit_ok
                and combo_component_audit_ok
                and site_prior_unchanged
                and not combo_projection_leakage_mutations
            ),
            "site_single_component_only": bool(site_component_audit_ok),
            "combo_multi_component_only": bool(combo_component_audit_ok),
            "site_prior_unchanged_from_stage2": bool(site_prior_unchanged),
            "combo_projection_leakage_count": int(len(combo_projection_leakage_mutations)),
            "combo_projection_leakage_mutations": combo_projection_leakage_mutations,
            "viral_no_cosmic_depmap_leakage": bool(
                True
                if not site_rank.get("domain_type", pd.Series(dtype=str)).fillna("").eq("viral").any()
                else viral_allowed_sources_ok
                and not (_contains_forbidden_viral_sources(site_rank) or _contains_forbidden_viral_sources(combo_rank))
            ),
            "viral_allowed_prior_sources_only": bool(viral_allowed_sources_ok),
            "fitness_conservation_primary_source": conservation_primary_source,
            "alphafold_conservation_status": str(conservation_profile.get("source") or "missing"),
        }
        llm_input_path = case_output_root / "mutation_story_input.json"
        llm_payload_path = case_output_root / "mutation_story_payload.json"
        conservation_profile_path = case_output_root / "conservation_profile.json"
        story_path = case_output_root / "mutation_story.md"
        llm_input, llm_payload, llm_record = proposal_agent.run(
            case_entry=case_entry,
            site_rank=site_rank,
            residue_risk=residue_risk,
            combo_rank=combo_rank,
            qc_payload=qc_payload,
            structural_context_summary=structural_context_summary,
        )
        json_dump(llm_input_path, llm_input)
        json_dump(llm_payload_path, llm_payload)
        qc_payload.update(
            {
                "mutation_story_model": llm_record.model,
                "mutation_story_prompt_hash": llm_record.prompt_hash,
                "mutation_story_tokens": llm_record.tokens,
                "mutation_story_latency_seconds": float(llm_record.latency_seconds),
                "mutation_story_retry_count": int(llm_record.retry_count),
                "mutation_story_thinking": {"type": str(stage4["agent_thinking"])},
            }
        )
        json_dump(case_output_root / "stage4_qc.json", qc_payload)
        story_lines = _story_lines(case_entry, site_rank, combo_rank, qc_payload, llm_payload)
        text_dump(story_path, "\n".join(story_lines) + "\n")
        _write_case_state(
            root=root,
            case_entry=case_entry,
            args_config=args.config,
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            stage4=stage4,
            case_output_root=case_output_root,
            qc_payload=qc_payload,
            site_rank=site_rank,
            combo_rank=combo_rank,
            llm_payload=llm_payload,
            llm_record=llm_record,
            llm_input_path=llm_input_path,
            llm_payload_path=llm_payload_path,
            conservation_profile_path=conservation_profile_path,
            story_path=story_path,
            software_versions=software_versions,
            commands=commands,
        )
        _write_case_manifest(
            root=root,
            case_entry=case_entry,
            args_config=args.config,
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            stage4=stage4,
            case_output_root=case_output_root,
            started_at=case_started_at,
            software_versions=software_versions,
            env_snapshot=env_snapshot,
            git_commit=git_commit,
            git_status=git_status,
            commands=commands,
            llm_record=llm_record,
            llm_input_path=llm_input_path,
            llm_payload_path=llm_payload_path,
            conservation_profile_path=conservation_profile_path,
            story_path=story_path,
        )
        _validate_stage4_case_artifacts(
            root,
            root / f"outputs/{case_id}/state.json",
            root / f"outputs/{case_id}/run_manifest.json",
        )
        summary_cases.append(qc_payload)

    mean_latency = (
        0.0
        if not summary_cases
        else float(sum(float(case["mutation_story_latency_seconds"]) for case in summary_cases) / len(summary_cases))
    )
    total_tokens = int(sum(int(case["mutation_story_tokens"] or 0) for case in summary_cases))
    total_proxy_retries = int(sum(int(case.get("site_proxy_total_retries") or 0) for case in summary_cases))
    json_dump(
        outputs_root / "stage4_qc.json",
        {
            "generated_at": iso_now(),
            "cases": summary_cases,
            "mutation_story_model_set": sorted({str(case["mutation_story_model"]) for case in summary_cases}),
            "mutation_story_total_tokens": total_tokens,
            "mutation_story_mean_latency_seconds": mean_latency,
            "site_proxy_total_retries": total_proxy_retries,
        },
    )


if __name__ == "__main__":
    main()
