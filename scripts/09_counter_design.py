#!/usr/bin/env python3
"""counter-design step counter-design search and objective ablation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# counter-design step is a CPU/Vina-heavy search. Do not let multiprocessing workers
# implicitly create CUDA/OpenMM contexts unless a GPU run is explicitly allowed.
if os.environ.get("RESISTGPT_STAGE6_ALLOW_GPU") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OPENMM_DEFAULT_PLATFORM", "CPU")

import pandas as pd
from jsonschema import validate as jsonschema_validate
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.counter_design_agent import CounterDesignAgent, CounterDesignAgentConfig
from tools.runtime import detect_git_commit, ensure_dir, iso_now, json_dump, load_yaml, project_root
from tools.stage5_physics import gnina_score_only
from tools.stage5_utils import hashed_inputs, native_value, relative_path, stage5_for_case
from tools.stage6_postmortem import ensure_stage6_health_columns, stage6_executability_metrics
from tools.stage6_utils import (
    apply_action_to_molecule,
    candidate_id,
    canonical_smiles,
    ensure_stage6_reference_affinities,
    enumerate_candidate_actions,
    evaluate_candidate,
    evaluate_candidate_job,
    case_context_for_round,
    load_stage6_case_context,
    murcko_scaffold,
    objective_ablation_rows,
    parse_json_payload,
    rank_actions,
    read_csv_optional,
    recompute_cached_candidate_scores,
    recompute_cached_leaderboard_scores,
    render_sar_rules,
    run_stage6_dynamics_lite_probe,
    stage6_software_versions,
    top_scaffold_diversity,
    write_candidate_sdf,
    write_top_sdf,
)

OBJECTIVES = ("robust", "naive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--proposal-count", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--site-panel-top-n", type=int, default=None)
    parser.add_argument("--combo-panel-top-n", type=int, default=None)
    parser.add_argument("--max-parallel-candidates", type=int, default=None)
    parser.add_argument("--target-parallel-workers", type=int, default=None)
    parser.add_argument("--disable-llm", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--stage6-subdir", default="stage6")
    parser.add_argument("--search-seed", type=int, default=None)
    parser.add_argument("--search-rank-jitter", type=float, default=None)
    return parser.parse_args()


def selected_cases(cases_config: dict[str, object], case_id: str | None) -> list[dict[str, object]]:
    cases = list(cases_config.get("set_d", []))
    if case_id is None:
        return cases
    return [case for case in cases if str(case.get("case_id")) == str(case_id)]


def temperature_for_round(round_index: int, stage6: dict[str, Any]) -> float:
    exploration_rounds = int(stage6.get("exploration_rounds", 6))
    if round_index <= exploration_rounds:
        return float(stage6.get("exploration_temperature", 0.7))
    return float(stage6.get("exploitation_temperature", 0.25))


def load_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def objective_sequence(stage6: dict[str, Any]) -> tuple[str, ...]:
    configured = list(stage6.get("objective_execution_order") or OBJECTIVES)
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_name in configured:
        name = str(raw_name)
        if name not in OBJECTIVES:
            raise ValueError(f"Unsupported counter-design step objective in objective_execution_order: {name}")
        if name in seen:
            continue
        ordered.append(name)
        seen.add(name)
    for name in OBJECTIVES:
        if name not in seen:
            ordered.append(name)
    return tuple(ordered)


def validate_objective_plan(stage6: dict[str, Any]) -> tuple[str, ...]:
    execution_order = objective_sequence(stage6)
    objective_positions = {objective_name: index for index, objective_name in enumerate(execution_order)}
    injections = dict(stage6.get("cross_objective_seed_injection") or {})
    for target_objective, raw_config in injections.items():
        target_name = str(target_objective)
        if target_name not in OBJECTIVES:
            raise ValueError(f"Unsupported counter-design step objective in cross_objective_seed_injection: {target_name}")
        config = dict(raw_config or {})
        for raw_source in list(config.get("source_objectives") or []):
            source_name = str(raw_source)
            if source_name not in OBJECTIVES:
                raise ValueError(f"Unsupported seed source objective for {target_name}: {source_name}")
            if objective_positions[source_name] >= objective_positions[target_name]:
                raise ValueError(
                    "Cross-objective seed injection requires source objectives to run before their target: "
                    f"{source_name} -> {target_name}, current order={list(execution_order)}"
                )
    return execution_order


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
                f"RESISTGPT_STAGE6_DISABLE_LLM={os.environ.get('RESISTGPT_STAGE6_DISABLE_LLM', '')}",
            ]
        ),
    }


def write_table(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def _last_action_family(action_sequence_json: Any) -> str:
    payload = parse_json_payload(action_sequence_json) or []
    if not isinstance(payload, list) or not payload:
        return "LEAD"
    last = payload[-1]
    if not isinstance(last, dict):
        return "UNKNOWN"
    return str(last.get("edit_family") or last.get("action_label") or "UNKNOWN")


def write_case_transform_library_artifacts(
    *,
    case_stage6_root: Path,
    case_context: dict[str, Any],
    stage6: dict[str, Any],
) -> tuple[Path, Path]:
    library_csv = case_stage6_root / "case_transform_library.csv"
    summary_json = case_stage6_root / "case_transform_library_summary.json"
    rows = [
        {
            "edit_family": signature[0],
            "pattern": signature[1],
            "fragment": signature[2],
            "normalized_score": float(score),
        }
        for signature, score in sorted(
            dict(case_context.get("transform_prior") or {}).items(),
            key=lambda item: (-float(item[1]), item[0]),
        )
    ]
    write_table(pd.DataFrame.from_records(rows), library_csv)
    json_dump(
        summary_json,
        {
            "case_id": str(case_context.get("case_id") or ""),
            "transform_prior_summary": list(dict(case_context.get("action_space") or {}).get("transform_prior_summary") or []),
            "transform_prior_top": list(dict(case_context.get("action_space") or {}).get("transform_prior_top") or []),
            "partner_chain_residues": list(case_context.get("partner_chain_residues") or []),
            "partner_chain_positions": list(case_context.get("partner_chain_positions") or []),
            "pocket_profile": dict(dict(case_context.get("action_space") or {}).get("pocket_profile") or {}),
            "case_specific_action_hints": list(dict(case_context.get("action_space") or {}).get("case_specific_action_hints") or []),
            "search_regularization": {
                "proposal_transform_prior_weight": float(stage6.get("proposal_transform_prior_weight", 10.0)),
                "proposal_diversity_enabled": bool(stage6.get("proposal_diversity_enabled", True)),
                "proposal_diversity_apply_to_naive": bool(stage6.get("proposal_diversity_apply_to_naive", False)),
                "search_beam_diversity_enabled": bool(stage6.get("search_beam_diversity_enabled", True)),
                "search_beam_diversity_apply_to_naive": bool(stage6.get("search_beam_diversity_apply_to_naive", False)),
            },
        },
    )
    return library_csv, summary_json


def _normalize_signature_weights(weights: dict[tuple[str, str, str], float]) -> dict[tuple[str, str, str], float]:
    positive = {
        signature: float(value)
        for signature, value in weights.items()
        if float(value) > 0.0
    }
    total = float(sum(positive.values()))
    if total <= 0.0:
        return {}
    return {signature: float(value / total) for signature, value in positive.items()}


def _template_family_targets(templates: list[dict[str, Any]], coverage_count: int) -> list[str]:
    if coverage_count <= 0:
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for row in sorted(
        [dict(item) for item in templates if isinstance(item, dict)],
        key=lambda item: (
            int(item.get("priority") or 999),
            str(item.get("edit_family") or ""),
            str(item.get("pattern") or ""),
            str(item.get("fragment") or ""),
        ),
    ):
        family = str(row.get("edit_family") or "")
        if not family or family in seen:
            continue
        seen.add(family)
        ordered.append(family)
        if len(ordered) >= coverage_count:
            break
    return ordered


def build_dynamic_transform_prior(
    *,
    beam_frame: pd.DataFrame,
    case_context: dict[str, Any],
    stage6: dict[str, Any],
    objective_name: str,
) -> dict[tuple[str, str, str], float]:
    if beam_frame.empty or "target_scores_json" not in beam_frame.columns:
        return {}
    focus_row = beam_frame.sort_values("objective_reward", ascending=False).iloc[0]
    target_scores = parse_json_payload(focus_row.get("target_scores_json")) or {}
    if not isinstance(target_scores, dict):
        return {}
    worst_target_count = max(1, int(stage6.get("failure_context_top_n", 5)))
    ordered_targets = sorted(
        [dict(item) for item in target_scores.values() if isinstance(item, dict)],
        key=lambda row: (
            str(row.get("docking_status") or "") == "ok",
            float(native_value(row.get("score")) or -1.0),
            float(native_value(row.get("keep_ifp")) or -1.0),
        ),
    )[:worst_target_count]
    if not ordered_targets:
        return {}

    weights: dict[tuple[str, str, str], float] = {}

    def boost(edit_family: str, pattern: str, fragment: str, weight: float) -> None:
        signature = (str(edit_family), str(pattern), str(fragment))
        weights[signature] = float(weights.get(signature, 0.0) + float(weight))

    mechanism_counts: dict[str, int] = {}
    partner_sensitive_count = 0
    uncertain_count = 0
    for row in ordered_targets:
        if bool(row.get("is_partner_chain_sensitive", False)):
            partner_sensitive_count += 1
        if bool(row.get("target_uncertain", False)):
            uncertain_count += 1
        for label in list(row.get("mechanism_labels") or []):
            label_text = str(label)
            mechanism_counts[label_text] = int(mechanism_counts.get(label_text, 0) + 1)

    if int(mechanism_counts.get("anchor_loss", 0)) > 0:
        boost("SMALL_POLAR_SCAN", "small_polar_scan", "O", 3.0)
        boost("SMALL_POLAR_SCAN", "small_polar_scan", "N", 2.0)
        boost("LINKER_HETERO_SCAN", "linker_hetero_scan", "N", 2.0)
        boost("REPLACE", "replace_leaf_with_polar", "N", 1.4)
    if int(mechanism_counts.get("steric_clash", 0)) > 0:
        boost("CONSTRAINED_TAIL_TRIM", "constrained_tail_trim", "", 3.0)
        boost("DELETE", "trim_steric_bulk", "", 2.2)
        boost("HALOGEN_SCAN", "halogen_scan", "F", 1.2)
        boost("N_METHYL_SCAN", "n_methyl_scan", "-C", 1.8)
    if int(mechanism_counts.get("electrostatic_shift", 0)) > 0:
        boost("SMALL_POLAR_SCAN", "small_polar_scan", "O", 2.4)
        boost("SMALL_POLAR_SCAN", "small_polar_scan", "OC", 2.2)
        boost("SMALL_POLAR_SCAN", "small_polar_scan", "N", 1.8)
        boost("LINKER_HETERO_SCAN", "linker_hetero_scan", "N", 1.8)
        boost("RING_EDIT", "aza_scan_ring", "N", 1.2)
    if int(mechanism_counts.get("pocket_rearrangement", 0)) > 0:
        boost("LINKER_HETERO_SCAN", "linker_hetero_scan", "N", 1.4)
        boost("RING_EDIT", "aza_scan_ring", "N", 1.2)
        boost("CONSTRAINED_TAIL_TRIM", "constrained_tail_trim", "", 1.2)
    if partner_sensitive_count > 0:
        boost("SMALL_POLAR_SCAN", "small_polar_scan", "OC", 2.6)
        boost("SMALL_POLAR_SCAN", "small_polar_scan", "O", 1.8)
        boost("LINKER_HETERO_SCAN", "linker_hetero_scan", "N", 1.8)
        boost("N_METHYL_SCAN", "n_methyl_scan", "-C", 1.2)
    if uncertain_count > 0:
        boost("SMALL_POLAR_SCAN", "small_polar_scan", "O", 1.0)
        boost("LINKER_HETERO_SCAN", "linker_hetero_scan", "N", 0.8)
        boost("CONSTRAINED_TAIL_TRIM", "constrained_tail_trim", "", 0.8)

    hotspot_fraction = float(native_value(focus_row.get("hotspot_fraction")) or 0.0)
    new_nonhotspot_score = float(native_value(focus_row.get("new_nonhotspot_contact_score")) or 0.0)
    if hotspot_fraction >= float(stage6.get("dynamic_hotspot_fraction_trigger", 0.45)):
        boost("RING_EDIT", "aza_scan_ring", "N", 1.6)
        boost("SMALL_POLAR_SCAN", "small_polar_scan", "O", 1.4)
        boost("LINKER_HETERO_SCAN", "linker_hetero_scan", "N", 1.2)
    if new_nonhotspot_score <= float(stage6.get("dynamic_new_nonhotspot_min", 1.0)):
        boost("SMALL_POLAR_SCAN", "small_polar_scan", "O", 1.2)
        boost("SMALL_POLAR_SCAN", "small_polar_scan", "N", 1.0)
        boost("RING_EDIT", "aza_scan_ring", "N", 0.9)
        boost("CONSTRAINED_TAIL_TRIM", "constrained_tail_trim", "", 0.8)

    if str(objective_name) == "robust":
        boost("SMALL_POLAR_SCAN", "small_polar_scan", "O", 0.8)
        boost("LINKER_HETERO_SCAN", "linker_hetero_scan", "N", 0.7)
        boost("RING_EDIT", "aza_scan_ring", "N", 0.6)

    hints = list(dict(case_context.get("action_space") or {}).get("case_specific_action_hints") or [])
    for hint in hints[:6]:
        if not isinstance(hint, dict):
            continue
        signature = (
            str(hint.get("edit_family") or ""),
            str(hint.get("pattern") or ""),
            str(hint.get("fragment") or ""),
        )
        if not signature[0] or not signature[1]:
            continue
        hint_weight = float(hint.get("weight") or 0.0)
        weights[signature] = float(weights.get(signature, 0.0) + 0.25 * hint_weight)

    return _normalize_signature_weights(weights)


def _proposal_adjusted_rank_score(
    *,
    proposal: dict[str, Any],
    beam_scaffold_counts: Counter[str],
    family_counts: dict[str, int],
    scaffold_counts: dict[str, int],
    parent_counts: dict[str, int],
    stage6: dict[str, Any],
    diversity_active: bool,
) -> float:
    adjusted_score = float(proposal.get("base_rank_score") or 0.0)
    if not diversity_active:
        return adjusted_score
    scaffold = str(proposal.get("scaffold_smiles") or "")
    family = str(proposal.get("edit_family") or "")
    parent_candidate_id = str(proposal.get("parent_candidate_id") or "")
    beam_scaffold_penalty = float(stage6.get("proposal_beam_scaffold_repeat_penalty", 0.15))
    scaffold_penalty = float(stage6.get("proposal_scaffold_repeat_penalty", 0.25))
    family_penalty = float(stage6.get("proposal_family_repeat_penalty", 0.10))
    parent_penalty = float(stage6.get("proposal_parent_repeat_penalty", 0.08))
    scaffold_novelty_bonus = float(stage6.get("proposal_scaffold_novelty_bonus", 0.08))
    family_novelty_bonus = float(stage6.get("proposal_family_novelty_bonus", 0.04))
    adjusted_score -= beam_scaffold_penalty * float(beam_scaffold_counts.get(scaffold, 0))
    adjusted_score -= scaffold_penalty * float(scaffold_counts.get(scaffold, 0))
    adjusted_score -= family_penalty * float(family_counts.get(family, 0))
    adjusted_score -= parent_penalty * float(parent_counts.get(parent_candidate_id, 0))
    if scaffold and beam_scaffold_counts.get(scaffold, 0) == 0 and scaffold_counts.get(scaffold, 0) == 0:
        adjusted_score += scaffold_novelty_bonus
    if family and family_counts.get(family, 0) == 0:
        adjusted_score += family_novelty_bonus
    return adjusted_score


def select_diverse_proposals(
    *,
    proposals: list[dict[str, Any]],
    beam_frame: pd.DataFrame,
    templates: list[dict[str, Any]],
    stage6: dict[str, Any],
    objective_name: str,
    proposal_count: int,
) -> list[dict[str, Any]]:
    if not proposals or proposal_count <= 0:
        return []
    diversity_active = bool(stage6.get("proposal_diversity_enabled", True)) and (
        str(objective_name) == "robust" or bool(stage6.get("proposal_diversity_apply_to_naive", False))
    )
    max_per_pattern = max(1, int(stage6.get("max_proposals_per_pattern", 2)))
    max_per_family = max(1, int(stage6.get("max_proposals_per_family", 3)))
    max_per_scaffold = max(1, int(stage6.get("max_proposals_per_scaffold", 2)))
    max_per_parent = max(1, int(stage6.get("max_proposals_per_parent", 3)))
    relaxed_max_per_family = max(max_per_family, int(stage6.get("proposal_relaxed_max_per_family", max_per_family)))
    relaxed_max_per_scaffold = max(max_per_scaffold, int(stage6.get("proposal_relaxed_max_per_scaffold", max_per_scaffold)))
    relaxed_max_per_parent = max(max_per_parent, int(stage6.get("proposal_relaxed_max_per_parent", max_per_parent)))
    unique_parents = {
        str(proposal.get("parent_candidate_id") or "")
        for proposal in proposals
        if str(proposal.get("parent_candidate_id") or "")
    }
    if len(unique_parents) <= 1:
        max_per_parent = max(max_per_parent, int(proposal_count))
        relaxed_max_per_parent = max(relaxed_max_per_parent, int(proposal_count))
    beam_scaffold_counts = Counter(str(value or "") for value in beam_frame.get("scaffold_smiles", pd.Series(dtype=object)).tolist())
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    family_counts: dict[str, int] = {}
    pattern_counts: dict[str, int] = {}
    scaffold_counts: dict[str, int] = {}
    parent_counts: dict[str, int] = {}

    family_targets = _template_family_targets(
        templates,
        int(stage6.get("proposal_template_family_coverage_count", 0)),
    )

    # Reserve one slot for top-ranked template families so proposal fallback
    # does not collapse everything back to a single chemotype family.
    for target_family in family_targets:
        if len(selected) >= proposal_count:
            break
        best_index = None
        best_key = None
        for index, proposal in enumerate(proposals):
            if index in selected_indices:
                continue
            family = str(proposal.get("edit_family") or "")
            pattern = str(proposal.get("pattern") or "")
            if family != target_family:
                continue
            if pattern_counts.get(pattern, 0) >= max_per_pattern:
                continue
            adjusted_score = _proposal_adjusted_rank_score(
                proposal=proposal,
                beam_scaffold_counts=beam_scaffold_counts,
                family_counts=family_counts,
                scaffold_counts=scaffold_counts,
                parent_counts=parent_counts,
                stage6=stage6,
                diversity_active=diversity_active,
            )
            proposal["selection_rank_score"] = float(adjusted_score)
            key = (
                float(adjusted_score),
                bool(proposal.get("preview_prefilter_pass", False)),
                -_preview_total_penalty(proposal),
                -int(proposal.get("preview_warning_count") or 0),
                str(proposal.get("action_label") or ""),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        if best_index is None:
            continue
        proposal = dict(proposals[best_index])
        selected_indices.add(best_index)
        selected.append(proposal)
        family = str(proposal.get("edit_family") or "")
        pattern = str(proposal.get("pattern") or "")
        scaffold = str(proposal.get("scaffold_smiles") or "")
        parent_candidate_id = str(proposal.get("parent_candidate_id") or "")
        family_counts[family] = int(family_counts.get(family, 0) + 1)
        pattern_counts[pattern] = int(pattern_counts.get(pattern, 0) + 1)
        if scaffold:
            scaffold_counts[scaffold] = int(scaffold_counts.get(scaffold, 0) + 1)
        if parent_candidate_id:
            parent_counts[parent_candidate_id] = int(parent_counts.get(parent_candidate_id, 0) + 1)

    while len(selected) < proposal_count:
        best_index = None
        best_key = None
        for index, proposal in enumerate(proposals):
            if index in selected_indices:
                continue
            family = str(proposal.get("edit_family") or "")
            pattern = str(proposal.get("pattern") or "")
            scaffold = str(proposal.get("scaffold_smiles") or "")
            parent_candidate_id = str(proposal.get("parent_candidate_id") or "")
            if family_counts.get(family, 0) >= max_per_family:
                continue
            if pattern_counts.get(pattern, 0) >= max_per_pattern:
                continue
            if scaffold and scaffold_counts.get(scaffold, 0) >= max_per_scaffold:
                continue
            if parent_candidate_id and parent_counts.get(parent_candidate_id, 0) >= max_per_parent:
                continue
            adjusted_score = _proposal_adjusted_rank_score(
                proposal=proposal,
                beam_scaffold_counts=beam_scaffold_counts,
                family_counts=family_counts,
                scaffold_counts=scaffold_counts,
                parent_counts=parent_counts,
                stage6=stage6,
                diversity_active=diversity_active,
            )
            proposal["selection_rank_score"] = float(adjusted_score)
            key = (
                float(adjusted_score),
                bool(proposal.get("preview_prefilter_pass", False)),
                -_preview_total_penalty(proposal),
                -int(proposal.get("preview_warning_count") or 0),
                float(proposal.get("dynamic_transform_prior_score") or 0.0),
                float(proposal.get("transform_prior_score") or 0.0),
                float(proposal.get("template_priority_score") or 0.0),
                str(proposal.get("action_label") or ""),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        if best_index is None:
            break
        proposal = dict(proposals[best_index])
        selected_indices.add(best_index)
        selected.append(proposal)
        family = str(proposal.get("edit_family") or "")
        pattern = str(proposal.get("pattern") or "")
        scaffold = str(proposal.get("scaffold_smiles") or "")
        parent_candidate_id = str(proposal.get("parent_candidate_id") or "")
        family_counts[family] = int(family_counts.get(family, 0) + 1)
        pattern_counts[pattern] = int(pattern_counts.get(pattern, 0) + 1)
        if scaffold:
            scaffold_counts[scaffold] = int(scaffold_counts.get(scaffold, 0) + 1)
        if parent_candidate_id:
            parent_counts[parent_candidate_id] = int(parent_counts.get(parent_candidate_id, 0) + 1)

        if len(selected) >= proposal_count:
            return selected[:proposal_count]

    # If strict diversity caps underfill the beam, relax family/pattern caps
    # but keep scaffold/parent limits bounded so search does not collapse back
    # to one chemotype.
    while len(selected) < proposal_count:
        best_index = None
        best_key = None
        for index, proposal in enumerate(proposals):
            if index in selected_indices:
                continue
            family = str(proposal.get("edit_family") or "")
            scaffold = str(proposal.get("scaffold_smiles") or "")
            parent_candidate_id = str(proposal.get("parent_candidate_id") or "")
            if family_counts.get(family, 0) >= relaxed_max_per_family:
                continue
            if scaffold and scaffold_counts.get(scaffold, 0) >= relaxed_max_per_scaffold:
                continue
            if parent_candidate_id and parent_counts.get(parent_candidate_id, 0) >= relaxed_max_per_parent:
                continue
            adjusted_score = _proposal_adjusted_rank_score(
                proposal=proposal,
                beam_scaffold_counts=beam_scaffold_counts,
                family_counts=family_counts,
                scaffold_counts=scaffold_counts,
                parent_counts=parent_counts,
                stage6=stage6,
                diversity_active=diversity_active,
            )
            proposal["selection_rank_score"] = float(adjusted_score)
            key = (
                float(adjusted_score),
                bool(proposal.get("preview_prefilter_pass", False)),
                -_preview_total_penalty(proposal),
                -int(proposal.get("preview_warning_count") or 0),
                str(proposal.get("action_label") or ""),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        if best_index is None:
            break
        proposal = dict(proposals[best_index])
        selected_indices.add(best_index)
        selected.append(proposal)
        family = str(proposal.get("edit_family") or "")
        pattern = str(proposal.get("pattern") or "")
        scaffold = str(proposal.get("scaffold_smiles") or "")
        parent_candidate_id = str(proposal.get("parent_candidate_id") or "")
        family_counts[family] = int(family_counts.get(family, 0) + 1)
        pattern_counts[pattern] = int(pattern_counts.get(pattern, 0) + 1)
        if scaffold:
            scaffold_counts[scaffold] = int(scaffold_counts.get(scaffold, 0) + 1)
        if parent_candidate_id:
            parent_counts[parent_candidate_id] = int(parent_counts.get(parent_candidate_id, 0) + 1)
    return selected[:proposal_count]


def apply_search_beam_diversity(
    *,
    beam_source: pd.DataFrame,
    beam_width: int,
    stage6: dict[str, Any],
    objective_name: str,
) -> pd.DataFrame:
    if beam_source.empty:
        return beam_source.copy()
    ordered = beam_source.copy()
    beam_reward = pd.to_numeric(ordered.get("beam_reranked_objective_reward", pd.Series(dtype=float)), errors="coerce")
    objective_reward = pd.to_numeric(ordered.get("objective_reward", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    if beam_reward.empty:
        ordered["_beam_selection_reward"] = objective_reward
    else:
        ordered["_beam_selection_reward"] = beam_reward.fillna(objective_reward)
    ordered = ordered.sort_values(
        ["_beam_selection_reward", "objective_reward", "robust_score", "keep_ifp", "candidate_id"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    if beam_width <= 0 or len(ordered) <= beam_width:
        return ordered.head(max(beam_width, 0)).drop(columns=["_beam_selection_reward"], errors="ignore").copy().reset_index(drop=True)
    diversity_active = bool(stage6.get("search_beam_diversity_enabled", True)) and (
        str(objective_name) == "robust" or bool(stage6.get("search_beam_diversity_apply_to_naive", False))
    )
    if not diversity_active:
        return ordered.head(beam_width).drop(columns=["_beam_selection_reward"], errors="ignore").copy().reset_index(drop=True)

    lock_top_n = max(1, int(stage6.get("search_beam_diversity_lock_top_n", 1)))
    lock_top_n = min(lock_top_n, beam_width, len(ordered))
    selected_rows = [dict(row) for row in ordered.head(lock_top_n).to_dict(orient="records")]
    selected_scaffolds = Counter(str(row.get("scaffold_smiles") or "") for row in selected_rows)
    remaining_rows = [dict(row) for row in ordered.iloc[lock_top_n:].to_dict(orient="records")]
    scaffold_penalty = float(stage6.get("search_beam_scaffold_penalty", 0.12))
    novelty_bonus = float(stage6.get("search_beam_scaffold_novelty_bonus", 0.05))

    while remaining_rows and len(selected_rows) < beam_width:
        best_index = None
        best_key = None
        for index, row in enumerate(remaining_rows):
            scaffold = str(row.get("scaffold_smiles") or "")
            repeat_count = int(selected_scaffolds.get(scaffold, 0))
            base_reward = native_value(row.get("beam_reranked_objective_reward"))
            if base_reward is None:
                base_reward = native_value(row.get("objective_reward")) or 0.0
            adjusted_reward = float(base_reward) - scaffold_penalty * repeat_count
            if scaffold and repeat_count == 0:
                adjusted_reward += novelty_bonus
            key = (
                adjusted_reward,
                float(native_value(row.get("robust_score")) or 0.0),
                float(native_value(row.get("keep_ifp")) or 0.0),
                -float(native_value(row.get("dep")) or 0.0),
                str(row.get("candidate_id") or ""),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        row = remaining_rows.pop(int(best_index))
        selected_rows.append(row)
        scaffold = str(row.get("scaffold_smiles") or "")
        selected_scaffolds[scaffold] += 1

    return pd.DataFrame.from_records(selected_rows).drop(columns=["_beam_selection_reward"], errors="ignore").reset_index(drop=True)


def _preview_total_penalty(proposal: dict[str, Any]) -> float:
    total = native_value(proposal.get("preview_total_penalty"))
    if total is not None:
        return float(total)
    return float(native_value(proposal.get("preview_admet_penalty")) or 0.0)


def _dynamics_lite_config(stage6: dict[str, Any]) -> dict[str, Any]:
    payload = dict(stage6.get("dynamics_lite") or {})
    return {
        "enabled": bool(payload.get("enabled", False)),
        "every_n_rounds": max(1, int(payload.get("every_n_rounds", 4))),
        "beam_top_n": max(1, int(payload.get("beam_top_n", 5))),
        "final_top_n": max(1, int(payload.get("final_top_n", payload.get("beam_top_n", 5)))),
        "reward_weight": float(payload.get("reward_weight", 0.20)),
        "reward_center": float(payload.get("reward_center", 0.50)),
    }


def apply_dynamics_lite_rerank(
    *,
    root: Path,
    case_stage6_root: Path,
    case_context: dict[str, Any],
    stage5: dict[str, Any],
    stage6: dict[str, Any],
    robust_frame: pd.DataFrame,
    reward_column: str,
    top_n: int,
) -> pd.DataFrame:
    reranked = robust_frame.copy()
    if reranked.empty:
        return reranked
    dynamics_cfg = _dynamics_lite_config(stage6)
    if not bool(dynamics_cfg["enabled"]):
        return reranked
    if reward_column not in reranked.columns:
        reranked[reward_column] = reranked["objective_reward"] if "objective_reward" in reranked.columns else pd.NA
    optional_columns = [
        "dynamics_lite_available",
        "dynamics_lite_score",
        "dynamics_lite_contact_survival",
        "dynamics_lite_anchor_contact_survival",
        "dynamics_lite_occupancy_persistence",
        "dynamics_lite_anchor_persistence",
        "dynamics_lite_ifp_occupancy_shift_mean_abs",
        "dynamics_lite_ifp_occupancy_anchor_loss",
        "dynamics_lite_relaxation_mode",
        "dynamics_lite_local_sampling_fallback_reason",
        "dynamics_lite_fallback_penalty",
        "dynamics_lite_reranked_objective_reward",
    ]
    for column in optional_columns:
        if column not in reranked.columns:
            reranked[column] = pd.NA
    reward_weight = float(dynamics_cfg["reward_weight"])
    reward_center = float(dynamics_cfg["reward_center"])
    for index, row in reranked.head(max(1, int(top_n))).iterrows():
        candidate_id_text = str(row.get("candidate_id") or "")
        if not candidate_id_text:
            continue
        probe = run_stage6_dynamics_lite_probe(
            root=root,
            case_context=case_context,
            stage5=stage5,
            stage6=stage6,
            candidate_id_text=candidate_id_text,
        )
        reranked.at[index, "dynamics_lite_available"] = bool(probe.get("available", False))
        reranked.at[index, "dynamics_lite_relaxation_mode"] = str(probe.get("relaxation_mode") or "")
        reranked.at[index, "dynamics_lite_local_sampling_fallback_reason"] = str(
            probe.get("local_sampling_fallback_reason") or ""
        )
        reranked.at[index, "dynamics_lite_fallback_penalty"] = native_value(probe.get("fallback_penalty"))
        if not bool(probe.get("available", False)):
            continue
        reranked.at[index, "dynamics_lite_score"] = float(probe.get("score") or 0.0)
        reranked.at[index, "dynamics_lite_contact_survival"] = float(probe.get("contact_survival") or 0.0)
        reranked.at[index, "dynamics_lite_anchor_contact_survival"] = float(
            probe.get("anchor_contact_survival") or 0.0
        )
        reranked.at[index, "dynamics_lite_occupancy_persistence"] = float(
            probe.get("occupancy_persistence") or 0.0
        )
        reranked.at[index, "dynamics_lite_anchor_persistence"] = float(probe.get("anchor_persistence") or 0.0)
        reranked.at[index, "dynamics_lite_ifp_occupancy_shift_mean_abs"] = float(
            probe.get("ifp_occupancy_shift_mean_abs") or 0.0
        )
        reranked.at[index, "dynamics_lite_ifp_occupancy_anchor_loss"] = float(
            probe.get("ifp_occupancy_anchor_loss") or 0.0
        )
        base_reward = native_value(reranked.at[index, reward_column])
        if base_reward is None:
            base_reward = native_value(row.get("objective_reward")) or 0.0
        adjusted_reward = float(base_reward) + reward_weight * (float(probe.get("score") or 0.0) - reward_center)
        reranked.at[index, reward_column] = float(adjusted_reward)
        reranked.at[index, "dynamics_lite_reranked_objective_reward"] = float(adjusted_reward)
    sort_columns = [reward_column]
    ascending = [False]
    for column, direction in [("objective_reward", False), ("robust_score", False), ("dep", True), ("candidate_id", True)]:
        if column in reranked.columns:
            sort_columns.append(column)
            ascending.append(direction)
    return reranked.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def prefilter_audit_row(row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    payload = dict(row) if isinstance(row, dict) else dict(row.to_dict())
    proposal_action_label = str(payload.get("proposal_action_label") or "")
    if bool(payload.get("seed_injected", False)) and not proposal_action_label:
        source_objective = str(payload.get("seed_source_objective") or "")
        proposal_action_label = f"SEED_INJECT:{source_objective}" if source_objective else "SEED_INJECT"
    audit_row = {
        "case_id": str(payload["case_id"]),
        "objective_name": str(payload.get("objective_name") or ""),
        "round_index": int(payload.get("round_index") or 0),
        "candidate_id": str(payload["candidate_id"]),
        "parent_candidate_id": str(payload.get("parent_candidate_id") or ""),
        "smiles": str(payload["smiles"]),
        "scaffold_smiles": str(payload.get("scaffold_smiles") or ""),
        "proposal_action_label": proposal_action_label,
        "action_sequence_json": str(payload.get("action_sequence_json") or "[]"),
        "prefilter_pass": bool(payload.get("prefilter_pass", False)),
        "prefilter_fail_reason": str(payload.get("prefilter_fail_reason") or ""),
        "prefilter_fail_reasons": str(json.dumps(payload.get("prefilter_fail_reasons") or [], ensure_ascii=True)),
        "prefilter_warning_reason": str(payload.get("prefilter_warning_reason") or ""),
        "prefilter_warning_reasons": str(json.dumps(payload.get("prefilter_warning_reasons") or [], ensure_ascii=True)),
        "docking_skipped": bool(payload.get("docking_skipped", False)),
        "prefilter_score": float(payload.get("prefilter_score") or 0.0),
        "admet_penalty": float(payload.get("admet_penalty") or 0.0),
        "synthesis_penalty": float(payload.get("synthesis_penalty") or 0.0),
        "total_penalty": float(payload.get("total_penalty") or payload.get("admet_penalty") or 0.0),
        "qed": payload.get("qed"),
        "sa_score": payload.get("sa_score"),
        "ra_score": payload.get("ra_score"),
        "scscore": payload.get("scscore"),
        "retrosynthesis_plausibility": payload.get("retrosynthesis_plausibility"),
        "series_likeness_score": payload.get("series_likeness_score"),
        "series_similarity": payload.get("series_similarity"),
        "series_scaffold_similarity": payload.get("series_scaffold_similarity"),
        "series_scaffold_match": payload.get("series_scaffold_match"),
        "clogp": payload.get("clogp"),
        "hbd": payload.get("hbd"),
        "hba": payload.get("hba"),
        "tpsa": payload.get("tpsa"),
        "mw": payload.get("mw"),
        "rotatable_bonds": payload.get("rotatable_bonds"),
        "pains_alert_count": payload.get("pains_alert_count"),
        "pains_alerts": str(json.dumps(payload.get("pains_alerts") or [], ensure_ascii=True)),
        "medchem_blacklist_count": payload.get("medchem_blacklist_count"),
        "medchem_blacklist_labels": str(json.dumps(payload.get("medchem_blacklist_labels") or [], ensure_ascii=True)),
        "unstable_motif_count": payload.get("unstable_motif_count"),
        "unstable_motif_labels": str(json.dumps(payload.get("unstable_motif_labels") or [], ensure_ascii=True)),
        "candidate_valid": bool(payload.get("candidate_valid", False)),
        "objective_reward": float(payload.get("objective_reward") or 0.0),
    }
    for optional_column in ["seed_injected", "seed_source_objective", "seed_source_rank", "seed_source_round_index"]:
        if optional_column in payload:
            audit_row[optional_column] = payload.get(optional_column)
    return audit_row


def binding_score_from_affinities(candidate_affinity: float | None, lead_affinity: float | None, scale: float) -> float | None:
    if candidate_affinity is None or lead_affinity is None:
        return None
    delta = float(lead_affinity) - float(candidate_affinity)
    return float(1.0 / (1.0 + pow(2.718281828459045, -delta / max(scale, 1.0e-6))))


def merge_nested_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_nested_dicts(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def stage6_for_case(stage6: dict[str, Any], case_id: str) -> dict[str, Any]:
    overrides = dict(stage6.get("case_overrides", {}))
    case_override = dict(overrides.get(case_id, {}))
    if not case_override:
        merged = copy.deepcopy(stage6)
        merged.pop("case_overrides", None)
        return merged
    merged = merge_nested_dicts(stage6, case_override)
    merged.pop("case_overrides", None)
    return merged


def build_candidate_proposals(
    *,
    beam_frame: pd.DataFrame,
    case_context: dict[str, Any],
    stage6: dict[str, Any],
    objective_name: str,
    templates: list[dict[str, Any]],
    proposal_count: int,
    evaluated_smiles: set[str],
    lead_descriptors: dict[str, Any],
    transform_prior: dict[tuple[str, str, str], float] | None,
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    seen: set[str] = set()
    dynamic_transform_prior = build_dynamic_transform_prior(
        beam_frame=beam_frame,
        case_context=case_context,
        stage6=stage6,
        objective_name=objective_name,
    )
    transform_prior_weight = float(stage6.get("proposal_transform_prior_weight", 10.0))
    for _, row in beam_frame.sort_values("objective_reward", ascending=False).iterrows():
        parent_smiles = canonical_smiles(str(row["smiles"]))
        parent_mol = Chem.MolFromSmiles(parent_smiles)
        if parent_mol is None:
            continue
        ranked_actions = rank_actions(
            enumerate_candidate_actions(parent_mol, stage6),
            templates,
            transform_prior=transform_prior,
            search_seed=native_value(stage6.get("search_seed")),
            rank_jitter=float(stage6.get("search_rank_jitter", 0.0) or 0.0),
            jitter_salt=(
                f"{case_context.get('case_id')}|{objective_name}|"
                f"{row.get('candidate_id')}|{row.get('round_index')}"
            ),
        )
        parent_sequence = list(parse_json_payload(row.get("action_sequence_json")) or [])
        for action in ranked_actions:
            child_mol = apply_action_to_molecule(parent_mol, action)
            if child_mol is None:
                continue
            try:
                child_smiles = canonical_smiles(child_mol)
            except Exception:
                continue
            if child_smiles == parent_smiles or child_smiles in seen or child_smiles in evaluated_smiles:
                continue
            seen.add(child_smiles)
            preview = evaluate_candidate_preview(
                smiles=child_smiles,
                stage6=stage6,
                lead_descriptors=lead_descriptors,
            )
            signature = (
                str(action.get("edit_family") or ""),
                str(action.get("pattern") or ""),
                str(action.get("fragment") or ""),
            )
            static_prior_score = float(action.get("transform_prior_score") or 0.0)
            dynamic_prior_score = float(dynamic_transform_prior.get(signature, 0.0))
            base_rank_score = (
                float(action.get("template_priority_score") or 0.0)
                + transform_prior_weight * float(static_prior_score + dynamic_prior_score)
                + (1.5 if bool(preview.get("prefilter_pass", False)) else -1.0)
                - 0.5 * float(preview.get("total_penalty") or preview.get("admet_penalty") or 0.0)
                - 0.05 * int(len(preview.get("prefilter_warning_reasons") or []))
            )
            proposals.append(
                {
                    "parent_candidate_id": str(row["candidate_id"]),
                    "smiles": child_smiles,
                    "scaffold_smiles": murcko_scaffold(child_smiles),
                    "action_sequence": parent_sequence + [action],
                    "action_label": str(action.get("action_label") or ""),
                    "edit_family": str(action.get("edit_family") or ""),
                    "pattern": str(action.get("pattern") or ""),
                    "template_priority_score": float(action.get("template_priority_score") or 0.0),
                    "transform_prior_score": float(static_prior_score),
                    "dynamic_transform_prior_score": float(dynamic_prior_score),
                    "base_rank_score": float(base_rank_score),
                    "preview_prefilter_pass": bool(preview.get("prefilter_pass", False)),
                    "preview_admet_penalty": float(preview.get("admet_penalty") or 0.0),
                    "preview_synthesis_penalty": float(preview.get("synthesis_penalty") or 0.0),
                    "preview_total_penalty": float(preview.get("total_penalty") or preview.get("admet_penalty") or 0.0),
                    "preview_warning_count": int(len(preview.get("prefilter_warning_reasons") or [])),
                }
            )
    proposals.sort(
        key=lambda item: (
            -float(item.get("base_rank_score") or 0.0),
            not bool(item.get("preview_prefilter_pass", False)),
            _preview_total_penalty(item),
            int(item.get("preview_warning_count") or 0),
            -float(item.get("dynamic_transform_prior_score") or 0.0),
            -float(item.get("transform_prior_score") or 0.0),
            -float(item.get("template_priority_score") or 0.0),
            str(item.get("action_label") or ""),
        )
    )
    return select_diverse_proposals(
        proposals=proposals,
        beam_frame=beam_frame,
        templates=templates,
        stage6=stage6,
        objective_name=objective_name,
        proposal_count=proposal_count,
    )


def evaluate_candidate_preview(*, smiles: str, stage6: dict[str, Any], lead_descriptors: dict[str, Any]) -> dict[str, Any]:
    from tools.filters import apply_prefilters

    return apply_prefilters(smiles, stage6, baseline_descriptors=lead_descriptors)


def _seed_sort_columns(objective_name: str) -> tuple[list[str], list[bool]]:
    if str(objective_name) == "robust":
        return (
            ["candidate_valid", "robust_score", "robust_objective_reward", "keep_ifp", "dep", "candidate_id"],
            [False, False, False, False, True, True],
        )
    return (
        ["candidate_valid", "naive_mean_affinity", "naive_objective_reward", "keep_ifp", "dep", "candidate_id"],
        [False, False, False, False, True, True],
    )


def _seed_reward_column(objective_name: str) -> str:
    return "robust_objective_reward" if str(objective_name) == "robust" else "naive_objective_reward"


def _clone_seed_candidate_for_objective(
    row: dict[str, Any] | pd.Series,
    *,
    objective_name: str,
    case_context: dict[str, Any],
    stage6: dict[str, Any],
    source_objective: str,
    source_rank: int,
) -> dict[str, Any]:
    payload = dict(row) if isinstance(row, dict) else dict(row.to_dict())
    source_round_index = native_value(payload.get("round_index"))
    reward_column = _seed_reward_column(objective_name)
    reward_value = native_value(payload.get(reward_column))
    if reward_value is None:
        payload.update(
            recompute_cached_candidate_scores(
                payload,
                case_context=case_context,
                stage6=stage6,
            )
        )
        reward_value = native_value(payload.get(reward_column))
    if reward_value is None:
        reward_value = native_value(payload.get("objective_reward")) or 0.0
    payload["objective_name"] = str(objective_name)
    payload["objective_reward"] = float(reward_value)
    payload["round_index"] = 0
    payload["seed_injected"] = True
    payload["seed_source_objective"] = str(source_objective)
    payload["seed_source_rank"] = int(source_rank)
    payload["seed_source_round_index"] = int(source_round_index or 0)
    return payload


def build_cross_objective_seed_rows(
    *,
    objective_name: str,
    prior_histories: dict[str, pd.DataFrame],
    case_context: dict[str, Any],
    stage6: dict[str, Any],
) -> list[dict[str, Any]]:
    injection_config = dict((stage6.get("cross_objective_seed_injection") or {}).get(objective_name) or {})
    if not injection_config:
        return []
    source_objectives = [
        str(source_name)
        for source_name in list(injection_config.get("source_objectives") or [])
        if str(source_name) in OBJECTIVES and str(source_name) != str(objective_name)
    ]
    if not source_objectives:
        return []
    seed_frames: list[pd.DataFrame] = []
    for source_objective in source_objectives:
        source_frame = prior_histories.get(source_objective)
        if source_frame is None or source_frame.empty:
            continue
        tagged = source_frame.copy()
        tagged["seed_source_objective"] = source_objective
        seed_frames.append(tagged)
    if not seed_frames:
        return []
    seed_frame = pd.concat(seed_frames, ignore_index=True)
    if seed_frame.empty:
        return []
    if "candidate_id" not in seed_frame.columns:
        return []
    lead_candidate_id = str(case_context.get("lead_candidate_id") or "")
    seed_frame = seed_frame[seed_frame["candidate_id"].astype(str) != lead_candidate_id].copy()
    require_valid = bool(injection_config.get("require_valid", True))
    if require_valid and "candidate_valid" in seed_frame.columns:
        seed_frame = seed_frame[seed_frame["candidate_valid"].eq(True)].copy()
    if seed_frame.empty:
        return []
    sort_columns, ascending = _seed_sort_columns(objective_name)
    for column in sort_columns:
        if column not in seed_frame.columns:
            seed_frame[column] = pd.NA
    seed_frame["candidate_valid"] = seed_frame["candidate_valid"].eq(True)
    for column in ["robust_score", "robust_objective_reward", "naive_mean_affinity", "naive_objective_reward", "keep_ifp", "dep"]:
        if column in seed_frame.columns:
            seed_frame[column] = pd.to_numeric(seed_frame[column], errors="coerce")
    seed_frame = (
        seed_frame.sort_values(sort_columns, ascending=ascending, na_position="last")
        .drop_duplicates(subset=["candidate_id"], keep="first")
        .reset_index(drop=True)
    )
    beam_width = int(stage6.get("beam_width", 8))
    top_k = int(injection_config.get("top_k", max(1, beam_width - 1)))
    if bool(injection_config.get("preserve_lead", True)):
        top_k = min(top_k, max(0, beam_width - 1))
    if top_k <= 0:
        return []
    seed_rows: list[dict[str, Any]] = []
    for source_rank, (_, row) in enumerate(seed_frame.head(top_k).iterrows(), start=1):
        seed_rows.append(
            _clone_seed_candidate_for_objective(
                row,
                objective_name=objective_name,
                case_context=case_context,
                stage6=stage6,
                source_objective=str(row.get("seed_source_objective") or ""),
                source_rank=source_rank,
            )
        )
    return seed_rows


def build_objective_guardrails(
    *,
    objective_name: str,
    seed_rows: list[dict[str, Any]],
    stage6: dict[str, Any],
) -> dict[str, Any]:
    if str(objective_name) != "robust":
        return {}
    injection_config = dict((stage6.get("cross_objective_seed_injection") or {}).get(objective_name) or {})
    guardrail_config = dict(injection_config.get("reward_guardrails") or {})
    if not bool(guardrail_config.get("enabled", False)) or not seed_rows:
        return {}
    seed_frame = pd.DataFrame.from_records(seed_rows)
    if seed_frame.empty:
        return {}
    robust_scores = pd.to_numeric(seed_frame.get("robust_score"), errors="coerce").dropna()
    s_wt_values = pd.to_numeric(seed_frame.get("s_wt"), errors="coerce").dropna()
    if robust_scores.empty or s_wt_values.empty:
        return {}
    return {
        "enabled": True,
        "source": "cross_objective_seed_floor",
        "seed_count": int(len(seed_frame)),
        "robust_score_floor": float(robust_scores.quantile(float(guardrail_config.get("robust_score_floor_quantile", 0.0)))),
        "s_wt_floor": float(s_wt_values.quantile(float(guardrail_config.get("s_wt_floor_quantile", 0.0)))),
        "robust_score_penalty_weight": float(guardrail_config.get("robust_score_penalty_weight", 2.0)),
        "s_wt_penalty_weight": float(guardrail_config.get("s_wt_penalty_weight", 2.0)),
        "disable_compensation_below_floor": bool(guardrail_config.get("disable_compensation_below_floor", True)),
    }


def apply_seed_dep_rerank(
    *,
    robust_frame: pd.DataFrame,
    stage6: dict[str, Any],
) -> pd.DataFrame:
    reranked = robust_frame.copy()
    if reranked.empty:
        return reranked
    rerank_config = dict(stage6.get("objective_final_rerank", {}).get("robust", {}) or {})
    if not bool(rerank_config.get("enabled", False)):
        return reranked
    for column in ["dep_reranked_objective_reward", "seed_dep_ceiling", "dep_rerank_penalty"]:
        if column not in reranked.columns:
            reranked[column] = pd.NA
    if "seed_injected" not in reranked.columns:
        return reranked
    seeded = reranked[reranked["seed_injected"].eq(True)].copy()
    if seeded.empty or "dep" not in seeded.columns:
        return reranked
    dep_values = pd.to_numeric(seeded["dep"], errors="coerce").dropna()
    if dep_values.empty:
        return reranked
    dep_ceiling = float(dep_values.quantile(float(rerank_config.get("dep_ceiling_quantile", 0.5))))
    penalty_weight = float(rerank_config.get("dep_penalty_weight", 2.0))
    dep_penalty = penalty_weight * pd.to_numeric(reranked["dep"], errors="coerce").sub(dep_ceiling).clip(lower=0.0)
    base_reward = pd.to_numeric(reranked["reranked_objective_reward"], errors="coerce")
    dep_reranked = base_reward - dep_penalty
    reranked["seed_dep_ceiling"] = dep_ceiling
    reranked["dep_rerank_penalty"] = dep_penalty
    reranked["dep_reranked_objective_reward"] = dep_reranked
    has_dep_rerank = dep_reranked.notna()
    reranked.loc[has_dep_rerank, "reranked_objective_reward"] = dep_reranked[has_dep_rerank].tolist()
    reranked = reranked.sort_values(
        ["reranked_objective_reward", "robust_score", "dep", "candidate_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return reranked


def apply_dep_focus_rerank(
    *,
    robust_frame: pd.DataFrame,
    case_context: dict[str, Any],
    stage6: dict[str, Any],
) -> pd.DataFrame:
    reranked = robust_frame.copy()
    if reranked.empty:
        return reranked
    rerank_config = dict(stage6.get("objective_final_rerank", {}).get("robust", {}) or {})
    dep_focus_config = dict(rerank_config.get("dep_focus") or {})
    if not bool(dep_focus_config.get("enabled", False)):
        return reranked
    if "reranked_objective_reward" not in reranked.columns:
        reranked["reranked_objective_reward"] = reranked["objective_reward"] if "objective_reward" in reranked.columns else pd.NA
    required_numeric = [
        "dep",
        "hotspot_fraction",
        "effective_compensation_gain",
        "new_nonhotspot_score",
        "keep_ifp_nonhotspot",
    ]
    for column in required_numeric:
        if column not in reranked.columns:
            reranked[column] = pd.NA
    for column in [
        "dep_focus_reranked_objective_reward",
        "dep_focus_dep_ceiling",
        "dep_focus_hotspot_ceiling",
        "dep_focus_dep_penalty",
        "dep_focus_hotspot_penalty",
        "dep_focus_bonus",
    ]:
        if column not in reranked.columns:
            reranked[column] = pd.NA
    lock_top_n = max(0, int(dep_focus_config.get("lock_top_n", 0)))
    apply_top_n = max(lock_top_n, int(dep_focus_config.get("apply_top_n", 48)))
    if len(reranked) <= lock_top_n:
        return reranked

    dep_series = pd.to_numeric(reranked["dep"], errors="coerce")
    hotspot_series = pd.to_numeric(reranked["hotspot_fraction"], errors="coerce")
    dep_window = dep_series.head(apply_top_n).dropna()
    hotspot_window = hotspot_series.head(apply_top_n).dropna()
    if dep_window.empty:
        return reranked

    lead_candidate_id = str(case_context.get("lead_candidate_id") or "")
    lead_row = reranked[reranked["candidate_id"].astype(str).eq(lead_candidate_id)].head(1)
    lead_dep = float(pd.to_numeric(lead_row["dep"], errors="coerce").iloc[0]) if not lead_row.empty and pd.notna(pd.to_numeric(lead_row["dep"], errors="coerce").iloc[0]) else None
    lead_hotspot = (
        float(pd.to_numeric(lead_row["hotspot_fraction"], errors="coerce").iloc[0])
        if not lead_row.empty and "hotspot_fraction" in lead_row.columns and pd.notna(pd.to_numeric(lead_row["hotspot_fraction"], errors="coerce").iloc[0])
        else None
    )

    dep_ceiling = float(dep_window.quantile(float(dep_focus_config.get("dep_ceiling_quantile", 0.35))))
    if lead_dep is not None:
        dep_ceiling = min(dep_ceiling, float(lead_dep) * float(dep_focus_config.get("dep_lead_scale", 1.10)))
    hotspot_ceiling = float(hotspot_window.quantile(float(dep_focus_config.get("hotspot_fraction_ceiling_quantile", 0.50)))) if not hotspot_window.empty else 1.0
    if lead_hotspot is not None:
        hotspot_ceiling = min(
            hotspot_ceiling,
            float(lead_hotspot) * float(dep_focus_config.get("hotspot_fraction_lead_scale", 1.10)),
        )

    dep_penalty_weight = float(dep_focus_config.get("dep_penalty_weight", 1.0))
    hotspot_penalty_weight = float(dep_focus_config.get("hotspot_fraction_penalty_weight", 0.25))
    compensation_bonus_weight = float(dep_focus_config.get("compensation_bonus_weight", 0.20))
    nonhotspot_bonus_weight = float(dep_focus_config.get("nonhotspot_bonus_weight", 0.15))
    keep_ifp_nonhotspot_bonus_weight = float(dep_focus_config.get("keep_ifp_nonhotspot_bonus_weight", 0.10))

    target_index = reranked.index[lock_top_n:apply_top_n]
    target_dep = pd.to_numeric(reranked.loc[target_index, "dep"], errors="coerce")
    target_hotspot = pd.to_numeric(reranked.loc[target_index, "hotspot_fraction"], errors="coerce")
    target_comp = pd.to_numeric(reranked.loc[target_index, "effective_compensation_gain"], errors="coerce").fillna(0.0)
    target_nonhotspot = pd.to_numeric(reranked.loc[target_index, "new_nonhotspot_score"], errors="coerce").fillna(0.0)
    target_keep_nonhotspot = pd.to_numeric(reranked.loc[target_index, "keep_ifp_nonhotspot"], errors="coerce").fillna(0.0)
    dep_penalty = dep_penalty_weight * target_dep.sub(dep_ceiling).clip(lower=0.0)
    hotspot_penalty = hotspot_penalty_weight * target_hotspot.sub(hotspot_ceiling).clip(lower=0.0)
    dep_bonus = (
        compensation_bonus_weight * target_comp
        + nonhotspot_bonus_weight * target_nonhotspot
        + keep_ifp_nonhotspot_bonus_weight * target_keep_nonhotspot
    )
    base_reward = pd.to_numeric(reranked.loc[target_index, "reranked_objective_reward"], errors="coerce")
    dep_focus_reward = base_reward - dep_penalty.fillna(0.0) - hotspot_penalty.fillna(0.0) + dep_bonus.fillna(0.0)
    reranked.loc[target_index, "dep_focus_dep_ceiling"] = float(dep_ceiling)
    reranked.loc[target_index, "dep_focus_hotspot_ceiling"] = float(hotspot_ceiling)
    reranked.loc[target_index, "dep_focus_dep_penalty"] = dep_penalty.tolist()
    reranked.loc[target_index, "dep_focus_hotspot_penalty"] = hotspot_penalty.tolist()
    reranked.loc[target_index, "dep_focus_bonus"] = dep_bonus.tolist()
    reranked.loc[target_index, "dep_focus_reranked_objective_reward"] = dep_focus_reward.tolist()
    valid_dep_focus = dep_focus_reward.notna()
    reranked.loc[target_index[valid_dep_focus], "reranked_objective_reward"] = dep_focus_reward[valid_dep_focus].tolist()
    prefix = reranked.head(lock_top_n).copy()
    target_tail = reranked.iloc[lock_top_n:apply_top_n].copy()
    suffix = reranked.iloc[apply_top_n:].copy()
    if not target_tail.empty:
        target_tail = target_tail.sort_values(
            ["reranked_objective_reward", "robust_score", "dep", "candidate_id"],
            ascending=[False, False, True, True],
        )
    return pd.concat([prefix, target_tail, suffix], ignore_index=True)


def apply_scaffold_diversity_tail_rerank(
    *,
    robust_frame: pd.DataFrame,
    stage6: dict[str, Any],
) -> pd.DataFrame:
    reranked = robust_frame.copy()
    if reranked.empty:
        return reranked
    rerank_config = dict(stage6.get("objective_final_rerank", {}).get("robust", {}) or {})
    diversity_config = dict(rerank_config.get("scaffold_tail_diversity") or {})
    if not bool(diversity_config.get("enabled", False)):
        return reranked
    if "reranked_objective_reward" not in reranked.columns:
        return reranked
    lock_top_n = max(0, int(diversity_config.get("lock_top_n", 20)))
    if len(reranked) <= lock_top_n:
        return reranked
    penalty_weight = float(diversity_config.get("penalty_weight", 0.2))
    prefix = reranked.head(lock_top_n).copy().reset_index(drop=True)
    tail = reranked.iloc[lock_top_n:].copy().reset_index(drop=True)
    for column in [
        "scaffold_diversity_repeat_count",
        "scaffold_diversity_rank",
        "scaffold_diversity_reranked_objective_reward",
    ]:
        if column not in reranked.columns:
            reranked[column] = pd.NA
    selected_scaffolds = [str(value or "") for value in prefix.get("scaffold_smiles", pd.Series(dtype=object)).tolist()]
    selected_rows: list[dict[str, Any]] = [dict(row) for row in prefix.to_dict(orient="records")]
    tail_rows = [dict(row) for row in tail.to_dict(orient="records")]
    tail_anchor = float(pd.to_numeric(prefix["reranked_objective_reward"], errors="coerce").min()) if not prefix.empty else 0.0
    tail_rank = 0
    while tail_rows:
        best_index = None
        best_key = None
        for index, row in enumerate(tail_rows):
            scaffold = str(row.get("scaffold_smiles") or "")
            repeat_count = int(selected_scaffolds.count(scaffold))
            base_reward = float(native_value(row.get("reranked_objective_reward")) or 0.0)
            novelty_score = float(base_reward - penalty_weight * repeat_count)
            key = (
                novelty_score,
                float(native_value(row.get("robust_score")) or 0.0),
                -float(native_value(row.get("dep")) or 0.0),
                -repeat_count,
                str(row.get("candidate_id") or ""),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        row = tail_rows.pop(int(best_index))
        scaffold = str(row.get("scaffold_smiles") or "")
        repeat_count = int(selected_scaffolds.count(scaffold))
        tail_rank += 1
        row["scaffold_diversity_repeat_count"] = repeat_count
        row["scaffold_diversity_rank"] = int(lock_top_n + tail_rank)
        row["scaffold_diversity_reranked_objective_reward"] = float(tail_anchor - tail_rank * 1.0e-6)
        row["reranked_objective_reward"] = float(row["scaffold_diversity_reranked_objective_reward"])
        selected_rows.append(row)
        selected_scaffolds.append(scaffold)
    selected_frame = pd.DataFrame.from_records(selected_rows)
    if not prefix.empty:
        selected_frame.loc[: lock_top_n - 1, "scaffold_diversity_rank"] = range(1, lock_top_n + 1)
        selected_frame.loc[: lock_top_n - 1, "scaffold_diversity_repeat_count"] = pd.NA
        selected_frame.loc[: lock_top_n - 1, "scaffold_diversity_reranked_objective_reward"] = selected_frame.loc[
            : lock_top_n - 1, "reranked_objective_reward"
        ]
    selected_frame = selected_frame.sort_values(
        ["scaffold_diversity_rank", "reranked_objective_reward", "candidate_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return selected_frame


def rerank_uncertainty_heavy_candidates(
    *,
    root: Path,
    case_stage6_root: Path,
    case_context: dict[str, Any],
    stage5: dict[str, Any] | None = None,
    stage6: dict[str, Any],
    robust_frame: pd.DataFrame,
) -> pd.DataFrame:
    reranked = robust_frame.copy()
    if "wt_gnina_affinity_kcal_mol" not in reranked.columns:
        reranked["wt_gnina_affinity_kcal_mol"] = pd.NA
    if "wt_gnina_score" not in reranked.columns:
        reranked["wt_gnina_score"] = pd.NA
    if "reranked_objective_reward" not in reranked.columns:
        reranked["reranked_objective_reward"] = reranked["objective_reward"] if "objective_reward" in reranked.columns else pd.NA
    if reranked.empty:
        return reranked
    if bool((case_context.get("scoring_policy") or {}).get("uncertainty_heavy", False)):
        top_n = int(stage6.get("uncertainty_heavy_rerank_top_n", 12))
        weight = float(stage6.get("uncertainty_heavy_gnina_weight", 0.15))
        lead_summary_path = case_stage6_root / "cache" / "candidates" / str(case_context["lead_candidate_id"]) / "wt" / "summary.json"
        if lead_summary_path.exists():
            lead_summary = json.loads(lead_summary_path.read_text(encoding="utf-8"))
            lead_receptor = root / str(lead_summary.get("receptor_pdb") or "")
            lead_pose = root / str(lead_summary.get("pose_sdf") or "")
            if lead_receptor.exists() and lead_pose.exists():
                lead_work_root = ensure_dir(case_stage6_root / "rerank_gnina" / str(case_context["lead_candidate_id"]))
                lead_gnina = gnina_score_only(lead_receptor, lead_pose, lead_work_root)
                lead_affinity = lead_gnina.get("affinity_kcal_mol")
                if lead_affinity is not None:
                    for index, row in reranked.head(top_n).iterrows():
                        candidate_id_text = str(row["candidate_id"])
                        summary_path = case_stage6_root / "cache" / "candidates" / candidate_id_text / "wt" / "summary.json"
                        if not summary_path.exists():
                            continue
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        receptor_pdb = root / str(summary.get("receptor_pdb") or "")
                        pose_sdf = root / str(summary.get("pose_sdf") or "")
                        if not receptor_pdb.exists() or not pose_sdf.exists():
                            continue
                        gnina_payload = gnina_score_only(
                            receptor_pdb,
                            pose_sdf,
                            ensure_dir(case_stage6_root / "rerank_gnina" / candidate_id_text),
                        )
                        candidate_affinity = gnina_payload.get("affinity_kcal_mol")
                        gnina_score = binding_score_from_affinities(
                            candidate_affinity,
                            lead_affinity,
                            float(stage6.get("binding_score_scale_kcal_mol", 1.5)),
                        )
                        reranked.at[index, "wt_gnina_affinity_kcal_mol"] = candidate_affinity
                        reranked.at[index, "wt_gnina_score"] = gnina_score
                        if gnina_score is not None:
                            reranked.at[index, "reranked_objective_reward"] = float(row["objective_reward"]) + weight * float(gnina_score)
    reranked = apply_dynamics_lite_rerank(
        root=root,
        case_stage6_root=case_stage6_root,
        case_context=case_context,
        stage5=dict(stage5 or {}),
        stage6=stage6,
        robust_frame=reranked,
        reward_column="reranked_objective_reward",
        top_n=_dynamics_lite_config(stage6)["final_top_n"],
    )
    reranked = apply_dep_focus_rerank(
        robust_frame=reranked,
        case_context=case_context,
        stage6=stage6,
    )
    reranked = apply_seed_dep_rerank(
        robust_frame=reranked,
        stage6=stage6,
    )
    reranked = apply_scaffold_diversity_tail_rerank(
        robust_frame=reranked,
        stage6=stage6,
    )
    for column in ["robust_score", "dep"]:
        if column not in reranked.columns:
            reranked[column] = pd.NA
    reranked = reranked.sort_values(
        ["reranked_objective_reward", "robust_score", "dep", "candidate_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return reranked


def merge_optional_rerank_columns(
    *,
    leaderboard: pd.DataFrame,
    robust_frame: pd.DataFrame,
) -> pd.DataFrame:
    merged = leaderboard.copy()
    optional_columns = [
        "reranked_objective_reward",
        "wt_gnina_affinity_kcal_mol",
        "wt_gnina_score",
        "dynamics_lite_available",
        "dynamics_lite_score",
        "dynamics_lite_contact_survival",
        "dynamics_lite_anchor_contact_survival",
        "dynamics_lite_occupancy_persistence",
        "dynamics_lite_anchor_persistence",
        "dynamics_lite_ifp_occupancy_shift_mean_abs",
        "dynamics_lite_ifp_occupancy_anchor_loss",
        "dynamics_lite_relaxation_mode",
        "dynamics_lite_local_sampling_fallback_reason",
        "dynamics_lite_fallback_penalty",
        "dynamics_lite_reranked_objective_reward",
        "dep_focus_reranked_objective_reward",
        "dep_focus_dep_ceiling",
        "dep_focus_hotspot_ceiling",
        "dep_focus_dep_penalty",
        "dep_focus_hotspot_penalty",
        "dep_focus_bonus",
        "dep_reranked_objective_reward",
        "seed_dep_ceiling",
        "dep_rerank_penalty",
        "scaffold_diversity_repeat_count",
        "scaffold_diversity_rank",
        "scaffold_diversity_reranked_objective_reward",
    ]
    for column in optional_columns:
        if column not in merged.columns:
            merged[column] = pd.NA
    if robust_frame.empty:
        return merged
    rerank_lookup = robust_frame.set_index("candidate_id").copy()
    for column in optional_columns:
        if column not in rerank_lookup.columns:
            rerank_lookup[column] = pd.NA
    robust_mask = merged["objective_name"].eq("robust")
    for column in optional_columns:
        merged.loc[robust_mask, column] = merged.loc[robust_mask, "candidate_id"].map(rerank_lookup[column].to_dict())
    reranked_values = pd.to_numeric(merged.loc[robust_mask, "reranked_objective_reward"], errors="coerce")
    has_rerank = reranked_values.notna()
    merged.loc[robust_mask & has_rerank, "objective_reward"] = reranked_values[has_rerank].tolist()
    merged.loc[robust_mask & has_rerank, "robust_objective_reward"] = reranked_values[has_rerank].tolist()
    return merged


def evaluate_proposals(
    *,
    root: Path,
    case_context: dict[str, Any],
    stage6: dict[str, Any],
    objective_name: str,
    round_index: int,
    proposals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not proposals:
        return []
    max_workers = max(1, min(int(stage6.get("max_parallel_candidates", 4)), len(proposals)))
    jobs = [
        {
            "root": str(root),
            "case_context": case_context,
            "stage6": stage6,
            "smiles": str(proposal["smiles"]),
            "action_sequence": list(proposal["action_sequence"]),
            "round_index": int(round_index),
            "objective_name": str(objective_name),
        }
        for proposal in proposals
    ]
    rows: list[dict[str, Any]] = []
    if max_workers == 1:
        for job in jobs:
            rows.append(evaluate_candidate_job(job))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(evaluate_candidate_job, job): job for job in jobs}
            for future in as_completed(future_map):
                rows.append(future.result())
    proposal_lookup = {str(proposal["smiles"]): proposal for proposal in proposals}
    for row in rows:
        proposal = proposal_lookup.get(str(row["smiles"]))
        row["parent_candidate_id"] = str(proposal["parent_candidate_id"]) if proposal else ""
        row["proposal_action_label"] = str(proposal["action_label"]) if proposal else ""
    return rows


def run_objective_search(
    *,
    root: Path,
    case_entry: dict[str, Any],
    case_context: dict[str, Any],
    stage5: dict[str, Any],
    stage6: dict[str, Any],
    objective_name: str,
    agent: CounterDesignAgent,
    disable_llm: bool,
    case_stage6_root: Path,
    seed_rows: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    llm_records: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    prefilter_rows: list[dict[str, Any]] = []
    seed_rows = list(seed_rows or [])
    objective_case_context = dict(case_context)
    objective_case_context["objective_guardrails"] = build_objective_guardrails(
        objective_name=objective_name,
        seed_rows=seed_rows,
        stage6=stage6,
    )
    lead_case_context = case_context_for_round(
        case_context=objective_case_context,
        stage6=stage6,
        round_index=1,
    )

    lead_row = evaluate_candidate(
        root=root,
        case_context=lead_case_context,
        stage6=stage6,
        smiles=str(case_context["lead_smiles"]),
        action_sequence=[],
        round_index=0,
        objective_name=objective_name,
    )
    history_rows.append(lead_row)
    prefilter_rows.append(prefilter_audit_row(lead_row))
    evaluated: dict[str, dict[str, Any]] = {str(lead_row["smiles"]): lead_row}
    for seed_row in seed_rows:
        seed_smiles = str(seed_row.get("smiles") or "")
        if not seed_smiles or seed_smiles in evaluated:
            continue
        history_rows.append(seed_row)
        prefilter_rows.append(prefilter_audit_row(seed_row))
        evaluated[seed_smiles] = seed_row
    beam_source = pd.DataFrame.from_records(list(evaluated.values()))
    beam_frame = apply_search_beam_diversity(
        beam_source=beam_source,
        beam_width=int(stage6.get("beam_width", 8)),
        stage6=stage6,
        objective_name=objective_name,
    )
    best_score = float(lead_row.get("objective_reward") or 0.0)
    if not beam_frame.empty:
        best_score = float(beam_frame.iloc[0]["objective_reward"])
    stagnation_rounds = 0
    exploration_round_used = False
    max_rounds = int(stage6.get("max_rounds", 12))
    proposal_count = int(stage6.get("proposal_count", 32))
    beam_width = int(stage6.get("beam_width", 8))
    llm_root = ensure_dir(case_stage6_root / "llm" / objective_name)
    previous_beam_size = int(len(beam_frame))

    for round_index in range(1, max_rounds + 1):
        round_case_context = case_context_for_round(
            case_context=objective_case_context,
            stage6=stage6,
            round_index=round_index,
        )
        if stagnation_rounds >= int(stage6.get("early_stop_rounds", 3)):
            if exploration_round_used:
                break
            exploration_round_used = True
            current_temperature = float(stage6.get("exploration_temperature", 0.7))
        else:
            current_temperature = temperature_for_round(round_index, stage6)

        prompt_input, payload, llm_record = agent.run(
            case_entry=case_entry,
            objective_name=objective_name,
            round_index=round_index,
            mechanism_summary=dict(round_case_context["mechanism_summary"]),
            hotspot_residues=list(round_case_context["hotspot_residues"]),
            beam_frame=beam_frame,
            action_space=dict(round_case_context["action_space"]),
            qc_payload={
                "beam_size": int(previous_beam_size),
                "best_reward": float(best_score),
                "proposal_count": int(proposal_count),
                "panel_curriculum": dict(round_case_context.get("panel_curriculum") or {}),
            },
            lead_descriptors=dict(case_context.get("lead_descriptors") or {}),
            temperature=current_temperature,
            disable_llm=disable_llm,
        )
        prompt_path = llm_root / f"round_{round_index:02d}_input.json"
        payload_path = llm_root / f"round_{round_index:02d}_payload.json"
        json_dump(prompt_path, prompt_input)
        json_dump(payload_path, payload)
        if llm_record is not None:
            llm_records.append(
                {
                    "objective_name": objective_name,
                    "round_index": int(round_index),
                    "prompt_input": relative_path(prompt_path, root),
                    "prompt_payload": relative_path(payload_path, root),
                    "model": llm_record.model,
                    "prompt_hash": llm_record.prompt_hash,
                    "tokens": llm_record.tokens,
                    "latency_seconds": float(llm_record.latency_seconds),
                    "retry_count": int(llm_record.retry_count),
                    "thinking": {"type": str(stage6["agent_thinking"])},
                }
            )
        proposals = build_candidate_proposals(
            beam_frame=beam_frame,
            case_context=case_context,
            stage6=stage6,
            objective_name=objective_name,
            templates=list(payload.get("action_templates") or []),
            proposal_count=proposal_count,
            evaluated_smiles=set(evaluated),
            lead_descriptors=dict(case_context.get("lead_descriptors") or {}),
            transform_prior=dict(case_context.get("transform_prior") or {}),
        )
        if not proposals:
            break
        evaluated_rows = evaluate_proposals(
            root=root,
            case_context=round_case_context,
            stage6=stage6,
            objective_name=objective_name,
            round_index=round_index,
            proposals=proposals,
        )
        for row in evaluated_rows:
            evaluated[str(row["smiles"])] = row
            history_rows.append(row)
            prefilter_rows.append(prefilter_audit_row(row))

        beam_source = pd.DataFrame.from_records(list(evaluated.values()))
        if str(objective_name) == "robust":
            dynamics_cfg = _dynamics_lite_config(stage6)
            if bool(dynamics_cfg["enabled"]) and int(round_index) % int(dynamics_cfg["every_n_rounds"]) == 0:
                beam_source = apply_dynamics_lite_rerank(
                    root=root,
                    case_stage6_root=case_stage6_root,
                    case_context=round_case_context,
                    stage5=stage5,
                    stage6=stage6,
                    robust_frame=beam_source,
                    reward_column="beam_reranked_objective_reward",
                    top_n=int(dynamics_cfg["beam_top_n"]),
                )
        beam_frame = apply_search_beam_diversity(
            beam_source=beam_source,
            beam_width=beam_width,
            stage6=stage6,
            objective_name=objective_name,
        )
        previous_beam_size = int(len(beam_frame))
        current_best = float(beam_frame.iloc[0]["objective_reward"]) if not beam_frame.empty else 0.0
        if current_best > best_score + 1e-6:
            best_score = current_best
            stagnation_rounds = 0
            exploration_round_used = False
        else:
            stagnation_rounds += 1

    history_frame = pd.DataFrame.from_records(history_rows)
    if history_frame.empty:
        return pd.DataFrame(), pd.DataFrame(), llm_records
    history_frame = history_frame.sort_values(
        ["objective_reward", "robust_score", "keep_ifp", "candidate_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    history_frame["objective_rank"] = range(1, len(history_frame) + 1)
    prefilter_frame = pd.DataFrame.from_records(prefilter_rows)
    return history_frame, prefilter_frame, llm_records


def write_case_state(
    *,
    root: Path,
    case_entry: dict[str, Any],
    args_config: str,
    stage6: dict[str, Any],
    case_stage6_root: Path,
    leaderboard_path: Path,
    designed_sdf_path: Path,
    prefilter_path: Path,
    objective_ablation_path: Path,
    sar_rules_path: Path,
    qc_payload: dict[str, Any],
    llm_decisions: list[dict[str, Any]],
    software_versions: dict[str, Any],
    commands: list[str],
) -> None:
    case_id = str(case_entry["case_id"])
    state_path = root / f"outputs/{case_id}/state.json"
    state = load_json_or_empty(state_path)
    state.update(
        {
            "project_id": case_id,
            "stage": "stage6",
            "inputs": {
                "config": args_config,
                "cases": load_yaml(root / args_config)["stage2"]["cases_frozen_config"],
                "wt": {
                    "complex_pdb": relative_path(root / f"outputs/{case_id}/stage3_5/wt_complex.pdb", root),
                    "ifp": relative_path(root / f"outputs/{case_id}/stage3_5/wt_ifp.json", root),
                },
                "mutations": {
                    "topk_rank_csv": relative_path(root / f"outputs/{case_id}/stage4/mutation_rank.csv", root),
                    "combo_rank_csv": relative_path(root / f"outputs/{case_id}/stage4/combo_rank.csv", root),
                },
                "effects": {
                    "json": relative_path(root / f"outputs/{case_id}/stage5/mutation_effects.json", root),
                    "ifp_diff_csv": relative_path(root / f"outputs/{case_id}/stage5/ifp_diff.csv", root),
                },
                "priors": {
                    "global": "outputs/tables/global_prior.parquet",
                    "benchmark_blind": "outputs/tables/benchmark_prior_blind.parquet",
                    "fitness_formula_version": str(stage6.get("fitness_formula_version", "resistagent_default_v1")),
                },
            },
            "artifacts": {
                "stage6": {
                    "leaderboard_csv": relative_path(leaderboard_path, root),
                    "designed_top200_sdf": relative_path(designed_sdf_path, root),
                    "design_prefilter_audit_csv": relative_path(prefilter_path, root),
                    "objective_ablation_csv": relative_path(objective_ablation_path, root),
                    "robust_sar_rules_md": relative_path(sar_rules_path, root),
                    "stage6_qc_json": relative_path(case_stage6_root / "stage6_qc.json", root),
                    "search_trajectory_csv": relative_path(case_stage6_root / "search_trajectory.csv", root),
                }
            },
            "qc": {"stage6": qc_payload},
            "software_versions": software_versions,
            "seeds": {"python": 42},
            "commands": commands,
            "llm_decisions": llm_decisions,
        }
    )
    json_dump(state_path, state)


def write_case_manifest(
    *,
    root: Path,
    case_entry: dict[str, Any],
    started_at: str,
    software_versions: dict[str, Any],
    commands: list[str],
    inputs: list[Path],
    outputs: list[Path],
) -> None:
    case_id = str(case_entry["case_id"])
    manifest_path = root / f"outputs/{case_id}/run_manifest.json"
    manifest = load_json_or_empty(manifest_path)
    stage_runs = manifest.get("stage_runs") if isinstance(manifest.get("stage_runs"), dict) else {}
    stage_runs["stage6"] = {
        "started_at": started_at,
        "finished_at": iso_now(),
        "outputs": [relative_path(path, root) for path in outputs],
    }
    input_hashes = manifest.get("input_hashes") if isinstance(manifest.get("input_hashes"), dict) else {}
    input_hashes.update(hashed_inputs(inputs, root))
    git_commit, git_status = detect_git_commit(root)
    manifest.update(
        {
            "project_id": case_id,
            "stage": "stage6",
            "git_commit": git_commit,
            "git_status": git_status,
            "software_versions": software_versions,
            "env_snapshot": load_env_snapshot(manifest),
            "random_seeds": {"python": 42},
            "input_hashes": input_hashes,
            "commands": commands,
            "started_at": manifest.get("started_at", started_at),
            "finished_at": iso_now(),
            "stage_runs": stage_runs,
        }
    )
    json_dump(manifest_path, manifest)


def write_auxiliary_stage6_manifest(
    *,
    root: Path,
    case_entry: dict[str, Any],
    case_stage6_root: Path,
    started_at: str,
    software_versions: dict[str, Any],
    commands: list[str],
    qc_payload: dict[str, Any],
    stage6: dict[str, Any],
    outputs: list[Path],
) -> None:
    manifest_path = case_stage6_root / "run_manifest.json"
    git_commit, git_status = detect_git_commit(root)
    json_dump(
        manifest_path,
        {
            "project_id": str(case_entry["case_id"]),
            "stage": "stage6",
            "stage6_subdir": case_stage6_root.name,
            "auxiliary_run": True,
            "started_at": started_at,
            "finished_at": iso_now(),
            "git_commit": git_commit,
            "git_status": git_status,
            "software_versions": software_versions,
            "commands": commands,
            "random_seeds": {
                "python": 42,
                "search_seed": native_value(stage6.get("search_seed")),
            },
            "search_rank_jitter": native_value(stage6.get("search_rank_jitter")),
            "qc": qc_payload,
            "outputs": [relative_path(path, root) for path in outputs],
        },
    )


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
    stage5 = dict(config["stage5"])
    stage6 = dict(config.get("stage6", {}))
    if args.max_rounds is not None:
        stage6["max_rounds"] = int(args.max_rounds)
    if args.proposal_count is not None:
        stage6["proposal_count"] = int(args.proposal_count)
    if args.beam_width is not None:
        stage6["beam_width"] = int(args.beam_width)
    if args.site_panel_top_n is not None:
        stage6["site_panel_top_n"] = int(args.site_panel_top_n)
    if args.combo_panel_top_n is not None:
        stage6["combo_panel_top_n"] = int(args.combo_panel_top_n)
    if args.max_parallel_candidates is not None:
        stage6["max_parallel_candidates"] = int(args.max_parallel_candidates)
    if args.target_parallel_workers is not None:
        stage6["target_parallel_workers"] = int(args.target_parallel_workers)
    if args.search_seed is not None:
        stage6["search_seed"] = int(args.search_seed)
    if args.search_rank_jitter is not None:
        stage6["search_rank_jitter"] = float(args.search_rank_jitter)

    cases_config = load_yaml(root / config["stage2"]["cases_frozen_config"])
    cases = selected_cases(cases_config, args.case_id)
    if not cases:
        raise SystemExit(f"No counter-design step case matched --case-id={args.case_id}")

    software_versions = stage6_software_versions()
    commands = [
        f"python scripts/09_counter_design.py --config {args.config}"
        + (f" --case-id {args.case_id}" if args.case_id else "")
        + (f" --stage6-subdir {args.stage6_subdir}" if str(args.stage6_subdir) != "stage6" else "")
        + (f" --search-seed {args.search_seed}" if args.search_seed is not None else ""),
    ]
    disable_llm = bool(args.disable_llm or os.environ.get("RESISTGPT_STAGE6_DISABLE_LLM") == "1")
    case_qc_rows: list[dict[str, Any]] = []
    for case_entry in cases:
        started_at = iso_now()
        case_id = str(case_entry["case_id"])
        case_stage5 = stage5_for_case(stage5, case_id)
        case_stage6 = stage6_for_case(stage6, case_id)
        case_stage6["output_dirname"] = str(args.stage6_subdir)
        agent = CounterDesignAgent(
            config=CounterDesignAgentConfig(
                temperature=float(case_stage6.get("exploration_temperature", 0.7)),
                max_tokens=int(case_stage6.get("agent_max_tokens", 2600)),
                thinking={"type": str(case_stage6.get("agent_thinking", "enabled"))},
                proposal_count=int(case_stage6.get("proposal_count", 32)),
            )
        )
        case_stage6_root = ensure_dir(root / "outputs" / case_id / str(args.stage6_subdir))
        case_context = load_stage6_case_context(
            root=root,
            case_entry=case_entry,
            stage5=case_stage5,
            stage6=case_stage6,
        )
        transform_library_csv, transform_library_summary_json = write_case_transform_library_artifacts(
            case_stage6_root=case_stage6_root,
            case_context=case_context,
            stage6=case_stage6,
        )
        oracle_mode = str((case_context.get("scoring_policy") or {}).get("oracle_v2_mode") or "hypothesis_only")
        if oracle_mode == "hypothesis_only" and not bool(case_stage6.get("allow_hypothesis_only_stage6_search", False)):
            raise RuntimeError(
                f"Stage6 oracle gate forbids full search for {case_id}: oracle_v2_mode=hypothesis_only"
            )
        reference_affinities = ensure_stage6_reference_affinities(
            root=root,
            case_context=case_context,
            stage6=case_stage6,
        )
        case_context["lead_wt_affinity_kcal_mol"] = reference_affinities.get(
            "wt_affinity_kcal_mol",
            case_context.get("lead_wt_affinity_kcal_mol"),
        )
        case_context["lead_mt_affinities"] = {
            **dict(case_context.get("lead_mt_affinities") or {}),
            **dict(reference_affinities.get("target_affinities") or {}),
        }

        llm_decisions: list[dict[str, Any]] = []
        if args.finalize_existing:
            leaderboard = read_csv_optional(case_stage6_root / "leaderboard.csv")
            prefilter_audit = read_csv_optional(case_stage6_root / "design_prefilter_audit.csv")
            if leaderboard.empty:
                raise RuntimeError(f"No existing leaderboard available for --finalize-existing at {case_stage6_root}")
            leaderboard = ensure_stage6_health_columns(leaderboard)
            leaderboard = recompute_cached_leaderboard_scores(
                leaderboard,
                case_context=case_context,
                stage6=case_stage6,
            )
        else:
            leaderboard_frames: list[pd.DataFrame] = []
            prefilter_frames: list[pd.DataFrame] = []
            objective_histories: dict[str, pd.DataFrame] = {}
            objective_seed_counts: dict[str, int] = {}
            execution_order = validate_objective_plan(case_stage6)
            for objective_name in execution_order:
                seed_rows = build_cross_objective_seed_rows(
                    objective_name=objective_name,
                    prior_histories=objective_histories,
                    case_context=case_context,
                    stage6=case_stage6,
                )
                objective_seed_counts[objective_name] = int(len(seed_rows))
                objective_history, prefilter_frame, objective_llm = run_objective_search(
                    root=root,
                    case_entry=case_entry,
                    case_context=case_context,
                    stage5=case_stage5,
                    stage6=case_stage6,
                    objective_name=objective_name,
                    agent=agent,
                    disable_llm=disable_llm,
                    case_stage6_root=case_stage6_root,
                    seed_rows=seed_rows,
                )
                if not objective_history.empty:
                    leaderboard_frames.append(objective_history)
                    objective_histories[objective_name] = objective_history
                if not prefilter_frame.empty:
                    prefilter_frames.append(prefilter_frame)
                llm_decisions.extend(objective_llm)

            leaderboard = pd.concat(leaderboard_frames, ignore_index=True) if leaderboard_frames else pd.DataFrame()
            if not leaderboard.empty:
                leaderboard = leaderboard.sort_values(
                    ["objective_name", "objective_reward", "robust_score", "keep_ifp", "candidate_id"],
                    ascending=[True, False, False, False, True],
                ).reset_index(drop=True)
                leaderboard = ensure_stage6_health_columns(leaderboard)
            prefilter_audit = pd.concat(prefilter_frames, ignore_index=True) if prefilter_frames else pd.DataFrame()
        if args.finalize_existing:
            execution_order = validate_objective_plan(case_stage6)
            objective_seed_counts = {}
        write_table(leaderboard, case_stage6_root / "leaderboard.csv")
        write_table(prefilter_audit, case_stage6_root / "design_prefilter_audit.csv")
        write_table(leaderboard, case_stage6_root / "search_trajectory.csv")

        robust_frame = leaderboard[leaderboard["objective_name"].eq("robust")].copy() if not leaderboard.empty else pd.DataFrame()
        if not robust_frame.empty:
            robust_frame = robust_frame.sort_values("objective_reward", ascending=False).reset_index(drop=True)
            robust_frame = rerank_uncertainty_heavy_candidates(
                root=root,
                case_stage6_root=case_stage6_root,
                case_context=case_context,
                stage5=case_stage5,
                stage6=case_stage6,
                robust_frame=robust_frame,
            )
            leaderboard = merge_optional_rerank_columns(
                leaderboard=leaderboard,
                robust_frame=robust_frame,
            )
            leaderboard = leaderboard.sort_values(
                ["objective_name", "objective_reward", "robust_score", "dep", "keep_ifp", "candidate_id"],
                ascending=[True, False, False, True, False, True],
            ).reset_index(drop=True)
            robust_frame = leaderboard[leaderboard["objective_name"].eq("robust")].copy().reset_index(drop=True)
        write_top_sdf(robust_frame, case_stage6_root / "designed_top200.sdf", int(case_stage6.get("top_k_export", 200)))
        (case_stage6_root / "robust_sar_rules.md").write_text(
            render_sar_rules(case_entry, robust_frame),
            encoding="utf-8",
        )

        ablation_rows = objective_ablation_rows(
            leaderboard,
            str(case_context["lead_candidate_id"]),
            case_stage6,
            bool(str(case_entry.get("evaluation_unit") or "") == "observed_combo"),
        )
        objective_ablation = pd.DataFrame.from_records(ablation_rows)
        write_table(objective_ablation, case_stage6_root / "objective_ablation.csv")

        qc_payload = {
            "case_id": case_id,
            "proposal_count": int(case_stage6.get("proposal_count", 32)),
            "beam_width": int(case_stage6.get("beam_width", 8)),
            "max_rounds": int(case_stage6.get("max_rounds", 12)),
            "max_parallel_candidates": int(case_stage6.get("max_parallel_candidates", 4)),
            "target_parallel_workers": int(case_stage6.get("target_parallel_workers", 4)),
            "site_panel_top_n": int(case_stage6.get("site_panel_max_n", case_stage6.get("site_panel_top_n", 20))),
            "combo_panel_top_n": int(case_stage6.get("combo_panel_max_n", case_stage6.get("combo_panel_top_n", 20)))
            if str(case_entry.get("evaluation_unit") or "") == "observed_combo"
            else 0,
            "candidate_count": int(leaderboard["candidate_id"].nunique()) if not leaderboard.empty else 0,
            "valid_candidate_rate": float(leaderboard["candidate_valid"].fillna(False).astype(bool).mean()) if not leaderboard.empty else 0.0,
            "prefilter_fail_count": int(prefilter_audit["prefilter_pass"].fillna(False).eq(False).sum()) if not prefilter_audit.empty else 0,
            "robust_best_candidate_id": None if robust_frame.empty else str(robust_frame.iloc[0]["candidate_id"]),
            "robust_best_reward": None if robust_frame.empty else float(robust_frame.iloc[0]["objective_reward"]),
            "robust_top50_scaffold_unique": int(top_scaffold_diversity(robust_frame, int(case_stage6.get("scaffold_diversity_top_n", 50)))),
            "combo_dense_case": bool(str(case_entry.get("evaluation_unit") or "") == "observed_combo"),
            "agent_thinking": {"type": str(case_stage6.get("agent_thinking", "enabled"))},
            "disable_llm": bool(disable_llm),
            "objective_execution_order": list(execution_order),
            "objective_seed_counts": objective_seed_counts,
            "search_seed": native_value(case_stage6.get("search_seed")),
            "search_rank_jitter": native_value(case_stage6.get("search_rank_jitter")),
            "oracle_v2_mode": oracle_mode,
            "oracle_v2_effective_trust_score": native_value((case_context.get("scoring_policy") or {}).get("oracle_v2_effective_trust_score")),
            "oracle_v2_case_holdout_trust_score": native_value((case_context.get("scoring_policy") or {}).get("oracle_v2_case_holdout_trust_score")),
            "oracle_v2_domain_holdout_trust_score": native_value((case_context.get("scoring_policy") or {}).get("oracle_v2_domain_holdout_trust_score")),
            "oracle_v2_ensemble_holdout_trust_score": native_value((case_context.get("scoring_policy") or {}).get("oracle_v2_ensemble_holdout_trust_score")),
            "receptor_ensemble_enabled": bool(case_context.get("receptor_ensemble_enabled", False)),
            "receptor_ensemble_target_count": int(len(case_context.get("receptor_ensemble_members") or {})),
        }
        qc_payload.update(stage6_executability_metrics(leaderboard))
        json_dump(case_stage6_root / "stage6_qc.json", qc_payload)
        case_qc_rows.append(qc_payload)

        stage6_outputs = [
            case_stage6_root / "leaderboard.csv",
            case_stage6_root / "designed_top200.sdf",
            case_stage6_root / "design_prefilter_audit.csv",
            case_stage6_root / "robust_sar_rules.md",
            case_stage6_root / "objective_ablation.csv",
            case_stage6_root / "search_trajectory.csv",
            case_stage6_root / "stage6_qc.json",
            transform_library_csv,
            transform_library_summary_json,
        ]
        if str(args.stage6_subdir) == "stage6":
            write_case_state(
                root=root,
                case_entry=case_entry,
                args_config=args.config,
                stage6=case_stage6,
                case_stage6_root=case_stage6_root,
                leaderboard_path=case_stage6_root / "leaderboard.csv",
                designed_sdf_path=case_stage6_root / "designed_top200.sdf",
                prefilter_path=case_stage6_root / "design_prefilter_audit.csv",
                objective_ablation_path=case_stage6_root / "objective_ablation.csv",
                sar_rules_path=case_stage6_root / "robust_sar_rules.md",
                qc_payload=qc_payload,
                llm_decisions=llm_decisions,
                software_versions=software_versions,
                commands=commands,
            )
            write_case_manifest(
                root=root,
                case_entry=case_entry,
                started_at=started_at,
                software_versions=software_versions,
                commands=commands,
                inputs=[
                    root / args.config,
                    root / f"outputs/{case_id}/stage3_5/wt_ifp.json",
                    root / f"outputs/{case_id}/stage5/ifp_diff.csv",
                    root / f"outputs/{case_id}/stage5/mutation_effects.json",
                ],
                outputs=stage6_outputs,
            )
            validate_case_artifacts(root, case_id)
        else:
            write_auxiliary_stage6_manifest(
                root=root,
                case_entry=case_entry,
                case_stage6_root=case_stage6_root,
                started_at=started_at,
                software_versions=software_versions,
                commands=commands,
                qc_payload=qc_payload,
                stage6=case_stage6,
                outputs=stage6_outputs,
            )

    if case_qc_rows and str(args.stage6_subdir) == "stage6":
        summary_path = root / str(stage6.get("summary_report", "outputs/stage6/stage6_qc.json"))
        existing_summary = load_json_or_empty(summary_path)
        case_map: dict[str, dict[str, Any]] = {}
        for row in list(existing_summary.get("cases") or []):
            if isinstance(row, dict) and row.get("case_id"):
                case_map[str(row["case_id"])] = dict(row)
        for row in case_qc_rows:
            case_map[str(row["case_id"])] = dict(row)
        merged_cases = [case_map[key] for key in sorted(case_map)]
        summary = {
            "case_count": int(len(merged_cases)),
            "disable_llm": bool(disable_llm),
            "agent_thinking": {"type": str(stage6.get("agent_thinking", "enabled"))},
            "candidate_count_total": int(sum(int(row.get("candidate_count") or 0) for row in merged_cases)),
            "prefilter_fail_count_total": int(sum(int(row.get("prefilter_fail_count") or 0) for row in merged_cases)),
            "valid_candidate_rate_mean": float(
                sum(float(row.get("valid_candidate_rate") or 0.0) for row in merged_cases) / max(1, len(merged_cases))
            ),
            "chemical_valid_rate_mean": float(
                sum(float(row.get("chemical_valid_rate") or 0.0) for row in merged_cases) / max(1, len(merged_cases))
            ),
            "prefilter_pass_rate_mean": float(
                sum(float(row.get("prefilter_pass_rate") or 0.0) for row in merged_cases) / max(1, len(merged_cases))
            ),
            "wt_pass_rate_mean": float(
                sum(float(row.get("wt_pass_rate") or 0.0) for row in merged_cases) / max(1, len(merged_cases))
            ),
            "panel_coverage_pass_rate_mean": float(
                sum(float(row.get("panel_coverage_pass_rate") or 0.0) for row in merged_cases) / max(1, len(merged_cases))
            ),
            "panel_passing_rate_mean": float(
                sum(float(row.get("panel_passing_rate") or 0.0) for row in merged_cases) / max(1, len(merged_cases))
            ),
            "updated_case_ids": [str(row["case_id"]) for row in case_qc_rows],
            "cases": merged_cases,
        }
        ensure_dir(summary_path.parent)
        json_dump(summary_path, summary)


if __name__ == "__main__":
    main()
