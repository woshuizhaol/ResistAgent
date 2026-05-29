#!/usr/bin/env python3
"""counter-design step.5 validation helpers."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from rdkit import Chem

from tools.runtime import command_exists, ensure_dir, iso_now, json_dump, load_yaml, project_root
from tools.stage35_utils import ifp_frequency, plip_ifp, save_chain_protein
from tools.stage5_physics import probe_ligand_parameterization
from tools.stage5_utils import (
    build_hiv_reference,
    ensure_stage5_relaxation,
    ensure_stage5_scoring_payload,
    first_protein_chain_id,
    materialize_stage5_modeled_sample,
    native_value,
    stable_target_slug,
    stage5_case_context,
    stage5_for_case,
    write_json,
)
from tools.stage6_utils import (
    _dock_candidate,
    _target_box,
    candidate_id,
    parse_json_payload,
    read_csv_optional,
    residue_number_from_label,
    write_candidate_sdf,
)

LIGAND_RESNAME = "LIG"
WT_REFERENCE_KEY = "WT_REFERENCE"


def selected_cases(cases_config: dict[str, object], case_id: str | None) -> list[dict[str, object]]:
    cases = list(cases_config.get("set_d", []))
    if case_id is None:
        return [case for case in cases if str(case.get("case_id")) in {"egfr_erlotinib", "hiv_rt_rilpivirine"}]
    return [case for case in cases if str(case.get("case_id")) == str(case_id)]


def _safe_float(value: Any, default: float = 0.0) -> float:
    native = native_value(value)
    if native is None:
        return float(default)
    return float(native)


def _amber_even_electron_compatible(smiles: str | None) -> bool:
    text = str(smiles or "").strip()
    if not text:
        return False
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return False
    total_electrons = 0
    total_formal_charge = 0
    for atom in molecule.GetAtoms():
        total_electrons += int(atom.GetAtomicNum())
        total_electrons += int(atom.GetTotalNumHs(includeNeighbors=True))
        total_formal_charge += int(atom.GetFormalCharge())
        if int(atom.GetNumRadicalElectrons()) > 0:
            return False
    electron_count = int(total_electrons - total_formal_charge)
    return bool(electron_count % 2 == 0)


def _logit01(value: float) -> float:
    clipped = min(1.0 - 1.0e-6, max(1.0e-6, float(value)))
    return float(math.log(clipped / (1.0 - clipped)))


def load_stage6_leaderboard(case_stage6_root: Path) -> pd.DataFrame:
    frame = read_csv_optional(case_stage6_root / "leaderboard.csv")
    if frame.empty:
        return frame
    for column in ["objective_reward", "objective_rank", "s_wt", "wt_affinity_kcal_mol"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_case_manifest(case_entry: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = project_root() if root is None else root
    manifest_path = root / str(case_entry["manifest_path"])
    return load_yaml(manifest_path)


def _case_candidate_priority(stage6_5: dict[str, Any], case_id: str, key: str) -> list[str]:
    overrides = dict(stage6_5.get("case_candidate_overrides") or {})
    case_payload = dict(overrides.get(case_id) or {})
    values = list(case_payload.get(key) or [])
    return [str(value) for value in values if str(value).strip()]


def load_stage6_5_candidate_history(
    *,
    root: Path,
    case_id: str,
    stage6_5: dict[str, Any],
) -> pd.DataFrame:
    case_root = root / "outputs" / case_id
    output_dirname = str(stage6_5.get("output_dirname", "stage6_5"))
    candidate_roots: list[Path] = []
    current_root = case_root / output_dirname
    if current_root.exists():
        candidate_roots.append(current_root)
    candidate_roots.extend(sorted(case_root.glob(f"{output_dirname}_backup_*"), reverse=True))
    history_root = next((path for path in candidate_roots if (path / "validation_manifest.json").exists()), None)
    if history_root is None:
        return pd.DataFrame()

    manifest = json.loads((history_root / "validation_manifest.json").read_text(encoding="utf-8"))
    label_to_candidate: dict[str, str] = {}
    attempted_candidates: set[str] = set()
    for item in list(manifest.get("molecules") or []) + list(manifest.get("energy_molecules") or []):
        if not isinstance(item, dict):
            continue
        candidate_id_text = str(item.get("candidate_id") or "").strip()
        molecule_label = str(item.get("molecule_label") or "").strip()
        if candidate_id_text and molecule_label:
            label_to_candidate[molecule_label] = candidate_id_text
        if candidate_id_text and candidate_id_text != "lead":
            attempted_candidates.add(candidate_id_text)

    occupancy = read_csv_optional(history_root / "md" / "occupancy.csv")
    if not occupancy.empty:
        occupancy["candidate_id"] = occupancy["molecule_label"].astype(str).map(label_to_candidate)
        occupancy = occupancy[occupancy["candidate_id"].astype(str).ne("")]
    mmgbsa = read_csv_optional(history_root / "energy" / "mmgbsa_summary.csv")
    if not mmgbsa.empty and "candidate_id" not in mmgbsa.columns:
        mmgbsa["candidate_id"] = mmgbsa["molecule_label"].astype(str).map(label_to_candidate)
    if not mmgbsa.empty:
        mmgbsa = mmgbsa[mmgbsa["candidate_id"].astype(str).ne("")]

    md_success_counts: dict[str, int] = {}
    if not occupancy.empty:
        md_success_counts = (
            occupancy[["candidate_id", "target_key"]]
            .drop_duplicates()
            .groupby("candidate_id", dropna=False)
            .size()
            .astype(int)
            .to_dict()
        )
    mmgbsa_available_counts: dict[str, int] = {}
    mmgbsa_attempt_counts: dict[str, int] = {}
    if not mmgbsa.empty:
        mmgbsa_attempt_counts = mmgbsa.groupby("candidate_id", dropna=False).size().astype(int).to_dict()
        available = mmgbsa[mmgbsa["delta_mmgbsa_binding_kcal_mol"].notna()].copy()
        if not available.empty:
            mmgbsa_available_counts = available.groupby("candidate_id", dropna=False).size().astype(int).to_dict()

    rows: list[dict[str, Any]] = []
    for candidate_id_text in sorted(
        attempted_candidates | set(md_success_counts) | set(mmgbsa_attempt_counts) | set(mmgbsa_available_counts)
    ):
        if not candidate_id_text or candidate_id_text == "lead":
            continue
        md_success_pair_count = int(md_success_counts.get(candidate_id_text, 0))
        mmgbsa_attempt_count = int(mmgbsa_attempt_counts.get(candidate_id_text, 0))
        mmgbsa_available_count = int(mmgbsa_available_counts.get(candidate_id_text, 0))
        success_flag = bool(md_success_pair_count > 0 or mmgbsa_available_count > 0)
        attempted_flag = bool(candidate_id_text in attempted_candidates or mmgbsa_attempt_count > 0)
        rows.append(
            {
                "candidate_id": candidate_id_text,
                "stage6_5_history_source": str(history_root.relative_to(root)),
                "stage6_5_prior_attempted": attempted_flag,
                "stage6_5_prior_md_success_pair_count": md_success_pair_count,
                "stage6_5_prior_mmgbsa_attempt_count": mmgbsa_attempt_count,
                "stage6_5_prior_mmgbsa_available_count": mmgbsa_available_count,
                "stage6_5_prior_success_flag": success_flag,
                "stage6_5_prior_failure_only": bool(attempted_flag and not success_flag),
            }
        )
    return pd.DataFrame.from_records(rows)


def _preferred_candidate_row(
    frame: pd.DataFrame,
    *,
    candidate_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> pd.Series | None:
    if frame.empty:
        return None
    if candidate_filter is None:
        return frame.iloc[0]
    for _, row in frame.iterrows():
        if candidate_filter(row.to_dict()):
            return row
    return frame.iloc[0]


def _preferred_candidate_row_by_ids(
    frame: pd.DataFrame,
    *,
    preferred_candidate_ids: list[str],
    candidate_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> pd.Series | None:
    if frame.empty or not preferred_candidate_ids:
        return None
    preferred_lookup = {str(candidate_id_text) for candidate_id_text in preferred_candidate_ids}
    subset = frame[frame["candidate_id"].astype(str).isin(preferred_lookup)].copy()
    if subset.empty:
        return None
    return _preferred_candidate_row(subset, candidate_filter=candidate_filter)


def _ordered_rows_by_candidate_ids(frame: pd.DataFrame, candidate_ids: list[str]) -> list[pd.Series]:
    if frame.empty or not candidate_ids:
        return []
    rows: list[pd.Series] = []
    for candidate_id_text in candidate_ids:
        subset = frame[frame["candidate_id"].astype(str).eq(str(candidate_id_text))].head(1)
        if subset.empty:
            continue
        rows.append(subset.iloc[0])
    return rows


def _sort_columns_present(frame: pd.DataFrame, columns: list[tuple[str, bool]]) -> tuple[list[str], list[bool]]:
    present = [(column, ascending) for column, ascending in columns if column in frame.columns]
    return [column for column, _ in present], [ascending for _, ascending in present]


def select_validation_targets(
    *,
    case_entry: dict[str, Any],
    manifest: dict[str, Any],
    combo_panel: pd.DataFrame,
    stage6_5: dict[str, Any],
) -> list[dict[str, Any]]:
    case_id = str(case_entry["case_id"])
    rows: list[dict[str, Any]] = [
        {
            "effect_scope": "wt_reference",
            "target_key": WT_REFERENCE_KEY,
            "selection_bucket": "wt_reference",
            "rank": 0,
            "source": "stage3_5_wt_reference",
            "representative_sample_id": "",
            "known_hotspot": False,
        }
    ]
    site_top_risk = list(dict(manifest.get("site_pool") or {}).get("top_risk20") or [])
    if case_id == "hiv_rt_rilpivirine":
        high_risk_sites = [dict(row) for row in site_top_risk if isinstance(row, dict) and str(row.get("mutation_key") or "")]
        chosen_site = None
        for row in high_risk_sites:
            if bool(row.get("known_hotspot", False)):
                chosen_site = row
                break
        if chosen_site is None and high_risk_sites:
            chosen_site = high_risk_sites[0]
        if chosen_site is not None:
            rows.append(
                {
                    "effect_scope": "site",
                    "target_key": str(chosen_site["mutation_key"]),
                    "selection_bucket": "high_risk_single",
                    "rank": int(chosen_site.get("rank") or 0),
                    "source": "site_pool.top_risk20",
                    "representative_sample_id": str(chosen_site.get("representative_sample_id") or ""),
                    "known_hotspot": bool(chosen_site.get("known_hotspot", False)),
                }
            )
        combo_rows = combo_panel[combo_panel["case_id"].astype(str).eq(case_id)].copy()
        if not combo_rows.empty:
            sort_columns = ["combo_rank"] + (["count"] if "count" in combo_rows.columns else [])
            ascending = [True] + ([False] if "count" in combo_rows.columns else [])
            combo_rows = combo_rows.sort_values(sort_columns, ascending=ascending)
        for _, row in combo_rows.head(int(stage6_5.get("validation_combo_topcombo_n", 2))).iterrows():
            rows.append(
                {
                    "effect_scope": "combo",
                    "target_key": str(row["combination_key"]),
                    "selection_bucket": "topcombo_validation",
                    "rank": int(row.get("combo_rank") or 0),
                    "source": "combo_panel",
                    "representative_sample_id": str(row.get("representative_sample_id") or ""),
                    "known_hotspot": False,
                }
            )
        return rows

    site_rows = [dict(row) for row in site_top_risk if isinstance(row, dict) and str(row.get("mutation_key") or "")]
    seen_targets = {WT_REFERENCE_KEY}
    high_risk_needed = int(stage6_5.get("validation_noncombo_high_risk_n", 2))
    moderate_needed = int(stage6_5.get("validation_noncombo_moderate_n", 1))
    for row in site_rows:
        if len([item for item in rows if item["selection_bucket"] == "high_risk_single"]) >= high_risk_needed:
            break
        target_key = str(row["mutation_key"])
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        rows.append(
            {
                "effect_scope": "site",
                "target_key": target_key,
                "selection_bucket": "high_risk_single",
                "rank": int(row.get("rank") or 0),
                "source": "site_pool.top_risk20",
                "representative_sample_id": str(row.get("representative_sample_id") or ""),
                "known_hotspot": bool(row.get("known_hotspot", False)),
            }
        )
    moderate_candidates = [row for row in site_rows if int(row.get("rank") or 0) >= 10]
    for row in moderate_candidates:
        if len([item for item in rows if item["selection_bucket"] == "moderate_single"]) >= moderate_needed:
            break
        target_key = str(row["mutation_key"])
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        rows.append(
            {
                "effect_scope": "site",
                "target_key": target_key,
                "selection_bucket": "moderate_single",
                "rank": int(row.get("rank") or 0),
                "source": "site_pool.top_risk20",
                "representative_sample_id": str(row.get("representative_sample_id") or ""),
                "known_hotspot": bool(row.get("known_hotspot", False)),
            }
        )
    return rows


def select_validation_molecules(
    *,
    case_entry: dict[str, Any],
    leaderboard: pd.DataFrame,
    stage6_5: dict[str, Any],
    candidate_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_id = str(case_entry["case_id"])
    blocked_candidate_ids = set(_case_candidate_priority(stage6_5, case_id, "blocked_candidate_ids"))
    blocked_energy_candidate_ids = set(_case_candidate_priority(stage6_5, case_id, "blocked_energy_candidate_ids"))
    preferred_energy_candidate_ids = _case_candidate_priority(stage6_5, case_id, "preferred_energy_candidate_ids")
    preferred_robust_best_candidate_ids = _case_candidate_priority(stage6_5, case_id, "preferred_robust_best_candidate_ids")
    preferred_wt_control_candidate_ids = _case_candidate_priority(stage6_5, case_id, "preferred_wt_control_candidate_ids")
    molecules: list[dict[str, Any]] = [
        {
            "molecule_label": "lead",
            "molecule_role": "lead",
            "candidate_id": "lead",
            "smiles": None,
            "source": "stage1_5.lead_ligand",
            "objective_name": "lead",
            "objective_reward": None,
            "s_wt": None,
        }
    ]
    energy_panel: list[dict[str, Any]] = []
    if leaderboard.empty:
        return molecules, energy_panel

    valid = leaderboard[leaderboard.get("candidate_valid", pd.Series(dtype=bool)).fillna(False).astype(bool)].copy()
    if valid.empty:
        valid = leaderboard.copy()
    if blocked_candidate_ids:
        filtered = valid[~valid["candidate_id"].astype(str).isin(blocked_candidate_ids)].copy()
        if not filtered.empty:
            valid = filtered
    if "smiles" in valid.columns:
        compatible_mask = valid["smiles"].astype(str).map(_amber_even_electron_compatible)
        compatible_valid = valid[compatible_mask].copy()
        if not compatible_valid.empty:
            valid = compatible_valid
    if "stage6_5_prior_failure_only" in valid.columns:
        failure_only_mask = valid["stage6_5_prior_failure_only"].fillna(False).astype(bool)
        if (~failure_only_mask).any():
            valid = valid[~failure_only_mask].copy()
    robust = valid[valid["objective_name"].astype(str).eq("robust")].copy()
    if robust.empty:
        robust = valid.copy()
    robust_sort_columns, robust_ascending = _sort_columns_present(
        robust,
        [
            ("stage6_5_prior_success_flag", False),
            ("objective_reward", False),
            ("objective_rank", True),
            ("robust_score", False),
            ("candidate_id", True),
        ],
    )
    robust = robust.sort_values(robust_sort_columns, ascending=robust_ascending, na_position="last")
    chosen_labels = {str(item["candidate_id"]) for item in molecules}
    best_robust = _preferred_candidate_row_by_ids(
        robust,
        preferred_candidate_ids=preferred_robust_best_candidate_ids,
        candidate_filter=candidate_filter,
    )
    if best_robust is None:
        best_robust = _preferred_candidate_row(robust, candidate_filter=candidate_filter)
    if best_robust is not None:
        row = best_robust
        molecules.append(
            {
                "molecule_label": "robust_best",
                "molecule_role": "robust_best",
                "candidate_id": str(row["candidate_id"]),
                "smiles": str(row["smiles"]),
                "source": "stage6.leaderboard.robust_best",
                "objective_name": str(row.get("objective_name") or ""),
                "objective_reward": native_value(row.get("objective_reward")),
                "s_wt": native_value(row.get("s_wt")),
            }
        )
        chosen_labels.add(str(row["candidate_id"]))

    wt_sort_columns, wt_ascending = _sort_columns_present(
        valid,
        [
            ("stage6_5_prior_mmgbsa_available_count", False),
            ("stage6_5_prior_md_success_pair_count", False),
            ("stage6_5_prior_success_flag", False),
            ("s_wt", False),
            ("objective_reward", False),
            ("wt_affinity_kcal_mol", True),
            ("candidate_id", True),
        ],
    )
    wt_candidates = valid.sort_values(wt_sort_columns, ascending=wt_ascending, na_position="last")
    wt_best_pool = wt_candidates[~wt_candidates["candidate_id"].astype(str).isin(chosen_labels)].copy()
    wt_best = _preferred_candidate_row_by_ids(
        wt_best_pool,
        preferred_candidate_ids=preferred_wt_control_candidate_ids,
        candidate_filter=candidate_filter,
    )
    if wt_best is None:
        wt_best = _preferred_candidate_row(wt_best_pool, candidate_filter=candidate_filter)
    if wt_best is None:
        wt_best = _preferred_candidate_row_by_ids(
            wt_candidates,
            preferred_candidate_ids=preferred_wt_control_candidate_ids,
            candidate_filter=candidate_filter,
        )
    if wt_best is None:
        wt_best = _preferred_candidate_row(wt_candidates, candidate_filter=candidate_filter)
    if wt_best is not None:
        row = wt_best
        molecules.append(
            {
                "molecule_label": "wt_best_control",
                "molecule_role": "wt_best_control",
                "candidate_id": str(row["candidate_id"]),
                "smiles": str(row["smiles"]),
                "source": "stage6.leaderboard.max_s_wt",
                "objective_name": str(row.get("objective_name") or ""),
                "objective_reward": native_value(row.get("objective_reward")),
                "s_wt": native_value(row.get("s_wt")),
            }
        )

    top_energy_n = max(1, int(stage6_5.get("energy_top_molecule_n", 5)))
    energy_panel.append(dict(molecules[0]))
    chosen_ids = {str(item["candidate_id"]) for item in energy_panel}
    candidate_pool = robust.copy()
    if wt_best is not None:
        candidate_pool = pd.concat([wt_best.to_frame().T, candidate_pool], ignore_index=True)
    candidate_pool = candidate_pool.drop_duplicates(subset=["candidate_id"], keep="first")
    if blocked_energy_candidate_ids:
        filtered_pool = candidate_pool[~candidate_pool["candidate_id"].astype(str).isin(blocked_energy_candidate_ids)].copy()
        if not filtered_pool.empty:
            candidate_pool = filtered_pool
    candidate_pool_sort_columns, candidate_pool_ascending = _sort_columns_present(
        candidate_pool,
        [
            ("stage6_5_prior_mmgbsa_available_count", False),
            ("stage6_5_prior_md_success_pair_count", False),
            ("stage6_5_prior_success_flag", False),
            ("objective_reward", False),
            ("objective_rank", True),
            ("s_wt", False),
            ("candidate_id", True),
        ],
    )
    candidate_pool = candidate_pool.sort_values(
        candidate_pool_sort_columns,
        ascending=candidate_pool_ascending,
        na_position="last",
    )
    ordered_candidate_pool_rows: list[pd.Series] = []
    if preferred_energy_candidate_ids:
        ordered_candidate_pool_rows.extend(_ordered_rows_by_candidate_ids(candidate_pool, preferred_energy_candidate_ids))
    remaining_candidate_pool = candidate_pool[
        ~candidate_pool["candidate_id"].astype(str).isin({str(row["candidate_id"]) for row in ordered_candidate_pool_rows})
    ].copy()
    ordered_candidate_pool_rows.extend([row for _, row in remaining_candidate_pool.iterrows()])
    for filter_required in [True, False]:
        for row in ordered_candidate_pool_rows:
            if len(energy_panel) >= top_energy_n:
                break
            cid = str(row["candidate_id"])
            if cid in chosen_ids:
                continue
            if filter_required and candidate_filter is not None and not candidate_filter(row.to_dict()):
                continue
            chosen_ids.add(cid)
            energy_panel.append(
                {
                    "molecule_label": f"energy_{len(energy_panel)}",
                    "molecule_role": "energy_panel",
                    "candidate_id": cid,
                    "smiles": str(row["smiles"]),
                    "source": f"stage6.leaderboard.{case_id}",
                    "objective_name": str(row.get("objective_name") or ""),
                    "objective_reward": native_value(row.get("objective_reward")),
                    "s_wt": native_value(row.get("s_wt")),
                }
            )
        if len(energy_panel) >= top_energy_n or candidate_filter is None:
            break
    return molecules, energy_panel


def select_energy_targets(
    *,
    case_entry: dict[str, Any],
    manifest: dict[str, Any],
    combo_panel: pd.DataFrame,
    stage6_5: dict[str, Any],
) -> list[dict[str, Any]]:
    case_id = str(case_entry["case_id"])
    top_n = max(1, int(stage6_5.get("energy_top_target_n", 5)))
    targets: list[dict[str, Any]] = []
    site_rows = [dict(row) for row in list(dict(manifest.get("site_pool") or {}).get("top_risk20") or []) if isinstance(row, dict)]
    if case_id == "hiv_rt_rilpivirine":
        chosen_site = None
        for row in site_rows:
            if bool(row.get("known_hotspot", False)):
                chosen_site = row
                break
        if chosen_site is not None:
            targets.append(
                {
                    "effect_scope": "site",
                    "target_key": str(chosen_site["mutation_key"]),
                    "selection_bucket": "energy_single",
                    "rank": int(chosen_site.get("rank") or 0),
                }
            )
        combo_rows = combo_panel[combo_panel["case_id"].astype(str).eq(case_id)].copy()
        if not combo_rows.empty:
            sort_columns = ["combo_rank"] + (["count"] if "count" in combo_rows.columns else [])
            ascending = [True] + ([False] if "count" in combo_rows.columns else [])
            combo_rows = combo_rows.sort_values(sort_columns, ascending=ascending)
        for _, row in combo_rows.head(top_n - len(targets)).iterrows():
            targets.append(
                {
                    "effect_scope": "combo",
                    "target_key": str(row["combination_key"]),
                    "selection_bucket": "energy_combo",
                    "rank": int(row.get("combo_rank") or 0),
                }
            )
        return targets[:top_n]
    for row in site_rows[:top_n]:
        targets.append(
            {
                "effect_scope": "site",
                "target_key": str(row["mutation_key"]),
                "selection_bucket": "energy_site",
                "rank": int(row.get("rank") or 0),
            }
        )
    return targets[:top_n]


def stage6_5_case_root(root: Path, case_id: str, stage6_5: dict[str, Any]) -> Path:
    return ensure_dir(root / "outputs" / case_id / str(stage6_5.get("output_dirname", "stage6_5")))


def _build_candidate_parameter_filter(
    *,
    root: Path,
    case_id: str,
    stage6_5_root: Path,
    stage6_5: dict[str, Any],
) -> tuple[Callable[[dict[str, Any]], bool] | None, list[dict[str, Any]]]:
    if not bool(stage6_5.get("molecule_parameter_probe_enabled", True)):
        return None, []
    if not command_exists("antechamber") or not command_exists("parmchk2"):
        return None, []
    timeout_sec = float(stage6_5.get("molecule_parameter_probe_timeout_sec", 45.0))
    probe_root = ensure_dir(stage6_5_root / "selection_probes")
    cache_root = ensure_dir(stage6_5_root / "md" / "systems" / "ligand_param_cache")
    audit_rows: list[dict[str, Any]] = []
    cached_payloads: dict[str, dict[str, Any]] = {}

    def candidate_filter(candidate: dict[str, Any]) -> bool:
        candidate_id_text = str(candidate.get("candidate_id") or "")
        if candidate_id_text in {"", "lead"}:
            return True
        if candidate_id_text in cached_payloads:
            return bool(cached_payloads[candidate_id_text].get("available", False))
        smiles = str(candidate.get("smiles") or "")
        payload: dict[str, Any] = {
            "case_id": case_id,
            "candidate_id": candidate_id_text,
            "objective_name": str(candidate.get("objective_name") or ""),
            "objective_reward": native_value(candidate.get("objective_reward")),
            "s_wt": native_value(candidate.get("s_wt")),
            "smiles": smiles,
            "probe_timeout_sec": float(timeout_sec),
        }
        if not smiles:
            payload.update({"available": False, "error": "missing_smiles"})
        else:
            candidate_root = ensure_dir(probe_root / candidate_id_text)
            input_sdf = candidate_root / "candidate.sdf"
            write_candidate_sdf(smiles, input_sdf)
            payload.update(
                probe_ligand_parameterization(
                    input_sdf=input_sdf,
                    work_root=candidate_root,
                    ligand_parameter_cache_root=cache_root,
                    timeout_sec=timeout_sec,
                )
            )
        cached_payloads[candidate_id_text] = payload
        audit_rows.append(payload)
        return bool(payload.get("available", False))

    return candidate_filter, audit_rows


def _copy_or_write_ligand(
    *,
    root: Path,
    case_id: str,
    molecule: dict[str, Any],
    ligands_root: Path,
) -> Path:
    role = str(molecule["molecule_role"])
    if role == "lead":
        source = root / "outputs" / case_id / "stage1_5" / "raw" / "ligand.sdf"
        target = ligands_root / "lead.sdf"
        ensure_dir(target.parent)
        shutil.copyfile(source, target)
        return target
    candidate_path = ligands_root / f"{str(molecule['candidate_id'])}.sdf"
    write_candidate_sdf(str(molecule["smiles"]), candidate_path)
    return candidate_path


def materialize_wt_reference_sample(
    *,
    root: Path,
    case_id: str,
    sample_root: Path,
) -> tuple[Path, str]:
    if (sample_root / "WT.pdb").exists() and (sample_root / "MT.pdb").exists():
        return sample_root, "wt_reference"
    ensure_dir(sample_root)
    case_context = stage5_case_context(root, case_id)
    wt_complex_path = root / "outputs" / case_id / "stage3_5" / "wt_complex.pdb"
    ligand_input = root / "outputs" / case_id / "stage1_5" / "raw" / "ligand.sdf"
    chain_id = str(case_context["chain_id"]) if str(case_context.get("chain_id") or "") else first_protein_chain_id(wt_complex_path)
    wt_pdb = sample_root / "WT.pdb"
    mt_pdb = sample_root / "MT.pdb"
    wt_complex = sample_root / "WT_complex.pdb"
    mt_complex = sample_root / "MT_complex.pdb"
    save_chain_protein(wt_complex_path, chain_id, wt_pdb)
    shutil.copyfile(wt_pdb, mt_pdb)
    shutil.copyfile(wt_complex_path, wt_complex)
    shutil.copyfile(wt_complex_path, mt_complex)
    shutil.copyfile(ligand_input, sample_root / "ligand.sdf")
    write_json(
        sample_root / "model_manifest.json",
        {
            "model_kind": "wt_reference",
            "case_id": case_id,
            "target_key": WT_REFERENCE_KEY,
            "created_at": iso_now(),
        },
    )
    return sample_root, "wt_reference"


def materialize_validation_sample(
    *,
    root: Path,
    case_id: str,
    target: dict[str, Any],
    samples_root: Path,
) -> tuple[Path, str]:
    effect_scope = str(target["effect_scope"])
    target_key = str(target["target_key"])
    sample_root = ensure_dir(samples_root / stable_target_slug(effect_scope, target_key))
    if target_key == WT_REFERENCE_KEY:
        return materialize_wt_reference_sample(root=root, case_id=case_id, sample_root=sample_root)
    if (sample_root / "model_manifest.json").exists():
        try:
            payload = json.loads((sample_root / "model_manifest.json").read_text(encoding="utf-8"))
            return sample_root, str(payload.get("model_kind") or "stage5_modeled")
        except Exception:
            pass
    model_kind, _ = materialize_stage5_modeled_sample(
        root=root,
        case_id=case_id,
        effect_scope=effect_scope,
        target_key=target_key,
        sample_root=sample_root,
    )
    return sample_root, model_kind


def _gpu_id() -> int | str | None:
    cuda_visible_devices = str((os.environ.get("CUDA_VISIBLE_DEVICES") or "")).strip()
    if not cuda_visible_devices:
        return None
    first = cuda_visible_devices.split(",")[0].strip()
    if not first:
        return None
    if first.isdigit():
        return int(first)
    return first


def _stage6_5_gpu_ids(stage5: dict[str, Any]) -> list[int | str]:
    env_value = str(os.environ.get("RESISTGPT_STAGE6_5_GPU_IDS") or "").strip()
    if env_value:
        values = [token.strip() for token in env_value.split(",") if token.strip()]
    else:
        raw_values = list(stage5.get("local_sampling_gpu_ids", []) or [])
        values = [str(value).strip() for value in raw_values if str(value).strip()]
    normalized: list[int | str] = []
    for value in values:
        normalized.append(int(value) if str(value).isdigit() else str(value))
    return normalized


def _stage6_5_max_workers(stage6_5: dict[str, Any], task_count: int) -> int:
    env_value = str(os.environ.get("RESISTGPT_STAGE6_5_MAX_WORKERS") or "").strip()
    configured = int(env_value) if env_value else int(stage6_5.get("pair_max_workers", 4))
    return max(1, min(configured, int(task_count)))


def _pose_rows_from_summary(root: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    path = root / str(summary.get("pose_rows_json") or "")
    if not path.exists():
        return []
    payload = parse_json_payload(path.read_text(encoding="utf-8")) or []
    if not isinstance(payload, list):
        return []
    return [dict(row) for row in payload if isinstance(row, dict)]


def run_validation_pair(
    *,
    root: Path,
    case_id: str,
    case_entry: dict[str, Any],
    target: dict[str, Any],
    molecule: dict[str, Any],
    ligand_sdf: Path,
    stage5: dict[str, Any],
    stage6: dict[str, Any],
    stage6_5_root: Path,
    hiv_reference: dict[str, Any] | None,
    gpu_id: int | str | None = None,
) -> dict[str, Any]:
    samples_root = ensure_dir(stage6_5_root / "samples")
    sample_root, model_kind = materialize_validation_sample(
        root=root,
        case_id=case_id,
        target=target,
        samples_root=samples_root,
    )
    pair_root = ensure_dir(
        stage6_5_root
        / "md"
        / "systems"
        / str(molecule["molecule_label"])
        / stable_target_slug(str(target["effect_scope"]), str(target["target_key"]))
    )
    docking_box = _target_box(sample_root, pair_root / "reference_box", stage6)
    wt_summary = _dock_candidate(
        root=root,
        receptor_pdb=sample_root / "WT.pdb",
        docking_box=docking_box,
        ligand_input_sdf=ligand_sdf,
        output_root=pair_root / "wt_docking",
        stage6=stage6,
        hiv_reference=hiv_reference,
    )
    mt_summary = _dock_candidate(
        root=root,
        receptor_pdb=sample_root / "MT.pdb",
        docking_box=docking_box,
        ligand_input_sdf=ligand_sdf,
        output_root=pair_root / "mt_docking",
        stage6=stage6,
        hiv_reference=hiv_reference,
    )
    pair_record = {
        "case_id": case_id,
        "target_key": str(target["target_key"]),
        "effect_scope": str(target["effect_scope"]),
        "selection_bucket": str(target.get("selection_bucket") or ""),
        "molecule_label": str(molecule["molecule_label"]),
        "molecule_role": str(molecule["molecule_role"]),
        "candidate_id": str(molecule["candidate_id"]),
        "sample_root": str(sample_root),
        "stage5_run_root": str(pair_root.relative_to(root)),
        "stage4_local_rmsd_a": 0.0,
        "component_count": max(0, str(target["target_key"]).count("+") + (0 if str(target["target_key"]) == WT_REFERENCE_KEY else 1)),
        "stage5_model_kind": model_kind,
        "wt_receptor_pdb": str(wt_summary.get("receptor_pdb") or ""),
        "wt_pose_sdf": str(wt_summary.get("pose_sdf") or ""),
        "wt_complex_docked_pdb": str(wt_summary.get("complex_pdb") or ""),
        "mt_receptor_pdb": str(mt_summary.get("receptor_pdb") or ""),
        "mt_pose_sdf": str(mt_summary.get("pose_sdf") or ""),
        "mt_complex_docked_pdb": str(mt_summary.get("complex_pdb") or ""),
        "stage5_status": "ok" if wt_summary.get("docking_status") == "ok" and mt_summary.get("docking_status") == "ok" else "failed",
        "stage5_error": None,
        "wt_best_affinity_kcal_mol": native_value(wt_summary.get("best_affinity_kcal_mol")),
        "mt_best_affinity_kcal_mol": native_value(mt_summary.get("best_affinity_kcal_mol")),
        "delta_dock_kcal_mol": None,
    }
    if pair_record["wt_best_affinity_kcal_mol"] is not None and pair_record["mt_best_affinity_kcal_mol"] is not None:
        pair_record["delta_dock_kcal_mol"] = float(
            float(pair_record["mt_best_affinity_kcal_mol"]) - float(pair_record["wt_best_affinity_kcal_mol"])
        )
    if pair_record["stage5_status"] != "ok":
        pair_record["stage5_error"] = ";".join(
            filter(
                None,
                [str(wt_summary.get("docking_error") or ""), str(mt_summary.get("docking_error") or "")],
            )
        )
        json_dump(pair_root / "pair_summary.json", pair_record)
        return pair_record

    relaxation_payload = ensure_stage5_relaxation(
        root=root,
        docking_row=pair_record,
        stage5=stage5,
        gpu_id=gpu_id,
    )
    scoring_payload = ensure_stage5_scoring_payload(
        root=root,
        docking_row=pair_record,
        stage5=stage5,
        gpu_id=gpu_id,
    )
    pair_record.update(scoring_payload)
    pair_record["wt_pose_rows_json"] = str(wt_summary.get("pose_rows_json") or "")
    pair_record["mt_pose_rows_json"] = str(mt_summary.get("pose_rows_json") or "")
    pair_record["wt_relaxation_mode"] = str(relaxation_payload["wt"]["relaxation"].get("relaxation_mode") or "")
    pair_record["mt_relaxation_mode"] = str(relaxation_payload["mt"]["relaxation"].get("relaxation_mode") or "")
    json_dump(pair_root / "pair_summary.json", pair_record)
    return pair_record


def _run_stage6_5_pair_job(job: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_validation_pair(**job)
    except Exception as exc:
        target = dict(job["target"])
        molecule = dict(job["molecule"])
        return {
            "case_id": str(job["case_id"]),
            "target_key": str(target["target_key"]),
            "effect_scope": str(target["effect_scope"]),
            "selection_bucket": str(target.get("selection_bucket") or ""),
            "molecule_label": str(molecule["molecule_label"]),
            "molecule_role": str(molecule["molecule_role"]),
            "candidate_id": str(molecule["candidate_id"]),
            "sample_root": "",
            "stage5_run_root": "",
            "stage5_status": "failed",
            "stage5_error": f"{type(exc).__name__}: {exc}",
            "delta_dock_kcal_mol": None,
            "wt_mmgbsa_binding_kcal_mol": None,
            "mt_mmgbsa_binding_kcal_mol": None,
            "delta_mmgbsa_binding_kcal_mol": None,
            "wt_gnina_affinity_kcal_mol": None,
            "mt_gnina_affinity_kcal_mol": None,
            "delta_gnina_affinity_kcal_mol": None,
            "consensus_direction": "",
            "high_uncertainty": True,
        }


def _execute_stage6_5_jobs(
    *,
    jobs: list[dict[str, Any]],
    stage6_5: dict[str, Any],
    stage5: dict[str, Any],
    label: str,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    gpu_ids = _stage6_5_gpu_ids(stage5)
    max_workers = _stage6_5_max_workers(stage6_5, len(jobs))
    submitted_jobs: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        gpu_id = gpu_ids[index % len(gpu_ids)] if gpu_ids else _gpu_id()
        submitted_jobs.append({**job, "gpu_id": gpu_id})
    print(
        f"[stage6_5] launch {label}: jobs={len(submitted_jobs)} max_workers={max_workers} "
        f"gpu_ids={gpu_ids if gpu_ids else ['env/default']}",
        flush=True,
    )
    records: list[dict[str, Any]] = []
    if max_workers <= 1:
        for index, job in enumerate(submitted_jobs, start=1):
            record = _run_stage6_5_pair_job(job)
            records.append(record)
            print(
                f"[stage6_5] done {label} {index}/{len(submitted_jobs)} "
                f"case={record['case_id']} target={record['target_key']} molecule={record['molecule_label']} "
                f"status={record.get('stage5_status')}",
                flush=True,
            )
        return records
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_run_stage6_5_pair_job, job): job for job in submitted_jobs}
        for index, future in enumerate(as_completed(future_map), start=1):
            record = future.result()
            records.append(record)
            print(
                f"[stage6_5] done {label} {index}/{len(submitted_jobs)} "
                f"case={record['case_id']} target={record['target_key']} molecule={record['molecule_label']} "
                f"status={record.get('stage5_status')}",
                flush=True,
            )
    return records


def _cpptraj_extract_frames(
    *,
    prmtop: Path,
    trajectory_path: Path,
    output_root: Path,
    stride: int,
) -> list[Path]:
    if not command_exists("cpptraj"):
        return []
    ensure_dir(output_root)
    input_path = output_root / "extract.in"
    input_path.write_text(
        "\n".join(
            [
                f"parm {prmtop}",
                f"trajin {trajectory_path} 1 last {max(1, int(stride))}",
                f"trajout {output_root / 'frame.pdb'} pdb multi",
                "run",
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["cpptraj", "-i", str(input_path)],
        cwd=str(output_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return sorted(output_root.glob("frame.pdb*"))


def _heavy_atom_coords_by_residue(pdb_path: Path) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    ligand_coords: list[np.ndarray] = []
    residue_heavy: dict[int, list[np.ndarray]] = {}
    residue_ca: dict[int, np.ndarray] = {}
    for atom in structure.get_atoms():
        residue = atom.get_parent()
        residue_name = str(residue.resname).strip().upper()
        element = str(getattr(atom, "element", "")).strip().upper()
        if element in {"", "H", "D"}:
            continue
        coord = np.asarray(atom.get_coord(), dtype=float)
        if residue_name == LIGAND_RESNAME:
            ligand_coords.append(coord)
            continue
        if residue.id[0] != " ":
            continue
        residue_number = int(residue.id[1])
        residue_heavy.setdefault(residue_number, []).append(coord)
        if atom.get_name().strip().upper() == "CA":
            residue_ca[residue_number] = coord
    ligand_array = np.asarray(ligand_coords, dtype=float) if ligand_coords else np.zeros((0, 3), dtype=float)
    heavy_arrays = {key: np.asarray(value, dtype=float) for key, value in residue_heavy.items()}
    return ligand_array, heavy_arrays, residue_ca


def _aligned_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    if reference.shape != mobile.shape or reference.size == 0:
        return 0.0
    ref_centered = reference - reference.mean(axis=0, keepdims=True)
    mob_centered = mobile - mobile.mean(axis=0, keepdims=True)
    covariance = mob_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt
    aligned = mob_centered @ rotation
    diff = aligned - ref_centered
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _select_distance_residues(case_context: dict[str, Any], target_key: str, top_n: int) -> list[int]:
    numbers: list[int] = []
    target_token = str(target_key)
    if ":" in target_token:
        target_token = target_token.split(":", 1)[1]
    for token in [part.strip() for part in target_token.split("+") if part.strip()]:
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            numbers.append(int(digits))
    for label in list(case_context.get("anchor_residues") or []):
        residue_number = residue_number_from_label(str(label))
        if residue_number is not None and residue_number not in numbers:
            numbers.append(int(residue_number))
        if len(numbers) >= top_n:
            break
    deduped: list[int] = []
    seen: set[int] = set()
    for number in numbers:
        if number in seen:
            continue
        deduped.append(number)
        seen.add(number)
        if len(deduped) >= top_n:
            break
    return deduped


def summarize_trajectory_system(
    *,
    root: Path,
    case_context: dict[str, Any],
    pair_root: Path,
    pose_set: str,
    pair_record: dict[str, Any],
    stage6_5: dict[str, Any],
) -> dict[str, Any]:
    physics_root = pair_root / f"{pose_set}_physics"
    amber_prep_path = physics_root / "amber_prep" / "amber_prep_manifest.json"
    relaxation_path = physics_root / "relaxation" / "relaxation_manifest.json"
    if not amber_prep_path.exists() or not relaxation_path.exists():
        return {
            "system_status": "missing_physics",
            "frame_count": 0,
            "occupancy_map": {},
            "mean_ligand_rmsd_a": None,
            "mean_pocket_rmsf_a": None,
            "distance_rows": [],
        }
    amber_prep = json.loads(amber_prep_path.read_text(encoding="utf-8"))
    relaxation = json.loads(relaxation_path.read_text(encoding="utf-8"))
    prmtop = Path(str(amber_prep["complex_prmtop"]))
    trajectory_path = Path(str(relaxation["trajectory_path"]))
    frames_root = ensure_dir(physics_root / "trajectory_frames")
    frame_paths = _cpptraj_extract_frames(
        prmtop=prmtop,
        trajectory_path=trajectory_path,
        output_root=frames_root,
        stride=int(stage6_5.get("trajectory_frame_stride", 4)),
    )
    if not frame_paths:
        refined_complex = Path(str(relaxation.get("refined_complex_pdb") or ""))
        if refined_complex.exists():
            fallback_path = frames_root / "frame.pdb.1"
            shutil.copyfile(refined_complex, fallback_path)
            frame_paths = [fallback_path]
    if not frame_paths:
        return {
            "system_status": "missing_frames",
            "frame_count": 0,
            "occupancy_map": {},
            "mean_ligand_rmsd_a": None,
            "mean_pocket_rmsf_a": None,
            "distance_rows": [],
        }

    pocket_numbers = {
        number
        for number in [residue_number_from_label(str(label)) for label in list(case_context.get("pocket_residues_universe") or [])]
        if number is not None
    }
    distance_numbers = _select_distance_residues(
        case_context=case_context,
        target_key=str(pair_record["target_key"]),
        top_n=int(stage6_5.get("distance_top_anchor_n", 3)),
    )

    reference_ligand, _, _ = _heavy_atom_coords_by_residue(frame_paths[0])
    residue_series: dict[int, list[np.ndarray]] = {number: [] for number in pocket_numbers}
    ligand_rmsd_values: list[float] = []
    distance_rows: list[dict[str, Any]] = []
    ifp_payloads: list[dict[str, Any]] = []
    total_ns = float(stage6_5.get("local_sampling_ns", 0.2))
    time_values = np.linspace(0.0, total_ns, num=len(frame_paths), endpoint=True) if len(frame_paths) > 1 else np.asarray([0.0])

    for frame_index, frame_path in enumerate(frame_paths, start=1):
        ligand_coords, residue_heavy, residue_ca = _heavy_atom_coords_by_residue(frame_path)
        ligand_rmsd_values.append(_aligned_rmsd(reference_ligand, ligand_coords))
        for number in pocket_numbers:
            if number in residue_ca:
                residue_series[number].append(np.asarray(residue_ca[number], dtype=float))
        ifp_payloads.append(plip_ifp(frame_path))
        if ligand_coords.size > 0:
            for residue_number in distance_numbers:
                residue_coords = residue_heavy.get(residue_number)
                if residue_coords is None or residue_coords.size == 0:
                    continue
                diff = ligand_coords[:, None, :] - residue_coords[None, :, :]
                distance = float(np.sqrt(np.min(np.sum(diff * diff, axis=2))))
                distance_rows.append(
                    {
                        "frame_index": int(frame_index),
                        "time_ns": float(time_values[min(frame_index - 1, len(time_values) - 1)]),
                        "residue_number": int(residue_number),
                        "distance_a": float(distance),
                    }
                )

    residue_rmsf: list[dict[str, Any]] = []
    for residue_number, coords_list in residue_series.items():
        if not coords_list:
            continue
        stacked = np.stack(coords_list, axis=0)
        centered = stacked - stacked.mean(axis=0, keepdims=True)
        rmsf = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        residue_rmsf.append({"residue_number": int(residue_number), "rmsf_a": float(rmsf)})

    return {
        "system_status": "ok",
        "frame_count": int(len(frame_paths)),
        "occupancy_map": ifp_frequency(ifp_payloads),
        "mean_ligand_rmsd_a": float(np.mean(ligand_rmsd_values)) if ligand_rmsd_values else None,
        "mean_pocket_rmsf_a": float(np.mean([row["rmsf_a"] for row in residue_rmsf])) if residue_rmsf else None,
        "residue_rmsf_rows": residue_rmsf,
        "distance_rows": distance_rows,
    }


def plot_distance_timeseries(frame: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    if frame.empty:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No trajectory distance data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return
    targets = list(dict.fromkeys(frame["target_key"].astype(str).tolist()))
    fig, axes = plt.subplots(len(targets), 1, figsize=(9, max(3.0, 2.8 * len(targets))), sharex=True)
    if len(targets) == 1:
        axes = [axes]
    for axis, target_key in zip(axes, targets):
        subset = frame[frame["target_key"].astype(str).eq(target_key)].copy()
        for molecule_label, group in subset.groupby("molecule_label", dropna=False):
            ordered = group.sort_values("time_ns")
            axis.plot(
                ordered["time_ns"],
                ordered["distance_a"],
                linewidth=1.6,
                label=str(molecule_label),
            )
        axis.set_ylabel("Min Dist (A)")
        axis.set_title(str(target_key))
        axis.grid(alpha=0.25, linewidth=0.6)
    axes[-1].set_xlabel("Time (ns)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, min(3, len(labels))))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_rmsd_rmsf(summary_frame: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    if summary_frame.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No RMSD/RMSF data", ha="center", va="center")
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_path, dpi=220)
        plt.close(figure)
        return
    plot_frame = summary_frame.copy()
    plot_frame["system_name"] = plot_frame["molecule_label"].astype(str) + " | " + plot_frame["target_key"].astype(str)
    ordered = plot_frame.sort_values(["target_key", "molecule_label"]).reset_index(drop=True)
    axes[0].barh(ordered["system_name"], ordered["mean_ligand_rmsd_a"].fillna(0.0), color="#4C78A8")
    axes[0].set_title("Ligand RMSD")
    axes[0].set_xlabel("Mean RMSD (A)")
    axes[0].grid(axis="x", alpha=0.25, linewidth=0.6)
    axes[1].barh(ordered["system_name"], ordered["mean_pocket_rmsf_a"].fillna(0.0), color="#F58518")
    axes[1].set_title("Pocket RMSF")
    axes[1].set_xlabel("Mean RMSF (A)")
    axes[1].grid(axis="x", alpha=0.25, linewidth=0.6)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_consistency(frame: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(6.4, 5.2))
    if frame.empty:
        plt.text(0.5, 0.5, "No MM/GBSA consistency data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=220)
        plt.close()
        return
    for molecule_label, group in frame.groupby("molecule_label", dropna=False):
        plt.scatter(
            group["delta_dock_kcal_mol"],
            group["delta_mmgbsa_binding_kcal_mol"],
            s=34,
            alpha=0.8,
            label=str(molecule_label),
        )
    plt.axhline(0.0, color="#999999", linewidth=0.8)
    plt.axvline(0.0, color="#999999", linewidth=0.8)
    plt.xlabel("Delta Dock (kcal/mol)")
    plt.ylabel("Delta MM/GBSA (kcal/mol)")
    plt.title("counter-design step.5 Consistency")
    plt.grid(alpha=0.2, linewidth=0.6)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _stage6_5_pair_root(stage6_5_root: Path, molecule: dict[str, Any], target: dict[str, Any]) -> Path:
    return (
        stage6_5_root
        / "md"
        / "systems"
        / str(molecule["molecule_label"])
        / stable_target_slug(str(target["effect_scope"]), str(target["target_key"]))
    )


def _missing_stage6_5_pair_record(
    *,
    root: Path,
    case_id: str,
    stage6_5_root: Path,
    target: dict[str, Any],
    molecule: dict[str, Any],
    error: str,
) -> dict[str, Any]:
    pair_root = _stage6_5_pair_root(stage6_5_root, molecule, target)
    return {
        "case_id": case_id,
        "target_key": str(target["target_key"]),
        "effect_scope": str(target["effect_scope"]),
        "selection_bucket": str(target.get("selection_bucket") or ""),
        "molecule_label": str(molecule["molecule_label"]),
        "molecule_role": str(molecule["molecule_role"]),
        "candidate_id": str(molecule["candidate_id"]),
        "sample_root": "",
        "stage5_run_root": str(pair_root.relative_to(root)),
        "stage5_status": "failed",
        "stage5_error": str(error),
        "delta_dock_kcal_mol": None,
        "wt_mmgbsa_binding_kcal_mol": None,
        "mt_mmgbsa_binding_kcal_mol": None,
        "delta_mmgbsa_binding_kcal_mol": None,
        "wt_gnina_affinity_kcal_mol": None,
        "mt_gnina_affinity_kcal_mol": None,
        "delta_gnina_affinity_kcal_mol": None,
        "consensus_direction": "",
        "high_uncertainty": True,
    }


def load_stage6_5_cached_pair_records(
    *,
    root: Path,
    case_id: str,
    stage6_5_root: Path,
    targets: list[dict[str, Any]],
    molecules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for target in targets:
        for molecule in molecules:
            pair_root = _stage6_5_pair_root(stage6_5_root, molecule, target)
            summary_path = pair_root / "pair_summary.json"
            if not summary_path.exists():
                records.append(
                    _missing_stage6_5_pair_record(
                        root=root,
                        case_id=case_id,
                        stage6_5_root=stage6_5_root,
                        target=target,
                        molecule=molecule,
                        error="pair_summary_missing",
                    )
                )
                continue
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception as exc:
                records.append(
                    _missing_stage6_5_pair_record(
                        root=root,
                        case_id=case_id,
                        stage6_5_root=stage6_5_root,
                        target=target,
                        molecule=molecule,
                        error=f"pair_summary_unreadable:{type(exc).__name__}",
                    )
                )
                continue
            record = dict(payload)
            record.setdefault("case_id", case_id)
            record.setdefault("target_key", str(target["target_key"]))
            record.setdefault("effect_scope", str(target["effect_scope"]))
            record.setdefault("selection_bucket", str(target.get("selection_bucket") or ""))
            record.setdefault("molecule_label", str(molecule["molecule_label"]))
            record.setdefault("molecule_role", str(molecule["molecule_role"]))
            record.setdefault("candidate_id", str(molecule["candidate_id"]))
            record.setdefault("sample_root", "")
            record.setdefault("stage5_run_root", str(pair_root.relative_to(root)))
            records.append(record)
    return records


def finalize_stage6_5_case_outputs(
    *,
    root: Path,
    case_id: str,
    stage6_5_root: Path,
    case_context: dict[str, Any],
    stage6_5: dict[str, Any],
    validation_targets: list[dict[str, Any]],
    molecules: list[dict[str, Any]],
    energy_targets: list[dict[str, Any]],
    energy_molecules: list[dict[str, Any]],
    pair_records: list[dict[str, Any]],
    energy_pair_records: list[dict[str, Any]],
) -> dict[str, Any]:
    md_root = ensure_dir(stage6_5_root / "md")
    energy_root = ensure_dir(stage6_5_root / "energy")

    occupancy_rows: list[dict[str, Any]] = []
    trajectory_summary_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    for pair_record in pair_records:
        pair_root = root / str(pair_record["stage5_run_root"])
        if str(pair_record.get("stage5_status") or "") != "ok":
            continue
        for pose_set in ["wt", "mt"]:
            summary = summarize_trajectory_system(
                root=root,
                case_context=case_context,
                pair_root=pair_root,
                pose_set=pose_set,
                pair_record=pair_record,
                stage6_5=stage6_5,
            )
            occupancy_rows.append(
                {
                    "case_id": case_id,
                    "target_key": str(pair_record["target_key"]),
                    "effect_scope": str(pair_record["effect_scope"]),
                    "molecule_label": str(pair_record["molecule_label"]),
                    "molecule_role": str(pair_record["molecule_role"]),
                    "system_label": pose_set,
                    "frame_count": int(summary.get("frame_count") or 0),
                    "mean_ligand_rmsd_a": native_value(summary.get("mean_ligand_rmsd_a")),
                    "mean_pocket_rmsf_a": native_value(summary.get("mean_pocket_rmsf_a")),
                    "occupancy_map_json": json.dumps(summary.get("occupancy_map") or {}, ensure_ascii=True, sort_keys=True),
                }
            )
            trajectory_summary_rows.append(
                {
                    "case_id": case_id,
                    "target_key": str(pair_record["target_key"]),
                    "molecule_label": str(pair_record["molecule_label"]),
                    "system_label": pose_set,
                    "mean_ligand_rmsd_a": native_value(summary.get("mean_ligand_rmsd_a")),
                    "mean_pocket_rmsf_a": native_value(summary.get("mean_pocket_rmsf_a")),
                    "frame_count": int(summary.get("frame_count") or 0),
                }
            )
            for row in list(summary.get("distance_rows") or []):
                distance_rows.append(
                    {
                        "case_id": case_id,
                        "target_key": str(pair_record["target_key"]),
                        "molecule_label": str(pair_record["molecule_label"]),
                        "system_label": pose_set,
                        **row,
                    }
                )

    occupancy_frame = pd.DataFrame.from_records(occupancy_rows)
    if not occupancy_frame.empty:
        occupancy_frame = occupancy_frame.sort_values(
            ["target_key", "system_label", "molecule_label"],
            ascending=[True, True, True],
            na_position="last",
        )
    occupancy_frame.to_csv(md_root / "occupancy.csv", index=False)

    distance_frame = pd.DataFrame.from_records(distance_rows)
    distance_plot_frame = (
        distance_frame[distance_frame["system_label"].astype(str).eq(str(stage6_5.get("md_summary_target_system", "mt")))].copy()
        if not distance_frame.empty
        else pd.DataFrame()
    )
    distance_plot_frame = (
        distance_plot_frame.groupby(["target_key", "molecule_label", "frame_index", "time_ns"], dropna=False)["distance_a"]
        .min()
        .reset_index()
        if not distance_plot_frame.empty
        else distance_plot_frame
    )
    plot_distance_timeseries(distance_plot_frame, md_root / "distance_timeseries.png")

    rmsd_rmsf_frame = pd.DataFrame.from_records(trajectory_summary_rows)
    rmsd_rmsf_frame = (
        rmsd_rmsf_frame[rmsd_rmsf_frame["system_label"].astype(str).eq(str(stage6_5.get("md_summary_target_system", "mt")))].copy()
        if not rmsd_rmsf_frame.empty
        else pd.DataFrame()
    )
    plot_rmsd_rmsf(rmsd_rmsf_frame, md_root / "rmsd_rmsf.png")

    energy_records: list[dict[str, Any]] = []
    for record in energy_pair_records:
        energy_records.append(
            {
                "case_id": case_id,
                "target_key": str(record["target_key"]),
                "effect_scope": str(record["effect_scope"]),
                "molecule_label": str(record["molecule_label"]),
                "molecule_role": str(record["molecule_role"]),
                "candidate_id": str(record["candidate_id"]),
                "delta_dock_kcal_mol": native_value(record.get("delta_dock_kcal_mol")),
                "wt_mmgbsa_binding_kcal_mol": native_value(record.get("wt_mmgbsa_binding_kcal_mol")),
                "mt_mmgbsa_binding_kcal_mol": native_value(record.get("mt_mmgbsa_binding_kcal_mol")),
                "delta_mmgbsa_binding_kcal_mol": native_value(record.get("delta_mmgbsa_binding_kcal_mol")),
                "wt_gnina_affinity_kcal_mol": native_value(record.get("wt_gnina_affinity_kcal_mol")),
                "mt_gnina_affinity_kcal_mol": native_value(record.get("mt_gnina_affinity_kcal_mol")),
                "delta_gnina_affinity_kcal_mol": native_value(record.get("delta_gnina_affinity_kcal_mol")),
                "consensus_direction": str(record.get("consensus_direction") or ""),
                "high_uncertainty": bool(record.get("high_uncertainty", False)),
            }
        )
    mmgbsa_frame = pd.DataFrame.from_records(energy_records)
    if not mmgbsa_frame.empty:
        mmgbsa_frame = mmgbsa_frame.sort_values(
            ["target_key", "molecule_label"],
            ascending=[True, True],
            na_position="last",
        )
    mmgbsa_frame.to_csv(energy_root / "mmgbsa_summary.csv", index=False)
    consistency_frame = mmgbsa_frame.dropna(subset=["delta_dock_kcal_mol", "delta_mmgbsa_binding_kcal_mol"]).copy()
    plot_consistency(consistency_frame, energy_root / "consistency_plot.png")

    qc_payload = {
        "case_id": case_id,
        "validation_mode": str(stage6_5.get("validation_mode", "")),
        "validation_note": str(stage6_5.get("validation_note", "")),
        "validation_target_count": int(len(validation_targets)),
        "validation_molecule_count": int(len(molecules)),
        "energy_target_count": int(len(energy_targets)),
        "energy_molecule_count": int(len(energy_molecules)),
        "md_pair_count": int(len(pair_records)),
        "md_pair_success_count": int(sum(1 for row in pair_records if str(row.get("stage5_status") or "") == "ok")),
        "trajectory_system_count": int(len(occupancy_frame)),
        "trajectory_nonempty_system_count": int(occupancy_frame["frame_count"].fillna(0).gt(0).sum()) if not occupancy_frame.empty else 0,
        "mmgbsa_point_count": int(len(mmgbsa_frame)),
        "mmgbsa_available_count": int(mmgbsa_frame["delta_mmgbsa_binding_kcal_mol"].notna().sum()) if not mmgbsa_frame.empty else 0,
        "dock_vs_mmgbsa_pearson_r": native_value(
            consistency_frame["delta_dock_kcal_mol"].corr(consistency_frame["delta_mmgbsa_binding_kcal_mol"], method="pearson")
        ) if len(consistency_frame) >= 2 else None,
        "dock_vs_mmgbsa_spearman_r": native_value(
            consistency_frame["delta_dock_kcal_mol"].corr(consistency_frame["delta_mmgbsa_binding_kcal_mol"], method="spearman")
        ) if len(consistency_frame) >= 2 else None,
        "generated_at": iso_now(),
    }
    json_dump(stage6_5_root / "stage6_5_qc.json", qc_payload)
    return qc_payload


def finalize_existing_stage6_5_case(
    *,
    root: Path,
    config: dict[str, Any],
    case_entry: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(case_entry["case_id"])
    stage6_5 = dict(config["stage6_5"])
    stage6_5_root = stage6_5_case_root(root, case_id, stage6_5)
    manifest_path = stage6_5_root / "validation_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"counter-design step.5 validation manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_targets = [dict(item) for item in list(manifest.get("targets") or []) if isinstance(item, dict)]
    molecules = [dict(item) for item in list(manifest.get("molecules") or []) if isinstance(item, dict)]
    energy_targets = [dict(item) for item in list(manifest.get("energy_targets") or []) if isinstance(item, dict)]
    energy_molecules = [dict(item) for item in list(manifest.get("energy_molecules") or []) if isinstance(item, dict)]
    case_context = stage5_case_context(root, case_id)
    pair_records = load_stage6_5_cached_pair_records(
        root=root,
        case_id=case_id,
        stage6_5_root=stage6_5_root,
        targets=validation_targets,
        molecules=molecules,
    )
    energy_pair_records = load_stage6_5_cached_pair_records(
        root=root,
        case_id=case_id,
        stage6_5_root=stage6_5_root,
        targets=energy_targets,
        molecules=energy_molecules,
    )
    qc_payload = finalize_stage6_5_case_outputs(
        root=root,
        case_id=case_id,
        stage6_5_root=stage6_5_root,
        case_context=case_context,
        stage6_5=stage6_5,
        validation_targets=validation_targets,
        molecules=molecules,
        energy_targets=energy_targets,
        energy_molecules=energy_molecules,
        pair_records=pair_records,
        energy_pair_records=energy_pair_records,
    )
    print(
        f"[stage6_5] case={case_id} finalized from cache md_pair_success_count={qc_payload['md_pair_success_count']} "
        f"mmgbsa_available_count={qc_payload['mmgbsa_available_count']}",
        flush=True,
    )
    return qc_payload


def run_stage6_5_case(
    *,
    root: Path,
    config: dict[str, Any],
    case_entry: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(case_entry["case_id"])
    stage5 = stage5_for_case(dict(config["stage5"]), case_id)
    stage6 = dict(config["stage6"])
    stage6_case_overrides = dict(stage6.get("case_overrides", {})).get(case_id, {})
    stage6.update(stage6_case_overrides if isinstance(stage6_case_overrides, dict) else {})
    stage6_5 = dict(config["stage6_5"])
    print(f"[stage6_5] case={case_id} start", flush=True)
    stage6["local_sampling_ns"] = float(stage6_5.get("local_sampling_ns", stage5.get("local_sampling_ns", 0.2)))
    stage6["local_sampling_backbone_restraint_kcal_mol_a2"] = float(
        stage6_5.get(
            "local_sampling_backbone_restraint_kcal_mol_a2",
            stage5.get("local_sampling_backbone_restraint_kcal_mol_a2", 1.0),
        )
    )
    stage5["local_sampling_enabled"] = True
    stage5["local_sampling_ns"] = float(stage6_5.get("local_sampling_ns", stage5.get("local_sampling_ns", 0.2)))
    stage5["local_sampling_backbone_restraint_kcal_mol_a2"] = float(
        stage6_5.get(
            "local_sampling_backbone_restraint_kcal_mol_a2",
            stage5.get("local_sampling_backbone_restraint_kcal_mol_a2", 1.0),
        )
    )
    stage5["require_gnina_for_stage5_scoring"] = bool(stage6_5.get("require_gnina_for_stage5_scoring", False))

    manifest = load_case_manifest(case_entry, root=root)
    combo_panel = read_csv_optional(root / "outputs" / "case_manifests" / "combo_panel.csv")
    leaderboard = load_stage6_leaderboard(root / "outputs" / case_id / "stage6")
    stage6_5_root = stage6_5_case_root(root, case_id, stage6_5)
    ensure_dir(stage6_5_root / "md")
    ensure_dir(stage6_5_root / "energy")
    ligands_root = ensure_dir(stage6_5_root / "ligands")
    history_frame = load_stage6_5_candidate_history(
        root=root,
        case_id=case_id,
        stage6_5=stage6_5,
    )
    if not history_frame.empty and not leaderboard.empty:
        leaderboard = leaderboard.merge(history_frame, on="candidate_id", how="left")
        for column in [
            "stage6_5_prior_md_success_pair_count",
            "stage6_5_prior_mmgbsa_attempt_count",
            "stage6_5_prior_mmgbsa_available_count",
        ]:
            if column in leaderboard.columns:
                leaderboard[column] = pd.to_numeric(leaderboard[column], errors="coerce").fillna(0).astype(int)
        for column in [
            "stage6_5_prior_attempted",
            "stage6_5_prior_success_flag",
            "stage6_5_prior_failure_only",
        ]:
            if column in leaderboard.columns:
                leaderboard[column] = leaderboard[column].fillna(False).astype(bool)
        history_frame.sort_values(
            ["stage6_5_prior_success_flag", "stage6_5_prior_mmgbsa_available_count", "candidate_id"],
            ascending=[False, False, True],
            na_position="last",
        ).to_csv(stage6_5_root / "molecule_history_audit.csv", index=False)
    candidate_filter, candidate_probe_rows = _build_candidate_parameter_filter(
        root=root,
        case_id=case_id,
        stage6_5_root=stage6_5_root,
        stage6_5=stage6_5,
    )
    case_context = stage5_case_context(root, case_id)
    hiv_reference = build_hiv_reference(
        root=root,
        case_entry=case_entry,
        stage2=dict(config["stage2"]),
        stage3_5=dict(config["stage3_5"]),
    )

    validation_targets = select_validation_targets(
        case_entry=case_entry,
        manifest=manifest,
        combo_panel=combo_panel,
        stage6_5=stage6_5,
    )
    molecules, energy_molecules = select_validation_molecules(
        case_entry=case_entry,
        leaderboard=leaderboard,
        stage6_5=stage6_5,
        candidate_filter=candidate_filter,
    )
    if candidate_probe_rows:
        selected_labels: dict[str, list[str]] = {}
        for item in molecules + energy_molecules:
            candidate_id_text = str(item.get("candidate_id") or "")
            if candidate_id_text in {"", "lead"}:
                continue
            selected_labels.setdefault(candidate_id_text, []).append(str(item["molecule_label"]))
        for row in candidate_probe_rows:
            labels = selected_labels.get(str(row.get("candidate_id") or ""), [])
            row["selected_labels"] = ";".join(labels)
            row["selected_for_validation"] = bool(any(label in {"robust_best", "wt_best_control"} for label in labels))
            row["selected_for_energy"] = bool(any(label.startswith("energy_") for label in labels))
        pd.DataFrame.from_records(candidate_probe_rows).sort_values(
            ["available", "objective_reward", "candidate_id"],
            ascending=[False, False, True],
            na_position="last",
        ).to_csv(stage6_5_root / "molecule_probe_audit.csv", index=False)
    energy_targets = select_energy_targets(
        case_entry=case_entry,
        manifest=manifest,
        combo_panel=combo_panel,
        stage6_5=stage6_5,
    )
    validation_manifest = {
        "case_id": case_id,
        "stage6_source": str(root / "outputs" / case_id / "stage6"),
        "validation_mode": str(stage6_5.get("validation_mode", "")),
        "validation_note": str(stage6_5.get("validation_note", "")),
        "targets": validation_targets,
        "molecules": molecules,
        "energy_targets": energy_targets,
        "energy_molecules": energy_molecules,
        "generated_at": iso_now(),
    }
    json_dump(stage6_5_root / "validation_manifest.json", validation_manifest)

    samples_root = ensure_dir(stage6_5_root / "samples")
    seen_target_keys: set[tuple[str, str]] = set()
    for target in validation_targets + energy_targets:
        target_identity = (str(target["effect_scope"]), str(target["target_key"]))
        if target_identity in seen_target_keys:
            continue
        seen_target_keys.add(target_identity)
        sample_root, model_kind = materialize_validation_sample(
            root=root,
            case_id=case_id,
            target=target,
            samples_root=samples_root,
        )
        print(
            f"[stage6_5] prepared sample case={case_id} target={target['target_key']} "
            f"sample_root={sample_root} model_kind={model_kind}",
            flush=True,
        )

    ligand_paths = {
        str(molecule["molecule_label"]): _copy_or_write_ligand(
            root=root,
            case_id=case_id,
            molecule=molecule,
            ligands_root=ligands_root,
        )
        for molecule in {item["molecule_label"]: item for item in molecules + energy_molecules}.values()
    }

    validation_jobs: list[dict[str, Any]] = []
    for target in validation_targets:
        for molecule in molecules:
            validation_jobs.append(
                {
                    "root": root,
                    "case_id": case_id,
                    "case_entry": case_entry,
                    "target": target,
                    "molecule": molecule,
                    "ligand_sdf": ligand_paths[str(molecule["molecule_label"])],
                    "stage5": stage5,
                    "stage6": stage6,
                    "stage6_5_root": stage6_5_root,
                    "hiv_reference": hiv_reference,
                }
            )
    pair_records = _execute_stage6_5_jobs(
        jobs=validation_jobs,
        stage6_5=stage6_5,
        stage5=stage5,
        label=f"{case_id}:validation",
    )

    energy_jobs: list[dict[str, Any]] = []
    for target in energy_targets:
        for molecule in energy_molecules:
            energy_jobs.append(
                {
                    "root": root,
                    "case_id": case_id,
                    "case_entry": case_entry,
                    "target": target,
                    "molecule": molecule,
                    "ligand_sdf": ligand_paths[str(molecule["molecule_label"])],
                    "stage5": stage5,
                    "stage6": stage6,
                    "stage6_5_root": stage6_5_root,
                    "hiv_reference": hiv_reference,
                }
            )
    energy_pair_records = _execute_stage6_5_jobs(
        jobs=energy_jobs,
        stage6_5=stage6_5,
        stage5=stage5,
        label=f"{case_id}:energy",
    )
    qc_payload = finalize_stage6_5_case_outputs(
        root=root,
        case_id=case_id,
        stage6_5_root=stage6_5_root,
        case_context=case_context,
        stage6_5=stage6_5,
        validation_targets=validation_targets,
        molecules=molecules,
        energy_targets=energy_targets,
        energy_molecules=energy_molecules,
        pair_records=pair_records,
        energy_pair_records=energy_pair_records,
    )
    print(
        f"[stage6_5] case={case_id} finished md_pair_success_count={qc_payload['md_pair_success_count']} "
        f"mmgbsa_available_count={qc_payload['mmgbsa_available_count']}",
        flush=True,
    )
    return qc_payload
