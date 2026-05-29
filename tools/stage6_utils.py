#!/usr/bin/env python3
"""counter-design step counter-design helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import DataStructs
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from tools.filters import apply_prefilters, descriptor_payload
from tools.runtime import ensure_dir, iso_now, json_dump, load_yaml, software_version
from tools.stage35_utils import (
    PLIP_INTERACTION_TYPES,
    build_box_from_ligand_coords,
    classify_hiv_pose,
    ensure_ligand_3d,
    load_rdkit_molecule,
    merge_pdb_fragments,
    mol_coordinates,
    plip_ifp,
    pose_pdb_from_sdf,
    prepare_ligand_pdbqt,
    prepare_receptor_pdbqt,
    run_pdbfixer,
    run_vina_redocking,
    save_chain_protein,
    save_chain_set_protein,
    select_best_pose,
    standardize_reference_ligand,
    vina_result_rows,
)
from tools.stage5_physics import gnina_score_only, prepare_amber_complex, run_amber_relaxation
from tools.stage5_utils import (
    build_hiv_reference,
    build_stage5_target_panel,
    first_protein_chain_id,
    local_pocket_metrics,
    native_value,
    occupancy_frequency_payload,
    occupancy_shift_metrics,
    relative_path,
    stable_target_slug,
    stage5_reference_ligand_input,
    write_json,
)
from tools.stage6_calibrator import fit_case_calibrator, load_case_calibrator
from tools.stage6_oracle_v2 import load_stage6_oracle_v2
from tools.stage6_postmortem import stage6_executability_metrics
from tools.stage6_receptor_ensemble import (
    aggregate_ensemble_value,
    load_receptor_ensemble_members,
    select_representative_member,
)
from tools.stage6_reward_v2 import (
    LAYER_NAMES,
    alt_anchor_score,
    oracle_uncertainty_score,
    reward_v2_components,
    reward_v2_layer_weights,
    weighted_keep_ifp,
)

EDIT_FAMILIES = (
    "ADD",
    "REPLACE",
    "DELETE",
    "HETERO_SWAP",
    "RING_EDIT",
    "RIGIDIFY_LINKER",
    "VECTOR_FLIP",
    "BIOISOSTERE_SWAP",
    "RING_EXPANSION",
    "RING_CONTRACTION",
    "HETEROARYL_SWAP",
    "BACKBONE_SEEKER",
    "WATER_BRIDGE_PROBE",
    "HALOGEN_SCAN",
    "SMALL_POLAR_SCAN",
    "LINKER_HETERO_SCAN",
    "N_METHYL_SCAN",
    "CONSTRAINED_TAIL_TRIM",
)
FRAGMENT_FAMILY_PATTERNS = {
    "F": "add_hydrophobic_cap",
    "Cl": "add_hydrophobic_cap",
    "Br": "add_hydrophobic_cap",
    "C": "add_hydrophobic_cap",
    "OC": "add_hbond_acceptor",
    "O": "add_polar_cap",
    "N": "add_polar_cap",
    "C#N": "add_hbond_acceptor",
}
REPLACE_FAMILY_PATTERNS = {
    "F": "replace_leaf_with_hydrophobe",
    "Cl": "replace_leaf_with_hydrophobe",
    "Br": "replace_leaf_with_hydrophobe",
    "C": "replace_leaf_with_hydrophobe",
    "OC": "replace_leaf_with_polar",
    "O": "replace_leaf_with_polar",
    "N": "replace_leaf_with_polar",
    "C#N": "replace_leaf_with_polar",
}
HALOGEN_FRAGMENTS = ("F", "Cl")
SMALL_POLAR_FRAGMENTS = ("O", "OC", "N", "C#N")
N_METHYL_SMILES = "C"
POCKET_RESIDUE_CLASSES = {
    "acidic": {"ASP", "GLU"},
    "aromatic": {"HIS", "PHE", "TRP", "TYR"},
    "basic": {"ARG", "HIS", "LYS"},
    "hydrophobic": {"ALA", "ILE", "LEU", "MET", "PRO", "VAL"},
    "polar": {"ASN", "GLN", "SER", "THR", "TYR"},
}


def read_csv_optional(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns or [])
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns or [])


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def stage6_software_versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "vina": software_version("vina"),
        "gnina": software_version("gnina", ["--version"]),
        "obabel": software_version("obabel"),
        "plip": software_version("plip", ["--version"]),
    }


def canonical_smiles(mol: Chem.Mol | str) -> str:
    if isinstance(mol, str):
        RDLogger.DisableLog("rdApp.*")
        try:
            molecule = Chem.MolFromSmiles(str(mol))
        finally:
            RDLogger.EnableLog("rdApp.*")
    else:
        molecule = Chem.Mol(mol)
    if molecule is None:
        raise ValueError("Unable to canonicalize invalid molecule.")
    return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(molecule)), canonical=True)


def candidate_id(smiles: str) -> str:
    return hashlib.sha1(canonical_smiles(smiles).encode("utf-8")).hexdigest()[:12]


def residue_number_from_label(label: str) -> int | None:
    match = re.search(r"(-?\d+)$", str(label))
    if not match:
        return None
    return int(match.group(1))


def residue_name_from_label(label: str) -> str | None:
    match = re.match(r"[^:]+:(?P<name>[A-Z]{3})-?\d+", str(label).strip())
    if not match:
        return None
    return str(match.group("name"))


def parse_json_payload(text: str | None) -> Any:
    if text in {None, "", "nan"}:
        return None
    return json.loads(str(text))


def _binding_score(candidate_affinity: float | None, lead_affinity: float | None, scale: float) -> float:
    if candidate_affinity is None or lead_affinity is None:
        return 0.0
    delta = float(lead_affinity) - float(candidate_affinity)
    return float(1.0 / (1.0 + math.exp(-delta / max(scale, 1e-6))))


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values or not weights:
        return 0.0
    total = sum(float(weight) for weight in weights)
    if total <= 0.0:
        return 0.0
    return float(sum(float(value) * float(weight) for value, weight in zip(values, weights)) / total)


def weighted_cvar(values: list[float], weights: list[float], tail_fraction: float) -> float:
    if not values or not weights:
        return 0.0
    paired = sorted(zip(values, weights), key=lambda item: float(item[0]))
    threshold_weight = max(1e-6, float(sum(weights)) * float(tail_fraction))
    cumulative = 0.0
    contribution = 0.0
    for value, weight in paired:
        remaining = threshold_weight - cumulative
        if remaining <= 0.0:
            break
        used = min(float(weight), float(remaining))
        contribution += float(value) * used
        cumulative += used
    if cumulative <= 0.0:
        return 0.0
    return float(contribution / cumulative)


def uncertainty_heavy_target_score(
    *,
    calibrated_score: float | None,
    raw_docking_score: float | None,
    keep_ifp_value: float,
    anchor_loss_inverse: float,
    use_raw_docking_floor: bool = True,
) -> float:
    base_candidates = [float(value) for value in [calibrated_score, raw_docking_score] if value is not None]
    if not base_candidates:
        base_score = 0.0
    elif bool(use_raw_docking_floor) and calibrated_score is not None and raw_docking_score is not None:
        base_score = float(max(float(calibrated_score), float(raw_docking_score)))
    else:
        base_score = float(base_candidates[0] if len(base_candidates) == 1 else float(calibrated_score))
    return float(0.4 * base_score + 0.4 * float(keep_ifp_value) + 0.2 * float(anchor_loss_inverse))


def objective_guardrail_adjustments(
    *,
    objective_name: str,
    robust_score: float,
    s_wt: float,
    compensation_gain: float,
    objective_guardrails: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = dict(objective_guardrails or {})
    adjustments = {
        "objective_guardrail_active": False,
        "objective_guardrail_floor_violation": False,
        "objective_guardrail_penalty": 0.0,
        "objective_guardrail_robust_score_floor": native_value(payload.get("robust_score_floor")),
        "objective_guardrail_s_wt_floor": native_value(payload.get("s_wt_floor")),
        "objective_guardrail_robust_score_gap": 0.0,
        "objective_guardrail_s_wt_gap": 0.0,
        "effective_compensation_gain": float(compensation_gain),
    }
    if str(objective_name) != "robust" or not payload or not bool(payload.get("enabled", True)):
        return adjustments

    robust_score_floor = native_value(payload.get("robust_score_floor"))
    s_wt_floor = native_value(payload.get("s_wt_floor"))
    robust_score_gap = float(max(0.0, float(robust_score_floor) - float(robust_score))) if robust_score_floor is not None else 0.0
    s_wt_gap = float(max(0.0, float(s_wt_floor) - float(s_wt))) if s_wt_floor is not None else 0.0
    floor_violation = bool(robust_score_gap > 0.0 or s_wt_gap > 0.0)
    effective_compensation_gain = float(compensation_gain)
    if floor_violation and bool(payload.get("disable_compensation_below_floor", True)):
        effective_compensation_gain = 0.0
    guardrail_penalty = (
        float(payload.get("robust_score_penalty_weight", 2.0)) * robust_score_gap
        + float(payload.get("s_wt_penalty_weight", 2.0)) * s_wt_gap
    )
    adjustments.update(
        {
            "objective_guardrail_active": True,
            "objective_guardrail_floor_violation": floor_violation,
            "objective_guardrail_penalty": float(guardrail_penalty),
            "objective_guardrail_robust_score_gap": float(robust_score_gap),
            "objective_guardrail_s_wt_gap": float(s_wt_gap),
            "effective_compensation_gain": float(effective_compensation_gain),
        }
    )
    return adjustments


def objective_guardrails_from_payload(
    payload: dict[str, Any] | pd.Series,
    *,
    case_context: dict[str, Any],
) -> dict[str, Any]:
    case_guardrails = dict(case_context.get("objective_guardrails") or {})
    if case_guardrails:
        return case_guardrails
    row = dict(payload) if isinstance(payload, dict) else dict(payload.to_dict())
    if not bool(row.get("objective_guardrail_active", False)):
        return {}
    return {
        "enabled": bool(row.get("objective_guardrail_active", False)),
        "robust_score_floor": native_value(row.get("objective_guardrail_robust_score_floor")),
        "s_wt_floor": native_value(row.get("objective_guardrail_s_wt_floor")),
        "robust_score_penalty_weight": native_value(row.get("objective_guardrail_robust_score_penalty_weight")) or 0.0,
        "s_wt_penalty_weight": native_value(row.get("objective_guardrail_s_wt_penalty_weight")) or 0.0,
        "disable_compensation_below_floor": bool(row.get("objective_guardrail_disable_compensation_below_floor", False)),
    }


def ifp_type_weights(stage6: dict[str, Any]) -> dict[str, float]:
    raw = dict(stage6.get("ifp_type_weights", {}))
    defaults = {
        "metal_complex": 1.4,
        "salt_bridge": 1.3,
        "hydrogen_bond": 1.0,
        "pi_cation": 0.9,
        "pi_stacking": 0.8,
        "hydrophobic": 0.6,
    }
    return {name: float(raw.get(name, defaults.get(name, 1.0))) for name in PLIP_INTERACTION_TYPES}


def ifp_residue_type_counts(ifp_payload: dict[str, Any], residues: set[str] | None = None) -> dict[str, dict[str, float]]:
    residue_filter = None if residues is None else {str(value) for value in residues}
    payload = {kind: {} for kind in PLIP_INTERACTION_TYPES}
    for row in list(ifp_payload.get("interactions") or []):
        kind = str(row.get("interaction_type") or "")
        residue = str(row.get("residue_label") or "")
        if kind not in payload or not residue:
            continue
        if residue_filter is not None and residue not in residue_filter:
            continue
        payload[kind][residue] = float(payload[kind].get(residue, 0.0) + 1.0)
    return payload


def cosine_similarity(lhs: dict[str, float], rhs: dict[str, float]) -> float:
    if not lhs and not rhs:
        return 1.0
    keys = set(lhs) | set(rhs)
    numerator = sum(float(lhs.get(key, 0.0)) * float(rhs.get(key, 0.0)) for key in keys)
    lhs_norm = math.sqrt(sum(float(value) ** 2 for value in lhs.values()))
    rhs_norm = math.sqrt(sum(float(value) ** 2 for value in rhs.values()))
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return 0.0
    return float(numerator / (lhs_norm * rhs_norm))


def weighted_ifp_cosine(
    *,
    baseline_ifp: dict[str, Any],
    candidate_ifp: dict[str, Any],
    residues: set[str],
    stage6: dict[str, Any],
) -> float:
    weights = ifp_type_weights(stage6)
    baseline_counts = ifp_residue_type_counts(baseline_ifp, residues)
    candidate_counts = ifp_residue_type_counts(candidate_ifp, residues)
    numerator = 0.0
    denominator = 0.0
    for interaction_type in PLIP_INTERACTION_TYPES:
        weight = float(weights.get(interaction_type, 1.0))
        numerator += weight * cosine_similarity(baseline_counts[interaction_type], candidate_counts[interaction_type])
        denominator += weight
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def anchor_contact_count(candidate_ifp: dict[str, Any], anchor_residues: set[str]) -> int:
    return int(len(set(candidate_ifp.get("residue_set", [])) & set(anchor_residues)))


def contact_score(candidate_ifp: dict[str, Any], residues: set[str], stage6: dict[str, Any]) -> float:
    weights = ifp_type_weights(stage6)
    per_type = ifp_residue_type_counts(candidate_ifp, residues)
    total = 0.0
    for interaction_type in PLIP_INTERACTION_TYPES:
        total += float(weights.get(interaction_type, 1.0)) * float(sum(per_type[interaction_type].values()))
    return float(total)


def residue_contact_mass(candidate_ifp: dict[str, Any], residues: set[str], stage6: dict[str, Any]) -> Counter[str]:
    weights = ifp_type_weights(stage6)
    mass: Counter[str] = Counter()
    for row in list(candidate_ifp.get("interactions") or []):
        residue = str(row.get("residue_label") or "")
        interaction_type = str(row.get("interaction_type") or "")
        if not residue or residue not in residues:
            continue
        mass[residue] += float(weights.get(interaction_type, 1.0))
    return mass


def hotspot_fraction(
    candidate_ifp: dict[str, Any],
    hotspot_residues: set[str],
    pocket_residues: set[str],
    stage6: dict[str, Any],
) -> float:
    mass = residue_contact_mass(candidate_ifp, pocket_residues, stage6)
    total = float(sum(mass.values()))
    if total <= 0.0:
        return 1.0
    hotspot_mass = float(sum(value for residue, value in mass.items() if residue in hotspot_residues))
    return float(hotspot_mass / total)


def anchor_diversity(candidate_ifp: dict[str, Any], hotspot_residues: set[str], stage6: dict[str, Any] | None = None) -> float:
    if not hotspot_residues:
        return 1.0
    if stage6 is None:
        residue_counts = Counter()
        for row in list(candidate_ifp.get("interactions") or []):
            residue = str(row.get("residue_label") or "")
            if residue in hotspot_residues:
                residue_counts[residue] += 1
    else:
        residue_counts = residue_contact_mass(candidate_ifp, hotspot_residues, stage6)
    if not residue_counts:
        return 0.0
    total = float(sum(residue_counts.values()))
    entropy = 0.0
    for count in residue_counts.values():
        probability = float(count) / total
        entropy -= probability * math.log(probability)
    normalizer = math.log(max(2, len(residue_counts)))
    return float(entropy / normalizer) if normalizer > 0.0 else 1.0


def dep_score(candidate_ifp: dict[str, Any], hotspot_residues: set[str], pocket_residues: set[str], stage6: dict[str, Any]) -> float:
    mass = residue_contact_mass(candidate_ifp, pocket_residues, stage6)
    total = float(sum(mass.values()))
    if total <= 0.0:
        return 1.0
    hotspot_mass_fraction = hotspot_fraction(candidate_ifp, hotspot_residues, pocket_residues, stage6)
    probabilities = [float(value) / total for value in mass.values() if float(value) > 0.0]
    if not probabilities:
        return 1.0
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    diversity_all = float(entropy / max(math.log(max(2, len(probabilities))), 1.0e-6))
    return float(0.5 * hotspot_mass_fraction + 0.5 * (1.0 - diversity_all))


def jaccard_loss(lhs: set[str], rhs: set[str]) -> float:
    union = set(lhs) | set(rhs)
    if not union:
        return 0.0
    intersection = set(lhs) & set(rhs)
    return float(1.0 - len(intersection) / len(union))


def ifp_occupancy_shift(
    baseline_ifp: dict[str, Any],
    candidate_ifp: dict[str, Any],
    anchor_residues: set[str],
) -> dict[str, Any]:
    baseline_map = {str(label): 1.0 for label in list(baseline_ifp.get("residue_set") or [])}
    candidate_map = {str(label): 1.0 for label in list(candidate_ifp.get("residue_set") or [])}
    union = sorted(set(baseline_map) | set(candidate_map))
    if not union:
        return {
            "ifp_occupancy_shift_mean_abs": 0.0,
            "ifp_occupancy_anchor_loss": 0.0,
        }
    diffs = [abs(float(candidate_map.get(label, 0.0)) - float(baseline_map.get(label, 0.0))) for label in union]
    anchor_diffs = [
        max(0.0, float(baseline_map.get(label, 0.0)) - float(candidate_map.get(label, 0.0)))
        for label in sorted(anchor_residues)
        if label in baseline_map or label in candidate_map
    ]
    return {
        "ifp_occupancy_shift_mean_abs": float(sum(diffs) / len(diffs)),
        "ifp_occupancy_anchor_loss": float(sum(anchor_diffs) / len(anchor_diffs)) if anchor_diffs else 0.0,
    }


def lost_anchor_labels(candidate_ifp: dict[str, Any], anchor_residues: set[str]) -> list[str]:
    candidate_labels = set(str(value) for value in list(candidate_ifp.get("residue_set") or []))
    return sorted(set(anchor_residues) - candidate_labels)


def anchor_loss_fraction(candidate_ifp: dict[str, Any], anchor_residues: set[str]) -> float:
    if not anchor_residues:
        return 0.0
    missing = lost_anchor_labels(candidate_ifp, anchor_residues)
    return float(len(missing) / max(1, len(anchor_residues)))


def new_nonhotspot_contact_metrics(
    *,
    baseline_ifp: dict[str, Any],
    candidate_ifp: dict[str, Any],
    hotspot_residues: set[str],
    stage6: dict[str, Any],
) -> dict[str, Any]:
    baseline_labels = set(str(value) for value in list(baseline_ifp.get("residue_set") or []))
    candidate_labels = set(str(value) for value in list(candidate_ifp.get("residue_set") or []))
    new_nonhotspot_residues = sorted((candidate_labels - baseline_labels) - set(hotspot_residues))
    return {
        "new_nonhotspot_contact_score": float(contact_score(candidate_ifp, set(new_nonhotspot_residues), stage6)),
        "new_nonhotspot_residue_count": int(len(new_nonhotspot_residues)),
        "new_nonhotspot_residues": new_nonhotspot_residues,
    }


def pocket_profile(
    *,
    residue_labels: list[str],
    partner_chain_residues: list[str],
) -> dict[str, Any]:
    unique_labels = sorted({str(label) for label in residue_labels if str(label)})
    class_counts: Counter[str] = Counter()
    residue_name_counts: Counter[str] = Counter()
    for label in unique_labels:
        residue_name = residue_name_from_label(label)
        if residue_name is None:
            continue
        residue_name_counts[residue_name] += 1
        for class_name, members in POCKET_RESIDUE_CLASSES.items():
            if residue_name in members:
                class_counts[class_name] += 1
    dominant_classes = [
        class_name
        for class_name, _ in sorted(class_counts.items(), key=lambda item: (-int(item[1]), item[0]))
        if int(class_counts[class_name]) > 0
    ]
    return {
        "residue_count": int(len(unique_labels)),
        "partner_chain_residue_count": int(len({str(label) for label in partner_chain_residues if str(label)})),
        "class_counts": {class_name: int(class_counts.get(class_name, 0)) for class_name in sorted(POCKET_RESIDUE_CLASSES)},
        "dominant_classes": dominant_classes[:3],
        "top_residue_names": [
            residue_name
            for residue_name, _ in sorted(residue_name_counts.items(), key=lambda item: (-int(item[1]), item[0]))[:6]
        ],
        "top_residue_labels": unique_labels[:12],
    }


def case_specific_action_hints(
    *,
    case_entry: dict[str, Any],
    lead_descriptors: dict[str, Any],
    pocket_profile_payload: dict[str, Any],
    partner_chain_positions: list[int],
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []

    def add(edit_family: str, pattern: str, fragment: str, weight: float, rationale: str) -> None:
        hints.append(
            {
                "edit_family": str(edit_family),
                "pattern": str(pattern),
                "fragment": str(fragment),
                "weight": float(weight),
                "rationale": str(rationale),
            }
        )

    target_domain = str(case_entry.get("target_domain") or "").lower()
    combo_dense = bool(str(case_entry.get("evaluation_unit") or "") == "observed_combo")
    class_counts = dict(pocket_profile_payload.get("class_counts") or {})
    partner_count = int(pocket_profile_payload.get("partner_chain_residue_count") or 0)
    rotatable = int(lead_descriptors.get("rotatable_bonds") or 0)

    if target_domain == "rt":
        add("SMALL_POLAR_SCAN", "small_polar_scan", "O", 3.0, "RT NNRTI pocket favors compact polar compensation.")
        add("LINKER_HETERO_SCAN", "linker_hetero_scan", "N", 2.5, "RT pocket often benefits from localized polarity tuning.")
        add("RING_EDIT", "aza_scan_ring", "N", 2.0, "Aza scans can redistribute hotspot pressure in NNRTI-like chemotypes.")
        add("WATER_BRIDGE_PROBE", "water_bridge_probe", "OC", 2.2, "RT pocket often benefits from water-mediated polar reach.")
        add("BACKBONE_SEEKER", "backbone_seeker", "O", 1.8, "RT rescue often needs backbone-facing donor/acceptor probes.")
        if partner_count > 0 or partner_chain_positions:
            add("SMALL_POLAR_SCAN", "small_polar_scan", "OC", 2.5, "Partner-chain-sensitive RT cases need peripheral polar reach.")
            add("N_METHYL_SCAN", "n_methyl_scan", "-C", 1.2, "Trim N-methyl bulk before probing partner-chain compensation.")
            add("BACKBONE_SEEKER", "backbone_seeker", "N", 1.6, "Partner-chain-sensitive RT cases benefit from directed backbone seekers.")
    if combo_dense:
        add("SMALL_POLAR_SCAN", "small_polar_scan", "N", 2.2, "Combo-dense panels reward compact polarity over bulk.")
        add("LINKER_HETERO_SCAN", "linker_hetero_scan", "N", 1.8, "Combo panels benefit from linker polarity retuning.")
        add("WATER_BRIDGE_PROBE", "water_bridge_probe", "O", 1.6, "Combo escape often uses water-mediated compensation.")
    if int(class_counts.get("aromatic", 0)) >= 3 and target_domain != "rt":
        add("HALOGEN_SCAN", "halogen_scan", "F", 1.8, "Aromatic pockets can exploit compact halogen fills.")
        add("HALOGEN_SCAN", "halogen_scan", "Cl", 1.2, "Compact chlorine scans probe shallow hydrophobic space.")
        add("HETEROARYL_SWAP", "heteroaryl_swap", "N", 1.4, "Aromatic pockets can benefit from heteroaryl polarity tuning.")
    if int(class_counts.get("acidic", 0)) + int(class_counts.get("polar", 0)) >= 3:
        add("SMALL_POLAR_SCAN", "small_polar_scan", "N", 1.6, "Polar or acidic pockets favor donor/acceptor probes.")
        add("WATER_BRIDGE_PROBE", "water_bridge_probe", "OC", 1.5, "Polar pockets can stabilize water-bridge probes.")
    if rotatable >= 6:
        add("CONSTRAINED_TAIL_TRIM", "constrained_tail_trim", "", 1.6, "Flexible leads should trim tail entropy before expansion.")
        add("N_METHYL_SCAN", "n_methyl_scan", "-C", 1.0, "Reducing N-methyl bulk can recover pocket-adapted conformations.")
        add("RIGIDIFY_LINKER", "rigidify_linker", "", 1.8, "Flexible series should rigidify solvent-exposed linkers first.")
    if target_domain.endswith("kinase"):
        add("CONSTRAINED_TAIL_TRIM", "constrained_tail_trim", "", 1.4, "Kinase pockets often reward localized steric cleanup.")
        add("LINKER_HETERO_SCAN", "linker_hetero_scan", "N", 1.3, "Kinase hinge-facing linkers benefit from hetero tuning.")
        add("BIOISOSTERE_SWAP", "bioisostere_swap", "N", 1.5, "Kinase series often need linker bioisosteres instead of small caps.")
        add("BACKBONE_SEEKER", "backbone_seeker", "N", 1.3, "Kinase hinge rescue often comes from backbone-seeking donors.")

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for hint in sorted(
        hints,
        key=lambda item: (-float(item["weight"]), str(item["edit_family"]), str(item["pattern"]), str(item["fragment"])),
    ):
        signature = (str(hint["edit_family"]), str(hint["pattern"]), str(hint["fragment"]))
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(hint)
    return deduped[:10]


def target_positions(target_key: str) -> list[int]:
    return sorted({int(value) for value in re.findall(r"(\d+)", str(target_key))})


def target_mechanism_labels(
    *,
    delta_dock_kcal_mol: float | None,
    keep_ifp_value: float,
    anchor_loss_fraction_value: float,
    occupancy_shift_mean_abs: float,
    occupancy_anchor_loss: float,
    solvent_proxy_shift: float,
) -> list[str]:
    labels: list[str] = []
    if anchor_loss_fraction_value >= 0.25 or occupancy_anchor_loss >= 0.25:
        labels.append("anchor_loss")
    if delta_dock_kcal_mol is not None and float(delta_dock_kcal_mol) >= 0.75 and keep_ifp_value <= 0.5:
        labels.append("steric_clash")
    if solvent_proxy_shift >= 0.20:
        labels.append("electrostatic_shift")
    if occupancy_shift_mean_abs >= 0.35 or keep_ifp_value <= 0.45:
        labels.append("pocket_rearrangement")
    return sorted(set(labels)) or ["pocket_rearrangement"]


def murcko_scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold, canonical=True)


def _action_signature(action: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(action.get("edit_family") or ""),
        str(action.get("pattern") or ""),
        str(action.get("fragment") or ""),
        int(action.get("anchor_atom_idx") or -1),
        int(action.get("atom_idx") or -1),
        int(action.get("neighbor_idx") or -1),
        int(action.get("source_anchor_idx") or -1),
        int(action.get("target_anchor_idx") or -1),
        str(action.get("swap_to") or ""),
    )


def _fragment_smiles(fragment: str) -> str:
    mapping = {
        "F": "F",
        "Cl": "Cl",
        "Br": "Br",
        "C": "C",
        "OC": "OC",
        "O": "O",
        "N": "N",
        "C#N": "C#N",
    }
    return str(mapping.get(fragment, fragment))


def _sanitize_candidate(mol: Chem.Mol) -> Chem.Mol | None:
    try:
        copied = Chem.Mol(mol)
        RDLogger.DisableLog("rdApp.*")
        try:
            Chem.SanitizeMol(copied)
        finally:
            RDLogger.EnableLog("rdApp.*")
        return copied
    except Exception:
        return None


def _attach_fragment(mol: Chem.Mol, anchor_idx: int, fragment: str) -> Chem.Mol | None:
    fragment_mol = Chem.MolFromSmiles(_fragment_smiles(fragment))
    if fragment_mol is None:
        return None
    combined = Chem.CombineMols(Chem.Mol(mol), fragment_mol)
    editable = Chem.RWMol(combined)
    editable.AddBond(int(anchor_idx), int(mol.GetNumAtoms()), order=Chem.BondType.SINGLE)
    return _sanitize_candidate(editable.GetMol())


def _delete_leaf_atom(mol: Chem.Mol, atom_idx: int) -> Chem.Mol | None:
    editable = Chem.RWMol(Chem.Mol(mol))
    editable.RemoveAtom(int(atom_idx))
    return _sanitize_candidate(editable.GetMol())


def _replace_leaf_atom(mol: Chem.Mol, atom_idx: int, neighbor_idx: int, fragment: str) -> Chem.Mol | None:
    editable = Chem.RWMol(Chem.Mol(mol))
    editable.RemoveAtom(int(atom_idx))
    reduced = editable.GetMol()
    if int(atom_idx) < int(neighbor_idx):
        neighbor_idx = int(neighbor_idx) - 1
    return _attach_fragment(reduced, int(neighbor_idx), fragment)


def _swap_atom(mol: Chem.Mol, atom_idx: int, atomic_num: int) -> Chem.Mol | None:
    editable = Chem.RWMol(Chem.Mol(mol))
    atom = editable.GetAtomWithIdx(int(atom_idx))
    atom.SetAtomicNum(int(atomic_num))
    return _sanitize_candidate(editable.GetMol())


def _remove_n_terminal_methyl(mol: Chem.Mol, atom_idx: int) -> Chem.Mol | None:
    atom = mol.GetAtomWithIdx(int(atom_idx))
    carbon_neighbors = [
        neighbor
        for neighbor in atom.GetNeighbors()
        if int(neighbor.GetAtomicNum()) == 6 and not neighbor.GetIsAromatic() and neighbor.GetDegree() == 1
    ]
    if not carbon_neighbors:
        return None
    return _delete_leaf_atom(mol, int(carbon_neighbors[0].GetIdx()))


def _rigidify_linker_bond(mol: Chem.Mol, atom_idx: int, neighbor_idx: int) -> Chem.Mol | None:
    editable = Chem.RWMol(Chem.Mol(mol))
    bond = editable.GetBondBetweenAtoms(int(atom_idx), int(neighbor_idx))
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
        return None
    if editable.GetAtomWithIdx(int(atom_idx)).GetIsAromatic() or editable.GetAtomWithIdx(int(neighbor_idx)).GetIsAromatic():
        return None
    bond.SetBondType(Chem.BondType.DOUBLE)
    return _sanitize_candidate(editable.GetMol())


def _vector_flip_leaf_attachment(
    mol: Chem.Mol,
    *,
    atom_idx: int,
    source_anchor_idx: int,
    target_anchor_idx: int,
) -> Chem.Mol | None:
    editable = Chem.RWMol(Chem.Mol(mol))
    if editable.GetBondBetweenAtoms(int(atom_idx), int(source_anchor_idx)) is None:
        return None
    if editable.GetBondBetweenAtoms(int(atom_idx), int(target_anchor_idx)) is not None:
        return None
    editable.RemoveBond(int(atom_idx), int(source_anchor_idx))
    editable.AddBond(int(atom_idx), int(target_anchor_idx), order=Chem.BondType.SINGLE)
    return _sanitize_candidate(editable.GetMol())


def _insert_ring_atom(mol: Chem.Mol, atom_idx: int, neighbor_idx: int) -> Chem.Mol | None:
    editable = Chem.RWMol(Chem.Mol(mol))
    bond = editable.GetBondBetweenAtoms(int(atom_idx), int(neighbor_idx))
    if bond is None or not bond.IsInRing() or bond.GetIsAromatic():
        return None
    editable.RemoveBond(int(atom_idx), int(neighbor_idx))
    new_idx = int(editable.AddAtom(Chem.Atom(6)))
    editable.AddBond(int(atom_idx), new_idx, order=Chem.BondType.SINGLE)
    editable.AddBond(new_idx, int(neighbor_idx), order=Chem.BondType.SINGLE)
    return _sanitize_candidate(editable.GetMol())


def _contract_ring_atom(mol: Chem.Mol, atom_idx: int) -> Chem.Mol | None:
    atom = mol.GetAtomWithIdx(int(atom_idx))
    if not atom.IsInRing() or atom.GetIsAromatic() or atom.GetDegree() != 2:
        return None
    neighbors = [int(neighbor.GetIdx()) for neighbor in atom.GetNeighbors()]
    if len(neighbors) != 2 or mol.GetBondBetweenAtoms(neighbors[0], neighbors[1]) is not None:
        return None
    editable = Chem.RWMol(Chem.Mol(mol))
    editable.RemoveAtom(int(atom_idx))
    left, right = neighbors
    if int(atom_idx) < left:
        left -= 1
    if int(atom_idx) < right:
        right -= 1
    editable.AddBond(int(left), int(right), order=Chem.BondType.SINGLE)
    return _sanitize_candidate(editable.GetMol())


def enumerate_candidate_actions(mol: Chem.Mol, stage6: dict[str, Any]) -> list[dict[str, Any]]:
    fragments_add = [str(value) for value in list(stage6.get("fragment_library", {}).get("add", []))]
    fragments_replace = [str(value) for value in list(stage6.get("fragment_library", {}).get("replace", []))]
    actions: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for atom in mol.GetAtoms():
        atom_idx = int(atom.GetIdx())
        symbol = str(atom.GetSymbol())
        total_hs = int(atom.GetTotalNumHs())
        is_aromatic = bool(atom.GetIsAromatic())
        if is_aromatic and total_hs > 0 and symbol in {"C", "N"}:
            for fragment in fragments_add:
                action = {
                    "edit_family": "ADD",
                    "pattern": FRAGMENT_FAMILY_PATTERNS.get(fragment, "add_polar_cap"),
                    "fragment": fragment,
                    "anchor_atom_idx": atom_idx,
                    "atom_idx": atom_idx,
                    "action_label": f"ADD({fragment})@{atom_idx}",
                }
                signature = _action_signature(action)
                if signature not in seen:
                    seen.add(signature)
                    actions.append(action)
            for fragment in HALOGEN_FRAGMENTS:
                halogen_action = {
                    "edit_family": "HALOGEN_SCAN",
                    "pattern": "halogen_scan",
                    "fragment": fragment,
                    "anchor_atom_idx": atom_idx,
                    "atom_idx": atom_idx,
                    "action_label": f"HALOGEN_SCAN({fragment})@{atom_idx}",
                }
                signature = _action_signature(halogen_action)
                if signature not in seen:
                    seen.add(signature)
                    actions.append(halogen_action)
            for fragment in SMALL_POLAR_FRAGMENTS:
                polar_action = {
                    "edit_family": "SMALL_POLAR_SCAN",
                    "pattern": "small_polar_scan",
                    "fragment": fragment,
                    "anchor_atom_idx": atom_idx,
                    "atom_idx": atom_idx,
                    "action_label": f"SMALL_POLAR_SCAN({fragment})@{atom_idx}",
                }
                signature = _action_signature(polar_action)
                if signature not in seen:
                    seen.add(signature)
                    actions.append(polar_action)
            if symbol == "C":
                action = {
                    "edit_family": "RING_EDIT",
                    "pattern": "aza_scan_ring",
                    "fragment": "N",
                    "atom_idx": atom_idx,
                    "swap_to": "N",
                    "action_label": f"RING_EDIT(N)@{atom_idx}",
                }
                signature = _action_signature(action)
                if signature not in seen:
                    seen.add(signature)
                    actions.append(action)
                heteroaryl_action = {
                    "edit_family": "HETEROARYL_SWAP",
                    "pattern": "heteroaryl_swap",
                    "fragment": "N",
                    "atom_idx": atom_idx,
                    "swap_to": "N",
                    "action_label": f"HETEROARYL_SWAP(N)@{atom_idx}",
                }
                signature = _action_signature(heteroaryl_action)
                if signature not in seen:
                    seen.add(signature)
                    actions.append(heteroaryl_action)
            backbone_action = {
                "edit_family": "BACKBONE_SEEKER",
                "pattern": "backbone_seeker",
                "fragment": "O",
                "anchor_atom_idx": atom_idx,
                "atom_idx": atom_idx,
                "action_label": f"BACKBONE_SEEKER(O)@{atom_idx}",
            }
            signature = _action_signature(backbone_action)
            if signature not in seen:
                seen.add(signature)
                actions.append(backbone_action)
            water_action = {
                "edit_family": "WATER_BRIDGE_PROBE",
                "pattern": "water_bridge_probe",
                "fragment": "OC",
                "anchor_atom_idx": atom_idx,
                "atom_idx": atom_idx,
                "action_label": f"WATER_BRIDGE_PROBE(OC)@{atom_idx}",
            }
            signature = _action_signature(water_action)
            if signature not in seen:
                seen.add(signature)
                actions.append(water_action)
        if not atom.IsInRing() and atom.GetDegree() == 1 and atom.GetAtomicNum() > 1:
            neighbor_idx = int(atom.GetNeighbors()[0].GetIdx())
            delete_pattern = "trim_steric_bulk" if atom.GetAtomicNum() > 9 else "trim_flexible_tail"
            delete_action = {
                "edit_family": "DELETE",
                "pattern": delete_pattern,
                "fragment": "",
                "atom_idx": atom_idx,
                "neighbor_idx": neighbor_idx,
                "action_label": f"DELETE@{atom_idx}",
            }
            signature = _action_signature(delete_action)
            if signature not in seen:
                seen.add(signature)
                actions.append(delete_action)
            trim_action = {
                "edit_family": "CONSTRAINED_TAIL_TRIM",
                "pattern": "constrained_tail_trim",
                "fragment": "",
                "atom_idx": atom_idx,
                "neighbor_idx": neighbor_idx,
                "action_label": f"CONSTRAINED_TAIL_TRIM@{atom_idx}",
            }
            signature = _action_signature(trim_action)
            if signature not in seen:
                seen.add(signature)
                actions.append(trim_action)
            for fragment in fragments_replace:
                replace_action = {
                    "edit_family": "REPLACE",
                    "pattern": REPLACE_FAMILY_PATTERNS.get(fragment, "replace_leaf_with_polar"),
                    "fragment": fragment,
                    "atom_idx": atom_idx,
                    "neighbor_idx": neighbor_idx,
                    "action_label": f"REPLACE({fragment})@{atom_idx}",
                }
                signature = _action_signature(replace_action)
                if signature not in seen:
                    seen.add(signature)
                    actions.append(replace_action)
            if int(atom.GetAtomicNum()) in {6, 7, 8, 16}:
                swap_to = {6: "N", 7: "C", 8: "N", 16: "O"}.get(int(atom.GetAtomicNum()))
                if swap_to:
                    bioisostere_action = {
                        "edit_family": "BIOISOSTERE_SWAP",
                        "pattern": "bioisostere_swap",
                        "fragment": swap_to,
                        "atom_idx": atom_idx,
                        "swap_to": swap_to,
                        "action_label": f"BIOISOSTERE_SWAP({swap_to})@{atom_idx}",
                    }
                    signature = _action_signature(bioisostere_action)
                    if signature not in seen:
                        seen.add(signature)
                        actions.append(bioisostere_action)
            anchor_atom = atom.GetNeighbors()[0]
            if anchor_atom.GetIsAromatic() and anchor_atom.IsInRing():
                for target_anchor in anchor_atom.GetNeighbors():
                    if int(target_anchor.GetIdx()) == atom_idx:
                        continue
                    if not target_anchor.GetIsAromatic() or not target_anchor.IsInRing() or int(target_anchor.GetTotalNumHs()) <= 0:
                        continue
                    vector_flip = {
                        "edit_family": "VECTOR_FLIP",
                        "pattern": "vector_flip_adjacent",
                        "fragment": "",
                        "atom_idx": atom_idx,
                        "source_anchor_idx": neighbor_idx,
                        "target_anchor_idx": int(target_anchor.GetIdx()),
                        "action_label": f"VECTOR_FLIP@{atom_idx}:{neighbor_idx}->{int(target_anchor.GetIdx())}",
                    }
                    signature = _action_signature(vector_flip)
                    if signature not in seen:
                        seen.add(signature)
                        actions.append(vector_flip)
        if not atom.IsInRing() and atom.GetDegree() in {1, 2} and symbol in {"O", "S"}:
            swap_to = "S" if symbol == "O" else "O"
            action = {
                "edit_family": "HETERO_SWAP",
                "pattern": "hetero_swap_linker",
                "fragment": swap_to,
                "atom_idx": atom_idx,
                "swap_to": swap_to,
                "action_label": f"HETERO_SWAP({swap_to})@{atom_idx}",
            }
            signature = _action_signature(action)
            if signature not in seen:
                seen.add(signature)
                actions.append(action)
        if not atom.IsInRing() and atom.GetDegree() in {1, 2} and symbol in {"C", "N", "O", "S"}:
            swap_targets = {"C": "N", "N": "C", "O": "N", "S": "O"}
            swap_to = swap_targets.get(symbol)
            if swap_to:
                linker_action = {
                    "edit_family": "LINKER_HETERO_SCAN",
                    "pattern": "linker_hetero_scan",
                    "fragment": swap_to,
                    "atom_idx": atom_idx,
                    "swap_to": swap_to,
                    "action_label": f"LINKER_HETERO_SCAN({swap_to})@{atom_idx}",
                }
                signature = _action_signature(linker_action)
                if signature not in seen:
                    seen.add(signature)
                    actions.append(linker_action)
            if int(atom.GetDegree()) == 2 and symbol in {"C", "N"}:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetIdx() == atom_idx:
                        continue
                    rigidify = {
                        "edit_family": "RIGIDIFY_LINKER",
                        "pattern": "rigidify_linker",
                        "fragment": "",
                        "atom_idx": atom_idx,
                        "neighbor_idx": int(neighbor.GetIdx()),
                        "action_label": f"RIGIDIFY_LINKER@{atom_idx}-{int(neighbor.GetIdx())}",
                    }
                    signature = _action_signature(rigidify)
                    if signature not in seen:
                        seen.add(signature)
                        actions.append(rigidify)
        if symbol == "N" and not atom.GetIsAromatic():
            if int(atom.GetDegree()) <= 2:
                methyl_action = {
                    "edit_family": "N_METHYL_SCAN",
                    "pattern": "n_methyl_scan",
                    "fragment": "C",
                    "anchor_atom_idx": atom_idx,
                    "atom_idx": atom_idx,
                    "action_label": f"N_METHYL_SCAN(+CH3)@{atom_idx}",
                }
                signature = _action_signature(methyl_action)
                if signature not in seen:
                    seen.add(signature)
                    actions.append(methyl_action)
            remove_action = {
                "edit_family": "N_METHYL_SCAN",
                "pattern": "n_methyl_scan",
                "fragment": "-C",
                "atom_idx": atom_idx,
                "action_label": f"N_METHYL_SCAN(-CH3)@{atom_idx}",
            }
            signature = _action_signature(remove_action)
            if signature not in seen:
                seen.add(signature)
                actions.append(remove_action)
        if atom.IsInRing() and not atom.GetIsAromatic() and atom.GetDegree() == 2:
            ring_contract = {
                "edit_family": "RING_CONTRACTION",
                "pattern": "ring_contraction",
                "fragment": "",
                "atom_idx": atom_idx,
                "action_label": f"RING_CONTRACTION@{atom_idx}",
            }
            signature = _action_signature(ring_contract)
            if signature not in seen:
                seen.add(signature)
                actions.append(ring_contract)
            for neighbor in atom.GetNeighbors():
                if not neighbor.IsInRing() or neighbor.GetIsAromatic() or int(neighbor.GetIdx()) < atom_idx:
                    continue
                ring_expand = {
                    "edit_family": "RING_EXPANSION",
                    "pattern": "ring_expansion",
                    "fragment": "C",
                    "atom_idx": atom_idx,
                    "neighbor_idx": int(neighbor.GetIdx()),
                    "action_label": f"RING_EXPANSION@{atom_idx}-{int(neighbor.GetIdx())}",
                }
                signature = _action_signature(ring_expand)
                if signature not in seen:
                    seen.add(signature)
                    actions.append(ring_expand)
    return actions


def apply_action_to_molecule(mol: Chem.Mol, action: dict[str, Any]) -> Chem.Mol | None:
    family = str(action.get("edit_family") or "")
    if family == "ADD":
        return _attach_fragment(mol, int(action["anchor_atom_idx"]), str(action.get("fragment") or ""))
    if family == "DELETE":
        return _delete_leaf_atom(mol, int(action["atom_idx"]))
    if family == "REPLACE":
        return _replace_leaf_atom(mol, int(action["atom_idx"]), int(action["neighbor_idx"]), str(action.get("fragment") or ""))
    if family == "HETERO_SWAP":
        swap_to = str(action.get("swap_to") or action.get("fragment") or "")
        atomic_num = {"O": 8, "N": 7, "S": 16, "C": 6}.get(swap_to)
        if atomic_num is None:
            return None
        return _swap_atom(mol, int(action["atom_idx"]), int(atomic_num))
    if family == "RING_EDIT":
        swap_to = str(action.get("swap_to") or action.get("fragment") or "N")
        atomic_num = {"N": 7, "C": 6}.get(swap_to)
        if atomic_num is None:
            return None
        return _swap_atom(mol, int(action["atom_idx"]), int(atomic_num))
    if family == "RIGIDIFY_LINKER":
        return _rigidify_linker_bond(mol, int(action["atom_idx"]), int(action["neighbor_idx"]))
    if family == "VECTOR_FLIP":
        return _vector_flip_leaf_attachment(
            mol,
            atom_idx=int(action["atom_idx"]),
            source_anchor_idx=int(action["source_anchor_idx"]),
            target_anchor_idx=int(action["target_anchor_idx"]),
        )
    if family == "BIOISOSTERE_SWAP":
        swap_to = str(action.get("swap_to") or action.get("fragment") or "")
        atomic_num = {"C": 6, "N": 7, "O": 8, "S": 16}.get(swap_to)
        if atomic_num is None:
            return None
        return _swap_atom(mol, int(action["atom_idx"]), int(atomic_num))
    if family == "RING_EXPANSION":
        return _insert_ring_atom(mol, int(action["atom_idx"]), int(action["neighbor_idx"]))
    if family == "RING_CONTRACTION":
        return _contract_ring_atom(mol, int(action["atom_idx"]))
    if family == "HETEROARYL_SWAP":
        swap_to = str(action.get("swap_to") or action.get("fragment") or "")
        atomic_num = {"C": 6, "N": 7}.get(swap_to)
        if atomic_num is None:
            return None
        return _swap_atom(mol, int(action["atom_idx"]), int(atomic_num))
    if family == "BACKBONE_SEEKER":
        return _attach_fragment(mol, int(action["anchor_atom_idx"]), str(action.get("fragment") or "O"))
    if family == "WATER_BRIDGE_PROBE":
        return _attach_fragment(mol, int(action["anchor_atom_idx"]), str(action.get("fragment") or "OC"))
    if family in {"HALOGEN_SCAN", "SMALL_POLAR_SCAN"}:
        return _attach_fragment(mol, int(action["anchor_atom_idx"]), str(action.get("fragment") or ""))
    if family == "LINKER_HETERO_SCAN":
        swap_to = str(action.get("swap_to") or action.get("fragment") or "")
        atomic_num = {"C": 6, "N": 7, "O": 8, "S": 16}.get(swap_to)
        if atomic_num is None:
            return None
        return _swap_atom(mol, int(action["atom_idx"]), int(atomic_num))
    if family == "N_METHYL_SCAN":
        fragment = str(action.get("fragment") or "C")
        if fragment == "-C":
            return _remove_n_terminal_methyl(mol, int(action["atom_idx"]))
        return _attach_fragment(mol, int(action["anchor_atom_idx"]), N_METHYL_SMILES)
    if family == "CONSTRAINED_TAIL_TRIM":
        return _delete_leaf_atom(mol, int(action["atom_idx"]))
    return None


def _template_priority_score(action: dict[str, Any], templates: list[dict[str, Any]]) -> float:
    score = 0.0
    for row in templates:
        weight = 1.0 / max(1, int(row.get("priority") or 1))
        if str(action.get("edit_family") or "") == str(row.get("edit_family") or ""):
            score += 5.0 * weight
        if str(action.get("pattern") or "") == str(row.get("pattern") or ""):
            score += 9.0 * weight
        if str(action.get("fragment") or "") == str(row.get("fragment") or ""):
            score += 3.0 * weight
    return float(score)


def _prior_signature(action: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(action.get("edit_family") or ""),
        str(action.get("pattern") or ""),
        str(action.get("fragment") or ""),
    )


def rank_actions(
    actions: list[dict[str, Any]],
    templates: list[dict[str, Any]],
    transform_prior: dict[tuple[str, str, str], float] | None = None,
    search_seed: int | None = None,
    rank_jitter: float = 0.0,
    jitter_salt: str = "",
) -> list[dict[str, Any]]:
    prior = transform_prior or {}
    ranked = []
    for action in actions:
        signature = _prior_signature(action)
        jitter = 0.0
        if search_seed is not None and float(rank_jitter) > 0.0:
            seed_material = json.dumps(
                {
                    "seed": int(search_seed),
                    "salt": str(jitter_salt),
                    "signature": signature,
                    "atom_idx": int(action.get("atom_idx") or -1),
                    "anchor_atom_idx": int(action.get("anchor_atom_idx") or -1),
                    "action_label": str(action.get("action_label") or ""),
                },
                sort_keys=True,
            )
            digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
            jitter = (random.Random(int(digest[:16], 16)).random() - 0.5) * float(rank_jitter)
        ranked.append(
            {
                **action,
                "template_priority_score": _template_priority_score(action, templates),
                "transform_prior_score": float(prior.get(signature, 0.0)),
                "search_rank_jitter": float(jitter),
            }
        )
    ranked.sort(
        key=lambda row: (
            -(
                float(row.get("template_priority_score") or 0.0)
                + float(row.get("transform_prior_score") or 0.0)
                + float(row.get("search_rank_jitter") or 0.0)
            ),
            str(row.get("edit_family") or ""),
            str(row.get("pattern") or ""),
            str(row.get("fragment") or ""),
            int(row.get("atom_idx") or -1),
        )
    )
    return ranked


def write_candidate_sdf(smiles: str, path: Path) -> dict[str, Any]:
    RDLogger.DisableLog("rdApp.*")
    try:
        molecule = Chem.MolFromSmiles(smiles)
    finally:
        RDLogger.EnableLog("rdApp.*")
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    molecule = Chem.AddHs(molecule)
    molecule = ensure_ligand_3d(molecule)
    ensure_dir(path.parent)
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()
    return {
        "candidate_id": candidate_id(smiles),
        "smiles": canonical_smiles(smiles),
        "path": str(path),
        "heavy_atom_count": int(Chem.RemoveHs(Chem.Mol(molecule)).GetNumHeavyAtoms()),
    }


def lead_wt_affinity(stage3_5_root: Path) -> float | None:
    pose_stats = read_csv_optional(stage3_5_root / "wt_pose_stats.csv")
    if pose_stats.empty or "affinity_kcal_mol" not in pose_stats.columns:
        return None
    final_rows = pose_stats[pose_stats.get("selected_as_final_pose", pd.Series(dtype=bool)).fillna(False).astype(bool)]
    if not final_rows.empty:
        return float(final_rows.iloc[0]["affinity_kcal_mol"])
    return float(pose_stats["affinity_kcal_mol"].min())


def _reference_smiles_candidates(root: Path, stage6: dict[str, Any]) -> list[str]:
    csv_path = str(stage6.get("transform_prior_reference_smiles_csv") or "").strip()
    if not csv_path:
        return []
    path = root / csv_path
    frame = read_csv_optional(path)
    if frame.empty:
        return []
    smiles_column = "smiles" if "smiles" in frame.columns else ("SMILES" if "SMILES" in frame.columns else "")
    if not smiles_column:
        return []
    return [str(value) for value in frame[smiles_column].dropna().astype(str).tolist()]


def _pdbbind_reference_mols(root: Path, lead_mol: Chem.Mol, stage6: dict[str, Any]) -> list[Chem.Mol]:
    pdbbind_root = root / str(stage6.get("transform_prior_pdbbind_root", "data/PDBbind/extracted/P-L"))
    if not pdbbind_root.exists():
        return []
    scan_limit = max(0, int(stage6.get("transform_prior_pdbbind_scan_limit", 240)))
    top_k = max(0, int(stage6.get("transform_prior_pdbbind_top_k", 24)))
    similarity_min = float(stage6.get("transform_prior_pdbbind_similarity_min", 0.35))
    if scan_limit <= 0 or top_k <= 0:
        return []
    lead_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.RemoveHs(Chem.Mol(lead_mol)), 2, nBits=2048)
    candidates: list[tuple[float, Chem.Mol]] = []
    for index, path in enumerate(pdbbind_root.glob("*/*/*_ligand.sdf")):
        if index >= scan_limit:
            break
        try:
            molecule = Chem.RemoveHs(load_rdkit_molecule(path, sanitize=False))
        except Exception:
            continue
        if molecule is None or molecule.GetNumHeavyAtoms() <= 4:
            continue
        similarity = float(
            DataStructs.TanimotoSimilarity(
                lead_fp,
                AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048),
            )
        )
        if similarity < similarity_min:
            continue
        candidates.append((similarity, molecule))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [molecule for _, molecule in candidates[:top_k]]


def _heuristic_reference_prior(lead_mol: Chem.Mol, reference_mols: list[Chem.Mol]) -> Counter[tuple[str, str, str]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    lead_heavy_atoms = int(Chem.RemoveHs(Chem.Mol(lead_mol)).GetNumHeavyAtoms())
    lead_has_halogen = any(int(atom.GetAtomicNum()) in {9, 17, 35} for atom in lead_mol.GetAtoms())
    lead_polar_atoms = sum(1 for atom in lead_mol.GetAtoms() if int(atom.GetAtomicNum()) in {7, 8, 16})
    lead_ring_n = sum(1 for atom in lead_mol.GetAtoms() if atom.GetIsAromatic() and atom.GetSymbol() == "N")
    for molecule in reference_mols:
        ref_heavy_atoms = int(Chem.RemoveHs(Chem.Mol(molecule)).GetNumHeavyAtoms())
        ref_has_halogen = any(int(atom.GetAtomicNum()) in {9, 17, 35} for atom in molecule.GetAtoms())
        ref_polar_atoms = sum(1 for atom in molecule.GetAtoms() if int(atom.GetAtomicNum()) in {7, 8, 16})
        ref_ring_n = sum(1 for atom in molecule.GetAtoms() if atom.GetIsAromatic() and atom.GetSymbol() == "N")
        if ref_has_halogen and not lead_has_halogen:
            counts[("HALOGEN_SCAN", "halogen_scan", "F")] += 1
        if ref_polar_atoms > lead_polar_atoms:
            counts[("SMALL_POLAR_SCAN", "small_polar_scan", "O")] += 1
        if ref_ring_n > lead_ring_n:
            counts[("RING_EDIT", "aza_scan_ring", "N")] += 1
        if ref_heavy_atoms < lead_heavy_atoms:
            counts[("CONSTRAINED_TAIL_TRIM", "constrained_tail_trim", "")] += 1
    return counts


def build_transform_prior(
    *,
    root: Path,
    case_stage6_root: Path,
    lead_smiles: str,
    stage6: dict[str, Any],
    case_specific_action_hints_payload: list[dict[str, Any]] | None = None,
) -> tuple[dict[tuple[str, str, str], float], list[dict[str, Any]]]:
    lead_mol = Chem.MolFromSmiles(lead_smiles)
    if lead_mol is None:
        return {}, []
    prior: Counter[tuple[str, str, str]] = Counter()
    summary: list[dict[str, Any]] = []

    lead_actions = enumerate_candidate_actions(lead_mol, stage6)
    lead_counter = Counter(_prior_signature(action) for action in lead_actions)
    for signature, count in lead_counter.items():
        family = signature[0]
        if family in {"HALOGEN_SCAN", "SMALL_POLAR_SCAN", "LINKER_HETERO_SCAN", "N_METHYL_SCAN", "CONSTRAINED_TAIL_TRIM"}:
            prior[signature] += float(min(3, count))
    if lead_counter:
        summary.append({"source": "lead_neighborhood", "rule_count": int(sum(lead_counter.values()))})

    hint_rows = [dict(row) for row in list(case_specific_action_hints_payload or []) if isinstance(row, dict)]
    if hint_rows:
        for hint in hint_rows:
            signature = (
                str(hint.get("edit_family") or ""),
                str(hint.get("pattern") or ""),
                str(hint.get("fragment") or ""),
            )
            if not signature[0] or not signature[1]:
                continue
            prior[signature] += float(hint.get("weight") or 0.0)
        summary.append({"source": "case_specific_pocket", "rule_count": int(len(hint_rows))})

    historical_path = case_stage6_root / "leaderboard.csv"
    historical = read_csv_optional(historical_path)
    if bool(stage6.get("transform_prior_use_history", False)) and not historical.empty and "action_sequence_json" in historical.columns:
        ordered = historical.sort_values("objective_reward", ascending=False).head(int(stage6.get("transform_prior_history_top_n", 25)))
        for rank, row in enumerate(ordered.to_dict(orient="records"), start=1):
            weight = 1.0 / rank
            for action in list(parse_json_payload(row.get("action_sequence_json")) or []):
                prior[_prior_signature(dict(action))] += weight
        summary.append({"source": "historical_stage6", "rule_count": int(len(ordered))})

    reference_mols: list[Chem.Mol] = []
    for smiles in _reference_smiles_candidates(root, stage6):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            reference_mols.append(molecule)
    reference_mols.extend(_pdbbind_reference_mols(root, lead_mol, stage6))
    if reference_mols:
        reference_prior = _heuristic_reference_prior(lead_mol, reference_mols)
        for signature, count in reference_prior.items():
            prior[signature] += float(count)
        summary.append({"source": "reference_ligands", "rule_count": int(sum(reference_prior.values()))})

    normalized: dict[tuple[str, str, str], float] = {}
    total = float(sum(prior.values()))
    if total > 0.0:
        for signature, value in prior.items():
            normalized[signature] = float(value / total)
    return normalized, summary


def _select_risk_mass_panel(frame: pd.DataFrame, effect_scope: str, stage6: dict[str, Any]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    risk_coverage = float(
        stage6.get(
            f"{effect_scope}_panel_risk_coverage",
            stage6.get("panel_risk_coverage", 0.80),
        )
    )
    min_n = int(stage6.get(f"{effect_scope}_panel_min_n", 6 if effect_scope == "site" else 5))
    max_n = int(stage6.get(f"{effect_scope}_panel_max_n", 20))
    ordered = frame.sort_values(["stage4_rank", "target_key"]).reset_index(drop=True)
    risk_series = pd.to_numeric(ordered.get("risk_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    risk_total = float(risk_series.sum())
    if risk_total <= 0.0:
        limit = min(len(ordered), max(max_n, min_n))
        return ordered.head(limit).copy()
    selected_indices: list[int] = []
    cumulative = 0.0
    for index, (_, row) in enumerate(ordered.iterrows()):
        selected_indices.append(index)
        cumulative += float(max(0.0, float(row.get("risk_score") or 0.0)) / risk_total)
        if len(selected_indices) >= min_n and cumulative >= risk_coverage:
            break
        if len(selected_indices) >= max_n:
            break
    if len(selected_indices) < min_n:
        selected_indices = list(range(min(len(ordered), min_n)))
    if len(selected_indices) > max_n:
        selected_indices = selected_indices[:max_n]
    return ordered.iloc[selected_indices].copy().reset_index(drop=True)


def _scope_weight_map(frame: pd.DataFrame, total_weight: float, stage6: dict[str, Any]) -> dict[str, float]:
    if frame.empty or total_weight <= 0.0:
        return {}
    risk_series = pd.to_numeric(frame.get("risk_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    tau = max(float(stage6.get("panel_weight_temperature", 1.0)), 1.0e-6)
    eps = float(stage6.get("panel_weight_eps", 1.0e-6))
    if float(risk_series.sum()) <= 0.0:
        each = float(total_weight / max(1, len(frame)))
        return {str(row["target_key"]): each for _, row in frame.iterrows()}
    logits = [math.log(max(float(value), eps) + eps) / tau for value in risk_series.tolist()]
    max_logit = max(logits)
    exp_values = [math.exp(value - max_logit) for value in logits]
    norm = sum(exp_values)
    if norm <= 0.0:
        each = float(total_weight / max(1, len(frame)))
        return {str(row["target_key"]): each for _, row in frame.iterrows()}
    weights: dict[str, float] = {}
    for (_, row), exp_value in zip(frame.iterrows(), exp_values):
        weights[str(row["target_key"])] = float(total_weight * exp_value / norm)
    return weights


def _panel_weights(panel_frame: pd.DataFrame, case_entry: dict[str, Any], stage6: dict[str, Any]) -> dict[str, float]:
    if panel_frame.empty:
        return {}
    combo_dense = bool(str(case_entry.get("evaluation_unit") or "") == "observed_combo")
    combo_rows = panel_frame[panel_frame["effect_scope"].eq("combo")].copy()
    site_rows = panel_frame[panel_frame["effect_scope"].eq("site")].copy()
    if combo_dense and not combo_rows.empty:
        combo_total = float(stage6.get("combo_reward_weight_floor", 0.6))
        if site_rows.empty:
            combo_total = 1.0
        combo_total = min(1.0, max(0.0, combo_total))
        site_total = max(0.0, 1.0 - combo_total) if not site_rows.empty else 0.0
    else:
        combo_total = 0.0
        site_total = 1.0
    weights = {
        **_scope_weight_map(site_rows, site_total, stage6),
        **_scope_weight_map(combo_rows, combo_total, stage6),
    }
    return weights


def _ordered_scope_panel_rows(panel_rows: list[dict[str, Any]], effect_scope: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in panel_rows if str(row.get("effect_scope") or "") == str(effect_scope)]
    rows.sort(
        key=lambda row: (
            int(row.get("stage4_rank") or 999999),
            str(row.get("target_key") or ""),
        )
    )
    return rows


def panel_curriculum_limits(
    *,
    case_context: dict[str, Any],
    stage6: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
    case_id = str(case_context.get("case_id") or "")
    oracle_mode = str((case_context.get("scoring_policy") or {}).get("oracle_v2_mode") or "")
    site_full = len(_ordered_scope_panel_rows(list(case_context.get("panel_rows_full") or case_context.get("panel_rows") or []), "site"))
    combo_full = len(_ordered_scope_panel_rows(list(case_context.get("panel_rows_full") or case_context.get("panel_rows") or []), "combo"))
    if case_id == "egfr_erlotinib":
        schedule = [
            (4, 6, 0),
            (8, 10, 0),
            (12, 14, 0),
        ]
    elif case_id == "hiv_rt_rilpivirine":
        schedule = [
            (4, 4, 4),
            (8, 8, 8),
            (12, 10, 12),
        ]
    elif case_id == "abl1_nilotinib":
        if oracle_mode != "normal_robust_optimization":
            site_cap = min(
                site_full,
                int(stage6.get("uncertainty_heavy_site_panel_max_n", site_full or 0)),
            )
            return {
                "stage_name": "oracle_capped",
                "site_limit": int(site_cap),
                "combo_limit": 0,
            }
        schedule = [
            (4, 6, 0),
            (8, 10, 0),
            (12, 14, 0),
        ]
    else:
        return {
            "stage_name": "full",
            "site_limit": int(site_full),
            "combo_limit": int(combo_full),
        }
    for upper_round, site_limit, combo_limit in schedule:
        if int(round_index) <= int(upper_round):
            return {
                "stage_name": f"rounds_1_{upper_round}" if int(round_index) == 1 else f"through_{upper_round}",
                "site_limit": int(min(site_full, site_limit)),
                "combo_limit": int(min(combo_full, combo_limit)),
            }
    return {
        "stage_name": "full",
        "site_limit": int(site_full),
        "combo_limit": int(combo_full),
    }


def case_context_for_round(
    *,
    case_context: dict[str, Any],
    stage6: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
    panel_rows_full = list(case_context.get("panel_rows_full") or case_context.get("panel_rows") or [])
    limits = panel_curriculum_limits(case_context=case_context, stage6=stage6, round_index=round_index)
    site_limit = int(limits.get("site_limit", 0))
    combo_limit = int(limits.get("combo_limit", 0))
    site_rows = _ordered_scope_panel_rows(panel_rows_full, "site")
    combo_rows = _ordered_scope_panel_rows(panel_rows_full, "combo")
    selected_rows = [*site_rows[:site_limit], *combo_rows[:combo_limit]]
    selected_frame = pd.DataFrame.from_records(selected_rows)
    if selected_frame.empty:
        selected_weights = {}
    else:
        selected_weights = _panel_weights(
            selected_frame,
            {"evaluation_unit": str(case_context.get("evaluation_unit") or "")},
            stage6,
        )
    updated = dict(case_context)
    updated["panel_rows"] = selected_rows
    updated["panel_weights"] = selected_weights
    curriculum_summary = {
        "round_index": int(round_index),
        "stage_name": str(limits.get("stage_name") or ""),
        "site_limit": int(site_limit),
        "combo_limit": int(combo_limit),
        "site_full_count": int(len(site_rows)),
        "combo_full_count": int(len(combo_rows)),
    }
    updated["panel_curriculum"] = curriculum_summary
    mechanism_summary = dict(case_context.get("mechanism_summary") or {})
    mechanism_summary.update(
        {
            "curriculum_stage_name": str(curriculum_summary["stage_name"]),
            "curriculum_site_limit": int(site_limit),
            "curriculum_combo_limit": int(combo_limit),
            "ready_site_count": int(site_limit),
            "ready_combo_count": int(combo_limit),
        }
    )
    updated["mechanism_summary"] = mechanism_summary
    action_space = dict(case_context.get("action_space") or {})
    action_space["panel_curriculum"] = curriculum_summary
    updated["action_space"] = action_space
    return updated


def _oracle_v2_scoring_policy(*, case_root: Path, stage6: dict[str, Any]) -> dict[str, Any]:
    oracle_root = case_root / str(stage6.get("oracle_v2_output_dirname", "stage6_oracle_v2"))
    metadata_path = oracle_root / "stage6_oracle_v2.json"
    holdout_path = oracle_root / "holdout_eval.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    holdout = json.loads(holdout_path.read_text(encoding="utf-8")) if holdout_path.exists() else {}
    case_holdout_trust = native_value(dict(holdout.get("case_metrics") or {}).get("trust_score"))
    domain_holdout_trust = native_value(dict(holdout.get("domain_metrics") or {}).get("trust_score"))
    ensemble_holdout_trust = native_value(dict(holdout.get("ensemble_metrics") or {}).get("trust_score"))
    training_trust = native_value(metadata.get("trust_score"))
    holdout_scores = [float(value) for value in [case_holdout_trust, domain_holdout_trust, ensemble_holdout_trust] if value is not None]
    effective_trust = float(min(holdout_scores)) if holdout_scores else (None if training_trust is None else float(training_trust))
    normal_threshold = float(stage6.get("oracle_v2_normal_trust_min", 0.60))
    uncertainty_threshold = float(stage6.get("oracle_v2_uncertainty_trust_min", 0.45))
    if effective_trust is None:
        mode = str(metadata.get("mode") or "hypothesis_only")
    elif effective_trust >= normal_threshold:
        mode = "normal_robust_optimization"
    elif effective_trust >= uncertainty_threshold:
        mode = "uncertainty_heavy"
    else:
        mode = "hypothesis_only"
    oracle_model_path = oracle_root / "stage6_oracle_v2.pkl"
    oracle_available = bool(metadata and oracle_model_path.exists() and mode != "hypothesis_only")
    return {
        "oracle_v2_root": str(oracle_root),
        "oracle_v2_model_path": str(oracle_model_path),
        "oracle_v2_metadata": metadata,
        "oracle_v2_holdout": holdout,
        "oracle_v2_training_trust_score": training_trust,
        "oracle_v2_case_holdout_trust_score": case_holdout_trust,
        "oracle_v2_domain_holdout_trust_score": domain_holdout_trust,
        "oracle_v2_ensemble_holdout_trust_score": ensemble_holdout_trust,
        "oracle_v2_effective_trust_score": effective_trust,
        "oracle_v2_mode": mode,
        "oracle_v2_available": bool(oracle_available),
    }


def load_stage6_case_context(
    *,
    root: Path,
    case_entry: dict[str, Any],
    stage5: dict[str, Any],
    stage6: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(case_entry["case_id"])
    case_root = root / "outputs" / case_id
    stage3_5_root = case_root / "stage3_5"
    stage5_root = case_root / "stage5"
    case_stage6_root = ensure_dir(case_root / str(stage6.get("output_dirname", "stage6")))
    site_rank = read_csv_optional(case_root / "stage4" / "mutation_rank.csv")
    combo_rank = read_csv_optional(case_root / "stage4" / "combo_rank.csv")
    mutation_status = read_csv_optional(case_root / "stage3_2" / "mutation_site_status.csv")
    ifp_diff = read_csv_optional(stage5_root / "ifp_diff.csv")
    stage5_qc_path = stage5_root / "stage5_qc.json"
    stage5_qc = json.loads(stage5_qc_path.read_text(encoding="utf-8")) if stage5_qc_path.exists() else {}
    dock_mmgbsa_r = native_value(stage5_qc.get("dock_vs_mmgbsa_pearson_r"))
    uncertainty_heavy = bool(
        dock_mmgbsa_r is not None
        and float(dock_mmgbsa_r) < float(stage6.get("uncertainty_heavy_dock_mmgbsa_threshold", 0.6))
    )
    oracle_policy = _oracle_v2_scoring_policy(case_root=case_root, stage6=stage6)
    oracle_mode = str(oracle_policy.get("oracle_v2_mode") or "hypothesis_only")
    if oracle_mode == "uncertainty_heavy":
        uncertainty_heavy = True

    panel_stage5 = dict(stage5)
    panel_stage5["site_top_n"] = int(stage6.get("site_panel_max_n", stage6.get("site_panel_top_n", 20)))
    panel_stage5["combo_top_n"] = (
        int(stage6.get("combo_panel_max_n", stage6.get("combo_panel_top_n", 20)))
        if str(case_entry.get("evaluation_unit") or "") == "observed_combo"
        else 0
    )
    if uncertainty_heavy:
        panel_stage5["site_top_n"] = min(
            int(panel_stage5["site_top_n"]),
            int(stage6.get("uncertainty_heavy_site_panel_max_n", panel_stage5["site_top_n"])),
        )
        if int(panel_stage5["combo_top_n"]) > 0:
            panel_stage5["combo_top_n"] = min(
                int(panel_stage5["combo_top_n"]),
                int(stage6.get("uncertainty_heavy_combo_panel_max_n", panel_stage5["combo_top_n"])),
            )
    panel_frame = build_stage5_target_panel(
        root=root,
        case_id=case_id,
        site_rank=site_rank,
        combo_rank=combo_rank,
        mutation_status=mutation_status,
        stage5=panel_stage5,
    )
    panel_frame = panel_frame[panel_frame["stage5_ready"].fillna(False).astype(bool)].copy()
    site_panel = _select_risk_mass_panel(panel_frame[panel_frame["effect_scope"].eq("site")].copy(), "site", stage6)
    combo_panel = _select_risk_mass_panel(panel_frame[panel_frame["effect_scope"].eq("combo")].copy(), "combo", stage6)
    panel_frame = pd.concat([site_panel, combo_panel], ignore_index=True) if (not site_panel.empty or not combo_panel.empty) else pd.DataFrame()
    panel_frame = panel_frame.sort_values(["effect_scope", "stage4_rank", "target_key"]).reset_index(drop=True)
    receptor_ensemble_enabled = bool(stage6.get("receptor_ensemble_enabled", False))
    receptor_ensemble_members = load_receptor_ensemble_members(case_root) if receptor_ensemble_enabled else {}
    receptor_ensemble_summary_path = case_root / "stage6_receptor_ensemble" / "receptor_ensemble_summary.json"
    receptor_ensemble_summary = (
        json.loads(receptor_ensemble_summary_path.read_text(encoding="utf-8"))
        if receptor_ensemble_summary_path.exists()
        else {}
    )

    use_multichain_receptor = bool(
        str(case_entry.get("target_domain") or "").lower() == "rt"
        and (stage3_5_root / "wt_ifp_multichain.json").exists()
        and (stage3_5_root / "wt_receptor_stage6_multichain.pdb").exists()
    )
    wt_ifp_path = stage3_5_root / ("wt_ifp_multichain.json" if use_multichain_receptor else "wt_ifp.json")
    wt_complex_path = stage3_5_root / ("wt_complex_multichain.pdb" if use_multichain_receptor else "wt_complex.pdb")
    wt_receptor_stage6_path = stage3_5_root / (
        "wt_receptor_stage6_multichain.pdb" if use_multichain_receptor else "wt_receptor_stage6.pdb"
    )
    wt_ifp_payload = json.loads(wt_ifp_path.read_text(encoding="utf-8"))
    baseline_ifp = dict(wt_ifp_payload.get("baseline_ifp") or {})
    anchor_residues = [str(value) for value in list(wt_ifp_payload.get("anchor_residues") or [])]
    validated_pocket_residues: set[str] = set()
    mechanism_counts: dict[str, int] = {}
    if not ifp_diff.empty:
        for value in ifp_diff.get("mechanism_labels_json", pd.Series(dtype=str)).fillna("[]").tolist():
            for label in list(parse_json_payload(value) or []):
                mechanism_counts[str(label)] = int(mechanism_counts.get(str(label), 0) + 1)
        for column in ["wt_ifp_occupancy_json", "mt_ifp_occupancy_json"]:
            for value in ifp_diff.get(column, pd.Series(dtype=str)).fillna("{}").tolist():
                payload = parse_json_payload(value) or {}
                if isinstance(payload, dict):
                    validated_pocket_residues.update(str(label) for label in payload.keys())

    top_site_positions: set[int] = set()
    site_limit = int(panel_stage5["site_top_n"])
    if not site_rank.empty and "target_position" in site_rank.columns:
        for value in site_rank.sort_values("mutation_rank").head(site_limit)["target_position"].tolist():
            if value is None or pd.isna(value):
                continue
            top_site_positions.add(int(value))
    hotspot_residues = [
        residue
        for residue in anchor_residues
        if residue_number_from_label(residue) in top_site_positions
    ]

    docking_scores = read_csv_optional(stage5_root / "docking_scores.csv")
    lead_mt_affinities = {
        str(row["target_key"]): native_value(row.get("mt_best_affinity_kcal_mol"))
        for row in docking_scores.to_dict(orient="records")
    }

    lead_smiles = canonical_smiles(load_rdkit_molecule(case_root / "stage1_5" / "raw" / "ligand.sdf", sanitize=False))
    lead_id = candidate_id(lead_smiles)
    lead_descriptors = descriptor_payload(Chem.MolFromSmiles(lead_smiles))
    calibrator_metadata = fit_case_calibrator(
        stage5_root=stage5_root,
        stage6_root=case_stage6_root,
        case_id=case_id,
        stage6=stage6,
        stage5_qc=stage5_qc,
    )
    panel_weights = _panel_weights(panel_frame, case_entry, stage6)
    pocket_residues_universe = sorted(
        set(str(value) for value in list(wt_ifp_payload.get("pocket_residue_universe") or []))
        | set(str(value) for value in list(baseline_ifp.get("residue_set") or []))
        | validated_pocket_residues
    )
    nonhotspot_residues = sorted(set(pocket_residues_universe) - set(hotspot_residues))
    multichain_summary = dict(wt_ifp_payload.get("reference_template") or {})
    partner_chain_residues = [
        str(value)
        for value in list(
            wt_ifp_payload.get("partner_chain_residues")
            or multichain_summary.get("partner_chain_residues")
            or []
        )
        if str(value)
    ]
    wt_receptor_chain_ids = [str(value) for value in list(multichain_summary.get("reference_holo_chain_ids") or [])]
    if not wt_receptor_chain_ids:
        wt_receptor_chain_ids = [str(case_entry.get("wt_template", {}).get("chain_id") or first_protein_chain_id(wt_complex_path))]
    wt_chain_id = str(case_entry.get("wt_template", {}).get("chain_id") or wt_receptor_chain_ids[0])
    partner_chain_ids = [chain_id for chain_id in wt_receptor_chain_ids if chain_id != wt_chain_id]
    partner_chain_positions = sorted(
        {
            position
            for position in (residue_number_from_label(label) for label in partner_chain_residues)
            if position is not None
        }
    )
    pocket_profile_payload = pocket_profile(
        residue_labels=pocket_residues_universe,
        partner_chain_residues=partner_chain_residues,
    )
    case_specific_action_hints_payload = case_specific_action_hints(
        case_entry=case_entry,
        lead_descriptors=lead_descriptors,
        pocket_profile_payload=pocket_profile_payload,
        partner_chain_positions=partner_chain_positions,
    )
    transform_prior, transform_prior_summary = build_transform_prior(
        root=root,
        case_stage6_root=case_stage6_root,
        lead_smiles=lead_smiles,
        stage6=stage6,
        case_specific_action_hints_payload=case_specific_action_hints_payload,
    )

    return {
        "case_id": case_id,
        "target_domain": str(case_entry.get("target_domain") or ""),
        "evaluation_unit": str(case_entry.get("evaluation_unit") or ""),
        "stage3_5_root": str(stage3_5_root),
        "stage5_root": str(stage5_root),
        "stage6_root": str(case_stage6_root),
        "lead_smiles": lead_smiles,
        "lead_candidate_id": lead_id,
        "lead_descriptors": lead_descriptors,
        "lead_ligand_sdf": str(case_root / "stage1_5" / "raw" / "ligand.sdf"),
        "lead_wt_affinity_kcal_mol": lead_wt_affinity(stage3_5_root),
        "lead_mt_affinities": lead_mt_affinities,
        "panel_rows_full": panel_frame.to_dict(orient="records"),
        "panel_rows": panel_frame.to_dict(orient="records"),
        "panel_weights": panel_weights,
        "anchor_residues": anchor_residues,
        "hotspot_residues": hotspot_residues,
        "nonhotspot_residues": nonhotspot_residues,
        "pocket_residues_universe": pocket_residues_universe,
        "validated_pocket_residues": sorted(validated_pocket_residues),
        "partner_chain_residues": partner_chain_residues,
        "partner_chain_positions": partner_chain_positions,
        "baseline_ifp": baseline_ifp,
        "docking_box": dict(wt_ifp_payload.get("docking_box") or json.loads((stage3_5_root / "docking_box.json").read_text(encoding="utf-8"))),
        "wt_ifp_path": str(wt_ifp_path),
        "wt_complex_path": str(wt_complex_path),
        "wt_receptor_stage6_path": str(wt_receptor_stage6_path),
        "use_multichain_receptor": bool(use_multichain_receptor),
        "wt_receptor_chain_ids": wt_receptor_chain_ids,
        "partner_chain_ids": partner_chain_ids,
        "mechanism_summary": {
            "mechanism_label_counts": dict(sorted(mechanism_counts.items())),
            "anchor_residue_count": int(len(anchor_residues)),
            "hotspot_residue_count": int(len(hotspot_residues)),
            "validated_pocket_residue_count": int(len(validated_pocket_residues)),
            "ready_site_count": int(panel_frame["effect_scope"].eq("site").sum()) if not panel_frame.empty else 0,
            "ready_combo_count": int(panel_frame["effect_scope"].eq("combo").sum()) if not panel_frame.empty else 0,
            "receptor_ensemble_target_count": int(len(receptor_ensemble_members)),
        },
        "action_space": {
            "edit_families": list(EDIT_FAMILIES),
            "fragments": sorted(
                {
                    *[str(value) for value in list(stage6.get("fragment_library", {}).get("add", []))],
                    *[str(value) for value in list(stage6.get("fragment_library", {}).get("replace", []))],
                }
            ),
            "partner_chain_residues": partner_chain_residues,
            "partner_chain_positions": partner_chain_positions,
            "pocket_profile": pocket_profile_payload,
            "case_specific_action_hints": case_specific_action_hints_payload,
            "transform_prior_summary": transform_prior_summary,
            "transform_prior_top": [
                {
                    "edit_family": signature[0],
                    "pattern": signature[1],
                    "fragment": signature[2],
                    "score": float(score),
                }
                for signature, score in sorted(transform_prior.items(), key=lambda item: (-float(item[1]), item[0]))[:8]
            ],
        },
        "transform_prior": transform_prior,
        "calibrator_path": str(case_stage6_root / "calibrator" / "stage6_calibrator.pkl"),
        "calibrator_metadata": calibrator_metadata,
        "receptor_ensemble_enabled": bool(receptor_ensemble_enabled),
        "receptor_ensemble_members": receptor_ensemble_members,
        "receptor_ensemble_summary": receptor_ensemble_summary,
        "scoring_policy": {
            "dock_vs_mmgbsa_pearson_r": dock_mmgbsa_r,
            "uncertainty_heavy": bool(uncertainty_heavy),
            "calibrator_available": bool(calibrator_metadata.get("available", False)),
            **oracle_policy,
        },
        "hiv_reference": build_hiv_reference(
            root=root,
            case_entry=case_entry,
            stage2=load_yaml(root / "configs/base.yaml")["stage2"],
            stage3_5=load_yaml(root / "configs/base.yaml")["stage3_5"],
        ),
        "wt_chain_id": wt_chain_id,
    }


def _stage6_docking_config(stage6: dict[str, Any]) -> dict[str, Any]:
    return {
        "protein_prep_ph": float(stage6.get("protein_prep_ph", 7.4)),
        "use_pdbfixer": bool(stage6.get("use_pdbfixer", True)),
        "vina_seeds": [int(value) for value in list(stage6.get("vina_seeds", [11, 23, 37]))],
        "vina_exhaustiveness": int(stage6.get("vina_exhaustiveness", 8)),
        "vina_num_modes": int(stage6.get("vina_num_modes", 5)),
        "vina_energy_range": int(stage6.get("vina_energy_range", 3)),
        "vina_cpu_threads": int(stage6.get("vina_cpu_threads", 1)),
        "hiv_pose_contact_cutoff_a": float(stage6.get("hiv_pose_contact_cutoff_a", 6.0)),
        "hiv_required_pose_label": str(stage6.get("hiv_required_pose_label", "NNRTI_pocket")),
    }


def _stage6_dynamics_lite_config(stage6: dict[str, Any]) -> dict[str, Any]:
    raw = dict(stage6.get("dynamics_lite") or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "every_n_rounds": max(1, int(raw.get("every_n_rounds", 4))),
        "beam_top_n": max(1, int(raw.get("beam_top_n", 5))),
        "final_top_n": max(1, int(raw.get("final_top_n", raw.get("beam_top_n", 5)))),
        "reward_weight": float(raw.get("reward_weight", 0.20)),
        "reward_center": float(raw.get("reward_center", 0.50)),
        "contact_survival_weight": float(raw.get("contact_survival_weight", 0.45)),
        "occupancy_persistence_weight": float(raw.get("occupancy_persistence_weight", 0.30)),
        "anchor_persistence_weight": float(raw.get("anchor_persistence_weight", 0.25)),
        "fallback_penalty": float(raw.get("fallback_penalty", 0.05)),
        "local_sampling_ns": float(raw.get("local_sampling_ns", 0.1)),
        "local_sampling_backbone_restraint_kcal_mol_a2": float(
            raw.get("local_sampling_backbone_restraint_kcal_mol_a2", 1.0)
        ),
    }


def _stage6_dynamics_stage5_config(stage5: dict[str, Any], stage6: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(stage5)
    dynamics_cfg = _stage6_dynamics_lite_config(stage6)
    cfg["local_sampling_enabled"] = True
    cfg["local_sampling_ns"] = float(dynamics_cfg["local_sampling_ns"])
    cfg["local_sampling_backbone_restraint_kcal_mol_a2"] = float(
        dynamics_cfg["local_sampling_backbone_restraint_kcal_mol_a2"]
    )
    return cfg


def _stage6_pose_rows_json_path(output_root: Path) -> Path:
    return output_root / "all_pose_rows.json"


def _serialize_pose_rows(rows: list[dict[str, Any]], *, root: Path) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        pose_sdf = Path(str(row.get("pose_sdf") or ""))
        payload.append(
            {
                "seed": int(row.get("seed") or 0),
                "mode_rank": int(row.get("mode_rank") or 0),
                "affinity_kcal_mol": native_value(row.get("affinity_kcal_mol")),
                "rmsd_lb": native_value(row.get("rmsd_lb")),
                "rmsd_ub": native_value(row.get("rmsd_ub")),
                "pose_sdf": relative_path(pose_sdf, root) if pose_sdf.exists() else str(row.get("pose_sdf") or ""),
            }
        )
    return payload


def _load_stage6_pose_rows(*, root: Path, output_root: Path) -> list[dict[str, Any]]:
    pose_rows_path = _stage6_pose_rows_json_path(output_root)
    if pose_rows_path.exists():
        payload = parse_json_payload(pose_rows_path.read_text(encoding="utf-8")) or []
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, dict)]
    vina_root = output_root / "vina_runs"
    if not vina_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(vina_root.glob("seed_*")):
        match = re.search(r"seed_(\d+)$", seed_dir.name)
        if not match:
            continue
        seed = int(match.group(1))
        out_pdbqt = seed_dir / "vina_out.pdbqt"
        if not out_pdbqt.exists():
            continue
        vina_rows = vina_result_rows(out_pdbqt)
        pose_sd_files = sorted(seed_dir.glob("pose*.sdf"))
        for pose_row, pose_sdf in zip(vina_rows, pose_sd_files):
            rows.append(
                {
                    **pose_row,
                    "seed": int(seed),
                    "pose_sdf": relative_path(pose_sdf, root),
                }
            )
    if rows:
        json_dump(pose_rows_path, rows)
    return rows


def _dynamics_lite_reference_root(case_context: dict[str, Any]) -> Path:
    return ensure_dir(Path(str(case_context["stage6_root"])) / "dynamics_lite" / "lead_reference")


def _candidate_dynamics_root(case_context: dict[str, Any], candidate_id_text: str) -> Path:
    return ensure_dir(Path(str(case_context["stage6_root"])) / "cache" / "candidates" / candidate_id_text / "dynamics_lite")


def _dynamics_lite_lead_reference(
    *,
    root: Path,
    case_context: dict[str, Any],
) -> dict[str, Any]:
    reference_root = _dynamics_lite_reference_root(case_context)
    summary_path = reference_root / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("available") is True:
            return payload
    lead_candidate_id = str(case_context["lead_candidate_id"])
    lead_docking_root = Path(str(case_context["stage6_root"])) / "cache" / "candidates" / lead_candidate_id / "wt"
    lead_summary_path = lead_docking_root / "summary.json"
    if not lead_summary_path.exists():
        payload = {"available": False, "reason": "missing_lead_summary"}
        json_dump(summary_path, payload)
        return payload
    lead_summary = json.loads(lead_summary_path.read_text(encoding="utf-8"))
    receptor_pdb = root / str(lead_summary.get("receptor_pdb") or "")
    pose_rows = _load_stage6_pose_rows(root=root, output_root=lead_docking_root)
    if not receptor_pdb.exists() or not pose_rows:
        payload = {
            "available": False,
            "reason": "missing_lead_pose_rows",
            "lead_candidate_id": lead_candidate_id,
        }
        json_dump(summary_path, payload)
        return payload
    occupancy_payload = occupancy_frequency_payload(
        root=root,
        run_root=reference_root,
        receptor_pdb=receptor_pdb,
        pose_rows=pose_rows,
        pose_set="lead",
    )
    payload = {
        "available": True,
        "lead_candidate_id": lead_candidate_id,
        "top_seed_count": int(occupancy_payload.get("top_seed_count", 0)),
        "occupancy_map": dict(occupancy_payload.get("occupancy_map") or {}),
    }
    json_dump(summary_path, payload)
    return payload


def run_stage6_dynamics_lite_probe(
    *,
    root: Path,
    case_context: dict[str, Any],
    stage5: dict[str, Any],
    stage6: dict[str, Any],
    candidate_id_text: str,
) -> dict[str, Any]:
    dynamics_cfg = _stage6_dynamics_lite_config(stage6)
    dynamics_root = _candidate_dynamics_root(case_context, candidate_id_text)
    summary_path = dynamics_root / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("available") is True:
            return payload
    candidate_docking_root = Path(str(case_context["stage6_root"])) / "cache" / "candidates" / candidate_id_text / "wt"
    docking_summary_path = candidate_docking_root / "summary.json"
    if not docking_summary_path.exists():
        payload = {"available": False, "candidate_id": candidate_id_text, "reason": "missing_candidate_summary"}
        json_dump(summary_path, payload)
        return payload
    docking_summary = json.loads(docking_summary_path.read_text(encoding="utf-8"))
    receptor_pdb = root / str(docking_summary.get("receptor_pdb") or "")
    pose_sdf = root / str(docking_summary.get("pose_sdf") or "")
    if not receptor_pdb.exists() or not pose_sdf.exists():
        payload = {"available": False, "candidate_id": candidate_id_text, "reason": "missing_candidate_pose"}
        json_dump(summary_path, payload)
        return payload
    reference_payload = _dynamics_lite_lead_reference(root=root, case_context=case_context)
    if not bool(reference_payload.get("available", False)):
        payload = {"available": False, "candidate_id": candidate_id_text, "reason": "missing_lead_reference"}
        json_dump(summary_path, payload)
        return payload
    candidate_pose_rows = _load_stage6_pose_rows(root=root, output_root=candidate_docking_root)
    if not candidate_pose_rows:
        payload = {"available": False, "candidate_id": candidate_id_text, "reason": "missing_candidate_pose_rows"}
        json_dump(summary_path, payload)
        return payload
    try:
        stage5_cfg = _stage6_dynamics_stage5_config(stage5, stage6)
        ligand_parameter_cache_root = ensure_dir(Path(str(case_context["stage6_root"])) / "dynamics_lite" / "ligand_param_cache")
        amber_prep = prepare_amber_complex(
            receptor_pdb=receptor_pdb,
            ligand_pose_sdf=pose_sdf,
            work_root=dynamics_root / "amber_prep",
            ligand_template_sdf=root / str(case_context.get("lead_ligand_sdf") or ""),
            ligand_parameter_cache_root=ligand_parameter_cache_root,
        )
        relaxation_payload = run_amber_relaxation(
            amber_prep=amber_prep,
            work_root=dynamics_root / "relaxation",
            stage5=stage5_cfg,
            run_local_sampling=True,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        refined_complex_pdb = Path(str(relaxation_payload.get("refined_complex_pdb") or ""))
        if not refined_complex_pdb.exists():
            raise RuntimeError("missing_refined_complex")
        refined_ifp = plip_ifp(refined_complex_pdb)
        candidate_occupancy = occupancy_frequency_payload(
            root=root,
            run_root=dynamics_root,
            receptor_pdb=receptor_pdb,
            pose_rows=candidate_pose_rows,
            pose_set="candidate",
        )
        occupancy_shift = occupancy_shift_metrics(
            wt_payload={
                "top_seed_count": int(reference_payload.get("top_seed_count", 0)),
                "occupancy_map": dict(reference_payload.get("occupancy_map") or {}),
            },
            mt_payload=candidate_occupancy,
            anchor_labels=set(case_context.get("anchor_residues") or []),
        )
        contact_survival = weighted_ifp_cosine(
            baseline_ifp=dict(case_context.get("baseline_ifp") or {}),
            candidate_ifp=refined_ifp,
            residues=set(case_context.get("pocket_residues_universe") or []),
            stage6=stage6,
        )
        anchor_contact_survival = weighted_ifp_cosine(
            baseline_ifp=dict(case_context.get("baseline_ifp") or {}),
            candidate_ifp=refined_ifp,
            residues=set(case_context.get("anchor_residues") or []),
            stage6=stage6,
        )
        occupancy_persistence = _clamp01(1.0 - float(occupancy_shift.get("ifp_occupancy_shift_mean_abs") or 0.0))
        anchor_persistence = _clamp01(1.0 - float(occupancy_shift.get("ifp_occupancy_anchor_loss") or 0.0))
        score = (
            float(dynamics_cfg["contact_survival_weight"]) * float(contact_survival)
            + float(dynamics_cfg["occupancy_persistence_weight"]) * float(occupancy_persistence)
            + float(dynamics_cfg["anchor_persistence_weight"]) * float(anchor_persistence)
        )
        fallback_penalty = (
            float(dynamics_cfg["fallback_penalty"])
            if str(relaxation_payload.get("relaxation_mode") or "") != "local_sampling_implicit_md"
            else 0.0
        )
        score = float(max(0.0, min(1.0, score - fallback_penalty)))
        payload = {
            "available": True,
            "candidate_id": candidate_id_text,
            "contact_survival": float(contact_survival),
            "anchor_contact_survival": float(anchor_contact_survival),
            "occupancy_persistence": float(occupancy_persistence),
            "anchor_persistence": float(anchor_persistence),
            "ifp_occupancy_shift_mean_abs": float(occupancy_shift.get("ifp_occupancy_shift_mean_abs") or 0.0),
            "ifp_occupancy_anchor_loss": float(occupancy_shift.get("ifp_occupancy_anchor_loss") or 0.0),
            "score": float(score),
            "fallback_penalty": float(fallback_penalty),
            "relaxation_mode": str(relaxation_payload.get("relaxation_mode") or ""),
            "run_local_sampling": bool(relaxation_payload.get("run_local_sampling", False)),
            "local_sampling_fallback_reason": str(relaxation_payload.get("local_sampling_fallback_reason") or ""),
        }
    except Exception as exc:
        payload = {
            "available": False,
            "candidate_id": candidate_id_text,
            "reason": f"dynamics_lite_failed:{type(exc).__name__}",
            "local_sampling_fallback_reason": str(exc),
        }
    json_dump(summary_path, payload)
    return payload


def _dock_candidate(
    *,
    root: Path,
    receptor_pdb: Path,
    docking_box: dict[str, Any],
    ligand_input_sdf: Path,
    output_root: Path,
    stage6: dict[str, Any],
    hiv_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    if _stage6_docking_cache_is_healthy(output_root):
        return json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    preserved_receptor_text: str | None = None
    try:
        receptor_within_output = receptor_pdb.exists() and receptor_pdb.resolve().is_relative_to(output_root.resolve())
    except Exception:
        receptor_within_output = False
    if receptor_within_output:
        preserved_receptor_text = receptor_pdb.read_text(encoding="utf-8")
    if output_root.exists():
        shutil.rmtree(output_root, ignore_errors=True)
    ensure_dir(output_root)
    if preserved_receptor_text is not None:
        ensure_dir(receptor_pdb.parent)
        receptor_pdb.write_text(preserved_receptor_text, encoding="utf-8")
    docking_cfg = _stage6_docking_config(stage6)
    try:
        receptor_input = receptor_pdb
        if bool(docking_cfg["use_pdbfixer"]):
            fixed_pdb = output_root / "receptor_fixed.pdb"
            try:
                run_pdbfixer(receptor_pdb, fixed_pdb, float(docking_cfg["protein_prep_ph"]))
                receptor_input = fixed_pdb
            except Exception:
                receptor_input = receptor_pdb
        receptor_pdbqt = output_root / "receptor.pdbqt"
        ligand_standardized = output_root / "ligand_standardized.sdf"
        ligand_pdbqt = output_root / "ligand.pdbqt"
        prepare_receptor_pdbqt(receptor_input, receptor_pdbqt, float(docking_cfg["protein_prep_ph"]))
        standardize_reference_ligand(ligand_input_sdf, ligand_standardized)
        prepare_ligand_pdbqt(ligand_standardized, ligand_pdbqt, float(docking_cfg["protein_prep_ph"]))
        rows = run_vina_redocking(
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=ligand_pdbqt,
            docking_box=docking_box,
            output_root=output_root / "vina_runs",
            seeds=list(docking_cfg["vina_seeds"]),
            exhaustiveness=int(docking_cfg["vina_exhaustiveness"]),
            num_modes=int(docking_cfg["vina_num_modes"]),
            energy_range=int(docking_cfg["vina_energy_range"]),
            cpu_threads=int(docking_cfg["vina_cpu_threads"]),
        )
        pose_rows_path = _stage6_pose_rows_json_path(output_root)
        json_dump(pose_rows_path, _serialize_pose_rows(rows, root=root))
        if hiv_reference is not None:
            for row in rows:
                row.update(
                    classify_hiv_pose(
                        Path(str(row["pose_sdf"])),
                        hiv_reference["nnrti_residue_coords"],
                        hiv_reference["active_site_residue_coords"],
                        float(docking_cfg["hiv_pose_contact_cutoff_a"]),
                    )
                )
        best_row = select_best_pose(rows, hiv_mode=hiv_reference is not None)
        if best_row is None:
            raise RuntimeError("No valid counter-design step pose selected.")
        if hiv_reference is not None and str(best_row.get("pose_label") or "") != str(docking_cfg["hiv_required_pose_label"]):
            raise RuntimeError(f"HIV pose left required pocket: {best_row.get('pose_label')}")
        pose_sdf = Path(str(best_row["pose_sdf"]))
        stable_pose_sdf = output_root / "best_pose.sdf"
        shutil.copyfile(pose_sdf, stable_pose_sdf)
        stable_receptor = output_root / "receptor_stage6.pdb"
        shutil.copyfile(receptor_input, stable_receptor)
        pose_pdb = output_root / "best_pose.pdb"
        pose_pdb_from_sdf(stable_pose_sdf, pose_pdb)
        complex_pdb = output_root / "complex_docked.pdb"
        merge_pdb_fragments([stable_receptor, pose_pdb], complex_pdb)
        ifp_payload = plip_ifp(complex_pdb)
        ifp_path = output_root / "ifp.json"
        json_dump(ifp_path, ifp_payload)
        summary = {
            "docking_status": "ok",
            "docking_error": None,
            "best_affinity_kcal_mol": float(best_row["affinity_kcal_mol"]),
            "pose_label": str(best_row.get("pose_label") or ""),
            "pose_sdf": relative_path(stable_pose_sdf, root),
            "receptor_pdb": relative_path(stable_receptor, root),
            "complex_pdb": relative_path(complex_pdb, root),
            "ifp_json": relative_path(ifp_path, root),
            "pose_rows_json": relative_path(pose_rows_path, root),
            "finished_at": iso_now(),
        }
    except Exception as exc:
        summary = {
            "docking_status": "failed",
            "docking_error": f"{type(exc).__name__}: {exc}",
            "best_affinity_kcal_mol": None,
            "pose_label": "",
            "pose_sdf": "",
            "receptor_pdb": relative_path(receptor_pdb, root) if receptor_pdb.exists() else "",
            "complex_pdb": "",
            "ifp_json": "",
            "pose_rows_json": "",
            "finished_at": iso_now(),
        }
    json_dump(output_root / "summary.json", summary)
    return summary


def _has_atom_records(path: Path) -> bool:
    if not path.exists():
        return False
    return any(line.startswith(("ATOM", "HETATM")) for line in path.read_text(encoding="utf-8", errors="ignore").splitlines())


def _pdb_has_duplicate_atom_names(path: Path) -> bool:
    if not path.exists():
        return True
    seen: set[tuple[str, str, str, str, str]] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        key = (
            line[12:16],
            line[17:20],
            line[21],
            line[22:26],
            line[26],
        )
        if key in seen:
            return True
        seen.add(key)
    return False


def _stage6_docking_cache_is_healthy(output_root: Path) -> bool:
    summary_path = output_root / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if str(summary.get("docking_status") or "") != "ok":
        return False
    required = [
        output_root / "best_pose.sdf",
        output_root / "best_pose.pdb",
        output_root / "receptor_stage6.pdb",
        output_root / "complex_docked.pdb",
        output_root / "ifp.json",
    ]
    if any(not path.exists() for path in required):
        return False
    pose_pdb = output_root / "best_pose.pdb"
    if _pdb_has_duplicate_atom_names(pose_pdb):
        return False
    return True


def _wt_receptor_pdb(case_context: dict[str, Any]) -> Path:
    stage3_5_root = Path(str(case_context["stage3_5_root"]))
    receptor_hint = Path(str(case_context.get("wt_receptor_stage6_path") or stage3_5_root / "wt_receptor_stage6.pdb"))
    source_complex = Path(str(case_context.get("wt_complex_path") or stage3_5_root / "wt_complex.pdb"))

    if _has_atom_records(receptor_hint):
        return receptor_hint

    receptor_pdb = stage3_5_root / "wt_receptor_stage6.pdb"
    if _has_atom_records(receptor_pdb):
        return receptor_pdb

    preferred_chain = str(case_context["wt_chain_id"])
    save_chain_protein(source_complex, preferred_chain, receptor_pdb)
    if _has_atom_records(receptor_pdb):
        return receptor_pdb

    fallback_chain = first_protein_chain_id(source_complex)
    if fallback_chain and str(fallback_chain) != preferred_chain:
        save_chain_protein(source_complex, str(fallback_chain), receptor_pdb)
        if _has_atom_records(receptor_pdb):
            return receptor_pdb

    shutil.copyfile(source_complex, receptor_pdb)
    return receptor_pdb


def _target_box(sample_root: Path, work_root: Path, stage6: dict[str, Any]) -> dict[str, Any]:
    lead_sdf = stage5_reference_ligand_input(sample_root, work_root)
    reference_standardized = work_root / "reference_ligand.sdf"
    standardize_reference_ligand(lead_sdf, reference_standardized)
    try:
        reference_ligand = load_rdkit_molecule(reference_standardized, sanitize=True)
    except ValueError:
        reference_ligand = load_rdkit_molecule(reference_standardized, sanitize=False)
    coords = mol_coordinates(reference_ligand)
    return build_box_from_ligand_coords(
        coords,
        default_box_size_a=float(stage6.get("default_box_size_a", 22.0)),
        ligand_padding_a=float(stage6.get("ligand_box_padding_a", 8.0)),
        source="stage6_reference_ligand",
    )


def _target_parallel_workers(stage6: dict[str, Any], panel_size: int) -> int:
    return max(1, min(int(stage6.get("target_parallel_workers", 4)), int(panel_size)))


def _target_panel_receptor_pdb(
    *,
    row: dict[str, Any],
    case_context: dict[str, Any],
    output_root: Path,
) -> Path:
    sample_root = Path(str(row["sample_root"]))
    base_receptor = sample_root / "MT.pdb"
    if not bool(case_context.get("use_multichain_receptor", False)):
        return base_receptor
    partner_chain_ids = [str(value) for value in list(case_context.get("partner_chain_ids") or []) if str(value)]
    wt_multichain_receptor = Path(str(case_context.get("wt_receptor_stage6_path") or ""))
    if not partner_chain_ids or not wt_multichain_receptor.exists():
        return base_receptor
    output_pdb = output_root / "receptor_multichain_stage6.pdb"
    if _has_atom_records(output_pdb):
        return output_pdb
    primary_chain_id = str(case_context.get("wt_chain_id") or first_protein_chain_id(base_receptor))
    primary_chain_pdb = output_root / "receptor_primary_chain.pdb"
    partner_chain_pdb = output_root / "receptor_partner_chains.pdb"
    save_chain_protein(base_receptor, primary_chain_id, primary_chain_pdb)
    save_chain_set_protein(wt_multichain_receptor, partner_chain_ids, partner_chain_pdb)
    merge_pdb_fragments([primary_chain_pdb, partner_chain_pdb], output_pdb)
    return output_pdb


def _dock_target_panel_row(
    *,
    root: Path,
    row: dict[str, Any],
    candidate_sdf: Path,
    case_context: dict[str, Any],
    stage6: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    target_key = str(row["target_key"])
    effect_scope = str(row.get("effect_scope") or "")
    target_root = candidate_sdf.parent / "targets" / stable_target_slug(effect_scope, target_key)
    ensemble_members = list(dict(case_context.get("receptor_ensemble_members") or {}).get(target_key) or [])
    runtime_member_cap = int(
        stage6.get(
            "receptor_ensemble_runtime_max_members",
            stage6.get("receptor_ensemble_max_members", 0) or 0,
        )
    )
    if runtime_member_cap > 0 and ensemble_members:
        ensemble_members = sorted(
            ensemble_members,
            key=lambda member: (
                int(member.get("member_rank") or 9999),
                str(member.get("member_id") or ""),
            ),
        )[:runtime_member_cap]
    if len(ensemble_members) > 1:
        member_summaries: list[dict[str, Any]] = []
        for member in ensemble_members:
            member_id = str(member.get("member_id") or member.get("sample_id") or f"member_{len(member_summaries) + 1}")
            member_row = {**row, "sample_root": str(member.get("sample_root") or row.get("sample_root") or "")}
            member_root = target_root / "ensemble" / member_id
            member_sample_root = Path(str(member_row["sample_root"]))
            receptor_pdb = _target_panel_receptor_pdb(
                row=member_row,
                case_context=case_context,
                output_root=member_root,
            )
            summary = _dock_candidate(
                root=root,
                receptor_pdb=receptor_pdb,
                docking_box=_target_box(member_sample_root, member_root, stage6),
                ligand_input_sdf=candidate_sdf,
                output_root=member_root,
                stage6=stage6,
                hiv_reference=dict(case_context["hiv_reference"]) if case_context.get("hiv_reference") else None,
            )
            member_summaries.append(
                {
                    **dict(member),
                    **summary,
                }
            )
        success_count = sum(1 for member in member_summaries if str(member.get("docking_status") or "") == "ok")
        return target_key, effect_scope, {
            "docking_status": "ok" if success_count > 0 else "failed",
            "ensemble_member_count": int(len(member_summaries)),
            "ensemble_member_cap": int(runtime_member_cap) if runtime_member_cap > 0 else int(len(member_summaries)),
            "ensemble_success_count": int(success_count),
            "ensemble_aggregate": str(stage6.get("receptor_ensemble_aggregate", "median")),
            "ensemble_members": member_summaries,
        }
    sample_root = Path(str(row["sample_root"]))
    receptor_pdb = _target_panel_receptor_pdb(
        row=row,
        case_context=case_context,
        output_root=target_root,
    )
    summary = _dock_candidate(
        root=root,
        receptor_pdb=receptor_pdb,
        docking_box=_target_box(sample_root, target_root, stage6),
        ligand_input_sdf=candidate_sdf,
        output_root=target_root,
        stage6=stage6,
        hiv_reference=dict(case_context["hiv_reference"]) if case_context.get("hiv_reference") else None,
    )
    return target_key, effect_scope, summary


def ensure_stage6_reference_affinities(
    *,
    root: Path,
    case_context: dict[str, Any],
    stage6: dict[str, Any],
) -> dict[str, Any]:
    stage6_root = Path(str(case_context["stage6_root"]))
    reference_path = stage6_root / "reference_lead_affinities.json"
    if reference_path.exists():
        payload = json.loads(reference_path.read_text(encoding="utf-8"))
        if str(payload.get("lead_candidate_id") or "") == str(case_context["lead_candidate_id"]):
            return payload

    lead_smiles = str(case_context["lead_smiles"])
    lead_candidate_id = str(case_context["lead_candidate_id"])
    candidate_root = ensure_dir(stage6_root / "cache" / "candidates" / lead_candidate_id)
    candidate_sdf = candidate_root / "candidate.sdf"
    if not candidate_sdf.exists():
        write_candidate_sdf(lead_smiles, candidate_sdf)

    wt_summary = _dock_candidate(
        root=root,
        receptor_pdb=_wt_receptor_pdb(case_context),
        docking_box=dict(case_context["docking_box"]),
        ligand_input_sdf=candidate_sdf,
        output_root=candidate_root / "wt",
        stage6=stage6,
        hiv_reference=dict(case_context["hiv_reference"]) if case_context.get("hiv_reference") else None,
    )
    fallback_target_affinities = dict(case_context.get("lead_mt_affinities") or {})
    target_affinities: dict[str, float | None] = {}
    target_statuses: dict[str, str] = {}
    panel_rows = list(case_context["panel_rows"])
    max_workers = _target_parallel_workers(stage6, len(panel_rows))
    if max_workers == 1:
        results = [
            _dock_target_panel_row(
                root=root,
                row=row,
                candidate_sdf=candidate_sdf,
                case_context=case_context,
                stage6=stage6,
            )
            for row in panel_rows
        ]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _dock_target_panel_row,
                    root=root,
                    row=row,
                    candidate_sdf=candidate_sdf,
                    case_context=case_context,
                    stage6=stage6,
                )
                for row in panel_rows
            ]
            for future in as_completed(futures):
                results.append(future.result())
    for target_key, _, target_summary in results:
        target_statuses[target_key] = str(target_summary.get("docking_status") or "failed")
        affinity = native_value(target_summary.get("best_affinity_kcal_mol"))
        if affinity is None:
            affinity = native_value(fallback_target_affinities.get(target_key))
        target_affinities[target_key] = affinity

    payload = {
        "lead_candidate_id": lead_candidate_id,
        "generated_at": iso_now(),
        "wt_affinity_kcal_mol": native_value(wt_summary.get("best_affinity_kcal_mol"))
        if str(wt_summary.get("docking_status") or "") == "ok"
        else native_value(case_context.get("lead_wt_affinity_kcal_mol")),
        "wt_docking_status": str(wt_summary.get("docking_status") or "failed"),
        "target_affinities": target_affinities,
        "target_docking_statuses": target_statuses,
    }
    json_dump(reference_path, payload)
    return payload


def _safe_local_pocket_metrics(receptor_pdb: Path, ligand_sdf: Path) -> dict[str, Any]:
    try:
        return local_pocket_metrics(
            receptor_pdb=receptor_pdb,
            ligand_sdf=ligand_sdf,
            pocket_contact_distance_a=6.0,
            polar_contact_distance_a=4.0,
            solvent_exposure_neighbor_threshold=4,
        )
    except Exception:
        return {
            "pocket_volume_proxy_a3": 0.0,
            "contact_density": 0.0,
            "polar_atom_count": 0,
            "polar_exposed_fraction": 0.0,
            "pocket_contact_atom_count": 0,
        }


def _layer_keep_ifp_scores(
    *,
    baseline_ifp: dict[str, Any],
    candidate_ifp: dict[str, Any],
    anchor_residues: set[str],
    backbone_support_residues: set[str],
    partner_chain_residues: set[str],
    nonhotspot_residues: set[str],
    case_context: dict[str, Any],
    stage6: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    residue_sets = {
        "keep_ifp_anchor": set(anchor_residues),
        "keep_ifp_backbone": set(backbone_support_residues),
        "keep_ifp_partner_chain": set(partner_chain_residues),
        "keep_ifp_nonhotspot": set(nonhotspot_residues),
    }
    layer_weights = reward_v2_layer_weights(
        case_id=str(case_context.get("case_id") or ""),
        target_domain=str(case_context.get("target_domain") or ""),
        stage6=stage6,
        layer_available={name: bool(residues) for name, residues in residue_sets.items()},
    )
    layer_scores = {
        name: (
            float(
                weighted_ifp_cosine(
                    baseline_ifp=baseline_ifp,
                    candidate_ifp=candidate_ifp,
                    residues=residue_sets[name],
                    stage6=stage6,
                )
            )
            if residue_sets[name]
            else 0.0
        )
        for name in LAYER_NAMES
    }
    return layer_scores, layer_weights


def _target_candidate_features(
    *,
    root: Path,
    wt_summary: dict[str, Any],
    wt_ifp: dict[str, Any],
    wt_affinity: float | None,
    wt_pocket: dict[str, Any],
    target_summary: dict[str, Any],
    target_ifp: dict[str, Any],
    anchor_residues: set[str],
    backbone_support_residues: set[str],
    partner_chain_residues: set[str],
    nonhotspot_residues: set[str],
    case_context: dict[str, Any],
    row: dict[str, Any],
    stage6: dict[str, Any],
) -> dict[str, Any]:
    target_affinity = native_value(target_summary.get("best_affinity_kcal_mol"))
    target_receptor = root / str(target_summary.get("receptor_pdb") or "")
    target_pose = root / str(target_summary.get("pose_sdf") or "")
    target_pocket = _safe_local_pocket_metrics(target_receptor, target_pose) if target_receptor.exists() and target_pose.exists() else {
        "pocket_volume_proxy_a3": 0.0,
        "polar_exposed_fraction": 0.0,
    }
    volume_change_fraction = 0.0
    wt_volume = float(wt_pocket.get("pocket_volume_proxy_a3") or 0.0)
    if wt_volume > 0.0:
        volume_change_fraction = float((float(target_pocket.get("pocket_volume_proxy_a3") or 0.0) - wt_volume) / wt_volume)
    solvent_proxy_shift = float(
        float(target_pocket.get("polar_exposed_fraction") or 0.0) - float(wt_pocket.get("polar_exposed_fraction") or 0.0)
    )
    occupancy_payload = ifp_occupancy_shift(wt_ifp, target_ifp, anchor_residues)
    keep_ifp_value = weighted_ifp_cosine(
        baseline_ifp=wt_ifp,
        candidate_ifp=target_ifp,
        residues=anchor_residues,
        stage6=stage6,
    )
    layer_scores, layer_weights = _layer_keep_ifp_scores(
        baseline_ifp=wt_ifp,
        candidate_ifp=target_ifp,
        anchor_residues=anchor_residues,
        backbone_support_residues=backbone_support_residues,
        partner_chain_residues=partner_chain_residues,
        nonhotspot_residues=nonhotspot_residues,
        case_context=case_context,
        stage6=stage6,
    )
    anchor_loss_fraction_value = anchor_loss_fraction(target_ifp, anchor_residues)
    delta_dock = None
    if target_affinity is not None and wt_affinity is not None:
        delta_dock = float(target_affinity) - float(wt_affinity)
    mechanism_labels = target_mechanism_labels(
        delta_dock_kcal_mol=delta_dock,
        keep_ifp_value=float(keep_ifp_value),
        anchor_loss_fraction_value=float(anchor_loss_fraction_value),
        occupancy_shift_mean_abs=float(occupancy_payload["ifp_occupancy_shift_mean_abs"]),
        occupancy_anchor_loss=float(occupancy_payload["ifp_occupancy_anchor_loss"]),
        solvent_proxy_shift=float(solvent_proxy_shift),
    )
    return {
        "effect_scope": str(row.get("effect_scope") or ""),
        "stage4_local_rmsd_a": native_value(row.get("stage4_local_rmsd_a")),
        "delta_dock_kcal_mol": native_value(delta_dock),
        "delta_gnina_affinity_kcal_mol": None,
        "ifp_jaccard_loss": float(jaccard_loss(set(wt_ifp.get("residue_set", [])), set(target_ifp.get("residue_set", [])))),
        "ifp_occupancy_shift_mean_abs": float(occupancy_payload["ifp_occupancy_shift_mean_abs"]),
        "ifp_occupancy_anchor_loss": float(occupancy_payload["ifp_occupancy_anchor_loss"]),
        "anchor_loss_fraction": float(anchor_loss_fraction_value),
        "pocket_volume_change_fraction": float(volume_change_fraction),
        "solvent_proxy_shift": float(solvent_proxy_shift),
        "keep_ifp": float(weighted_keep_ifp(layer_scores, layer_weights)),
        **layer_scores,
        "lost_anchor_labels": lost_anchor_labels(target_ifp, anchor_residues),
        "mechanism_labels": mechanism_labels,
        "candidate_affinity_kcal_mol": target_affinity,
    }


def evaluate_candidate(
    *,
    root: Path,
    case_context: dict[str, Any],
    stage6: dict[str, Any],
    smiles: str,
    action_sequence: list[dict[str, Any]],
    round_index: int,
    objective_name: str,
) -> dict[str, Any]:
    smiles = canonical_smiles(smiles)
    cid = candidate_id(smiles)
    case_root = Path(str(case_context["stage6_root"]))
    candidate_root = ensure_dir(case_root / "cache" / "candidates" / cid)
    candidate_sdf = candidate_root / "candidate.sdf"
    write_candidate_sdf(smiles, candidate_sdf)
    scale = float(stage6.get("binding_score_scale_kcal_mol", 1.5))
    scoring_policy = dict(case_context.get("scoring_policy") or {})
    calibrator = load_case_calibrator(str(case_context.get("calibrator_path") or "")) if str(case_context.get("calibrator_path") or "") else None
    oracle_v2 = (
        load_stage6_oracle_v2(str(scoring_policy.get("oracle_v2_model_path") or ""))
        if bool(scoring_policy.get("oracle_v2_available", False))
        else None
    )
    prefilter = apply_prefilters(
        smiles,
        stage6,
        baseline_descriptors=dict(case_context.get("lead_descriptors") or {}),
    )
    lead_wt_affinity_value = native_value(case_context.get("lead_wt_affinity_kcal_mol"))

    base_row = {
        "case_id": str(case_context["case_id"]),
        "candidate_id": cid,
        "smiles": smiles,
        "round_index": int(round_index),
        "objective_name": str(objective_name),
        "action_sequence_json": json.dumps(action_sequence, ensure_ascii=True),
        "scaffold_smiles": murcko_scaffold(smiles),
        **prefilter,
    }

    def failure_row(*, mechanism_focus: str, wt_status: str, docking_success: bool) -> dict[str, Any]:
        return {
            **base_row,
            "chemical_valid": True,
            "docking_success": bool(docking_success),
            "wt_docking_status": str(wt_status),
            "wt_affinity_kcal_mol": None,
            "lead_wt_affinity_kcal_mol": lead_wt_affinity_value,
            "s_wt": 0.0,
            "wt_keep_ifp": 0.0,
            "wt_hard_constraint_pass": False,
            "wt_pass": False,
            "keep_ifp": 0.0,
            "keep_ifp_constraint_pass": False,
            "key_anchor_count": 0,
            "hotspot_contact_score": 0.0,
            "nonhotspot_contact_score": 0.0,
            "hotspot_fraction": 1.0,
            "dep": 1.0,
            "hotspot_drop": 0.0,
            "new_nonhotspot_contact_score": 0.0,
            "new_nonhotspot_residue_count": 0,
            "compensation_gain": 0.0,
            "compensation_constraint_pass": False,
            "panel_coverage": 0.0,
            "coverage_pass": False,
            "panel_coverage_pass": False,
            "high_uncertainty": True,
            "target_success_count": 0,
            "target_uncertain_count": int(len(case_context.get("panel_rows") or [])),
            "robust_core": 0.0,
            "robust_score": 0.0,
            "naive_mean_affinity": 0.0,
            "combo_robust_core": None,
            "combo_naive_mean_affinity": None,
            "coverage_penalty": float(stage6.get("coverage_penalty", 0.25)),
            "constraint_penalty": 0.0,
            "robust_objective_reward": 0.0,
            "naive_objective_reward": 0.0,
            "objective_reward": 0.0,
            "panel_passing": False,
            "candidate_valid": False,
            "mechanism_risk_focus": mechanism_focus,
            "target_scores_json": "{}",
        }

    if not bool(prefilter["prefilter_pass"]):
        return failure_row(mechanism_focus="prefilter_failed", wt_status="skipped", docking_success=False)

    wt_root = candidate_root / "wt"
    wt_summary = _dock_candidate(
        root=root,
        receptor_pdb=_wt_receptor_pdb(case_context),
        docking_box=dict(case_context["docking_box"]),
        ligand_input_sdf=candidate_sdf,
        output_root=wt_root,
        stage6=stage6,
        hiv_reference=dict(case_context["hiv_reference"]) if case_context.get("hiv_reference") else None,
    )
    if wt_summary.get("docking_status") != "ok":
        return failure_row(
            mechanism_focus="wt_docking_failed",
            wt_status=str(wt_summary.get("docking_status") or "failed"),
            docking_success=False,
        )

    wt_ifp = json.loads((root / str(wt_summary["ifp_json"])).read_text(encoding="utf-8"))
    anchor_residues = set(case_context["anchor_residues"])
    hotspot_residues = set(case_context["hotspot_residues"])
    pocket_residues = set(case_context.get("pocket_residues_universe") or []) | set(wt_ifp.get("residue_set", []))
    nonhotspot_residues = pocket_residues - hotspot_residues
    partner_chain_residues = set(case_context.get("partner_chain_residues") or [])
    baseline_ifp = dict(case_context["baseline_ifp"])
    backbone_support_residues = (set(baseline_ifp.get("residue_set", [])) - hotspot_residues - partner_chain_residues) or nonhotspot_residues
    wt_layer_scores, layer_weights = _layer_keep_ifp_scores(
        baseline_ifp=baseline_ifp,
        candidate_ifp=wt_ifp,
        anchor_residues=anchor_residues,
        backbone_support_residues=backbone_support_residues,
        partner_chain_residues=partner_chain_residues,
        nonhotspot_residues=nonhotspot_residues,
        case_context=case_context,
        stage6=stage6,
    )
    wt_keep_ifp = float(weighted_keep_ifp(wt_layer_scores, layer_weights))
    key_anchor_count = anchor_contact_count(wt_ifp, anchor_residues)
    lead_hotspot_score = contact_score(baseline_ifp, hotspot_residues, stage6)
    lead_nonhotspot_score = contact_score(baseline_ifp, nonhotspot_residues, stage6)
    candidate_hotspot_score = contact_score(wt_ifp, hotspot_residues, stage6)
    candidate_nonhotspot_score = contact_score(wt_ifp, nonhotspot_residues, stage6)
    dep_value = float(dep_score(wt_ifp, hotspot_residues, pocket_residues, stage6))
    hotspot_fraction_value = float(hotspot_fraction(wt_ifp, hotspot_residues, pocket_residues, stage6))
    hotspot_drop = float(max(0.0, lead_hotspot_score - candidate_hotspot_score))
    new_nonhotspot_metrics = new_nonhotspot_contact_metrics(
        baseline_ifp=baseline_ifp,
        candidate_ifp=wt_ifp,
        hotspot_residues=hotspot_residues,
        stage6=stage6,
    )
    compensation_gain = float(max(0.0, candidate_nonhotspot_score - lead_nonhotspot_score))
    wt_affinity = native_value(wt_summary.get("best_affinity_kcal_mol"))
    wt_receptor_pdb = root / str(wt_summary.get("receptor_pdb") or "")
    wt_pose_sdf = root / str(wt_summary.get("pose_sdf") or "")
    wt_pocket = _safe_local_pocket_metrics(wt_receptor_pdb, wt_pose_sdf) if wt_receptor_pdb.exists() and wt_pose_sdf.exists() else {
        "pocket_volume_proxy_a3": 0.0,
        "polar_exposed_fraction": 0.0,
    }

    target_scores: dict[str, Any] = {}
    mutant_scores: list[float] = []
    mutant_weights: list[float] = []
    mutant_keep_ifp_values: list[float] = []
    site_scores: list[float] = []
    site_weights: list[float] = []
    layer_mutant_values: dict[str, list[float]] = {name: [] for name in LAYER_NAMES}
    combo_scores: list[float] = []
    combo_weights: list[float] = []
    oracle_pred_stds: list[float] = []
    panel_rows = list(case_context["panel_rows"])
    panel_weights = dict(case_context.get("panel_weights") or {})
    panel_total_weight = float(sum(float(weight) for weight in panel_weights.values()))
    panel_success_weight = 0.0
    target_success_count = 0
    target_uncertain_count = 0
    max_workers = _target_parallel_workers(stage6, len(panel_rows))
    if max_workers == 1:
        target_results = [
            _dock_target_panel_row(
                root=root,
                row=row,
                candidate_sdf=candidate_sdf,
                case_context=case_context,
                stage6=stage6,
            )
            for row in panel_rows
        ]
    else:
        target_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _dock_target_panel_row,
                    root=root,
                    row=row,
                    candidate_sdf=candidate_sdf,
                    case_context=case_context,
                    stage6=stage6,
                )
                for row in panel_rows
            ]
            for future in as_completed(futures):
                target_results.append(future.result())
    row_lookup = {str(row["target_key"]): row for row in panel_rows}

    def _score_target_summary(
        *,
        target_key: str,
        effect_scope: str,
        row: dict[str, Any],
        target_summary: dict[str, Any],
    ) -> dict[str, Any] | None:
        if target_summary.get("docking_status") != "ok":
            return None
        target_positions_value = target_positions(target_key)
        target_ifp = json.loads((root / str(target_summary["ifp_json"])).read_text(encoding="utf-8"))
        features = _target_candidate_features(
            root=root,
            wt_summary=wt_summary,
            wt_ifp=wt_ifp,
            wt_affinity=wt_affinity,
            wt_pocket=wt_pocket,
            target_summary=target_summary,
            target_ifp=target_ifp,
            anchor_residues=anchor_residues,
            backbone_support_residues=backbone_support_residues,
            partner_chain_residues=partner_chain_residues,
            nonhotspot_residues=nonhotspot_residues,
            case_context=case_context,
            row=row,
            stage6=stage6,
        )
        keep_ifp_value = float(features["keep_ifp"])
        raw_docking_score = _binding_score(
            native_value(target_summary.get("best_affinity_kcal_mol")),
            native_value(case_context["lead_mt_affinities"].get(target_key)),
            scale,
        )
        legacy_calibrated_score = None if calibrator is None else calibrator.predict_score(features, scale)
        oracle_prediction = (
            oracle_v2.predict(features, scale)
            if oracle_v2 is not None and bool(oracle_v2.available)
            else {
                "available": False,
                "pred_mean": None,
                "pred_std": None,
                "pred_conservative": None,
                "score_mean": None,
                "score_conservative": None,
                "trust_score": native_value(scoring_policy.get("oracle_v2_effective_trust_score")),
                "member_predictions": {},
                "provenance": [],
            }
        )
        calibrated_score = native_value(oracle_prediction.get("score_conservative"))
        if calibrated_score is None:
            calibrated_score = native_value(legacy_calibrated_score)
        anchor_loss_inverse = float(max(0.0, 1.0 - float(features["anchor_loss_fraction"])))
        if bool(scoring_policy.get("uncertainty_heavy", False)):
            score_value = uncertainty_heavy_target_score(
                calibrated_score=native_value(calibrated_score),
                raw_docking_score=float(raw_docking_score),
                keep_ifp_value=float(keep_ifp_value),
                anchor_loss_inverse=float(anchor_loss_inverse),
                use_raw_docking_floor=bool(stage6.get("uncertainty_heavy_raw_docking_floor", True)),
            )
        elif calibrated_score is not None:
            score_value = float(calibrated_score)
        else:
            score_value = float(raw_docking_score)
        partner_chain_contact_labels = sorted(
            set(str(value) for value in list(target_ifp.get("residue_set") or []))
            & set(case_context.get("partner_chain_residues") or [])
        )
        return {
            "target_key": target_key,
            "effect_scope": effect_scope,
            "docking_status": "ok",
            "candidate_affinity_kcal_mol": native_value(target_summary.get("best_affinity_kcal_mol")),
            "lead_affinity_kcal_mol": native_value(case_context["lead_mt_affinities"].get(target_key)),
            "raw_docking_score": float(raw_docking_score),
            "calibrated_score": native_value(calibrated_score),
            "legacy_calibrated_score": native_value(legacy_calibrated_score),
            "oracle_v2_pred_mean_ddg": native_value(oracle_prediction.get("pred_mean")),
            "oracle_v2_pred_std_ddg": native_value(oracle_prediction.get("pred_std")),
            "oracle_v2_pred_conservative_ddg": native_value(oracle_prediction.get("pred_conservative")),
            "oracle_v2_score_mean": native_value(oracle_prediction.get("score_mean")),
            "oracle_v2_score_conservative": native_value(oracle_prediction.get("score_conservative")),
            "oracle_v2_trust_score": native_value(oracle_prediction.get("trust_score")),
            "oracle_v2_provenance": list(oracle_prediction.get("provenance") or []),
            "score": float(score_value),
            "keep_ifp": float(keep_ifp_value),
            **{name: float(features.get(name) or 0.0) for name in LAYER_NAMES},
            "lost_anchor_labels": list(features["lost_anchor_labels"]),
            "mechanism_labels": list(features["mechanism_labels"]),
            "target_uncertain": False,
            "is_partner_chain_sensitive": bool(
                set(target_positions_value) & set(case_context.get("partner_chain_positions") or [])
            )
            or bool(partner_chain_contact_labels),
            "mutation_positions": target_positions_value,
            "partner_chain_contact_labels": partner_chain_contact_labels,
            "partner_chain_contact_count": int(len(partner_chain_contact_labels)),
        }

    for target_key, effect_scope, target_summary in target_results:
        row = row_lookup[target_key]
        target_weight = float(panel_weights.get(target_key, 0.0))
        target_positions_value = target_positions(target_key)
        if list(target_summary.get("ensemble_members") or []):
            scored_members: list[dict[str, Any]] = []
            for member_index, member_summary in enumerate(list(target_summary.get("ensemble_members") or []), start=1):
                scored = _score_target_summary(
                    target_key=target_key,
                    effect_scope=effect_scope,
                    row=row,
                    target_summary=member_summary,
                )
                if scored is None:
                    continue
                scored.update(
                    {
                        "member_rank": int(member_summary.get("member_rank") or member_index),
                        "member_id": str(member_summary.get("member_id") or member_summary.get("sample_id") or f"member_{member_index}"),
                        "pdb_id": str(member_summary.get("pdb_id") or ""),
                    }
                )
                scored_members.append(scored)
            if not scored_members:
                target_uncertain_count += 1
                target_scores[target_key] = {
                    "target_key": target_key,
                    "effect_scope": effect_scope,
                    "docking_status": str(target_summary.get("docking_status") or "failed"),
                    "candidate_affinity_kcal_mol": None,
                    "lead_affinity_kcal_mol": native_value(case_context["lead_mt_affinities"].get(target_key)),
                    "score": None,
                    "raw_docking_score": None,
                    "calibrated_score": None,
                    "keep_ifp": None,
                    "lost_anchor_labels": [],
                    "mechanism_labels": [],
                    "target_uncertain": True,
                    "is_partner_chain_sensitive": bool(
                        set(target_positions_value) & set(case_context.get("partner_chain_positions") or [])
                    ),
                    "mutation_positions": target_positions_value,
                    "partner_chain_contact_labels": [],
                    "partner_chain_contact_count": 0,
                    "ensemble_member_count": int(target_summary.get("ensemble_member_count") or 0),
                    "ensemble_success_count": 0,
                    "ensemble_aggregate": str(target_summary.get("ensemble_aggregate") or stage6.get("receptor_ensemble_aggregate", "median")),
                }
                continue
            aggregate_mode = str(target_summary.get("ensemble_aggregate") or stage6.get("receptor_ensemble_aggregate", "median"))
            aggregate_score = aggregate_ensemble_value([float(member["score"]) for member in scored_members], aggregate_mode)
            representative = select_representative_member(scored_members, score_key="score", aggregate_value=aggregate_score) or dict(scored_members[0])
            aggregate_keep_ifp = aggregate_ensemble_value([float(member["keep_ifp"]) for member in scored_members], aggregate_mode)
            aggregate_affinity = aggregate_ensemble_value(
                [float(member["candidate_affinity_kcal_mol"]) for member in scored_members if member.get("candidate_affinity_kcal_mol") is not None],
                aggregate_mode,
            )
            aggregate_raw = aggregate_ensemble_value([float(member["raw_docking_score"]) for member in scored_members], aggregate_mode)
            aggregate_calibrated = aggregate_ensemble_value(
                [float(member["calibrated_score"]) for member in scored_members if member.get("calibrated_score") is not None],
                aggregate_mode,
            )
            aggregate_legacy = aggregate_ensemble_value(
                [float(member["legacy_calibrated_score"]) for member in scored_members if member.get("legacy_calibrated_score") is not None],
                aggregate_mode,
            )
            target_success_count += 1
            panel_success_weight += target_weight
            representative.update(
                {
                    "candidate_affinity_kcal_mol": native_value(aggregate_affinity),
                    "raw_docking_score": native_value(aggregate_raw),
                    "calibrated_score": native_value(aggregate_calibrated),
                    "legacy_calibrated_score": native_value(aggregate_legacy),
                    "score": native_value(aggregate_score),
                    "keep_ifp": native_value(aggregate_keep_ifp),
                    "ensemble_member_count": int(target_summary.get("ensemble_member_count") or len(scored_members)),
                    "ensemble_success_count": int(len(scored_members)),
                    "ensemble_aggregate": aggregate_mode,
                    "ensemble_score_std": float(pd.Series([member["score"] for member in scored_members], dtype=float).std(ddof=0))
                    if len(scored_members) > 1
                    else 0.0,
                    "ensemble_conformer_ids": [str(member.get("member_id") or "") for member in scored_members],
                    "ensemble_conformer_pdb_ids": [str(member.get("pdb_id") or "") for member in scored_members if str(member.get("pdb_id") or "")],
                }
            )
            target_scores[target_key] = representative
            mutant_scores.append(float(aggregate_score or 0.0))
            mutant_weights.append(target_weight)
            mutant_keep_ifp_values.append(float(aggregate_keep_ifp or 0.0))
            for layer_name in LAYER_NAMES:
                layer_value = aggregate_ensemble_value(
                    [
                        float(member.get(layer_name) or 0.0)
                        for member in scored_members
                    ],
                    aggregate_mode,
                )
                layer_mutant_values[layer_name].append(float(layer_value or 0.0))
            pred_std_aggregate = aggregate_ensemble_value(
                [
                    float(member.get("oracle_v2_pred_std_ddg") or 0.0)
                    for member in scored_members
                    if member.get("oracle_v2_pred_std_ddg") is not None
                ],
                aggregate_mode,
            )
            if pred_std_aggregate is not None:
                oracle_pred_stds.append(float(pred_std_aggregate))
            if effect_scope == "site":
                site_scores.append(float(aggregate_score or 0.0))
                site_weights.append(target_weight)
            if effect_scope == "combo":
                combo_scores.append(float(aggregate_score or 0.0))
                combo_weights.append(target_weight)
            continue

        scored_target = _score_target_summary(
            target_key=target_key,
            effect_scope=effect_scope,
            row=row,
            target_summary=target_summary,
        )
        if scored_target is None:
            target_uncertain_count += 1
            target_scores[target_key] = {
                "target_key": target_key,
                "effect_scope": effect_scope,
                "docking_status": str(target_summary.get("docking_status") or "failed"),
                "candidate_affinity_kcal_mol": None,
                "lead_affinity_kcal_mol": native_value(case_context["lead_mt_affinities"].get(target_key)),
                "score": None,
                "raw_docking_score": None,
                "calibrated_score": None,
                "keep_ifp": None,
                "lost_anchor_labels": [],
                "mechanism_labels": [],
                "target_uncertain": True,
                "is_partner_chain_sensitive": bool(
                    set(target_positions_value) & set(case_context.get("partner_chain_positions") or [])
                ),
                "mutation_positions": target_positions_value,
                "partner_chain_contact_labels": [],
                "partner_chain_contact_count": 0,
            }
            continue
        target_success_count += 1
        panel_success_weight += target_weight
        target_scores[target_key] = scored_target
        mutant_scores.append(float(scored_target["score"]))
        mutant_weights.append(target_weight)
        mutant_keep_ifp_values.append(float(scored_target["keep_ifp"]))
        for layer_name in LAYER_NAMES:
            layer_mutant_values[layer_name].append(float(scored_target.get(layer_name) or 0.0))
        pred_std_value = native_value(scored_target.get("oracle_v2_pred_std_ddg"))
        if pred_std_value is not None:
            oracle_pred_stds.append(float(pred_std_value))
        if effect_scope == "site":
            site_scores.append(float(scored_target["score"]))
            site_weights.append(target_weight)
        if effect_scope == "combo":
            combo_scores.append(float(scored_target["score"]))
            combo_weights.append(target_weight)

    s_wt = _binding_score(
        wt_affinity,
        lead_wt_affinity_value,
        scale,
    )
    mutant_keep_ifp = _weighted_mean(mutant_keep_ifp_values, mutant_weights)
    mutant_layer_scores = {
        name: _weighted_mean(layer_mutant_values[name], mutant_weights) if layer_mutant_values[name] else 0.0
        for name in LAYER_NAMES
    }
    keep_ifp_layer_scores = {
        name: float((float(wt_layer_scores.get(name, 0.0)) + float(mutant_layer_scores.get(name, 0.0))) / 2.0)
        for name in LAYER_NAMES
    }
    keep_ifp = float(weighted_keep_ifp(keep_ifp_layer_scores, layer_weights))
    robust_core = weighted_cvar(mutant_scores, mutant_weights, float(stage6.get("cvar_tail_fraction", 0.2))) if mutant_scores else float(s_wt)
    robust_score = float(min(s_wt, robust_core))
    robust_site_core = weighted_cvar(site_scores, site_weights, float(stage6.get("cvar_tail_fraction", 0.2))) if site_scores else float(robust_core)
    naive_mean = float((s_wt + _weighted_mean(mutant_scores, mutant_weights)) / 2.0) if mutant_scores else float(s_wt)
    combo_robust_core = weighted_cvar(combo_scores, combo_weights, float(stage6.get("cvar_tail_fraction", 0.2))) if combo_scores else None
    combo_naive_mean = _weighted_mean(combo_scores, combo_weights) if combo_scores else None

    epsilon = float(stage6.get("wt_hard_constraint_epsilon_kcal_mol", 0.5))
    wt_hard_constraint_pass = (
        wt_affinity is not None and lead_wt_affinity_value is not None and float(wt_affinity) <= float(lead_wt_affinity_value) + epsilon
    )
    keep_ifp_constraint_pass = bool(keep_ifp >= float(stage6.get("keep_ifp_min", 0.5)) or key_anchor_count >= int(stage6.get("min_anchor_count", 1)))
    hotspot_contact_delta = float(candidate_hotspot_score - lead_hotspot_score)
    compensation_constraint_pass = bool(hotspot_contact_delta >= 0.0 or compensation_gain > 0.0)
    if panel_total_weight <= 0.0:
        panel_coverage = 1.0
    else:
        panel_coverage = float(panel_success_weight / panel_total_weight)
    coverage_pass = bool(panel_coverage >= float(stage6.get("panel_min_coverage", 0.80)))
    high_uncertainty = bool(panel_coverage < float(stage6.get("high_uncertainty_coverage", 0.95)))
    candidate_valid = bool(wt_hard_constraint_pass and coverage_pass)

    delta = float(stage6.get("reward_weights", {}).get("delta", 0.3))
    objective_guardrails = dict(case_context.get("objective_guardrails") or {})
    guardrail_adjustments = objective_guardrail_adjustments(
        objective_name=str(objective_name),
        robust_score=float(robust_score),
        s_wt=float(s_wt),
        compensation_gain=float(compensation_gain),
        objective_guardrails=objective_guardrails,
    )
    effective_compensation_gain = float(guardrail_adjustments["effective_compensation_gain"])
    oracle_uncertainty_terms = oracle_uncertainty_score(
        effective_trust_score=native_value(scoring_policy.get("oracle_v2_effective_trust_score")),
        pred_std_mean=(sum(oracle_pred_stds) / len(oracle_pred_stds)) if oracle_pred_stds else None,
        stage6=stage6,
    )
    alt_anchor_terms = alt_anchor_score(
        layer_scores=keep_ifp_layer_scores,
        new_nonhotspot_score=float(new_nonhotspot_metrics["new_nonhotspot_contact_score"]),
        compensation_gain=float(effective_compensation_gain),
        hotspot_fraction=float(hotspot_fraction_value),
        stage6=stage6,
    )
    reward_v2_terms = reward_v2_components(
        s_wt=float(s_wt),
        robust_site_core=float(robust_site_core),
        robust_combo_core=native_value(combo_robust_core),
        combo_dense_case=bool(str(case_context.get("evaluation_unit") or "") == "observed_combo"),
        alt_anchor_score_value=float(alt_anchor_terms["alt_anchor_score"]),
        new_nonhotspot_score=float(alt_anchor_terms["new_nonhotspot_score"]),
        hotspot_fraction=float(hotspot_fraction_value),
        oracle_uncertainty=float(oracle_uncertainty_terms["oracle_uncertainty"]),
        synth_penalty=float(prefilter.get("total_penalty") or 0.0),
    )
    robust_reward_raw = float(reward_v2_terms["reward_v2_raw"])
    naive_reward_raw = naive_mean - delta * float(prefilter.get("total_penalty") or 0.0)
    coverage_penalty = float(stage6.get("coverage_penalty", 0.25)) * float(max(0.0, 1.0 - panel_coverage))
    constraint_penalty = 0.0
    if not keep_ifp_constraint_pass:
        constraint_penalty += float(stage6.get("keep_ifp_penalty", 0.20))
    if not compensation_constraint_pass:
        constraint_penalty += float(stage6.get("compensation_penalty", 0.10))
    hotspot_penalty = float(stage6.get("hotspot_drop_penalty", 0.15)) * float(hotspot_drop)
    robust_reward = (
        robust_reward_raw
        - coverage_penalty
        - constraint_penalty
        - hotspot_penalty
        - float(guardrail_adjustments["objective_guardrail_penalty"])
    )
    naive_reward = naive_reward_raw - coverage_penalty - constraint_penalty
    if not candidate_valid:
        robust_reward = min(robust_reward, 0.0)
        naive_reward = min(naive_reward, 0.0)
    objective_reward = robust_reward if str(objective_name) == "robust" else naive_reward

    mechanism_focus = "balanced"
    if not wt_hard_constraint_pass:
        mechanism_focus = "wt_constraint_failed"
    elif not coverage_pass:
        mechanism_focus = "coverage_failed"
    elif bool(guardrail_adjustments["objective_guardrail_floor_violation"]):
        mechanism_focus = "seed_floor_failed"
    elif dep_value > 0.7:
        mechanism_focus = "hotspot_dependent"
    elif compensation_gain > 0.0:
        mechanism_focus = "nonhotspot_compensation"

    return {
        **base_row,
        "chemical_valid": True,
        "docking_success": bool(target_success_count > 0),
        "wt_docking_status": str(wt_summary.get("docking_status") or ""),
        "wt_affinity_kcal_mol": wt_affinity,
        "lead_wt_affinity_kcal_mol": lead_wt_affinity_value,
        "s_wt": float(s_wt),
        "wt_keep_ifp": float(wt_keep_ifp),
        **{f"wt_{name}": float(wt_layer_scores.get(name, 0.0)) for name in LAYER_NAMES},
        "key_anchor_count": int(key_anchor_count),
        "hotspot_contact_score": float(candidate_hotspot_score),
        "nonhotspot_contact_score": float(candidate_nonhotspot_score),
        "hotspot_fraction": float(hotspot_fraction_value),
        "dep": float(dep_value),
        "hotspot_drop": float(hotspot_drop),
        "new_nonhotspot_contact_score": float(new_nonhotspot_metrics["new_nonhotspot_contact_score"]),
        "new_nonhotspot_residue_count": int(new_nonhotspot_metrics["new_nonhotspot_residue_count"]),
        "compensation_gain": float(compensation_gain),
        "effective_compensation_gain": float(effective_compensation_gain),
        "keep_ifp": float(keep_ifp),
        **{name: float(keep_ifp_layer_scores.get(name, 0.0)) for name in LAYER_NAMES},
        **{f"reward_v2_weight_{name}": float(layer_weights.get(name, 0.0)) for name in LAYER_NAMES},
        "panel_coverage": float(panel_coverage),
        "coverage_pass": bool(coverage_pass),
        "high_uncertainty": bool(high_uncertainty),
        "target_success_count": int(target_success_count),
        "target_uncertain_count": int(target_uncertain_count),
        "robust_core": float(robust_core),
        "robust_site_core": float(robust_site_core),
        "robust_score": float(robust_score),
        "naive_mean_affinity": float(naive_mean),
        "combo_robust_core": native_value(combo_robust_core),
        "combo_naive_mean_affinity": native_value(combo_naive_mean),
        "oracle_uncertainty": float(oracle_uncertainty_terms["oracle_uncertainty"]),
        "oracle_uncertainty_trust_penalty": float(oracle_uncertainty_terms["oracle_uncertainty_trust_penalty"]),
        "oracle_uncertainty_pred_std_penalty": float(oracle_uncertainty_terms["oracle_uncertainty_pred_std_penalty"]),
        "oracle_pred_std_mean": None if not oracle_pred_stds else float(sum(oracle_pred_stds) / len(oracle_pred_stds)),
        "alt_anchor_score": float(alt_anchor_terms["alt_anchor_score"]),
        "new_nonhotspot_score": float(alt_anchor_terms["new_nonhotspot_score"]),
        "compensation_gain_score": float(alt_anchor_terms["compensation_gain_score"]),
        **reward_v2_terms,
        "coverage_penalty": float(coverage_penalty),
        "constraint_penalty": float(constraint_penalty + hotspot_penalty),
        "objective_guardrail_active": bool(guardrail_adjustments["objective_guardrail_active"]),
        "objective_guardrail_floor_violation": bool(guardrail_adjustments["objective_guardrail_floor_violation"]),
        "objective_guardrail_penalty": float(guardrail_adjustments["objective_guardrail_penalty"]),
        "objective_guardrail_robust_score_floor": native_value(guardrail_adjustments["objective_guardrail_robust_score_floor"]),
        "objective_guardrail_s_wt_floor": native_value(guardrail_adjustments["objective_guardrail_s_wt_floor"]),
        "objective_guardrail_robust_score_gap": float(guardrail_adjustments["objective_guardrail_robust_score_gap"]),
        "objective_guardrail_s_wt_gap": float(guardrail_adjustments["objective_guardrail_s_wt_gap"]),
        "objective_guardrail_robust_score_penalty_weight": native_value(objective_guardrails.get("robust_score_penalty_weight")),
        "objective_guardrail_s_wt_penalty_weight": native_value(objective_guardrails.get("s_wt_penalty_weight")),
        "objective_guardrail_disable_compensation_below_floor": bool(objective_guardrails.get("disable_compensation_below_floor", False)),
        "robust_objective_reward": float(robust_reward),
        "naive_objective_reward": float(naive_reward),
        "objective_reward": float(objective_reward),
        "wt_hard_constraint_pass": bool(wt_hard_constraint_pass),
        "wt_pass": bool(wt_hard_constraint_pass),
        "keep_ifp_constraint_pass": bool(keep_ifp_constraint_pass),
        "compensation_constraint_pass": bool(compensation_constraint_pass),
        "panel_coverage_pass": bool(coverage_pass),
        "panel_passing": bool(candidate_valid),
        "candidate_valid": bool(candidate_valid),
        "mechanism_risk_focus": mechanism_focus,
        "target_scores_json": json.dumps(target_scores, ensure_ascii=True, sort_keys=True),
    }


def evaluate_candidate_job(job: dict[str, Any]) -> dict[str, Any]:
    return evaluate_candidate(
        root=Path(str(job["root"])),
        case_context=dict(job["case_context"]),
        stage6=dict(job["stage6"]),
        smiles=str(job["smiles"]),
        action_sequence=list(job.get("action_sequence") or []),
        round_index=int(job["round_index"]),
        objective_name=str(job["objective_name"]),
    )


def recompute_cached_candidate_scores(
    row: dict[str, Any] | pd.Series,
    *,
    case_context: dict[str, Any],
    stage6: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(row) if isinstance(row, dict) else dict(row.to_dict())
    target_scores = parse_json_payload(payload.get("target_scores_json")) or {}
    if not isinstance(target_scores, dict):
        target_scores = {}
    scoring_policy = dict(case_context.get("scoring_policy") or {})
    anchor_count = max(1, len(case_context.get("anchor_residues") or []))
    panel_weights = dict(case_context.get("panel_weights") or {})
    mutant_scores: list[float] = []
    mutant_weights: list[float] = []
    site_scores: list[float] = []
    site_weights: list[float] = []
    layer_mutant_values: dict[str, list[float]] = {name: [] for name in LAYER_NAMES}
    combo_scores: list[float] = []
    combo_weights: list[float] = []
    oracle_pred_stds: list[float] = []
    for target_key, item in target_scores.items():
        if not isinstance(item, dict) or bool(item.get("target_uncertain", False)):
            continue
        raw_docking_score = native_value(item.get("raw_docking_score"))
        calibrated_score = native_value(item.get("calibrated_score"))
        keep_ifp_value = float(native_value(item.get("keep_ifp")) or 0.0)
        lost_anchor_count = int(len(item.get("lost_anchor_labels") or []))
        anchor_loss_inverse = float(max(0.0, 1.0 - lost_anchor_count / anchor_count))
        if bool(scoring_policy.get("uncertainty_heavy", False)):
            score_value = uncertainty_heavy_target_score(
                calibrated_score=calibrated_score,
                raw_docking_score=raw_docking_score,
                keep_ifp_value=keep_ifp_value,
                anchor_loss_inverse=anchor_loss_inverse,
                use_raw_docking_floor=bool(stage6.get("uncertainty_heavy_raw_docking_floor", True)),
            )
        elif calibrated_score is not None:
            score_value = float(calibrated_score)
        elif raw_docking_score is not None:
            score_value = float(raw_docking_score)
        else:
            continue
        item["score"] = float(score_value)
        weight = float(panel_weights.get(str(target_key), 0.0))
        mutant_scores.append(float(score_value))
        mutant_weights.append(weight)
        for layer_name in LAYER_NAMES:
            layer_mutant_values[layer_name].append(float(native_value(item.get(layer_name)) or 0.0))
        pred_std_value = native_value(item.get("oracle_v2_pred_std_ddg"))
        if pred_std_value is not None:
            oracle_pred_stds.append(float(pred_std_value))
        if str(item.get("effect_scope") or "") == "site":
            site_scores.append(float(score_value))
            site_weights.append(weight)
        if str(item.get("effect_scope") or "") == "combo":
            combo_scores.append(float(score_value))
            combo_weights.append(weight)
    s_wt = float(native_value(payload.get("s_wt")) or 0.0)
    robust_core = weighted_cvar(mutant_scores, mutant_weights, float(stage6.get("cvar_tail_fraction", 0.2))) if mutant_scores else s_wt
    robust_score = float(min(s_wt, robust_core))
    robust_site_core = weighted_cvar(site_scores, site_weights, float(stage6.get("cvar_tail_fraction", 0.2))) if site_scores else float(robust_core)
    naive_mean = float((s_wt + _weighted_mean(mutant_scores, mutant_weights)) / 2.0) if mutant_scores else s_wt
    combo_robust_core = weighted_cvar(combo_scores, combo_weights, float(stage6.get("cvar_tail_fraction", 0.2))) if combo_scores else None
    combo_naive_mean = _weighted_mean(combo_scores, combo_weights) if combo_scores else None

    delta = float(stage6.get("reward_weights", {}).get("delta", 0.3))
    coverage_penalty = float(stage6.get("coverage_penalty", 0.25)) * float(max(0.0, 1.0 - float(native_value(payload.get("panel_coverage")) or 0.0)))
    constraint_penalty = 0.0
    if not bool(payload.get("keep_ifp_constraint_pass", False)):
        constraint_penalty += float(stage6.get("keep_ifp_penalty", 0.20))
    if not bool(payload.get("compensation_constraint_pass", False)):
        constraint_penalty += float(stage6.get("compensation_penalty", 0.10))
    hotspot_penalty = float(stage6.get("hotspot_drop_penalty", 0.15)) * float(native_value(payload.get("hotspot_drop")) or 0.0)
    compensation_gain = float(native_value(payload.get("compensation_gain")) or 0.0)
    total_penalty = float(
        native_value(payload.get("total_penalty"))
        or native_value(payload.get("admet_penalty"))
        or 0.0
    )
    layer_available = {
        name: bool(case_context.get("partner_chain_residues") or []) if name == "keep_ifp_partner_chain" else True
        for name in LAYER_NAMES
    }
    layer_weights = reward_v2_layer_weights(
        case_id=str(case_context.get("case_id") or payload.get("case_id") or ""),
        target_domain=str(case_context.get("target_domain") or ""),
        stage6=stage6,
        layer_available=layer_available,
    )
    wt_layer_scores = {
        name: float(native_value(payload.get(f"wt_{name}")) or 0.0)
        for name in LAYER_NAMES
    }
    mutant_layer_scores = {
        name: _weighted_mean(layer_mutant_values[name], mutant_weights) if layer_mutant_values[name] else float(native_value(payload.get(name)) or 0.0)
        for name in LAYER_NAMES
    }
    keep_ifp_layer_scores = {
        name: float((float(wt_layer_scores.get(name, 0.0)) + float(mutant_layer_scores.get(name, 0.0))) / 2.0)
        for name in LAYER_NAMES
    }
    keep_ifp_value = float(weighted_keep_ifp(keep_ifp_layer_scores, layer_weights))
    objective_guardrails = objective_guardrails_from_payload(payload, case_context=case_context)
    guardrail_adjustments = objective_guardrail_adjustments(
        objective_name=str(payload.get("objective_name") or ""),
        robust_score=float(robust_score),
        s_wt=float(s_wt),
        compensation_gain=float(compensation_gain),
        objective_guardrails=objective_guardrails,
    )
    effective_compensation_gain = float(guardrail_adjustments["effective_compensation_gain"])
    oracle_uncertainty_terms = oracle_uncertainty_score(
        effective_trust_score=native_value(payload.get("oracle_v2_effective_trust_score") or case_context.get("oracle_v2_effective_trust_score")),
        pred_std_mean=(sum(oracle_pred_stds) / len(oracle_pred_stds)) if oracle_pred_stds else native_value(payload.get("oracle_pred_std_mean")),
        stage6=stage6,
    )
    alt_anchor_terms = alt_anchor_score(
        layer_scores=keep_ifp_layer_scores,
        new_nonhotspot_score=float(native_value(payload.get("new_nonhotspot_contact_score")) or 0.0),
        compensation_gain=float(effective_compensation_gain),
        hotspot_fraction=float(native_value(payload.get("hotspot_fraction")) or 0.0),
        stage6=stage6,
    )
    reward_v2_terms = reward_v2_components(
        s_wt=float(s_wt),
        robust_site_core=float(robust_site_core),
        robust_combo_core=native_value(combo_robust_core),
        combo_dense_case=bool(str(case_context.get("evaluation_unit") or "") == "observed_combo"),
        alt_anchor_score_value=float(alt_anchor_terms["alt_anchor_score"]),
        new_nonhotspot_score=float(alt_anchor_terms["new_nonhotspot_score"]),
        hotspot_fraction=float(native_value(payload.get("hotspot_fraction")) or 0.0),
        oracle_uncertainty=float(oracle_uncertainty_terms["oracle_uncertainty"]),
        synth_penalty=float(total_penalty),
    )
    robust_reward = float(
        reward_v2_terms["reward_v2_raw"]
        - coverage_penalty
        - constraint_penalty
        - hotspot_penalty
        - float(guardrail_adjustments["objective_guardrail_penalty"])
    )
    naive_reward = float(naive_mean - delta * total_penalty - coverage_penalty - constraint_penalty)
    if not bool(payload.get("candidate_valid", False)):
        robust_reward = min(robust_reward, 0.0)
        naive_reward = min(naive_reward, 0.0)
    payload.update(
        {
            "target_scores_json": json.dumps(target_scores, ensure_ascii=True, sort_keys=True),
            "robust_core": float(robust_core),
            "robust_site_core": float(robust_site_core),
            "robust_score": float(robust_score),
            "naive_mean_affinity": float(naive_mean),
            "combo_robust_core": native_value(combo_robust_core),
            "combo_naive_mean_affinity": native_value(combo_naive_mean),
            "effective_compensation_gain": float(effective_compensation_gain),
            "keep_ifp": float(keep_ifp_value),
            **{name: float(keep_ifp_layer_scores.get(name, 0.0)) for name in LAYER_NAMES},
            "oracle_uncertainty": float(oracle_uncertainty_terms["oracle_uncertainty"]),
            "oracle_uncertainty_trust_penalty": float(oracle_uncertainty_terms["oracle_uncertainty_trust_penalty"]),
            "oracle_uncertainty_pred_std_penalty": float(oracle_uncertainty_terms["oracle_uncertainty_pred_std_penalty"]),
            "oracle_pred_std_mean": None if not oracle_pred_stds else float(sum(oracle_pred_stds) / len(oracle_pred_stds)),
            "alt_anchor_score": float(alt_anchor_terms["alt_anchor_score"]),
            "new_nonhotspot_score": float(alt_anchor_terms["new_nonhotspot_score"]),
            "compensation_gain_score": float(alt_anchor_terms["compensation_gain_score"]),
            **reward_v2_terms,
            "objective_guardrail_active": bool(guardrail_adjustments["objective_guardrail_active"]),
            "objective_guardrail_floor_violation": bool(guardrail_adjustments["objective_guardrail_floor_violation"]),
            "objective_guardrail_penalty": float(guardrail_adjustments["objective_guardrail_penalty"]),
            "objective_guardrail_robust_score_floor": native_value(guardrail_adjustments["objective_guardrail_robust_score_floor"]),
            "objective_guardrail_s_wt_floor": native_value(guardrail_adjustments["objective_guardrail_s_wt_floor"]),
            "objective_guardrail_robust_score_gap": float(guardrail_adjustments["objective_guardrail_robust_score_gap"]),
            "objective_guardrail_s_wt_gap": float(guardrail_adjustments["objective_guardrail_s_wt_gap"]),
            "objective_guardrail_robust_score_penalty_weight": native_value(objective_guardrails.get("robust_score_penalty_weight")),
            "objective_guardrail_s_wt_penalty_weight": native_value(objective_guardrails.get("s_wt_penalty_weight")),
            "objective_guardrail_disable_compensation_below_floor": bool(objective_guardrails.get("disable_compensation_below_floor", False)),
            "robust_objective_reward": float(robust_reward),
            "naive_objective_reward": float(naive_reward),
            "objective_reward": float(robust_reward if str(payload.get("objective_name") or "") == "robust" else naive_reward),
        }
    )
    return payload


def recompute_cached_leaderboard_scores(
    frame: pd.DataFrame,
    *,
    case_context: dict[str, Any],
    stage6: dict[str, Any],
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rows = [recompute_cached_candidate_scores(row, case_context=case_context, stage6=stage6) for row in frame.to_dict(orient="records")]
    return pd.DataFrame.from_records(rows)


def top_scaffold_diversity(frame: pd.DataFrame, top_n: int) -> int:
    if frame.empty or top_n <= 0:
        return 0
    subset = frame.head(top_n)
    return int(subset["scaffold_smiles"].fillna("").replace("", pd.NA).dropna().nunique())


def objective_ablation_rows(frame: pd.DataFrame, lead_candidate_id: str, stage6: dict[str, Any], combo_dense: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    top20_n = int(stage6.get("top20_report_n", 20))
    top50_n = int(stage6.get("scaffold_diversity_top_n", 50))
    for objective_name, group in frame.groupby("objective_name"):
        ranked = group.sort_values("objective_reward", ascending=False).reset_index(drop=True)
        top20 = ranked.head(top20_n)
        lead_row = ranked[ranked["candidate_id"].eq(lead_candidate_id)].head(1)
        lead_robust = float(lead_row.iloc[0]["robust_score"]) if not lead_row.empty else 0.0
        lead_dep = float(lead_row.iloc[0]["dep"]) if not lead_row.empty else 1.0
        robust_median = float(top20["robust_score"].median()) if not top20.empty else 0.0
        robust_core_median = float(top20.get("robust_core", pd.Series(dtype=float)).median()) if not top20.empty else 0.0
        dep_median = float(top20["dep"].median()) if not top20.empty else 1.0
        keep_ifp_median = float(top20.get("keep_ifp", pd.Series(dtype=float)).median()) if not top20.empty else 0.0
        dynamics_lite_median = (
            float(top20.get("dynamics_lite_score", pd.Series(dtype=float)).dropna().median())
            if not top20.empty and "dynamics_lite_score" in top20.columns and not top20.get("dynamics_lite_score", pd.Series(dtype=float)).dropna().empty
            else None
        )
        total_penalty_median = (
            float(top20.get("total_penalty", pd.Series(dtype=float)).dropna().median())
            if not top20.empty and "total_penalty" in top20.columns and not top20.get("total_penalty", pd.Series(dtype=float)).dropna().empty
            else None
        )
        candidate_valid = ranked.get("candidate_valid", pd.Series(False, index=ranked.index))
        wt_pass = top20.get("wt_hard_constraint_pass", pd.Series(False, index=top20.index))
        wt_pass_rate = float(wt_pass.fillna(False).astype(bool).mean()) if not top20.empty else 0.0
        top20_valid_rate = float(top20.get("candidate_valid", pd.Series(False, index=top20.index)).fillna(False).astype(bool).mean()) if not top20.empty else 0.0
        executability = stage6_executability_metrics(ranked)
        combo_core_median = None
        if combo_dense and "combo_robust_core" in top20.columns:
            combo_values = top20["combo_robust_core"].dropna()
            combo_core_median = None if combo_values.empty else float(combo_values.median())
        rows.append(
            {
                "objective_name": str(objective_name),
                "candidate_count": int(len(ranked)),
                "valid_candidate_rate": float(candidate_valid.fillna(False).astype(bool).mean()) if not ranked.empty else 0.0,
                "chemical_valid_rate": float(executability["chemical_valid_rate"]),
                "prefilter_pass_rate": float(executability["prefilter_pass_rate"]),
                "wt_pass_rate": float(executability["wt_pass_rate"]),
                "panel_coverage_pass_rate": float(executability["panel_coverage_pass_rate"]),
                "panel_passing_rate": float(executability["panel_passing_rate"]),
                "top20_valid_candidate_rate": top20_valid_rate,
                "top20_wt_constraint_pass_rate": wt_pass_rate,
                "top20_robust_score_median": robust_median,
                "top20_robust_core_median": robust_core_median,
                "top20_keep_ifp_median": keep_ifp_median,
                "top20_dynamics_lite_score_median": dynamics_lite_median,
                "top20_total_penalty_median": total_penalty_median,
                "lead_robust_score": lead_robust,
                "top20_robust_score_gain_fraction": None if lead_robust == 0.0 else float(robust_median / lead_robust - 1.0),
                "top20_dep_median": dep_median,
                "lead_dep": lead_dep,
                "top20_dep_reduction_fraction": None if lead_dep == 0.0 else float(1.0 - dep_median / lead_dep),
                "top50_scaffold_unique": int(top_scaffold_diversity(ranked, top50_n)),
                "best_candidate_id": str(ranked.iloc[0]["candidate_id"]),
                "best_objective_reward": float(ranked.iloc[0]["objective_reward"]),
                "combo_dense_case": bool(combo_dense),
                "topcombo20_robust_core_median": combo_core_median,
            }
        )
    return rows


def write_top_sdf(frame: pd.DataFrame, output_sdf: Path, top_k: int) -> None:
    ensure_dir(output_sdf.parent)
    writer = Chem.SDWriter(str(output_sdf))
    for _, row in frame.head(top_k).iterrows():
        molecule = Chem.MolFromSmiles(str(row["smiles"]))
        if molecule is None:
            continue
        molecule = Chem.AddHs(molecule)
        molecule = ensure_ligand_3d(molecule)
        for field in [
            "candidate_id",
            "objective_name",
            "objective_reward",
            "robust_score",
            "naive_mean_affinity",
            "keep_ifp",
            "dep",
            "compensation_gain",
        ]:
            molecule.SetProp(str(field), str(native_value(row.get(field))))
        writer.write(molecule)
    writer.close()


def render_sar_rules(case_entry: dict[str, Any], frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# counter-design step SAR Rules\n\nNo valid candidate survived counter-design step search.\n"
    top = frame.head(10).copy()
    common_patterns = top["action_sequence_json"].fillna("[]").tolist()
    lines = [
        f"# {case_entry['case_id']} counter-design step Robust SAR Rules",
        "",
        f"- 目标：`{case_entry['target_name']}`",
        f"- 药物：`{case_entry['drug_name']}`",
        f"- Top robust 候选数：`{len(frame)}`",
        "",
        "## Top Findings",
        f"- 最优候选：`{top.iloc[0]['candidate_id']}`，reward=`{native_value(top.iloc[0]['objective_reward'])}`，`Dep={native_value(top.iloc[0]['dep'])}`，`KeepIFP={native_value(top.iloc[0]['keep_ifp'])}`",
        f"- Top10 `Dep` 中位数：`{native_value(top['dep'].median())}`",
        f"- Top10 `RobustScore` 中位数：`{native_value(top['robust_score'].median())}`",
        "",
        "## Suggested SAR Rules",
        "- 优先保留或恢复 anchor-compatible 极性相互作用，避免直接削弱 WT 锚点。",
        "- 若热点接触下降，必须同时引入非热点补偿接触，否则 reward 不成立。",
        "- 结构上优先选择紧凑的 peripheral edit，避免用更长更柔性的尾部换平均亲和力。",
        "- 对芳环优先尝试 aza-scan / 小型极性取代，而不是大体积扩张。",
        "",
        "## Action Trace Samples",
    ]
    for candidate_id, action_json in zip(top["candidate_id"].tolist()[:5], common_patterns[:5]):
        lines.append(f"- `{candidate_id}`: `{action_json}`")
    lines.append("")
    return "\n".join(lines)
