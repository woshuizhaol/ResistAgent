#!/usr/bin/env python3
"""counter-design step unit tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Geometry import Point3D

from agents.counter_design_agent import CounterDesignAgent, CounterDesignAgentConfig
from tools.filters import apply_prefilters
from tools.runtime import load_yaml
from tools import stage6_utils
from tools import stage35_utils
from tools.stage35_utils import nearby_protein_residue_labels, pose_pdb_from_sdf, standardize_reference_ligand
from tools.stage6_utils import (
    _wt_receptor_pdb,
    _target_panel_receptor_pdb,
    _target_box,
    _oracle_v2_scoring_policy,
    _dock_candidate,
    apply_action_to_molecule,
    build_transform_prior,
    canonical_smiles,
    case_specific_action_hints,
    dep_score,
    enumerate_candidate_actions,
    murcko_scaffold,
    new_nonhotspot_contact_metrics,
    objective_ablation_rows,
    objective_guardrail_adjustments,
    pocket_profile,
    recompute_cached_candidate_scores,
    uncertainty_heavy_target_score,
    weighted_cvar,
    weighted_ifp_cosine,
)
from tools.stage6_calibrator import fit_case_calibrator, load_case_calibrator
from tools.stage6_oracle_v2 import conservative_ensemble_holdout, fit_stage6_oracle_v2, load_stage6_oracle_v2
from tools.stage6_postmortem import llm_failure_context_audit_frame, stage6_executability_metrics
from tools.stage6_receptor_ensemble import aggregate_ensemble_value, build_receptor_ensemble_members
from tools.stage6_reward_v2 import alt_anchor_score, oracle_uncertainty_score, reward_v2_components, reward_v2_layer_weights

SCRIPT_09_PATH = Path(__file__).resolve().parents[1] / "scripts" / "09_counter_design.py"
SCRIPT_09_SPEC = importlib.util.spec_from_file_location("stage6_counter_design", SCRIPT_09_PATH)
assert SCRIPT_09_SPEC is not None and SCRIPT_09_SPEC.loader is not None
SCRIPT_09_MODULE = importlib.util.module_from_spec(SCRIPT_09_SPEC)
SCRIPT_09_SPEC.loader.exec_module(SCRIPT_09_MODULE)


class DummyClient:
    def chat_json(self, *args, **kwargs):  # pragma: no cover - fallback path should bypass this
        raise AssertionError("Dummy client should not be called when disable_llm=True")


def stage6_config() -> dict:
    return {
        "prefilter": {
            "qed_min": 0.35,
            "sa_max": 6.5,
            "ra_score_min": 0.30,
            "scscore_max": 4.5,
            "retrosynthesis_plausibility_min": 0.30,
            "series_likeness_min": 0.25,
            "medchem_blacklist_hard_fail": True,
            "unstable_motif_hard_fail": False,
            "clogp_min": -0.5,
            "clogp_max": 5.5,
            "hbd_max": 6,
            "hba_max": 12,
            "tpsa_max": 160.0,
            "mw_max": 650.0,
            "rotatable_bonds_max": 14,
            "baseline_tolerance": {
                "qed_drop": 0.03,
                "sa_increase": 0.5,
                "clogp_decrease": 0.4,
                "clogp_increase": 0.4,
                "hbd_increase": 1,
                "hba_increase": 1,
                "tpsa_increase": 15.0,
                "mw_increase": 60.0,
                "rotatable_bonds_increase": 2,
            },
        },
        "fragment_library": {
            "add": ["F", "Cl", "OC", "N"],
            "replace": ["F", "OC", "N"],
        },
        "ifp_type_weights": {
            "metal_complex": 1.4,
            "salt_bridge": 1.3,
            "hydrogen_bond": 1.0,
            "pi_cation": 0.9,
            "pi_stacking": 0.8,
            "hydrophobic": 0.6,
        },
        "proposal_count": 8,
        "top20_report_n": 20,
        "scaffold_diversity_top_n": 50,
        "panel_min_coverage": 0.8,
        "high_uncertainty_coverage": 0.95,
        "coverage_penalty": 0.25,
        "keep_ifp_penalty": 0.20,
        "compensation_penalty": 0.10,
        "hotspot_drop_penalty": 0.15,
        "uncertainty_heavy_raw_docking_floor": True,
        "keep_ifp_min": 0.5,
        "min_anchor_count": 1,
        "wt_hard_constraint_epsilon_kcal_mol": 0.5,
        "required_numeric_features": ["delta_gnina_affinity_kcal_mol"],
        "oracle_v2_conservative_residual_weight": 1.0,
        "receptor_ensemble_enabled": False,
        "receptor_ensemble_min_members": 2,
        "receptor_ensemble_max_members": 4,
        "receptor_ensemble_runtime_max_members": 4,
        "receptor_ensemble_aggregate": "median",
        "transform_prior_use_history": False,
        "failure_context_top_n": 5,
        "proposal_transform_prior_weight": 10.0,
        "dynamic_hotspot_fraction_trigger": 0.45,
        "dynamic_new_nonhotspot_min": 1.0,
        "proposal_diversity_enabled": True,
        "proposal_diversity_apply_to_naive": False,
        "proposal_beam_scaffold_repeat_penalty": 0.15,
        "proposal_scaffold_repeat_penalty": 0.25,
        "proposal_family_repeat_penalty": 0.10,
        "proposal_parent_repeat_penalty": 0.08,
        "proposal_scaffold_novelty_bonus": 0.08,
        "proposal_family_novelty_bonus": 0.04,
        "proposal_template_family_coverage_count": 0,
        "proposal_relaxed_max_per_family": 3,
        "proposal_relaxed_max_per_scaffold": 2,
        "proposal_relaxed_max_per_parent": 3,
        "max_proposals_per_scaffold": 2,
        "max_proposals_per_parent": 3,
        "search_beam_diversity_enabled": True,
        "search_beam_diversity_apply_to_naive": False,
        "search_beam_diversity_lock_top_n": 1,
        "search_beam_scaffold_penalty": 0.12,
        "search_beam_scaffold_novelty_bonus": 0.05,
        "dynamics_lite": {
            "enabled": False,
            "every_n_rounds": 4,
            "beam_top_n": 5,
            "final_top_n": 5,
            "reward_weight": 0.20,
            "reward_center": 0.50,
        },
        "reward_weights": {
            "alpha": 1.2,
            "beta": 0.6,
            "eta": 0.4,
            "gamma": 0.2,
            "delta": 0.3,
        },
    }


def test_apply_prefilters_passes_reasonable_molecule() -> None:
    payload = apply_prefilters("Oc1ccccc1", stage6_config())
    assert payload["prefilter_pass"] is True
    assert payload["docking_skipped"] is False
    assert payload["prefilter_fail_reasons"] == []


def test_apply_prefilters_flags_extreme_hydrophobe() -> None:
    payload = apply_prefilters("CCCCCCCCCCCCCCCCCCCC", stage6_config())
    assert payload["prefilter_pass"] is False
    assert payload["docking_skipped"] is True
    assert "clogp_above_max" in payload["prefilter_fail_reasons"]


def test_apply_prefilters_allows_lead_like_baseline_outlier() -> None:
    lead_like = "Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1"
    first_pass = apply_prefilters(lead_like, stage6_config())
    assert first_pass["prefilter_pass"] is False
    second_pass = apply_prefilters(lead_like, stage6_config(), baseline_descriptors=first_pass)
    assert second_pass["prefilter_pass"] is True
    assert "qed_below_min" in second_pass["prefilter_warning_reasons"]
    assert "clogp_above_max" in second_pass["prefilter_warning_reasons"]
    assert second_pass["prefilter_fail_reasons"] == []


def test_apply_prefilters_flags_medchem_blacklist_and_synthesis_fields() -> None:
    payload = apply_prefilters("CC(=O)Cl", stage6_config())
    assert payload["prefilter_pass"] is False
    assert "medchem_blacklist" in payload["prefilter_fail_reasons"]
    assert payload["medchem_blacklist_count"] >= 1
    assert payload["ra_score"] is not None
    assert payload["scscore"] is not None
    assert payload["retrosynthesis_plausibility"] is not None
    assert payload["synthesis_penalty"] > 0.0
    assert payload["total_penalty"] >= payload["synthesis_penalty"]


def test_apply_prefilters_warns_on_unstable_motif_without_hard_fail() -> None:
    payload = apply_prefilters("O=CC1=CC=CC=C1", stage6_config())
    assert payload["unstable_motif_count"] >= 1
    assert "unstable_motif_present" in payload["prefilter_warning_reasons"]


def test_apply_prefilters_computes_series_likeness_against_baseline() -> None:
    baseline = apply_prefilters("Oc1ccccc1", stage6_config())
    payload = apply_prefilters("Oc1cccc(Cl)c1", stage6_config(), baseline_descriptors=baseline)
    assert payload["series_likeness_score"] > 0.25
    assert payload["series_similarity"] > 0.2


def test_pose_pdb_from_sdf_writes_unique_atom_names(tmp_path: Path) -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("CC(F)F"))
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom_index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(atom_index, Point3D(float(atom_index), 0.0, 0.0))
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)

    pose_sdf = tmp_path / "pose.sdf"
    pose_pdb = tmp_path / "pose.pdb"
    writer = Chem.SDWriter(str(pose_sdf))
    writer.write(molecule)
    writer.close()

    pose_pdb_from_sdf(pose_sdf, pose_pdb)
    atom_lines = [line for line in pose_pdb.read_text(encoding="utf-8").splitlines() if line.startswith(("ATOM", "HETATM"))]
    atom_names = [line[12:16].strip() for line in atom_lines]

    assert len(atom_names) == len(set(atom_names))
    assert "F1" in atom_names
    assert "F2" in atom_names


def test_stage6_docking_cache_health_rejects_duplicate_atom_pose(tmp_path: Path) -> None:
    output_root = tmp_path / "dock"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps({"docking_status": "ok"}), encoding="utf-8")
    (output_root / "best_pose.sdf").write_text("placeholder", encoding="utf-8")
    (output_root / "best_pose.pdb").write_text(
        "\n".join(
            [
                "HETATM    1  C   UNL     1      0.000   0.000   0.000  1.00  0.00           C  ",
                "HETATM    2  C   UNL     1      1.000   0.000   0.000  1.00  0.00           C  ",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "receptor_stage6.pdb").write_text("ATOM      1  N   MET A   1      0.0  0.0  0.0  1.00  0.00           N\nEND\n", encoding="utf-8")
    (output_root / "complex_docked.pdb").write_text("ATOM      1  N   MET A   1      0.0  0.0  0.0  1.00  0.00           N\nEND\n", encoding="utf-8")
    (output_root / "ifp.json").write_text("{}", encoding="utf-8")

    assert stage6_utils._stage6_docking_cache_is_healthy(output_root) is False


def test_stage6_docking_cache_health_accepts_unique_pose(tmp_path: Path) -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("CC(F)F"))
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom_index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(atom_index, Point3D(float(atom_index), 0.0, 0.0))
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)

    output_root = tmp_path / "dock"
    output_root.mkdir(parents=True, exist_ok=True)
    pose_sdf = output_root / "best_pose.sdf"
    writer = Chem.SDWriter(str(pose_sdf))
    writer.write(molecule)
    writer.close()
    pose_pdb_from_sdf(pose_sdf, output_root / "best_pose.pdb")
    (output_root / "summary.json").write_text(json.dumps({"docking_status": "ok"}), encoding="utf-8")
    (output_root / "receptor_stage6.pdb").write_text("ATOM      1  N   MET A   1      0.0  0.0  0.0  1.00  0.00           N\nEND\n", encoding="utf-8")
    (output_root / "complex_docked.pdb").write_text("ATOM      1  N   MET A   1      0.0  0.0  0.0  1.00  0.00           N\nEND\n", encoding="utf-8")
    (output_root / "ifp.json").write_text("{}", encoding="utf-8")

    assert stage6_utils._stage6_docking_cache_is_healthy(output_root) is True


def test_enumerate_and_apply_candidate_action_changes_smiles() -> None:
    mol = Chem.MolFromSmiles("Oc1ccccc1")
    actions = enumerate_candidate_actions(mol, stage6_config())
    assert actions
    families = {str(action["edit_family"]) for action in actions}
    assert {"HALOGEN_SCAN", "SMALL_POLAR_SCAN", "HETEROARYL_SWAP", "BACKBONE_SEEKER", "WATER_BRIDGE_PROBE"} <= families
    child = apply_action_to_molecule(mol, actions[0])
    assert child is not None
    assert canonical_smiles(child) != canonical_smiles(mol)


def test_new_action_families_apply_meaningful_graph_changes() -> None:
    linker = Chem.MolFromSmiles("CCNC")
    rigidified = apply_action_to_molecule(
        linker,
        {"edit_family": "RIGIDIFY_LINKER", "atom_idx": 1, "neighbor_idx": 2},
    )
    assert rigidified is not None
    assert canonical_smiles(rigidified) != canonical_smiles(linker)

    heteroaryl = Chem.MolFromSmiles("c1ccccc1")
    swapped = apply_action_to_molecule(
        heteroaryl,
        {"edit_family": "HETEROARYL_SWAP", "atom_idx": 0, "swap_to": "N"},
    )
    assert swapped is not None
    assert canonical_smiles(swapped) != canonical_smiles(heteroaryl)

    ring = Chem.MolFromSmiles("C1CCCCC1")
    expanded = apply_action_to_molecule(
        ring,
        {"edit_family": "RING_EXPANSION", "atom_idx": 0, "neighbor_idx": 1},
    )
    contracted = apply_action_to_molecule(
        ring,
        {"edit_family": "RING_CONTRACTION", "atom_idx": 0},
    )
    assert expanded is not None
    assert contracted is not None
    assert expanded.GetNumAtoms() == ring.GetNumAtoms() + 1
    assert contracted.GetNumAtoms() == ring.GetNumAtoms() - 1

    vector_source = Chem.MolFromSmiles("Cc1cccc(O)c1")
    leaf_idx = next(atom.GetIdx() for atom in vector_source.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetDegree() == 1)
    anchor_idx = next(nei.GetIdx() for nei in vector_source.GetAtomWithIdx(leaf_idx).GetNeighbors())
    target_idx = next(
        nei.GetIdx()
        for nei in vector_source.GetAtomWithIdx(anchor_idx).GetNeighbors()
        if nei.GetIsAromatic() and nei.GetIdx() != leaf_idx and nei.GetTotalNumHs() > 0
    )
    flipped = apply_action_to_molecule(
        vector_source,
        {
            "edit_family": "VECTOR_FLIP",
            "atom_idx": leaf_idx,
            "source_anchor_idx": anchor_idx,
            "target_anchor_idx": target_idx,
        },
    )
    assert flipped is not None
    assert canonical_smiles(flipped) != canonical_smiles(vector_source)


def test_dep_decreases_when_hotspot_contacts_disappear_but_nonhotspot_contacts_remain() -> None:
    stage6 = stage6_config()
    hotspot_only_ifp = {
        "interactions": [
            {"interaction_type": "hydrogen_bond", "residue_label": "A:THR315"},
            {"interaction_type": "hydrogen_bond", "residue_label": "A:THR315"},
        ]
    }
    diversified_ifp = {
        "interactions": [
            {"interaction_type": "hydrogen_bond", "residue_label": "A:MET318"},
            {"interaction_type": "hydrophobic", "residue_label": "A:GLU286"},
        ]
    }
    pocket = {"A:THR315", "A:MET318", "A:GLU286"}
    hotspot = {"A:THR315"}
    assert dep_score(diversified_ifp, hotspot, pocket, stage6) < dep_score(hotspot_only_ifp, hotspot, pocket, stage6)


def test_new_nonhotspot_contacts_contribute_to_compensation_gain() -> None:
    stage6 = stage6_config()
    baseline_ifp = {
        "interactions": [
            {"interaction_type": "hydrogen_bond", "residue_label": "A:THR315"},
        ],
        "residue_set": ["A:THR315"],
    }
    candidate_ifp = {
        "interactions": [
            {"interaction_type": "hydrogen_bond", "residue_label": "A:THR315"},
            {"interaction_type": "hydrophobic", "residue_label": "A:GLU286"},
        ],
        "residue_set": ["A:THR315", "A:GLU286"],
    }
    payload = new_nonhotspot_contact_metrics(
        baseline_ifp=baseline_ifp,
        candidate_ifp=candidate_ifp,
        hotspot_residues={"A:THR315"},
        stage6=stage6,
    )
    assert payload["new_nonhotspot_residue_count"] == 1
    assert payload["new_nonhotspot_contact_score"] > 0.0


def test_weighted_ifp_cosine_rewards_anchor_overlap() -> None:
    stage6 = stage6_config()
    baseline_ifp = {
        "interactions": [
            {"interaction_type": "hydrogen_bond", "residue_label": "A:LYS721"},
            {"interaction_type": "hydrophobic", "residue_label": "A:MET769"},
        ]
    }
    candidate_ifp = {
        "interactions": [
            {"interaction_type": "hydrogen_bond", "residue_label": "A:LYS721"},
            {"interaction_type": "hydrophobic", "residue_label": "A:MET769"},
        ]
    }
    score = weighted_ifp_cosine(
        baseline_ifp=baseline_ifp,
        candidate_ifp=candidate_ifp,
        residues={"A:LYS721", "A:MET769"},
        stage6=stage6,
    )
    assert score == 1.0


def test_weighted_cvar_uses_weighted_worst_tail() -> None:
    score = weighted_cvar([0.1, 0.2, 0.9], [0.2, 0.3, 0.5], 0.2)
    assert abs(score - 0.1) < 1e-9


def test_uncertainty_heavy_target_score_uses_raw_docking_floor_when_calibrator_collapses() -> None:
    score = uncertainty_heavy_target_score(
        calibrated_score=4.138020019377218e-05,
        raw_docking_score=0.5443828762475713,
        keep_ifp_value=0.6,
        anchor_loss_inverse=0.0,
        use_raw_docking_floor=True,
    )
    assert score > 0.24
    assert round(score, 6) == round(0.4 * 0.5443828762475713 + 0.4 * 0.6, 6)


def test_recompute_cached_candidate_scores_refreshes_uncertainty_heavy_objectives() -> None:
    row = {
        "objective_name": "robust",
        "candidate_valid": True,
        "s_wt": 0.5,
        "keep_ifp": 0.9,
        "dep": 0.17,
        "compensation_gain": 0.0,
        "admet_penalty": 0.0,
        "panel_coverage": 1.0,
        "keep_ifp_constraint_pass": True,
        "compensation_constraint_pass": True,
        "hotspot_drop": 0.0,
        "target_scores_json": json.dumps(
            {
                "MUT1": {
                    "effect_scope": "site",
                    "target_uncertain": False,
                    "raw_docking_score": 0.54,
                    "calibrated_score": 4.138020019377218e-05,
                    "keep_ifp": 0.6,
                    "lost_anchor_labels": ["A:LYS721"],
                },
                "MUT2": {
                    "effect_scope": "site",
                    "target_uncertain": False,
                    "raw_docking_score": 0.52,
                    "calibrated_score": 4.138020019377218e-05,
                    "keep_ifp": 0.6,
                    "lost_anchor_labels": ["A:LYS721"],
                },
            },
            ensure_ascii=True,
        ),
    }
    case_context = {
        "anchor_residues": ["A:LYS721"],
        "panel_weights": {"MUT1": 0.5, "MUT2": 0.5},
        "scoring_policy": {"uncertainty_heavy": True},
    }

    refreshed = recompute_cached_candidate_scores(row, case_context=case_context, stage6=stage6_config())

    assert refreshed["robust_core"] > 0.24
    assert refreshed["robust_score"] > 0.24
    payload = json.loads(refreshed["target_scores_json"])
    assert payload["MUT1"]["score"] > 0.24
    assert refreshed["objective_reward"] == refreshed["robust_objective_reward"]


def test_objective_ablation_rows_reports_gain_and_diversity() -> None:
    frame = pd.DataFrame.from_records(
        [
            {
                "objective_name": "robust",
                "candidate_id": "lead",
                "objective_reward": 1.0,
                "robust_score": 0.5,
                "dep": 0.8,
                "candidate_valid": True,
                "wt_hard_constraint_pass": True,
                "scaffold_smiles": murcko_scaffold("Oc1ccccc1"),
            },
            {
                "objective_name": "robust",
                "candidate_id": "cand1",
                "objective_reward": 1.5,
                "robust_score": 0.65,
                "robust_core": 0.60,
                "dep": 0.5,
                "keep_ifp": 0.8,
                "candidate_valid": True,
                "wt_hard_constraint_pass": True,
                "scaffold_smiles": murcko_scaffold("Fc1ccccc1O"),
            },
            {
                "objective_name": "naive",
                "candidate_id": "lead",
                "objective_reward": 1.0,
                "robust_score": 0.5,
                "robust_core": 0.48,
                "dep": 0.8,
                "keep_ifp": 0.6,
                "candidate_valid": True,
                "wt_hard_constraint_pass": True,
                "scaffold_smiles": murcko_scaffold("Oc1ccccc1"),
            },
        ]
    )
    rows = objective_ablation_rows(frame, "lead", stage6_config(), combo_dense=False)
    assert len(rows) == 2
    robust_row = next(row for row in rows if row["objective_name"] == "robust")
    assert robust_row["best_candidate_id"] == "cand1"
    assert robust_row["top20_robust_score_gain_fraction"] > 0.0
    assert robust_row["top20_robust_core_median"] >= 0.5
    assert robust_row["top20_keep_ifp_median"] >= 0.7
    assert robust_row["panel_passing_rate"] == robust_row["valid_candidate_rate"]
    assert robust_row["prefilter_pass_rate"] == 1.0


def test_stage6_executability_metrics_split_rates() -> None:
    frame = pd.DataFrame.from_records(
        [
            {
                "chemical_valid": True,
                "prefilter_pass": True,
                "wt_pass": True,
                "panel_coverage_pass": True,
                "panel_passing": True,
            },
            {
                "chemical_valid": True,
                "prefilter_pass": True,
                "wt_pass": False,
                "panel_coverage_pass": False,
                "panel_passing": False,
            },
            {
                "chemical_valid": True,
                "prefilter_pass": False,
                "wt_pass": False,
                "panel_coverage_pass": False,
                "panel_passing": False,
            },
        ]
    )
    metrics = stage6_executability_metrics(frame)
    assert metrics["chemical_valid_rate"] == 1.0
    assert metrics["prefilter_pass_rate"] == 2 / 3
    assert metrics["wt_pass_rate"] == 1 / 3
    assert metrics["panel_coverage_pass_rate"] == 1 / 3
    assert metrics["panel_passing_rate"] == 1 / 3
    assert metrics["valid_candidate_rate"] == metrics["panel_passing_rate"]


def test_build_transform_prior_skips_historical_stage6_when_disabled(tmp_path) -> None:
    case_stage6_root = tmp_path / "outputs" / "demo_case" / "stage6"
    case_stage6_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(
        [
            {
                "objective_reward": 1.5,
                "action_sequence_json": json.dumps([{"edit_family": "ADD", "pattern": "add_polar_cap", "fragment": "N"}]),
            }
        ]
    ).to_csv(case_stage6_root / "leaderboard.csv", index=False)
    prior, summary = build_transform_prior(
        root=tmp_path,
        case_stage6_root=case_stage6_root,
        lead_smiles="Oc1ccccc1",
        stage6=stage6_config(),
    )
    assert prior
    assert all(item["source"] != "historical_stage6" for item in summary)


def test_build_transform_prior_includes_case_specific_pocket_hints(tmp_path) -> None:
    case_stage6_root = tmp_path / "outputs" / "demo_case" / "stage6"
    case_stage6_root.mkdir(parents=True, exist_ok=True)
    prior, summary = build_transform_prior(
        root=tmp_path,
        case_stage6_root=case_stage6_root,
        lead_smiles="Oc1ccccc1",
        stage6=stage6_config(),
        case_specific_action_hints_payload=[
            {
                "edit_family": "SMALL_POLAR_SCAN",
                "pattern": "small_polar_scan",
                "fragment": "O",
                "weight": 3.0,
                "rationale": "synthetic hint",
            }
        ],
    )

    assert any(item["source"] == "case_specific_pocket" for item in summary)
    assert ("SMALL_POLAR_SCAN", "small_polar_scan", "O") in prior


def test_build_dynamic_transform_prior_promotes_partner_chain_polar_transforms() -> None:
    beam_frame = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "cand1",
                "objective_reward": 1.2,
                "hotspot_fraction": 0.62,
                "new_nonhotspot_contact_score": 0.2,
                "target_scores_json": json.dumps(
                    {
                        "GAG-POL_RT:E138K": {
                            "target_key": "GAG-POL_RT:E138K",
                            "docking_status": "ok",
                            "score": 0.31,
                            "keep_ifp": 0.44,
                            "mechanism_labels": ["electrostatic_shift", "anchor_loss"],
                            "target_uncertain": False,
                            "is_partner_chain_sensitive": True,
                        }
                    }
                ),
            }
        ]
    )
    prior = SCRIPT_09_MODULE.build_dynamic_transform_prior(
        beam_frame=beam_frame,
        case_context={
            "action_space": {
                "case_specific_action_hints": [
                    {
                        "edit_family": "SMALL_POLAR_SCAN",
                        "pattern": "small_polar_scan",
                        "fragment": "OC",
                        "weight": 3.0,
                    }
                ]
            }
        },
        stage6=stage6_config(),
        objective_name="robust",
    )

    assert ("SMALL_POLAR_SCAN", "small_polar_scan", "OC") in prior
    assert ("LINKER_HETERO_SCAN", "linker_hetero_scan", "N") in prior


def test_counter_design_agent_disable_llm_uses_fallback_templates() -> None:
    agent = CounterDesignAgent(client=DummyClient(), config=CounterDesignAgentConfig(proposal_count=6))
    _, payload, record = agent.run(
        case_entry={
            "case_id": "abl1_nilotinib",
            "target_name": "ABL1",
            "drug_name": "Nilotinib",
            "target_domain": "kinase",
            "evaluation_unit": "site",
        },
        objective_name="robust",
        round_index=1,
        mechanism_summary={"mechanism_label_counts": {"anchor_loss": 2, "steric_clash": 1}},
        hotspot_residues=["A:THR315"],
        beam_frame=pd.DataFrame(),
        action_space={"fragments": ["F", "Cl", "OC", "N"]},
        qc_payload={"beam_size": 1},
        lead_descriptors={"qed": 0.28, "clogp": 6.2, "sa_score": 6.4, "rotatable_bonds": 6},
        temperature=0.7,
        disable_llm=True,
    )
    assert record is None
    assert payload["action_templates"]
    assert payload["action_templates"][0]["edit_family"] in {
        "ADD",
        "REPLACE",
        "DELETE",
        "HETERO_SWAP",
        "RING_EDIT",
        "HALOGEN_SCAN",
        "SMALL_POLAR_SCAN",
        "LINKER_HETERO_SCAN",
        "N_METHYL_SCAN",
        "CONSTRAINED_TAIL_TRIM",
    }


def test_counter_design_prompt_input_contains_failure_and_pocket_context() -> None:
    agent = CounterDesignAgent(client=DummyClient(), config=CounterDesignAgentConfig(proposal_count=6))
    beam_frame = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "cand1",
                "objective_reward": 0.9,
                "robust_score": 0.7,
                "keep_ifp": 0.8,
                "dep": 0.2,
                "panel_coverage": 0.95,
                "new_nonhotspot_contact_score": 0.4,
                "hotspot_fraction": 0.3,
                "mechanism_risk_focus": "balanced",
                "target_scores_json": json.dumps(
                    {
                        "GAG-POL_RT:E138K": {
                            "target_key": "GAG-POL_RT:E138K",
                            "effect_scope": "site",
                            "score": 0.45,
                            "keep_ifp": 0.66,
                            "lost_anchor_labels": ["A:PRO95"],
                            "mechanism_labels": ["electrostatic_shift"],
                            "target_uncertain": False,
                            "is_partner_chain_sensitive": True,
                        }
                    }
                ),
                "wt_hard_constraint_pass": True,
                "coverage_pass": False,
                "keep_ifp_constraint_pass": False,
                "compensation_constraint_pass": True,
            }
        ]
    )
    prompt_input = agent.build_prompt_input(
        case_entry={
            "case_id": "hiv_rt_rilpivirine",
            "target_name": "HIV RT",
            "drug_name": "Rilpivirine",
            "target_domain": "rt",
            "evaluation_unit": "observed_combo",
        },
        objective_name="robust",
        round_index=2,
        mechanism_summary={"mechanism_label_counts": {"electrostatic_shift": 3}},
        hotspot_residues=["A:LYS101"],
        beam_frame=beam_frame,
        action_space={
            "fragments": ["F", "Cl", "OC", "N"],
            "partner_chain_residues": ["B:GLU138"],
            "partner_chain_positions": [138],
            "pocket_profile": {"dominant_classes": ["polar"]},
            "case_specific_action_hints": [
                {"edit_family": "SMALL_POLAR_SCAN", "pattern": "small_polar_scan", "fragment": "OC", "weight": 2.5}
            ],
        },
        qc_payload={"beam_size": 1},
        lead_descriptors={"qed": 0.4},
    )

    assert prompt_input["failure_context"]["partner_chain_sensitive_count"] == 1
    assert prompt_input["constraint_breakdown"]["coverage_fail_count"] == 1
    assert prompt_input["pocket_context"]["partner_chain_positions"] == [138]
    assert prompt_input["pocket_context"]["case_specific_action_hints"][0]["edit_family"] == "SMALL_POLAR_SCAN"


def test_llm_failure_context_audit_reads_saved_round_inputs(tmp_path) -> None:
    stage6_root = tmp_path / "stage6"
    input_path = stage6_root / "llm" / "robust" / "round_01_input.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(
        json.dumps(
            {
                "search_state": {"beam_size": 4},
                "constraint_breakdown": {
                    "wt_fail_count": 1,
                    "coverage_fail_count": 2,
                    "keep_ifp_fail_count": 3,
                    "compensation_fail_count": 0,
                },
                "failure_context": {
                    "worst_targets": [
                        {
                            "target_key": "GAG-POL_RT:E138K",
                            "effect_scope": "site",
                            "target_uncertain": False,
                            "is_partner_chain_sensitive": True,
                            "mechanism_labels": ["electrostatic_shift"],
                        },
                        {
                            "target_key": "GAG-POL_RT:K101E+Y181C",
                            "effect_scope": "combo",
                            "target_uncertain": True,
                            "is_partner_chain_sensitive": False,
                            "mechanism_labels": ["anchor_loss", "steric_clash"],
                        },
                    ]
                },
                "pocket_context": {
                    "partner_chain_residues": ["B:GLU138"],
                    "case_specific_action_hints": [
                        {"edit_family": "SMALL_POLAR_SCAN", "pattern": "small_polar_scan", "fragment": "OC"}
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    frame = llm_failure_context_audit_frame(stage6_root)

    assert len(frame) == 1
    row = frame.iloc[0].to_dict()
    assert row["objective_name"] == "robust"
    assert row["worst_target_count"] == 2
    assert row["worst_combo_target_count"] == 1
    assert row["partner_chain_sensitive_worst_target_count"] == 1
    assert row["steric_clash_worst_target_count"] == 1
    assert row["partner_chain_residue_count"] == 1


def test_candidate_not_killed_by_single_target_failure_if_coverage_passes(tmp_path, monkeypatch) -> None:
    root = tmp_path
    case_stage6_root = root / "outputs" / "demo_case" / "stage6"
    case_stage6_root.mkdir(parents=True, exist_ok=True)
    receptor_path = root / "receptor.pdb"
    receptor_path.write_text("ATOM      1  N   MET A   1      0.0  0.0  0.0  1.00  0.00           N\nEND\n", encoding="utf-8")
    pose_path = root / "pose.sdf"
    pose_path.write_text("", encoding="utf-8")

    wt_ifp_path = root / "wt_ifp.json"
    wt_ifp_path.write_text(
        json.dumps(
            {
                "interactions": [
                    {"interaction_type": "hydrogen_bond", "residue_label": "A:LYS721"},
                    {"interaction_type": "hydrophobic", "residue_label": "A:MET769"},
                ],
                "residue_set": ["A:LYS721", "A:MET769"],
            }
        ),
        encoding="utf-8",
    )
    target1_ifp_path = root / "target1_ifp.json"
    target1_ifp_path.write_text(
        json.dumps(
            {
                "interactions": [
                    {"interaction_type": "hydrogen_bond", "residue_label": "A:LYS721"},
                    {"interaction_type": "hydrophobic", "residue_label": "A:MET769"},
                ],
                "residue_set": ["A:LYS721", "A:MET769"],
            }
        ),
        encoding="utf-8",
    )
    target2_ifp_path = root / "target2_ifp.json"
    target2_ifp_path.write_text(
        json.dumps(
            {
                "interactions": [
                    {"interaction_type": "hydrogen_bond", "residue_label": "A:LYS721"},
                    {"interaction_type": "hydrophobic", "residue_label": "A:GLU738"},
                ],
                "residue_set": ["A:LYS721", "A:GLU738"],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        stage6_utils,
        "apply_prefilters",
        lambda smiles, stage6, baseline_descriptors=None: {
            "prefilter_pass": True,
            "prefilter_fail_reason": "",
            "prefilter_fail_reasons": [],
            "prefilter_warning_reason": "",
            "prefilter_warning_reasons": [],
            "docking_skipped": False,
            "prefilter_score": 1.0,
            "admet_penalty": 0.0,
        },
    )
    monkeypatch.setattr(stage6_utils, "_wt_receptor_pdb", lambda case_context: receptor_path)
    monkeypatch.setattr(stage6_utils, "_safe_local_pocket_metrics", lambda receptor_pdb, ligand_sdf: {
        "pocket_volume_proxy_a3": 10.0,
        "polar_exposed_fraction": 0.2,
    })

    def fake_dock_candidate(**kwargs):
        return {
            "docking_status": "ok",
            "best_affinity_kcal_mol": -9.0,
            "ifp_json": str(wt_ifp_path.relative_to(root)),
            "receptor_pdb": str(receptor_path.relative_to(root)),
            "pose_sdf": str(pose_path.relative_to(root)),
        }

    def fake_target_row(**kwargs):
        row = kwargs["row"]
        if row["target_key"] == "MUT3":
            return row["target_key"], row["effect_scope"], {"docking_status": "failed", "best_affinity_kcal_mol": None}
        ifp_path = target1_ifp_path if row["target_key"] == "MUT1" else target2_ifp_path
        affinity = -8.8 if row["target_key"] == "MUT1" else -8.7
        return row["target_key"], row["effect_scope"], {
            "docking_status": "ok",
            "best_affinity_kcal_mol": affinity,
            "ifp_json": str(ifp_path.relative_to(root)),
            "receptor_pdb": str(receptor_path.relative_to(root)),
            "pose_sdf": str(pose_path.relative_to(root)),
        }

    monkeypatch.setattr(stage6_utils, "_dock_candidate", fake_dock_candidate)
    monkeypatch.setattr(stage6_utils, "_dock_target_panel_row", fake_target_row)

    case_context = {
        "case_id": "demo_case",
        "stage6_root": str(case_stage6_root),
        "stage3_5_root": str(root),
        "lead_descriptors": {},
        "lead_wt_affinity_kcal_mol": -8.5,
        "lead_mt_affinities": {"MUT1": -8.0, "MUT2": -8.0, "MUT3": -8.0},
        "panel_rows": [
            {"target_key": "MUT1", "effect_scope": "site", "sample_root": str(root), "stage4_local_rmsd_a": 0.1},
            {"target_key": "MUT2", "effect_scope": "site", "sample_root": str(root), "stage4_local_rmsd_a": 0.2},
            {"target_key": "MUT3", "effect_scope": "site", "sample_root": str(root), "stage4_local_rmsd_a": 0.3},
        ],
        "panel_weights": {"MUT1": 0.45, "MUT2": 0.45, "MUT3": 0.10},
        "anchor_residues": ["A:LYS721"],
        "hotspot_residues": ["A:LYS721"],
        "nonhotspot_residues": ["A:MET769", "A:GLU738"],
        "pocket_residues_universe": ["A:LYS721", "A:MET769", "A:GLU738"],
        "baseline_ifp": {
            "interactions": [
                {"interaction_type": "hydrogen_bond", "residue_label": "A:LYS721"},
                {"interaction_type": "hydrophobic", "residue_label": "A:MET769"},
            ],
            "residue_set": ["A:LYS721", "A:MET769"],
        },
        "docking_box": {},
        "hiv_reference": None,
        "wt_chain_id": "A",
        "scoring_policy": {"uncertainty_heavy": False},
        "calibrator_path": "",
    }

    row = stage6_utils.evaluate_candidate(
        root=root,
        case_context=case_context,
        stage6=stage6_config(),
        smiles="Oc1ccccc1",
        action_sequence=[],
        round_index=1,
        objective_name="robust",
    )

    assert row["candidate_valid"] is True
    assert row["coverage_pass"] is True
    assert row["target_uncertain_count"] == 1
    assert row["panel_coverage"] >= 0.8


def test_uncertain_partner_chain_target_stays_flagged_in_target_scores(tmp_path, monkeypatch) -> None:
    root = tmp_path
    wt_ifp_path = root / "wt_ifp.json"
    wt_ifp_path.write_text(
        json.dumps(
            {
                "interactions": [
                    {"interaction_type": "hydrogen_bond", "residue_label": "A:LYS101"},
                    {"interaction_type": "hydrophobic", "residue_label": "A:PRO95"},
                ],
                "residue_set": ["A:LYS101", "A:PRO95"],
            }
        ),
        encoding="utf-8",
    )
    receptor_path = root / "receptor.pdb"
    receptor_path.write_text("ATOM      1  N   MET A   1      0.0  0.0  0.0  1.00  0.00           N\nEND\n", encoding="utf-8")
    pose_path = root / "pose.sdf"
    pose_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        stage6_utils,
        "_dock_candidate",
        lambda **kwargs: {
            "docking_status": "ok",
            "best_affinity_kcal_mol": -10.5,
            "ifp_json": str(wt_ifp_path.relative_to(root)),
            "receptor_pdb": str(receptor_path.relative_to(root)),
            "pose_sdf": str(pose_path.relative_to(root)),
        },
    )
    monkeypatch.setattr(
        stage6_utils,
        "_dock_target_panel_row",
        lambda **kwargs: (
            "GAG-POL_RT:E138K",
            "site",
            {"docking_status": "failed"},
        ),
    )
    monkeypatch.setattr(
        stage6_utils,
        "_safe_local_pocket_metrics",
        lambda receptor_pdb, ligand_sdf: {
            "pocket_volume_proxy_a3": 10.0,
            "polar_exposed_fraction": 0.2,
        },
    )
    monkeypatch.setattr(stage6_utils, "_wt_receptor_pdb", lambda case_context: receptor_path)

    row = stage6_utils.evaluate_candidate(
        root=root,
        case_context={
            "case_id": "hiv_rt_rilpivirine",
            "stage6_root": str(root / "outputs" / "hiv_rt_rilpivirine" / "stage6"),
            "stage3_5_root": str(root),
            "lead_descriptors": {},
            "lead_wt_affinity_kcal_mol": -10.0,
            "docking_box": {"center_x": 0.0, "center_y": 0.0, "center_z": 0.0, "size_x": 10.0, "size_y": 10.0, "size_z": 10.0},
            "hiv_reference": None,
            "anchor_residues": ["A:LYS101", "A:PRO95"],
            "hotspot_residues": ["A:LYS101"],
            "baseline_ifp": json.loads(wt_ifp_path.read_text(encoding="utf-8")),
            "panel_rows": [{"target_key": "GAG-POL_RT:E138K", "effect_scope": "site", "sample_root": str(root)}],
            "panel_weights": {"GAG-POL_RT:E138K": 1.0},
            "lead_mt_affinities": {"GAG-POL_RT:E138K": -9.5},
            "pocket_residues_universe": ["A:LYS101", "A:PRO95", "B:GLU138"],
            "partner_chain_positions": [138],
            "partner_chain_residues": ["B:GLU138"],
            "calibrator_path": "",
            "scoring_policy": {},
        },
        stage6=stage6_config(),
        smiles="CCO",
        action_sequence=[],
        round_index=1,
        objective_name="robust",
    )

    target_scores = json.loads(row["target_scores_json"])
    assert target_scores["GAG-POL_RT:E138K"]["target_uncertain"] is True
    assert target_scores["GAG-POL_RT:E138K"]["is_partner_chain_sensitive"] is True
    assert target_scores["GAG-POL_RT:E138K"]["mutation_positions"] == [138]


def test_hiv_multichain_receptor_retains_partner_chain_hotspot(tmp_path) -> None:
    sample_root = tmp_path / "sample"
    sample_root.mkdir(parents=True, exist_ok=True)
    mt_receptor = sample_root / "MT.pdb"
    mt_receptor.write_text(
        "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N\nEND\n",
        encoding="utf-8",
    )
    wt_multichain = tmp_path / "wt_receptor_stage6_multichain.pdb"
    wt_multichain.write_text(
        "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  N   GLU B 138       1.000   0.000   0.000  1.00  0.00           N\nEND\n",
        encoding="utf-8",
    )

    result = _target_panel_receptor_pdb(
        row={"sample_root": str(sample_root)},
        case_context={
            "use_multichain_receptor": True,
            "partner_chain_ids": ["B"],
            "wt_receptor_stage6_path": str(wt_multichain),
            "wt_chain_id": "A",
        },
        output_root=tmp_path / "target",
    )

    lines = [line for line in result.read_text(encoding="utf-8").splitlines() if line.startswith("ATOM")]
    assert {line[21] for line in lines} == {"A", "B"}


def test_stage6_calibrator_trains_and_predicts(tmp_path) -> None:
    stage5_root = tmp_path / "stage5"
    stage6_root = tmp_path / "stage6"
    stage5_root.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "case_id": "demo",
            "effect_scope": "site",
            "target_key": f"MUT{i}",
            "representative_sample_id": f"S{i}",
            "delta_dock_kcal_mol": 0.2 * i,
            "delta_gnina_affinity_kcal_mol": 0.1 * i,
            "ifp_jaccard_loss": 0.05 * i,
            "ifp_occupancy_shift_mean_abs": 0.03 * i,
            "ifp_occupancy_anchor_loss": 0.02 * i,
            "anchor_loss_fraction": 0.04 * i,
            "pocket_volume_change_fraction": -0.01 * i,
            "solvent_proxy_shift": 0.01 * i,
            "stage4_local_rmsd_a": 0.1 * i,
            "delta_mmgbsa_binding_kcal_mol": 0.25 * i,
        }
        for i in range(1, 7)
    ]
    pd.DataFrame.from_records(records).to_csv(stage5_root / "ifp_diff.csv", index=False)
    pd.DataFrame.from_records(records).to_csv(stage5_root / "scoring_calibration.csv", index=False)
    (stage5_root / "stage5_qc.json").write_text('{"dock_vs_mmgbsa_pearson_r": 0.75}', encoding="utf-8")

    metadata = fit_case_calibrator(
        stage5_root=stage5_root,
        stage6_root=stage6_root,
        case_id="demo",
        stage6=stage6_config(),
        stage5_qc={"dock_vs_mmgbsa_pearson_r": 0.75},
    )
    calibrator = load_case_calibrator(str(stage6_root / "calibrator" / "stage6_calibrator.pkl"))

    assert metadata["available"] is True
    score = calibrator.predict_score(
        {
            "effect_scope": "site",
            "delta_dock_kcal_mol": 0.8,
            "delta_gnina_affinity_kcal_mol": 0.4,
            "ifp_jaccard_loss": 0.2,
            "ifp_occupancy_shift_mean_abs": 0.1,
            "ifp_occupancy_anchor_loss": 0.1,
            "anchor_loss_fraction": 0.15,
            "pocket_volume_change_fraction": -0.05,
            "solvent_proxy_shift": 0.05,
            "stage4_local_rmsd_a": 0.3,
        },
        scale=1.5,
    )
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_stage6_calibrator_rejects_missing_required_feature(tmp_path) -> None:
    stage5_root = tmp_path / "stage5"
    stage6_root = tmp_path / "stage6"
    stage5_root.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "case_id": "demo",
            "effect_scope": "site",
            "target_key": f"MUT{i}",
            "representative_sample_id": f"S{i}",
            "delta_dock_kcal_mol": 0.2 * i,
            "delta_gnina_affinity_kcal_mol": None,
            "ifp_jaccard_loss": 0.05 * i,
            "ifp_occupancy_shift_mean_abs": 0.03 * i,
            "ifp_occupancy_anchor_loss": 0.02 * i,
            "anchor_loss_fraction": 0.04 * i,
            "pocket_volume_change_fraction": -0.01 * i,
            "solvent_proxy_shift": 0.01 * i,
            "stage4_local_rmsd_a": 0.1 * i,
            "delta_mmgbsa_binding_kcal_mol": 0.25 * i,
        }
        for i in range(1, 7)
    ]
    pd.DataFrame.from_records(records).to_csv(stage5_root / "ifp_diff.csv", index=False)
    pd.DataFrame.from_records(records).to_csv(stage5_root / "scoring_calibration.csv", index=False)

    try:
        fit_case_calibrator(
            stage5_root=stage5_root,
            stage6_root=stage6_root,
            case_id="demo",
            stage6=stage6_config(),
            stage5_qc={"dock_vs_mmgbsa_pearson_r": 0.75},
        )
    except RuntimeError as exc:
        assert "delta_gnina_affinity_kcal_mol" in str(exc)
    else:
        raise AssertionError("fit_case_calibrator should fail fast when required features are entirely missing")


def test_rerank_uncertainty_heavy_candidates_preserves_rerank_columns_when_disabled(tmp_path) -> None:
    case_stage6_root = tmp_path / "stage6"
    case_stage6_root.mkdir(parents=True, exist_ok=True)
    robust_frame = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "cand1",
                "objective_reward": 1.25,
                "robust_score": 0.8,
            }
        ]
    )
    reranked = SCRIPT_09_MODULE.rerank_uncertainty_heavy_candidates(
        root=tmp_path,
        case_stage6_root=case_stage6_root,
        case_context={"scoring_policy": {"uncertainty_heavy": False}, "lead_candidate_id": "lead"},
        stage6=stage6_config(),
        robust_frame=robust_frame,
    )

    assert "reranked_objective_reward" in reranked.columns
    assert "wt_gnina_affinity_kcal_mol" in reranked.columns
    assert "wt_gnina_score" in reranked.columns
    assert reranked.at[0, "reranked_objective_reward"] == 1.25
    assert pd.isna(reranked.at[0, "wt_gnina_affinity_kcal_mol"])
    assert pd.isna(reranked.at[0, "wt_gnina_score"])


def test_apply_search_beam_diversity_prefers_beam_reranked_reward() -> None:
    beam_source = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "base_best",
                "objective_reward": 1.00,
                "beam_reranked_objective_reward": 0.90,
                "robust_score": 0.60,
                "keep_ifp": 0.85,
                "dep": 0.10,
                "scaffold_smiles": "scaf_a",
            },
            {
                "candidate_id": "reranked_best",
                "objective_reward": 0.95,
                "beam_reranked_objective_reward": 1.05,
                "robust_score": 0.59,
                "keep_ifp": 0.84,
                "dep": 0.11,
                "scaffold_smiles": "scaf_b",
            },
        ]
    )

    selected = SCRIPT_09_MODULE.apply_search_beam_diversity(
        beam_source=beam_source,
        beam_width=1,
        stage6=stage6_config(),
        objective_name="robust",
    )

    assert selected.iloc[0]["candidate_id"] == "reranked_best"


def test_apply_dynamics_lite_rerank_reorders_top_candidates(monkeypatch, tmp_path: Path) -> None:
    def fake_probe(*, candidate_id_text: str, **kwargs):
        if candidate_id_text == "cand_a":
            return {
                "available": True,
                "score": 0.0,
                "contact_survival": 0.2,
                "anchor_contact_survival": 0.2,
                "occupancy_persistence": 0.1,
                "anchor_persistence": 0.1,
                "ifp_occupancy_shift_mean_abs": 0.9,
                "ifp_occupancy_anchor_loss": 0.9,
                "relaxation_mode": "local_sampling_implicit_md",
                "local_sampling_fallback_reason": "",
                "fallback_penalty": 0.0,
            }
        return {
            "available": True,
            "score": 1.0,
            "contact_survival": 0.9,
            "anchor_contact_survival": 0.8,
            "occupancy_persistence": 0.9,
            "anchor_persistence": 0.8,
            "ifp_occupancy_shift_mean_abs": 0.1,
            "ifp_occupancy_anchor_loss": 0.2,
            "relaxation_mode": "local_sampling_implicit_md",
            "local_sampling_fallback_reason": "",
            "fallback_penalty": 0.0,
        }

    monkeypatch.setattr(SCRIPT_09_MODULE, "run_stage6_dynamics_lite_probe", fake_probe)
    stage6 = stage6_config()
    stage6["dynamics_lite"]["enabled"] = True
    stage6["dynamics_lite"]["final_top_n"] = 2
    robust_frame = pd.DataFrame.from_records(
        [
            {"candidate_id": "cand_a", "objective_reward": 1.00, "robust_score": 0.60, "dep": 0.10},
            {"candidate_id": "cand_b", "objective_reward": 0.95, "robust_score": 0.59, "dep": 0.11},
        ]
    )

    reranked = SCRIPT_09_MODULE.apply_dynamics_lite_rerank(
        root=tmp_path,
        case_stage6_root=tmp_path / "stage6",
        case_context={"stage6_root": str(tmp_path / "stage6")},
        stage5={},
        stage6=stage6,
        robust_frame=robust_frame,
        reward_column="reranked_objective_reward",
        top_n=2,
    )

    assert reranked.iloc[0]["candidate_id"] == "cand_b"
    assert "dynamics_lite_score" in reranked.columns
    assert reranked.loc[reranked["candidate_id"].eq("cand_b"), "dynamics_lite_score"].iloc[0] == 1.0


def test_apply_dep_focus_rerank_penalizes_high_dep_but_preserves_locked_top() -> None:
    robust_frame = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "lock",
                "reranked_objective_reward": 1.20,
                "objective_reward": 1.20,
                "robust_score": 0.52,
                "dep": 0.14,
                "hotspot_fraction": 0.52,
                "effective_compensation_gain": 0.02,
                "new_nonhotspot_score": 0.01,
                "keep_ifp_nonhotspot": 0.40,
            },
            {
                "candidate_id": "high_dep",
                "reranked_objective_reward": 1.16,
                "objective_reward": 1.16,
                "robust_score": 0.51,
                "dep": 0.28,
                "hotspot_fraction": 0.72,
                "effective_compensation_gain": 0.00,
                "new_nonhotspot_score": 0.00,
                "keep_ifp_nonhotspot": 0.20,
            },
            {
                "candidate_id": "low_dep_bonus",
                "reranked_objective_reward": 1.15,
                "objective_reward": 1.15,
                "robust_score": 0.50,
                "dep": 0.16,
                "hotspot_fraction": 0.48,
                "effective_compensation_gain": 0.10,
                "new_nonhotspot_score": 0.12,
                "keep_ifp_nonhotspot": 0.70,
            },
        ]
    )

    reranked = SCRIPT_09_MODULE.apply_dep_focus_rerank(
        robust_frame=robust_frame,
        case_context={"lead_candidate_id": "lock"},
        stage6={
            "objective_final_rerank": {
                "robust": {
                    "dep_focus": {
                        "enabled": True,
                        "lock_top_n": 1,
                        "apply_top_n": 3,
                        "dep_ceiling_quantile": 0.35,
                        "dep_lead_scale": 1.10,
                        "dep_penalty_weight": 1.0,
                        "hotspot_fraction_ceiling_quantile": 0.5,
                        "hotspot_fraction_lead_scale": 1.10,
                        "hotspot_fraction_penalty_weight": 0.3,
                        "compensation_bonus_weight": 0.2,
                        "nonhotspot_bonus_weight": 0.2,
                        "keep_ifp_nonhotspot_bonus_weight": 0.1,
                    }
                }
            }
        },
    )

    assert reranked.iloc[0]["candidate_id"] == "lock"
    assert reranked.iloc[1]["candidate_id"] == "low_dep_bonus"
    assert reranked.iloc[2]["candidate_id"] == "high_dep"
    assert "dep_focus_reranked_objective_reward" in reranked.columns
    high_dep_row = reranked.loc[reranked["candidate_id"].eq("high_dep")].iloc[0]
    assert high_dep_row["dep_focus_dep_penalty"] > 0.0


def test_merge_optional_rerank_columns_includes_dep_focus_fields() -> None:
    leaderboard = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "rob1",
                "objective_name": "robust",
                "objective_reward": 1.1,
            }
        ]
    )
    robust_frame = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "rob1",
                "reranked_objective_reward": 1.0,
                "dep_focus_reranked_objective_reward": 1.0,
                "dep_focus_dep_ceiling": 0.12,
                "dep_focus_hotspot_ceiling": 0.5,
                "dep_focus_dep_penalty": 0.1,
                "dep_focus_hotspot_penalty": 0.05,
                "dep_focus_bonus": 0.03,
            }
        ]
    )

    merged = SCRIPT_09_MODULE.merge_optional_rerank_columns(
        leaderboard=leaderboard,
        robust_frame=robust_frame,
    )

    assert merged.loc[0, "dep_focus_dep_penalty"] == 0.1
    assert merged.loc[0, "dep_focus_bonus"] == 0.03


def test_objective_sequence_honors_configured_order() -> None:
    assert SCRIPT_09_MODULE.objective_sequence({}) == ("robust", "naive")
    assert SCRIPT_09_MODULE.objective_sequence({"objective_execution_order": ["naive"]}) == ("naive", "robust")


def test_build_cross_objective_seed_rows_injects_valid_naive_hits_for_robust() -> None:
    config = stage6_config()
    config["beam_width"] = 3
    config["cross_objective_seed_injection"] = {
        "robust": {
            "source_objectives": ["naive"],
            "top_k": 8,
        }
    }
    prior_histories = {
        "naive": pd.DataFrame.from_records(
            [
                {
                    "candidate_id": "lead",
                    "smiles": "Oc1ccccc1",
                    "objective_name": "naive",
                    "objective_reward": 0.40,
                    "naive_objective_reward": 0.40,
                    "robust_objective_reward": 0.55,
                    "robust_score": 0.40,
                    "naive_mean_affinity": 0.52,
                    "keep_ifp": 0.90,
                    "dep": 0.30,
                    "candidate_valid": True,
                    "round_index": 0,
                    "action_sequence_json": "[]",
                },
                {
                    "candidate_id": "cand_best",
                    "smiles": "Fc1ccccc1O",
                    "objective_name": "naive",
                    "objective_reward": 0.62,
                    "naive_objective_reward": 0.62,
                    "robust_objective_reward": 1.12,
                    "robust_score": 0.62,
                    "naive_mean_affinity": 0.70,
                    "keep_ifp": 0.88,
                    "dep": 0.12,
                    "candidate_valid": True,
                    "round_index": 6,
                    "action_sequence_json": json.dumps([{"edit_family": "ADD", "fragment": "F"}]),
                },
                {
                    "candidate_id": "cand_second",
                    "smiles": "Clc1ccccc1O",
                    "objective_name": "naive",
                    "objective_reward": 0.58,
                    "naive_objective_reward": 0.58,
                    "robust_objective_reward": 1.04,
                    "robust_score": 0.57,
                    "naive_mean_affinity": 0.67,
                    "keep_ifp": 0.87,
                    "dep": 0.15,
                    "candidate_valid": True,
                    "round_index": 5,
                    "action_sequence_json": json.dumps([{"edit_family": "REPLACE", "fragment": "Cl"}]),
                },
                {
                    "candidate_id": "cand_invalid",
                    "smiles": "Nc1ccccc1O",
                    "objective_name": "naive",
                    "objective_reward": 0.54,
                    "naive_objective_reward": 0.54,
                    "robust_objective_reward": 1.30,
                    "robust_score": 0.80,
                    "naive_mean_affinity": 0.60,
                    "keep_ifp": 0.84,
                    "dep": 0.25,
                    "candidate_valid": False,
                    "round_index": 7,
                    "action_sequence_json": json.dumps([{"edit_family": "ADD", "fragment": "N"}]),
                },
            ]
        )
    }

    seed_rows = SCRIPT_09_MODULE.build_cross_objective_seed_rows(
        objective_name="robust",
        prior_histories=prior_histories,
        case_context={
            "lead_candidate_id": "lead",
            "anchor_residues": ["A:LYS721"],
            "panel_weights": {},
            "scoring_policy": {"uncertainty_heavy": False},
        },
        stage6=config,
    )

    assert [row["candidate_id"] for row in seed_rows] == ["cand_best", "cand_second"]
    assert all(row["objective_name"] == "robust" for row in seed_rows)
    assert all(row["seed_injected"] is True for row in seed_rows)
    assert all(row["seed_source_objective"] == "naive" for row in seed_rows)
    assert seed_rows[0]["objective_reward"] == prior_histories["naive"].iloc[1]["robust_objective_reward"]
    assert seed_rows[0]["round_index"] == 0
    assert seed_rows[0]["seed_source_round_index"] == 6


def test_build_objective_guardrails_uses_seed_floor_minima_for_robust() -> None:
    guardrails = SCRIPT_09_MODULE.build_objective_guardrails(
        objective_name="robust",
        seed_rows=[
            {"candidate_id": "a", "robust_score": 0.52, "s_wt": 0.61},
            {"candidate_id": "b", "robust_score": 0.49, "s_wt": 0.54},
        ],
        stage6={
            "cross_objective_seed_injection": {
                "robust": {
                    "reward_guardrails": {
                        "enabled": True,
                        "robust_score_floor_quantile": 0.0,
                        "s_wt_floor_quantile": 0.0,
                        "robust_score_penalty_weight": 2.0,
                        "s_wt_penalty_weight": 1.5,
                        "disable_compensation_below_floor": True,
                    }
                }
            }
        },
    )

    assert guardrails["robust_score_floor"] == 0.49
    assert guardrails["s_wt_floor"] == 0.54
    assert guardrails["seed_count"] == 2
    assert guardrails["disable_compensation_below_floor"] is True


def test_apply_seed_dep_rerank_uses_seed_dep_ceiling() -> None:
    robust_frame = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "seed_low_dep",
                "reranked_objective_reward": 1.08,
                "objective_reward": 1.08,
                "robust_score": 0.50,
                "dep": 0.15,
                "seed_injected": True,
            },
            {
                "candidate_id": "seed_high_dep",
                "reranked_objective_reward": 1.07,
                "objective_reward": 1.07,
                "robust_score": 0.51,
                "dep": 0.19,
                "seed_injected": True,
            },
            {
                "candidate_id": "nonseed",
                "reranked_objective_reward": 1.09,
                "objective_reward": 1.09,
                "robust_score": 0.52,
                "dep": 0.20,
                "seed_injected": False,
            },
        ]
    )

    reranked = SCRIPT_09_MODULE.apply_seed_dep_rerank(
        robust_frame=robust_frame,
        stage6={
            "objective_final_rerank": {
                "robust": {
                    "enabled": True,
                    "dep_ceiling_quantile": 0.5,
                    "dep_penalty_weight": 2.0,
                }
            }
        },
    )

    assert "dep_reranked_objective_reward" in reranked.columns
    assert "seed_dep_ceiling" in reranked.columns
    assert "dep_rerank_penalty" in reranked.columns
    nonseed_row = reranked.loc[reranked["candidate_id"].eq("nonseed")].iloc[0]
    assert nonseed_row["dep_rerank_penalty"] > 0.0
    assert nonseed_row["dep_reranked_objective_reward"] < 1.09


def test_apply_scaffold_diversity_tail_rerank_preserves_prefix_and_diversifies_tail() -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "candidate_id": f"lock{index}",
                "reranked_objective_reward": 2.0 - index * 0.01,
                "objective_reward": 2.0 - index * 0.01,
                "robust_score": 0.60 - index * 0.001,
                "dep": 0.15,
                "scaffold_smiles": "scaf_lock",
            }
        )
    rows.extend(
        [
            {
                "candidate_id": "tail_a1",
                "reranked_objective_reward": 0.90,
                "objective_reward": 0.90,
                "robust_score": 0.50,
                "dep": 0.16,
                "scaffold_smiles": "scaf_a",
            },
            {
                "candidate_id": "tail_a2",
                "reranked_objective_reward": 0.89,
                "objective_reward": 0.89,
                "robust_score": 0.49,
                "dep": 0.16,
                "scaffold_smiles": "scaf_a",
            },
            {
                "candidate_id": "tail_b",
                "reranked_objective_reward": 0.88,
                "objective_reward": 0.88,
                "robust_score": 0.48,
                "dep": 0.16,
                "scaffold_smiles": "scaf_b",
            },
            {
                "candidate_id": "tail_c",
                "reranked_objective_reward": 0.87,
                "objective_reward": 0.87,
                "robust_score": 0.47,
                "dep": 0.16,
                "scaffold_smiles": "scaf_c",
            },
        ]
    )
    reranked = SCRIPT_09_MODULE.apply_scaffold_diversity_tail_rerank(
        robust_frame=pd.DataFrame.from_records(rows),
        stage6={
            "objective_final_rerank": {
                "robust": {
                    "scaffold_tail_diversity": {
                        "enabled": True,
                        "lock_top_n": 20,
                        "penalty_weight": 0.2,
                    }
                }
            }
        },
    )

    assert reranked.head(20)["candidate_id"].tolist() == [f"lock{i}" for i in range(20)]
    assert reranked.iloc[20]["candidate_id"] == "tail_a1"
    assert reranked.iloc[21]["candidate_id"] == "tail_b"
    assert "scaffold_diversity_rank" in reranked.columns
    assert "scaffold_diversity_reranked_objective_reward" in reranked.columns


def test_select_diverse_proposals_penalizes_beam_scaffold_repeats() -> None:
    proposals = [
        {
            "candidate_id": "repeat1",
            "parent_candidate_id": "p1",
            "edit_family": "SMALL_POLAR_SCAN",
            "pattern": "small_polar_scan",
            "scaffold_smiles": "scaf_repeat",
            "action_label": "SMALL_POLAR_SCAN(O)@1",
            "base_rank_score": 1.0,
            "preview_prefilter_pass": True,
            "preview_admet_penalty": 0.0,
            "preview_warning_count": 0,
            "dynamic_transform_prior_score": 0.1,
            "transform_prior_score": 0.0,
            "template_priority_score": 1.0,
        },
        {
            "candidate_id": "novel1",
            "parent_candidate_id": "p2",
            "edit_family": "LINKER_HETERO_SCAN",
            "pattern": "linker_hetero_scan",
            "scaffold_smiles": "scaf_novel",
            "action_label": "LINKER_HETERO_SCAN(N)@2",
            "base_rank_score": 0.95,
            "preview_prefilter_pass": True,
            "preview_admet_penalty": 0.0,
            "preview_warning_count": 0,
            "dynamic_transform_prior_score": 0.05,
            "transform_prior_score": 0.0,
            "template_priority_score": 1.0,
        },
    ]
    beam_frame = pd.DataFrame.from_records(
        [
            {"candidate_id": "beam1", "scaffold_smiles": "scaf_repeat"},
            {"candidate_id": "beam2", "scaffold_smiles": "scaf_repeat"},
        ]
    )
    stage6 = stage6_config()
    stage6["proposal_beam_scaffold_repeat_penalty"] = 0.3
    stage6["proposal_scaffold_repeat_penalty"] = 0.3

    selected = SCRIPT_09_MODULE.select_diverse_proposals(
        proposals=proposals,
        beam_frame=beam_frame,
        templates=[],
        stage6=stage6,
        objective_name="robust",
        proposal_count=2,
    )

    assert selected[0]["candidate_id"] == "novel1"


def test_select_diverse_proposals_reserves_top_template_families() -> None:
    proposals = [
        {
            "candidate_id": "small_1",
            "parent_candidate_id": "p1",
            "edit_family": "SMALL_POLAR_SCAN",
            "pattern": "small_polar_scan",
            "scaffold_smiles": "scaf_a",
            "action_label": "SMALL_POLAR_SCAN(O)@1",
            "base_rank_score": 1.20,
            "preview_prefilter_pass": True,
            "preview_admet_penalty": 0.0,
            "preview_warning_count": 0,
            "dynamic_transform_prior_score": 0.15,
            "transform_prior_score": 0.10,
            "template_priority_score": 2.0,
        },
        {
            "candidate_id": "small_2",
            "parent_candidate_id": "p2",
            "edit_family": "SMALL_POLAR_SCAN",
            "pattern": "small_polar_scan",
            "scaffold_smiles": "scaf_b",
            "action_label": "SMALL_POLAR_SCAN(N)@2",
            "base_rank_score": 1.10,
            "preview_prefilter_pass": True,
            "preview_admet_penalty": 0.0,
            "preview_warning_count": 0,
            "dynamic_transform_prior_score": 0.12,
            "transform_prior_score": 0.08,
            "template_priority_score": 1.8,
        },
        {
            "candidate_id": "ring_1",
            "parent_candidate_id": "p3",
            "edit_family": "RING_EDIT",
            "pattern": "aza_scan_ring",
            "scaffold_smiles": "scaf_c",
            "action_label": "RING_EDIT(N)@20",
            "base_rank_score": 0.70,
            "preview_prefilter_pass": True,
            "preview_admet_penalty": 0.0,
            "preview_warning_count": 0,
            "dynamic_transform_prior_score": 0.02,
            "transform_prior_score": 0.01,
            "template_priority_score": 1.0,
        },
        {
            "candidate_id": "link_1",
            "parent_candidate_id": "p4",
            "edit_family": "LINKER_HETERO_SCAN",
            "pattern": "linker_hetero_scan",
            "scaffold_smiles": "scaf_d",
            "action_label": "LINKER_HETERO_SCAN(N)@4",
            "base_rank_score": 0.68,
            "preview_prefilter_pass": True,
            "preview_admet_penalty": 0.0,
            "preview_warning_count": 0,
            "dynamic_transform_prior_score": 0.02,
            "transform_prior_score": 0.01,
            "template_priority_score": 0.9,
        },
    ]
    templates = [
        {"edit_family": "SMALL_POLAR_SCAN", "pattern": "small_polar_scan", "fragment": "O", "priority": 1},
        {"edit_family": "RING_EDIT", "pattern": "aza_scan_ring", "fragment": "N", "priority": 2},
        {"edit_family": "LINKER_HETERO_SCAN", "pattern": "linker_hetero_scan", "fragment": "N", "priority": 3},
    ]
    stage6 = stage6_config()
    stage6["proposal_template_family_coverage_count"] = 3

    selected = SCRIPT_09_MODULE.select_diverse_proposals(
        proposals=proposals,
        beam_frame=pd.DataFrame(),
        templates=templates,
        stage6=stage6,
        objective_name="robust",
        proposal_count=3,
    )

    families = {row["edit_family"] for row in selected}
    assert {"SMALL_POLAR_SCAN", "RING_EDIT", "LINKER_HETERO_SCAN"} <= families


def test_select_diverse_proposals_family_coverage_ignores_scaffold_cap_for_reserved_slots() -> None:
    proposals = [
        {
            "candidate_id": "nm_1",
            "parent_candidate_id": "p1",
            "edit_family": "N_METHYL_SCAN",
            "pattern": "n_methyl_scan",
            "scaffold_smiles": "scaf_a",
            "action_label": "N_METHYL_SCAN(-CH3)@11",
            "base_rank_score": 1.20,
            "preview_prefilter_pass": True,
            "preview_admet_penalty": 0.0,
            "preview_warning_count": 0,
            "dynamic_transform_prior_score": 0.15,
            "transform_prior_score": 0.10,
            "template_priority_score": 2.0,
        },
        {
            "candidate_id": "ring_1",
            "parent_candidate_id": "p2",
            "edit_family": "RING_EDIT",
            "pattern": "aza_scan_ring",
            "scaffold_smiles": "scaf_b",
            "action_label": "RING_EDIT(N)@20",
            "base_rank_score": 1.10,
            "preview_prefilter_pass": True,
            "preview_admet_penalty": 0.0,
            "preview_warning_count": 0,
            "dynamic_transform_prior_score": 0.12,
            "transform_prior_score": 0.08,
            "template_priority_score": 1.8,
        },
        {
            "candidate_id": "small_1",
            "parent_candidate_id": "p3",
            "edit_family": "SMALL_POLAR_SCAN",
            "pattern": "small_polar_scan",
            "scaffold_smiles": "scaf_a",
            "action_label": "SMALL_POLAR_SCAN(O)@15",
            "base_rank_score": 1.00,
            "preview_prefilter_pass": True,
            "preview_admet_penalty": 0.0,
            "preview_warning_count": 0,
            "dynamic_transform_prior_score": 0.10,
            "transform_prior_score": 0.06,
            "template_priority_score": 1.6,
        },
        {
            "candidate_id": "link_1",
            "parent_candidate_id": "p4",
            "edit_family": "LINKER_HETERO_SCAN",
            "pattern": "linker_hetero_scan",
            "scaffold_smiles": "scaf_a",
            "action_label": "LINKER_HETERO_SCAN(N)@4",
            "base_rank_score": 0.95,
            "preview_prefilter_pass": True,
            "preview_admet_penalty": 0.0,
            "preview_warning_count": 0,
            "dynamic_transform_prior_score": 0.09,
            "transform_prior_score": 0.05,
            "template_priority_score": 1.5,
        },
    ]
    templates = [
        {"edit_family": "N_METHYL_SCAN", "pattern": "n_methyl_scan", "fragment": "-C", "priority": 1},
        {"edit_family": "RING_EDIT", "pattern": "aza_scan_ring", "fragment": "N", "priority": 2},
        {"edit_family": "SMALL_POLAR_SCAN", "pattern": "small_polar_scan", "fragment": "O", "priority": 3},
        {"edit_family": "LINKER_HETERO_SCAN", "pattern": "linker_hetero_scan", "fragment": "N", "priority": 4},
    ]
    stage6 = stage6_config()
    stage6["proposal_template_family_coverage_count"] = 4
    stage6["max_proposals_per_scaffold"] = 1

    selected = SCRIPT_09_MODULE.select_diverse_proposals(
        proposals=proposals,
        beam_frame=pd.DataFrame(),
        templates=templates,
        stage6=stage6,
        objective_name="robust",
        proposal_count=4,
    )

    assert {row["edit_family"] for row in selected} == {
        "N_METHYL_SCAN",
        "RING_EDIT",
        "SMALL_POLAR_SCAN",
        "LINKER_HETERO_SCAN",
    }


def test_select_diverse_proposals_relaxed_fill_adds_more_after_family_coverage() -> None:
    proposals = [
        {
            "candidate_id": "nm_1",
            "parent_candidate_id": "p1",
            "edit_family": "N_METHYL_SCAN",
            "pattern": "n_methyl_scan",
            "scaffold_smiles": "scaf_a",
            "action_label": "N_METHYL_SCAN(-CH3)@11",
            "base_rank_score": 1.20,
            "preview_prefilter_pass": True,
            "preview_admet_penalty": 0.0,
            "preview_warning_count": 0,
            "dynamic_transform_prior_score": 0.15,
            "transform_prior_score": 0.10,
            "template_priority_score": 2.0,
        },
        {
            "candidate_id": "ring_1",
            "parent_candidate_id": "p2",
            "edit_family": "RING_EDIT",
            "pattern": "aza_scan_ring",
            "scaffold_smiles": "scaf_b",
            "action_label": "RING_EDIT(N)@20",
            "base_rank_score": 1.10,
            "preview_prefilter_pass": True,
            "preview_admet_penalty": 0.0,
            "preview_warning_count": 0,
            "dynamic_transform_prior_score": 0.12,
            "transform_prior_score": 0.08,
            "template_priority_score": 1.8,
        },
        {
            "candidate_id": "small_1",
            "parent_candidate_id": "p3",
            "edit_family": "SMALL_POLAR_SCAN",
            "pattern": "small_polar_scan",
            "scaffold_smiles": "scaf_a",
            "action_label": "SMALL_POLAR_SCAN(O)@15",
            "base_rank_score": 1.00,
            "preview_prefilter_pass": True,
            "preview_admet_penalty": 0.0,
            "preview_warning_count": 0,
            "dynamic_transform_prior_score": 0.10,
            "transform_prior_score": 0.06,
            "template_priority_score": 1.6,
        },
        {
            "candidate_id": "link_1",
            "parent_candidate_id": "p4",
            "edit_family": "LINKER_HETERO_SCAN",
            "pattern": "linker_hetero_scan",
            "scaffold_smiles": "scaf_c",
            "action_label": "LINKER_HETERO_SCAN(N)@4",
            "base_rank_score": 0.95,
            "preview_prefilter_pass": True,
            "preview_admet_penalty": 0.0,
            "preview_warning_count": 0,
            "dynamic_transform_prior_score": 0.09,
            "transform_prior_score": 0.05,
            "template_priority_score": 1.5,
        },
        {
            "candidate_id": "small_2",
            "parent_candidate_id": "p5",
            "edit_family": "SMALL_POLAR_SCAN",
            "pattern": "small_polar_scan",
            "scaffold_smiles": "scaf_a",
            "action_label": "SMALL_POLAR_SCAN(N)@20",
            "base_rank_score": 0.90,
            "preview_prefilter_pass": True,
            "preview_admet_penalty": 0.0,
            "preview_warning_count": 0,
            "dynamic_transform_prior_score": 0.08,
            "transform_prior_score": 0.05,
            "template_priority_score": 1.4,
        },
    ]
    templates = [
        {"edit_family": "N_METHYL_SCAN", "pattern": "n_methyl_scan", "fragment": "-C", "priority": 1},
        {"edit_family": "RING_EDIT", "pattern": "aza_scan_ring", "fragment": "N", "priority": 2},
        {"edit_family": "SMALL_POLAR_SCAN", "pattern": "small_polar_scan", "fragment": "O", "priority": 3},
        {"edit_family": "LINKER_HETERO_SCAN", "pattern": "linker_hetero_scan", "fragment": "N", "priority": 4},
    ]
    stage6 = stage6_config()
    stage6["proposal_template_family_coverage_count"] = 4
    stage6["max_proposals_per_scaffold"] = 1
    stage6["proposal_relaxed_max_per_scaffold"] = 3

    selected = SCRIPT_09_MODULE.select_diverse_proposals(
        proposals=proposals,
        beam_frame=pd.DataFrame(),
        templates=templates,
        stage6=stage6,
        objective_name="robust",
        proposal_count=5,
    )

    assert len(selected) == 5
    families = [row["edit_family"] for row in selected]
    assert {"N_METHYL_SCAN", "RING_EDIT", "SMALL_POLAR_SCAN", "LINKER_HETERO_SCAN"} <= set(families)


def test_select_diverse_proposals_single_parent_does_not_bind_parent_cap() -> None:
    proposals = []
    for index, family in enumerate(
        [
            "N_METHYL_SCAN",
            "RING_EDIT",
            "SMALL_POLAR_SCAN",
            "LINKER_HETERO_SCAN",
            "SMALL_POLAR_SCAN",
            "SMALL_POLAR_SCAN",
        ],
        start=1,
    ):
        proposals.append(
            {
                "candidate_id": f"cand_{index}",
                "parent_candidate_id": "only_parent",
                "edit_family": family,
                "pattern": {
                    "N_METHYL_SCAN": "n_methyl_scan",
                    "RING_EDIT": "aza_scan_ring",
                    "SMALL_POLAR_SCAN": "small_polar_scan",
                    "LINKER_HETERO_SCAN": "linker_hetero_scan",
                }[family],
                "scaffold_smiles": f"scaf_{index}",
                "action_label": f"{family}@{index}",
                "base_rank_score": 1.2 - index * 0.05,
                "preview_prefilter_pass": True,
                "preview_admet_penalty": 0.0,
                "preview_warning_count": 0,
                "dynamic_transform_prior_score": 0.1,
                "transform_prior_score": 0.05,
                "template_priority_score": 1.0,
            }
        )
    templates = [
        {"edit_family": "N_METHYL_SCAN", "pattern": "n_methyl_scan", "fragment": "-C", "priority": 1},
        {"edit_family": "RING_EDIT", "pattern": "aza_scan_ring", "fragment": "N", "priority": 2},
        {"edit_family": "SMALL_POLAR_SCAN", "pattern": "small_polar_scan", "fragment": "O", "priority": 3},
        {"edit_family": "LINKER_HETERO_SCAN", "pattern": "linker_hetero_scan", "fragment": "N", "priority": 4},
    ]
    stage6 = stage6_config()
    stage6["proposal_template_family_coverage_count"] = 4
    stage6["max_proposals_per_parent"] = 3
    stage6["proposal_relaxed_max_per_parent"] = 4

    selected = SCRIPT_09_MODULE.select_diverse_proposals(
        proposals=proposals,
        beam_frame=pd.DataFrame(),
        templates=templates,
        stage6=stage6,
        objective_name="robust",
        proposal_count=6,
    )

    assert len(selected) == 6


def test_select_diverse_proposals_relaxed_fill_respects_relaxed_family_cap() -> None:
    proposals = []
    for index in range(6):
        proposals.append(
            {
                "candidate_id": f"ring_{index}",
                "parent_candidate_id": "only_parent",
                "edit_family": "RING_EDIT",
                "pattern": "aza_scan_ring",
                "scaffold_smiles": f"scaf_ring_{index}",
                "action_label": f"RING_EDIT(N)@{index}",
                "base_rank_score": 1.2 - index * 0.05,
                "preview_prefilter_pass": True,
                "preview_admet_penalty": 0.0,
                "preview_warning_count": 0,
                "dynamic_transform_prior_score": 0.1,
                "transform_prior_score": 0.05,
                "template_priority_score": 1.0,
            }
        )
    proposals.extend(
        [
            {
                "candidate_id": "link_1",
                "parent_candidate_id": "only_parent",
                "edit_family": "LINKER_HETERO_SCAN",
                "pattern": "linker_hetero_scan",
                "scaffold_smiles": "scaf_link_1",
                "action_label": "LINKER_HETERO_SCAN(N)@1",
                "base_rank_score": 0.8,
                "preview_prefilter_pass": True,
                "preview_admet_penalty": 0.0,
                "preview_warning_count": 0,
                "dynamic_transform_prior_score": 0.1,
                "transform_prior_score": 0.05,
                "template_priority_score": 1.0,
            },
            {
                "candidate_id": "small_1",
                "parent_candidate_id": "only_parent",
                "edit_family": "SMALL_POLAR_SCAN",
                "pattern": "small_polar_scan",
                "scaffold_smiles": "scaf_small_1",
                "action_label": "SMALL_POLAR_SCAN(O)@1",
                "base_rank_score": 0.75,
                "preview_prefilter_pass": True,
                "preview_admet_penalty": 0.0,
                "preview_warning_count": 0,
                "dynamic_transform_prior_score": 0.1,
                "transform_prior_score": 0.05,
                "template_priority_score": 1.0,
            },
        ]
    )
    templates = [
        {"edit_family": "RING_EDIT", "pattern": "aza_scan_ring", "fragment": "N", "priority": 1},
        {"edit_family": "LINKER_HETERO_SCAN", "pattern": "linker_hetero_scan", "fragment": "N", "priority": 2},
        {"edit_family": "SMALL_POLAR_SCAN", "pattern": "small_polar_scan", "fragment": "O", "priority": 3},
    ]
    stage6 = stage6_config()
    stage6["proposal_template_family_coverage_count"] = 3
    stage6["proposal_relaxed_max_per_family"] = 3
    stage6["proposal_relaxed_max_per_parent"] = 10

    selected = SCRIPT_09_MODULE.select_diverse_proposals(
        proposals=proposals,
        beam_frame=pd.DataFrame(),
        templates=templates,
        stage6=stage6,
        objective_name="robust",
        proposal_count=6,
    )

    families = [row["edit_family"] for row in selected]
    assert families.count("RING_EDIT") <= 3


def test_apply_search_beam_diversity_keeps_prefix_and_promotes_novel_scaffold() -> None:
    beam_source = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "lock",
                "objective_reward": 1.00,
                "robust_score": 0.60,
                "keep_ifp": 0.80,
                "dep": 0.20,
                "scaffold_smiles": "scaf_a",
            },
            {
                "candidate_id": "repeat",
                "objective_reward": 0.96,
                "robust_score": 0.59,
                "keep_ifp": 0.79,
                "dep": 0.20,
                "scaffold_smiles": "scaf_a",
            },
            {
                "candidate_id": "novel",
                "objective_reward": 0.91,
                "robust_score": 0.58,
                "keep_ifp": 0.78,
                "dep": 0.20,
                "scaffold_smiles": "scaf_b",
            },
        ]
    )
    stage6 = stage6_config()
    stage6["search_beam_scaffold_penalty"] = 0.10
    stage6["search_beam_scaffold_novelty_bonus"] = 0.05

    selected = SCRIPT_09_MODULE.apply_search_beam_diversity(
        beam_source=beam_source,
        beam_width=2,
        stage6=stage6,
        objective_name="robust",
    )

    assert selected["candidate_id"].tolist() == ["lock", "novel"]


def test_objective_guardrail_adjustments_zero_compensation_below_seed_floor() -> None:
    adjustments = objective_guardrail_adjustments(
        objective_name="robust",
        robust_score=0.41,
        s_wt=0.42,
        compensation_gain=1.2,
        objective_guardrails={
            "enabled": True,
            "robust_score_floor": 0.49,
            "s_wt_floor": 0.54,
            "robust_score_penalty_weight": 2.0,
            "s_wt_penalty_weight": 2.0,
            "disable_compensation_below_floor": True,
        },
    )

    assert adjustments["objective_guardrail_active"] is True
    assert adjustments["objective_guardrail_floor_violation"] is True
    assert adjustments["effective_compensation_gain"] == 0.0
    assert adjustments["objective_guardrail_penalty"] > 0.0


def test_merge_optional_rerank_columns_tolerates_missing_dep_and_scaffold_columns() -> None:
    leaderboard = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "rob1",
                "objective_name": "robust",
                "objective_reward": 1.0,
                "robust_objective_reward": 1.0,
            },
            {
                "candidate_id": "nav1",
                "objective_name": "naive",
                "objective_reward": 0.8,
                "robust_objective_reward": 0.8,
            },
        ]
    )
    robust_frame = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "rob1",
                "objective_name": "robust",
                "reranked_objective_reward": 1.2,
                "wt_gnina_affinity_kcal_mol": -10.5,
                "wt_gnina_score": 0.7,
            }
        ]
    )

    merged = SCRIPT_09_MODULE.merge_optional_rerank_columns(
        leaderboard=leaderboard,
        robust_frame=robust_frame,
    )

    assert "dep_reranked_objective_reward" in merged.columns
    assert "scaffold_diversity_reranked_objective_reward" in merged.columns
    assert float(merged.loc[merged["candidate_id"].eq("rob1"), "objective_reward"].iloc[0]) == 1.2
    assert pd.isna(merged.loc[merged["candidate_id"].eq("rob1"), "dep_reranked_objective_reward"].iloc[0])


def test_validate_objective_plan_rejects_forward_reference_seed_injection() -> None:
    try:
        SCRIPT_09_MODULE.validate_objective_plan(
            {
                "objective_execution_order": ["robust", "naive"],
                "cross_objective_seed_injection": {
                    "robust": {
                        "source_objectives": ["naive"],
                    }
                },
            }
        )
    except ValueError as exc:
        assert "source objectives to run before their target" in str(exc)
    else:
        raise AssertionError("validate_objective_plan should reject seed sources that run after their target")


def test_base_stage6_case_override_keeps_seed_injection_and_wt_override() -> None:
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    abl_override = config["stage6"]["case_overrides"]["abl1_nilotinib"]

    assert abl_override["wt_hard_constraint_epsilon_kcal_mol"] == 0.75
    assert abl_override["proposal_count"] == 16
    assert abl_override["beam_width"] == 6
    assert abl_override["max_rounds"] == 6
    assert abl_override["uncertainty_heavy_site_panel_max_n"] == 8
    assert abl_override["receptor_ensemble_enabled"] is True
    assert abl_override["receptor_ensemble_max_members"] == 4
    assert abl_override["receptor_ensemble_runtime_max_members"] == 2
    assert abl_override["vina_seeds"] == [11]
    assert abl_override["objective_execution_order"] == ["naive", "robust"]
    assert abl_override["cross_objective_seed_injection"]["robust"]["source_objectives"] == ["naive"]
    assert abl_override["cross_objective_seed_injection"]["robust"]["reward_guardrails"]["enabled"] is True
    assert abl_override["objective_final_rerank"]["robust"]["dep_penalty_weight"] == 2.0
    assert abl_override["objective_final_rerank"]["robust"]["scaffold_tail_diversity"]["penalty_weight"] == 0.2


def test_load_yaml_rejects_duplicate_mapping_keys(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "stage6:\n"
        "  case_overrides: {}\n"
        "  case_overrides: {}\n",
        encoding="utf-8",
    )

    try:
        load_yaml(bad_yaml)
    except ValueError as exc:
        assert "Duplicate YAML key" in str(exc)
    else:
        raise AssertionError("load_yaml should reject duplicate YAML keys")


def test_standardize_reference_ligand_falls_back_to_unsanitized_parse(tmp_path, monkeypatch) -> None:
    input_sdf = tmp_path / "input.sdf"
    input_sdf.write_text("", encoding="utf-8")
    output_sdf = tmp_path / "standardized.sdf"

    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom_idx in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(atom_idx, (float(atom_idx), 0.0, 0.0))
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)

    def fake_load(path, sanitize=True):
        if sanitize:
            raise ValueError("synthetic sanitize failure")
        return Chem.Mol(molecule)

    monkeypatch.setattr(stage35_utils, "load_rdkit_molecule", fake_load)

    payload = standardize_reference_ligand(input_sdf, output_sdf)

    assert output_sdf.exists()
    assert payload["heavy_atom_count"] == 3
    assert payload["canonical_smiles"] == "CCO"


def test_nearby_protein_residue_labels_captures_partner_chain_contacts(tmp_path) -> None:
    protein_pdb = tmp_path / "protein_multichain.pdb"
    protein_pdb.write_text(
        "ATOM      1  N   MET A   1       0.000   0.000   8.000  1.00  0.00           N\n"
        "ATOM      2  N   GLU B 138       0.000   0.000   2.500  1.00  0.00           N\n"
        "END\n",
        encoding="utf-8",
    )

    labels = nearby_protein_residue_labels(
        protein_pdb,
        ["A", "B"],
        ligand_coords=[(0.0, 0.0, 0.0)],
        distance_cutoff_a=4.0,
    )

    assert "B:GLU138" in labels
    assert "A:MET1" not in labels


def test_dock_candidate_preserves_receptor_built_inside_output_root(tmp_path, monkeypatch) -> None:
    root = tmp_path
    output_root = root / "dock"
    output_root.mkdir(parents=True, exist_ok=True)
    receptor_pdb = output_root / "receptor_multichain_stage6.pdb"
    receptor_pdb.write_text(
        "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N\nEND\n",
        encoding="utf-8",
    )
    ligand_sdf = root / "ligand.sdf"
    ligand_sdf.write_text("", encoding="utf-8")

    monkeypatch.setattr(stage6_utils, "_stage6_docking_cache_is_healthy", lambda output_root: False)
    monkeypatch.setattr(stage6_utils, "_stage6_docking_config", lambda stage6: {"use_pdbfixer": False, "protein_prep_ph": 7.4})

    def fake_prepare_receptor_pdbqt(input_pdb, output_pdbqt, ph):
        text = Path(input_pdb).read_text(encoding="utf-8")
        assert "ATOM" in text
        raise RuntimeError("stop_after_receptor_check")

    monkeypatch.setattr(stage6_utils, "prepare_receptor_pdbqt", fake_prepare_receptor_pdbqt)

    summary = _dock_candidate(
        root=root,
        receptor_pdb=receptor_pdb,
        docking_box={"center_x": 0.0, "center_y": 0.0, "center_z": 0.0, "size_x": 10.0, "size_y": 10.0, "size_z": 10.0},
        ligand_input_sdf=ligand_sdf,
        output_root=output_root,
        stage6={},
        hiv_reference=None,
    )

    assert summary["docking_status"] == "failed"
    assert "stop_after_receptor_check" in str(summary["docking_error"])
    assert receptor_pdb.exists()


def test_case_specific_action_hints_favor_partner_chain_rt_context() -> None:
    hints = case_specific_action_hints(
        case_entry={
            "case_id": "hiv_rt_rilpivirine",
            "target_domain": "rt",
            "evaluation_unit": "observed_combo",
        },
        lead_descriptors={"rotatable_bonds": 7},
        pocket_profile_payload=pocket_profile(
            residue_labels=["A:LYS101", "A:TYR181", "B:GLU138"],
            partner_chain_residues=["B:GLU138"],
        ),
        partner_chain_positions=[138],
    )

    families = {row["edit_family"] for row in hints}
    assert {"SMALL_POLAR_SCAN", "LINKER_HETERO_SCAN", "BACKBONE_SEEKER", "WATER_BRIDGE_PROBE", "RIGIDIFY_LINKER"} <= families


def test_target_box_falls_back_to_unsanitized_standardized_ligand(tmp_path, monkeypatch) -> None:
    sample_root = tmp_path / "sample"
    sample_root.mkdir(parents=True, exist_ok=True)
    standardized = tmp_path / "work" / "reference_ligand.sdf"

    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom_idx in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(atom_idx, (float(atom_idx), 1.0, -1.0))
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)

    monkeypatch.setattr(stage6_utils, "stage5_reference_ligand_input", lambda sample_root, work_root: sample_root / "ligand.sdf")

    def fake_standardize(input_sdf, output_sdf):
        output_sdf.parent.mkdir(parents=True, exist_ok=True)
        output_sdf.write_text("placeholder", encoding="utf-8")
        return {"path": str(output_sdf), "heavy_atom_count": 3, "canonical_smiles": ""}

    def fake_load(path, sanitize=True):
        if path == standardized and sanitize:
            raise ValueError("synthetic sanitize failure")
        return Chem.Mol(molecule)

    monkeypatch.setattr(stage6_utils, "standardize_reference_ligand", fake_standardize)
    monkeypatch.setattr(stage6_utils, "load_rdkit_molecule", fake_load)

    box = _target_box(sample_root, tmp_path / "work", stage6_config())

    assert box["source"] == "stage6_reference_ligand"
    assert box["size_x"] > 0.0
    assert box["size_y"] > 0.0
    assert box["size_z"] > 0.0


def test_wt_receptor_pdb_regenerates_invalid_cached_file(tmp_path, monkeypatch) -> None:
    stage3_5_root = tmp_path / "stage3_5"
    stage3_5_root.mkdir(parents=True, exist_ok=True)
    source_complex = stage3_5_root / "wt_complex.pdb"
    source_complex.write_text("ATOM      1  N   MET A   1      0.0  0.0  0.0  1.00  0.00           N\nEND\n", encoding="utf-8")
    cached = stage3_5_root / "wt_receptor_stage6.pdb"
    cached.write_text("END\n", encoding="utf-8")

    def fake_save_chain_protein(source_pdb, chain_id, output_pdb):
        if str(chain_id) == "A":
            output_pdb.write_text("ATOM      1  N   MET A   1      0.0  0.0  0.0  1.00  0.00           N\nEND\n", encoding="utf-8")
        else:
            output_pdb.write_text("END\n", encoding="utf-8")

    monkeypatch.setattr(stage6_utils, "save_chain_protein", fake_save_chain_protein)
    monkeypatch.setattr(stage6_utils, "first_protein_chain_id", lambda path: "A")

    result = _wt_receptor_pdb({"stage3_5_root": str(stage3_5_root), "wt_chain_id": "C"})

    assert result == cached
    assert "ATOM" in cached.read_text(encoding="utf-8")


def test_stage6_oracle_v2_trains_and_predicts(tmp_path) -> None:
    root = tmp_path
    case_entries = [
        {"case_id": "demo_case", "target_name": "Demo", "target_domain": "kinase", "wt_template": {"chain_id": "A"}},
        {"case_id": "demo_peer", "target_name": "Peer", "target_domain": "kinase", "wt_template": {"chain_id": "A"}},
    ]
    for case_id, scale in [("demo_case", 1.0), ("demo_peer", 1.5)]:
        stage5_root = root / "outputs" / case_id / "stage5"
        stage5_root.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "case_id": case_id,
                "effect_scope": "site",
                "target_key": f"MUT{i}",
                "sample_source": "observed",
                "used_synthetic_combo_model": False,
                "delta_dock_kcal_mol": 0.1 * i * scale,
                "delta_gnina_affinity_kcal_mol": 0.05 * i * scale,
                "ifp_jaccard_loss": 0.02 * i,
                "ifp_occupancy_shift_mean_abs": 0.01 * i,
                "ifp_occupancy_anchor_loss": 0.01 * i,
                "anchor_loss_fraction": 0.03 * i,
                "pocket_volume_change_fraction": -0.01 * i,
                "solvent_proxy_shift": 0.01 * i,
                "stage4_local_rmsd_a": 0.1 * i,
                "high_uncertainty": False,
                "delta_mmgbsa_binding_kcal_mol": 0.12 * i * scale,
            }
            for i in range(1, 7)
        ]
        pd.DataFrame.from_records(records).to_csv(stage5_root / "ifp_diff.csv", index=False)
        pd.DataFrame.from_records(records).to_csv(stage5_root / "scoring_calibration.csv", index=False)

    metadata = fit_stage6_oracle_v2(
        root=root,
        case_entries=case_entries,
        focus_case_id="demo_case",
        stage6=stage6_config(),
    )
    oracle = load_stage6_oracle_v2(str(root / "outputs" / "demo_case" / "stage6_oracle_v2" / "stage6_oracle_v2.pkl"))
    prediction = oracle.predict(
        {
            "effect_scope": "site",
            "sample_source": "observed",
            "target_domain": "kinase",
            "is_partner_chain_sensitive": False,
            "used_synthetic_combo_model": False,
            "delta_dock_kcal_mol": 0.25,
            "delta_gnina_affinity_kcal_mol": 0.12,
            "ifp_jaccard_loss": 0.08,
            "ifp_occupancy_shift_mean_abs": 0.04,
            "ifp_occupancy_anchor_loss": 0.03,
            "anchor_loss_fraction": 0.05,
            "pocket_volume_change_fraction": -0.03,
            "solvent_proxy_shift": 0.02,
            "stage4_local_rmsd_a": 0.3,
        },
        scale=1.5,
    )

    assert metadata["trust_score"] > 0.0
    assert metadata["available_models"]
    assert metadata["ensemble_model"]["available"] is True
    assert oracle.available is True
    assert prediction["available"] is True
    assert prediction["pred_mean"] is not None
    assert prediction["pred_std"] is not None
    assert prediction["pred_conservative"] is not None
    assert 0.0 <= prediction["score_mean"] <= 1.0
    assert 0.0 <= prediction["score_conservative"] <= 1.0
    assert prediction["score_conservative"] <= prediction["score_mean"]


def test_oracle_v2_scoring_policy_uses_holdout_gate(tmp_path) -> None:
    case_root = tmp_path / "outputs" / "demo_case"
    oracle_root = case_root / "stage6_oracle_v2"
    oracle_root.mkdir(parents=True, exist_ok=True)
    (oracle_root / "stage6_oracle_v2.json").write_text(
        json.dumps({"trust_score": 0.90, "mode": "normal_robust_optimization"}),
        encoding="utf-8",
    )
    (oracle_root / "holdout_eval.json").write_text(
        json.dumps(
            {
                "case_metrics": {"trust_score": 0.49},
                "domain_metrics": {"trust_score": 0.52},
                "ensemble_metrics": {"trust_score": 0.51},
            }
        ),
        encoding="utf-8",
    )

    policy = _oracle_v2_scoring_policy(case_root=case_root, stage6=stage6_config())

    assert policy["oracle_v2_mode"] == "uncertainty_heavy"
    assert abs(policy["oracle_v2_effective_trust_score"] - 0.49) < 1e-9
    assert abs(policy["oracle_v2_ensemble_holdout_trust_score"] - 0.51) < 1e-9


def test_conservative_ensemble_holdout_uses_min_trust_score() -> None:
    payload = conservative_ensemble_holdout(
        case_metrics={"available": True, "trust_score": 0.62},
        domain_metrics={"available": True, "trust_score": 0.55},
        residual_weight=1.0,
    )

    assert payload["available"] is True
    assert payload["component_models"] == ["case", "domain"]
    assert abs(payload["trust_score"] - 0.55) < 1e-9


def test_build_receptor_ensemble_members_prefers_distinct_pdbs(tmp_path) -> None:
    mutation_pool_root = tmp_path / "outputs" / "case_manifests"
    mutation_pool_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(
        [
            {
                "case_id": "abl1_nilotinib",
                "effect_scope": "site",
                "mutation_key": "ABL1:T315I",
                "sample_id": "S1",
                "pdb_id": "3CS9",
                "selection_bucket": "TopRisk20",
                "risk_score": 0.9,
            },
            {
                "case_id": "abl1_nilotinib",
                "effect_scope": "site",
                "mutation_key": "ABL1:T315I",
                "sample_id": "S2",
                "pdb_id": "2FO0",
                "selection_bucket": "MediumRisk80",
                "risk_score": 0.7,
            },
            {
                "case_id": "abl1_nilotinib",
                "effect_scope": "site",
                "mutation_key": "ABL1:T315I",
                "sample_id": "S3",
                "pdb_id": "3CS9",
                "selection_bucket": "Tail100",
                "risk_score": 0.6,
            },
        ]
    ).to_csv(mutation_pool_root / "mutation_pool.csv", index=False)
    for sample_id in ["S1", "S2", "S3"]:
        sample_root = tmp_path / "outputs" / "structures" / sample_id
        sample_root.mkdir(parents=True, exist_ok=True)
        for filename in ["WT.pdb", "MT.pdb", "ligand.sdf", "WT_complex.pdb", "MT_complex.pdb"]:
            (sample_root / filename).write_text("X\n", encoding="utf-8")

    members = build_receptor_ensemble_members(
        root=tmp_path,
        case_entry={
            "case_id": "abl1_nilotinib",
            "seed_pdb_candidates": [
                {"pdb_id": "3CS9", "seed_rank": 1},
                {"pdb_id": "2FO0", "seed_rank": 2},
            ],
        },
        panel_frame=pd.DataFrame.from_records(
            [
                {
                    "effect_scope": "site",
                    "target_key": "ABL1:T315I",
                    "sample_root": str(tmp_path / "outputs" / "structures" / "S1"),
                    "sample_source": "observed_sample",
                    "representative_sample_id": "S1",
                    "stage5_selection_bucket": "primary_topk",
                }
            ]
        ),
        stage6={**stage6_config(), "receptor_ensemble_enabled": True, "receptor_ensemble_max_members": 3},
        mutation_pool=pd.read_csv(mutation_pool_root / "mutation_pool.csv"),
    )

    assert list(members["member_id"]) == ["S1", "S2"]
    assert list(members["pdb_id"]) == ["3CS9", "2FO0"]


def test_aggregate_ensemble_value_supports_median_and_cvar() -> None:
    values = [0.2, 0.5, 0.7, 0.9]

    assert abs(aggregate_ensemble_value(values, "median") - 0.6) < 1e-9
    assert abs(aggregate_ensemble_value(values, "cvar") - 0.35) < 1e-9


def test_reward_v2_layer_weights_are_case_specific() -> None:
    egfr = reward_v2_layer_weights(
        case_id="egfr_erlotinib",
        target_domain="kinase",
        stage6=stage6_config(),
        layer_available={name: True for name in ["keep_ifp_anchor", "keep_ifp_backbone", "keep_ifp_partner_chain", "keep_ifp_nonhotspot"]},
    )
    hiv = reward_v2_layer_weights(
        case_id="hiv_rt_rilpivirine",
        target_domain="rt",
        stage6=stage6_config(),
        layer_available={name: True for name in ["keep_ifp_anchor", "keep_ifp_backbone", "keep_ifp_partner_chain", "keep_ifp_nonhotspot"]},
    )

    assert egfr["keep_ifp_anchor"] > hiv["keep_ifp_anchor"]
    assert hiv["keep_ifp_partner_chain"] > egfr["keep_ifp_partner_chain"]
    assert abs(sum(egfr.values()) - 1.0) < 1e-9
    assert abs(sum(hiv.values()) - 1.0) < 1e-9


def test_reward_v2_components_penalize_oracle_uncertainty_and_hotspot_fraction() -> None:
    alt_anchor = alt_anchor_score(
        layer_scores={
            "keep_ifp_anchor": 0.4,
            "keep_ifp_backbone": 0.6,
            "keep_ifp_partner_chain": 0.3,
            "keep_ifp_nonhotspot": 0.7,
        },
        new_nonhotspot_score=3.0,
        compensation_gain=2.0,
        hotspot_fraction=0.2,
        stage6=stage6_config(),
    )
    oracle = oracle_uncertainty_score(
        effective_trust_score=0.55,
        pred_std_mean=1.5,
        stage6=stage6_config(),
    )
    reward = reward_v2_components(
        s_wt=0.8,
        robust_site_core=0.7,
        robust_combo_core=0.6,
        combo_dense_case=True,
        alt_anchor_score_value=alt_anchor["alt_anchor_score"],
        new_nonhotspot_score=alt_anchor["new_nonhotspot_score"],
        hotspot_fraction=0.2,
        oracle_uncertainty=oracle["oracle_uncertainty"],
        synth_penalty=0.1,
    )

    assert alt_anchor["alt_anchor_score"] > 0.0
    assert oracle["oracle_uncertainty"] > 0.0
    assert reward["reward_v2_raw"] > 0.0


def test_case_context_for_round_egfr_curriculum_limits_site_panel() -> None:
    case_context = {
        "case_id": "egfr_erlotinib",
        "target_domain": "kinase",
        "evaluation_unit": "site",
        "scoring_policy": {"oracle_v2_mode": "uncertainty_heavy"},
        "panel_rows_full": [
            {"effect_scope": "site", "target_key": f"S{i}", "stage4_rank": i, "risk_score": 1.0}
            for i in range(1, 17)
        ],
        "panel_rows": [],
        "panel_weights": {},
        "mechanism_summary": {},
        "action_space": {},
        "hotspot_residues": [],
    }

    round4 = stage6_utils.case_context_for_round(case_context=case_context, stage6=stage6_config(), round_index=4)
    round13 = stage6_utils.case_context_for_round(case_context=case_context, stage6=stage6_config(), round_index=13)

    assert len(round4["panel_rows"]) == 6
    assert round4["panel_curriculum"]["site_limit"] == 6
    assert len(round13["panel_rows"]) == 16
    assert round13["panel_curriculum"]["site_limit"] == 16


def test_case_context_for_round_hiv_curriculum_splits_site_and_combo() -> None:
    case_context = {
        "case_id": "hiv_rt_rilpivirine",
        "target_domain": "rt",
        "evaluation_unit": "observed_combo",
        "scoring_policy": {"oracle_v2_mode": "normal_robust_optimization"},
        "panel_rows_full": [
            *[
                {"effect_scope": "site", "target_key": f"S{i}", "stage4_rank": i, "risk_score": 1.0}
                for i in range(1, 13)
            ],
            *[
                {"effect_scope": "combo", "target_key": f"C{i}", "stage4_rank": i, "risk_score": 1.0}
                for i in range(1, 15)
            ],
        ],
        "panel_rows": [],
        "panel_weights": {},
        "mechanism_summary": {},
        "action_space": {},
        "hotspot_residues": [],
    }

    round1 = stage6_utils.case_context_for_round(case_context=case_context, stage6=stage6_config(), round_index=1)
    round9 = stage6_utils.case_context_for_round(case_context=case_context, stage6=stage6_config(), round_index=9)

    assert int(sum(1 for row in round1["panel_rows"] if row["effect_scope"] == "site")) == 4
    assert int(sum(1 for row in round1["panel_rows"] if row["effect_scope"] == "combo")) == 4
    assert int(sum(1 for row in round9["panel_rows"] if row["effect_scope"] == "site")) == 10
    assert int(sum(1 for row in round9["panel_rows"] if row["effect_scope"] == "combo")) == 12


def test_case_context_for_round_abl_holds_at_capped_panel_when_oracle_is_uncertainty_heavy() -> None:
    cfg = {**stage6_config(), "uncertainty_heavy_site_panel_max_n": 8}
    case_context = {
        "case_id": "abl1_nilotinib",
        "target_domain": "kinase",
        "evaluation_unit": "site",
        "scoring_policy": {"oracle_v2_mode": "uncertainty_heavy"},
        "panel_rows_full": [
            {"effect_scope": "site", "target_key": f"S{i}", "stage4_rank": i, "risk_score": 1.0}
            for i in range(1, 21)
        ],
        "panel_rows": [],
        "panel_weights": {},
        "mechanism_summary": {},
        "action_space": {},
        "hotspot_residues": [],
    }

    round1 = stage6_utils.case_context_for_round(case_context=case_context, stage6=cfg, round_index=1)
    round12 = stage6_utils.case_context_for_round(case_context=case_context, stage6=cfg, round_index=12)

    assert len(round1["panel_rows"]) == 8
    assert len(round12["panel_rows"]) == 8


def test_target_panel_ensemble_runtime_cap_limits_members(tmp_path, monkeypatch) -> None:
    candidate_sdf = tmp_path / "candidate.sdf"
    candidate_sdf.write_text("", encoding="utf-8")
    call_member_ids: list[str] = []

    def fake_target_box(sample_root, work_root, stage6):
        return {"source": "test"}

    def fake_target_receptor(row, case_context, output_root):
        return Path(str(row["sample_root"])) / "MT.pdb"

    def fake_dock_candidate(**kwargs):
        member_id = Path(kwargs["output_root"]).name
        call_member_ids.append(member_id)
        return {"docking_status": "ok"}

    monkeypatch.setattr(stage6_utils, "_target_box", fake_target_box)
    monkeypatch.setattr(stage6_utils, "_target_panel_receptor_pdb", fake_target_receptor)
    monkeypatch.setattr(stage6_utils, "_dock_candidate", fake_dock_candidate)

    row = {"target_key": "ABL1:T315I", "effect_scope": "site", "sample_root": str(tmp_path / "base")}
    case_context = {
        "receptor_ensemble_members": {
            "ABL1:T315I": [
                {"member_rank": 1, "member_id": "m1", "sample_root": str(tmp_path / "m1")},
                {"member_rank": 2, "member_id": "m2", "sample_root": str(tmp_path / "m2")},
                {"member_rank": 3, "member_id": "m3", "sample_root": str(tmp_path / "m3")},
            ]
        },
        "hiv_reference": None,
    }
    stage6 = {**stage6_config(), "receptor_ensemble_runtime_max_members": 2}

    _, _, summary = stage6_utils._dock_target_panel_row(
        root=tmp_path,
        row=row,
        candidate_sdf=candidate_sdf,
        case_context=case_context,
        stage6=stage6,
    )

    assert call_member_ids == ["m1", "m2"]
    assert summary["ensemble_member_count"] == 2
    assert summary["ensemble_member_cap"] == 2
