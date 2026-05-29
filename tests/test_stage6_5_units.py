#!/usr/bin/env python3

from __future__ import annotations

import json

import pandas as pd

from tools.stage5_utils import stable_target_slug
from tools.stage6_5_utils import (
    WT_REFERENCE_KEY,
    load_stage6_5_cached_pair_records,
    select_energy_targets,
    select_validation_molecules,
    select_validation_targets,
)


def test_select_validation_targets_for_noncombo_case_uses_wt_two_high_risk_and_one_moderate() -> None:
    manifest = {
        "site_pool": {
            "top_risk20": [
                {"rank": 1, "mutation_key": "EGFR:T790M", "representative_sample_id": "A", "known_hotspot": True},
                {"rank": 2, "mutation_key": "EGFR:G719S", "representative_sample_id": "B", "known_hotspot": True},
                {"rank": 12, "mutation_key": "EGFR:L861Q", "representative_sample_id": "C", "known_hotspot": True},
            ]
        }
    }

    rows = select_validation_targets(
        case_entry={"case_id": "egfr_erlotinib"},
        manifest=manifest,
        combo_panel=pd.DataFrame(),
        stage6_5={
            "validation_noncombo_high_risk_n": 2,
            "validation_noncombo_moderate_n": 1,
        },
    )

    assert [row["target_key"] for row in rows] == [WT_REFERENCE_KEY, "EGFR:T790M", "EGFR:G719S", "EGFR:L861Q"]
    assert [row["selection_bucket"] for row in rows] == [
        "wt_reference",
        "high_risk_single",
        "high_risk_single",
        "moderate_single",
    ]


def test_select_validation_targets_for_hiv_uses_hotspot_single_and_topcombo() -> None:
    manifest = {
        "site_pool": {
            "top_risk20": [
                {"rank": 1, "mutation_key": "GAG-POL_RT:M184V", "representative_sample_id": "A", "known_hotspot": False},
                {"rank": 2, "mutation_key": "GAG-POL_RT:E138K", "representative_sample_id": "B", "known_hotspot": True},
            ]
        }
    }
    combo_panel = pd.DataFrame.from_records(
        [
            {"case_id": "hiv_rt_rilpivirine", "combo_rank": 2, "combination_key": "GAG-POL_RT:K101H+G190A"},
            {"case_id": "hiv_rt_rilpivirine", "combo_rank": 1, "combination_key": "GAG-POL_RT:V106M+V179D"},
        ]
    )

    rows = select_validation_targets(
        case_entry={"case_id": "hiv_rt_rilpivirine"},
        manifest=manifest,
        combo_panel=combo_panel,
        stage6_5={"validation_combo_topcombo_n": 2},
    )

    assert [row["target_key"] for row in rows] == [
        WT_REFERENCE_KEY,
        "GAG-POL_RT:E138K",
        "GAG-POL_RT:V106M+V179D",
        "GAG-POL_RT:K101H+G190A",
    ]


def test_select_validation_molecules_deduplicates_wt_control_from_robust_best() -> None:
    leaderboard = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "robust1",
                "smiles": "CCO",
                "objective_name": "robust",
                "objective_reward": 2.0,
                "objective_rank": 1,
                "robust_score": 0.9,
                "s_wt": 0.95,
                "wt_affinity_kcal_mol": -10.0,
                "candidate_valid": True,
            },
            {
                "candidate_id": "wt2",
                "smiles": "CCC",
                "objective_name": "naive",
                "objective_reward": 1.0,
                "objective_rank": 2,
                "robust_score": 0.4,
                "s_wt": 0.90,
                "wt_affinity_kcal_mol": -9.5,
                "candidate_valid": True,
            },
            {
                "candidate_id": "robust3",
                "smiles": "CCN",
                "objective_name": "robust",
                "objective_reward": 1.5,
                "objective_rank": 3,
                "robust_score": 0.8,
                "s_wt": 0.85,
                "wt_affinity_kcal_mol": -9.0,
                "candidate_valid": True,
            },
        ]
    )

    molecules, energy_panel = select_validation_molecules(
        case_entry={"case_id": "egfr_erlotinib"},
        leaderboard=leaderboard,
        stage6_5={"energy_top_molecule_n": 4},
    )

    assert [row["molecule_label"] for row in molecules] == ["lead", "robust_best", "wt_best_control"]
    assert molecules[1]["candidate_id"] == "robust1"
    assert molecules[2]["candidate_id"] == "wt2"
    assert energy_panel[0]["candidate_id"] == "lead"
    assert {row["candidate_id"] for row in energy_panel} >= {"lead", "robust1", "wt2", "robust3"}


def test_select_validation_molecules_skips_probe_failing_candidates_for_control_and_energy() -> None:
    leaderboard = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "robust1",
                "smiles": "CCO",
                "objective_name": "robust",
                "objective_reward": 2.0,
                "objective_rank": 1,
                "robust_score": 0.9,
                "s_wt": 0.96,
                "wt_affinity_kcal_mol": -10.0,
                "candidate_valid": True,
            },
            {
                "candidate_id": "wt_bad",
                "smiles": "CCC",
                "objective_name": "robust",
                "objective_reward": 1.9,
                "objective_rank": 2,
                "robust_score": 0.8,
                "s_wt": 0.95,
                "wt_affinity_kcal_mol": -9.9,
                "candidate_valid": True,
            },
            {
                "candidate_id": "wt_good",
                "smiles": "CCN",
                "objective_name": "naive",
                "objective_reward": 1.2,
                "objective_rank": 3,
                "robust_score": 0.5,
                "s_wt": 0.94,
                "wt_affinity_kcal_mol": -9.8,
                "candidate_valid": True,
            },
            {
                "candidate_id": "energy_ok",
                "smiles": "CCCl",
                "objective_name": "robust",
                "objective_reward": 1.1,
                "objective_rank": 4,
                "robust_score": 0.4,
                "s_wt": 0.90,
                "wt_affinity_kcal_mol": -9.0,
                "candidate_valid": True,
            },
        ]
    )

    molecules, energy_panel = select_validation_molecules(
        case_entry={"case_id": "hiv_rt_rilpivirine"},
        leaderboard=leaderboard,
        stage6_5={"energy_top_molecule_n": 4},
        candidate_filter=lambda row: str(row["candidate_id"]) != "wt_bad",
    )

    assert [row["candidate_id"] for row in molecules] == ["lead", "robust1", "wt_good"]
    assert {row["candidate_id"] for row in energy_panel} == {"lead", "robust1", "wt_good", "energy_ok"}


def test_select_validation_molecules_uses_stage6_5_history_to_avoid_failure_only_candidate() -> None:
    leaderboard = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "robust1",
                "smiles": "CCO",
                "objective_name": "robust",
                "objective_reward": 2.0,
                "objective_rank": 1,
                "robust_score": 0.9,
                "s_wt": 0.96,
                "wt_affinity_kcal_mol": -10.0,
                "candidate_valid": True,
                "stage6_5_prior_success_flag": True,
                "stage6_5_prior_failure_only": False,
            },
            {
                "candidate_id": "wt_failed",
                "smiles": "CCC",
                "objective_name": "robust",
                "objective_reward": 1.9,
                "objective_rank": 2,
                "robust_score": 0.8,
                "s_wt": 0.95,
                "wt_affinity_kcal_mol": -9.9,
                "candidate_valid": True,
                "stage6_5_prior_success_flag": False,
                "stage6_5_prior_failure_only": True,
            },
            {
                "candidate_id": "wt_stable",
                "smiles": "CCN",
                "objective_name": "robust",
                "objective_reward": 1.5,
                "objective_rank": 3,
                "robust_score": 0.7,
                "s_wt": 0.70,
                "wt_affinity_kcal_mol": -9.3,
                "candidate_valid": True,
                "stage6_5_prior_success_flag": True,
                "stage6_5_prior_failure_only": False,
            },
            {
                "candidate_id": "untested",
                "smiles": "CCCl",
                "objective_name": "naive",
                "objective_reward": 1.0,
                "objective_rank": 4,
                "robust_score": 0.4,
                "s_wt": 0.92,
                "wt_affinity_kcal_mol": -9.0,
                "candidate_valid": True,
                "stage6_5_prior_success_flag": False,
                "stage6_5_prior_failure_only": False,
            },
        ]
    )

    molecules, energy_panel = select_validation_molecules(
        case_entry={"case_id": "hiv_rt_rilpivirine"},
        leaderboard=leaderboard,
        stage6_5={"energy_top_molecule_n": 4},
    )

    assert [row["candidate_id"] for row in molecules] == ["lead", "robust1", "wt_stable"]
    assert "wt_failed" not in {row["candidate_id"] for row in energy_panel}


def test_select_validation_molecules_prefers_mmgbsa_stable_control_over_higher_swt_unstable_candidate() -> None:
    leaderboard = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "robust1",
                "smiles": "CCO",
                "objective_name": "robust",
                "objective_reward": 2.0,
                "objective_rank": 1,
                "robust_score": 0.9,
                "s_wt": 0.96,
                "wt_affinity_kcal_mol": -10.0,
                "candidate_valid": True,
                "stage6_5_prior_success_flag": True,
                "stage6_5_prior_failure_only": False,
                "stage6_5_prior_md_success_pair_count": 4,
                "stage6_5_prior_mmgbsa_available_count": 5,
            },
            {
                "candidate_id": "wt_high_swt_but_unstable",
                "smiles": "CCC",
                "objective_name": "naive",
                "objective_reward": 0.5,
                "objective_rank": 10,
                "robust_score": 0.2,
                "s_wt": 0.95,
                "wt_affinity_kcal_mol": -9.8,
                "candidate_valid": True,
                "stage6_5_prior_success_flag": True,
                "stage6_5_prior_failure_only": False,
                "stage6_5_prior_md_success_pair_count": 2,
                "stage6_5_prior_mmgbsa_available_count": 0,
            },
            {
                "candidate_id": "wt_mmgbsa_stable",
                "smiles": "CCN",
                "objective_name": "robust",
                "objective_reward": 1.4,
                "objective_rank": 3,
                "robust_score": 0.6,
                "s_wt": 0.70,
                "wt_affinity_kcal_mol": -9.2,
                "candidate_valid": True,
                "stage6_5_prior_success_flag": True,
                "stage6_5_prior_failure_only": False,
                "stage6_5_prior_md_success_pair_count": 0,
                "stage6_5_prior_mmgbsa_available_count": 5,
            },
            {
                "candidate_id": "energy_ok",
                "smiles": "CCCl",
                "objective_name": "robust",
                "objective_reward": 1.1,
                "objective_rank": 4,
                "robust_score": 0.4,
                "s_wt": 0.69,
                "wt_affinity_kcal_mol": -9.0,
                "candidate_valid": True,
                "stage6_5_prior_success_flag": False,
                "stage6_5_prior_failure_only": False,
                "stage6_5_prior_md_success_pair_count": 0,
                "stage6_5_prior_mmgbsa_available_count": 0,
            },
        ]
    )

    molecules, energy_panel = select_validation_molecules(
        case_entry={"case_id": "hiv_rt_rilpivirine"},
        leaderboard=leaderboard,
        stage6_5={"energy_top_molecule_n": 4},
    )

    assert [row["candidate_id"] for row in molecules] == ["lead", "robust1", "wt_mmgbsa_stable"]
    assert "wt_high_swt_but_unstable" not in {row["candidate_id"] for row in energy_panel[:3]}


def test_select_validation_molecules_applies_case_specific_energy_blocklist_and_priority() -> None:
    leaderboard = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "robust1",
                "smiles": "CCO",
                "objective_name": "robust",
                "objective_reward": 2.0,
                "objective_rank": 1,
                "robust_score": 0.9,
                "s_wt": 0.96,
                "wt_affinity_kcal_mol": -10.0,
                "candidate_valid": True,
            },
            {
                "candidate_id": "blocked_energy",
                "smiles": "CCC",
                "objective_name": "robust",
                "objective_reward": 1.9,
                "objective_rank": 2,
                "robust_score": 0.8,
                "s_wt": 0.92,
                "wt_affinity_kcal_mol": -9.8,
                "candidate_valid": True,
            },
            {
                "candidate_id": "preferred_energy",
                "smiles": "CCN",
                "objective_name": "robust",
                "objective_reward": 1.3,
                "objective_rank": 5,
                "robust_score": 0.6,
                "s_wt": 0.70,
                "wt_affinity_kcal_mol": -9.2,
                "candidate_valid": True,
            },
            {
                "candidate_id": "fallback_energy",
                "smiles": "CCCl",
                "objective_name": "robust",
                "objective_reward": 1.4,
                "objective_rank": 4,
                "robust_score": 0.7,
                "s_wt": 0.71,
                "wt_affinity_kcal_mol": -9.3,
                "candidate_valid": True,
            },
            {
                "candidate_id": "wt_control",
                "smiles": "CCBr",
                "objective_name": "naive",
                "objective_reward": 0.5,
                "objective_rank": 20,
                "robust_score": 0.2,
                "s_wt": 0.80,
                "wt_affinity_kcal_mol": -9.1,
                "candidate_valid": True,
            },
        ]
    )

    molecules, energy_panel = select_validation_molecules(
        case_entry={"case_id": "hiv_rt_rilpivirine"},
        leaderboard=leaderboard,
        stage6_5={
            "energy_top_molecule_n": 4,
            "case_candidate_overrides": {
                "hiv_rt_rilpivirine": {
                    "blocked_energy_candidate_ids": ["blocked_energy"],
                    "preferred_energy_candidate_ids": ["preferred_energy"],
                }
            },
        },
    )

    assert [row["candidate_id"] for row in molecules] == ["lead", "robust1", "blocked_energy"]
    assert [row["candidate_id"] for row in energy_panel] == ["lead", "preferred_energy", "robust1", "fallback_energy"]


def test_select_validation_molecules_applies_case_specific_validation_preferences() -> None:
    leaderboard = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "robust_default",
                "smiles": "CCO",
                "objective_name": "robust",
                "objective_reward": 2.0,
                "objective_rank": 1,
                "robust_score": 0.9,
                "s_wt": 0.96,
                "wt_affinity_kcal_mol": -10.0,
                "candidate_valid": True,
            },
            {
                "candidate_id": "robust_preferred",
                "smiles": "CCN",
                "objective_name": "robust",
                "objective_reward": 1.4,
                "objective_rank": 5,
                "robust_score": 0.5,
                "s_wt": 0.75,
                "wt_affinity_kcal_mol": -9.0,
                "candidate_valid": True,
            },
            {
                "candidate_id": "wt_default",
                "smiles": "CCCl",
                "objective_name": "naive",
                "objective_reward": 0.5,
                "objective_rank": 20,
                "robust_score": 0.2,
                "s_wt": 0.92,
                "wt_affinity_kcal_mol": -9.5,
                "candidate_valid": True,
            },
            {
                "candidate_id": "wt_preferred",
                "smiles": "CCBr",
                "objective_name": "robust",
                "objective_reward": 1.0,
                "objective_rank": 10,
                "robust_score": 0.3,
                "s_wt": 0.70,
                "wt_affinity_kcal_mol": -8.8,
                "candidate_valid": True,
            },
        ]
    )

    molecules, _ = select_validation_molecules(
        case_entry={"case_id": "hiv_rt_rilpivirine"},
        leaderboard=leaderboard,
        stage6_5={
            "energy_top_molecule_n": 4,
            "case_candidate_overrides": {
                "hiv_rt_rilpivirine": {
                    "preferred_robust_best_candidate_ids": ["robust_preferred"],
                    "preferred_wt_control_candidate_ids": ["wt_preferred"],
                }
            },
        },
    )

    assert [row["candidate_id"] for row in molecules] == ["lead", "robust_preferred", "wt_preferred"]


def test_select_energy_targets_prefers_hiv_single_plus_combo_panel() -> None:
    manifest = {
        "site_pool": {
            "top_risk20": [
                {"rank": 1, "mutation_key": "GAG-POL_RT:M184V", "representative_sample_id": "A", "known_hotspot": False},
                {"rank": 2, "mutation_key": "GAG-POL_RT:E138K", "representative_sample_id": "B", "known_hotspot": True},
            ]
        }
    }
    combo_panel = pd.DataFrame.from_records(
        [
            {"case_id": "hiv_rt_rilpivirine", "combo_rank": 1, "combination_key": "combo1", "count": 10},
            {"case_id": "hiv_rt_rilpivirine", "combo_rank": 2, "combination_key": "combo2", "count": 9},
            {"case_id": "hiv_rt_rilpivirine", "combo_rank": 3, "combination_key": "combo3", "count": 8},
        ]
    )

    targets = select_energy_targets(
        case_entry={"case_id": "hiv_rt_rilpivirine"},
        manifest=manifest,
        combo_panel=combo_panel,
        stage6_5={"energy_top_target_n": 3},
    )

    assert [row["target_key"] for row in targets] == ["GAG-POL_RT:E138K", "combo1", "combo2"]


def test_load_stage6_5_cached_pair_records_marks_missing_pairs_failed(tmp_path) -> None:
    root = tmp_path
    stage6_5_root = root / "outputs" / "hiv_rt_rilpivirine" / "stage6_5"
    target = {"effect_scope": "site", "target_key": "GAG-POL_RT:E138K", "selection_bucket": "energy_single"}
    molecule = {"molecule_label": "energy_1", "molecule_role": "energy_panel", "candidate_id": "cand1"}
    pair_root = stage6_5_root / "md" / "systems" / "energy_1" / stable_target_slug("site", "GAG-POL_RT:E138K")
    pair_root.mkdir(parents=True, exist_ok=True)
    (pair_root / "pair_summary.json").write_text(
        json.dumps(
            {
                "case_id": "hiv_rt_rilpivirine",
                "target_key": "GAG-POL_RT:E138K",
                "effect_scope": "site",
                "molecule_label": "energy_1",
                "molecule_role": "energy_panel",
                "candidate_id": "cand1",
                "stage5_run_root": str(pair_root.relative_to(root)),
                "stage5_status": "ok",
                "delta_mmgbsa_binding_kcal_mol": 1.25,
            }
        ),
        encoding="utf-8",
    )

    records = load_stage6_5_cached_pair_records(
        root=root,
        case_id="hiv_rt_rilpivirine",
        stage6_5_root=stage6_5_root,
        targets=[target, {"effect_scope": "combo", "target_key": "combo_missing", "selection_bucket": "energy_combo"}],
        molecules=[molecule],
    )

    assert len(records) == 2
    assert records[0]["stage5_status"] == "ok"
    assert records[0]["delta_mmgbsa_binding_kcal_mol"] == 1.25
    assert records[1]["stage5_status"] == "failed"
    assert records[1]["stage5_error"] == "pair_summary_missing"
