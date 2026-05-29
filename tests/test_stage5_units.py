#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from agents.mutation_effect_agent import MutationEffectAgent
from tools import stage5_physics, stage5_utils
from tools.stage5_physics import (
    _mmpbsa_input_text,
    _md_mdin,
    _min_mdin,
    _prepare_ligand_parameter_files,
    amber_md_instability_reason,
    ensure_explicit_hydrogens_sdf,
    multi_score_consensus,
    normalize_special_receptor_residues,
    parse_fpocket_info,
    parse_mmpbsa_results,
    probe_ligand_parameterization,
    strip_hydrogens_from_pdb,
    template_pose_sdf,
    write_ligand_pose_pdb_from_sdf,
)
from tools.stage5_utils import _build_site_panel, calibration_frame, compute_ifp_effect_row, occupancy_shift_metrics


def _touch_stage5_ready_sample(sample_root: Path) -> None:
    sample_root.mkdir(parents=True, exist_ok=True)
    for name in ["WT.pdb", "MT.pdb", "ligand.sdf", "WT_complex.pdb", "MT_complex.pdb"]:
        (sample_root / name).write_text("stub\n", encoding="utf-8")


SCRIPT_08B_PATH = Path(__file__).resolve().parents[1] / "scripts" / "08b_calibrate_scoring.py"
SCRIPT_08B_SPEC = importlib.util.spec_from_file_location("stage5_calibrate_scoring", SCRIPT_08B_PATH)
assert SCRIPT_08B_SPEC is not None and SCRIPT_08B_SPEC.loader is not None
SCRIPT_08B_MODULE = importlib.util.module_from_spec(SCRIPT_08B_SPEC)
SCRIPT_08B_SPEC.loader.exec_module(SCRIPT_08B_MODULE)


def test_build_site_panel_preserves_modeled_sample_root_without_representative_sample(
    tmp_path: Path,
    monkeypatch,
) -> None:
    modeled_root = tmp_path / "outputs" / "egfr_erlotinib" / "stage5" / "modeled_samples" / "site_example"
    _touch_stage5_ready_sample(modeled_root)

    def fake_resolve_stage5_target_sample(**_: object) -> tuple[Path, str, bool, str]:
        return modeled_root, "stage5_modeled", True, "deletion"

    monkeypatch.setattr(stage5_utils, "resolve_stage5_target_sample", fake_resolve_stage5_target_sample)

    site_rank = pd.DataFrame(
        [
            {
                "mutation_key": "EGFR:E746_A750delELREA",
                "mutation_rank": 1,
                "representative_sample_id": "",
                "risk_score": 0.95,
                "impact_evidence_tier": "A",
            }
        ]
    )
    rows = _build_site_panel(
        root=tmp_path,
        case_id="egfr_erlotinib",
        site_rank=site_rank,
        stage5={"site_top_n": 1, "model_missing_structures": True},
        mutation_status=pd.DataFrame(columns=["sample_id", "eligible_for_stage5"]),
    )

    assert len(rows) == 1
    assert rows[0]["sample_source"] == "stage5_modeled"
    assert rows[0]["sample_root"] == str(modeled_root)
    assert rows[0]["stage5_ready"] is True
    assert rows[0]["used_stage5_modeled_sample"] is True
    assert rows[0]["stage5_model_kind"] == "deletion"


def test_stage5_for_case_applies_case_override_without_mutating_base() -> None:
    stage5 = {
        "site_top_n": 8,
        "agent_top_site_n": 6,
        "reward": {"alpha": 1.0},
        "case_overrides": {
            "abl1_nilotinib": {
                "site_top_n": 20,
                "reward": {"alpha": 2.0},
            }
        },
    }

    merged = stage5_utils.stage5_for_case(stage5, "abl1_nilotinib")

    assert merged["site_top_n"] == 20
    assert merged["agent_top_site_n"] == 6
    assert merged["reward"] == {"alpha": 2.0}
    assert "case_overrides" not in merged
    assert stage5["site_top_n"] == 8
    assert stage5["reward"] == {"alpha": 1.0}


def test_parse_fpocket_info_extracts_top_level_volumes(tmp_path: Path) -> None:
    info_path = tmp_path / "sample_info.txt"
    info_path.write_text(
        "\n".join(
            [
                "Pocket 1 :",
                "  Score : 12.0",
                "  Volume : 123.4",
                "",
                "Pocket 2 :",
                "  Volume : 98.7",
                "",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_fpocket_info(info_path)

    assert parsed == [
        {"pocket_id": 1, "volume_a3": 123.4},
        {"pocket_id": 2, "volume_a3": 98.7},
    ]


def test_parse_mmpbsa_results_reads_delta_total(tmp_path: Path) -> None:
    result_path = tmp_path / "FINAL_RESULTS_MMPBSA.dat"
    result_path.write_text(
        "\n".join(
            [
                "GENERALIZED BORN:",
                "DELTA TOTAL      -12.3456",
                "",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_mmpbsa_results(result_path)

    assert parsed == {"delta_total_kcal_mol": -12.3456}


def test_mmpbsa_input_text_uses_sander_backend_when_requested() -> None:
    payload = _mmpbsa_input_text({"mmgbsa_frame_interval": 2, "mmgbsa_igb": 8, "mmgbsa_saltcon": 0.15}, use_sander=True)

    assert "interval=2" in payload
    assert "igb=8" in payload
    assert "saltcon=0.150" in payload
    assert "use_sander=1" in payload


def test_multi_score_consensus_reports_resistance_like_agreement() -> None:
    payload = multi_score_consensus(
        {"vina": 1.2, "mmgbsa": 0.8, "gnina": 0.05},
        neutral_threshold=0.25,
        consensus_threshold=0.70,
    )

    assert payload == {
        "available_score_count": 3,
        "nonzero_direction_count": 2,
        "consensus_fraction": 1.0,
        "consensus_direction": "resistance_like",
        "high_uncertainty": False,
    }


def test_occupancy_shift_metrics_computes_mean_shift_and_anchor_loss() -> None:
    payload = occupancy_shift_metrics(
        wt_payload={"top_seed_count": 4, "occupancy_map": {"A": 1.0, "B": 0.5}},
        mt_payload={"top_seed_count": 4, "occupancy_map": {"A": 0.25, "C": 1.0}},
        anchor_labels={"A", "B"},
    )

    assert payload["wt_top_seed_count"] == 4
    assert payload["mt_top_seed_count"] == 4
    assert payload["ifp_occupancy_shift_mean_abs"] == 0.75
    assert payload["ifp_occupancy_anchor_loss"] == 0.625
    assert json.loads(payload["wt_ifp_occupancy_json"]) == {"A": 1.0, "B": 0.5}
    assert json.loads(payload["mt_ifp_occupancy_json"]) == {"A": 0.25, "C": 1.0}


def test_mutation_effect_agent_constrains_mechanism_labels_to_deterministic_payload() -> None:
    agent = MutationEffectAgent(client=object())
    prompt_input = {
        "site_effect_candidates": [
            {
                "mutation_key": "GAG-POL_RT:Y181I",
                "mechanism_labels_json": "[\"anchor_loss\"]",
            }
        ],
        "combo_effect_candidates": [
            {
                "combination_key": "GAG-POL_RT:Y181I+M184V",
                "mechanism_labels_json": "[\"pocket_rearrangement\"]",
                "epistasis_flag": "unresolved",
            }
        ],
    }
    payload = {
        "site_effects": [
            {
                "mutation_key": "Y181I",
                "mechanism_labels": ["steric_clash", "anchor_loss"],
            }
        ],
        "combo_effects": [
            {
                "combination_key": "Y181I+M184V",
                "mechanism_labels": ["steric_clash"],
                "epistasis_flag": "additive_like",
            }
        ],
    }

    payload = agent._canonicalize_payload_keys(payload, prompt_input)
    payload = agent._constrain_payload_to_deterministic_values(payload, prompt_input)

    assert payload["site_effects"][0]["mutation_key"] == "GAG-POL_RT:Y181I"
    assert payload["site_effects"][0]["mechanism_labels"] == ["anchor_loss"]
    assert payload["combo_effects"][0]["combination_key"] == "GAG-POL_RT:Y181I+M184V"
    assert payload["combo_effects"][0]["mechanism_labels"] == ["pocket_rearrangement"]
    assert payload["combo_effects"][0]["epistasis_flag"] == "unresolved"


def test_calibration_frame_tolerates_missing_scoring_columns() -> None:
    effect_frame = pd.DataFrame(
        [
            {
                "case_id": "hiv_rt_rilpivirine",
                "effect_scope": "site",
                "target_key": "GAG-POL_RT:Y181I",
                "representative_sample_id": "MdrDB000001",
                "delta_dock_kcal_mol": 1.0,
                "ifp_jaccard_loss": 0.2,
                "anchor_loss_fraction": 0.1,
                "pocket_volume_change_fraction": 0.0,
                "solvent_proxy_shift": 0.0,
            }
        ]
    )

    frame = calibration_frame(effect_frame=effect_frame, ddg_lookup=pd.DataFrame())

    assert "delta_mmgbsa_binding_kcal_mol" in frame.columns
    assert frame.empty


def test_configured_gpu_ids_prefers_env_override(monkeypatch) -> None:
    monkeypatch.setenv("RESISTGPT_STAGE5_GPU_IDS", "3,5")

    gpu_ids = SCRIPT_08B_MODULE.configured_gpu_ids({"local_sampling_gpu_ids": [2, 4, 6]})

    assert gpu_ids == [3, 5]
    monkeypatch.delenv("RESISTGPT_STAGE5_GPU_IDS")


def test_configured_max_workers_supports_env_override(monkeypatch) -> None:
    monkeypatch.setenv("RESISTGPT_STAGE5_MAX_WORKERS", "8")

    max_workers = SCRIPT_08B_MODULE.configured_max_workers(
        {"mmgbsa_max_parallel_jobs": 12},
        scoring_job_count=20,
        gpu_ids=[2, 3, 4],
    )

    assert max_workers == 8
    monkeypatch.delenv("RESISTGPT_STAGE5_MAX_WORKERS")


def test_run_scoring_job_degrades_mmgbsa_failure_to_high_uncertainty(monkeypatch) -> None:
    def fail_scoring_payload(**_: object) -> dict[str, object]:
        raise RuntimeError("MMPBSA.py parse failed: *************")

    monkeypatch.setattr(SCRIPT_08B_MODULE, "ensure_stage5_scoring_payload", fail_scoring_payload)

    payload = SCRIPT_08B_MODULE.run_scoring_job(
        {
            "case_id": "egfr_erlotinib",
            "effect_scope": "combo",
            "target_key": "EGFR:L747_E749delLRE+A750P",
            "docking_row": {"delta_dock_kcal_mol": 1.4},
            "stage5": {
                "multi_score_neutral_threshold_kcal_mol": 0.25,
                "multi_score_consensus_threshold": 0.7,
            },
            "root": "/tmp",
        }
    )

    assert payload["scoring_status"] == "ok"
    assert payload["delta_mmgbsa_binding_kcal_mol"] is None
    assert payload["consensus_direction"] == "resistance_like"
    assert payload["high_uncertainty"] is True
    assert "MMPBSA.py" in str(payload["scoring_error"])


def test_run_mutation_effect_agent_degrades_llm_failure_to_empty_payload() -> None:
    class DummyAgent:
        def build_prompt_input(self, **kwargs):
            return {"site_effect_candidates": [], "combo_effect_candidates": []}

        def run(self, **kwargs):
            raise RuntimeError("synthetic agent outage")

    site_effect_frame = pd.DataFrame.from_records(
        [
            {
                "ifp_status": "ok",
                "stage4_rank": 1,
                "target_key": "EGFR:T790M",
            }
        ]
    )
    combo_effect_frame = pd.DataFrame(columns=["ifp_status", "stage4_rank", "target_key"])

    agent_input, llm_payload, llm_record, agent_status, agent_error = SCRIPT_08B_MODULE.run_mutation_effect_agent(
        effect_agent=DummyAgent(),
        case_entry={"case_id": "egfr_erlotinib", "target_name": "EGFR", "drug_name": "Erlotinib"},
        site_effect_frame=site_effect_frame,
        combo_effect_frame=combo_effect_frame,
        qc_payload={},
        calibration_summary={"calibration_sample_count": 0},
    )

    assert agent_input == {"site_effect_candidates": [], "combo_effect_candidates": []}
    assert llm_payload["site_effects"] == []
    assert llm_payload["combo_effects"] == []
    assert llm_record is None
    assert agent_status == "failed"
    assert "synthetic agent outage" in str(agent_error)


def test_gnina_score_only_respects_cuda_visible_devices(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class DummyResult:
        stdout = "Affinity: -8.70\n"

    monkeypatch.setattr(stage5_physics, "command_exists", lambda name: True)

    def fake_checked_command(command, cwd=None, extra_env=None):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["extra_env"] = extra_env
        return DummyResult()

    monkeypatch.setattr(stage5_physics, "checked_command", fake_checked_command)

    payload = stage5_physics.gnina_score_only(
        receptor_pdb=tmp_path / "receptor.pdb",
        ligand_sdf=tmp_path / "ligand.sdf",
        work_root=tmp_path,
        cuda_visible_devices=5,
    )

    assert payload["available"] is True
    assert payload["affinity_kcal_mol"] == -8.7
    assert captured["extra_env"] == {"CUDA_VISIBLE_DEVICES": "5"}


def test_gnina_score_only_retries_on_cpu_for_gpu_runtime_error(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class DummyResult:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    monkeypatch.setattr(stage5_physics, "command_exists", lambda name: True)

    def fake_checked_command(command, cwd=None, extra_env=None):
        calls.append({"command": command, "cwd": cwd, "extra_env": extra_env})
        if len(calls) == 1:
            raise RuntimeError("Command failed (-6): gnina ... CUBLAS_STATUS_NOT_INITIALIZED")
        return DummyResult("Affinity: -7.25\n")

    monkeypatch.setattr(stage5_physics, "checked_command", fake_checked_command)

    payload = stage5_physics.gnina_score_only(
        receptor_pdb=tmp_path / "receptor.pdb",
        ligand_sdf=tmp_path / "ligand.sdf",
        work_root=tmp_path,
        cuda_visible_devices=1,
    )

    assert len(calls) == 2
    assert calls[0]["extra_env"] == {"CUDA_VISIBLE_DEVICES": "1"}
    assert calls[1]["extra_env"] == {"CUDA_VISIBLE_DEVICES": ""}
    assert payload["available"] is True
    assert payload["affinity_kcal_mol"] == -7.25
    assert payload["used_cpu_fallback"] is True


def test_ensure_stage5_relaxation_reruns_invalid_cached_relaxation(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    for pose_set in ["wt", "mt"]:
        prep_root = run_root / f"{pose_set}_physics" / "amber_prep"
        prep_root.mkdir(parents=True, exist_ok=True)
        (prep_root / "amber_prep_manifest.json").write_text(
            json.dumps({"complex_pdb": "complex.pdb", "receptor_pdb": "receptor.pdb", "ligand_sdf": "ligand.sdf"}),
            encoding="utf-8",
        )
        relaxation_root = run_root / f"{pose_set}_physics" / "relaxation"
        relaxation_root.mkdir(parents=True, exist_ok=True)
        (relaxation_root / "relaxation_manifest.json").write_text(
            json.dumps(
                {
                    "refined_complex_pdb": str(relaxation_root / "refined_complex.pdb"),
                    "refined_receptor_pdb": str(relaxation_root / "refined_receptor.pdb"),
                    "refined_ligand_sdf": str(relaxation_root / "refined_ligand.sdf"),
                }
            ),
            encoding="utf-8",
        )
        (relaxation_root / "refined_complex.pdb").write_text("END\n", encoding="utf-8")
        (relaxation_root / "refined_receptor.pdb").write_text("END\n", encoding="utf-8")
        (relaxation_root / "refined_ligand.sdf").write_text("", encoding="utf-8")

    rerun_calls: list[tuple[str, bool]] = []

    def fake_run_amber_relaxation(*, amber_prep, work_root, stage5, run_local_sampling, cuda_visible_devices):
        rerun_calls.append((str(work_root), bool(run_local_sampling)))
        refined_complex = work_root / "refined_complex.pdb"
        refined_receptor = work_root / "refined_receptor.pdb"
        refined_ligand = work_root / "refined_ligand.sdf"
        refined_complex.write_text("ATOM      1  N   MET A   1      0.0  0.0  0.0  1.00  0.00           N\nEND\n", encoding="utf-8")
        refined_receptor.write_text("ATOM      1  N   MET A   1      0.0  0.0  0.0  1.00  0.00           N\nEND\n", encoding="utf-8")
        molecule = Chem.MolFromSmiles("CCO")
        writer = Chem.SDWriter(str(refined_ligand))
        writer.write(molecule)
        writer.close()
        return {
            "refined_complex_pdb": str(refined_complex),
            "refined_receptor_pdb": str(refined_receptor),
            "refined_ligand_sdf": str(refined_ligand),
        }

    monkeypatch.setattr(stage5_utils, "run_amber_relaxation", fake_run_amber_relaxation)

    payload = stage5_utils.ensure_stage5_relaxation(
        root=tmp_path,
        docking_row={
            "stage5_run_root": str(run_root.relative_to(tmp_path)),
            "wt_receptor_pdb": "wt_receptor.pdb",
            "wt_pose_sdf": "wt_pose.sdf",
            "mt_receptor_pdb": "mt_receptor.pdb",
            "mt_pose_sdf": "mt_pose.sdf",
            "stage4_local_rmsd_a": 0.0,
            "component_count": 1,
            "stage5_model_kind": "single_substitution",
        },
        stage5={"local_sampling_rmsd_trigger_a": 0.8, "local_sampling_component_count_trigger": 2},
        gpu_id=3,
    )

    assert len(rerun_calls) == 2
    assert payload["wt"]["relaxation"]["refined_ligand_sdf"].endswith("refined_ligand.sdf")
    assert payload["mt"]["relaxation"]["refined_receptor_pdb"].endswith("refined_receptor.pdb")


def test_template_pose_sdf_preserves_template_bond_orders_and_pose_coordinates(tmp_path: Path) -> None:
    template = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    pose = Chem.Mol(template)
    template_conf = Chem.Conformer(template.GetNumAtoms())
    pose_conf = Chem.Conformer(pose.GetNumAtoms())
    for atom_index in range(template.GetNumAtoms()):
        template_conf.SetAtomPosition(atom_index, Point3D(float(atom_index), 0.0, 0.0))
        pose_conf.SetAtomPosition(atom_index, Point3D(float(atom_index), 1.5, -2.0))
    template.RemoveAllConformers()
    pose.RemoveAllConformers()
    template.AddConformer(template_conf, assignId=True)
    pose.AddConformer(pose_conf, assignId=True)

    template_sdf = tmp_path / "template.sdf"
    pose_sdf = tmp_path / "pose.sdf"
    output_sdf = tmp_path / "mapped.sdf"
    template_writer = Chem.SDWriter(str(template_sdf))
    template_writer.write(template)
    template_writer.close()
    pose_writer = Chem.SDWriter(str(pose_sdf))
    pose_writer.write(pose)
    pose_writer.close()

    template_pose_sdf(template_sdf, pose_sdf, output_sdf)
    mapped = Chem.SDMolSupplier(str(output_sdf), removeHs=False)[0]

    assert mapped is not None
    assert mapped.GetBondBetweenAtoms(0, 1).GetBondType() == template.GetBondBetweenAtoms(0, 1).GetBondType()
    mapped_position = mapped.GetConformer().GetAtomPosition(3)
    assert mapped_position.x == 3.0
    assert mapped_position.y == 1.5
    assert mapped_position.z == -2.0


def test_template_pose_sdf_maps_heavy_atoms_when_pose_hydrogens_do_not_match(tmp_path: Path) -> None:
    template = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    template_conf = Chem.Conformer(template.GetNumAtoms())
    for atom_index in range(template.GetNumAtoms()):
        template_conf.SetAtomPosition(atom_index, Point3D(float(atom_index), 0.0, 0.0))
    template.RemoveAllConformers()
    template.AddConformer(template_conf, assignId=True)

    pose = Chem.RemoveHs(Chem.Mol(template), sanitize=False)
    pose_conf = Chem.Conformer(pose.GetNumAtoms())
    for atom_index in range(pose.GetNumAtoms()):
        pose_conf.SetAtomPosition(atom_index, Point3D(float(atom_index), 4.0, -1.0))
    pose.RemoveAllConformers()
    pose.AddConformer(pose_conf, assignId=True)

    template_sdf = tmp_path / "template_h.sdf"
    pose_sdf = tmp_path / "pose_heavy.sdf"
    output_sdf = tmp_path / "mapped_h.sdf"
    template_writer = Chem.SDWriter(str(template_sdf))
    template_writer.write(template)
    template_writer.close()
    pose_writer = Chem.SDWriter(str(pose_sdf))
    pose_writer.write(pose)
    pose_writer.close()

    template_pose_sdf(template_sdf, pose_sdf, output_sdf)
    mapped = Chem.SDMolSupplier(str(output_sdf), removeHs=False)[0]

    assert mapped is not None
    assert mapped.GetNumAtoms() == template.GetNumAtoms()
    heavy_indices = [atom.GetIdx() for atom in mapped.GetAtoms() if atom.GetAtomicNum() > 1]
    assert len(heavy_indices) == pose.GetNumAtoms()
    first_heavy_position = mapped.GetConformer().GetAtomPosition(heavy_indices[0])
    assert first_heavy_position.x == 0.0
    assert first_heavy_position.y == 4.0
    assert first_heavy_position.z == -1.0
    assert sum(1 for atom in mapped.GetAtoms() if atom.GetAtomicNum() == 1) > 0


def test_template_pose_sdf_maps_template_when_pose_heavy_atom_order_differs(tmp_path: Path) -> None:
    template = Chem.AddHs(Chem.MolFromSmiles("CC(=O)N"))
    template_conf = Chem.Conformer(template.GetNumAtoms())
    for atom_index in range(template.GetNumAtoms()):
        template_conf.SetAtomPosition(atom_index, Point3D(float(atom_index), 0.0, 0.0))
    template.RemoveAllConformers()
    template.AddConformer(template_conf, assignId=True)

    pose_base = Chem.RemoveHs(Chem.Mol(template), sanitize=False)
    reorder = [2, 0, 3, 1]
    pose = Chem.RenumberAtoms(pose_base, reorder)
    pose_conf = Chem.Conformer(pose.GetNumAtoms())
    for atom_index in range(pose.GetNumAtoms()):
        pose_conf.SetAtomPosition(atom_index, Point3D(float(atom_index), 7.0, -3.0))
    pose.RemoveAllConformers()
    pose.AddConformer(pose_conf, assignId=True)

    template_sdf = tmp_path / "template_reorder.sdf"
    pose_sdf = tmp_path / "pose_reorder.sdf"
    output_sdf = tmp_path / "mapped_reorder.sdf"
    template_writer = Chem.SDWriter(str(template_sdf))
    template_writer.write(template)
    template_writer.close()
    pose_writer = Chem.SDWriter(str(pose_sdf))
    pose_writer.write(pose)
    pose_writer.close()

    template_pose_sdf(template_sdf, pose_sdf, output_sdf)
    mapped = Chem.SDMolSupplier(str(output_sdf), removeHs=False)[0]

    assert mapped is not None
    assert mapped.GetNumAtoms() == template.GetNumAtoms()
    mapped_heavy = Chem.RemoveHs(Chem.Mol(mapped), sanitize=True)
    pose_heavy = Chem.RemoveHs(Chem.Mol(pose), sanitize=False)
    pattern = Chem.MolFromSmarts("NC(C)=O")
    mapped_match = mapped_heavy.GetSubstructMatch(pattern)
    pose_match = pose_heavy.GetSubstructMatch(pattern)
    assert mapped_match
    assert pose_match
    mapped_carbonyl_oxygen = mapped_heavy.GetConformer().GetAtomPosition(mapped_match[3])
    pose_carbonyl_oxygen = pose_heavy.GetConformer().GetAtomPosition(pose_match[3])
    assert mapped_carbonyl_oxygen.x == pose_carbonyl_oxygen.x
    assert mapped_carbonyl_oxygen.y == pose_carbonyl_oxygen.y
    assert mapped_carbonyl_oxygen.z == pose_carbonyl_oxygen.z


def test_write_ligand_pose_pdb_from_sdf_preserves_two_letter_halogen_atom_names(tmp_path: Path) -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("Clc1ccccc1Cl"))
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom_index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(atom_index, Point3D(float(atom_index), 0.0, 0.0))
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)

    input_sdf = tmp_path / "ligand.sdf"
    output_pdb = tmp_path / "ligand.pdb"
    writer = Chem.SDWriter(str(input_sdf))
    writer.write(molecule)
    writer.close()

    write_ligand_pose_pdb_from_sdf(input_sdf, output_pdb)
    atom_lines = [line for line in output_pdb.read_text(encoding="utf-8").splitlines() if line.startswith("ATOM")]

    chlorine_names = [line[12:16].strip() for line in atom_lines if line[76:78].strip() == "CL"]
    assert chlorine_names == ["Cl1", "Cl2"]


def test_prepare_ligand_parameter_files_reuses_case_cache(tmp_path: Path, monkeypatch) -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("Clc1ccccc1Cl"))
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom_index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(atom_index, Point3D(float(atom_index), 0.0, 0.0))
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)

    input_sdf = tmp_path / "ligand_cache.sdf"
    writer = Chem.SDWriter(str(input_sdf))
    writer.write(molecule)
    writer.close()

    command_calls: list[str] = []

    def fake_checked_command(
        command: list[str],
        cwd: Path | None = None,
        extra_env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ):
        command_calls.append(str(command[0]))
        output_flag = command.index("-o") + 1
        Path(command[output_flag]).write_text(f"{command[0]}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(stage5_physics, "checked_command", fake_checked_command)

    cache_root = tmp_path / "ligand_param_cache"
    first_payload = _prepare_ligand_parameter_files(
        amber_ligand_input=input_sdf,
        ligand_mol2=tmp_path / "first" / "ligand.mol2",
        ligand_frcmod=tmp_path / "first" / "ligand.frcmod",
        ligand_parameter_cache_root=cache_root,
    )
    second_payload = _prepare_ligand_parameter_files(
        amber_ligand_input=input_sdf,
        ligand_mol2=tmp_path / "second" / "ligand.mol2",
        ligand_frcmod=tmp_path / "second" / "ligand.frcmod",
        ligand_parameter_cache_root=cache_root,
    )

    assert command_calls == ["antechamber", "parmchk2"]
    assert first_payload["ligand_parameter_cache_hit"] is False
    assert second_payload["ligand_parameter_cache_hit"] is True
    assert Path(first_payload["ligand_parameter_source_mol2"]).exists()
    assert Path(second_payload["ligand_parameter_source_mol2"]).exists()


def test_prepare_ligand_parameter_files_recovers_from_partial_cache(tmp_path: Path, monkeypatch) -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom_index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(atom_index, Point3D(float(atom_index), 0.0, 0.0))
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)

    input_sdf = tmp_path / "ligand_partial_cache.sdf"
    writer = Chem.SDWriter(str(input_sdf))
    writer.write(molecule)
    writer.close()

    cache_root = tmp_path / "ligand_param_cache"
    cache_key = stage5_physics._ligand_parameter_cache_key(input_sdf)
    cache_dir = cache_root / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "ligand.mol2").write_text("partial\n", encoding="utf-8")

    command_calls: list[str] = []

    def fake_checked_command(
        command: list[str],
        cwd: Path | None = None,
        extra_env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ):
        command_calls.append(str(command[0]))
        output_flag = command.index("-o") + 1
        Path(command[output_flag]).write_text(f"{command[0]}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(stage5_physics, "checked_command", fake_checked_command)

    payload = _prepare_ligand_parameter_files(
        amber_ligand_input=input_sdf,
        ligand_mol2=tmp_path / "work" / "ligand.mol2",
        ligand_frcmod=tmp_path / "work" / "ligand.frcmod",
        ligand_parameter_cache_root=cache_root,
    )

    assert payload["ligand_parameter_cache_hit"] is False
    assert command_calls == ["antechamber", "parmchk2"]
    assert (cache_dir / "ligand.mol2").read_text(encoding="utf-8") == "antechamber\n"
    assert (cache_dir / "ligand.frcmod").read_text(encoding="utf-8") == "parmchk2\n"


def test_probe_ligand_parameterization_reports_error_without_raising(tmp_path: Path, monkeypatch) -> None:
    input_sdf = tmp_path / "ligand.sdf"
    input_sdf.write_text("stub\n", encoding="utf-8")

    def fake_hydrogenate(source: Path, destination: Path) -> dict[str, object]:
        destination.write_text("prepared\n", encoding="utf-8")
        return {"ligand_hydrogenation_method": "stub"}

    def fake_prepare(**_: object) -> dict[str, object]:
        raise RuntimeError("sqm timeout")

    monkeypatch.setattr(stage5_physics, "ensure_explicit_hydrogens_sdf", fake_hydrogenate)
    monkeypatch.setattr(stage5_physics, "_prepare_ligand_parameter_files", fake_prepare)

    payload = probe_ligand_parameterization(
        input_sdf=input_sdf,
        work_root=tmp_path / "probe",
        timeout_sec=5.0,
    )

    assert payload["available"] is False
    assert "sqm timeout" in str(payload["error"])
    assert (tmp_path / "probe" / "ligand_parameterization_probe.json").exists()


def test_reset_incomplete_stage5_workdir_clears_partial_directory(tmp_path: Path) -> None:
    work_root = tmp_path / "amber_prep"
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "stale.tmp").write_text("stale\n", encoding="utf-8")
    manifest_path = work_root / "amber_prep_manifest.json"

    stage5_utils._reset_incomplete_stage5_workdir(work_root, manifest_path)

    assert work_root.exists()
    assert list(work_root.iterdir()) == []
    manifest_path.write_text("{}", encoding="utf-8")
    (work_root / "keep.tmp").write_text("keep\n", encoding="utf-8")
    stage5_utils._reset_incomplete_stage5_workdir(work_root, manifest_path)
    assert (work_root / "keep.tmp").exists()


def test_strip_hydrogens_from_pdb_removes_hydrogen_records(tmp_path: Path) -> None:
    input_pdb = tmp_path / "input.pdb"
    output_pdb = tmp_path / "output.pdb"
    input_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   ASP A 233      12.766 -10.506  59.812  1.00  0.00           N  ",
                "ATOM      2  H   ASP A 233      13.187 -11.584  60.118  1.00  0.00           H  ",
                "ATOM      3  H2  ASP A 233      11.810 -10.783  59.135  1.00  0.00           H  ",
                "ATOM      4  CA  ASP A 233      13.678  -9.657  58.941  1.00  0.00           C  ",
                "TER",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    strip_hydrogens_from_pdb(input_pdb, output_pdb)
    lines = output_pdb.read_text(encoding="utf-8").splitlines()

    assert any(" N   ASP " in line for line in lines)
    assert any(" CA  ASP " in line for line in lines)
    assert not any(" H   ASP " in line or " H2  ASP " in line for line in lines)
    assert "TER" in lines
    assert lines[-1] == "END"


def test_normalize_special_receptor_residues_maps_cy0_to_cys(tmp_path: Path) -> None:
    input_pdb = tmp_path / "input_cy0.pdb"
    output_pdb = tmp_path / "output_cy0.pdb"
    input_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   GLY A 101      10.000  10.000  10.000  1.00  0.00           N  ",
                "ATOM      2  CA  GLY A 101      11.000  10.000  10.000  1.00  0.00           C  ",
                "HETATM    3  N   CY0 A 102      12.000  10.000  10.000  1.00  0.00           N  ",
                "HETATM    4  CA  CY0 A 102      13.000  10.000  10.000  1.00  0.00           C  ",
                "HETATM    5  C   CY0 A 102      14.000  10.000  10.000  1.00  0.00           C  ",
                "HETATM    6  O   CY0 A 102      15.000  10.000  10.000  1.00  0.00           O  ",
                "HETATM    7  CB  CY0 A 102      13.000  11.000  10.000  1.00  0.00           C  ",
                "HETATM    8 SAU  CY0 A 102      13.000  12.500  10.000  1.00  0.00           S  ",
                "HETATM    9 CAE  CY0 A 102      16.000  10.000  10.000  1.00  0.00           C  ",
                "ATOM     10  N   LEU A 103      14.500  11.000  10.000  1.00  0.00           N  ",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    payload = normalize_special_receptor_residues(input_pdb, output_pdb)
    text = output_pdb.read_text(encoding="utf-8")

    assert payload["normalized_receptor_residues"] == {"CY0": 6}
    assert payload["normalized_receptor_atom_count"] == 6
    assert payload["removed_receptor_atom_count"] == 1
    lines = text.splitlines()
    assert any(line[17:20].strip() == "CYS" and line[22:26].strip() == "102" for line in lines if line.startswith("ATOM"))
    assert any(line[12:16].strip() == "SG" and line[17:20].strip() == "CYS" for line in lines if line.startswith("ATOM"))
    assert not any("CY0" in line for line in lines)
    assert not any(line[12:16].strip() == "CAE" for line in lines if line.startswith("ATOM"))


def test_normalize_special_receptor_residues_maps_sep_to_ser_and_drops_phosphate(tmp_path: Path) -> None:
    input_pdb = tmp_path / "input_sep.pdb"
    output_pdb = tmp_path / "output_sep.pdb"
    input_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   GLY A   4      10.000  10.000  10.000  1.00  0.00           N  ",
                "HETATM    2  N   SEP A   5      11.000  10.000  10.000  1.00  0.00           N  ",
                "HETATM    3  CA  SEP A   5      12.000  10.000  10.000  1.00  0.00           C  ",
                "HETATM    4  C   SEP A   5      13.000  10.000  10.000  1.00  0.00           C  ",
                "HETATM    5  O   SEP A   5      14.000  10.000  10.000  1.00  0.00           O  ",
                "HETATM    6  CB  SEP A   5      12.000  11.000  10.000  1.00  0.00           C  ",
                "HETATM    7  OG  SEP A   5      12.000  12.000  10.000  1.00  0.00           O  ",
                "HETATM    8  P   SEP A   5      13.000  13.000  10.000  1.00  0.00           P  ",
                "HETATM    9  O1P SEP A   5      14.000  13.000  10.000  1.00  0.00           O  ",
                "HETATM   10  O2P SEP A   5      13.000  14.000  10.000  1.00  0.00           O  ",
                "HETATM   11  O3P SEP A   5      12.000  13.000  10.000  1.00  0.00           O  ",
                "ATOM     12  N   LEU A   6      13.500  11.000  10.000  1.00  0.00           N  ",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    payload = normalize_special_receptor_residues(input_pdb, output_pdb)
    text = output_pdb.read_text(encoding="utf-8")

    assert payload["normalized_receptor_residues"] == {"SEP": 6}
    assert payload["normalized_receptor_atom_count"] == 6
    assert payload["removed_receptor_atom_count"] == 4
    lines = text.splitlines()
    assert any(line[17:20].strip() == "SER" and line[22:26].strip() == "5" for line in lines if line.startswith("ATOM"))
    assert any(line[12:16].strip() == "OG" and line[17:20].strip() == "SER" for line in lines if line.startswith("ATOM"))
    assert not any("SEP" in line for line in lines)
    assert not any(line[12:16].strip() in {"P", "O1P", "O2P", "O3P"} for line in lines if line.startswith("ATOM"))


def test_ensure_explicit_hydrogens_sdf_adds_missing_hydrogens(tmp_path: Path) -> None:
    heavy_only = Chem.MolFromSmiles("CCO")
    conformer = Chem.Conformer(heavy_only.GetNumAtoms())
    for atom_index in range(heavy_only.GetNumAtoms()):
        conformer.SetAtomPosition(atom_index, Point3D(float(atom_index), 2.0, -1.0))
    heavy_only.AddConformer(conformer, assignId=True)

    input_sdf = tmp_path / "heavy_only.sdf"
    output_sdf = tmp_path / "with_h.sdf"
    writer = Chem.SDWriter(str(input_sdf))
    writer.write(heavy_only)
    writer.close()

    payload = ensure_explicit_hydrogens_sdf(input_sdf, output_sdf)
    hydrated = Chem.SDMolSupplier(str(output_sdf), removeHs=False)[0]

    assert payload["ligand_hydrogenation_method"] == "rdkit_add_hs"
    assert payload["ligand_input_atom_count"] == 3
    assert payload["ligand_output_atom_count"] > payload["ligand_input_atom_count"]
    assert hydrated is not None
    assert hydrated.GetNumAtoms() == payload["ligand_output_atom_count"]
    assert sum(1 for atom in hydrated.GetAtoms() if atom.GetAtomicNum() == 1) > 0
    first_heavy = next(atom.GetIdx() for atom in hydrated.GetAtoms() if atom.GetAtomicNum() > 1)
    first_position = hydrated.GetConformer().GetAtomPosition(first_heavy)
    assert first_position.x == 0.0
    assert first_position.y == 2.0
    assert first_position.z == -1.0


def test_amber_md_instability_reason_detects_masked_numeric_overflow(tmp_path: Path) -> None:
    md_out = tmp_path / "md.out"
    md_out.write_text(
        "\n".join(
            [
                " NSTEP =    15000   TIME(PS) =      30.000  TEMP(K) =*********  PRESS =     0.0",
                " Etot   = **************",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert amber_md_instability_reason(md_out) == "masked_numeric_overflow"


def test_amber_md_instability_reason_ignores_stable_output(tmp_path: Path) -> None:
    md_out = tmp_path / "stable_md.out"
    md_out.write_text(
        "\n".join(
            [
                " NSTEP =    10000   TIME(PS) =      20.000  TEMP(K) =   301.21  PRESS =     0.0",
                " Etot   =     -1234.5678",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert amber_md_instability_reason(md_out) is None


def test_compute_ifp_effect_row_falls_back_to_docked_when_relaxed_outputs_are_empty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path
    run_root = root / "outputs" / "egfr_erlotinib" / "stage5" / "docking_runs" / "combo_example"
    run_root.mkdir(parents=True, exist_ok=True)

    docked_paths = {
        "wt_complex_docked_pdb": run_root / "wt_complex_docked.pdb",
        "mt_complex_docked_pdb": run_root / "mt_complex_docked.pdb",
        "wt_receptor_pdb": run_root / "wt_receptor_prepared.pdb",
        "mt_receptor_pdb": run_root / "mt_receptor_prepared.pdb",
        "wt_pose_sdf": run_root / "wt_pose.sdf",
        "mt_pose_sdf": run_root / "mt_pose.sdf",
    }
    for path in docked_paths.values():
        path.write_text("stub\n", encoding="utf-8")

    empty_wt_relax_root = run_root / "wt_physics" / "relaxation"
    empty_mt_relax_root = run_root / "mt_physics" / "relaxation"
    empty_wt_relax_root.mkdir(parents=True, exist_ok=True)
    empty_mt_relax_root.mkdir(parents=True, exist_ok=True)
    for path in [
        empty_wt_relax_root / "refined_complex.pdb",
        empty_wt_relax_root / "refined_receptor.pdb",
        empty_mt_relax_root / "refined_complex.pdb",
        empty_mt_relax_root / "refined_receptor.pdb",
    ]:
        path.write_text("END\n", encoding="utf-8")
    for path in [empty_wt_relax_root / "refined_ligand.sdf", empty_mt_relax_root / "refined_ligand.sdf"]:
        path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        stage5_utils,
        "ensure_stage5_relaxation",
        lambda **_: {
            "wt": {
                "relaxation": {
                    "refined_complex_pdb": str(empty_wt_relax_root / "refined_complex.pdb"),
                    "refined_receptor_pdb": str(empty_wt_relax_root / "refined_receptor.pdb"),
                    "refined_ligand_sdf": str(empty_wt_relax_root / "refined_ligand.sdf"),
                }
            },
            "mt": {
                "relaxation": {
                    "refined_complex_pdb": str(empty_mt_relax_root / "refined_complex.pdb"),
                    "refined_receptor_pdb": str(empty_mt_relax_root / "refined_receptor.pdb"),
                    "refined_ligand_sdf": str(empty_mt_relax_root / "refined_ligand.sdf"),
                }
            },
        },
    )

    plip_calls: list[str] = []

    def fake_plip_ifp(path: Path) -> dict[str, object]:
        plip_calls.append(Path(path).name)
        counts = {name: 0 for name in stage5_utils.PLIP_INTERACTION_TYPES}
        counts["hydrophobic"] = 1
        residue_label = "A:MET1" if "wt_" in Path(path).name else "A:MET2"
        return {
            "residue_set": [residue_label],
            "interactions": [{"interaction_type": "hydrophobic", "residue_label": residue_label}],
            "interaction_type_counts": counts,
        }

    monkeypatch.setattr(stage5_utils, "plip_ifp", fake_plip_ifp)
    monkeypatch.setattr(
        stage5_utils,
        "local_pocket_metrics",
        lambda **_: {"pocket_volume_proxy_a3": 10.0, "contact_density": 1.0, "polar_exposed_fraction": 0.2},
    )
    monkeypatch.setattr(stage5_utils, "fpocket_top_pocket_metrics", lambda *_: {"volume_a3": 100.0})
    monkeypatch.setattr(
        stage5_utils,
        "occupancy_frequency_payload",
        lambda **_: {"top_seed_count": 1, "occupancy_map": {"A:MET1": 1.0}},
    )
    monkeypatch.setattr(stage5_utils, "deterministic_mechanism_labels", lambda **_: ["anchor_loss"])

    try:
        compute_ifp_effect_row(
            root=root,
            case_id="egfr_erlotinib",
            docking_row={
                "effect_scope": "combo",
                "target_key": "EGFR:L747_E749delLRE+A750P",
                "target_slug": "combo_example",
                "stage5_status": "ok",
                "impact_evidence_tier": "modeled",
                "sample_source": "stage5_modeled",
                "used_synthetic_combo_model": False,
                "used_stage5_modeled_sample": True,
                "stage5_model_kind": "mixed_deletion_substitution",
                "stage4_rank": 1,
                "risk_score": 0.9,
                "delta_dock_kcal_mol": 1.2,
                "stage4_local_rmsd_a": 0.5,
                "stage5_run_root": "outputs/egfr_erlotinib/stage5/docking_runs/combo_example",
                **{key: str(path.relative_to(root)) for key, path in docked_paths.items()},
            },
            anchor_labels={"A:MET1"},
            stage5={
                "pocket_contact_distance_a": 4.0,
                "polar_contact_distance_a": 3.5,
                "solvent_exposure_neighbor_threshold": 2,
                "local_sampling_enabled": True,
                "local_sampling_component_count_trigger": 2,
                "local_sampling_rmsd_trigger_a": 0.8,
                "pocket_volume_change_threshold": 0.2,
                "solvent_proxy_shift_threshold": 0.1,
                "local_rmsd_threshold_a": 0.8,
            },
            pose_ensemble=pd.DataFrame(),
        )
    except RuntimeError as exc:
        assert "Invalid Stage5 relaxation outputs" in str(exc)
    else:
        raise AssertionError("compute_ifp_effect_row should fail on invalid relaxation outputs in strict mode")
    assert plip_calls == []


def test_ensure_stage5_scoring_payload_marks_mmgbsa_unavailable_as_high_uncertainty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path
    run_root = root / "outputs" / "egfr_erlotinib" / "stage5" / "docking_runs" / "combo_example"
    amber_root = run_root / "wt_physics" / "amber_prep"
    amber_root.mkdir(parents=True, exist_ok=True)
    relax_root = run_root / "wt_physics" / "relaxation"
    relax_root.mkdir(parents=True, exist_ok=True)
    receptor_pdb = relax_root / "refined_receptor.pdb"
    complex_pdb = relax_root / "refined_complex.pdb"
    ligand_sdf = relax_root / "refined_ligand.sdf"
    receptor_pdb.write_text("ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N  \nEND\n", encoding="utf-8")
    complex_pdb.write_text("HETATM    1  C1  LIG Z 900       0.000   0.000   0.000  1.00  0.00           C  \nEND\n", encoding="utf-8")
    mol = Chem.MolFromSmiles("CC")
    writer = Chem.SDWriter(str(ligand_sdf))
    writer.write(mol)
    writer.close()
    amber_prep = {
        "complex_prmtop": str(amber_root / "complex.prmtop"),
        "receptor_prmtop": str(amber_root / "receptor.prmtop"),
        "ligand_prmtop": str(amber_root / "ligand.prmtop"),
    }
    relaxation_payload = {
        "trajectory_path": str(relax_root / "min.rst7"),
        "refined_complex_pdb": str(complex_pdb),
        "refined_receptor_pdb": str(receptor_pdb),
        "refined_ligand_sdf": str(ligand_sdf),
    }

    monkeypatch.setattr(
        stage5_utils,
        "ensure_stage5_relaxation",
        lambda **_: {
            "wt": {"amber_prep": dict(amber_prep), "relaxation": dict(relaxation_payload)},
            "mt": {"amber_prep": dict(amber_prep), "relaxation": dict(relaxation_payload)},
        },
    )
    monkeypatch.setattr(stage5_utils, "run_mmgbsa", lambda **_: (_ for _ in ()).throw(RuntimeError("MMPBSA stars")))
    monkeypatch.setattr(stage5_utils, "gnina_score_only", lambda **_: {"available": False, "affinity_kcal_mol": None})

    payload = stage5_utils.ensure_stage5_scoring_payload(
        root=root,
        docking_row={
            "stage5_run_root": "outputs/egfr_erlotinib/stage5/docking_runs/combo_example",
            "delta_dock_kcal_mol": 1.25,
        },
        stage5={
            "multi_score_neutral_threshold_kcal_mol": 0.25,
            "multi_score_consensus_threshold": 0.7,
        },
    )

    assert payload["wt_mmgbsa_binding_kcal_mol"] is None
    assert payload["mt_mmgbsa_binding_kcal_mol"] is None
    assert payload["delta_mmgbsa_binding_kcal_mol"] is None
    assert payload["consensus_direction"] == "resistance_like"
    assert payload["high_uncertainty"] is True
    wt_manifest = root / "outputs" / "egfr_erlotinib" / "stage5" / "docking_runs" / "combo_example" / "wt_physics" / "mmgbsa" / "mmgbsa_manifest.json"
    mt_manifest = root / "outputs" / "egfr_erlotinib" / "stage5" / "docking_runs" / "combo_example" / "mt_physics" / "mmgbsa" / "mmgbsa_manifest.json"
    assert json.loads(wt_manifest.read_text())["available"] is False
    assert json.loads(mt_manifest.read_text())["available"] is False


def test_ensure_stage5_scoring_payload_recomputes_stale_gnina_manifest_when_binary_is_restored(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path
    run_root = root / "outputs" / "egfr_erlotinib" / "stage5" / "docking_runs" / "combo_example"
    amber_root = run_root / "wt_physics" / "amber_prep"
    amber_root.mkdir(parents=True, exist_ok=True)
    relax_root_wt = run_root / "wt_physics" / "relaxation"
    relax_root_mt = run_root / "mt_physics" / "relaxation"
    relax_root_wt.mkdir(parents=True, exist_ok=True)
    relax_root_mt.mkdir(parents=True, exist_ok=True)
    for relax_root in [relax_root_wt, relax_root_mt]:
        (relax_root / "refined_receptor.pdb").write_text(
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N  \nEND\n",
            encoding="utf-8",
        )
        (relax_root / "refined_complex.pdb").write_text(
            "HETATM    1  C1  LIG Z 900       0.000   0.000   0.000  1.00  0.00           C  \nEND\n",
            encoding="utf-8",
        )
        writer = Chem.SDWriter(str(relax_root / "refined_ligand.sdf"))
        writer.write(Chem.MolFromSmiles("CC"))
        writer.close()
    for pose_set in ["wt", "mt"]:
        stale_manifest = run_root / f"{pose_set}_physics" / "gnina" / "gnina_manifest.json"
        stale_manifest.parent.mkdir(parents=True, exist_ok=True)
        stale_manifest.write_text('{"available": false, "affinity_kcal_mol": null}', encoding="utf-8")

    amber_prep = {
        "complex_prmtop": str(amber_root / "complex.prmtop"),
        "receptor_prmtop": str(amber_root / "receptor.prmtop"),
        "ligand_prmtop": str(amber_root / "ligand.prmtop"),
    }
    wt_relaxation = {
        "trajectory_path": str(relax_root_wt / "min.rst7"),
        "refined_complex_pdb": str(relax_root_wt / "refined_complex.pdb"),
        "refined_receptor_pdb": str(relax_root_wt / "refined_receptor.pdb"),
        "refined_ligand_sdf": str(relax_root_wt / "refined_ligand.sdf"),
    }
    mt_relaxation = {
        "trajectory_path": str(relax_root_mt / "min.rst7"),
        "refined_complex_pdb": str(relax_root_mt / "refined_complex.pdb"),
        "refined_receptor_pdb": str(relax_root_mt / "refined_receptor.pdb"),
        "refined_ligand_sdf": str(relax_root_mt / "refined_ligand.sdf"),
    }

    monkeypatch.setattr(
        stage5_utils,
        "ensure_stage5_relaxation",
        lambda **_: {
            "wt": {"amber_prep": dict(amber_prep), "relaxation": dict(wt_relaxation)},
            "mt": {"amber_prep": dict(amber_prep), "relaxation": dict(mt_relaxation)},
        },
    )
    monkeypatch.setattr(stage5_utils, "run_mmgbsa", lambda **_: {"available": True, "delta_total_kcal_mol": -12.0})
    monkeypatch.setattr(stage5_utils, "command_exists", lambda name: name == "gnina")
    affinities = iter([-8.1, -7.4])
    monkeypatch.setattr(
        stage5_utils,
        "gnina_score_only",
        lambda **_: {"available": True, "affinity_kcal_mol": next(affinities)},
    )

    payload = stage5_utils.ensure_stage5_scoring_payload(
        root=root,
        docking_row={
            "stage5_run_root": "outputs/egfr_erlotinib/stage5/docking_runs/combo_example",
            "delta_dock_kcal_mol": 0.4,
        },
        stage5={
            "multi_score_neutral_threshold_kcal_mol": 0.25,
            "multi_score_consensus_threshold": 0.7,
            "require_gnina_for_stage5_scoring": True,
        },
    )

    assert payload["wt_gnina_affinity_kcal_mol"] == -8.1
    assert payload["mt_gnina_affinity_kcal_mol"] == -7.4
    assert payload["delta_gnina_affinity_kcal_mol"] == pytest.approx(0.7)
    assert json.loads((run_root / "wt_physics" / "gnina" / "gnina_manifest.json").read_text())["available"] is True
    assert json.loads((run_root / "mt_physics" / "gnina" / "gnina_manifest.json").read_text())["available"] is True


def test_ensure_stage5_scoring_payload_rejects_invalid_relaxed_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path
    run_root = root / "outputs" / "egfr_erlotinib" / "stage5" / "docking_runs" / "combo_example"
    run_root.mkdir(parents=True, exist_ok=True)
    amber_root = run_root / "wt_physics" / "amber_prep"
    amber_root.mkdir(parents=True, exist_ok=True)
    invalid_wt_relax_root = run_root / "wt_physics" / "relaxation"
    invalid_mt_relax_root = run_root / "mt_physics" / "relaxation"
    invalid_wt_relax_root.mkdir(parents=True, exist_ok=True)
    invalid_mt_relax_root.mkdir(parents=True, exist_ok=True)
    amber_prep = {
        "complex_prmtop": str(amber_root / "complex.prmtop"),
        "receptor_prmtop": str(amber_root / "receptor.prmtop"),
        "ligand_prmtop": str(amber_root / "ligand.prmtop"),
    }
    for path in [
        invalid_wt_relax_root / "refined_complex.pdb",
        invalid_wt_relax_root / "refined_receptor.pdb",
        invalid_mt_relax_root / "refined_complex.pdb",
        invalid_mt_relax_root / "refined_receptor.pdb",
    ]:
        path.write_text("END\n", encoding="utf-8")
    for path in [invalid_wt_relax_root / "refined_ligand.sdf", invalid_mt_relax_root / "refined_ligand.sdf"]:
        path.write_text("", encoding="utf-8")
    relaxation_payload = {
        "trajectory_path": str(run_root / "wt_physics" / "relaxation" / "min.rst7"),
        "refined_complex_pdb": str(invalid_wt_relax_root / "refined_complex.pdb"),
        "refined_receptor_pdb": str(invalid_wt_relax_root / "refined_receptor.pdb"),
        "refined_ligand_sdf": str(invalid_wt_relax_root / "refined_ligand.sdf"),
    }
    relaxation_payload_mt = {
        "trajectory_path": str(run_root / "mt_physics" / "relaxation" / "min.rst7"),
        "refined_complex_pdb": str(invalid_mt_relax_root / "refined_complex.pdb"),
        "refined_receptor_pdb": str(invalid_mt_relax_root / "refined_receptor.pdb"),
        "refined_ligand_sdf": str(invalid_mt_relax_root / "refined_ligand.sdf"),
    }

    monkeypatch.setattr(
        stage5_utils,
        "ensure_stage5_relaxation",
        lambda **_: {
            "wt": {"amber_prep": dict(amber_prep), "relaxation": dict(relaxation_payload)},
            "mt": {"amber_prep": dict(amber_prep), "relaxation": dict(relaxation_payload_mt)},
        },
    )
    monkeypatch.setattr(stage5_utils, "run_mmgbsa", lambda **_: {"delta_total_kcal_mol": -10.0})
    try:
        stage5_utils.ensure_stage5_scoring_payload(
            root=root,
            docking_row={
                "stage5_run_root": "outputs/egfr_erlotinib/stage5/docking_runs/combo_example",
                "delta_dock_kcal_mol": 1.25,
            },
            stage5={
                "multi_score_neutral_threshold_kcal_mol": 0.25,
                "multi_score_consensus_threshold": 0.7,
            },
        )
    except RuntimeError as exc:
        assert "Invalid Stage5 relaxation outputs" in str(exc)
    else:
        raise AssertionError("ensure_stage5_scoring_payload should fail on invalid relaxation outputs in strict mode")


def test_restart_to_pdb_uses_ambpdb_stdout(tmp_path: Path, monkeypatch) -> None:
    prmtop = tmp_path / "complex.prmtop"
    restart = tmp_path / "final.rst7"
    output_pdb = tmp_path / "refined_complex.pdb"
    prmtop.write_text("stub\n", encoding="utf-8")
    restart.write_text("stub\n", encoding="utf-8")

    monkeypatch.setattr(stage5_physics, "command_exists", lambda name: name == "ambpdb")
    monkeypatch.setattr(
        stage5_physics,
        "checked_command",
        lambda command, cwd=None, extra_env=None: type("Result", (), {"stdout": "ATOM      1  N   GLY A   1      0.0 0.0 0.0\nEND\n"})(),
    )

    stage5_physics._restart_to_pdb(prmtop, restart, output_pdb)

    text = output_pdb.read_text(encoding="utf-8")
    assert "ATOM" in text
    assert text.endswith("END\n")


def test_amber_relaxation_mdin_uses_namelist_safe_restraint_mask() -> None:
    stage5 = {"local_sampling_backbone_restraint_kcal_mol_a2": 2.5, "local_sampling_ns": 1.0}

    min_mdin = _min_mdin(stage5)
    md_mdin = _md_mdin(stage5)

    assert min_mdin.splitlines()[0] == "Stage5 restrained minimization"
    assert md_mdin.splitlines()[0] == "Stage5 restrained local sampling"
    assert min_mdin.splitlines()[1] == "&cntrl"
    assert md_mdin.splitlines()[1] == "&cntrl"
    assert "restraint_wt=2.500" in min_mdin
    assert "restraint_wt=2.500" in md_mdin
    assert "restraintmask='@CA,C,N,O'," in min_mdin
    assert "restraintmask='@CA,C,N,O'," in md_mdin
    assert "& !:LIG" not in min_mdin
    assert "& !:LIG" not in md_mdin
