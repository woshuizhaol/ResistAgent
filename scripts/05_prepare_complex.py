#!/usr/bin/env python3
"""Stage 3.5 WT baseline complex generation and mechanism extraction."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import ensure_dir, json_dump, load_yaml, project_root
from tools.stage35_utils import (
    anchor_residues,
    best_rmsd,
    build_box_from_ligand_coords,
    build_box_from_residue_coords,
    build_nearby_cofactor_set,
    build_water_retention_table,
    choose_chain_ligand,
    crystal_ligand_from_template,
    classify_hiv_pose,
    display_path,
    extract_ligand_to_sdf,
    fpocket_top_pocket_coords,
    fpocket_box,
    ifp_frequency,
    ligand_pose_pdb,
    load_rdkit_molecule,
    merge_pdb_fragments,
    mol_coordinates,
    nearby_protein_residue_labels,
    obrms_rmsd,
    nearby_protein_residue_ids,
    plip_ifp,
    pose_pdb_from_sdf,
    prepare_ligand_pdbqt,
    prepare_receptor_pdbqt,
    protein_chain_ids,
    residue_atom_coordinates,
    run_pdbfixer,
    run_vina_redocking,
    save_chain_protein,
    save_chain_set_protein,
    save_residue_subset,
    select_best_pose,
    standardize_reference_ligand,
    top_pose_per_seed,
    write_anchor_file,
    write_ifp_json,
    write_table,
)


POSE_STATS_COLUMNS = [
    "case_id",
    "pose_source",
    "redocking_attempt",
    "docking_box_source",
    "seed",
    "mode_rank",
    "affinity_kcal_mol",
    "rmsd_to_crystal_a",
    "pose_label",
    "selected_for_anchor",
    "selected_as_final_pose",
    "ifp_residue_count",
    "nnrti_min_distance_a",
    "nnrti_coverage_count",
    "nnrti_coverage_fraction",
    "active_site_min_distance_a",
    "active_site_coverage_count",
    "active_site_coverage_fraction",
    "pose_sdf",
]

HIV_QC_COLUMNS = [
    "case_id",
    "redocking_attempt",
    "docking_box_source",
    "seed",
    "mode_rank",
    "affinity_kcal_mol",
    "pose_label",
    "nnrti_min_distance_a",
    "nnrti_coverage_count",
    "nnrti_coverage_fraction",
    "active_site_min_distance_a",
    "active_site_coverage_count",
    "active_site_coverage_fraction",
    "pose_sdf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--case-id", default=None)
    return parser.parse_args()


def stage35_paths(root: Path, case_id: str) -> dict[str, Path]:
    stage_root = ensure_dir(root / "outputs" / case_id / "stage3_5")
    return {
        "root": stage_root,
        "wt_complex": stage_root / "wt_complex.pdb",
        "wt_complex_multichain": stage_root / "wt_complex_multichain.pdb",
        "wt_ifp": stage_root / "wt_ifp.json",
        "wt_ifp_multichain": stage_root / "wt_ifp_multichain.json",
        "wt_anchor_residues": stage_root / "wt_anchor_residues.txt",
        "wt_anchor_residues_multichain": stage_root / "wt_anchor_residues_multichain.txt",
        "wt_receptor_stage6_multichain": stage_root / "wt_receptor_stage6_multichain.pdb",
        "wt_pose_stats": stage_root / "wt_pose_stats.csv",
        "docking_box": stage_root / "docking_box.json",
        "water_retention_report": stage_root / "water_retention_report.csv",
        "hiv_pose_qc": stage_root / "hiv_nnrti_pose_qc.csv",
    }


def empty_hiv_qc(path: Path) -> None:
    pd.DataFrame(columns=HIV_QC_COLUMNS).to_csv(path, index=False)


def selected_cases(cases_config: dict[str, object], case_id: str | None) -> list[dict[str, object]]:
    cases = list(cases_config.get("set_d", []))
    if case_id is None:
        return cases
    return [case for case in cases if str(case.get("case_id")) == str(case_id)]


def resolve_structure_cache(root: Path, pdb_id: str) -> Path:
    candidates = [
        root / "outputs" / "case_manifests" / "rcsb_cache" / f"{pdb_id.upper()}.pdb",
        root / "outputs" / "stage1_5" / "rcsb_cache" / f"{pdb_id.upper()}.pdb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate cached structure for {pdb_id}")


def residue_ids_from_template_rows(
    residue_rows: list[dict[str, Any]],
    target_positions: list[int],
    rt_offset: int | None = None,
) -> set[tuple[str, int, str]]:
    targets = {int(value) for value in target_positions}
    residue_ids: set[tuple[str, int, str]] = set()
    for row in residue_rows:
        if row.get("uniprot_pos") is None:
            continue
        position = int(row["uniprot_pos"])
        if rt_offset is not None:
            position = int(position - int(rt_offset))
        if position in targets:
            residue_ids.add((" ", int(row["pdb_resnum"]), str(row.get("insertion_code") or "")))
    return residue_ids


def build_receptor_fragments(
    source_pdb: Path,
    chain_id: str,
    prepared_receptor_pdb: Path,
    water_report_path: Path,
    ligand_coords: list[tuple[float, float, float]],
    excluded_residue_ids: set[tuple[str, int, str]] | None,
    stage3_5: dict[str, Any],
    temp_dir: Path,
    apply_pdbfixer: bool,
) -> dict[str, Any]:
    protein_pdb = temp_dir / "protein_chain.pdb"
    protein_fixed_pdb = temp_dir / "protein_chain_fixed.pdb"
    water_pdb = temp_dir / "retained_waters.pdb"
    cofactor_pdb = temp_dir / "retained_cofactors.pdb"

    save_chain_protein(source_pdb, chain_id, protein_pdb)
    if apply_pdbfixer:
        run_pdbfixer(protein_pdb, protein_fixed_pdb, float(stage3_5["protein_prep_ph"]))
    else:
        protein_fixed_pdb.write_text(protein_pdb.read_text(encoding="utf-8"), encoding="utf-8")

    water_df, retained_waters = build_water_retention_table(
        source_pdb,
        chain_id,
        ligand_coords,
        distance_cutoff_a=float(stage3_5["bridge_water_distance_a"]),
        max_bfactor=float(stage3_5["bridge_water_bfactor_max"]),
    )
    write_table(water_df, water_report_path)
    save_residue_subset(source_pdb, chain_id, retained_waters, water_pdb)

    retained_cofactors = build_nearby_cofactor_set(
        source_pdb,
        chain_id,
        ligand_coords,
        distance_cutoff_a=float(stage3_5["cofactor_distance_a"]),
        excluded_residue_ids=excluded_residue_ids,
    )
    save_residue_subset(source_pdb, chain_id, retained_cofactors, cofactor_pdb)

    merge_pdb_fragments([protein_fixed_pdb, water_pdb, cofactor_pdb], prepared_receptor_pdb)
    return {
        "protein_fixed_pdb": protein_fixed_pdb,
        "water_pdb": water_pdb,
        "cofactor_pdb": cofactor_pdb,
        "retained_water_count": int(water_df["retained"].astype(bool).sum()) if not water_df.empty else 0,
        "retained_cofactor_count": int(len(retained_cofactors)),
    }


def reference_ligand_setup(
    root: Path,
    case_id: str,
    temp_dir: Path,
) -> dict[str, Any]:
    input_sdf = root / "outputs" / case_id / "stage1_5" / "raw" / "ligand.sdf"
    output_sdf = temp_dir / "reference_ligand_3d.sdf"
    ligand_qc = standardize_reference_ligand(input_sdf, output_sdf)
    molecule = load_rdkit_molecule(output_sdf, sanitize=True)
    return {
        "source_sdf": input_sdf,
        "standardized_sdf": output_sdf,
        "coords": mol_coordinates(molecule),
        "heavy_atom_count": ligand_qc["heavy_atom_count"],
        "canonical_smiles": ligand_qc["canonical_smiles"],
    }


def redocking_pose_ifps(
    prepared_receptor_pdb: Path,
    top_pose_rows: list[dict[str, Any]],
    temp_dir: Path,
    hiv_mode: bool,
) -> list[dict[str, Any]]:
    pose_ifps: list[dict[str, Any]] = []
    for row in top_pose_rows:
        if hiv_mode and row.get("pose_label") != "NNRTI_pocket":
            continue
        pose_pdb = temp_dir / f"seed_{row['seed']}_mode_{row['mode_rank']}.pdb"
        pose_pdb_from_sdf(Path(str(row["pose_sdf_path"])), pose_pdb)
        complex_pdb = temp_dir / f"seed_{row['seed']}_mode_{row['mode_rank']}_complex.pdb"
        merge_pdb_fragments([prepared_receptor_pdb, pose_pdb], complex_pdb)
        ifp = plip_ifp(complex_pdb)
        row["ifp_residue_count"] = int(len(ifp["residue_set"]))
        pose_ifps.append(ifp)
    return pose_ifps


def ensure_pose_stats_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in POSE_STATS_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame.loc[:, POSE_STATS_COLUMNS]


def ensure_hiv_qc_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in HIV_QC_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame.loc[:, HIV_QC_COLUMNS]


def expand_docking_box(docking_box: dict[str, Any], delta_a: float, attempt_index: int) -> dict[str, Any]:
    expanded = dict(docking_box)
    expanded["source"] = f"{docking_box['source']}_expanded_attempt_{attempt_index}"
    expanded["size_x"] = float(docking_box["size_x"]) + float(delta_a)
    expanded["size_y"] = float(docking_box["size_y"]) + float(delta_a)
    expanded["size_z"] = float(docking_box["size_z"]) + float(delta_a)
    expanded["expansion_delta_a"] = float(delta_a)
    expanded["attempt_index"] = int(attempt_index)
    return expanded


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage2 = config["stage2"]
    stage3_5 = config["stage3_5"]
    cases_frozen = load_yaml(root / config["stage2"]["cases_frozen_config"])
    cases = selected_cases(cases_frozen, args.case_id)
    if not cases:
        raise SystemExit(f"No Stage 3.5 case matched --case-id={args.case_id}")

    summary_rows: list[dict[str, Any]] = []

    for case in cases:
        case_id = str(case["case_id"])
        paths = stage35_paths(root, case_id)
        hiv_mode = case_id == "hiv_rt_rilpivirine"

        with tempfile.TemporaryDirectory(prefix=f"stage35_{case_id}_") as temp_dir_text:
            temp_dir = Path(temp_dir_text)
            receptor_source = root / str(case["wt_template"]["structure_path"])
            chain_id = str(case["wt_template"]["chain_id"])
            prepared_receptor_pdb = temp_dir / "prepared_receptor.pdb"
            prepared_receptor_pdbqt = temp_dir / "prepared_receptor.pdbqt"

            reference_ligand = reference_ligand_setup(root, case_id, temp_dir)

            baseline_pose_source = "crystal_pose"
            noncrystal_mode = False
            crystal_ligand_sdf = None
            crystal_ligand_pdb = None
            crystal_ifp = None
            reference_template_ifp = None
            reference_template_ifp_multichain = None
            redocking_input_sdf = None
            docking_box = None
            protein_build_meta = None
            hiv_qc_frame = pd.DataFrame(columns=HIV_QC_COLUMNS)
            nnrti_residue_coords: dict[str, list[tuple[float, float, float]]] = {}
            active_site_residue_coords: dict[str, list[tuple[float, float, float]]] = {}
            multichain_reference_summary = None

            if hiv_mode:
                baseline_pose_source = "validated_holo_crystal_pose"
                pocket_positions = [int(value) for value in stage2["hiv_pocket_positions"]]

                active_site_ligand = choose_chain_ligand(receptor_source, chain_id, preferred_residue_names=["TTP"])
                if active_site_ligand is None:
                    raise RuntimeError("HIV WT template is missing the expected TTP active-site ligand")
                active_site_coord_map = residue_atom_coordinates(receptor_source, chain_id, {active_site_ligand.residue_id})
                active_site_coords = next(iter(active_site_coord_map.values()))
                active_site_ids_wt = nearby_protein_residue_ids(
                    receptor_source,
                    chain_id,
                    active_site_coords,
                    distance_cutoff_a=float(stage3_5["hiv_pose_contact_cutoff_a"]),
                )

                holo_template_pdb = str(stage3_5["hiv_reference_holo_pdb"]).upper()
                holo_template_chain = str(stage3_5["hiv_reference_holo_chain"])
                holo_template_path = resolve_structure_cache(root, holo_template_pdb)
                holo_ligand = choose_chain_ligand(holo_template_path, holo_template_chain)
                if holo_ligand is None:
                    raise RuntimeError("HIV reference holo template is missing the NNRTI ligand")
                holo_ligand_sdf = temp_dir / f"{holo_template_pdb}_reference_ligand.sdf"
                crystal_ligand_from_template(
                    holo_template_path,
                    holo_template_chain,
                    holo_ligand,
                    reference_ligand["standardized_sdf"],
                    holo_ligand_sdf,
                    temp_dir,
                )
                holo_ligand_coord_map = residue_atom_coordinates(
                    holo_template_path,
                    holo_template_chain,
                    {holo_ligand.residue_id},
                )
                holo_ligand_coords = next(iter(holo_ligand_coord_map.values()))
                nnrti_residue_ids = {(" ", int(position), "") for position in pocket_positions}
                nnrti_residue_coords = {
                    f"{holo_template_chain}:{residue_id[1]}{residue_id[2]}": coords
                    for residue_id, coords in residue_atom_coordinates(
                        holo_template_path,
                        holo_template_chain,
                        nnrti_residue_ids,
                    ).items()
                }
                if len(nnrti_residue_coords) < len(pocket_positions):
                    raise RuntimeError("Unable to resolve the full HIV NNRTI pocket residue set on the validated holo template")
                active_site_positions = sorted({int(residue_id[1]) for residue_id in active_site_ids_wt if str(residue_id[0]).strip() == ""})
                active_site_ids_holo = {(" ", int(position), "") for position in active_site_positions}
                active_site_residue_coords = {
                    f"{holo_template_chain}:{residue_id[1]}{residue_id[2]}": coords
                    for residue_id, coords in residue_atom_coordinates(
                        holo_template_path,
                        holo_template_chain,
                        active_site_ids_holo,
                    ).items()
                }

                protein_build_meta = build_receptor_fragments(
                    holo_template_path,
                    holo_template_chain,
                    prepared_receptor_pdb,
                    paths["water_retention_report"],
                    ligand_coords=holo_ligand_coords,
                    excluded_residue_ids={holo_ligand.residue_id},
                    stage3_5=stage3_5,
                    temp_dir=temp_dir,
                    apply_pdbfixer=True,
                )
                docking_box = build_box_from_ligand_coords(
                    holo_ligand_coords,
                    default_box_size_a=float(stage3_5["default_box_size_a"]),
                    ligand_padding_a=float(stage3_5["ligand_box_padding_a"]),
                    source="hiv_validated_holo_crystal",
                )
                redocking_input_sdf = holo_ligand_sdf
                docking_receptor_pdb = prepared_receptor_pdb

                reference_holo_protein = temp_dir / "reference_holo_protein.pdb"
                reference_holo_protein_multichain = temp_dir / "reference_holo_protein_multichain.pdb"
                reference_holo_pose_pdb = temp_dir / "reference_holo_pose.pdb"
                save_chain_protein(holo_template_path, holo_template_chain, reference_holo_protein)
                save_chain_set_protein(
                    holo_template_path,
                    protein_chain_ids(holo_template_path) or [holo_template_chain],
                    reference_holo_protein_multichain,
                )
                save_residue_subset(holo_template_path, holo_template_chain, {holo_ligand.residue_id}, reference_holo_pose_pdb)
                reference_holo_complex = temp_dir / "reference_holo_complex.pdb"
                reference_holo_complex_multichain = temp_dir / "reference_holo_complex_multichain.pdb"
                merge_pdb_fragments([reference_holo_protein, reference_holo_pose_pdb], reference_holo_complex)
                merge_pdb_fragments([reference_holo_protein_multichain, reference_holo_pose_pdb], reference_holo_complex_multichain)
                reference_template_ifp = plip_ifp(reference_holo_complex)
                reference_template_ifp_multichain = plip_ifp(reference_holo_complex_multichain)
                merge_pdb_fragments([prepared_receptor_pdb, reference_holo_pose_pdb], paths["wt_complex"])
                shutil.copyfile(reference_holo_complex_multichain, paths["wt_complex_multichain"])
                shutil.copyfile(reference_holo_protein_multichain, paths["wt_receptor_stage6_multichain"])
                reference_chain_ids = protein_chain_ids(holo_template_path) or [holo_template_chain]
                partner_chain_ids = [chain for chain in reference_chain_ids if str(chain) != str(holo_template_chain)]
                multichain_pocket_residues = nearby_protein_residue_labels(
                    reference_holo_protein_multichain,
                    reference_chain_ids,
                    holo_ligand_coords,
                    distance_cutoff_a=float(stage3_5.get("hiv_partner_chain_contact_cutoff_a", 6.0)),
                )
                partner_chain_residues = sorted(
                    {
                        residue
                        for residue in multichain_pocket_residues
                        if str(residue).split(":", 1)[0] in set(partner_chain_ids)
                    }
                    | {
                        residue
                        for residue in reference_template_ifp_multichain.get("residue_set", [])
                        if str(residue).split(":", 1)[0] != str(holo_template_chain)
                    }
                )
                multichain_reference_summary = {
                    "reference_holo_pdb": str(stage3_5["hiv_reference_holo_pdb"]).upper(),
                    "reference_holo_chain": str(stage3_5["hiv_reference_holo_chain"]),
                    "reference_holo_chain_ids": reference_chain_ids,
                    "partner_chain_residues": partner_chain_residues,
                    "multichain_pocket_residues": sorted(multichain_pocket_residues),
                    "ifp": reference_template_ifp_multichain,
                }
            else:
                crystal_ligand = choose_chain_ligand(receptor_source, chain_id)
                if crystal_ligand is None:
                    noncrystal_mode = True
                    baseline_pose_source = "fpocket_redocking_pose"
                    protein_only_pdb = temp_dir / "protein_only_fpocket_input.pdb"
                    save_chain_protein(receptor_source, chain_id, protein_only_pdb)
                    pocket_coords = fpocket_top_pocket_coords(protein_only_pdb)
                    protein_build_meta = build_receptor_fragments(
                        receptor_source,
                        chain_id,
                        prepared_receptor_pdb,
                        paths["water_retention_report"],
                        ligand_coords=pocket_coords,
                        excluded_residue_ids=None,
                        stage3_5=stage3_5,
                        temp_dir=temp_dir,
                        apply_pdbfixer=True,
                    )
                    docking_box = build_box_from_residue_coords(
                        pocket_coords,
                        reference_ligand_coords=reference_ligand["coords"],
                        default_box_size_a=float(stage3_5["default_box_size_a"]),
                        ligand_padding_a=float(stage3_5["ligand_box_padding_a"]),
                        source="fpocket_top1",
                    )
                    redocking_input_sdf = reference_ligand["standardized_sdf"]
                    docking_receptor_pdb = prepared_receptor_pdb
                else:
                    crystal_ligand_sdf = temp_dir / "crystal_ligand.sdf"
                    crystal_ligand_pdb = temp_dir / "crystal_ligand.pdb"
                    crystal_ligand_from_template(
                        receptor_source,
                        chain_id,
                        crystal_ligand,
                        reference_ligand["standardized_sdf"],
                        crystal_ligand_sdf,
                        temp_dir,
                    )
                    save_residue_subset(receptor_source, chain_id, {crystal_ligand.residue_id}, crystal_ligand_pdb)
                    crystal_ligand_coord_map = residue_atom_coordinates(receptor_source, chain_id, {crystal_ligand.residue_id})
                    crystal_ligand_coords = next(iter(crystal_ligand_coord_map.values()))

                    protein_build_meta = build_receptor_fragments(
                        receptor_source,
                        chain_id,
                        prepared_receptor_pdb,
                        paths["water_retention_report"],
                        ligand_coords=crystal_ligand_coords,
                        excluded_residue_ids={crystal_ligand.residue_id},
                        stage3_5=stage3_5,
                        temp_dir=temp_dir,
                        apply_pdbfixer=True,
                    )

                    merge_pdb_fragments([prepared_receptor_pdb, crystal_ligand_pdb], paths["wt_complex"])
                    crystal_ifp = plip_ifp(paths["wt_complex"])
                    docking_box = build_box_from_ligand_coords(
                        crystal_ligand_coords,
                        default_box_size_a=float(stage3_5["default_box_size_a"]),
                        ligand_padding_a=float(stage3_5["ligand_box_padding_a"]),
                        source="crystal_ligand_centroid",
                    )
                    redocking_input_sdf = crystal_ligand_sdf
                    docking_receptor_pdb = prepared_receptor_pdb

            prepare_receptor_pdbqt(
                docking_receptor_pdb,
                prepared_receptor_pdbqt,
                ph=float(stage3_5["protein_prep_ph"]),
            )
            ligand_pdbqt = temp_dir / "redocking_input.pdbqt"
            prepare_ligand_pdbqt(redocking_input_sdf, ligand_pdbqt, ph=float(stage3_5["protein_prep_ph"]))

            seeds = [
                int(value)
                for value in (
                    stage3_5["noncrystal_redocking_seeds"]
                    if hiv_mode or noncrystal_mode
                    else stage3_5["crystal_redocking_seeds"]
                )
            ]
            baseline_ifp = reference_template_ifp if hiv_mode else crystal_ifp
            reference_summary = (
                {
                    "reference_holo_pdb": str(stage3_5["hiv_reference_holo_pdb"]).upper(),
                    "reference_holo_chain": str(stage3_5["hiv_reference_holo_chain"]),
                    "ifp": reference_template_ifp,
                }
                if hiv_mode
                else None
            )
            all_docking_rows: list[dict[str, Any]] = []
            attempt_history: list[dict[str, Any]] = []
            selected_docking_rows: list[dict[str, Any]] | None = None
            selected_top_rows: list[dict[str, Any]] = []
            selected_top_ifps: list[dict[str, Any]] = []
            selected_anchor_method: str | None = None
            selected_attempt_index: int | None = None
            best_row = None
            current_box = dict(docking_box)
            max_attempts = int(stage3_5["max_redocking_attempts"])

            for attempt_index in range(1, max_attempts + 1):
                attempt_rows = run_vina_redocking(
                    prepared_receptor_pdbqt,
                    ligand_pdbqt,
                    current_box,
                    output_root=temp_dir / f"vina_runs_attempt_{attempt_index}",
                    seeds=seeds,
                    exhaustiveness=int(stage3_5["vina_exhaustiveness"]),
                    num_modes=int(stage3_5["vina_num_modes"]),
                    energy_range=int(stage3_5["vina_energy_range"]),
                    cpu_threads=int(stage3_5["cpu_threads"]),
                )

                for row in attempt_rows:
                    row["case_id"] = case_id
                    row["pose_source"] = "redocking"
                    row["redocking_attempt"] = int(attempt_index)
                    row["docking_box_source"] = str(current_box["source"])
                    row["selected_for_anchor"] = False
                    row["selected_as_final_pose"] = False
                    row["ifp_residue_count"] = None
                    row["pose_label"] = None
                    row["nnrti_min_distance_a"] = None
                    row["nnrti_coverage_count"] = None
                    row["nnrti_coverage_fraction"] = None
                    row["active_site_min_distance_a"] = None
                    row["active_site_coverage_count"] = None
                    row["active_site_coverage_fraction"] = None
                    pose_sdf_path = Path(str(row["pose_sdf"]))
                    row["pose_sdf_path"] = str(pose_sdf_path)
                    row["pose_sdf"] = display_path(pose_sdf_path, root)
                    if crystal_ligand_sdf is not None:
                        row["rmsd_to_crystal_a"] = obrms_rmsd(crystal_ligand_sdf, pose_sdf_path) or best_rmsd(
                            crystal_ligand_sdf,
                            pose_sdf_path,
                        )
                    else:
                        row["rmsd_to_crystal_a"] = None
                    if hiv_mode:
                        metrics = classify_hiv_pose(
                            pose_sdf_path,
                            nnrti_residue_coords,
                            active_site_residue_coords,
                            contact_cutoff_a=float(stage3_5["hiv_pose_contact_cutoff_a"]),
                        )
                        row.update(metrics)

                all_docking_rows.extend(attempt_rows)

                if hiv_mode:
                    attempt_top_rows = top_pose_per_seed(attempt_rows)
                    accepted_top_rows = [
                        row for row in attempt_top_rows if row.get("pose_label") == str(stage3_5["hiv_required_pose_label"])
                    ]
                    attempt_top_ifps = redocking_pose_ifps(prepared_receptor_pdb, attempt_top_rows, temp_dir, hiv_mode=True)
                    attempt_frequency = ifp_frequency(attempt_top_ifps)
                    attempt_best_row = select_best_pose(attempt_rows, hiv_mode=True)
                    global_best = min(attempt_rows, key=lambda item: float(item["affinity_kcal_mol"]))
                    attempt_pass = (
                        attempt_best_row is not None
                        and global_best.get("pose_label") == str(stage3_5["hiv_required_pose_label"])
                        and len(accepted_top_rows) >= int(stage3_5["hiv_min_nnrti_seed_count"])
                    )
                    attempt_history.append(
                        {
                            "attempt_index": int(attempt_index),
                            "box": current_box,
                            "pose_count": int(len(attempt_rows)),
                            "top_pose_count": int(len(attempt_top_rows)),
                            "nnrti_top_pose_count": int(len(accepted_top_rows)),
                            "global_best_pose_label": global_best.get("pose_label"),
                            "best_affinity_kcal_mol": float(global_best["affinity_kcal_mol"]),
                            "overall_pass": bool(attempt_pass),
                        }
                    )
                    if attempt_pass:
                        selected_docking_rows = attempt_rows
                        selected_top_rows = attempt_top_rows
                        selected_top_ifps = attempt_top_ifps
                        selected_attempt_index = int(attempt_index)
                        best_row = attempt_best_row
                        docking_box = current_box
                        break
                elif not noncrystal_mode:
                    validation_rows = [
                        row
                        for row in attempt_rows
                        if row.get("rmsd_to_crystal_a") is not None
                        and float(row["rmsd_to_crystal_a"]) <= float(stage3_5["crystal_rmsd_threshold_a"])
                    ]
                    near_native_seed_count = len({int(row["seed"]) for row in validation_rows})
                    attempt_top_ifps = redocking_pose_ifps(prepared_receptor_pdb, validation_rows, temp_dir, hiv_mode=False)
                    attempt_anchor_method = None
                    attempt_frequency = ifp_frequency(attempt_top_ifps)
                    if attempt_top_ifps:
                        _, attempt_anchor_method, attempt_frequency = anchor_residues(
                            baseline_ifp,
                            attempt_top_ifps,
                            threshold=float(stage3_5["anchor_frequency_threshold"]),
                            require_intersection=True,
                        )
                    observed_best = min(
                        (
                            float(row["rmsd_to_crystal_a"])
                            for row in attempt_rows
                            if row.get("rmsd_to_crystal_a") is not None
                        ),
                        default=None,
                    )
                    attempt_pass = (
                        bool(validation_rows)
                        and near_native_seed_count >= int(stage3_5["crystal_min_consensus_seeds"])
                        and attempt_anchor_method == "crystal_redocking_intersection"
                    )
                    attempt_history.append(
                        {
                            "attempt_index": int(attempt_index),
                            "box": current_box,
                            "pose_count": int(len(attempt_rows)),
                            "near_native_pose_count": int(len(validation_rows)),
                            "near_native_seed_count": int(near_native_seed_count),
                            "best_rmsd_a": observed_best,
                            "anchor_method": attempt_anchor_method,
                            "overall_pass": bool(attempt_pass),
                        }
                    )
                    if attempt_pass:
                        selected_docking_rows = attempt_rows
                        selected_top_ifps = attempt_top_ifps
                        selected_anchor_method = attempt_anchor_method
                        selected_attempt_index = int(attempt_index)
                        best_row = min(
                            validation_rows,
                            key=lambda row: (
                                float(row["affinity_kcal_mol"]),
                                float(row["rmsd_to_crystal_a"]),
                                int(row["seed"]),
                                int(row["mode_rank"]),
                            ),
                        )
                        docking_box = current_box
                        break
                else:
                    attempt_top_rows = top_pose_per_seed(attempt_rows)
                    attempt_top_ifps = redocking_pose_ifps(prepared_receptor_pdb, attempt_top_rows, temp_dir, hiv_mode=False)
                    stable_anchors, attempt_anchor_method, attempt_frequency = anchor_residues(
                        {"residue_set": []},
                        attempt_top_ifps,
                        threshold=float(stage3_5["anchor_frequency_threshold"]),
                        require_intersection=False,
                    )
                    attempt_best_row = select_best_pose(attempt_rows, hiv_mode=False)
                    attempt_pass = bool(attempt_best_row is not None and stable_anchors and len(attempt_top_rows) == len(seeds))
                    attempt_history.append(
                        {
                            "attempt_index": int(attempt_index),
                            "box": current_box,
                            "pose_count": int(len(attempt_rows)),
                            "top_pose_count": int(len(attempt_top_rows)),
                            "stable_anchor_count": int(len(stable_anchors)),
                            "anchor_method": attempt_anchor_method,
                            "overall_pass": bool(attempt_pass),
                        }
                    )
                    if attempt_pass:
                        selected_docking_rows = attempt_rows
                        selected_top_rows = attempt_top_rows
                        selected_top_ifps = attempt_top_ifps
                        selected_anchor_method = attempt_anchor_method
                        selected_attempt_index = int(attempt_index)
                        best_row = attempt_best_row
                        docking_box = current_box
                        break

                if attempt_index < max_attempts:
                    current_box = expand_docking_box(
                        current_box,
                        delta_a=float(stage3_5["box_expansion_step_a"]),
                        attempt_index=attempt_index + 1,
                    )

            if selected_docking_rows is None:
                raise RuntimeError(
                    f"{case_id} Stage 3.5 redocking failed after {max_attempts} attempts: "
                    f"{json.dumps(attempt_history, ensure_ascii=True)}"
                )

            if hiv_mode:
                accepted_keys = {
                    (int(row["seed"]), int(row["mode_rank"]))
                    for row in selected_top_rows
                    if row.get("pose_label") == str(stage3_5["hiv_required_pose_label"])
                }
                for row in all_docking_rows:
                    if (
                        int(row.get("redocking_attempt") or 0) == int(selected_attempt_index)
                        and (int(row["seed"]), int(row["mode_rank"])) in accepted_keys
                    ):
                        row["selected_for_anchor"] = True
                if best_row is None:
                    raise RuntimeError("HIV Stage 3.5 failed: no NNRTI-pocket docked pose survived the strict gate")
                best_row["selected_as_final_pose"] = True
                final_pose_pdb = temp_dir / "final_hiv_pose.pdb"
                pose_pdb_from_sdf(Path(str(best_row["pose_sdf_path"])), final_pose_pdb)
                anchors, anchor_method, frequency_map = anchor_residues(
                    baseline_ifp,
                    selected_top_ifps,
                    threshold=float(stage3_5["anchor_frequency_threshold"]),
                    require_intersection=True,
                )
                hiv_qc_frame = ensure_hiv_qc_columns(pd.DataFrame.from_records(all_docking_rows))
                write_table(hiv_qc_frame, paths["hiv_pose_qc"])
            elif not noncrystal_mode:
                validation_rows = [
                    row
                    for row in selected_docking_rows
                    if row.get("rmsd_to_crystal_a") is not None
                    and float(row["rmsd_to_crystal_a"]) <= float(stage3_5["crystal_rmsd_threshold_a"])
                ]
                accepted_keys = {(int(row["seed"]), int(row["mode_rank"])) for row in validation_rows}
                for row in all_docking_rows:
                    if (
                        int(row.get("redocking_attempt") or 0) == int(selected_attempt_index)
                        and (int(row["seed"]), int(row["mode_rank"])) in accepted_keys
                    ):
                        row["selected_for_anchor"] = True
                if best_row is None:
                    raise RuntimeError(f"{case_id} Stage 3.5 failed: no near-native pose remained after selection")
                best_row["selected_as_final_pose"] = True
                anchors, anchor_method, frequency_map = anchor_residues(
                    baseline_ifp,
                    selected_top_ifps,
                    threshold=float(stage3_5["anchor_frequency_threshold"]),
                    require_intersection=True,
                )
                empty_hiv_qc(paths["hiv_pose_qc"])
            else:
                accepted_keys = {(int(row["seed"]), int(row["mode_rank"])) for row in selected_top_rows}
                for row in all_docking_rows:
                    if (
                        int(row.get("redocking_attempt") or 0) == int(selected_attempt_index)
                        and (int(row["seed"]), int(row["mode_rank"])) in accepted_keys
                    ):
                        row["selected_for_anchor"] = True
                if best_row is None:
                    raise RuntimeError(f"{case_id} Stage 3.5 failed: no high-confidence fpocket redocking pose remained")
                best_row["selected_as_final_pose"] = True
                final_pose_pdb = temp_dir / "final_noncrystal_pose.pdb"
                pose_pdb_from_sdf(Path(str(best_row["pose_sdf_path"])), final_pose_pdb)
                merge_pdb_fragments([prepared_receptor_pdb, final_pose_pdb], paths["wt_complex"])
                baseline_ifp = plip_ifp(paths["wt_complex"])
                anchors, anchor_method, frequency_map = anchor_residues(
                    baseline_ifp,
                    selected_top_ifps,
                    threshold=float(stage3_5["anchor_frequency_threshold"]),
                    require_intersection=False,
                )
                empty_hiv_qc(paths["hiv_pose_qc"])

            if selected_anchor_method is None:
                selected_anchor_method = anchor_method

            if hiv_mode and reference_template_ifp_multichain is not None:
                multichain_anchor_residues = sorted(
                    set(anchors)
                    | {
                        residue
                        for residue in reference_template_ifp_multichain.get("residue_set", [])
                        if str(residue).split(":", 1)[0] != str(stage3_5["hiv_reference_holo_chain"])
                    }
                )
                write_anchor_file(paths["wt_anchor_residues_multichain"], multichain_anchor_residues)
                write_ifp_json(
                    paths["wt_ifp_multichain"],
                    {
                        "case_id": case_id,
                        "baseline_pose_source": "validated_holo_multichain_crystal_pose",
                        "protein_prep": {
                            "ph": float(stage3_5["protein_prep_ph"]),
                            "retained_water_count": int(protein_build_meta["retained_water_count"]),
                            "retained_cofactor_count": int(protein_build_meta["retained_cofactor_count"]),
                        },
                        "reference_ligand": {
                            "source_path": display_path(reference_ligand["source_sdf"], root),
                            "standardized_path": display_path(reference_ligand["standardized_sdf"], root),
                            "heavy_atom_count": int(reference_ligand["heavy_atom_count"]),
                            "canonical_smiles": reference_ligand["canonical_smiles"],
                        },
                        "docking_box": docking_box,
                        "baseline_ifp": reference_template_ifp_multichain,
                        "redocking_anchor_frequency": frequency_map,
                        "anchor_residues": multichain_anchor_residues,
                        "anchor_method": "hiv_multichain_anchor_union",
                        "top_pose_count_used_for_anchor": len(selected_top_ifps),
                        "selected_attempt_index": selected_attempt_index,
                        "attempt_history": attempt_history,
                        "reference_template": multichain_reference_summary,
                        "partner_chain_residues": []
                        if multichain_reference_summary is None
                        else list(multichain_reference_summary.get("partner_chain_residues") or []),
                        "pocket_residue_universe": sorted(
                            set(reference_template_ifp_multichain.get("residue_set", []))
                            | set(reference_template_ifp.get("residue_set", []) if reference_template_ifp else [])
                            | set(multichain_reference_summary.get("multichain_pocket_residues", []) if multichain_reference_summary else [])
                        ),
                    },
                )

            json_dump(
                paths["docking_box"],
                {
                    **docking_box,
                    "selected_box": docking_box,
                    "attempt_history": attempt_history,
                    "selected_attempt_index": selected_attempt_index,
                },
            )

            pose_stats_rows = all_docking_rows + [
                {
                    "case_id": case_id,
                    "pose_source": baseline_pose_source,
                    "redocking_attempt": None,
                    "docking_box_source": docking_box["source"],
                    "seed": None,
                    "mode_rank": None,
                    "affinity_kcal_mol": None,
                    "rmsd_to_crystal_a": 0.0 if crystal_ligand_sdf is not None else None,
                    "pose_label": "NNRTI_pocket" if hiv_mode else ("redocking_pose" if noncrystal_mode else "crystal_pose"),
                    "selected_for_anchor": True,
                    "selected_as_final_pose": True,
                    "ifp_residue_count": int(len(baseline_ifp["residue_set"])),
                    "nnrti_min_distance_a": None,
                    "nnrti_coverage_count": None,
                    "nnrti_coverage_fraction": None,
                    "active_site_min_distance_a": None,
                    "active_site_coverage_count": None,
                    "active_site_coverage_fraction": None,
                    "pose_sdf": (
                        display_path(Path(str(best_row["pose_sdf_path"])), root)
                        if noncrystal_mode and best_row is not None
                        else (None if crystal_ligand_sdf is None else display_path(crystal_ligand_sdf, root))
                    ),
                }
            ]
            pose_stats_frame = ensure_pose_stats_columns(pd.DataFrame.from_records(pose_stats_rows))
            write_table(pose_stats_frame, paths["wt_pose_stats"])

            write_anchor_file(paths["wt_anchor_residues"], anchors)
            write_ifp_json(
                paths["wt_ifp"],
                {
                    "case_id": case_id,
                    "baseline_pose_source": baseline_pose_source,
                    "protein_prep": {
                        "ph": float(stage3_5["protein_prep_ph"]),
                        "retained_water_count": int(protein_build_meta["retained_water_count"]),
                        "retained_cofactor_count": int(protein_build_meta["retained_cofactor_count"]),
                    },
                    "reference_ligand": {
                        "source_path": display_path(reference_ligand["source_sdf"], root),
                        "standardized_path": display_path(reference_ligand["standardized_sdf"], root),
                        "heavy_atom_count": int(reference_ligand["heavy_atom_count"]),
                        "canonical_smiles": reference_ligand["canonical_smiles"],
                    },
                    "docking_box": docking_box,
                    "baseline_ifp": baseline_ifp,
                    "redocking_anchor_frequency": frequency_map,
                    "anchor_residues": anchors,
                    "anchor_method": anchor_method,
                    "top_pose_count_used_for_anchor": len(selected_top_ifps),
                    "selected_attempt_index": selected_attempt_index,
                    "attempt_history": attempt_history,
                    "reference_template": reference_summary,
                },
            )

            summary_rows.append(
                {
                    "case_id": case_id,
                    "baseline_pose_source": baseline_pose_source,
                    "wt_complex_path": display_path(paths["wt_complex"], root),
                    "anchor_count": int(len(anchors)),
                    "retained_water_count": int(protein_build_meta["retained_water_count"]),
                    "retained_cofactor_count": int(protein_build_meta["retained_cofactor_count"]),
                    "redocking_pose_count": int(len(selected_docking_rows)),
                    "redocking_attempt_count": int(len(attempt_history)),
                    "selected_attempt_index": int(selected_attempt_index),
                    "top_pose_count_used_for_anchor": int(len(selected_top_ifps)),
                    "best_redocking_affinity_kcal_mol": None if best_row is None else float(best_row["affinity_kcal_mol"]),
                    "best_redocking_rmsd_a": None if best_row is None else best_row.get("rmsd_to_crystal_a"),
                    "anchor_method": selected_anchor_method,
                    "box_expanded": bool(int(selected_attempt_index) > 1),
                    "overall_pass": True,
                }
            )

    summary_path = root / stage3_5["summary_report"]
    existing_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    case_map: dict[str, dict[str, Any]] = {}
    for row in list(existing_summary.get("cases") or []):
        if isinstance(row, dict) and row.get("case_id"):
            case_map[str(row["case_id"])] = dict(row)
    for row in summary_rows:
        case_map[str(row["case_id"])] = dict(row)
    json_dump(summary_path, {"cases": [case_map[key] for key in sorted(case_map)]})


if __name__ == "__main__":
    main()
