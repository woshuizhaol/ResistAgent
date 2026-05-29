#!/usr/bin/env python3
"""Stage 1.5 public structure/sequence/ligand assembly."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.public_data_utils import (
    chain_completeness,
    fetch_alphafold_model,
    fetch_pubchem_ligand,
    fetch_rcsb_files,
    fetch_uniprot_assets,
    load_mmcif_dict,
    load_public_projects,
    mmcif_entry_title,
    mmcif_resolution,
    normalize_sequence_text,
    pdb_chain_completeness,
    request_session,
    write_sequence_fasta,
    write_ligand_sdf,
)
from tools.runtime import ensure_dir, json_dump, load_yaml, project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def best_chain_row(chain_rows: list[dict[str, object]]) -> dict[str, object] | None:
    return max(
        chain_rows,
        key=lambda row: (
            0.0 if row["chain_completeness"] is None else float(row["chain_completeness"]),
            int(row["observed_residue_count"]),
            str(row["chain_id"]),
        ),
        default=None,
    )


def display_path(path: Path, root: Path) -> str:
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def resolve_project_sequence(
    project: dict[str, object],
    root: Path,
    raw_dir: Path,
    uniprot_assets: dict[str, object],
) -> dict[str, object]:
    explicit_fasta = project.get("fasta")
    fasta_path_value = project.get("fasta_path")
    sequence_source = "missing"
    source_path = None
    if explicit_fasta:
        sequence = normalize_sequence_text(str(explicit_fasta))
        sequence_source = "project_fasta"
    elif fasta_path_value:
        fasta_path = Path(str(fasta_path_value))
        if not fasta_path.is_absolute():
            fasta_path = root / fasta_path
        sequence = normalize_sequence_text(fasta_path.read_text(encoding="utf-8"))
        sequence_source = "project_fasta_path"
        source_path = display_path(fasta_path, root)
    else:
        sequence = str(uniprot_assets.get("sequence") or "")
        if sequence:
            sequence_source = "uniprot"

    fasta_output = None
    if sequence:
        fasta_output_path = raw_dir / "target.fasta"
        write_sequence_fasta(fasta_output_path, str(project["project_id"]), sequence)
        fasta_output = display_path(fasta_output_path, root)
    return {
        "sequence": sequence,
        "sequence_source": sequence_source,
        "source_path": source_path,
        "fasta_output_path": fasta_output,
    }


def experimental_structure_payload(
    project: dict[str, object],
    session,
    root: Path,
    stage1_5: dict[str, object],
    raw_dir: Path,
    timeout: int,
) -> dict[str, object]:
    rcsb_files = fetch_rcsb_files(
        session,
        str(project["pdb_id"]),
        root / stage1_5["rcsb_cache_root"],
        timeout,
    )
    structure_path = raw_dir / "structure.pdb"
    structure_cif_path = raw_dir / "structure.cif"
    shutil.copyfile(rcsb_files["pdb"], structure_path)
    shutil.copyfile(rcsb_files["cif"], structure_cif_path)
    cif_dict = load_mmcif_dict(rcsb_files["cif"])
    chain_rows = chain_completeness(cif_dict)
    return {
        "source": "rcsb_pdb",
        "pdb_id": str(project["pdb_id"]).upper(),
        "title": mmcif_entry_title(cif_dict),
        "resolution": mmcif_resolution(cif_dict),
        "best_chain": best_chain_row(chain_rows),
        "chain_completeness_rows": chain_rows,
        "structure_path": display_path(structure_path, root),
        "structure_cif_path": display_path(structure_cif_path, root),
        "structure_available": True,
        "modeling_required": False,
    }


def alphafold_structure_payload(
    project: dict[str, object],
    session,
    root: Path,
    stage1_5: dict[str, object],
    raw_dir: Path,
    timeout: int,
    sequence: str,
) -> dict[str, object] | None:
    model = fetch_alphafold_model(
        session,
        str(project.get("uniprot_id") or ""),
        root / stage1_5["alphafold_cache_root"],
        timeout,
    )
    if model is None:
        return None
    structure_path = raw_dir / "structure.pdb"
    shutil.copyfile(model["pdb"], structure_path)
    chain_rows = pdb_chain_completeness(structure_path, expected_sequence=sequence)
    record = dict(model["record"])
    return {
        "source": "alphafold_db",
        "pdb_id": None,
        "title": str(record.get("modelEntityId") or record.get("modelId") or "AlphaFold model"),
        "resolution": None,
        "best_chain": best_chain_row(chain_rows),
        "chain_completeness_rows": chain_rows,
        "structure_path": display_path(structure_path, root),
        "structure_cif_path": None,
        "structure_available": True,
        "modeling_required": False,
        "alphafold_record": record,
    }


def pending_modeling_payload(
    project: dict[str, object],
    root: Path,
    stage1_5: dict[str, object],
    sequence_info: dict[str, object],
) -> dict[str, object]:
    request_path = root / stage1_5["modeling_request_root"] / f"{project['project_id']}.json"
    request_payload = {
        "project_id": project["project_id"],
        "target_name": project.get("target_name"),
        "uniprot_id": project.get("uniprot_id"),
        "sequence_length": len(str(sequence_info["sequence"])),
        "sequence_source": sequence_info["sequence_source"],
        "sequence_fasta_path": sequence_info["fasta_output_path"],
        "requested_engines": ["AlphaFold", "ESMFold"],
        "pocket_localization_required": True,
        "status": "pending_modeling",
    }
    json_dump(request_path, request_payload)
    return {
        "source": "pending_modeling",
        "pdb_id": None,
        "title": None,
        "resolution": None,
        "best_chain": None,
        "chain_completeness_rows": [],
        "structure_path": None,
        "structure_cif_path": None,
        "structure_available": False,
        "modeling_required": True,
        "modeling_request_path": display_path(request_path, root),
    }


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage1_5 = config["stage1_5"]
    projects = load_public_projects(root / stage1_5["projects_root"])
    timeout = int(stage1_5["request_timeout_sec"])
    min_chain_completeness = float(stage1_5["min_chain_completeness"])
    min_ligand_heavy_atoms = int(stage1_5["min_ligand_heavy_atoms"])

    session = request_session()
    summary_rows = []

    for project in projects:
        project_id = str(project["project_id"])
        project_root_dir = root / "outputs" / project_id / "stage1_5"
        raw_dir = ensure_dir(project_root_dir / "raw")
        meta_dir = ensure_dir(project_root_dir / "meta")

        uniprot_assets = fetch_uniprot_assets(
            session,
            project.get("uniprot_id"),
            root / stage1_5["uniprot_cache_root"],
            timeout,
        )
        sequence_info = resolve_project_sequence(project, root, raw_dir, uniprot_assets)

        if project.get("pdb_id"):
            structure_payload = experimental_structure_payload(project, session, root, stage1_5, raw_dir, timeout)
        else:
            structure_payload = alphafold_structure_payload(
                project,
                session,
                root,
                stage1_5,
                raw_dir,
                timeout,
                str(sequence_info["sequence"]),
            )
            if structure_payload is None:
                structure_payload = pending_modeling_payload(project, root, stage1_5, sequence_info)

        ligand_payload = fetch_pubchem_ligand(
            session,
            project.get("ligand"),
            root / stage1_5["pubchem_cache_root"],
            timeout,
        )
        ligand_qc = write_ligand_sdf(ligand_payload, raw_dir / "ligand.sdf")

        best_chain = structure_payload["best_chain"]
        resolution = structure_payload["resolution"]
        structure_pass = (
            bool(structure_payload["structure_available"])
            and (
                resolution is not None
                or str(structure_payload["source"]) in {"alphafold_db"}
            )
            and best_chain is not None
            and best_chain["chain_completeness"] is not None
            and float(best_chain["chain_completeness"]) >= min_chain_completeness
        )
        ligand_pass = (
            ligand_qc["heavy_atom_count"] >= min_ligand_heavy_atoms and ligand_qc["fragment_count"] == 1
        )

        target_payload = {
            "project_id": project_id,
            "input": {key: value for key, value in project.items() if not key.startswith("_")},
            "structure": structure_payload,
            "sequence": {
                "uniprot_id": uniprot_assets["uniprot_id"],
                "sequence_length": len(str(sequence_info["sequence"])),
                "sequence_source": sequence_info["sequence_source"],
                "sequence_fasta_source_path": sequence_info["source_path"],
                "sequence_fasta_path": sequence_info["fasta_output_path"],
                "record": uniprot_assets["record"],
            },
            "ligand": {
                "query_name": ligand_payload.get("query_name"),
                "title": ligand_qc["title"],
                "canonical_smiles": ligand_qc["canonical_smiles"],
                "pubchem_cid": ligand_qc["pubchem_cid"],
                "heavy_atom_count": ligand_qc["heavy_atom_count"],
                "bond_count": ligand_qc["bond_count"],
                "fragment_count": ligand_qc["fragment_count"],
                "ligand_path": str((raw_dir / "ligand.sdf").relative_to(root)),
            },
            "checks": {
                "structure_available": bool(structure_payload["structure_available"]),
                "structure_resolution_ok": structure_pass,
                "chain_completeness_ok": structure_pass,
                "ligand_connectivity_ok": ligand_pass,
                "modeling_required": bool(structure_payload["modeling_required"]),
                "overall_pass": bool(structure_pass and ligand_pass),
            },
        }
        json_dump(meta_dir / "target.json", target_payload)

        summary_rows.append(
            {
                "project_id": project_id,
                "pdb_id": None if project.get("pdb_id") is None else str(project["pdb_id"]).upper(),
                "structure_source": structure_payload["source"],
                "resolution": resolution,
                "best_chain_id": None if best_chain is None else best_chain["chain_id"],
                "best_chain_completeness": None if best_chain is None else best_chain["chain_completeness"],
                "ligand_heavy_atom_count": ligand_qc["heavy_atom_count"],
                "ligand_fragment_count": ligand_qc["fragment_count"],
                "modeling_required": bool(structure_payload["modeling_required"]),
                "overall_pass": bool(structure_pass and ligand_pass),
                "target_json": str((meta_dir / "target.json").relative_to(root)),
            }
        )

    json_dump(root / stage1_5["summary_report"], {"projects": summary_rows})


if __name__ == "__main__":
    main()
