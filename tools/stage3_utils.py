#!/usr/bin/env python3
"""Shared extraction and residue-mapping helpers for Stage 3 and Stage 3.2."""

from __future__ import annotations

import gzip
import json
import math
import os
import shutil
import sqlite3
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml
from Bio import Align
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

from tools.mutation_parser import ParsedComponent, parse_component
from tools.runtime import ensure_dir, text_dump
from tools.stage0_common import classify_file_role
from tools.structure_io import validate_pdb_file, validate_sdf_file

ROLE_TO_FILENAME = {
    "WT": "WT.pdb",
    "MT": "MT.pdb",
    "WT_complex": "WT_complex.pdb",
    "MT_complex": "MT_complex.pdb",
    "ligand": "ligand.sdf",
}

CORE_ROLES = ("WT", "MT", "ligand")
OPTIONAL_ROLES = ("WT_complex", "MT_complex")
PDB_HETERO_SKIP_IDS = {
    "HOH",
    "DOD",
    "NA",
    "CL",
    "MG",
    "MN",
    "ZN",
    "CA",
    "K",
    "SO4",
    "PO4",
    "ACT",
    "EDO",
    "GOL",
    "PEG",
    "MPD",
}


@dataclass
class LocalRoleHit:
    sample_id: str
    role: str
    path: str
    source_kind: str


@dataclass
class ArchiveRoleHit:
    sample_id: str
    role: str
    archive_name: str
    archive_type: str
    member_path: str
    size_bytes: int


@dataclass
class ChainResidue:
    chain_id: str
    pdb_resnum: int
    insertion_code: str
    pdb_aa: str
    atom_count: int


def sample_list_paths(manifests_root: Path, case_id: str) -> dict[str, Path]:
    return {
        "TopRisk20": manifests_root / f"{case_id}_TopRisk20.sample_id_list.txt",
        "MediumRisk80": manifests_root / f"{case_id}_MediumRisk80.sample_id_list.txt",
        "Tail100": manifests_root / f"{case_id}_Tail100.sample_id_list.txt",
        "TopCombo20": manifests_root / f"{case_id}_TopCombo20.sample_id_list.txt",
    }


def load_case_manifests(root: Path, cases_frozen: dict[str, Any]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for case in cases_frozen.get("set_d", []):
        manifest_path = case.get("manifest_path")
        if not manifest_path:
            continue
        path = root / str(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing case manifest: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise TypeError(f"Expected mapping in {path}")
        payload["_manifest_path"] = str(path.relative_to(root))
        manifests.append(payload)
    return manifests


def sample_list_paths_from_manifest(root: Path, manifest: dict[str, Any], manifests_root: Path) -> dict[str, Path]:
    case_id = str(manifest["case_id"])
    mutation_pool = manifest.get("mutation_pool", {}) or {}
    return {
        "TopRisk20": root / str(mutation_pool.get("top_risk_sample_list", manifests_root / f"{case_id}_TopRisk20.sample_id_list.txt")),
        "MediumRisk80": root / str(mutation_pool.get("medium_sample_list", manifests_root / f"{case_id}_MediumRisk80.sample_id_list.txt")),
        "Tail100": root / str(mutation_pool.get("tail_sample_list", manifests_root / f"{case_id}_Tail100.sample_id_list.txt")),
        "TopCombo20": manifests_root / f"{case_id}_TopCombo20.sample_id_list.txt",
    }


def read_sample_id_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_component_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                decoded = json.loads(text.replace("'", '"'))
                if isinstance(decoded, list):
                    return [str(item) for item in decoded]
            except Exception:
                pass
        return [text]
    if isinstance(value, pd.Series):
        return [str(item) for item in value.tolist()]
    if hasattr(value, "tolist"):
        try:
            return [str(item) for item in value.tolist()]
        except Exception:
            pass
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_csv_with_columns(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    ensure_dir(path.parent)
    if frame.empty:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return
    missing_columns = [column for column in columns if column not in frame.columns]
    for column in missing_columns:
        frame[column] = None
    frame.loc[:, columns].to_csv(path, index=False)


def _normalize_match_token(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _type_priority(value: Any) -> int:
    order = {
        "Single Substitution": 0,
        "Multiple Substitution": 1,
        "Multiple Complex": 2,
        "Deletion": 3,
        "Insertion": 4,
        "Indel": 5,
    }
    return order.get(str(value), 99)


def selection_row_from_record(
    case_id: str,
    record: pd.Series,
    panel_names: list[str],
    selection_origin: str,
    set_n_case_id: str | None = None,
) -> dict[str, Any]:
    return {
        "case_id": str(case_id),
        "sample_id": str(record["SAMPLE_ID"]),
        "panel_names": sorted({str(value) for value in panel_names}),
        "selection_error": None,
        "selection_origin": str(selection_origin),
        "set_n_case_id": None if not set_n_case_id else str(set_n_case_id),
        "pdb_id": None if pd.isna(record["PDB_ID"]) else str(record["PDB_ID"]),
        "chain_id": None if pd.isna(record["CHAIN_ID"]) else str(record["CHAIN_ID"]),
        "type": str(record["TYPE"]),
        "mutation": str(record["MUTATION"]),
        "evaluation_unit": str(record["evaluation_unit"]),
        "combination_size": int(record["combination_size"]),
        "combination_key": None if pd.isna(record["combination_key"]) else str(record["combination_key"]),
        "component_mutations": normalize_component_list(record["component_mutations"]),
        "component_mutation_keys": normalize_component_list(record["component_mutation_keys"]),
        "uniprot_id": None if pd.isna(record["UNIPROT_ID"]) else str(record["UNIPROT_ID"]),
        "gene_symbol": None if pd.isna(record["gene_symbol"]) else str(record["gene_symbol"]),
        "drug_name": None if pd.isna(record["drug_name"]) else str(record["drug_name"]),
        "domain_type": None if pd.isna(record["domain_type"]) else str(record["domain_type"]),
        "target_domain": None if pd.isna(record["target_domain"]) else str(record["target_domain"]),
    }


def build_case_selection_table(
    root: Path,
    case_manifests: list[dict[str, Any]],
    manifests_root: Path,
    master_table: pd.DataFrame,
    supplemental_rows: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    columns = [
        "case_id",
        "sample_id",
        "panel_names",
        "selection_error",
        "selection_origin",
        "set_n_case_id",
        "pdb_id",
        "chain_id",
        "type",
        "mutation",
        "evaluation_unit",
        "combination_size",
        "combination_key",
        "component_mutations",
        "component_mutation_keys",
        "uniprot_id",
        "gene_symbol",
        "drug_name",
        "domain_type",
        "target_domain",
    ]
    selected_rows: list[dict[str, Any]] = []
    master_lookup = master_table.set_index("SAMPLE_ID", drop=False)

    for manifest in case_manifests:
        case_id = str(manifest["case_id"])
        panel_membership: dict[str, list[str]] = {}
        for panel_name, path in sample_list_paths_from_manifest(root, manifest, manifests_root).items():
            panel_membership[panel_name] = read_sample_id_list(path)

        sample_to_panels: dict[str, list[str]] = defaultdict(list)
        for panel_name, sample_ids in panel_membership.items():
            for sample_id in sample_ids:
                if panel_name not in sample_to_panels[sample_id]:
                    sample_to_panels[sample_id].append(panel_name)

        for sample_id, panels in sorted(sample_to_panels.items()):
            if sample_id not in master_lookup.index:
                selected_rows.append(
                    {
                        "case_id": case_id,
                        "sample_id": sample_id,
                        "panel_names": panels,
                        "selection_error": "sample_missing_from_master_table",
                        "selection_origin": "case_manifest",
                        "set_n_case_id": None,
                    }
                )
                continue
            record = master_lookup.loc[sample_id]
            if isinstance(record, pd.DataFrame):
                record = record.iloc[0]
            selected_rows.append(selection_row_from_record(case_id, record, panels, "case_manifest"))

    if supplemental_rows:
        selected_rows.extend(supplemental_rows)

    frame = pd.DataFrame.from_records(selected_rows, columns=columns)
    if frame.empty:
        return frame
    frame["panel_names"] = frame["panel_names"].apply(lambda values: sorted({str(value) for value in values}))
    frame = frame.sort_values(["case_id", "sample_id"]).reset_index(drop=True)
    return frame.drop_duplicates(subset=["case_id", "sample_id"], keep="first")


def write_extract_lists(
    selection_df: pd.DataFrame,
    extract_lists_root: Path,
    extract_list_names: dict[str, str],
) -> dict[str, str]:
    ensure_dir(extract_lists_root)
    output_map: dict[str, str] = {}
    for case_id, group in selection_df.groupby("case_id", dropna=False):
        output_name = extract_list_names.get(str(case_id), f"{case_id}.txt")
        output_path = extract_lists_root / output_name
        sample_ids = sorted(group["sample_id"].astype(str).unique().tolist())
        text_dump(output_path, "\n".join(sample_ids) + ("\n" if sample_ids else ""))
        output_map[str(case_id)] = str(output_path)
    return output_map


def plan_set_n_stage3_actions(
    cases_frozen: dict[str, Any],
    case_manifests: list[dict[str, Any]],
    selection_df: pd.DataFrame,
    master_table: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_lookup = {
        (_normalize_match_token(manifest.get("target_name")), _normalize_match_token(manifest.get("drug_name"))): manifest
        for manifest in case_manifests
    }
    supplemental_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    master_frame = master_table.copy()
    master_frame["PDB_ID_NORMALIZED"] = master_frame["PDB_ID"].astype(str).str.upper()
    already_selected = {
        (str(row.case_id), str(row.sample_id))
        for row in selection_df.itertuples(index=False)
        if not pd.isna(row.sample_id)
    }

    for set_n_case in cases_frozen.get("set_n", []):
        set_n_case_id = str(set_n_case["case_id"])
        pdb_id = str(set_n_case["pdb_id"]).upper()
        target_name = str(set_n_case.get("target_name") or "")
        drug_name = str(set_n_case.get("drug_name") or "")
        target_key = _normalize_match_token(target_name)
        drug_key = _normalize_match_token(drug_name)
        case_manifest = case_lookup.get((target_key, drug_key))

        matched = master_frame[master_frame["PDB_ID_NORMALIZED"].eq(pdb_id)].copy()
        exact_drug = matched[matched["drug_name"].map(_normalize_match_token).eq(drug_key)].copy() if not matched.empty else matched
        preferred = exact_drug if not exact_drug.empty else matched
        representative = None
        if not preferred.empty:
            preferred = preferred.assign(
                _type_priority=preferred["TYPE"].map(_type_priority),
                _drug_match_priority=preferred["drug_name"].map(
                    lambda value: 0 if _normalize_match_token(value) == drug_key else 1
                ),
            )
            preferred = preferred.sort_values(
                ["_drug_match_priority", "_type_priority", "SAMPLE_ID"]
            )
            representative = preferred.iloc[0]

        status = "external_test_set"
        status_note = "no_matching_mdrdb_sample_for_pdb_id"
        bridge_case_id = None
        bridge_sample_id = None
        if representative is not None and case_manifest is not None:
            bridge_case_id = str(case_manifest["case_id"])
            bridge_sample_id = str(representative["SAMPLE_ID"])
            if (bridge_case_id, bridge_sample_id) in already_selected:
                status = "already_in_stage3_selection"
                status_note = "matching_sample_already_selected"
            else:
                supplemental_rows.append(
                    selection_row_from_record(
                        bridge_case_id,
                        representative,
                        ["SetNBridge"],
                        selection_origin="set_n_bridge",
                        set_n_case_id=set_n_case_id,
                    )
                )
                already_selected.add((bridge_case_id, bridge_sample_id))
                status = "bridged_to_stage3_selection"
                status_note = "representative_sample_added_to_manifest_selection"
        elif representative is not None and case_manifest is None:
            status = "external_test_set"
            status_note = "matching_pdb_found_in_mdrdb_but_no_stage_d_case_matches_target_drug"

        report_rows.append(
            {
                "set_n_case_id": set_n_case_id,
                "pdb_id": pdb_id,
                "target_name": target_name,
                "drug_name": drug_name,
                "matching_master_rows": int(len(matched)),
                "matching_exact_drug_rows": int(len(exact_drug)),
                "bridge_case_id": bridge_case_id,
                "bridge_sample_id": bridge_sample_id,
                "status": status,
                "status_note": status_note,
            }
        )

    return supplemental_rows, report_rows


def build_hiv_allowed_sets(hiv_qc: pd.DataFrame, rt_domain_report: pd.DataFrame) -> tuple[set[str], set[tuple[str, str]]]:
    allowed_pdbs = set(hiv_qc.loc[hiv_qc["is_holo_nnrti"].astype(bool), "pdb_id"].astype(str).tolist())
    allowed_chains = {
        (str(row.pdb_id), str(row.chain_id))
        for row in rt_domain_report.itertuples(index=False)
        if bool(row.pass_rt_domain_gate)
    }
    return allowed_pdbs, allowed_chains


def apply_hiv_stage2_gate(
    selection_df: pd.DataFrame,
    hiv_qc: pd.DataFrame,
    rt_domain_report: pd.DataFrame,
) -> pd.DataFrame:
    frame = selection_df.copy()
    frame["is_excluded"] = False
    frame["exclusion_reason"] = ""
    frame["hiv_holo_whitelist_ok"] = True
    frame["hiv_rt_chain_ok"] = True
    if frame.empty:
        return frame

    allowed_pdbs, allowed_chains = build_hiv_allowed_sets(hiv_qc, rt_domain_report)

    hiv_mask = frame["case_id"].astype(str).eq("hiv_rt_rilpivirine")
    frame.loc[hiv_mask, "hiv_holo_whitelist_ok"] = frame.loc[hiv_mask, "pdb_id"].astype(str).isin(allowed_pdbs)
    frame.loc[hiv_mask, "hiv_rt_chain_ok"] = frame.loc[hiv_mask].apply(
        lambda row: (str(row["pdb_id"]), str(row["chain_id"])) in allowed_chains,
        axis=1,
    )

    holo_failed = hiv_mask & ~frame["hiv_holo_whitelist_ok"].astype(bool)
    frame.loc[holo_failed, "is_excluded"] = True
    frame.loc[holo_failed, "exclusion_reason"] = "hiv_stage2_holo_whitelist_failed"

    chain_failed = hiv_mask & frame["hiv_holo_whitelist_ok"].astype(bool) & ~frame["hiv_rt_chain_ok"].astype(bool)
    frame.loc[chain_failed, "is_excluded"] = True
    frame.loc[chain_failed, "exclusion_reason"] = "hiv_rt_domain_chain_failed"
    return frame


def index_local_structure_roots(search_roots: list[Path]) -> dict[tuple[str, str], LocalRoleHit]:
    mapping: dict[tuple[str, str], LocalRoleHit] = {}
    for root in search_roots:
        if not root.exists():
            continue
        source_kind = root.name
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            sample_id = next((part for part in path.parts if part.startswith("MdrDB") and len(part) == 11), None)
            if not sample_id:
                continue
            role = classify_file_role(str(path))
            if role not in ROLE_TO_FILENAME:
                continue
            mapping.setdefault(
                (sample_id, role),
                LocalRoleHit(sample_id=sample_id, role=role, path=str(path), source_kind=source_kind),
            )
    return mapping


def query_archive_hits(db_path: Path, sample_ids: Iterable[str]) -> dict[tuple[str, str], ArchiveRoleHit]:
    sample_ids = sorted({str(sample_id) for sample_id in sample_ids})
    if not sample_ids:
        return {}
    placeholders = ",".join("?" for _ in sample_ids)
    query = f"""
        SELECT sample_id, file_role, archive_name, archive_type, member_path, size_bytes
        FROM archive_members
        WHERE sample_id IN ({placeholders}) AND file_role IN ('WT', 'MT', 'WT_complex', 'MT_complex', 'ligand')
        ORDER BY sample_id, file_role, member_path
    """
    mapping: dict[tuple[str, str], ArchiveRoleHit] = {}
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(query, sample_ids).fetchall()
    for sample_id, role, archive_name, archive_type, member_path, size_bytes in rows:
        mapping.setdefault(
            (str(sample_id), str(role)),
            ArchiveRoleHit(
                sample_id=str(sample_id),
                role=str(role),
                archive_name=str(archive_name),
                archive_type=str(archive_type),
                member_path=str(member_path),
                size_bytes=int(size_bytes),
            ),
        )
    return mapping


def hardlink_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def archive_cache_path(sample_raw_root: Path, hit: ArchiveRoleHit) -> Path:
    return sample_raw_root / hit.sample_id / Path(hit.member_path).name


def _write_stream(stream, dst: Path) -> None:
    ensure_dir(dst.parent)
    with dst.open("wb") as handle:
        shutil.copyfileobj(stream, handle)


def extract_archive_group(archive_path: Path, archive_type: str, needed: dict[str, Path]) -> list[str]:
    extracted: list[str] = []
    if archive_type in {"tar", "tar.gz"}:
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                dst = needed.get(member.name)
                if dst is None:
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                with stream:
                    _write_stream(stream, dst)
                extracted.append(member.name)
        return extracted

    if archive_type == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            for member_name in archive.namelist():
                dst = needed.get(member_name)
                if dst is None:
                    continue
                with archive.open(member_name) as stream:
                    _write_stream(stream, dst)
                extracted.append(member_name)
        return extracted

    if archive_type == "gzip":
        member_name, dst = next(iter(needed.items()))
        with gzip.open(archive_path, "rb") as stream:
            _write_stream(stream, dst)
        extracted.append(member_name)
        return extracted

    raise ValueError(f"Unsupported archive type: {archive_type}")


def bulk_extract_archive_hits(
    root: Path,
    archive_hits: dict[tuple[str, str], ArchiveRoleHit],
    sample_raw_root: Path,
) -> dict[tuple[str, str], str]:
    resolved_raw_paths: dict[tuple[str, str], str] = {}
    grouped: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)

    for key, hit in archive_hits.items():
        raw_path = archive_cache_path(sample_raw_root, hit)
        if raw_path.exists() and raw_path.stat().st_size > 0:
            resolved_raw_paths[key] = str(raw_path)
            continue
        grouped[(hit.archive_name, hit.archive_type)][hit.member_path] = raw_path

    for (archive_name, archive_type), needed in grouped.items():
        archive_path = Path(archive_name)
        if not archive_path.is_absolute():
            archive_path = root / archive_path
        extracted_members = set(extract_archive_group(archive_path, archive_type, needed))
        for key, hit in archive_hits.items():
            if hit.archive_name != archive_name or hit.archive_type != archive_type:
                continue
            raw_path = archive_cache_path(sample_raw_root, hit)
            if raw_path.exists() and hit.member_path in extracted_members:
                resolved_raw_paths[key] = str(raw_path)

    for key, hit in archive_hits.items():
        raw_path = archive_cache_path(sample_raw_root, hit)
        if raw_path.exists() and raw_path.stat().st_size > 0:
            resolved_raw_paths[key] = str(raw_path)
    return resolved_raw_paths


def validate_standardized_role(path: Path, role: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {
            "read_ok": False,
            "exists": False,
            "size_bytes": 0,
            "error": "missing_or_empty",
        }
    if role == "ligand":
        validation = validate_sdf_file(path)
    else:
        validation = validate_pdb_file(path)
    validation["exists"] = True
    validation["size_bytes"] = int(path.stat().st_size)
    return validation


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


def load_case_target_payload(root: Path, case_id: str) -> dict[str, Any]:
    path = root / "outputs" / case_id / "stage1_5" / "meta" / "target.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_fasta_sequence(path: Path) -> str:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return "".join(line for line in lines if not line.startswith(">")).upper()


def case_uniprot_sequence(root: Path, case_id: str) -> str:
    target = load_case_target_payload(root, case_id)
    fasta_path = target.get("sequence", {}).get("sequence_fasta_path")
    if not fasta_path:
        return ""
    fasta = root / str(fasta_path)
    return read_fasta_sequence(fasta)


def residue_to_one_letter(residue_name: str) -> str | None:
    token = str(residue_name).strip().upper()
    if token == "MSE":
        return "M"
    if len(token) != 3:
        return None
    try:
        return seq1(token, custom_map={"UNK": "X"}).upper()
    except Exception:
        return None


def load_chain_residues(pdb_path: Path) -> dict[str, list[ChainResidue]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("sample", str(pdb_path))
    chains: dict[str, list[ChainResidue]] = defaultdict(list)
    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag, resseq, icode = residue.id
                if hetflag.strip():
                    continue
                aa = residue_to_one_letter(residue.resname)
                if aa is None:
                    continue
                chains[str(chain.id)].append(
                    ChainResidue(
                        chain_id=str(chain.id),
                        pdb_resnum=int(resseq),
                        insertion_code=str(icode).strip(),
                        pdb_aa=aa,
                        atom_count=sum(1 for _ in residue.get_atoms()),
                    )
                )
        break
    return {chain_id: sorted(rows, key=lambda row: (row.pdb_resnum, row.insertion_code)) for chain_id, rows in chains.items()}


def structure_ligand_coordinates(pdb_path: Path) -> list[tuple[float, float, float]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("template", str(pdb_path))
    candidate_residues: list[list[tuple[float, float, float]]] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag, _, _ = residue.id
                if not hetflag.strip():
                    continue
                residue_name = str(residue.resname).strip().upper()
                if residue_name in PDB_HETERO_SKIP_IDS:
                    continue
                atom_xyz = []
                for atom in residue.get_atoms():
                    coord = atom.get_coord()
                    atom_xyz.append((float(coord[0]), float(coord[1]), float(coord[2])))
                if len(atom_xyz) >= 5:
                    candidate_residues.append(atom_xyz)
        break

    if not candidate_residues:
        return []

    max_atoms = max(len(coords) for coords in candidate_residues)
    selected = [coords for coords in candidate_residues if len(coords) >= max(5, max_atoms // 2)]
    return [coord for coords in selected for coord in coords]


def template_pocket_residues(
    pdb_path: Path,
    chain_id: str,
    ligand_sdf_path: Path,
    distance_cutoff_a: float,
) -> list[ChainResidue]:
    ligand_xyz = structure_ligand_coordinates(pdb_path)
    if not ligand_xyz:
        return []
    cutoff_sq = float(distance_cutoff_a) ** 2
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("template", str(pdb_path))
    residues: list[ChainResidue] = []
    for model in structure:
        for chain in model:
            if str(chain.id) != str(chain_id):
                continue
            for residue in chain:
                hetflag, resseq, icode = residue.id
                if hetflag.strip():
                    continue
                aa = residue_to_one_letter(residue.resname)
                if aa is None:
                    continue
                min_distance_sq = None
                atom_count = 0
                for atom in residue.get_atoms():
                    atom_count += 1
                    atom_xyz = atom.get_coord()
                    for lig_x, lig_y, lig_z in ligand_xyz:
                        distance_sq = (
                            float(atom_xyz[0] - lig_x) ** 2
                            + float(atom_xyz[1] - lig_y) ** 2
                            + float(atom_xyz[2] - lig_z) ** 2
                        )
                        if min_distance_sq is None or distance_sq < min_distance_sq:
                            min_distance_sq = distance_sq
                if min_distance_sq is not None and min_distance_sq <= cutoff_sq:
                    residues.append(
                        ChainResidue(
                            chain_id=str(chain.id),
                            pdb_resnum=int(resseq),
                            insertion_code=str(icode).strip(),
                            pdb_aa=aa,
                            atom_count=atom_count,
                        )
                    )
            break
        break
    return sorted(residues, key=lambda row: (row.pdb_resnum, row.insertion_code))


def load_sifts_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="gzip", comment="#", low_memory=False)


def sifts_rows_for_chain(
    sifts_df: pd.DataFrame,
    pdb_id: str,
    chain_id: str,
    uniprot_id: str,
) -> pd.DataFrame:
    if sifts_df.empty:
        return sifts_df.copy()
    mask = (
        sifts_df["PDB"].astype(str).str.lower().eq(str(pdb_id).lower())
        & sifts_df["CHAIN"].astype(str).eq(str(chain_id))
        & sifts_df["SP_PRIMARY"].astype(str).eq(str(uniprot_id))
    )
    return sifts_df.loc[mask].copy()


def _sifts_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def align_chain_to_uniprot(chain_sequence: str, uniprot_sequence: str) -> dict[str, Any]:
    aligner = Align.PairwiseAligner(mode="local")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(uniprot_sequence, chain_sequence)[0]

    matches = 0
    residue_mappings: list[tuple[int, int]] = []
    for target_block, query_block in zip(alignment.aligned[0], alignment.aligned[1]):
        target_start, target_end = target_block
        query_start, query_end = query_block
        block_len = min(int(target_end - target_start), int(query_end - query_start))
        for offset in range(block_len):
            uniprot_idx = int(target_start + offset)
            chain_idx = int(query_start + offset)
            residue_mappings.append((chain_idx, uniprot_idx))
            if uniprot_sequence[uniprot_idx] == chain_sequence[chain_idx]:
                matches += 1

    identity = float(matches / len(residue_mappings)) if residue_mappings else 0.0
    query_coverage = float(len(residue_mappings) / len(chain_sequence)) if chain_sequence else 0.0
    return {
        "identity": identity,
        "query_coverage": query_coverage,
        "alignment_score": float(alignment.score),
        "index_map": residue_mappings,
    }


def hiv_rt_domain_bounds(rt_domain_report: pd.DataFrame, pdb_id: str, chain_id: str) -> dict[str, Any] | None:
    if rt_domain_report.empty:
        return None
    mask = (
        rt_domain_report["case_id"].astype(str).eq("hiv_rt_rilpivirine")
        & rt_domain_report["pdb_id"].astype(str).str.upper().eq(str(pdb_id).upper())
        & rt_domain_report["chain_id"].astype(str).eq(str(chain_id))
        & rt_domain_report["pass_rt_domain_gate"].astype(bool)
    )
    rows = rt_domain_report.loc[mask].copy()
    if rows.empty:
        return None
    if "selected_best_chain" in rows.columns:
        rows = rows.assign(
            _selected_chain_priority=rows["selected_best_chain"].astype(bool).map(lambda value: 0 if value else 1),
            _effective_score_rank=rows["sequence_effective_score"].fillna(0).map(lambda value: -float(value)),
        ).sort_values(["_selected_chain_priority", "_effective_score_rank", "chain_id"])
    row = rows.iloc[0]
    start = row.get("sequence_polyprotein_start")
    end = row.get("sequence_polyprotein_end")
    if pd.isna(start) or pd.isna(end):
        return None
    return {
        "polyprotein_start": int(start),
        "polyprotein_end": int(end),
        "sequence_best_domain": row.get("sequence_best_domain"),
        "selected_best_chain": bool(row.get("selected_best_chain", False)),
    }


def map_chain_residues(
    residues: list[ChainResidue],
    pdb_id: str,
    chain_id: str,
    uniprot_id: str,
    uniprot_sequence: str,
    sifts_chain_df: pd.DataFrame,
    sifts_observed_df: pd.DataFrame,
) -> dict[str, Any]:
    mapping_rows: list[dict[str, Any]] = []
    mapping_by_uniprot: dict[int, dict[str, Any]] = {}
    mapping_collisions: list[dict[str, Any]] = []
    chain_ranges: list[tuple[int, int]] = []
    observed_ranges: list[tuple[int, int]] = []
    chain_sequence = "".join(residue.pdb_aa for residue in residues)

    chain_segments = sifts_rows_for_chain(sifts_chain_df, pdb_id, chain_id, uniprot_id)
    observed_segments = sifts_rows_for_chain(sifts_observed_df, pdb_id, chain_id, uniprot_id)
    usable_chain_segments = []
    if not chain_segments.empty:
        for row in chain_segments.itertuples(index=False):
            pdb_beg = _sifts_int(row.PDB_BEG)
            pdb_end = _sifts_int(row.PDB_END)
            if pdb_beg is None or pdb_end is None:
                usable_chain_segments = []
                break
            usable_chain_segments.append((row, pdb_beg, pdb_end))

    if usable_chain_segments:
        for row in chain_segments.itertuples(index=False):
            chain_ranges.append((int(row.SP_BEG), int(row.SP_END)))
        for row in observed_segments.itertuples(index=False):
            observed_ranges.append((int(row.SP_BEG), int(row.SP_END)))
        if not observed_ranges:
            observed_ranges = list(chain_ranges)
        for residue in residues:
            mapped = False
            if residue.insertion_code:
                confidence = "unmapped_insertion_code"
                uniprot_pos = None
                uniprot_aa = None
            else:
                confidence = "unmapped"
                uniprot_pos = None
                uniprot_aa = None
                for row, pdb_beg, pdb_end in usable_chain_segments:
                    if pdb_beg <= residue.pdb_resnum <= pdb_end:
                        uniprot_pos = int(row.SP_BEG) + int(residue.pdb_resnum - pdb_beg)
                        if 1 <= uniprot_pos <= len(uniprot_sequence):
                            uniprot_aa = uniprot_sequence[uniprot_pos - 1]
                        confidence = "sifts_linear"
                        mapped = True
                        break
            row_payload = {
                "chain_id": chain_id,
                "pdb_resnum": residue.pdb_resnum,
                "insertion_code": residue.insertion_code,
                "pdb_aa": residue.pdb_aa,
                "uniprot_pos": uniprot_pos,
                "uniprot_aa": uniprot_aa,
                "confidence": confidence,
            }
            mapping_rows.append(row_payload)
            if mapped and uniprot_pos is not None:
                existing = mapping_by_uniprot.get(int(uniprot_pos))
                if existing is None:
                    mapping_by_uniprot[int(uniprot_pos)] = row_payload
                elif (
                    int(existing["pdb_resnum"]) != residue.pdb_resnum
                    or str(existing["insertion_code"]) != residue.insertion_code
                    or str(existing["pdb_aa"]) != residue.pdb_aa
                ):
                    mapping_collisions.append(
                        {
                            "uniprot_pos": int(uniprot_pos),
                            "first_pdb_resnum": int(existing["pdb_resnum"]),
                            "first_insertion_code": str(existing["insertion_code"]),
                            "second_pdb_resnum": residue.pdb_resnum,
                            "second_insertion_code": residue.insertion_code,
                        }
                    )
        return {
            "mapping_source": "sifts_linear",
            "mapping_rows": mapping_rows,
            "mapping_by_uniprot": mapping_by_uniprot,
            "chain_ranges": chain_ranges,
            "observed_ranges": observed_ranges,
            "alignment_identity": None,
            "alignment_query_coverage": None,
            "mapping_collisions": mapping_collisions,
            "unique_mapping_ok": not mapping_collisions,
        }

    alignment = align_chain_to_uniprot(chain_sequence, uniprot_sequence)
    for chain_index, residue in enumerate(residues):
        uniprot_pos = None
        uniprot_aa = None
        for residue_index, uniprot_index in alignment["index_map"]:
            if residue_index == chain_index:
                uniprot_pos = int(uniprot_index + 1)
                uniprot_aa = uniprot_sequence[uniprot_index]
                break
        row_payload = {
            "chain_id": chain_id,
            "pdb_resnum": residue.pdb_resnum,
            "insertion_code": residue.insertion_code,
            "pdb_aa": residue.pdb_aa,
            "uniprot_pos": uniprot_pos,
            "uniprot_aa": uniprot_aa,
            "confidence": "sequence_alignment" if uniprot_pos is not None else "unmapped",
        }
        mapping_rows.append(row_payload)
        if uniprot_pos is not None:
            existing = mapping_by_uniprot.get(uniprot_pos)
            if existing is None:
                mapping_by_uniprot[uniprot_pos] = row_payload
            elif (
                int(existing["pdb_resnum"]) != residue.pdb_resnum
                or str(existing["insertion_code"]) != residue.insertion_code
                or str(existing["pdb_aa"]) != residue.pdb_aa
            ):
                mapping_collisions.append(
                    {
                        "uniprot_pos": int(uniprot_pos),
                        "first_pdb_resnum": int(existing["pdb_resnum"]),
                        "first_insertion_code": str(existing["insertion_code"]),
                        "second_pdb_resnum": residue.pdb_resnum,
                        "second_insertion_code": residue.insertion_code,
                    }
                )

    if mapping_by_uniprot:
        mapped_positions = sorted(mapping_by_uniprot)
        chain_ranges.append((mapped_positions[0], mapped_positions[-1]))
        observed_ranges.append((mapped_positions[0], mapped_positions[-1]))
    return {
        "mapping_source": "sequence_alignment",
        "mapping_rows": mapping_rows,
        "mapping_by_uniprot": mapping_by_uniprot,
        "chain_ranges": chain_ranges,
        "observed_ranges": observed_ranges,
        "alignment_identity": alignment["identity"],
        "alignment_query_coverage": alignment["query_coverage"],
        "mapping_collisions": mapping_collisions,
        "unique_mapping_ok": not mapping_collisions,
    }


def position_in_ranges(position: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= position <= end for start, end in ranges)


def component_status(
    component: ParsedComponent,
    mapping_by_uniprot: dict[int, dict[str, Any]],
    chain_ranges: list[tuple[int, int]],
    observed_ranges: list[tuple[int, int]],
) -> dict[str, Any]:
    if component.start_pos is None:
        return {"status": "UNMAPPED", "mapped_positions": [], "observed_aa": None}

    target_positions = list(range(int(component.start_pos), int(component.end_pos or component.start_pos) + 1))
    if not all(position_in_ranges(position, chain_ranges) for position in target_positions):
        return {"status": "UNMAPPED", "mapped_positions": [], "observed_aa": None}
    if not all(position_in_ranges(position, observed_ranges) for position in target_positions):
        return {"status": "MISSING_RESIDUE", "mapped_positions": [], "observed_aa": None}

    mapped_positions = [mapping_by_uniprot.get(position) for position in target_positions]
    if any(position is None for position in mapped_positions):
        return {"status": "MISSING_RESIDUE", "mapped_positions": [], "observed_aa": None}

    first_row = mapped_positions[0]
    observed_aa = first_row["pdb_aa"]
    if component.mutation_class == "single_substitution" and component.alt_aa:
        if observed_aa == component.alt_aa:
            status = "MUTATED_IN_PDB"
        else:
            status = "PRESENT"
    else:
        status = "PRESENT"

    return {
        "status": status,
        "mapped_positions": [int(position) for position in target_positions],
        "observed_aa": observed_aa,
        "pdb_resnum": int(first_row["pdb_resnum"]),
        "insertion_code": str(first_row["insertion_code"]),
    }


def component_rows_for_sample(
    row: pd.Series,
    mapping_by_uniprot: dict[int, dict[str, Any]],
    chain_ranges: list[tuple[int, int]],
    observed_ranges: list[tuple[int, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    component_rows: list[dict[str, Any]] = []
    statuses: list[str] = []
    for component_text, component_key in zip(row["component_mutations"], row["component_mutation_keys"]):
        component = parse_component(str(component_text))
        status_payload = component_status(component, mapping_by_uniprot, chain_ranges, observed_ranges)
        statuses.append(str(status_payload["status"]))
        component_rows.append(
            {
                "case_id": row["case_id"],
                "sample_id": row["sample_id"],
                "pdb_id": row["pdb_id"],
                "chain_id": row["chain_id"],
                "component_mutation": str(component_text),
                "component_mutation_key": str(component_key),
                "mutation_class": component.mutation_class,
                "component_start_pos": component.start_pos,
                "component_end_pos": component.end_pos,
                "component_ref_aa": component.ref_aa,
                "component_alt_aa": component.alt_aa,
                "component_site_status": status_payload["status"],
                "mapped_positions": status_payload.get("mapped_positions", []),
                "observed_aa": status_payload.get("observed_aa"),
                "mapped_pdb_resnum": status_payload.get("pdb_resnum"),
                "mapped_insertion_code": status_payload.get("insertion_code"),
            }
        )

    all_components_mapped = all(status in {"PRESENT", "MUTATED_IN_PDB"} for status in statuses)
    if all(status == "MUTATED_IN_PDB" for status in statuses):
        sample_status = "MUTATED_IN_PDB"
    elif all_components_mapped:
        sample_status = "PRESENT"
    elif any(status == "MISSING_RESIDUE" for status in statuses):
        sample_status = "MISSING_RESIDUE"
    else:
        sample_status = "UNMAPPED"

    summary = {
        "mutation_site_status": sample_status,
        "all_components_mapped": bool(all_components_mapped),
        "component_statuses": statuses,
    }
    return component_rows, summary


def sentinel_rows_for_sample(
    row: pd.Series,
    mapping_by_uniprot: dict[int, dict[str, Any]],
    sentinel_positions: list[int],
    uniprot_sequence: str,
) -> list[dict[str, Any]]:
    results = []
    for position in sentinel_positions:
        mapped = mapping_by_uniprot.get(int(position))
        uniprot_aa = uniprot_sequence[position - 1] if 1 <= position <= len(uniprot_sequence) else None
        status = "PASS" if mapped is not None and mapped.get("uniprot_aa") == uniprot_aa else "FAIL"
        results.append(
            {
                "case_id": row["case_id"],
                "sample_id": row["sample_id"],
                "pdb_id": row["pdb_id"],
                "chain_id": row["chain_id"],
                "sentinel_pos": int(position),
                "expected_uniprot_aa": uniprot_aa,
                "mapped_pdb_resnum": None if mapped is None else int(mapped["pdb_resnum"]),
                "mapped_insertion_code": None if mapped is None else str(mapped["insertion_code"]),
                "mapped_pdb_aa": None if mapped is None else str(mapped["pdb_aa"]),
                "status": status,
            }
        )
    return results


def serialize_mapping_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return to_jsonable(payload)


def ensure_parent(path: Path) -> Path:
    return ensure_dir(path.parent)
