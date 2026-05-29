#!/usr/bin/env python3
"""mutation-proposal step mutation proposal helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from Bio.PDB import PDBParser, ShrakeRupley, Superimposer

from tools.mutation_parser import parse_component
from tools.runtime import ensure_dir, json_dump, sha256_file
from tools.stage35_utils import (
    build_box_from_ligand_coords,
    classify_hiv_pose,
    choose_chain_ligand,
    crystal_ligand_from_template,
    extract_ligand_to_sdf,
    load_rdkit_molecule,
    load_structure,
    merge_pdb_fragments,
    min_distance_between_sets,
    mol_coordinates,
    nearby_protein_residue_ids,
    plip_ifp,
    prepare_ligand_pdbqt,
    prepare_receptor_pdbqt,
    residue_atom_coordinates,
    run_pdbfixer,
    run_vina_redocking,
    save_chain_protein,
    save_residue_subset,
    select_best_pose,
    standardize_reference_ligand,
)

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AA1_TO_3 = {value: key for key, value in AA3_TO_1.items()}

AA_MAX_ASA = {
    "ALA": 121.0,
    "ARG": 265.0,
    "ASN": 187.0,
    "ASP": 187.0,
    "CYS": 148.0,
    "GLN": 214.0,
    "GLU": 214.0,
    "GLY": 97.0,
    "HIS": 216.0,
    "ILE": 195.0,
    "LEU": 191.0,
    "LYS": 230.0,
    "MET": 203.0,
    "PHE": 228.0,
    "PRO": 154.0,
    "SER": 143.0,
    "THR": 163.0,
    "TRP": 264.0,
    "TYR": 255.0,
    "VAL": 165.0,
}

AA_PROPERTIES = {
    "A": {"hydropathy": 1.8, "volume": 88.6, "charge": 0.0, "aromatic": 0.0, "hbond": 0.0},
    "R": {"hydropathy": -4.5, "volume": 173.4, "charge": 1.0, "aromatic": 0.0, "hbond": 1.0},
    "N": {"hydropathy": -3.5, "volume": 114.1, "charge": 0.0, "aromatic": 0.0, "hbond": 1.0},
    "D": {"hydropathy": -3.5, "volume": 111.1, "charge": -1.0, "aromatic": 0.0, "hbond": 1.0},
    "C": {"hydropathy": 2.5, "volume": 108.5, "charge": 0.0, "aromatic": 0.0, "hbond": 0.0},
    "Q": {"hydropathy": -3.5, "volume": 143.8, "charge": 0.0, "aromatic": 0.0, "hbond": 1.0},
    "E": {"hydropathy": -3.5, "volume": 138.4, "charge": -1.0, "aromatic": 0.0, "hbond": 1.0},
    "G": {"hydropathy": -0.4, "volume": 60.1, "charge": 0.0, "aromatic": 0.0, "hbond": 0.0},
    "H": {"hydropathy": -3.2, "volume": 153.2, "charge": 0.5, "aromatic": 1.0, "hbond": 1.0},
    "I": {"hydropathy": 4.5, "volume": 166.7, "charge": 0.0, "aromatic": 0.0, "hbond": 0.0},
    "L": {"hydropathy": 3.8, "volume": 166.7, "charge": 0.0, "aromatic": 0.0, "hbond": 0.0},
    "K": {"hydropathy": -3.9, "volume": 168.6, "charge": 1.0, "aromatic": 0.0, "hbond": 1.0},
    "M": {"hydropathy": 1.9, "volume": 162.9, "charge": 0.0, "aromatic": 0.0, "hbond": 0.0},
    "F": {"hydropathy": 2.8, "volume": 189.9, "charge": 0.0, "aromatic": 1.0, "hbond": 0.0},
    "P": {"hydropathy": -1.6, "volume": 112.7, "charge": 0.0, "aromatic": 0.0, "hbond": 0.0},
    "S": {"hydropathy": -0.8, "volume": 89.0, "charge": 0.0, "aromatic": 0.0, "hbond": 1.0},
    "T": {"hydropathy": -0.7, "volume": 116.1, "charge": 0.0, "aromatic": 0.0, "hbond": 1.0},
    "W": {"hydropathy": -0.9, "volume": 227.8, "charge": 0.0, "aromatic": 1.0, "hbond": 0.0},
    "Y": {"hydropathy": -1.3, "volume": 193.6, "charge": 0.0, "aromatic": 1.0, "hbond": 1.0},
    "V": {"hydropathy": 4.2, "volume": 140.0, "charge": 0.0, "aromatic": 0.0, "hbond": 0.0},
}

ANCHOR_LABEL_RE = re.compile(r"^(?P<chain>[^:]+):(?P<resname>[A-Z]{3})(?P<resnum>-?\d+)(?P<icode>[A-Za-z]?)$")
PLIP_LABEL_RE = re.compile(r"^(?P<chain>[^:]+):(?P<resname>[A-Z]{3})(?P<resnum>-?\d+)$")
ALPHAFOLD_PREDICTION_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
CONSERVATION_CACHE_VERSION = "stage4_conservation_v1"
PROXY_CACHE_VERSION = "stage4_proxy_v3"
COMBO_MODEL_VERSION = "stage4_combo_model_v1"
AA_ENTROPY_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class MutationComponent:
    raw: str
    position: int | None
    ref_aa: str | None
    alt_aa: str | None
    mutation_class: str


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if pd.isna(value):
        return None
    return float(value)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def rank_normalize(values: pd.Series) -> pd.Series:
    series = values.astype(float)
    valid = series.dropna()
    result = pd.Series(0.0, index=series.index, dtype=float)
    if valid.empty:
        return result
    if len(valid) == 1:
        result.loc[valid.index] = 0.5
        return result
    ranks = valid.rank(method="average", ascending=True)
    result.loc[valid.index] = (ranks - 1.0) / float(len(valid) - 1)
    return result


def ndcg_at_k(frame: pd.DataFrame, score_col: str, relevance_col: str, k: int) -> float:
    rows = frame[[score_col, relevance_col]].dropna(subset=[score_col]).sort_values(score_col, ascending=False).head(k)
    if rows.empty:
        return 0.0
    gains = [float(value) for value in rows[relevance_col].tolist()]
    dcg = sum((2.0**gain - 1.0) / math.log2(index + 2.0) for index, gain in enumerate(gains))
    ideal = sorted([float(value) for value in frame[relevance_col].fillna(0.0).tolist()], reverse=True)[:k]
    idcg = sum((2.0**gain - 1.0) / math.log2(index + 2.0) for index, gain in enumerate(ideal))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def target_position_from_row(row: dict[str, Any], numbering_system: str, rt_offset: int | None) -> int | None:
    uniprot_pos = row.get("uniprot_pos")
    if uniprot_pos is None:
        return None
    position = int(uniprot_pos)
    if str(numbering_system) == "rt_relative":
        if rt_offset is None:
            return None
        return int(position - rt_offset)
    return position


def residue_key_from_row(row: dict[str, Any]) -> tuple[str, int, str]:
    return (" ", int(row["pdb_resnum"]), str(row.get("insertion_code") or "").strip())


def residue_chain_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row["chain_id"]), int(row["pdb_resnum"]), str(row.get("insertion_code") or "").strip())


def parse_mutation_components(value: str | None) -> list[MutationComponent]:
    if value is None or not str(value).strip():
        return []
    text = str(value).strip()
    if ":" in text:
        text = text.split(":", 1)[1]
    parts = [token.strip() for token in text.split("+") if token.strip()]
    components: list[MutationComponent] = []
    for token in parts:
        parsed = parse_component(token)
        components.append(
            MutationComponent(
                raw=parsed.raw,
                position=parsed.start_pos,
                ref_aa=parsed.ref_aa,
                alt_aa=parsed.alt_aa,
                mutation_class=parsed.mutation_class,
            )
        )
    return components


def _sha256_json_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def physchem_severity(ref_aa: str | None, alt_aa: str | None, mutation_class: str) -> float:
    if mutation_class != "single_substitution":
        return 1.0
    if not ref_aa or not alt_aa or ref_aa == alt_aa:
        return 0.0
    lhs = AA_PROPERTIES.get(ref_aa)
    rhs = AA_PROPERTIES.get(alt_aa)
    if lhs is None or rhs is None:
        return 0.75
    hydropathy = abs(lhs["hydropathy"] - rhs["hydropathy"]) / 9.0
    volume = abs(lhs["volume"] - rhs["volume"]) / 170.0
    charge = abs(lhs["charge"] - rhs["charge"]) / 2.0
    aromatic = abs(lhs["aromatic"] - rhs["aromatic"])
    hbond = abs(lhs["hbond"] - rhs["hbond"])
    return float(min(1.0, 0.25 * hydropathy + 0.2 * volume + 0.2 * charge + 0.2 * aromatic + 0.15 * hbond))


def parse_secondary_structure_map(pdb_path: Path, chain_id: str) -> dict[tuple[int, str], str]:
    mapping: dict[tuple[int, str], str] = {}
    for line in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        record = line[:6].strip()
        if record == "HELIX":
            start_chain = line[19].strip()
            end_chain = line[31].strip()
            if start_chain != chain_id or end_chain != chain_id:
                continue
            start_seq = int(line[21:25].strip())
            end_seq = int(line[33:37].strip())
            for resnum in range(start_seq, end_seq + 1):
                mapping[(resnum, "")] = "helix"
        elif record == "SHEET":
            start_chain = line[21].strip()
            end_chain = line[32].strip()
            if start_chain != chain_id or end_chain != chain_id:
                continue
            start_seq = int(line[22:26].strip())
            end_seq = int(line[33:37].strip())
            for resnum in range(start_seq, end_seq + 1):
                mapping[(resnum, "")] = "sheet"
    return mapping


def compute_relative_sasa_map(pdb_path: Path, chain_id: str) -> dict[tuple[str, int, str], float]:
    structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    ShrakeRupley().compute(structure, level="R")
    model = next(structure.get_models())
    chain = model[str(chain_id)]
    mapping: dict[tuple[str, int, str], float] = {}
    for residue in chain:
        hetflag, resseq, icode = residue.id
        if str(hetflag).strip():
            continue
        residue_name = str(residue.resname).strip().upper()
        maximum = AA_MAX_ASA.get(residue_name)
        if maximum is None:
            continue
        sasa = float(getattr(residue, "sasa", 0.0))
        mapping[(" ", int(resseq), str(icode).strip())] = float(min(1.5, sasa / maximum))
    return mapping


def solvent_class(relative_sasa: float | None) -> str:
    if relative_sasa is None:
        return "unknown"
    if relative_sasa >= 0.25:
        return "exposed"
    if relative_sasa >= 0.1:
        return "intermediate"
    return "buried"


def position_lookup_from_rows(
    rows: list[dict[str, Any]],
    numbering_system: str,
    rt_offset: int | None,
) -> tuple[dict[int, dict[str, Any]], dict[tuple[str, int, str], int], dict[tuple[str, int], int]]:
    by_position: dict[int, dict[str, Any]] = {}
    by_key: dict[tuple[str, int, str], int] = {}
    by_plip: dict[tuple[str, int], int] = {}
    for row in rows:
        position = target_position_from_row(row, numbering_system, rt_offset)
        if position is None:
            continue
        by_position[int(position)] = row
        chain_id = str(row["chain_id"])
        pdb_resnum = int(row["pdb_resnum"])
        insertion_code = str(row.get("insertion_code") or "").strip()
        by_key[(chain_id, pdb_resnum, insertion_code)] = int(position)
        by_plip[(chain_id, pdb_resnum)] = int(position)
    return by_position, by_key, by_plip


def anchor_positions_from_file(
    anchor_path: Path,
    residue_rows: list[dict[str, Any]],
    numbering_system: str,
    rt_offset: int | None,
) -> set[int]:
    _, by_key, _ = position_lookup_from_rows(residue_rows, numbering_system, rt_offset)
    positions: set[int] = set()
    if not anchor_path.exists():
        return positions
    for line in anchor_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text:
            continue
        match = ANCHOR_LABEL_RE.match(text)
        if not match:
            continue
        key = (
            match.group("chain"),
            int(match.group("resnum")),
            str(match.group("icode") or "").strip(),
        )
        position = by_key.get(key)
        if position is not None:
            positions.add(int(position))
    return positions


def plip_positions_from_ifp(
    ifp_payload: dict[str, Any],
    residue_rows: list[dict[str, Any]],
    numbering_system: str,
    rt_offset: int | None,
) -> set[int]:
    _, _, by_plip = position_lookup_from_rows(residue_rows, numbering_system, rt_offset)
    positions: set[int] = set()
    for label in ifp_payload.get("residue_set", []):
        match = PLIP_LABEL_RE.match(str(label))
        if not match:
            continue
        position = by_plip.get((match.group("chain"), int(match.group("resnum"))))
        if position is not None:
            positions.add(int(position))
    return positions


def _available_chain_ids(pdb_path: Path) -> list[str]:
    chains: set[str] = set()
    for line in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain_id = line[21].strip()
        if chain_id:
            chains.add(chain_id)
    return sorted(chains)


def _effective_residue_rows(
    residue_rows: list[dict[str, Any]],
    preferred_chain_id: str,
    effective_chain_id: str,
) -> list[dict[str, Any]]:
    if preferred_chain_id == effective_chain_id:
        return residue_rows
    remapped: list[dict[str, Any]] = []
    for row in residue_rows:
        item = dict(row)
        if str(item.get("chain_id") or "") == str(preferred_chain_id):
            item["chain_id"] = effective_chain_id
        remapped.append(item)
    return remapped


def _chain_reference_aa_lookup(pdb_path: Path, chain_id: str) -> dict[tuple[int, str], str]:
    structure = load_structure(pdb_path)
    model = next(structure.get_models())
    chain = model[str(chain_id)]
    lookup: dict[tuple[int, str], str] = {}
    for residue in chain:
        hetflag, resseq, icode = residue.id
        if str(hetflag).strip():
            continue
        aa = AA3_TO_1.get(str(residue.resname).strip().upper())
        if aa is None:
            continue
        lookup[(int(resseq), str(icode).strip())] = aa
    return lookup


def _reference_holo_position_contacts(
    reference_holo_path: Path,
    ligand_chain_id: str,
) -> dict[int, dict[str, Any]]:
    ligand = choose_chain_ligand(reference_holo_path, ligand_chain_id)
    if ligand is None:
        raise RuntimeError(f"Unable to resolve reference holo ligand from {reference_holo_path}")
    ligand_coord_map = residue_atom_coordinates(reference_holo_path, ligand_chain_id, {ligand.residue_id})
    ligand_coords = next(iter(ligand_coord_map.values()))
    structure = load_structure(reference_holo_path)
    model = next(structure.get_models())
    contacts: dict[int, dict[str, Any]] = {}
    for chain in model:
        for residue in chain:
            hetflag, resseq, icode = residue.id
            if str(hetflag).strip():
                continue
            residue_coords = [
                (float(coord[0]), float(coord[1]), float(coord[2]))
                for coord in (atom.get_coord() for atom in residue.get_atoms())
            ]
            min_distance = min_distance_between_sets(residue_coords, ligand_coords)
            if min_distance is None:
                continue
            position = int(resseq)
            current = contacts.get(position)
            if current is None or float(min_distance) < float(current["min_ligand_distance_a"]):
                contacts[position] = {
                    "chain_id": str(chain.id),
                    "pdb_resnum": int(resseq),
                    "insertion_code": str(icode).strip(),
                    "min_ligand_distance_a": float(min_distance),
                }
    return contacts


def _sample_reference_rows_for_chain(
    sample_payloads: list[dict[str, Any]],
    *,
    effective_chain_id: str,
    numbering_system: str,
    wt_complex_path: Path,
) -> tuple[list[dict[str, Any]], int | None]:
    chain_aa_lookup = _chain_reference_aa_lookup(wt_complex_path, effective_chain_id)
    best_rows: list[dict[str, Any]] = []
    best_rt_offset: int | None = None
    best_score: tuple[int, int, int, int, str] | None = None
    for sample in sample_payloads:
        sample_chain_id = str(sample.get("resolved_chain_id") or sample.get("chain_id") or "")
        if sample_chain_id != effective_chain_id:
            continue
        if str(sample.get("target_numbering_system") or numbering_system) != numbering_system:
            continue
        sample_rt_offset = None if sample.get("rt_numbering_offset") is None else int(sample["rt_numbering_offset"])
        residues = list(sample.get("residues") or [])
        if not residues:
            continue
        normalized_rows: list[dict[str, Any]] = []
        match_count = 0
        cover_count = 0
        for row in residues:
            item = dict(row)
            item["chain_id"] = effective_chain_id
            ref_aa = str(item.get("uniprot_aa") or item.get("pdb_aa") or "").upper()
            if ref_aa:
                item["pdb_aa"] = ref_aa
            key = (int(item["pdb_resnum"]), str(item.get("insertion_code") or "").strip())
            actual_aa = chain_aa_lookup.get(key)
            if actual_aa is not None:
                cover_count += 1
                if not ref_aa or actual_aa == ref_aa:
                    match_count += 1
            normalized_rows.append(item)
        offset_score = -1 if sample_rt_offset is None else sample_rt_offset
        score = (match_count, cover_count, len(normalized_rows), offset_score, -len(str(sample.get("sample_id") or "")))
        if best_score is None or score > best_score:
            best_score = score
            best_rows = normalized_rows
            best_rt_offset = sample_rt_offset
    return best_rows, best_rt_offset


def build_case_context(
    case_id: str,
    wt_complex_path: Path,
    residue_map_path: Path,
    anchor_path: Path,
    wt_ifp_path: Path,
    pocket_cutoff_a: float,
    second_shell_cutoff_a: float,
    root: Path | None = None,
    hiv_reference_holo_pdb: str | None = None,
    hiv_reference_holo_chain: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(residue_map_path.read_text(encoding="utf-8"))
    wt_template = payload["wt_template"]
    sample_payloads = list(payload.get("samples") or [])
    residue_rows = wt_template["residues"]
    preferred_chain_id = str(wt_template["resolved_chain_id"])
    available_chain_ids = _available_chain_ids(wt_complex_path)
    chain_id = preferred_chain_id
    chain_id_source = "residue_map"
    if chain_id not in available_chain_ids:
        if len(available_chain_ids) == 1:
            chain_id = available_chain_ids[0]
            chain_id_source = "wt_complex_single_chain_fallback"
        else:
            raise RuntimeError(
                f"{case_id}: resolved_chain_id={preferred_chain_id} not found in {wt_complex_path.name}; "
                f"available_chains={available_chain_ids}"
            )
    numbering_system = str(wt_template["target_numbering_system"])
    rt_offset = wt_template.get("rt_numbering_offset")
    if numbering_system == "rt_relative" and rt_offset is None and wt_template.get("rt_polyprotein_start") is not None:
        rt_offset = int(wt_template["rt_polyprotein_start"]) - 1
    if chain_id_source == "wt_complex_single_chain_fallback":
        sample_reference_rows, sample_rt_offset = _sample_reference_rows_for_chain(
            sample_payloads,
            effective_chain_id=chain_id,
            numbering_system=numbering_system,
            wt_complex_path=wt_complex_path,
        )
        if sample_reference_rows:
            residue_rows = sample_reference_rows
            chain_id_source = "sample_residue_map_fallback"
            if numbering_system == "rt_relative" and sample_rt_offset is not None:
                rt_offset = int(sample_rt_offset)
        else:
            residue_rows = _effective_residue_rows(residue_rows, preferred_chain_id, chain_id)
    else:
        residue_rows = _effective_residue_rows(residue_rows, preferred_chain_id, chain_id)
    anchor_positions = anchor_positions_from_file(anchor_path, residue_rows, numbering_system, rt_offset)
    baseline_ifp = json.loads(wt_ifp_path.read_text(encoding="utf-8")).get("baseline_ifp", {})
    baseline_ifp_positions = plip_positions_from_ifp(baseline_ifp, residue_rows, numbering_system, rt_offset)
    ligand = choose_chain_ligand(wt_complex_path, chain_id)
    if ligand is None:
        raise RuntimeError(f"{case_id}: unable to resolve WT ligand from {wt_complex_path}")
    ligand_coord_map = residue_atom_coordinates(wt_complex_path, chain_id, {ligand.residue_id})
    ligand_coords = next(iter(ligand_coord_map.values()))
    residue_coord_map = residue_atom_coordinates(
        wt_complex_path,
        chain_id,
        {residue_key_from_row(row) for row in residue_rows},
    )
    sasa_map = compute_relative_sasa_map(wt_complex_path, chain_id)
    secondary_map = parse_secondary_structure_map(wt_complex_path, chain_id)
    reference_holo_contacts: dict[int, dict[str, Any]] = {}
    if (
        root is not None
        and hiv_reference_holo_pdb
        and hiv_reference_holo_chain
        and numbering_system == "rt_relative"
        and chain_id_source in {"wt_complex_single_chain_fallback", "sample_residue_map_fallback"}
    ):
        reference_holo_contacts = _reference_holo_position_contacts(
            _resolve_cached_structure_path(root, str(hiv_reference_holo_pdb)),
            str(hiv_reference_holo_chain),
        )
    rows: list[dict[str, Any]] = []
    for row in residue_rows:
        target_pos = target_position_from_row(row, numbering_system, rt_offset)
        if target_pos is None:
            continue
        key = residue_key_from_row(row)
        coords = residue_coord_map.get(key)
        min_distance = min_distance_between_sets(coords or [], ligand_coords)
        min_distance_source = "wt_complex_chain"
        min_distance_chain_id = chain_id
        reference_contact = reference_holo_contacts.get(int(target_pos))
        if reference_contact is not None and (
            min_distance is None or float(reference_contact["min_ligand_distance_a"]) + 1.0e-6 < float(min_distance)
        ):
            min_distance = float(reference_contact["min_ligand_distance_a"])
            min_distance_source = "reference_holo_multichain_override"
            min_distance_chain_id = str(reference_contact["chain_id"])
        layer = "other"
        if min_distance is not None and float(min_distance) <= float(pocket_cutoff_a):
            layer = "pocket"
        elif min_distance is not None and float(min_distance) <= float(second_shell_cutoff_a):
            layer = "second_shell"
        if int(target_pos) in anchor_positions:
            layer = "anchor"
        rows.append(
            {
                "case_id": case_id,
                "target_position": int(target_pos),
                "chain_id": str(row["chain_id"]),
                "pdb_resnum": int(row["pdb_resnum"]),
                "insertion_code": str(row.get("insertion_code") or "").strip(),
                "pdb_aa": str(row.get("pdb_aa") or ""),
                "uniprot_aa": str(row.get("uniprot_aa") or ""),
                "uniprot_position": None if row.get("uniprot_pos") is None else int(row["uniprot_pos"]),
                "structure_layer": layer,
                "min_ligand_distance_a": None if min_distance is None else float(min_distance),
                "min_ligand_distance_source": min_distance_source,
                "min_ligand_distance_chain_id": min_distance_chain_id,
                "relative_sasa": sasa_map.get(key),
                "solvent_accessibility": solvent_class(sasa_map.get(key)),
                "secondary_structure_context": secondary_map.get((int(row["pdb_resnum"]), str(row.get("insertion_code") or "").strip()), "loop"),
                "anchor_flag": bool(int(target_pos) in anchor_positions),
                "baseline_ifp_flag": bool(int(target_pos) in baseline_ifp_positions),
                "sample_proxy_eligible": bool(min_distance_source == "wt_complex_chain"),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    return {
        "case_id": case_id,
        "chain_id": chain_id,
        "chain_id_source": chain_id_source,
        "numbering_system": numbering_system,
        "rt_offset": None if rt_offset is None else int(rt_offset),
        "residue_rows": residue_rows,
        "anchor_positions": sorted(anchor_positions),
        "baseline_ifp_positions": sorted(baseline_ifp_positions),
        "context_frame": frame,
        "ligand_residue_name": ligand.residue_name,
    }


def _alphafold_prediction_payload(
    uniprot_id: str,
    cache_root: Path,
    timeout_sec: int,
) -> list[dict[str, Any]]:
    prediction_path = cache_root / f"{uniprot_id}.prediction.json"
    if prediction_path.exists():
        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "ResistAgent/Stage4"})
    response = session.get(ALPHAFOLD_PREDICTION_API_URL.format(uniprot_id=uniprot_id), timeout=timeout_sec)
    response.raise_for_status()
    payload = response.json()
    json_dump(prediction_path, payload)
    return payload if isinstance(payload, list) else []


def _choose_alphafold_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return max(
        records,
        key=lambda row: (
            int(row.get("uniprotEnd") or 0) - int(row.get("uniprotStart") or 0),
            int(row.get("sequenceEnd") or 0) - int(row.get("sequenceStart") or 0),
            str(row.get("modelEntityId") or ""),
        ),
    )


def _download_msa_a3m(msa_url: str, cache_path: Path, timeout_sec: int) -> Path:
    if cache_path.exists():
        return cache_path
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "ResistAgent/Stage4"})
    response = session.get(msa_url, timeout=timeout_sec)
    response.raise_for_status()
    ensure_dir(cache_path.parent)
    cache_path.write_bytes(response.content)
    return cache_path


def _a3m_sequences(path: Path) -> list[str]:
    sequences: list[str] = []
    parts: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith(">"):
            if parts:
                sequences.append("".join(parts))
                parts = []
            continue
        parts.append(text)
    if parts:
        sequences.append("".join(parts))
    return sequences


def _strip_a3m_insertions(sequence: str) -> str:
    return "".join(char for char in sequence if not char.islower())


def _alignment_position_entropy(a3m_path: Path) -> dict[int, float]:
    sequences = _a3m_sequences(a3m_path)
    if not sequences:
        return {}
    query_alignment = _strip_a3m_insertions(sequences[0])
    aligned_sequences = [_strip_a3m_insertions(sequence) for sequence in sequences]
    lookup: dict[int, float] = {}
    query_position = 0
    for column_index, query_char in enumerate(query_alignment):
        if query_char == "-":
            continue
        query_position += 1
        counts = Counter()
        for sequence in aligned_sequences:
            if column_index >= len(sequence):
                continue
            residue = sequence[column_index].upper()
            if residue in AA_ENTROPY_ALPHABET:
                counts[residue] += 1
        if not counts:
            continue
        total = float(sum(counts.values()))
        entropy = 0.0
        for count in counts.values():
            probability = float(count) / total
            entropy -= probability * math.log2(probability)
        normalized = entropy / math.log2(float(len(AA_ENTROPY_ALPHABET)))
        lookup[int(query_position)] = float(max(0.0, min(1.0, normalized)))
    return lookup


def load_alphafold_position_entropy(
    *,
    uniprot_id: str | None,
    cache_root: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    if not uniprot_id:
        return {
            "cache_version": CONSERVATION_CACHE_VERSION,
            "source": "missing_uniprot_id",
            "position_entropy": {},
        }
    entropy_path = cache_root / f"{uniprot_id}.msa_entropy.json"
    if entropy_path.exists():
        payload = json.loads(entropy_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("cache_version") == CONSERVATION_CACHE_VERSION:
            return payload
    try:
        records = _alphafold_prediction_payload(uniprot_id, cache_root, timeout_sec)
        record = _choose_alphafold_record(records)
        if record is None or not record.get("msaUrl"):
            payload = {
                "cache_version": CONSERVATION_CACHE_VERSION,
                "source": "alphafold_msa_unavailable",
                "position_entropy": {},
            }
            json_dump(entropy_path, payload)
            return payload
        msa_url = str(record["msaUrl"])
        msa_path = _download_msa_a3m(msa_url, cache_root / Path(msa_url).name, timeout_sec)
        position_entropy = _alignment_position_entropy(msa_path)
        payload = {
            "cache_version": CONSERVATION_CACHE_VERSION,
            "source": "alphafold_msa",
            "uniprot_id": uniprot_id,
            "model_entity_id": str(record.get("modelEntityId") or ""),
            "msa_path": str(msa_path),
            "position_entropy": {str(position): round(value, 6) for position, value in position_entropy.items()},
        }
        json_dump(entropy_path, payload)
        return payload
    except Exception as exc:  # pragma: no cover - network/cache state varies
        payload = {
            "cache_version": CONSERVATION_CACHE_VERSION,
            "source": "alphafold_msa_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "position_entropy": {},
        }
        json_dump(entropy_path, payload)
        return payload


def _iter_mutation_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item) for item in converted if item is not None]
        if converted is not None:
            return [str(converted)]
    return [str(value)]


def empirical_position_entropy_lookup(case_master: pd.DataFrame) -> dict[int, float]:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    mutation_series = None
    if "component_mutation_keys" in case_master.columns:
        mutation_series = case_master["component_mutation_keys"]
    elif "mutation_key" in case_master.columns:
        mutation_series = case_master["mutation_key"]
    if mutation_series is None:
        return {}
    for value in mutation_series.tolist():
        for token in _iter_mutation_tokens(value):
            for component in parse_mutation_components(str(token)):
                if component.position is None or not component.alt_aa:
                    continue
                counts[int(component.position)][str(component.alt_aa)] += 1
    lookup: dict[int, float] = {}
    for position, counter in counts.items():
        total = float(sum(counter.values()))
        if total <= 0.0:
            continue
        entropy = 0.0
        for count in counter.values():
            probability = float(count) / total
            entropy -= probability * math.log2(probability)
        normalized = entropy / math.log2(float(len(AA_ENTROPY_ALPHABET)))
        lookup[int(position)] = float(max(0.0, min(1.0, normalized)))
    return lookup


def conservation_values_for_positions(
    positions: list[int],
    *,
    numbering_system: str,
    rt_offset: int | None,
    alphafold_payload: dict[str, Any],
    empirical_lookup: dict[int, float],
) -> tuple[list[float | None], dict[str, int]]:
    alphafold_lookup = {
        int(key): float(value)
        for key, value in (alphafold_payload.get("position_entropy") or {}).items()
    }
    values: list[float | None] = []
    source_counts = {"alphafold_msa": 0, "empirical_mutation_entropy": 0, "missing": 0}
    for position in positions:
        lookup_position = int(position)
        if numbering_system == "rt_relative" and rt_offset is not None:
            lookup_position = int(position + int(rt_offset))
        if lookup_position in alphafold_lookup:
            values.append(float(alphafold_lookup[lookup_position]))
            source_counts["alphafold_msa"] += 1
            continue
        if position in empirical_lookup:
            values.append(float(empirical_lookup[position]))
            source_counts["empirical_mutation_entropy"] += 1
            continue
        values.append(None)
        source_counts["missing"] += 1
    return values, source_counts


def summarize_conservation_source(source_counts: dict[str, int]) -> str:
    if source_counts.get("alphafold_msa", 0) > 0:
        return "alphafold_msa"
    if source_counts.get("empirical_mutation_entropy", 0) > 0:
        return "empirical_mutation_entropy"
    return "missing"


def jaccard_loss(lhs: set[int], rhs: set[int]) -> float | None:
    if not lhs and not rhs:
        return None
    union = lhs | rhs
    if not union:
        return None
    return float(1.0 - (len(lhs & rhs) / float(len(union))))


def anchor_loss(anchor_positions: set[int], mt_positions: set[int]) -> float | None:
    if not anchor_positions:
        return None
    return float(1.0 - (len(anchor_positions & mt_positions) / float(len(anchor_positions))))


def build_ligand_box_from_sample(
    ligand_sdf_path: Path,
    default_box_size_a: float,
    ligand_padding_a: float,
) -> dict[str, Any]:
    ligand = load_rdkit_molecule(ligand_sdf_path, sanitize=True)
    return build_box_from_ligand_coords(
        mol_coordinates(ligand),
        default_box_size_a=default_box_size_a,
        ligand_padding_a=ligand_padding_a,
        source="sample_ligand_centroid",
    )


def _ca_lookup(structure_path: Path, chain_id: str) -> dict[tuple[int, str], Any]:
    structure = load_structure(structure_path)
    model = next(structure.get_models())
    chain = model[str(chain_id)]
    lookup: dict[tuple[int, str], Any] = {}
    for residue in chain:
        hetflag, resseq, icode = residue.id
        if str(hetflag).strip():
            continue
        if "CA" in residue:
            lookup[(int(resseq), str(icode).strip())] = residue["CA"]
    return lookup


def local_backbone_rmsd(
    wt_pdb: Path,
    mt_pdb: Path,
    sample_rows: list[dict[str, Any]],
    numbering_system: str,
    rt_offset: int | None,
    mutated_positions: list[int],
    radius_a: float,
) -> float | None:
    by_position, _, _ = position_lookup_from_rows(sample_rows, numbering_system, rt_offset)
    if not sample_rows:
        return None
    chain_id = str(sample_rows[0]["chain_id"])
    wt_structure = load_structure(wt_pdb)
    mt_structure = load_structure(mt_pdb)
    wt_chain = next(wt_structure.get_models())[chain_id]
    mt_chain = next(mt_structure.get_models())[chain_id]
    wt_ca: dict[tuple[int, str], Any] = {}
    mt_ca: dict[tuple[int, str], Any] = {}
    for residue in wt_chain:
        hetflag, resseq, icode = residue.id
        if str(hetflag).strip():
            continue
        if "CA" in residue:
            wt_ca[(int(resseq), str(icode).strip())] = residue["CA"]
    for residue in mt_chain:
        hetflag, resseq, icode = residue.id
        if str(hetflag).strip():
            continue
        if "CA" in residue:
            mt_ca[(int(resseq), str(icode).strip())] = residue["CA"]
    common_positions: list[int] = []
    for position, row in by_position.items():
        key = (int(row["pdb_resnum"]), str(row.get("insertion_code") or "").strip())
        if key in wt_ca and key in mt_ca:
            common_positions.append(int(position))
    if len(common_positions) < 3:
        return None
    common_positions = sorted(set(common_positions))
    wt_atoms = []
    mt_atoms = []
    for position in common_positions:
        row = by_position[position]
        key = (int(row["pdb_resnum"]), str(row.get("insertion_code") or "").strip())
        wt_atoms.append(wt_ca[key])
        mt_atoms.append(mt_ca[key])
    superimposer = Superimposer()
    superimposer.set_atoms(wt_atoms, mt_atoms)
    superimposer.apply(list(mt_structure.get_atoms()))
    mt_ca_transformed: dict[tuple[int, str], Any] = {}
    for residue in mt_chain:
        hetflag, resseq, icode = residue.id
        if str(hetflag).strip():
            continue
        if "CA" in residue:
            mt_ca_transformed[(int(resseq), str(icode).strip())] = residue["CA"]
    neighborhood_positions: set[int] = set()
    for mutated_position in mutated_positions:
        row = by_position.get(int(mutated_position))
        if row is None:
            continue
        key = (int(row["pdb_resnum"]), str(row.get("insertion_code") or "").strip())
        anchor_atom = wt_ca.get(key)
        if anchor_atom is None:
            continue
        anchor_coord = anchor_atom.get_coord()
        for position in common_positions:
            candidate_row = by_position[position]
            candidate_key = (int(candidate_row["pdb_resnum"]), str(candidate_row.get("insertion_code") or "").strip())
            atom = wt_ca[candidate_key]
            delta = atom.get_coord() - anchor_coord
            distance = float((delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2) ** 0.5)
            if distance <= float(radius_a):
                neighborhood_positions.add(position)
    if len(neighborhood_positions) < 3:
        neighborhood_positions = set(mutated_positions) & set(common_positions)
    if len(neighborhood_positions) < 1:
        return None
    squared = 0.0
    count = 0
    for position in sorted(neighborhood_positions):
        row = by_position.get(position)
        if row is None:
            continue
        key = (int(row["pdb_resnum"]), str(row.get("insertion_code") or "").strip())
        lhs = wt_ca.get(key)
        rhs = mt_ca_transformed.get(key)
        if lhs is None or rhs is None:
            continue
        delta = lhs.get_coord() - rhs.get_coord()
        squared += float(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)
        count += 1
    if count <= 0:
        return None
    return float(math.sqrt(squared / float(count)))


def estimate_ddg_surrogate(
    components: list[MutationComponent],
    relative_sasa_values: list[float | None],
    local_rmsd_a: float | None,
    position_entropy_values: list[float | None] | None = None,
) -> float:
    severity_values = [
        physchem_severity(component.ref_aa, component.alt_aa, component.mutation_class)
        for component in components
    ]
    severity = float(sum(severity_values) / max(1, len(severity_values)))
    observed_sasa = [float(value) for value in relative_sasa_values if value is not None]
    mean_relative_sasa = float(sum(observed_sasa) / len(observed_sasa)) if observed_sasa else 0.35
    burial = max(0.0, 1.0 - min(1.0, mean_relative_sasa))
    local_rmsd = 0.0 if local_rmsd_a is None else min(float(local_rmsd_a), 3.0)
    combo_penalty = max(0, len(components) - 1) * 0.6
    entropy_values = [float(value) for value in (position_entropy_values or []) if value is not None]
    mean_entropy = float(sum(entropy_values) / len(entropy_values)) if entropy_values else None
    conservation_penalty = 0.0 if mean_entropy is None else 1.2 * max(0.0, 1.0 - min(1.0, mean_entropy))
    estimate = 0.8 + 2.2 * burial + 1.7 * severity + 1.8 * local_rmsd + combo_penalty + conservation_penalty
    return float(max(0.0, min(8.0, estimate)))


def fitness_weight_from_ddg(ddg_fold: float) -> float:
    clipped = max(-2.0, min(8.0, float(ddg_fold)))
    return float(sigmoid(-(clipped - 3.0) / 1.2))


def heuristic_impact_from_context(
    components: list[MutationComponent],
    structure_layers: list[str],
    anchor_flags: list[bool],
    p_drug_selected: float,
) -> tuple[float, float]:
    severity_values = [
        physchem_severity(component.ref_aa, component.alt_aa, component.mutation_class)
        for component in components
    ]
    severity = float(sum(severity_values) / max(1, len(severity_values)))
    layer_weights = {"anchor": 1.0, "pocket": 0.8, "second_shell": 0.45, "other": 0.15}
    layer_weight = max(layer_weights.get(layer, 0.15) for layer in structure_layers) if structure_layers else 0.15
    anchor_bonus = 0.25 if any(anchor_flags) else 0.0
    prior_bonus = min(0.35, float(max(0.0, p_drug_selected)) * 0.5)
    delta_dock = float(0.2 + 1.6 * severity * layer_weight + anchor_bonus + prior_bonus)
    delta_ifp = float(min(1.0, 0.15 + 0.7 * severity * layer_weight + anchor_bonus))
    return delta_dock, delta_ifp


def _proxy_cache_fingerprint(job: dict[str, Any]) -> str:
    input_paths = [
        Path(str(job["sample_root"])) / "model_manifest.json",
        Path(str(job["sample_root"])) / "ligand.sdf",
        Path(str(job["sample_root"])) / "WT.pdb",
        Path(str(job["sample_root"])) / "MT.pdb",
        Path(str(job["sample_root"])) / "WT_complex.pdb",
        Path(str(job["sample_root"])) / "MT_complex.pdb",
    ]
    proxy_box_expansion_step = job.get("proxy_box_expansion_step_a")
    hiv_pose_contact_cutoff = job.get("hiv_pose_contact_cutoff_a")
    payload = {
        "cache_version": PROXY_CACHE_VERSION,
        "input_hashes": {str(path.name): sha256_file(path) if path.exists() else None for path in input_paths},
        "sample_residue_rows": list(job.get("sample_residue_rows") or []),
        "target_numbering_system": str(job.get("target_numbering_system") or ""),
        "rt_offset": job.get("rt_offset"),
        "case_anchor_positions": list(job.get("case_anchor_positions") or []),
        "default_box_size_a": float(job["default_box_size_a"]),
        "ligand_box_padding_a": float(job["ligand_box_padding_a"]),
        "use_pdbfixer": bool(job.get("use_pdbfixer", False)),
        "protein_prep_ph": float(job["protein_prep_ph"]),
        "quick_docking_seeds": list(job["quick_docking_seeds"]),
        "vina_exhaustiveness": int(job["vina_exhaustiveness"]),
        "vina_num_modes": int(job["vina_num_modes"]),
        "vina_energy_range": int(job["vina_energy_range"]),
        "vina_cpu_threads": int(job["vina_cpu_threads"]),
        "proxy_retry_limit": int(job.get("proxy_retry_limit", 0)),
        "proxy_box_expansion_step_a": 0.0 if proxy_box_expansion_step is None else float(proxy_box_expansion_step),
        "hiv_mode": bool(job.get("hiv_mode", False)),
        "hiv_required_pose_label": str(job.get("hiv_required_pose_label") or ""),
        "hiv_pose_contact_cutoff_a": 0.0 if hiv_pose_contact_cutoff is None else float(hiv_pose_contact_cutoff),
        "nnrti_residue_coords": job.get("nnrti_residue_coords") or {},
        "active_site_residue_coords": job.get("active_site_residue_coords") or {},
        "proxy_evidence_tier_success": str(job.get("proxy_evidence_tier_success") or "paired_observed_structure"),
        "allow_complex_ifp_only_fallback": bool(job.get("allow_complex_ifp_only_fallback", False)),
        "complex_ifp_only_evidence_tier": str(job.get("complex_ifp_only_evidence_tier") or ""),
    }
    return _sha256_json_payload(payload)


def _expand_proxy_box(docking_box: dict[str, Any], delta_a: float, attempt_index: int) -> dict[str, Any]:
    expanded = dict(docking_box)
    if delta_a > 0.0:
        expanded["size_x"] = float(expanded["size_x"]) + delta_a
        expanded["size_y"] = float(expanded["size_y"]) + delta_a
        expanded["size_z"] = float(expanded["size_z"]) + delta_a
    expanded["source"] = f"{docking_box['source']}_retry_{attempt_index}"
    expanded["retry_expansion_delta_a"] = float(delta_a)
    expanded["retry_attempt_index"] = int(attempt_index)
    return expanded


def _resolve_cached_structure_path(root: Path, pdb_id: str) -> Path:
    pdb_upper = str(pdb_id).upper()
    candidates = [
        root / "outputs" / "case_manifests" / "rcsb_cache" / f"{pdb_upper}.pdb",
        root / "outputs" / "stage1_5" / "rcsb_cache" / f"{pdb_upper}.pdb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate cached structure for {pdb_upper}")


def build_hiv_pose_reference(
    *,
    root: Path,
    case_entry: dict[str, Any],
    pocket_positions: list[int],
    reference_holo_pdb: str,
    reference_holo_chain: str,
    contact_cutoff_a: float,
) -> dict[str, Any]:
    receptor_source = root / str(case_entry["wt_template"]["structure_path"])
    chain_id = str(case_entry["wt_template"]["chain_id"])
    active_site_ligand = choose_chain_ligand(receptor_source, chain_id, preferred_residue_names=["TTP"])
    if active_site_ligand is None:
        raise RuntimeError("HIV WT template is missing the expected TTP active-site ligand")
    active_site_coord_map = residue_atom_coordinates(receptor_source, chain_id, {active_site_ligand.residue_id})
    active_site_coords = next(iter(active_site_coord_map.values()))
    active_site_ids_wt = nearby_protein_residue_ids(
        receptor_source,
        chain_id,
        active_site_coords,
        distance_cutoff_a=float(contact_cutoff_a),
    )
    holo_template_path = _resolve_cached_structure_path(root, reference_holo_pdb)
    holo_chain = str(reference_holo_chain)
    nnrti_residue_ids = {(" ", int(position), "") for position in pocket_positions}
    nnrti_residue_coords = {
        f"{holo_chain}:{residue_id[1]}{residue_id[2]}": coords
        for residue_id, coords in residue_atom_coordinates(
            holo_template_path,
            holo_chain,
            nnrti_residue_ids,
        ).items()
    }
    if len(nnrti_residue_coords) < len(pocket_positions):
        raise RuntimeError("Unable to resolve the full HIV NNRTI pocket residue set for mutation-proposal step pose gating")
    active_site_positions = sorted({int(residue_id[1]) for residue_id in active_site_ids_wt if str(residue_id[0]).strip() == ""})
    active_site_ids_holo = {(" ", int(position), "") for position in active_site_positions}
    active_site_residue_coords = {
        f"{holo_chain}:{residue_id[1]}{residue_id[2]}": coords
        for residue_id, coords in residue_atom_coordinates(
            holo_template_path,
            holo_chain,
            active_site_ids_holo,
        ).items()
    }
    return {
        "reference_holo_pdb": str(reference_holo_pdb).upper(),
        "reference_holo_chain": holo_chain,
        "contact_cutoff_a": float(contact_cutoff_a),
        "nnrti_residue_coords": nnrti_residue_coords,
        "active_site_residue_coords": active_site_residue_coords,
    }


def _hetero_residue_ids_for_chain(source_pdb: Path, chain_id: str) -> set[tuple[str, int, str]]:
    structure = load_structure(source_pdb)
    model = next(structure.get_models())
    chain = model[str(chain_id)]
    residue_ids: set[tuple[str, int, str]] = set()
    for residue in chain:
        hetflag, resseq, icode = residue.id
        if not str(hetflag).strip():
            continue
        residue_ids.add((str(hetflag), int(resseq), str(icode).strip()))
    return residue_ids


def _mutation_specs_for_components(
    components: list[MutationComponent],
    residue_rows: list[dict[str, Any]],
    numbering_system: str,
    rt_offset: int | None,
) -> list[str]:
    by_position, _, _ = position_lookup_from_rows(residue_rows, numbering_system, rt_offset)
    specs: list[str] = []
    for component in components:
        if component.position is None or component.mutation_class != "single_substitution" or not component.alt_aa:
            continue
        row = by_position.get(int(component.position))
        if row is None:
            raise RuntimeError(f"Unable to map mutation position {component.position} onto WT template residues")
        ref_aa = str(row.get("uniprot_aa") or row.get("pdb_aa") or component.ref_aa or "").upper()
        alt_aa = str(component.alt_aa).upper()
        ref_aa3 = AA1_TO_3.get(ref_aa)
        alt_aa3 = AA1_TO_3.get(alt_aa)
        if ref_aa3 is None or alt_aa3 is None:
            raise RuntimeError(f"Unsupported mutation code for combo modeling: {ref_aa}->{alt_aa}")
        specs.append(f"{ref_aa3}-{int(row['pdb_resnum'])}-{alt_aa3}")
    if not specs:
        raise RuntimeError("No single-substitution mutation specs available for combo modeling")
    return specs


def materialize_synthetic_combo_sample(
    *,
    combo_key: str,
    components: list[MutationComponent],
    sample_root: Path,
    wt_complex_path: Path,
    ligand_input_path: Path,
    chain_id: str,
    residue_rows: list[dict[str, Any]],
    numbering_system: str,
    rt_offset: int | None,
) -> None:
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    ensure_dir(sample_root)
    manifest = {
        "combo_model_version": COMBO_MODEL_VERSION,
        "combination_key": str(combo_key),
        "mutation_specs": _mutation_specs_for_components(components, residue_rows, numbering_system, rt_offset),
        "template_complex": str(wt_complex_path),
    }
    json_dump(sample_root / "model_manifest.json", manifest)
    wt_pdb = sample_root / "WT.pdb"
    mt_pdb = sample_root / "MT.pdb"
    wt_complex = sample_root / "WT_complex.pdb"
    mt_complex = sample_root / "MT_complex.pdb"
    ligand_output = sample_root / "ligand.sdf"

    save_chain_protein(wt_complex_path, chain_id, wt_pdb)
    shutil.copyfile(wt_complex_path, wt_complex)
    ligand = choose_chain_ligand(wt_complex_path, chain_id)
    if ligand is not None:
        try:
            crystal_ligand_from_template(
                wt_complex_path,
                chain_id,
                ligand,
                ligand_input_path,
                ligand_output,
                sample_root,
            )
        except Exception:
            extract_ligand_to_sdf(
                wt_complex_path,
                chain_id,
                ligand,
                ligand_output,
                sample_root,
            )
    else:
        shutil.copyfile(ligand_input_path, ligand_output)

    fixer = PDBFixer(filename=str(wt_pdb))
    fixer.applyMutations(list(manifest["mutation_specs"]), str(chain_id))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    with mt_pdb.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)

    hetero_ids = _hetero_residue_ids_for_chain(wt_complex_path, chain_id)
    hetero_pdb = sample_root / "hetero_fragment.pdb"
    save_residue_subset(wt_complex_path, chain_id, hetero_ids, hetero_pdb)
    merge_pdb_fragments([mt_pdb, hetero_pdb], mt_complex)


def _classify_complex_pose(
    complex_pdb: Path,
    chain_id: str,
    work_root: Path,
    label_prefix: str,
    reference_ligand_sdf: Path | None,
    nnrti_residue_coords: dict[str, list[tuple[float, float, float]]],
    active_site_residue_coords: dict[str, list[tuple[float, float, float]]],
    contact_cutoff_a: float,
) -> dict[str, Any]:
    ligand = choose_chain_ligand(complex_pdb, chain_id)
    if ligand is None:
        raise RuntimeError(f"No ligand found in complex for pose classification: {complex_pdb}")
    ligand_sdf = work_root / f"{label_prefix}_complex_pose.sdf"
    if reference_ligand_sdf is not None and reference_ligand_sdf.exists():
        try:
            crystal_ligand_from_template(
                complex_pdb,
                chain_id,
                ligand,
                reference_ligand_sdf,
                ligand_sdf,
                work_root,
            )
        except Exception:
            extract_ligand_to_sdf(complex_pdb, chain_id, ligand, ligand_sdf, work_root)
    else:
        extract_ligand_to_sdf(complex_pdb, chain_id, ligand, ligand_sdf, work_root)
    return classify_hiv_pose(
        ligand_sdf,
        nnrti_residue_coords,
        active_site_residue_coords,
        float(contact_cutoff_a),
    )


def _complex_only_proxy_payload(
    *,
    sample_id: str,
    wt_complex: Path,
    mt_complex: Path,
    reference_ligand_sdf: Path | None,
    sample_residue_rows: list[dict[str, Any]],
    target_numbering_system: str,
    rt_offset: int | None,
    case_anchor_positions: list[int],
    cache_path: Path,
    cache_fingerprint: str,
    attempt_history: list[dict[str, Any]],
    proxy_retry_count: int,
    proxy_evidence_tier: str,
    hiv_mode: bool,
    required_pose_label: str,
    nnrti_residue_coords: dict[str, list[tuple[float, float, float]]],
    active_site_residue_coords: dict[str, list[tuple[float, float, float]]],
    hiv_pose_contact_cutoff_a: float | None,
    work_root: Path,
) -> dict[str, Any]:
    chain_id = str(sample_residue_rows[0]["chain_id"]) if sample_residue_rows else ""
    if hiv_mode:
        wt_pose = _classify_complex_pose(
            wt_complex,
            chain_id,
            work_root,
            "wt",
            reference_ligand_sdf,
            nnrti_residue_coords,
            active_site_residue_coords,
            0.0 if hiv_pose_contact_cutoff_a is None else float(hiv_pose_contact_cutoff_a),
        )
        mt_pose = _classify_complex_pose(
            mt_complex,
            chain_id,
            work_root,
            "mt",
            reference_ligand_sdf,
            nnrti_residue_coords,
            active_site_residue_coords,
            0.0 if hiv_pose_contact_cutoff_a is None else float(hiv_pose_contact_cutoff_a),
        )
        if str(wt_pose.get("pose_label")) != required_pose_label or str(mt_pose.get("pose_label")) != required_pose_label:
            raise RuntimeError(
                f"Observed/model complex pose outside required HIV pocket: wt={wt_pose.get('pose_label')} mt={mt_pose.get('pose_label')}"
            )
    else:
        wt_pose = {}
        mt_pose = {}

    wt_ifp = plip_ifp(wt_complex)
    mt_ifp = plip_ifp(mt_complex)
    wt_ifp_positions = plip_positions_from_ifp(
        wt_ifp,
        sample_residue_rows,
        target_numbering_system,
        rt_offset,
    )
    mt_ifp_positions = plip_positions_from_ifp(
        mt_ifp,
        sample_residue_rows,
        target_numbering_system,
        rt_offset,
    )
    payload = {
        "sample_id": sample_id,
        "proxy_status": "ok",
        "proxy_evidence_tier": proxy_evidence_tier,
        "docking_box_source": "complex_ifp_only",
        "wt_best_affinity_kcal_mol": None,
        "mt_best_affinity_kcal_mol": None,
        "delta_dock_proxy": 0.0,
        "wt_pose_count": 0,
        "mt_pose_count": 0,
        "wt_ifp_positions": sorted(wt_ifp_positions),
        "mt_ifp_positions": sorted(mt_ifp_positions),
        "ifp_jaccard_loss": jaccard_loss(wt_ifp_positions, mt_ifp_positions),
        "anchor_loss_fraction": anchor_loss(set(int(value) for value in case_anchor_positions), mt_ifp_positions),
        "proxy_retry_count": int(proxy_retry_count),
        "attempt_history": attempt_history,
        "cache_version": PROXY_CACHE_VERSION,
        "cache_fingerprint": cache_fingerprint,
    }
    if hiv_mode:
        payload["wt_pose_label_counts"] = {str(wt_pose.get("pose_label") or "other"): 1}
        payload["mt_pose_label_counts"] = {str(mt_pose.get("pose_label") or "other"): 1}
    json_dump(cache_path, payload)
    return payload


def sample_proxy_worker(job: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(job["sample_id"])
    sample_root = Path(job["sample_root"])
    cache_path = Path(job["cache_path"])
    cache_fingerprint = _proxy_cache_fingerprint(job)
    if cache_path.exists():
        cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            isinstance(cached_payload, dict)
            and cached_payload.get("cache_version") == PROXY_CACHE_VERSION
            and cached_payload.get("cache_fingerprint") == cache_fingerprint
        ):
            return cached_payload

    work_root = ensure_dir(Path(job["work_root"]))
    ligand_input = sample_root / "ligand.sdf"
    wt_pdb = sample_root / "WT.pdb"
    mt_pdb = sample_root / "MT.pdb"
    wt_complex = sample_root / "WT_complex.pdb"
    mt_complex = sample_root / "MT_complex.pdb"
    if not (ligand_input.exists() and wt_pdb.exists() and mt_pdb.exists() and wt_complex.exists() and mt_complex.exists()):
        payload = {
            "sample_id": sample_id,
            "proxy_status": "missing_inputs",
            "proxy_evidence_tier": "structure_unavailable_fallback",
            "cache_version": PROXY_CACHE_VERSION,
            "cache_fingerprint": cache_fingerprint,
            "proxy_retry_count": 0,
            "attempt_history": [],
        }
        json_dump(cache_path, payload)
        return payload

    max_attempts = max(1, int(job.get("proxy_retry_limit", 0)) + 1)
    expansion_step_value = job.get("proxy_box_expansion_step_a")
    expansion_step_a = 0.0 if expansion_step_value is None else float(expansion_step_value)
    hiv_mode = bool(job.get("hiv_mode", False))
    required_pose_label = str(job.get("hiv_required_pose_label") or "NNRTI_pocket")
    nnrti_residue_coords = job.get("nnrti_residue_coords") or {}
    active_site_residue_coords = job.get("active_site_residue_coords") or {}
    attempt_history: list[dict[str, Any]] = []
    last_error = "unknown_error"
    ligand_standardized = work_root / "ligand_standardized.sdf"
    try:
        standardize_reference_ligand(ligand_input, ligand_standardized)
    except Exception as exc:
        payload = {
            "sample_id": sample_id,
            "proxy_status": "failed",
            "proxy_evidence_tier": "structure_unavailable_fallback",
            "proxy_error": f"{type(exc).__name__}: {exc}",
            "proxy_retry_count": 0,
            "attempt_history": attempt_history,
            "cache_version": PROXY_CACHE_VERSION,
            "cache_fingerprint": cache_fingerprint,
        }
        json_dump(cache_path, payload)
        return payload

    for attempt_index in range(1, max_attempts + 1):
        attempt_root = ensure_dir(work_root / f"attempt_{attempt_index}")
        try:
            docking_box = build_ligand_box_from_sample(
                ligand_standardized,
                default_box_size_a=float(job["default_box_size_a"]),
                ligand_padding_a=float(job["ligand_box_padding_a"]),
            )
            docking_box = _expand_proxy_box(docking_box, expansion_step_a * float(attempt_index - 1), attempt_index)

            wt_receptor_input = wt_pdb
            mt_receptor_input = mt_pdb
            if bool(job.get("use_pdbfixer", False)):
                wt_fixed = attempt_root / "WT_fixed.pdb"
                mt_fixed = attempt_root / "MT_fixed.pdb"
                run_pdbfixer(wt_pdb, wt_fixed, float(job["protein_prep_ph"]))
                run_pdbfixer(mt_pdb, mt_fixed, float(job["protein_prep_ph"]))
                wt_receptor_input = wt_fixed
                mt_receptor_input = mt_fixed

            wt_receptor_pdbqt = attempt_root / "WT.pdbqt"
            mt_receptor_pdbqt = attempt_root / "MT.pdbqt"
            ligand_pdbqt = attempt_root / "ligand.pdbqt"
            prepare_receptor_pdbqt(wt_receptor_input, wt_receptor_pdbqt, float(job["protein_prep_ph"]))
            prepare_receptor_pdbqt(mt_receptor_input, mt_receptor_pdbqt, float(job["protein_prep_ph"]))
            prepare_ligand_pdbqt(ligand_standardized, ligand_pdbqt, float(job["protein_prep_ph"]))

            wt_rows = run_vina_redocking(
                receptor_pdbqt=wt_receptor_pdbqt,
                ligand_pdbqt=ligand_pdbqt,
                docking_box=docking_box,
                output_root=attempt_root / "wt_docking",
                seeds=list(job["quick_docking_seeds"]),
                exhaustiveness=int(job["vina_exhaustiveness"]),
                num_modes=int(job["vina_num_modes"]),
                energy_range=int(job["vina_energy_range"]),
                cpu_threads=int(job["vina_cpu_threads"]),
            )
            mt_rows = run_vina_redocking(
                receptor_pdbqt=mt_receptor_pdbqt,
                ligand_pdbqt=ligand_pdbqt,
                docking_box=docking_box,
                output_root=attempt_root / "mt_docking",
                seeds=list(job["quick_docking_seeds"]),
                exhaustiveness=int(job["vina_exhaustiveness"]),
                num_modes=int(job["vina_num_modes"]),
                energy_range=int(job["vina_energy_range"]),
                cpu_threads=int(job["vina_cpu_threads"]),
            )
            if hiv_mode:
                for row in wt_rows:
                    row.update(
                        classify_hiv_pose(
                            Path(str(row["pose_sdf"])),
                            nnrti_residue_coords,
                            active_site_residue_coords,
                            float(job["hiv_pose_contact_cutoff_a"]),
                        )
                    )
                for row in mt_rows:
                    row.update(
                        classify_hiv_pose(
                            Path(str(row["pose_sdf"])),
                            nnrti_residue_coords,
                            active_site_residue_coords,
                            float(job["hiv_pose_contact_cutoff_a"]),
                        )
                    )
            best_wt = select_best_pose(wt_rows, hiv_mode=hiv_mode)
            best_mt = select_best_pose(mt_rows, hiv_mode=hiv_mode)
            if best_wt is None or best_mt is None:
                raise RuntimeError("No valid WT/MT pose selected after pocket gating")
            if hiv_mode:
                global_best_wt = select_best_pose(wt_rows, hiv_mode=False)
                global_best_mt = select_best_pose(mt_rows, hiv_mode=False)
                if global_best_wt is not None and str(global_best_wt.get("pose_label")) != required_pose_label:
                    raise RuntimeError(f"WT global-best pose left required HIV pocket: {global_best_wt.get('pose_label')}")
                if global_best_mt is not None and str(global_best_mt.get("pose_label")) != required_pose_label:
                    raise RuntimeError(f"MT global-best pose left required HIV pocket: {global_best_mt.get('pose_label')}")

            wt_ifp = plip_ifp(wt_complex)
            mt_ifp = plip_ifp(mt_complex)
            wt_ifp_positions = plip_positions_from_ifp(
                wt_ifp,
                list(job["sample_residue_rows"]),
                str(job["target_numbering_system"]),
                job.get("rt_offset"),
            )
            mt_ifp_positions = plip_positions_from_ifp(
                mt_ifp,
                list(job["sample_residue_rows"]),
                str(job["target_numbering_system"]),
                job.get("rt_offset"),
            )
            payload = {
                "sample_id": sample_id,
                "proxy_status": "ok",
                "proxy_evidence_tier": str(job.get("proxy_evidence_tier_success") or "paired_observed_structure"),
                "docking_box_source": str(docking_box["source"]),
                "wt_best_affinity_kcal_mol": float(best_wt["affinity_kcal_mol"]),
                "mt_best_affinity_kcal_mol": float(best_mt["affinity_kcal_mol"]),
                "delta_dock_proxy": float(best_mt["affinity_kcal_mol"] - best_wt["affinity_kcal_mol"]),
                "wt_pose_count": int(len(wt_rows)),
                "mt_pose_count": int(len(mt_rows)),
                "wt_ifp_positions": sorted(wt_ifp_positions),
                "mt_ifp_positions": sorted(mt_ifp_positions),
                "ifp_jaccard_loss": jaccard_loss(wt_ifp_positions, mt_ifp_positions),
                "anchor_loss_fraction": anchor_loss(set(int(value) for value in job["case_anchor_positions"]), mt_ifp_positions),
                "proxy_retry_count": int(attempt_index - 1),
                "attempt_history": attempt_history,
                "cache_version": PROXY_CACHE_VERSION,
                "cache_fingerprint": cache_fingerprint,
            }
            if hiv_mode:
                payload["wt_pose_label_counts"] = dict(Counter(str(row.get("pose_label") or "other") for row in wt_rows))
                payload["mt_pose_label_counts"] = dict(Counter(str(row.get("pose_label") or "other") for row in mt_rows))
            json_dump(cache_path, payload)
            return payload
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            attempt_history.append(
                {
                    "attempt_index": int(attempt_index),
                    "docking_box_source": f"retry_{attempt_index}",
                    "retry_expansion_delta_a": float(expansion_step_a * float(attempt_index - 1)),
                    "error": last_error,
                }
            )

    if bool(job.get("allow_complex_ifp_only_fallback", False)):
        try:
            return _complex_only_proxy_payload(
                sample_id=sample_id,
                wt_complex=wt_complex,
                mt_complex=mt_complex,
                reference_ligand_sdf=ligand_standardized,
                sample_residue_rows=list(job["sample_residue_rows"]),
                target_numbering_system=str(job["target_numbering_system"]),
                rt_offset=job.get("rt_offset"),
                case_anchor_positions=list(job["case_anchor_positions"]),
                cache_path=cache_path,
                cache_fingerprint=cache_fingerprint,
                attempt_history=attempt_history,
                proxy_retry_count=int(max_attempts - 1),
                proxy_evidence_tier=str(job.get("complex_ifp_only_evidence_tier") or "complex_ifp_only_structure"),
                hiv_mode=hiv_mode,
                required_pose_label=required_pose_label,
                nnrti_residue_coords=nnrti_residue_coords,
                active_site_residue_coords=active_site_residue_coords,
                hiv_pose_contact_cutoff_a=job.get("hiv_pose_contact_cutoff_a"),
                work_root=work_root,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            attempt_history.append(
                {
                    "attempt_index": int(max_attempts + 1),
                    "docking_box_source": "complex_ifp_only_fallback",
                    "retry_expansion_delta_a": None,
                    "error": last_error,
                }
            )

    payload = {
        "sample_id": sample_id,
        "proxy_status": "failed",
        "proxy_evidence_tier": "structure_unavailable_fallback",
        "proxy_error": last_error,
        "proxy_retry_count": int(max_attempts - 1),
        "attempt_history": attempt_history,
        "cache_version": PROXY_CACHE_VERSION,
        "cache_fingerprint": cache_fingerprint,
    }
    json_dump(cache_path, payload)
    return payload


def write_story_markdown(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
