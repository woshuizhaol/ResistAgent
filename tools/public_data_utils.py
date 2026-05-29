#!/usr/bin/env python3
"""Public-data fetch and parsing helpers used by Stage 1.5 and Stage 2."""

from __future__ import annotations

import json
import re
import shutil
import textwrap
from pathlib import Path
from typing import Any

import requests
import yaml
from Bio import Align
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from rdkit import Chem

from tools.runtime import ensure_dir, json_dump, text_dump

RCSB_MMCIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
ALPHAFOLD_PREDICTION_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
UNIPROT_JSON_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
PUBCHEM_NAME_TO_CID_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/cids/JSON"
PUBCHEM_CID_PROPERTIES_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Title,CanonicalSMILES/JSON"
)
PUBCHEM_CID_SDF_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=2d"

HETERO_SKIP_IDS = {
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


def request_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "ResistAgent/public-data"})
    return session


def load_public_projects(projects_root: Path) -> list[dict[str, Any]]:
    projects = []
    for path in sorted(projects_root.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Expected mapping in {path}")
        data["_config_path"] = str(path)
        projects.append(data)
    return projects


def _download_bytes(session: requests.Session, url: str, timeout: int) -> bytes:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def _download_text(session: requests.Session, url: str, timeout: int) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def cached_json(session: requests.Session, url: str, cache_path: Path, timeout: int) -> dict[str, Any]:
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    payload = json.loads(_download_text(session, url, timeout=timeout))
    json_dump(cache_path, payload)
    return payload


def cached_text(session: requests.Session, url: str, cache_path: Path, timeout: int) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    text = _download_text(session, url, timeout=timeout)
    text_dump(cache_path, text)
    return text


def cached_binary(session: requests.Session, url: str, cache_path: Path, timeout: int) -> Path:
    if cache_path.exists():
        return cache_path
    ensure_dir(cache_path.parent)
    cache_path.write_bytes(_download_bytes(session, url, timeout=timeout))
    return cache_path


def fetch_rcsb_files(session: requests.Session, pdb_id: str, cache_root: Path, timeout: int) -> dict[str, Path]:
    pdb_upper = str(pdb_id).upper()
    cif_path = cached_binary(session, RCSB_MMCIF_URL.format(pdb_id=pdb_upper), cache_root / f"{pdb_upper}.cif", timeout)
    pdb_path = cached_binary(session, RCSB_PDB_URL.format(pdb_id=pdb_upper), cache_root / f"{pdb_upper}.pdb", timeout)
    return {"cif": cif_path, "pdb": pdb_path}


def fetch_alphafold_model(
    session: requests.Session,
    uniprot_id: str | None,
    cache_root: Path,
    timeout: int,
) -> dict[str, Any] | None:
    if not uniprot_id:
        return None
    cache_root = ensure_dir(cache_root)
    try:
        api_payload = cached_json(
            session,
            ALPHAFOLD_PREDICTION_API_URL.format(uniprot_id=uniprot_id),
            cache_root / f"{uniprot_id}.prediction.json",
            timeout,
        )
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None
        raise
    if not isinstance(api_payload, list) or not api_payload:
        return None
    record = api_payload[0]
    pdb_url = record.get("pdbUrl")
    if not pdb_url:
        return None
    model_id = str(record.get("modelEntityId") or f"AF-{uniprot_id}-F1")
    pdb_path = cached_binary(session, str(pdb_url), cache_root / f"{model_id}.pdb", timeout)
    return {
        "source": "alphafold_db",
        "model_id": model_id,
        "record": record,
        "pdb": pdb_path,
    }


def fetch_uniprot_assets(
    session: requests.Session,
    uniprot_id: str | None,
    cache_root: Path,
    timeout: int,
) -> dict[str, Any]:
    if not uniprot_id:
        return {"uniprot_id": None, "sequence": None, "record": None}
    fasta_path = cache_root / f"{uniprot_id}.fasta"
    json_path = cache_root / f"{uniprot_id}.json"
    fasta_text = cached_text(session, UNIPROT_FASTA_URL.format(uniprot_id=uniprot_id), fasta_path, timeout)
    sequence = "".join(line.strip() for line in fasta_text.splitlines() if not line.startswith(">"))
    record = cached_json(session, UNIPROT_JSON_URL.format(uniprot_id=uniprot_id), json_path, timeout)
    return {"uniprot_id": uniprot_id, "sequence": sequence, "record": record}


def _canonicalize_smiles(smiles: str) -> tuple[str, Chem.Mol]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Unable to parse SMILES from PubChem")
    return Chem.MolToSmiles(molecule, canonical=True), molecule


def fetch_pubchem_ligand(
    session: requests.Session,
    ligand_spec: dict[str, Any] | None,
    cache_root: Path,
    timeout: int,
) -> dict[str, Any]:
    ligand_spec = ligand_spec or {}
    query_name = ligand_spec.get("query_name")
    smiles = ligand_spec.get("smiles")
    cid = ligand_spec.get("pubchem_cid")

    if smiles:
        canonical_smiles, molecule = _canonicalize_smiles(str(smiles))
        return {
            "cid": None,
            "query_name": query_name,
            "canonical_smiles": canonical_smiles,
            "molecule": molecule,
            "title": query_name or canonical_smiles,
            "sdf_source": "input_smiles",
        }

    if not cid and query_name:
        cid_payload = cached_json(
            session,
            PUBCHEM_NAME_TO_CID_URL.format(query=requests.utils.quote(str(query_name))),
            cache_root / f"name_{re.sub(r'[^A-Za-z0-9]+', '_', str(query_name)).strip('_')}.cid.json",
            timeout,
        )
        cid_list = cid_payload.get("IdentifierList", {}).get("CID", [])
        if not cid_list:
            raise ValueError(f"Unable to resolve PubChem CID for {query_name}")
        cid = int(cid_list[0])

    if cid is None:
        raise ValueError("Ligand spec must provide one of smiles, pubchem_cid, or query_name")

    properties = cached_json(
        session,
        PUBCHEM_CID_PROPERTIES_URL.format(cid=cid),
        cache_root / f"cid_{cid}.properties.json",
        timeout,
    )
    property_rows = properties.get("PropertyTable", {}).get("Properties", [])
    if not property_rows:
        raise ValueError(f"PubChem property lookup returned no rows for CID {cid}")
    property_row = property_rows[0]
    smiles_value = (
        property_row.get("CanonicalSMILES")
        or property_row.get("ConnectivitySMILES")
        or property_row.get("IsomericSMILES")
    )
    if not smiles_value:
        raise ValueError(f"PubChem property lookup returned no usable SMILES for CID {cid}")
    canonical_smiles, molecule = _canonicalize_smiles(smiles_value)
    sdf_path = cached_binary(session, PUBCHEM_CID_SDF_URL.format(cid=cid), cache_root / f"cid_{cid}.sdf", timeout)
    return {
        "cid": int(cid),
        "query_name": query_name,
        "canonical_smiles": canonical_smiles,
        "molecule": molecule,
        "title": property_row.get("Title") or query_name or f"CID:{cid}",
        "sdf_path": sdf_path,
        "sdf_source": "pubchem",
    }


def write_ligand_sdf(ligand_payload: dict[str, Any], output_path: Path) -> dict[str, Any]:
    ensure_dir(output_path.parent)
    sdf_source = str(ligand_payload.get("sdf_source", ""))
    if sdf_source == "pubchem":
        shutil.copyfile(Path(ligand_payload["sdf_path"]), output_path)
    else:
        molecule = ligand_payload["molecule"]
        mol_block = Chem.MolToMolBlock(molecule)
        text_dump(output_path, mol_block)

    molecule = ligand_payload["molecule"]
    fragment_count = len(Chem.GetMolFrags(molecule))
    return {
        "heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
        "bond_count": int(molecule.GetNumBonds()),
        "fragment_count": int(fragment_count),
        "canonical_smiles": ligand_payload["canonical_smiles"],
        "title": ligand_payload["title"],
        "pubchem_cid": ligand_payload.get("cid"),
    }


def normalize_sequence_text(text: str | None) -> str:
    value = str(text or "")
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if lines and lines[0].startswith(">"):
        lines = [line for line in lines if not line.startswith(">")]
    return re.sub(r"[^A-Za-z]", "", "".join(lines)).upper()


def fasta_text(header: str, sequence: str) -> str:
    wrapped = "\n".join(textwrap.wrap(sequence, width=80))
    return f">{header}\n{wrapped}\n"


def write_sequence_fasta(path: Path, header: str, sequence: str) -> None:
    text_dump(path, fasta_text(header, sequence))


def load_mmcif_dict(path: Path) -> MMCIF2Dict:
    return MMCIF2Dict(str(path))


def _list_value(cif_dict: MMCIF2Dict, key: str) -> list[str]:
    value = cif_dict.get(key, [])
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def mmcif_resolution(cif_dict: MMCIF2Dict) -> float | None:
    for key in ("_refine.ls_d_res_high", "_em_3d_reconstruction.resolution"):
        values = _list_value(cif_dict, key)
        if values and values[0] not in {"?", "."}:
            try:
                return float(values[0])
            except ValueError:
                continue
    return None


def mmcif_entry_title(cif_dict: MMCIF2Dict) -> str:
    for key in ("_struct.title", "_struct_keywords.pdbx_keywords"):
        values = _list_value(cif_dict, key)
        if values and values[0] not in {"?", "."}:
            return values[0].strip()
    return ""


def mmcif_entity_descriptions(cif_dict: MMCIF2Dict) -> dict[str, str]:
    entity_ids = _list_value(cif_dict, "_entity.id")
    descriptions = _list_value(cif_dict, "_entity.pdbx_description")
    return {entity_id: description.strip() for entity_id, description in zip(entity_ids, descriptions)}


def mmcif_chain_entity_map(cif_dict: MMCIF2Dict) -> dict[str, str]:
    label_chains = _list_value(cif_dict, "_struct_asym.id")
    entity_ids = _list_value(cif_dict, "_struct_asym.entity_id")
    label_to_entity = {chain: entity_id for chain, entity_id in zip(label_chains, entity_ids)}

    auth_chains = _list_value(cif_dict, "_atom_site.auth_asym_id")
    atom_label_chains = _list_value(cif_dict, "_atom_site.label_asym_id")
    auth_to_entity: dict[str, str] = {}
    for auth_chain, label_chain in zip(auth_chains, atom_label_chains):
        entity_id = label_to_entity.get(label_chain)
        if auth_chain not in auth_to_entity and entity_id is not None:
            auth_to_entity[auth_chain] = entity_id
    return auth_to_entity


def mmcif_entity_sequences(cif_dict: MMCIF2Dict) -> dict[str, str]:
    entity_ids = _list_value(cif_dict, "_entity_poly.entity_id")
    sequences = _list_value(cif_dict, "_entity_poly.pdbx_seq_one_letter_code_can")
    cleaned: dict[str, str] = {}
    for entity_id, sequence in zip(entity_ids, sequences):
        cleaned[entity_id] = re.sub(r"[^A-Za-z]", "", sequence).upper()
    return cleaned


def mmcif_chain_sequences(cif_dict: MMCIF2Dict) -> dict[str, str]:
    chain_entity = mmcif_chain_entity_map(cif_dict)
    entity_sequences = mmcif_entity_sequences(cif_dict)
    chain_sequences = {}
    for chain, entity_id in chain_entity.items():
        if entity_id in entity_sequences:
            chain_sequences[chain] = entity_sequences[entity_id]
    return chain_sequences


def mmcif_nonpoly_ligands(cif_dict: MMCIF2Dict) -> list[dict[str, str]]:
    comp_name_lookup = {
        comp_id: name
        for comp_id, name in zip(_list_value(cif_dict, "_chem_comp.id"), _list_value(cif_dict, "_chem_comp.name"))
    }
    entity_ids = _list_value(cif_dict, "_pdbx_entity_nonpoly.entity_id")
    comp_ids = _list_value(cif_dict, "_pdbx_entity_nonpoly.comp_id")
    ligands = []
    for entity_id, comp_id in zip(entity_ids, comp_ids):
        ligands.append(
            {
                "entity_id": entity_id,
                "comp_id": comp_id,
                "name": comp_name_lookup.get(comp_id, ""),
            }
        )
    return ligands


def mmcif_chain_residue_map(cif_dict: MMCIF2Dict) -> dict[str, dict[int, str]]:
    groups = _list_value(cif_dict, "_atom_site.group_PDB")
    chains = _list_value(cif_dict, "_atom_site.auth_asym_id")
    seq_ids = _list_value(cif_dict, "_atom_site.auth_seq_id")
    comp_ids = _list_value(cif_dict, "_atom_site.label_comp_id")
    chain_map: dict[str, dict[int, str]] = {}
    for group, chain, seq_id, comp_id in zip(groups, chains, seq_ids, comp_ids):
        if group != "ATOM" or seq_id in {"?", "."}:
            continue
        try:
            position = int(float(seq_id))
        except ValueError:
            continue
        residue = _residue_to_one_letter(comp_id)
        if residue is None:
            continue
        chain_map.setdefault(chain, {})
        chain_map[chain].setdefault(position, residue)
    return chain_map


def _residue_to_one_letter(comp_id: str) -> str | None:
    token = str(comp_id).strip().upper()
    if token in {"MSE"}:
        return "M"
    if len(token) != 3:
        return None
    try:
        return seq1(token, custom_map={"UNK": "X"}).upper()
    except Exception:
        return None


def chain_completeness(cif_dict: MMCIF2Dict) -> list[dict[str, Any]]:
    chain_entity = mmcif_chain_entity_map(cif_dict)
    entity_sequences = mmcif_entity_sequences(cif_dict)
    residue_map = mmcif_chain_residue_map(cif_dict)
    descriptions = mmcif_entity_descriptions(cif_dict)
    rows = []
    for chain, residues in sorted(residue_map.items()):
        entity_id = chain_entity.get(chain)
        full_sequence = entity_sequences.get(entity_id or "", "")
        observed_count = len(residues)
        expected_count = len(full_sequence)
        completeness = float(observed_count / expected_count) if expected_count else None
        rows.append(
            {
                "chain_id": chain,
                "entity_id": entity_id,
                "entity_description": descriptions.get(entity_id or "", ""),
                "observed_residue_count": observed_count,
                "expected_residue_count": expected_count,
                "chain_completeness": completeness,
            }
        )
    return rows


def pdb_chain_completeness(pdb_path: Path, expected_sequence: str | None = None) -> list[dict[str, Any]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", str(pdb_path))
    expected_length = len(expected_sequence or "")
    rows = []
    for model in structure:
        for chain in model:
            observed = []
            for residue in chain:
                hetflag = residue.id[0]
                if hetflag.strip():
                    continue
                observed.append(residue)
            observed_count = len(observed)
            completeness = float(observed_count / expected_length) if expected_length else None
            rows.append(
                {
                    "chain_id": str(chain.id),
                    "entity_id": None,
                    "entity_description": "predicted_model_chain",
                    "observed_residue_count": observed_count,
                    "expected_residue_count": expected_length,
                    "chain_completeness": completeness,
                }
            )
        break
    return rows


def select_best_template_chain(
    cif_dict: MMCIF2Dict,
    pocket_positions: list[int],
    rt_keywords: list[str],
    exclude_keywords: list[str],
) -> dict[str, Any]:
    descriptions = mmcif_entity_descriptions(cif_dict)
    chain_entity = mmcif_chain_entity_map(cif_dict)
    chain_residues = mmcif_chain_residue_map(cif_dict)
    best: dict[str, Any] | None = None
    for chain_id, residues in sorted(chain_residues.items()):
        entity_id = chain_entity.get(chain_id)
        description = descriptions.get(entity_id or "", "")
        lowered = description.lower()
        if any(keyword in lowered for keyword in exclude_keywords):
            continue
        keyword_hit = any(keyword in lowered for keyword in rt_keywords)
        covered_positions = [position for position in pocket_positions if position in residues]
        if not covered_positions:
            continue
        row = {
            "chain_id": chain_id,
            "entity_id": entity_id,
            "entity_description": description,
            "covered_positions": covered_positions,
            "coverage_fraction": len(covered_positions) / float(len(pocket_positions)),
            "keyword_hit": keyword_hit,
            "pocket_residues": {position: residues[position] for position in covered_positions},
        }
        score = (1 if keyword_hit else 0, row["coverage_fraction"], chain_id)
        if best is None or score > (1 if best["keyword_hit"] else 0, best["coverage_fraction"], best["chain_id"]):
            best = row
    if best is None:
        raise ValueError("Unable to determine template RT chain from mmCIF")
    return best


def _feature_interval(feature: dict[str, Any]) -> tuple[int, int] | None:
    location = feature.get("location") or {}
    start = location.get("start", {}).get("value")
    end = location.get("end", {}).get("value")
    if start is None or end is None:
        return None
    try:
        return int(start), int(end)
    except (TypeError, ValueError):
        return None


def hiv_reference_domains(uniprot_record: dict[str, Any]) -> list[dict[str, Any]]:
    sequence = str(((uniprot_record or {}).get("sequence") or {}).get("value") or "").upper()
    if not sequence:
        return []
    domains: list[dict[str, Any]] = []
    for feature in (uniprot_record or {}).get("features", []):
        feature_type = str(feature.get("type") or "")
        description = str(feature.get("description") or "")
        lowered = description.lower()
        interval = _feature_interval(feature)
        if interval is None or feature_type != "Chain":
            continue
        start, end = interval
        if "reverse transcriptase/ribonuclease h" in lowered:
            family = "rt"
            label = "rt_p66"
        elif "p51 rt" in lowered:
            family = "rt"
            label = "rt_p51"
        elif lowered == "protease":
            family = "protease"
            label = "protease"
        elif lowered == "integrase":
            family = "integrase"
            label = "integrase"
        else:
            continue
        domains.append(
            {
                "family": family,
                "label": label,
                "description": description,
                "start": start,
                "end": end,
                "sequence": sequence[start - 1 : end],
            }
        )
    return domains


def infer_hiv_annotation_label(title: str, description: str) -> str:
    description_text = str(description or "").lower()
    title_text = str(title or "").lower()
    primary_text = description_text if description_text.strip() else title_text
    if "protease" in primary_text:
        return "protease"
    if "integrase" in primary_text:
        return "integrase"
    if (
        "reverse transcriptase" in primary_text
        or "ribonuclease h" in primary_text
        or " p66" in primary_text
        or " p51" in primary_text
        or primary_text.startswith("rt ")
        or " rt " in primary_text
    ):
        return "rt"
    if "gag-pol" in primary_text or "polyprotein" in primary_text:
        return "gag_pol"
    if primary_text.strip():
        return "other"
    return "unknown"


def _alignment_exact_matches(alignment: Any) -> int:
    matches = 0
    target = str(alignment.target)
    query = str(alignment.query)
    for target_block, query_block in zip(alignment.aligned[0], alignment.aligned[1]):
        target_start, target_end = target_block
        query_start, query_end = query_block
        block_length = min(int(target_end - target_start), int(query_end - query_start))
        for offset in range(block_length):
            if target[target_start + offset] == query[query_start + offset]:
                matches += 1
    return matches


def align_chain_to_hiv_domains(chain_sequence: str, reference_domains: list[dict[str, Any]]) -> dict[str, Any]:
    sequence = str(chain_sequence or "").upper()
    if not sequence or not reference_domains:
        return {
            "sequence_best_family": "other",
            "sequence_best_domain": None,
            "sequence_identity": 0.0,
            "sequence_query_coverage": 0.0,
            "sequence_ref_coverage": 0.0,
            "sequence_effective_score": 0.0,
            "sequence_alignment_score": 0.0,
            "sequence_polyprotein_start": None,
            "sequence_polyprotein_end": None,
            "sequence_runner_up_family": None,
            "sequence_runner_up_score": 0.0,
        }

    aligner = Align.PairwiseAligner(mode="local")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5

    scored_rows = []
    for domain in reference_domains:
        alignment = aligner.align(domain["sequence"], sequence)[0]
        aligned_ref = sum(end - start for start, end in alignment.aligned[0])
        aligned_query = sum(end - start for start, end in alignment.aligned[1])
        exact_matches = _alignment_exact_matches(alignment)
        identity = float(exact_matches / aligned_query) if aligned_query else 0.0
        ref_coverage = float(aligned_ref / len(domain["sequence"])) if domain["sequence"] else 0.0
        query_coverage = float(aligned_query / len(sequence)) if sequence else 0.0
        effective_score = identity * ref_coverage * query_coverage
        polyprotein_start = None
        polyprotein_end = None
        if len(alignment.aligned[0]) > 0:
            first_block = alignment.aligned[0][0]
            last_block = alignment.aligned[0][-1]
            polyprotein_start = int(domain["start"] + first_block[0])
            polyprotein_end = int(domain["start"] + last_block[1] - 1)
        scored_rows.append(
            {
                "sequence_best_family": domain["family"],
                "sequence_best_domain": domain["label"],
                "sequence_identity": identity,
                "sequence_query_coverage": query_coverage,
                "sequence_ref_coverage": ref_coverage,
                "sequence_effective_score": effective_score,
                "sequence_alignment_score": float(alignment.score),
                "sequence_polyprotein_start": polyprotein_start,
                "sequence_polyprotein_end": polyprotein_end,
            }
        )
    scored_rows = sorted(
        scored_rows,
        key=lambda row: (
            row["sequence_effective_score"],
            row["sequence_identity"],
            row["sequence_query_coverage"],
            row["sequence_ref_coverage"],
            row["sequence_alignment_score"],
            row["sequence_best_domain"] or "",
        ),
        reverse=True,
    )
    best = dict(scored_rows[0])
    runner_up = scored_rows[1] if len(scored_rows) > 1 else None
    best["sequence_runner_up_family"] = None if runner_up is None else runner_up["sequence_best_family"]
    best["sequence_runner_up_score"] = 0.0 if runner_up is None else float(runner_up["sequence_effective_score"])
    return best


def evaluate_hiv_candidate_chains(
    cif_dict: MMCIF2Dict,
    title: str,
    reference_domains: list[dict[str, Any]],
    template_residues: dict[int, str],
    pocket_positions: list[int],
    rt_keywords: list[str],
    exclude_keywords: list[str],
    min_identity: float,
    min_query_coverage: float,
    min_effective_score: float,
) -> list[dict[str, Any]]:
    descriptions = mmcif_entity_descriptions(cif_dict)
    chain_entity = mmcif_chain_entity_map(cif_dict)
    chain_sequences = mmcif_chain_sequences(cif_dict)
    chain_residues = mmcif_chain_residue_map(cif_dict)
    rows = []
    for chain_id, residues in sorted(chain_residues.items()):
        entity_id = chain_entity.get(chain_id)
        description = descriptions.get(entity_id or "", "")
        lowered = f"{title} {description}".lower()
        excluded = any(keyword in lowered for keyword in exclude_keywords)
        keyword_hit = any(keyword in lowered for keyword in rt_keywords)
        annotation_label = infer_hiv_annotation_label(title, description)
        sequence_alignment = align_chain_to_hiv_domains(chain_sequences.get(chain_id, ""), reference_domains)
        covered_positions = [position for position in pocket_positions if position in template_residues and position in residues]
        matches = sum(1 for position in covered_positions if residues[position] == template_residues[position])
        pocket_similarity = (matches / float(len(covered_positions))) if covered_positions else 0.0
        pass_sequence_domain_gate = bool(
            sequence_alignment["sequence_best_family"] == "rt"
            and float(sequence_alignment["sequence_identity"]) >= min_identity
            and float(sequence_alignment["sequence_query_coverage"]) >= min_query_coverage
            and float(sequence_alignment["sequence_effective_score"]) >= min_effective_score
        )
        pass_rt_domain_gate = bool(
            annotation_label not in {"protease", "integrase", "other"}
            and pass_sequence_domain_gate
        )
        rows.append(
            {
                "chain_id": chain_id,
                "entity_id": entity_id,
                "entity_description": description,
                "excluded_keyword_hit": excluded,
                "rt_keyword_hit": keyword_hit,
                "annotation_label": annotation_label,
                "is_gag_pol_annotation": bool(annotation_label == "gag_pol"),
                "coverage_count": len(covered_positions),
                "coverage_fraction": len(covered_positions) / float(len(pocket_positions)),
                "match_count": matches,
                "pocket_similarity": pocket_similarity,
                "pass_sequence_domain_gate": pass_sequence_domain_gate,
                "pass_rt_domain_gate": pass_rt_domain_gate,
                **sequence_alignment,
            }
        )
    return rows


def nnrti_ligand_summary(cif_dict: MMCIF2Dict, alias_terms: list[str]) -> dict[str, Any]:
    ligands = [ligand for ligand in mmcif_nonpoly_ligands(cif_dict) if ligand["comp_id"].upper() not in HETERO_SKIP_IDS]
    alias_terms = [term.lower() for term in alias_terms]
    matched_ligands = []
    for ligand in ligands:
        text = f"{ligand['comp_id']} {ligand['name']}".lower()
        if any(term in text for term in alias_terms):
            matched_ligands.append(ligand)
    return {
        "ligands": ligands,
        "matched_nnrti_ligands": matched_ligands,
        "has_nonpoly_ligand": bool(ligands),
        "is_holo_nnrti": bool(matched_ligands),
    }
