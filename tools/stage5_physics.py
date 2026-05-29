#!/usr/bin/env python3
"""mutation-effect step structure refinement, fpocket, and MM/GBSA helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager, nullcontext
from collections import Counter
from pathlib import Path
from typing import Any

import fcntl
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Geometry import Point3D

from tools.runtime import command_exists, ensure_dir, iso_now, json_dump, text_dump
from tools.stage35_utils import load_rdkit_molecule, merge_pdb_fragments

LIGAND_RESIDUE_NAME = "LIG"
LIGAND_CHAIN_ID = "Z"
LIGAND_RESIDUE_NUMBER = 900
FPACKET_VOLUME_RE = re.compile(r"^\s*Volume\s*:\s*([0-9.+-Ee]+)")
MMPBSA_DELTA_TOTAL_RE = re.compile(r"^\s*DELTA TOTAL\s+(-?[0-9.+-Ee]+)")
GNINA_AFFINITY_RE = re.compile(r"Affinity:\s*(-?[0-9.+-Ee]+)")
GNINA_GPU_RUNTIME_ERROR_PATTERNS = (
    "CUBLAS_STATUS_NOT_INITIALIZED",
    "GET was unable to find an engine",
    "No GPU detected",
)
RECEPTOR_SPECIAL_RESIDUE_MAP = {
    "CY0": {
        "residue_name": "CYS",
        "record_name": "ATOM  ",
        "atom_map": {
            "N": "N",
            "CA": "CA",
            "C": "C",
            "O": "O",
            "CB": "CB",
            "SAU": "SG",
        },
    },
    "SEP": {
        "residue_name": "SER",
        "record_name": "ATOM  ",
        "atom_map": {
            "N": "N",
            "CA": "CA",
            "C": "C",
            "O": "O",
            "CB": "CB",
            "OG": "OG",
        },
    },
}


def checked_command(
    command: list[str],
    cwd: Path | None = None,
    extra_env: dict[str, str] | None = None,
    timeout_sec: float | None = None,
) -> subprocess.CompletedProcess[str]:
    env = None
    if extra_env:
        env = {**os.environ, **{str(key): str(value) for key, value in extra_env.items()}}
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=None if timeout_sec is None else float(timeout_sec),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        raise RuntimeError(
            f"Command timed out after {float(timeout_sec):.1f}s: {' '.join(command)}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


@contextmanager
def _file_lock(lock_path: Path):
    ensure_dir(lock_path.parent)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _unique_atom_names(molecule: Chem.Mol) -> list[str]:
    counter: Counter[str] = Counter()
    names: list[str] = []
    for atom in molecule.GetAtoms():
        # Preserve the element token casing so two-letter atoms like Cl/Br
        # stay aligned with antechamber mol2 atom names. Uppercasing them to
        # CL/BR makes tleap treat the PDB ligand atoms as template-external.
        symbol = str(atom.GetSymbol())
        counter[symbol] += 1
        names.append(f"{symbol}{counter[symbol]}")
    return names


def write_ligand_pose_pdb_from_sdf(input_sdf: Path, output_pdb: Path) -> None:
    molecule = load_rdkit_molecule(input_sdf, sanitize=False)
    atom_names = _unique_atom_names(molecule)
    for atom, atom_name in zip(molecule.GetAtoms(), atom_names):
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName(LIGAND_RESIDUE_NAME.rjust(3))
        info.SetChainId(LIGAND_CHAIN_ID)
        info.SetResidueNumber(int(LIGAND_RESIDUE_NUMBER))
        info.SetName(f"{atom_name[:4]:>4}")
        atom.SetMonomerInfo(info)
    text_dump(output_pdb, Chem.MolToPDBBlock(molecule))


def _has_finite_conformer_coords(molecule: Chem.Mol) -> bool:
    if molecule.GetNumConformers() <= 0:
        return False
    conformer = molecule.GetConformer()
    for atom_index in range(molecule.GetNumAtoms()):
        position = conformer.GetAtomPosition(atom_index)
        if not math.isfinite(float(position.x)) or not math.isfinite(float(position.y)) or not math.isfinite(float(position.z)):
            return False
    return True


def _fill_missing_hydrogen_coords(molecule: Chem.Mol) -> None:
    if molecule.GetNumConformers() <= 0:
        return
    conformer = molecule.GetConformer()
    hydrogen_counter: Counter[int] = Counter()
    offsets = [
        (0.90, 0.00, 0.00),
        (-0.30, 0.85, 0.00),
        (-0.30, -0.85, 0.00),
        (0.00, 0.00, 0.90),
    ]
    for atom in molecule.GetAtoms():
        if int(atom.GetAtomicNum()) != 1:
            continue
        position = conformer.GetAtomPosition(atom.GetIdx())
        if math.isfinite(float(position.x)) and math.isfinite(float(position.y)) and math.isfinite(float(position.z)):
            continue
        heavy_neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors() if int(neighbor.GetAtomicNum()) > 1]
        if not heavy_neighbors:
            conformer.SetAtomPosition(atom.GetIdx(), Point3D(0.0, 0.0, 0.0))
            continue
        anchor_index = heavy_neighbors[0]
        hydrogen_counter[anchor_index] += 1
        offset = offsets[(hydrogen_counter[anchor_index] - 1) % len(offsets)]
        anchor_position = conformer.GetAtomPosition(anchor_index)
        conformer.SetAtomPosition(
            atom.GetIdx(),
            Point3D(
                float(anchor_position.x) + offset[0],
                float(anchor_position.y) + offset[1],
                float(anchor_position.z) + offset[2],
            ),
        )


def template_pose_sdf(template_sdf: Path, pose_sdf: Path, output_sdf: Path) -> Path:
    template = load_rdkit_molecule(template_sdf, sanitize=True)
    pose = load_rdkit_molecule(pose_sdf, sanitize=False)

    atom_index_pairs: list[tuple[int, int]] | None = None
    template_symbols = [str(atom.GetSymbol()) for atom in template.GetAtoms()]
    pose_symbols = [str(atom.GetSymbol()) for atom in pose.GetAtoms()]
    if template.GetNumAtoms() == pose.GetNumAtoms() and template_symbols == pose_symbols:
        atom_index_pairs = [(atom_index, atom_index) for atom_index in range(template.GetNumAtoms())]
    else:
        template_heavy_atom_indices = [atom.GetIdx() for atom in template.GetAtoms() if int(atom.GetAtomicNum()) > 1]
        pose_heavy_atom_indices = [atom.GetIdx() for atom in pose.GetAtoms() if int(atom.GetAtomicNum()) > 1]
        template_heavy_symbols = [str(template.GetAtomWithIdx(atom_index).GetSymbol()) for atom_index in template_heavy_atom_indices]
        pose_heavy_symbols = [str(pose.GetAtomWithIdx(atom_index).GetSymbol()) for atom_index in pose_heavy_atom_indices]
        if len(template_heavy_atom_indices) == len(pose_heavy_atom_indices) and template_heavy_symbols == pose_heavy_symbols:
            atom_index_pairs = list(zip(template_heavy_atom_indices, pose_heavy_atom_indices))
        elif len(template_heavy_atom_indices) == len(pose_heavy_atom_indices):
            template_nohyd = Chem.RemoveHs(Chem.Mol(template), sanitize=True)
            pose_nohyd = Chem.RemoveHs(Chem.Mol(pose), sanitize=False)
            mcs = rdFMCS.FindMCS(
                [template_nohyd, pose_nohyd],
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareAny,
                ringMatchesRingOnly=False,
                completeRingsOnly=False,
                timeout=10,
            )
            if int(mcs.numAtoms) == template_nohyd.GetNumAtoms() == pose_nohyd.GetNumAtoms():
                pattern = Chem.MolFromSmarts(mcs.smartsString)
                template_match = template_nohyd.GetSubstructMatch(pattern)
                pose_match = pose_nohyd.GetSubstructMatch(pattern)
                if template_match and pose_match and len(template_match) == len(pose_match):
                    atom_index_pairs = [
                        (template_heavy_atom_indices[template_pos], pose_heavy_atom_indices[pose_pos])
                        for template_pos, pose_pos in zip(template_match, pose_match)
                    ]
    if atom_index_pairs is None:
        shutil.copyfile(pose_sdf, output_sdf)
        return output_sdf

    remapped = Chem.Mol(template)
    remapped.RemoveAllConformers()
    template_conf = template.GetConformer()
    pose_conf = pose.GetConformer()
    new_conf = Chem.Conformer(remapped.GetNumAtoms())
    for atom_index in range(remapped.GetNumAtoms()):
        position = template_conf.GetAtomPosition(atom_index)
        new_conf.SetAtomPosition(atom_index, position)
    for template_atom_index, pose_atom_index in atom_index_pairs:
        position = pose_conf.GetAtomPosition(pose_atom_index)
        new_conf.SetAtomPosition(template_atom_index, position)
    remapped.AddConformer(new_conf, assignId=True)
    if len(atom_index_pairs) != remapped.GetNumAtoms():
        try:
            rebuilt = Chem.AddHs(Chem.RemoveHs(remapped, sanitize=True), addCoords=True)
            if _has_finite_conformer_coords(rebuilt):
                remapped = rebuilt
        except Exception:
            pass
    writer = Chem.SDWriter(str(output_sdf))
    writer.write(remapped)
    writer.close()
    return output_sdf


def rewrite_ligand_pdb_tags(input_pdb: Path, output_pdb: Path) -> None:
    lines: list[str] = []
    for line in input_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        rewritten = list(f"{line:<80}"[:80])
        rewritten[17:20] = list(LIGAND_RESIDUE_NAME.rjust(3))
        rewritten[21] = LIGAND_CHAIN_ID
        rewritten[22:26] = list(f"{LIGAND_RESIDUE_NUMBER:4d}")
        lines.append("".join(rewritten).rstrip())
    lines.append("END")
    text_dump(output_pdb, "\n".join(lines) + "\n")


def strip_hydrogens_from_pdb(input_pdb: Path, output_pdb: Path) -> None:
    lines: list[str] = []
    for line in input_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM", "TER")):
            continue
        if line.startswith("TER"):
            lines.append(line.rstrip())
            continue
        atom_name = str(line[12:16]).strip().upper()
        element = str(line[76:78]).strip().upper()
        if element == "H" or atom_name.startswith("H") or atom_name[1:].startswith("H"):
            continue
        lines.append(line.rstrip())
    lines.append("END")
    text_dump(output_pdb, "\n".join(lines) + "\n")


def normalize_special_receptor_residues(input_pdb: Path, output_pdb: Path) -> dict[str, Any]:
    lines: list[str] = []
    normalized_residue_counts: Counter[str] = Counter()
    normalized_atom_count = 0
    removed_atom_count = 0
    for line in input_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("TER"):
            lines.append(line.rstrip())
            continue
        if not line.startswith(("ATOM", "HETATM")):
            continue
        residue_name = str(line[17:20]).strip().upper()
        residue_rule = RECEPTOR_SPECIAL_RESIDUE_MAP.get(residue_name)
        if residue_rule is None:
            lines.append(line.rstrip())
            continue
        atom_name = str(line[12:16]).strip().upper()
        mapped_atom_name = residue_rule["atom_map"].get(atom_name)
        if mapped_atom_name is None:
            removed_atom_count += 1
            continue
        rewritten = list(f"{line:<80}"[:80])
        rewritten[0:6] = list(str(residue_rule.get("record_name", line[0:6]))[:6].ljust(6))
        rewritten[12:16] = list(f"{mapped_atom_name[:4]:>4}")
        rewritten[17:20] = list(str(residue_rule["residue_name"]).rjust(3))
        lines.append("".join(rewritten).rstrip())
        normalized_residue_counts[residue_name] += 1
        normalized_atom_count += 1
    lines.append("END")
    text_dump(output_pdb, "\n".join(lines) + "\n")
    return {
        "normalized_receptor_residues": dict(sorted(normalized_residue_counts.items())),
        "normalized_receptor_atom_count": int(normalized_atom_count),
        "removed_receptor_atom_count": int(removed_atom_count),
    }


def prepare_receptor_for_amber(input_pdb: Path, output_pdb: Path) -> dict[str, Any]:
    nohyd_pdb = output_pdb.with_name(f"{output_pdb.stem}_nohyd.pdb")
    normalized_pdb = output_pdb.with_name(f"{output_pdb.stem}_normalized.pdb")
    strip_hydrogens_from_pdb(input_pdb, nohyd_pdb)
    normalization_payload = normalize_special_receptor_residues(nohyd_pdb, normalized_pdb)
    method = "strip_hydrogens"
    if command_exists("pdb4amber"):
        try:
            checked_command(
                [
                    "pdb4amber",
                    "-i",
                    str(normalized_pdb),
                    "-o",
                    str(output_pdb),
                    "-d",
                ],
                cwd=output_pdb.parent,
            )
            method = "pdb4amber"
        except Exception:
            shutil.copyfile(normalized_pdb, output_pdb)
            method = "strip_hydrogens_fallback"
    else:
        shutil.copyfile(normalized_pdb, output_pdb)
    return {
        "receptor_input_pdb": str(input_pdb),
        "receptor_amber_pdb": str(output_pdb),
        "receptor_prep_method": method,
        **normalization_payload,
    }


def strip_residue_from_pdb(input_pdb: Path, residue_name: str, output_pdb: Path) -> None:
    residue_token = str(residue_name).upper()
    lines: list[str] = []
    for line in input_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")) and str(line[17:20]).strip().upper() == residue_token:
            continue
        if line.startswith(("ATOM", "HETATM", "TER")):
            lines.append(line.rstrip())
    lines.append("END")
    text_dump(output_pdb, "\n".join(lines) + "\n")


def extract_residue_from_pdb(input_pdb: Path, residue_name: str, output_pdb: Path) -> None:
    residue_token = str(residue_name).upper()
    lines: list[str] = []
    for line in input_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")) and str(line[17:20]).strip().upper() == residue_token:
            lines.append(line.rstrip())
    lines.append("END")
    text_dump(output_pdb, "\n".join(lines) + "\n")


def parse_fpocket_info(info_path: Path) -> list[dict[str, Any]]:
    pockets: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in info_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("pocket ") and line.endswith(":"):
            if current is not None:
                pockets.append(current)
            pocket_id = int(re.findall(r"\d+", line)[0])
            current = {"pocket_id": pocket_id, "volume_a3": None}
            continue
        if current is None:
            continue
        match = FPACKET_VOLUME_RE.match(line)
        if match:
            current["volume_a3"] = float(match.group(1))
    if current is not None:
        pockets.append(current)
    return pockets


def fpocket_top_pocket_metrics(receptor_pdb: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="stage5_fpocket_") as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        work_pdb = temp_dir / receptor_pdb.name
        shutil.copyfile(receptor_pdb, work_pdb)
        checked_command(["fpocket", "-f", str(work_pdb)], cwd=temp_dir)
        info_path = temp_dir / f"{work_pdb.stem}_out" / f"{work_pdb.stem}_info.txt"
        if not info_path.exists():
            raise RuntimeError(f"fpocket did not produce {info_path.name} for {receptor_pdb}")
        pockets = parse_fpocket_info(info_path)
        if not pockets:
            raise RuntimeError(f"fpocket did not report any pocket volume for {receptor_pdb}")
        top = pockets[0]
        return {
            "pocket_id": int(top["pocket_id"]),
            "volume_a3": None if top.get("volume_a3") is None else float(top["volume_a3"]),
        }


def _tleap_script(prep_root: Path) -> str:
    return "\n".join(
        [
            "source leaprc.protein.ff14SB",
            "source leaprc.gaff2",
            "set default PBRadii mbondi3",
            f"{LIGAND_RESIDUE_NAME} = loadmol2 {prep_root / 'ligand.mol2'}",
            f"loadamberparams {prep_root / 'ligand.frcmod'}",
            f"receptor = loadpdb {prep_root / 'receptor_amber.pdb'}",
            f"complex = loadpdb {prep_root / 'complex_amber.pdb'}",
            f"saveamberparm receptor {prep_root / 'receptor.prmtop'} {prep_root / 'receptor.inpcrd'}",
            f"saveamberparm complex {prep_root / 'complex.prmtop'} {prep_root / 'complex.inpcrd'}",
            f"saveamberparm {LIGAND_RESIDUE_NAME} {prep_root / 'ligand.prmtop'} {prep_root / 'ligand.inpcrd'}",
            "quit",
            "",
        ]
    )


def ensure_explicit_hydrogens_sdf(input_sdf: Path, output_sdf: Path) -> dict[str, Any]:
    molecule = load_rdkit_molecule(input_sdf, sanitize=False)
    heavy_atom_count = sum(1 for atom in molecule.GetAtoms() if int(atom.GetAtomicNum()) > 1)
    if molecule.GetNumAtoms() > heavy_atom_count:
        if input_sdf.resolve() != output_sdf.resolve():
            shutil.copyfile(input_sdf, output_sdf)
        return {
            "ligand_hydrogenation_method": "preserved_explicit_hydrogens",
            "ligand_input_atom_count": int(molecule.GetNumAtoms()),
            "ligand_output_atom_count": int(molecule.GetNumAtoms()),
        }
    prepared = Chem.Mol(molecule)
    Chem.SanitizeMol(prepared)
    for conformer in prepared.GetConformers():
        conformer.Set3D(True)
    prepared = Chem.AddHs(Chem.RemoveHs(prepared, sanitize=True), addCoords=True)
    _fill_missing_hydrogen_coords(prepared)
    writer = Chem.SDWriter(str(output_sdf))
    writer.write(prepared)
    writer.close()
    return {
        "ligand_hydrogenation_method": "rdkit_add_hs",
        "ligand_input_atom_count": int(molecule.GetNumAtoms()),
        "ligand_output_atom_count": int(prepared.GetNumAtoms()),
    }


def _ligand_parameter_cache_key(input_sdf: Path) -> str:
    molecule = load_rdkit_molecule(input_sdf, sanitize=False)
    try:
        connectivity_smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(molecule)), canonical=True)
    except Exception:
        connectivity_smiles = Chem.MolToSmiles(Chem.Mol(molecule), canonical=True)
    payload = {
        "connectivity_smiles": connectivity_smiles,
        "atom_name_sequence": _unique_atom_names(molecule),
        "formal_charges": [int(atom.GetFormalCharge()) for atom in molecule.GetAtoms()],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return digest[:16]


def _prepare_ligand_parameter_files(
    *,
    amber_ligand_input: Path,
    ligand_mol2: Path,
    ligand_frcmod: Path,
    ligand_parameter_cache_root: Path | None = None,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    cache_key = None
    cache_hit = False
    source_mol2 = ligand_mol2
    source_frcmod = ligand_frcmod
    lock_path: Path | None = None

    if ligand_parameter_cache_root is not None:
        cache_key = _ligand_parameter_cache_key(amber_ligand_input)
        cache_dir = ensure_dir(ligand_parameter_cache_root / cache_key)
        source_mol2 = cache_dir / "ligand.mol2"
        source_frcmod = cache_dir / "ligand.frcmod"
        lock_path = cache_dir / ".prepare.lock"
        cache_hit = source_mol2.exists() and source_frcmod.exists()

    lock_context = _file_lock(lock_path) if lock_path is not None else nullcontext()
    with lock_context:
        cache_hit = source_mol2.exists() and source_frcmod.exists()
        if not source_mol2.exists() or not source_frcmod.exists():
            checked_command(
                [
                    "antechamber",
                    "-i",
                    str(amber_ligand_input),
                    "-fi",
                    "sdf",
                    "-o",
                    str(source_mol2),
                    "-fo",
                    "mol2",
                    "-at",
                    "gaff2",
                    "-c",
                    "bcc",
                    "-s",
                    "2",
                    "-rn",
                    LIGAND_RESIDUE_NAME,
                    "-dr",
                    "no",
                ],
                cwd=source_mol2.parent,
                timeout_sec=timeout_sec,
            )
            checked_command(
                [
                    "parmchk2",
                    "-i",
                    str(source_mol2),
                    "-f",
                    "mol2",
                    "-o",
                    str(source_frcmod),
                ],
                cwd=source_frcmod.parent,
                timeout_sec=timeout_sec,
            )
            cache_hit = False

    if source_mol2.resolve() != ligand_mol2.resolve():
        ensure_dir(ligand_mol2.parent)
        shutil.copyfile(source_mol2, ligand_mol2)
    if source_frcmod.resolve() != ligand_frcmod.resolve():
        ensure_dir(ligand_frcmod.parent)
        shutil.copyfile(source_frcmod, ligand_frcmod)

    return {
        "ligand_parameter_cache_root": None if ligand_parameter_cache_root is None else str(ligand_parameter_cache_root),
        "ligand_parameter_cache_key": cache_key,
        "ligand_parameter_cache_hit": bool(cache_hit),
        "ligand_parameter_source_mol2": str(source_mol2),
        "ligand_parameter_source_frcmod": str(source_frcmod),
    }


def probe_ligand_parameterization(
    *,
    input_sdf: Path,
    work_root: Path,
    ligand_parameter_cache_root: Path | None = None,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    probe_root = ensure_dir(work_root)
    amber_ligand_input = probe_root / "ligand_amber_input.sdf"
    ligand_mol2 = probe_root / "ligand.mol2"
    ligand_frcmod = probe_root / "ligand.frcmod"
    payload: dict[str, Any] = {
        "probe_at": iso_now(),
        "input_sdf": str(input_sdf),
        "available": False,
        "error": None,
    }
    try:
        hydrogenation_payload = ensure_explicit_hydrogens_sdf(input_sdf, amber_ligand_input)
        parameter_payload = _prepare_ligand_parameter_files(
            amber_ligand_input=amber_ligand_input,
            ligand_mol2=ligand_mol2,
            ligand_frcmod=ligand_frcmod,
            ligand_parameter_cache_root=ligand_parameter_cache_root,
            timeout_sec=timeout_sec,
        )
        payload.update(hydrogenation_payload)
        payload.update(parameter_payload)
        payload["available"] = True
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    json_dump(probe_root / "ligand_parameterization_probe.json", payload)
    return payload


def prepare_amber_complex(
    *,
    receptor_pdb: Path,
    ligand_pose_sdf: Path,
    work_root: Path,
    ligand_template_sdf: Path | None = None,
    ligand_parameter_cache_root: Path | None = None,
) -> dict[str, str]:
    prep_root = ensure_dir(work_root)
    ligand_mol2 = prep_root / "ligand.mol2"
    ligand_frcmod = prep_root / "ligand.frcmod"
    ligand_pose_pdb = prep_root / "ligand_pose.pdb"
    receptor_amber_pdb = prep_root / "receptor_amber.pdb"
    complex_amber_pdb = prep_root / "complex_amber.pdb"
    tleap_input = prep_root / "tleap.in"
    amber_ligand_input = prep_root / "ligand_amber_input.sdf"
    ligand_hydrogenation_payload: dict[str, Any]
    if ligand_template_sdf is not None and ligand_template_sdf.exists():
        mapped_ligand_input = template_pose_sdf(
            ligand_template_sdf,
            ligand_pose_sdf,
            amber_ligand_input,
        )
        ligand_hydrogenation_payload = ensure_explicit_hydrogens_sdf(mapped_ligand_input, amber_ligand_input)
    else:
        ligand_hydrogenation_payload = ensure_explicit_hydrogens_sdf(ligand_pose_sdf, amber_ligand_input)

    write_ligand_pose_pdb_from_sdf(amber_ligand_input, ligand_pose_pdb)
    receptor_prep = prepare_receptor_for_amber(receptor_pdb, receptor_amber_pdb)
    merge_pdb_fragments([receptor_amber_pdb, ligand_pose_pdb], complex_amber_pdb)
    ligand_parameter_payload = _prepare_ligand_parameter_files(
        amber_ligand_input=amber_ligand_input,
        ligand_mol2=ligand_mol2,
        ligand_frcmod=ligand_frcmod,
        ligand_parameter_cache_root=ligand_parameter_cache_root,
    )
    tleap_input.write_text(_tleap_script(prep_root), encoding="utf-8")
    checked_command(["tleap", "-f", str(tleap_input)], cwd=prep_root)

    payload = {
        "prepared_at": iso_now(),
        "receptor_pdb": str(receptor_pdb),
        "ligand_pose_sdf": str(ligand_pose_sdf),
        "ligand_template_sdf": None if ligand_template_sdf is None else str(ligand_template_sdf),
        "amber_ligand_input_sdf": str(amber_ligand_input),
        "receptor_amber_pdb": str(receptor_amber_pdb),
        "complex_amber_pdb": str(complex_amber_pdb),
        "ligand_pose_pdb": str(ligand_pose_pdb),
        "ligand_mol2": str(ligand_mol2),
        "ligand_frcmod": str(ligand_frcmod),
        "complex_prmtop": str(prep_root / "complex.prmtop"),
        "complex_inpcrd": str(prep_root / "complex.inpcrd"),
        "receptor_prmtop": str(prep_root / "receptor.prmtop"),
        "receptor_inpcrd": str(prep_root / "receptor.inpcrd"),
        "ligand_prmtop": str(prep_root / "ligand.prmtop"),
        "ligand_inpcrd": str(prep_root / "ligand.inpcrd"),
        **ligand_hydrogenation_payload,
        **ligand_parameter_payload,
        **receptor_prep,
    }
    json_dump(prep_root / "amber_prep_manifest.json", payload)
    return payload


def _min_mdin(stage5: dict[str, Any]) -> str:
    restraint = float(stage5.get("local_sampling_backbone_restraint_kcal_mol_a2", 1.0))
    return "\n".join(
        [
            "Stage5 restrained minimization",
            "&cntrl",
            "  imin=1, maxcyc=2500, ncyc=1000,",
            "  ntb=0, igb=8, cut=999.0,",
            "  ntpr=100, ntr=1,",
            f"  restraint_wt={restraint:.3f},",
            "  restraintmask='@CA,C,N,O',",
            "/",
            "",
        ]
    )


def _md_mdin(stage5: dict[str, Any]) -> str:
    restraint = float(stage5.get("local_sampling_backbone_restraint_kcal_mol_a2", 1.0))
    local_sampling_ns = float(stage5.get("local_sampling_ns", 5.0))
    nstlim = max(1, int(round(local_sampling_ns * 500000.0)))
    return "\n".join(
        [
            "Stage5 restrained local sampling",
            "&cntrl",
            "  imin=0, irest=0, ntx=1,",
            "  ntb=0, igb=8, cut=999.0,",
            "  dt=0.002, nstlim=%d," % nstlim,
            "  ntc=2, ntf=2,",
            "  ntt=3, gamma_ln=2.0, tempi=300.0, temp0=300.0,",
            "  ntpr=2500, ntwx=2500, ioutfm=1,",
            "  ntr=1,",
            f"  restraint_wt={restraint:.3f},",
            "  restraintmask='@CA,C,N,O',",
            "/",
            "",
        ]
    )


def _restart_to_pdb(prmtop: Path, restart_path: Path, output_pdb: Path) -> None:
    if not command_exists("ambpdb"):
        raise RuntimeError("ambpdb is required to convert AMBER restart files into refined PDB outputs.")
    result = checked_command(
        [
            "ambpdb",
            "-p",
            str(prmtop),
            "-c",
            str(restart_path),
        ],
        cwd=output_pdb.parent,
    )
    text_dump(output_pdb, result.stdout)


def amber_md_instability_reason(output_path: Path) -> str | None:
    if not output_path.exists():
        return "missing_md_output"
    text = output_path.read_text(encoding="utf-8", errors="ignore")
    if "********" in text:
        return "masked_numeric_overflow"
    lowered = text.lower()
    if "nan" in lowered:
        return "nan_detected"
    if "vlimit exceeded" in lowered:
        return "vlimit_exceeded"
    if "coordinate resetting cannot be accomplished" in lowered:
        return "coordinate_reset_failure"
    return None


def run_amber_relaxation(
    *,
    amber_prep: dict[str, str],
    work_root: Path,
    stage5: dict[str, Any],
    run_local_sampling: bool,
    cuda_visible_devices: str | int | None = None,
) -> dict[str, Any]:
    run_root = ensure_dir(work_root)
    min_in = run_root / "min.in"
    md_in = run_root / "md.in"
    min_in.write_text(_min_mdin(stage5), encoding="utf-8")
    md_in.write_text(_md_mdin(stage5), encoding="utf-8")
    complex_prmtop = Path(amber_prep["complex_prmtop"])
    complex_inpcrd = Path(amber_prep["complex_inpcrd"])
    min_rst7 = run_root / "min.rst7"
    refined_complex_pdb = run_root / "refined_complex.pdb"
    refined_receptor_pdb = run_root / "refined_receptor.pdb"
    refined_ligand_pdb = run_root / "refined_ligand.pdb"
    refined_ligand_sdf = run_root / "refined_ligand.sdf"

    min_engine = "sander"
    min_env = None
    if command_exists("pmemd.cuda") and cuda_visible_devices not in {None, ""}:
        min_engine = "pmemd.cuda"
        min_env = {"CUDA_VISIBLE_DEVICES": str(cuda_visible_devices)}

    checked_command(
        [
            min_engine,
            "-O",
            "-i",
            str(min_in),
            "-o",
            str(run_root / "min.out"),
            "-p",
            str(complex_prmtop),
            "-c",
            str(complex_inpcrd),
            "-r",
            str(min_rst7),
            "-ref",
            str(complex_inpcrd),
        ],
        cwd=run_root,
        extra_env=min_env,
    )

    trajectory_path = min_rst7
    final_restart_path = min_rst7
    relaxation_mode = "minimize_only"
    local_sampling_fallback_reason = None
    local_sampling_output_path = run_root / "md.out"
    local_sampling_restart_path = run_root / "md.rst7"
    if run_local_sampling:
        md_nc = run_root / "md.nc"
        md_engine = "pmemd.cuda" if command_exists("pmemd.cuda") else "sander"
        md_env = None
        if md_engine == "pmemd.cuda" and cuda_visible_devices not in {None, ""}:
            md_env = {"CUDA_VISIBLE_DEVICES": str(cuda_visible_devices)}
        try:
            checked_command(
                [
                    md_engine,
                    "-O",
                    "-i",
                    str(md_in),
                    "-o",
                    str(local_sampling_output_path),
                    "-p",
                    str(complex_prmtop),
                    "-c",
                    str(min_rst7),
                    "-r",
                    str(local_sampling_restart_path),
                    "-x",
                    str(md_nc),
                    "-ref",
                    str(min_rst7),
                ],
                cwd=run_root,
                extra_env=md_env,
            )
        except Exception as exc:
            local_sampling_fallback_reason = f"md_command_failed:{type(exc).__name__}"
        else:
            local_sampling_fallback_reason = amber_md_instability_reason(local_sampling_output_path)
            if local_sampling_fallback_reason is None and md_nc.exists() and md_nc.stat().st_size > 0:
                trajectory_path = md_nc
                final_restart_path = local_sampling_restart_path
                relaxation_mode = "local_sampling_implicit_md"
            else:
                local_sampling_fallback_reason = local_sampling_fallback_reason or "missing_md_trajectory"
        if local_sampling_fallback_reason is not None:
            trajectory_path = min_rst7
            final_restart_path = min_rst7
            relaxation_mode = "local_sampling_fallback_minimize_only"

    _restart_to_pdb(complex_prmtop, final_restart_path, refined_complex_pdb)
    strip_residue_from_pdb(refined_complex_pdb, LIGAND_RESIDUE_NAME, refined_receptor_pdb)
    extract_residue_from_pdb(refined_complex_pdb, LIGAND_RESIDUE_NAME, refined_ligand_pdb)
    checked_command(["obabel", str(refined_ligand_pdb), "-O", str(refined_ligand_sdf)], cwd=run_root)

    payload = {
        "relaxed_at": iso_now(),
        "relaxation_mode": relaxation_mode,
        "run_local_sampling": bool(run_local_sampling),
        "local_sampling_fallback_reason": local_sampling_fallback_reason,
        "min_engine": min_engine,
        "cuda_visible_devices": None if cuda_visible_devices in {None, ""} else str(cuda_visible_devices),
        "trajectory_path": str(trajectory_path),
        "final_restart_path": str(final_restart_path),
        "refined_complex_pdb": str(refined_complex_pdb),
        "refined_receptor_pdb": str(refined_receptor_pdb),
        "refined_ligand_pdb": str(refined_ligand_pdb),
        "refined_ligand_sdf": str(refined_ligand_sdf),
        "min_restart": str(min_rst7),
        "local_sampling_output": str(local_sampling_output_path),
        "local_sampling_restart": str(local_sampling_restart_path),
    }
    json_dump(run_root / "relaxation_manifest.json", payload)
    return payload


def parse_mmpbsa_results(path: Path) -> dict[str, Any]:
    delta_total = None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = MMPBSA_DELTA_TOTAL_RE.match(raw_line)
        if match:
            delta_total = float(match.group(1))
    if delta_total is None:
        raise RuntimeError(f"Unable to parse DELTA TOTAL from {path}")
    return {"delta_total_kcal_mol": float(delta_total)}


def run_mmgbsa(
    *,
    amber_prep: dict[str, str],
    relaxation_payload: dict[str, Any],
    work_root: Path,
    stage5: dict[str, Any],
) -> dict[str, Any]:
    mmgbsa_root = ensure_dir(work_root)
    input_path = mmgbsa_root / "mmpbsa.in"
    use_sander = bool(stage5.get("mmgbsa_use_sander", True))
    input_path.write_text(_mmpbsa_input_text(stage5, use_sander=use_sander), encoding="utf-8")
    output_path = mmgbsa_root / "FINAL_RESULTS_MMPBSA.dat"
    per_frame_path = mmgbsa_root / "FINAL_RESULTS_MMPBSA.csv"
    checked_command(
        [
            "MMPBSA.py",
            "-O",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "-eo",
            str(per_frame_path),
            "-cp",
            str(amber_prep["complex_prmtop"]),
            "-rp",
            str(amber_prep["receptor_prmtop"]),
            "-lp",
            str(amber_prep["ligand_prmtop"]),
            "-y",
            str(relaxation_payload["trajectory_path"]),
        ],
        cwd=mmgbsa_root,
    )
    parsed = parse_mmpbsa_results(output_path)
    payload = {
        "mmgbsa_at": iso_now(),
        "use_sander": use_sander,
        "final_results_path": str(output_path),
        "per_frame_csv": str(per_frame_path),
        **parsed,
    }
    json_dump(mmgbsa_root / "mmgbsa_manifest.json", payload)
    return payload


def _mmpbsa_input_text(stage5: dict[str, Any], *, use_sander: bool) -> str:
    general_tokens = [
        "startframe=1",
        "interval=%d" % int(stage5.get("mmgbsa_frame_interval", 1)),
        "verbose=1",
    ]
    if use_sander:
        general_tokens.append("use_sander=1")
    gb_tokens = [
        "igb=%d" % int(stage5.get("mmgbsa_igb", 8)),
        "saltcon=%.3f" % float(stage5.get("mmgbsa_saltcon", 0.150)),
    ]
    return "\n".join(
        [
            "&general",
            "  %s," % ", ".join(general_tokens),
            "/",
            "&gb",
            "  %s," % ", ".join(gb_tokens),
            "/",
            "",
        ]
    )


def gnina_score_only(
    receptor_pdb: Path,
    ligand_sdf: Path,
    work_root: Path,
    cuda_visible_devices: int | str | None = None,
) -> dict[str, Any]:
    if not command_exists("gnina"):
        return {"available": False, "affinity_kcal_mol": None}
    command = [
        "gnina",
        "--receptor",
        str(receptor_pdb),
        "--ligand",
        str(ligand_sdf),
        "--score_only",
    ]
    # Default to CPU-only scoring. Callers that intentionally want GPU gnina
    # must pass a concrete cuda_visible_devices value.
    extra_env = {"CUDA_VISIBLE_DEVICES": "" if cuda_visible_devices is None else str(cuda_visible_devices)}
    used_cpu_fallback = False
    try:
        result = checked_command(
            command,
            cwd=work_root,
            extra_env=extra_env,
        )
    except RuntimeError as exc:
        error_text = str(exc)
        gpu_requested = cuda_visible_devices not in {None, ""}
        if not gpu_requested or not any(pattern in error_text for pattern in GNINA_GPU_RUNTIME_ERROR_PATTERNS):
            raise
        used_cpu_fallback = True
        result = checked_command(
            command,
            cwd=work_root,
            extra_env={"CUDA_VISIBLE_DEVICES": ""},
        )
    affinities = [float(match.group(1)) for match in GNINA_AFFINITY_RE.finditer(result.stdout or "")]
    return {
        "available": True,
        "affinity_kcal_mol": affinities[-1] if affinities else None,
        "stdout": result.stdout,
        "used_cpu_fallback": bool(used_cpu_fallback),
    }


def score_direction(value: float | None, neutral_threshold: float) -> int:
    if value is None or not math.isfinite(float(value)):
        return 0
    if float(value) >= float(neutral_threshold):
        return 1
    if float(value) <= -float(neutral_threshold):
        return -1
    return 0


def multi_score_consensus(
    score_values: dict[str, float | None],
    neutral_threshold: float,
    consensus_threshold: float,
) -> dict[str, Any]:
    directions = {
        name: score_direction(value, neutral_threshold)
        for name, value in score_values.items()
        if value is not None and math.isfinite(float(value))
    }
    nonzero = {name: direction for name, direction in directions.items() if direction != 0}
    if not nonzero:
        return {
            "available_score_count": int(len(directions)),
            "nonzero_direction_count": 0,
            "consensus_fraction": None,
            "consensus_direction": "neutral",
            "high_uncertainty": True,
        }
    counts = Counter(nonzero.values())
    dominant_direction, dominant_count = max(counts.items(), key=lambda item: (item[1], item[0]))
    consensus_fraction = float(dominant_count / max(1, len(nonzero)))
    direction_label = {1: "resistance_like", -1: "sensitizing_like"}.get(dominant_direction, "neutral")
    return {
        "available_score_count": int(len(directions)),
        "nonzero_direction_count": int(len(nonzero)),
        "consensus_fraction": float(consensus_fraction),
        "consensus_direction": direction_label,
        "high_uncertainty": bool(consensus_fraction < float(consensus_threshold)),
    }
