#!/usr/bin/env python3
"""Stage 3.5 WT baseline complex, docking, and interaction helpers."""

from __future__ import annotations

import json
import io
import math
import shutil
import subprocess
import tempfile
import warnings
from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd
from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign

from tools.runtime import ensure_dir, json_dump, text_dump
from tools.stage3_utils import PDB_HETERO_SKIP_IDS

METAL_ELEMENTS = {
    "LI",
    "NA",
    "K",
    "RB",
    "CS",
    "BE",
    "MG",
    "CA",
    "SR",
    "BA",
    "ZN",
    "MN",
    "FE",
    "CO",
    "NI",
    "CU",
    "CD",
}

PLIP_INTERACTION_TYPES = (
    "hydrogen_bond",
    "salt_bridge",
    "hydrophobic",
    "pi_stacking",
    "pi_cation",
    "metal_complex",
)


@dataclass
class HeteroResidue:
    chain_id: str
    residue_name: str
    residue_number: int
    insertion_code: str
    atom_count: int
    heavy_atom_count: int
    centroid: tuple[float, float, float]
    residue_id: tuple[str, int, str]


class ChainProteinSelect(Select):
    def __init__(self, chain_id: str) -> None:
        self.chain_id = str(chain_id)

    def accept_chain(self, chain) -> bool:
        return str(chain.id) == self.chain_id

    def accept_residue(self, residue) -> bool:
        return residue.id[0] == " "


class ChainSetProteinSelect(Select):
    def __init__(self, chain_ids: Iterable[str]) -> None:
        self.chain_ids = {str(value) for value in chain_ids if str(value)}

    def accept_chain(self, chain) -> bool:
        return str(chain.id) in self.chain_ids

    def accept_residue(self, residue) -> bool:
        return residue.id[0] == " "


class ResidueSetSelect(Select):
    def __init__(self, chain_id: str, residue_ids: set[tuple[str, int, str]]) -> None:
        self.chain_id = str(chain_id)
        self.residue_ids = residue_ids

    def accept_chain(self, chain) -> bool:
        return str(chain.id) == self.chain_id

    def accept_residue(self, residue) -> bool:
        residue_key = (
            str(residue.id[0]),
            int(residue.id[1]),
            str(residue.id[2]).strip(),
        )
        return residue_key in self.residue_ids


def parser() -> PDBParser:
    return PDBParser(QUIET=True)


@contextmanager
def suppress_rdkit_logs() -> Iterator[None]:
    RDLogger.DisableLog("rdApp.*")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.*")


@contextmanager
def suppress_noisy_stderr() -> Iterator[None]:
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        yield


def load_structure(path: Path):
    return parser().get_structure(path.stem, str(path))


def first_model(structure):
    return next(structure.get_models())


def atom_coordinates(items: Iterable[Any]) -> list[tuple[float, float, float]]:
    coords: list[tuple[float, float, float]] = []
    for item in items:
        coord = item.get_coord()
        coords.append((float(coord[0]), float(coord[1]), float(coord[2])))
    return coords


def centroid(coords: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    if not coords:
        return (0.0, 0.0, 0.0)
    count = float(len(coords))
    return (
        sum(value[0] for value in coords) / count,
        sum(value[1] for value in coords) / count,
        sum(value[2] for value in coords) / count,
    )


def max_axis_span(coords: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    if not coords:
        return (0.0, 0.0, 0.0)
    xs = [value[0] for value in coords]
    ys = [value[1] for value in coords]
    zs = [value[2] for value in coords]
    return (
        float(max(xs) - min(xs)),
        float(max(ys) - min(ys)),
        float(max(zs) - min(zs)),
    )


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def min_distance_between_sets(
    lhs: list[tuple[float, float, float]],
    rhs: list[tuple[float, float, float]],
) -> float | None:
    if not lhs or not rhs:
        return None
    best = None
    for left in lhs:
        for right in rhs:
            current = distance(left, right)
            if best is None or current < best:
                best = current
    return best


def hetero_residues_for_chain(
    pdb_path: Path,
    chain_id: str,
    skip_residue_names: set[str] | None = None,
) -> list[HeteroResidue]:
    structure = load_structure(pdb_path)
    model = first_model(structure)
    chain = model[str(chain_id)]
    skip_residue_names = {value.upper() for value in (skip_residue_names or set())}
    residues: list[HeteroResidue] = []
    for residue in chain:
        hetflag, resseq, icode = residue.id
        if not hetflag.strip():
            continue
        residue_name = str(residue.resname).strip().upper()
        if residue_name in skip_residue_names:
            continue
        coords = atom_coordinates(list(residue.get_atoms()))
        heavy_atom_count = sum(
            1
            for atom in residue.get_atoms()
            if str(getattr(atom, "element", "")).strip().upper() not in {"", "H", "D"}
        )
        residues.append(
            HeteroResidue(
                chain_id=str(chain.id),
                residue_name=residue_name,
                residue_number=int(resseq),
                insertion_code=str(icode).strip(),
                atom_count=len(coords),
                heavy_atom_count=heavy_atom_count,
                centroid=centroid(coords),
                residue_id=(str(hetflag), int(resseq), str(icode).strip()),
            )
        )
    return residues


def choose_chain_ligand(
    pdb_path: Path,
    chain_id: str,
    preferred_residue_names: list[str] | None = None,
) -> HeteroResidue | None:
    preferred = [value.upper() for value in (preferred_residue_names or [])]
    candidates = hetero_residues_for_chain(
        pdb_path,
        chain_id,
        skip_residue_names=PDB_HETERO_SKIP_IDS,
    )
    if not candidates:
        return None
    if preferred:
        preferred_hits = [row for row in candidates if row.residue_name in preferred]
        if preferred_hits:
            candidates = preferred_hits
    return max(
        candidates,
        key=lambda row: (row.heavy_atom_count, row.atom_count, row.residue_name, row.residue_number),
        default=None,
    )


def save_selection(source_pdb: Path, select: Select, output_pdb: Path) -> None:
    structure = load_structure(source_pdb)
    io = PDBIO()
    io.set_structure(structure)
    ensure_dir(output_pdb.parent)
    io.save(str(output_pdb), select=select)


def save_chain_protein(source_pdb: Path, chain_id: str, output_pdb: Path) -> None:
    save_selection(source_pdb, ChainProteinSelect(chain_id), output_pdb)


def save_chain_set_protein(source_pdb: Path, chain_ids: Iterable[str], output_pdb: Path) -> None:
    save_selection(source_pdb, ChainSetProteinSelect(chain_ids), output_pdb)


def save_residue_subset(
    source_pdb: Path,
    chain_id: str,
    residue_ids: set[tuple[str, int, str]],
    output_pdb: Path,
) -> None:
    if not residue_ids:
        text_dump(output_pdb, "END\n")
        return
    save_selection(source_pdb, ResidueSetSelect(chain_id, residue_ids), output_pdb)


def residue_atom_coordinates(
    pdb_path: Path,
    chain_id: str,
    residue_ids: set[tuple[str, int, str]],
) -> dict[tuple[str, int, str], list[tuple[float, float, float]]]:
    structure = load_structure(pdb_path)
    model = first_model(structure)
    chain = model[str(chain_id)]
    payload: dict[tuple[str, int, str], list[tuple[float, float, float]]] = {}
    for residue in chain:
        residue_key = (
            str(residue.id[0]),
            int(residue.id[1]),
            str(residue.id[2]).strip(),
        )
        if residue_key in residue_ids:
            payload[residue_key] = atom_coordinates(list(residue.get_atoms()))
    return payload


def residue_number_lookup(
    residue_rows: list[dict[str, Any]],
    target_positions: list[int],
) -> set[tuple[str, int, str]]:
    target_set = {int(value) for value in target_positions}
    return {
        (" ", int(row["pdb_resnum"]), str(row.get("insertion_code") or ""))
        for row in residue_rows
        if row.get("uniprot_pos") is not None and int(row["uniprot_pos"]) in target_set
    }


def protein_chain_ids(pdb_path: Path) -> list[str]:
    structure = load_structure(pdb_path)
    model = first_model(structure)
    chain_ids: list[str] = []
    for chain in model:
        if any(residue.id[0] == " " for residue in chain):
            chain_ids.append(str(chain.id))
    return chain_ids


def extract_ligand_to_sdf(
    source_pdb: Path,
    chain_id: str,
    ligand: HeteroResidue,
    output_sdf: Path,
    work_dir: Path,
) -> Path:
    ligand_pdb = work_dir / f"{ligand.residue_name}_{ligand.residue_number}.pdb"
    save_residue_subset(source_pdb, chain_id, {ligand.residue_id}, ligand_pdb)
    run_command(
        [
            "obabel",
            str(ligand_pdb),
            "-O",
            str(output_sdf),
        ],
        cwd=work_dir,
    )
    return output_sdf


def crystal_ligand_from_template(
    source_pdb: Path,
    chain_id: str,
    ligand: HeteroResidue,
    reference_sdf: Path,
    output_sdf: Path,
    work_dir: Path,
) -> Path:
    ligand_pdb = work_dir / f"{ligand.residue_name}_{ligand.residue_number}.pdb"
    save_residue_subset(source_pdb, chain_id, {ligand.residue_id}, ligand_pdb)
    template = load_rdkit_molecule(reference_sdf, sanitize=True)
    pdb_molecule = Chem.MolFromPDBFile(str(ligand_pdb), sanitize=False, removeHs=False)
    if pdb_molecule is None:
        raise ValueError(f"Unable to parse crystal ligand PDB for {ligand.residue_name}")
    assigned = AllChem.AssignBondOrdersFromTemplate(
        Chem.RemoveHs(template),
        Chem.RemoveHs(pdb_molecule),
    )
    ensure_dir(output_sdf.parent)
    writer = Chem.SDWriter(str(output_sdf))
    writer.write(assigned)
    writer.close()
    return output_sdf


def load_rdkit_molecule(path: Path, sanitize: bool = True) -> Chem.Mol:
    suffix = path.suffix.lower()
    molecule = None
    if suffix == ".sdf":
        molecule = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=sanitize)
    elif suffix == ".pdb":
        molecule = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=sanitize)
    if molecule is None:
        raise ValueError(f"Unable to parse molecule from {path}")
    return molecule


def mol_coordinates(molecule: Chem.Mol) -> list[tuple[float, float, float]]:
    conformer = molecule.GetConformer()
    coords = []
    for atom_idx in range(molecule.GetNumAtoms()):
        position = conformer.GetAtomPosition(atom_idx)
        coords.append((float(position.x), float(position.y), float(position.z)))
    return coords


def ensure_ligand_3d(molecule: Chem.Mol) -> Chem.Mol:
    copied = Chem.Mol(molecule)
    if copied.GetNumConformers() <= 0:
        AllChem.EmbedMolecule(copied, AllChem.ETKDGv3())
    coords = mol_coordinates(copied)
    _, _, z_span = max_axis_span(coords)
    if z_span < 0.1:
        copied.RemoveAllConformers()
        status = AllChem.EmbedMolecule(copied, AllChem.ETKDGv3())
        if status != 0:
            raise ValueError("RDKit failed to generate 3D coordinates")
        try:
            AllChem.UFFOptimizeMolecule(copied, maxIters=500)
        except Exception:
            pass
    return copied


def standardize_reference_ligand(input_sdf: Path, output_sdf: Path) -> dict[str, Any]:
    sanitized = True
    try:
        with suppress_rdkit_logs():
            molecule = load_rdkit_molecule(input_sdf, sanitize=True)
    except ValueError:
        # Some public ligands carry valence issues that still preserve usable 3D coordinates.
        molecule = load_rdkit_molecule(input_sdf, sanitize=False)
        sanitized = False
    with suppress_noisy_stderr(), suppress_rdkit_logs():
        try:
            from rdkit.Chem.MolStandardize import rdMolStandardize

            largest = rdMolStandardize.LargestFragmentChooser()
            molecule = largest.choose(molecule)
            molecule = rdMolStandardize.Uncharger().uncharge(molecule)
        except Exception:
            pass
    molecule = ensure_ligand_3d(molecule)
    canonical_smiles = ""
    try:
        canonical_smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(molecule)), canonical=True)
    except Exception:
        if not sanitized:
            try:
                molecule.UpdatePropertyCache(strict=False)
            except Exception:
                pass
    ensure_dir(output_sdf.parent)
    writer = Chem.SDWriter(str(output_sdf))
    writer.write(molecule)
    writer.close()
    return {
        "path": str(output_sdf),
        "heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
        "canonical_smiles": canonical_smiles,
    }


def build_water_retention_table(
    source_pdb: Path,
    chain_id: str,
    ligand_coords: list[tuple[float, float, float]],
    distance_cutoff_a: float,
    max_bfactor: float,
) -> tuple[pd.DataFrame, set[tuple[str, int, str]]]:
    structure = load_structure(source_pdb)
    model = first_model(structure)
    chain = model[str(chain_id)]
    rows: list[dict[str, Any]] = []
    retained: set[tuple[str, int, str]] = set()
    for residue in chain:
        residue_name = str(residue.resname).strip().upper()
        if residue_name not in {"HOH", "DOD"}:
            continue
        residue_coords = atom_coordinates(list(residue.get_atoms()))
        min_distance = min_distance_between_sets(residue_coords, ligand_coords)
        mean_bfactor = float(sum(atom.get_bfactor() for atom in residue.get_atoms()) / max(1, len(residue_coords)))
        retained_flag = (
            min_distance is not None
            and float(min_distance) <= float(distance_cutoff_a)
            and mean_bfactor <= float(max_bfactor)
        )
        residue_key = (str(residue.id[0]), int(residue.id[1]), str(residue.id[2]).strip())
        if retained_flag:
            retained.add(residue_key)
        rows.append(
            {
                "chain_id": chain_id,
                "residue_name": residue_name,
                "residue_number": int(residue.id[1]),
                "insertion_code": str(residue.id[2]).strip(),
                "min_distance_to_ligand_a": None if min_distance is None else float(min_distance),
                "mean_bfactor": mean_bfactor,
                "retained": bool(retained_flag),
                "retention_reason": (
                    "bridge_water"
                    if retained_flag
                    else (
                        "distance_gt_cutoff"
                        if min_distance is None or float(min_distance) > float(distance_cutoff_a)
                        else "bfactor_gt_cutoff"
                    )
                ),
            }
        )
    return pd.DataFrame.from_records(rows), retained


def build_nearby_cofactor_set(
    source_pdb: Path,
    chain_id: str,
    ligand_coords: list[tuple[float, float, float]],
    distance_cutoff_a: float,
    excluded_residue_ids: set[tuple[str, int, str]] | None = None,
) -> set[tuple[str, int, str]]:
    structure = load_structure(source_pdb)
    model = first_model(structure)
    chain = model[str(chain_id)]
    excluded_residue_ids = excluded_residue_ids or set()
    retained: set[tuple[str, int, str]] = set()
    for residue in chain:
        hetflag, resseq, icode = residue.id
        residue_name = str(residue.resname).strip().upper()
        if not hetflag.strip():
            continue
        residue_key = (str(hetflag), int(resseq), str(icode).strip())
        if residue_key in excluded_residue_ids:
            continue
        if residue_name in PDB_HETERO_SKIP_IDS or residue_name in {"HOH", "DOD"}:
            continue
        residue_coords = atom_coordinates(list(residue.get_atoms()))
        min_distance = min_distance_between_sets(residue_coords, ligand_coords)
        elements = {str(getattr(atom, "element", "")).strip().upper() for atom in residue.get_atoms()}
        keep = False
        if min_distance is not None and float(min_distance) <= float(distance_cutoff_a):
            keep = True
        if elements & METAL_ELEMENTS and min_distance is not None and float(min_distance) <= float(distance_cutoff_a) + 1.5:
            keep = True
        if keep:
            retained.add(residue_key)
    return retained


def nearby_protein_residue_ids(
    source_pdb: Path,
    chain_id: str,
    ligand_coords: list[tuple[float, float, float]],
    distance_cutoff_a: float,
) -> set[tuple[str, int, str]]:
    structure = load_structure(source_pdb)
    model = first_model(structure)
    chain = model[str(chain_id)]
    retained: set[tuple[str, int, str]] = set()
    for residue in chain:
        hetflag, resseq, icode = residue.id
        if hetflag.strip():
            continue
        residue_coords = atom_coordinates(list(residue.get_atoms()))
        min_distance = min_distance_between_sets(residue_coords, ligand_coords)
        if min_distance is not None and float(min_distance) <= float(distance_cutoff_a):
            retained.add((" ", int(resseq), str(icode).strip()))
    return retained


def nearby_protein_residue_labels(
    source_pdb: Path,
    chain_ids: Iterable[str],
    ligand_coords: list[tuple[float, float, float]],
    distance_cutoff_a: float,
) -> set[str]:
    structure = load_structure(source_pdb)
    model = first_model(structure)
    retained: set[str] = set()
    for raw_chain_id in chain_ids:
        chain_id = str(raw_chain_id)
        try:
            chain = model[chain_id]
        except KeyError:
            continue
        for residue in chain:
            hetflag, resseq, icode = residue.id
            if str(hetflag).strip():
                continue
            residue_coords = atom_coordinates(list(residue.get_atoms()))
            min_distance = min_distance_between_sets(residue_coords, ligand_coords)
            if min_distance is None or float(min_distance) > float(distance_cutoff_a):
                continue
            insertion_code = str(icode).strip()
            residue_number = f"{int(resseq)}{insertion_code}" if insertion_code else f"{int(resseq)}"
            retained.add(f"{chain_id}:{str(residue.resname).strip().upper()}{residue_number}")
    return retained


def merge_pdb_fragments(input_paths: list[Path], output_pdb: Path) -> None:
    lines: list[str] = []
    for path in input_paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(("ATOM", "HETATM", "TER")):
                lines.append(line)
    lines.append("END")
    text_dump(output_pdb, "\n".join(lines) + "\n")


def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def run_pdbfixer(input_pdb: Path, output_pdb: Path, ph: float) -> None:
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    fixer = PDBFixer(filename=str(input_pdb))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(float(ph))
    ensure_dir(output_pdb.parent)
    with output_pdb.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)


def prepare_receptor_pdbqt(input_pdb: Path, output_pdbqt: Path, ph: float) -> None:
    raw_output = output_pdbqt.with_suffix(".raw.pdbqt")
    run_command(
        [
            "obabel",
            str(input_pdb),
            "-O",
            str(raw_output),
            "-p",
            str(ph),
        ]
    )
    lines = raw_output.read_text(encoding="utf-8", errors="ignore").splitlines()
    filtered = [line for line in lines if line.startswith(("ATOM", "HETATM", "TER"))]
    if not filtered:
        raise RuntimeError(f"Rigid receptor conversion produced no ATOM/HETATM records for {input_pdb}")
    text_dump(output_pdbqt, "\n".join(filtered) + "\n")


def prepare_ligand_pdbqt(input_sdf: Path, output_pdbqt: Path, ph: float) -> None:
    run_command(
        [
            "obabel",
            str(input_sdf),
            "-O",
            str(output_pdbqt),
            "-p",
            str(ph),
        ]
    )


def vina_result_rows(pdbqt_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mode_rank = 0
    for line in pdbqt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("REMARK VINA RESULT:"):
            mode_rank += 1
            parts = line.split()
            rows.append(
                {
                    "mode_rank": mode_rank,
                    "affinity_kcal_mol": float(parts[3]),
                    "rmsd_lb": float(parts[4]),
                    "rmsd_ub": float(parts[5]),
                }
            )
    return rows


def split_vina_poses(vina_pdbqt: Path, output_dir: Path) -> list[Path]:
    ensure_dir(output_dir)
    stem = output_dir / "pose.sdf"
    run_command(
        [
            "obabel",
            str(vina_pdbqt),
            "-O",
            str(stem),
            "-m",
        ],
        cwd=output_dir,
    )
    poses = sorted(output_dir.glob("pose*.sdf"))
    if not poses:
        raise RuntimeError(f"No SDF poses were generated from {vina_pdbqt}")
    return poses


def _unique_atom_names(molecule: Chem.Mol) -> list[str]:
    counter: Counter[str] = Counter()
    names: list[str] = []
    for atom in molecule.GetAtoms():
        symbol = str(atom.GetSymbol())
        counter[symbol] += 1
        names.append(f"{symbol}{counter[symbol]}")
    return names


def pose_pdb_from_sdf(pose_sdf: Path, output_pdb: Path) -> None:
    molecule = load_rdkit_molecule(pose_sdf, sanitize=False)
    atom_names = _unique_atom_names(molecule)
    for atom, atom_name in zip(molecule.GetAtoms(), atom_names):
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName("UNK")
        info.SetChainId("Z")
        info.SetResidueNumber(1)
        info.SetName(f"{atom_name[:4]:>4}")
        atom.SetMonomerInfo(info)
    text_dump(output_pdb, Chem.MolToPDBBlock(molecule))


def ligand_pose_pdb(
    molecule: Chem.Mol,
    output_pdb: Path,
    residue_name: str,
    chain_id: str,
    residue_number: int,
) -> None:
    mol = Chem.Mol(molecule)
    atom_names = _unique_atom_names(mol)
    for atom, atom_name in zip(mol.GetAtoms(), atom_names):
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName(str(residue_name)[:3].rjust(3))
        info.SetChainId(str(chain_id)[:1])
        info.SetResidueNumber(int(residue_number))
        info.SetName(f"{atom_name[:4]:>4}")
        atom.SetMonomerInfo(info)
    text_dump(output_pdb, Chem.MolToPDBBlock(mol))


def best_rmsd(reference_sdf: Path, pose_sdf: Path) -> float | None:
    reference = Chem.RemoveHs(load_rdkit_molecule(reference_sdf, sanitize=True))
    pose = Chem.RemoveHs(load_rdkit_molecule(pose_sdf, sanitize=True))
    if reference.GetNumAtoms() != pose.GetNumAtoms():
        return None
    try:
        return float(rdMolAlign.GetBestRMS(reference, pose))
    except Exception:
        return None


def obrms_rmsd(reference_path: Path, test_path: Path) -> float | None:
    result = subprocess.run(
        ["obrms", "-f", str(reference_path), str(test_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    tokens = (result.stdout or "").replace(",", " ").split()
    for token in reversed(tokens):
        try:
            return float(token)
        except ValueError:
            continue
    return None


def plip_ifp(complex_pdb: Path) -> dict[str, Any]:
    from plip.structure.preparation import PDBComplex

    mol = PDBComplex()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        mol.load_pdb(str(complex_pdb))
    ligands = [
        ligand
        for ligand in mol.ligands
        if str(getattr(ligand, "hetid", "")).strip().upper() not in PDB_HETERO_SKIP_IDS
    ]
    if not ligands:
        raise RuntimeError(f"PLIP found no eligible ligand in {complex_pdb}")
    ligand = max(
        ligands,
        key=lambda item: len(getattr(item, "can_to_pdb", {}) or {}),
    )
    mol.characterize_complex(ligand)
    interaction_key = next(iter(mol.interaction_sets))
    interaction_set = mol.interaction_sets[interaction_key]

    interactions: list[dict[str, Any]] = []

    def add_interaction(kind: str, collection: Iterable[Any], distance_attr: str | None = None) -> None:
        for item in collection:
            residue_label = f"{getattr(item, 'reschain', '?')}:{getattr(item, 'restype', 'UNK')}{getattr(item, 'resnr', '?')}"
            record = {
                "interaction_type": kind,
                "residue_label": residue_label,
                "chain_id": getattr(item, "reschain", None),
                "residue_name": getattr(item, "restype", None),
                "residue_number": getattr(item, "resnr", None),
                "distance_a": None,
            }
            if distance_attr is not None and hasattr(item, distance_attr):
                value = getattr(item, distance_attr)
                if value is not None:
                    record["distance_a"] = float(value)
            interactions.append(record)

    add_interaction("hydrogen_bond", interaction_set.hbonds_ldon, "distance_ad")
    add_interaction("hydrogen_bond", interaction_set.hbonds_pdon, "distance_ad")
    add_interaction("salt_bridge", interaction_set.saltbridge_lneg, None)
    add_interaction("salt_bridge", interaction_set.saltbridge_pneg, None)
    add_interaction("hydrophobic", interaction_set.hydrophobic_contacts, "distance")
    add_interaction("pi_stacking", interaction_set.pistacking, "distance")
    add_interaction("pi_cation", interaction_set.pication_laro, "distance")
    add_interaction("pi_cation", interaction_set.pication_paro, "distance")
    add_interaction("metal_complex", interaction_set.metal_complexes, "distance")

    residue_counter = Counter(row["residue_label"] for row in interactions)
    type_counter = Counter(row["interaction_type"] for row in interactions)
    return {
        "ligand_key": interaction_key,
        "interactions": interactions,
        "residue_set": sorted(residue_counter),
        "residue_counts": dict(sorted(residue_counter.items())),
        "interaction_type_counts": {key: int(type_counter.get(key, 0)) for key in PLIP_INTERACTION_TYPES},
    }


def write_ifp_json(path: Path, payload: dict[str, Any]) -> None:
    json_dump(path, payload)


def write_anchor_file(path: Path, residues: list[str]) -> None:
    text_dump(path, "\n".join(residues) + ("\n" if residues else ""))


def build_box_from_ligand_coords(
    ligand_coords: list[tuple[float, float, float]],
    default_box_size_a: float,
    ligand_padding_a: float,
    source: str,
) -> dict[str, Any]:
    size_x, size_y, size_z = max_axis_span(ligand_coords)
    center_x, center_y, center_z = centroid(ligand_coords)
    return {
        "source": source,
        "center_x": float(center_x),
        "center_y": float(center_y),
        "center_z": float(center_z),
        "size_x": float(max(default_box_size_a, size_x + ligand_padding_a)),
        "size_y": float(max(default_box_size_a, size_y + ligand_padding_a)),
        "size_z": float(max(default_box_size_a, size_z + ligand_padding_a)),
    }


def build_box_from_residue_coords(
    residue_coords: list[tuple[float, float, float]],
    reference_ligand_coords: list[tuple[float, float, float]],
    default_box_size_a: float,
    ligand_padding_a: float,
    source: str,
) -> dict[str, Any]:
    center_x, center_y, center_z = centroid(residue_coords)
    size_x, size_y, size_z = max_axis_span(reference_ligand_coords)
    return {
        "source": source,
        "center_x": float(center_x),
        "center_y": float(center_y),
        "center_z": float(center_z),
        "size_x": float(max(default_box_size_a, size_x + ligand_padding_a)),
        "size_y": float(max(default_box_size_a, size_y + ligand_padding_a)),
        "size_z": float(max(default_box_size_a, size_z + ligand_padding_a)),
    }


def fpocket_top_pocket_coords(receptor_pdb: Path) -> list[tuple[float, float, float]]:
    with tempfile.TemporaryDirectory(prefix="fpocket_stage35_") as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        work_pdb = temp_dir / receptor_pdb.name
        shutil.copyfile(receptor_pdb, work_pdb)
        run_command(["fpocket", "-f", str(work_pdb)], cwd=temp_dir)
        pockets = sorted((temp_dir / f"{work_pdb.stem}_out" / "pockets").glob("pocket*_atm.pdb"))
        if not pockets:
            raise RuntimeError(f"fpocket did not produce pockets for {receptor_pdb}")
        return atom_coordinates(list(load_structure(pockets[0]).get_atoms()))


def fpocket_box(
    receptor_pdb: Path,
    default_box_size_a: float,
    reference_ligand_coords: list[tuple[float, float, float]],
) -> dict[str, Any]:
    coords = fpocket_top_pocket_coords(receptor_pdb)
    return build_box_from_residue_coords(
        coords,
        reference_ligand_coords=reference_ligand_coords,
        default_box_size_a=default_box_size_a,
        ligand_padding_a=8.0,
        source="fpocket_top1",
    )


def run_vina_redocking(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    docking_box: dict[str, Any],
    output_root: Path,
    seeds: list[int],
    exhaustiveness: int,
    num_modes: int,
    energy_range: int,
    cpu_threads: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ensure_dir(output_root)
    for seed in seeds:
        seed_dir = ensure_dir(output_root / f"seed_{seed}")
        out_pdbqt = seed_dir / "vina_out.pdbqt"
        command = [
            "vina",
            "--receptor",
            str(receptor_pdbqt),
            "--ligand",
            str(ligand_pdbqt),
            "--center_x",
            str(docking_box["center_x"]),
            "--center_y",
            str(docking_box["center_y"]),
            "--center_z",
            str(docking_box["center_z"]),
            "--size_x",
            str(docking_box["size_x"]),
            "--size_y",
            str(docking_box["size_y"]),
            "--size_z",
            str(docking_box["size_z"]),
            "--exhaustiveness",
            str(exhaustiveness),
            "--num_modes",
            str(num_modes),
            "--energy_range",
            str(energy_range),
            "--cpu",
            str(cpu_threads),
            "--seed",
            str(seed),
            "--out",
            str(out_pdbqt),
        ]
        result = run_command(command, cwd=seed_dir)
        pose_sd_files = split_vina_poses(out_pdbqt, seed_dir)
        vina_rows = vina_result_rows(out_pdbqt)
        for pose_row, pose_sdf in zip(vina_rows, pose_sd_files):
            rows.append(
                {
                    **pose_row,
                    "seed": int(seed),
                    "pose_sdf": str(pose_sdf),
                    "vina_stdout": result.stdout,
                }
            )
    rows.sort(key=lambda row: (int(row["seed"]), int(row["mode_rank"])))
    return rows


def top_pose_per_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    top_rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in sorted(rows, key=lambda item: (int(item["seed"]), int(item["mode_rank"]))):
        seed = int(row["seed"])
        if seed in seen:
            continue
        seen.add(seed)
        top_rows.append(row)
    return top_rows


def ifp_frequency(top_pose_ifps: list[dict[str, Any]]) -> dict[str, float]:
    counter = Counter()
    for item in top_pose_ifps:
        for residue in item.get("residue_set", []):
            counter[str(residue)] += 1
    denominator = max(1, len(top_pose_ifps))
    return {key: float(value / denominator) for key, value in sorted(counter.items())}


def anchor_residues(
    baseline_ifp: dict[str, Any],
    redocking_ifps: list[dict[str, Any]],
    threshold: float,
    require_intersection: bool,
) -> tuple[list[str], str, dict[str, float]]:
    frequency_map = ifp_frequency(redocking_ifps)
    stable = {residue for residue, value in frequency_map.items() if float(value) > float(threshold)}
    baseline = set(str(value) for value in baseline_ifp.get("residue_set", []))
    if require_intersection:
        intersection = sorted(baseline & stable)
        if intersection:
            return intersection, "crystal_redocking_intersection", frequency_map
        return sorted(stable), "redocking_high_confidence_fallback", frequency_map
    return sorted(stable), "redocking_high_confidence_only", frequency_map


def pocket_residue_metrics(
    pose_coords: list[tuple[float, float, float]],
    residue_coord_map: dict[str, list[tuple[float, float, float]]],
    contact_cutoff_a: float,
) -> dict[str, Any]:
    min_distance = None
    covered = 0
    for coords in residue_coord_map.values():
        current = min_distance_between_sets(pose_coords, coords)
        if current is None:
            continue
        if min_distance is None or float(current) < float(min_distance):
            min_distance = float(current)
        if float(current) <= float(contact_cutoff_a):
            covered += 1
    residue_count = len(residue_coord_map)
    return {
        "min_distance_a": None if min_distance is None else float(min_distance),
        "coverage_count": int(covered),
        "coverage_fraction": float(covered / residue_count) if residue_count else 0.0,
        "residue_count": int(residue_count),
    }


def classify_hiv_pose(
    pose_sdf: Path,
    nnrti_residue_coords: dict[str, list[tuple[float, float, float]]],
    active_site_residue_coords: dict[str, list[tuple[float, float, float]]],
    contact_cutoff_a: float,
) -> dict[str, Any]:
    pose = load_rdkit_molecule(pose_sdf, sanitize=True)
    pose_coords = mol_coordinates(pose)
    nnrti = pocket_residue_metrics(pose_coords, nnrti_residue_coords, contact_cutoff_a)
    active = pocket_residue_metrics(pose_coords, active_site_residue_coords, contact_cutoff_a)
    label = "other"
    if nnrti["coverage_count"] > 0 and nnrti["coverage_fraction"] >= active["coverage_fraction"]:
        label = "NNRTI_pocket"
    if active["coverage_count"] > 0 and (
        active["coverage_fraction"] > nnrti["coverage_fraction"]
        or (
            nnrti["min_distance_a"] is not None
            and active["min_distance_a"] is not None
            and float(active["min_distance_a"]) + 0.5 < float(nnrti["min_distance_a"])
        )
    ):
        label = "active_site"
    return {
        "pose_label": label,
        "nnrti_min_distance_a": nnrti["min_distance_a"],
        "nnrti_coverage_count": nnrti["coverage_count"],
        "nnrti_coverage_fraction": nnrti["coverage_fraction"],
        "active_site_min_distance_a": active["min_distance_a"],
        "active_site_coverage_count": active["coverage_count"],
        "active_site_coverage_fraction": active["coverage_fraction"],
    }


def select_best_pose(rows: list[dict[str, Any]], hiv_mode: bool) -> dict[str, Any] | None:
    if not rows:
        return None
    candidates = rows
    if hiv_mode:
        candidates = [row for row in rows if row.get("pose_label") == "NNRTI_pocket"]
        if not candidates:
            return None
    return min(
        candidates,
        key=lambda row: (
            float(row.get("affinity_kcal_mol", 1.0e9)),
            int(row.get("seed", 0)),
            int(row.get("mode_rank", 0)),
        ),
    )


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def docking_summary_json(path: Path, payload: dict[str, Any]) -> None:
    json_dump(path, payload)
