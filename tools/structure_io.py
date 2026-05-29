#!/usr/bin/env python3
"""Lightweight structure and ligand IO helpers for Stage 3 and later."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Bio.PDB import PDBParser
from rdkit import Chem


def validate_pdb_file(path: Path) -> dict[str, Any]:
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("stage3", str(path))
    except Exception as exc:  # pragma: no cover - parser exceptions vary by file
        return {
            "read_ok": False,
            "chain_count": 0,
            "residue_count": 0,
            "atom_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }

    chain_count = 0
    residue_count = 0
    atom_count = 0
    for model in structure:
        for chain in model:
            observed_residues = []
            for residue in chain:
                if residue.id[0].strip():
                    continue
                observed_residues.append(residue)
            if observed_residues:
                chain_count += 1
                residue_count += len(observed_residues)
                atom_count += sum(1 for residue in observed_residues for _ in residue.get_atoms())
        break

    return {
        "read_ok": bool(chain_count > 0 and residue_count > 0 and atom_count > 0),
        "chain_count": int(chain_count),
        "residue_count": int(residue_count),
        "atom_count": int(atom_count),
        "error": None if chain_count > 0 else "no_protein_residues_detected",
    }


def validate_sdf_file(path: Path) -> dict[str, Any]:
    molecule = None
    error = None
    try:
        molecule = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=False)
    except Exception as exc:  # pragma: no cover - rdkit exceptions vary by file
        error = f"{type(exc).__name__}: {exc}"

    if molecule is None:
        try:
            supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
            molecule = next((mol for mol in supplier if mol is not None), None)
        except Exception as exc:  # pragma: no cover - rdkit exceptions vary by file
            error = f"{type(exc).__name__}: {exc}"

    if molecule is None:
        return {
            "read_ok": False,
            "atom_count": 0,
            "bond_count": 0,
            "conformer_count": 0,
            "error": error or "rdkit_returned_none",
        }

    return {
        "read_ok": bool(molecule.GetNumAtoms() > 0 and molecule.GetNumConformers() > 0),
        "atom_count": int(molecule.GetNumAtoms()),
        "bond_count": int(molecule.GetNumBonds()),
        "conformer_count": int(molecule.GetNumConformers()),
        "error": None,
    }
