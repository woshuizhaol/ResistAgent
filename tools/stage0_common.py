#!/usr/bin/env python3
"""Stage 0 helpers for auditing archives, aliases, and filesystem layout."""

from __future__ import annotations

import gzip
import os
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Iterator

ARCHIVE_EXTENSIONS = (".tar", ".tar.gz", ".tgz", ".zip", ".gz")
SAMPLE_ID_PATTERN = re.compile(r"(MdrDB\d{6})")


def is_ignored_path(path: Path, ignore_prefixes: list[str]) -> bool:
    return any(part.startswith(prefix) for part in path.parts for prefix in ignore_prefixes)


def relative_format(path: Path) -> str:
    suffixes = [suffix.lstrip(".").lower() for suffix in path.suffixes]
    if not suffixes:
        return "none"
    if suffixes[-1] in {"gz", "bz2", "xz"} and len(suffixes) >= 2:
        return f"{suffixes[-2]}.{suffixes[-1]}"
    if suffixes[-1] == "zip" and len(suffixes) >= 2 and suffixes[-2] in {"tsv", "csv", "txt"}:
        return f"{suffixes[-2]}.zip"
    if suffixes[-1] == "tar":
        return "tar"
    if len(suffixes) >= 2 and suffixes[-2:] == ["tar", "gz"]:
        return "tar.gz"
    return suffixes[-1]


def detect_archive_type(path: Path) -> tuple[str | None, str | None]:
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic[:2] == b"PK":
        return "zip", "unzip -q"
    if magic[:2] == b"\x1f\x8b":
        if tarfile.is_tarfile(path):
            return "tar.gz", "tar -xzf"
        return "gzip", "gzip -dc"
    if tarfile.is_tarfile(path):
        return "tar", "tar -xf"
    return None, None


def infer_sample_id(member_path: str) -> str | None:
    match = SAMPLE_ID_PATTERN.search(member_path)
    return match.group(1) if match else None


def classify_file_role(member_path: str) -> str:
    name = Path(member_path).name.lower()
    if name.endswith("_lig.sdf") or name.endswith(".sdf"):
        return "ligand"
    if name.startswith("wt_") and name.endswith("_complex.pdb"):
        return "WT_complex"
    if name.startswith("mt_") and name.endswith("_complex.pdb"):
        return "MT_complex"
    if name.startswith("wt_") and name.endswith(".pdb"):
        return "WT"
    if name.startswith("mt_") and name.endswith(".pdb"):
        return "MT"
    return "other"


def canonical_alias_entries(root: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str | bool]] = []
    candidates = [
        (
            root / "data/MdrDB/multiple-substitution",
            "Multiple_Substitution.tar.gz",
            "MdrDB multiple-substitution is a gzip-compressed tar stream with no extension",
        ),
        (
            root / "data/MdrDB/unpacked/MepMap_drug_cell_line_sensitivity.tsv",
            "DepMap_drug_cell_line_sensitivity.tsv",
            "Upstream unpacked typo should resolve to the DepMap semantic alias",
        ),
        (
            root / "data/MdrDB/unpacked/MirDB_drug_annotation_v1.0.2022.tsv",
            "MdrDB_drug_annotation_v1.0.2022.tsv",
            "Upstream unpacked typo should resolve to the MdrDB semantic alias",
        ),
        (
            root / "data/MdrDB/unpacked/MirDB_protein_annotation_v1.0.2022.tsv",
            "MdrDB_protein_annotation_v1.0.2022.tsv",
            "Upstream unpacked typo should resolve to the MdrDB semantic alias",
        ),
        (
            root / "data/external/sifts/pdbe_mappings/1m17.json",
            "1M17.json",
            "Canonicalize PDB identifiers to uppercase for downstream manifests",
        ),
        (
            root / "data/external/sifts/pdbe_mappings/1iep.json",
            "1IEP.json",
            "Canonicalize PDB identifiers to uppercase for downstream manifests",
        ),
        (
            root / "data/external/sifts/pdbe_mappings/1rtd.json",
            "1RTD.json",
            "Canonicalize PDB identifiers to uppercase for downstream manifests",
        ),
        (
            root / "data/external/sifts/pdbe_mappings/2zd1.json",
            "2ZD1.json",
            "Canonicalize PDB identifiers to uppercase for downstream manifests",
        ),
        (
            root / "data/external/sifts/pdbe_mappings/7tll.json",
            "7TLL.json",
            "Canonicalize PDB identifiers to uppercase for downstream manifests",
        ),
    ]
    for path, canonical_name, reason in candidates:
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "canonical_name": canonical_name,
                "reason": reason,
                "path_exists": path.exists(),
            }
        )
    return entries


def iter_archive_members(path: Path, archive_type: str) -> Iterator[tuple[str, int]]:
    if archive_type in {"tar", "tar.gz"}:
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                if member.isfile():
                    yield member.name, member.size
        return
    if archive_type == "zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if not member.is_dir():
                    yield member.filename, member.file_size
        return
    if archive_type == "gzip":
        inner_name = path.stem
        with gzip.open(path, "rb") as stream:
            size = 0
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
        yield inner_name, size
        return
    raise ValueError(f"Unsupported archive type: {archive_type}")


def candidate_archive(path: Path) -> bool:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if path.name == "multiple-substitution":
        return True
    if suffixes and suffixes[-1] in {".tar", ".zip", ".gz"}:
        return True
    return False
