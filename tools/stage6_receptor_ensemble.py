#!/usr/bin/env python3
"""counter-design step receptor-ensemble helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from tools.public_data_utils import fetch_rcsb_files, request_session
from tools.runtime import ensure_dir, iso_now, json_dump
from tools.stage35_utils import save_chain_protein
from tools.stage5_utils import first_protein_chain_id, sample_root_for_target, sample_root_ready, stable_target_slug, target_components

AA1_TO_3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _seed_rank_map(case_entry: dict[str, Any]) -> dict[str, int]:
    return {
        str(row.get("pdb_id") or "").upper(): int(row.get("seed_rank") or 9999)
        for row in list(case_entry.get("seed_pdb_candidates") or [])
        if str(row.get("pdb_id") or "")
    }


def _target_key_column(effect_scope: str) -> str:
    return "combination_key" if str(effect_scope) == "combo" else "mutation_key"


def _ensemble_member_ready(sample_root: Path) -> bool:
    return (
        sample_root.exists()
        and (sample_root / "WT.pdb").exists()
        and (sample_root / "MT.pdb").exists()
        and (sample_root / "ligand.sdf").exists()
    )


def _candidate_member_rows(
    *,
    root: Path,
    case_id: str,
    effect_scope: str,
    target_key: str,
    mutation_pool: pd.DataFrame,
    seed_rank_map: dict[str, int],
) -> list[dict[str, Any]]:
    if mutation_pool.empty:
        return []
    key_column = _target_key_column(effect_scope)
    if key_column not in mutation_pool.columns:
        return []
    subset = mutation_pool[
        mutation_pool.get("case_id", pd.Series(dtype=str)).astype(str).eq(str(case_id))
        & mutation_pool.get("effect_scope", pd.Series(dtype=str)).astype(str).eq(str(effect_scope))
        & mutation_pool[key_column].astype(str).eq(str(target_key))
    ].copy()
    if subset.empty:
        return []
    candidate_rows: list[dict[str, Any]] = []
    for row in subset.to_dict(orient="records"):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            continue
        sample_root, sample_source = sample_root_for_target(root, case_id, sample_id)
        if not sample_root_ready(sample_root):
            continue
        pdb_id = str(row.get("pdb_id") or "").upper()
        candidate_rows.append(
            {
                "sample_id": sample_id,
                "sample_root": str(sample_root),
                "sample_source": str(sample_source),
                "pdb_id": pdb_id,
                "seed_rank": int(seed_rank_map.get(pdb_id, 9999)),
                "selection_bucket": str(row.get("selection_bucket") or ""),
                "risk_score": float(row.get("risk_score") or row.get("risk_calibrated") or 0.0),
            }
        )
    candidate_rows.sort(
        key=lambda row: (
            int(row["seed_rank"]),
            -float(row["risk_score"]),
            str(row["sample_id"]),
        )
    )
    return candidate_rows


def _mutation_specs(target_key: str) -> list[str]:
    specs: list[str] = []
    for component in target_components(target_key):
        if component.mutation_class != "single_substitution" or component.start_pos is None or not component.alt_aa or not component.ref_aa:
            return []
        ref_aa3 = AA1_TO_3.get(str(component.ref_aa).upper())
        alt_aa3 = AA1_TO_3.get(str(component.alt_aa).upper())
        if ref_aa3 is None or alt_aa3 is None:
            return []
        specs.append(f"{ref_aa3}-{int(component.start_pos)}-{alt_aa3}")
    return specs


def materialize_modeled_member(
    *,
    root: Path,
    case_entry: dict[str, Any],
    effect_scope: str,
    target_key: str,
    pdb_id: str,
) -> Path | None:
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    if str(effect_scope) != "site":
        return None
    mutation_specs = _mutation_specs(target_key)
    if not mutation_specs:
        return None
    case_id = str(case_entry["case_id"])
    member_root = ensure_dir(
        root
        / "outputs"
        / case_id
        / "stage6_receptor_ensemble"
        / "modeled_members"
        / stable_target_slug(effect_scope, target_key)
        / str(pdb_id).upper()
    )
    if _ensemble_member_ready(member_root):
        return member_root
    cache_root = ensure_dir(root / "outputs" / "stage1_5" / "rcsb_cache")
    session = request_session()
    files = fetch_rcsb_files(session, str(pdb_id), cache_root, timeout=120)
    raw_pdb = Path(files["pdb"])
    chain_id = first_protein_chain_id(raw_pdb)
    wt_pdb = member_root / "WT.pdb"
    mt_pdb = member_root / "MT.pdb"
    ligand_sdf = member_root / "ligand.sdf"
    save_chain_protein(raw_pdb, chain_id, wt_pdb)
    shutil.copyfile(root / "outputs" / case_id / "stage1_5" / "raw" / "ligand.sdf", ligand_sdf)
    try:
        fixer = PDBFixer(filename=str(wt_pdb))
        fixer.applyMutations(mutation_specs, str(chain_id))
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        with mt_pdb.open("w", encoding="utf-8") as handle:
            PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    except Exception:
        return None
    json_dump(
        member_root / "member_manifest.json",
        {
            "case_id": case_id,
            "effect_scope": effect_scope,
            "target_key": target_key,
            "pdb_id": str(pdb_id).upper(),
            "chain_id": str(chain_id),
            "mutation_specs": mutation_specs,
            "generated_at": iso_now(),
        },
    )
    return member_root if _ensemble_member_ready(member_root) else None


def build_receptor_ensemble_members(
    *,
    root: Path,
    case_entry: dict[str, Any],
    panel_frame: pd.DataFrame,
    stage6: dict[str, Any],
    mutation_pool: pd.DataFrame,
) -> pd.DataFrame:
    if panel_frame.empty:
        return pd.DataFrame()
    max_members = max(1, int(stage6.get("receptor_ensemble_max_members", 4)))
    min_members = max(1, int(stage6.get("receptor_ensemble_min_members", 2)))
    case_id = str(case_entry["case_id"])
    seed_rank_map = _seed_rank_map(case_entry)
    rows: list[dict[str, Any]] = []
    for panel_row in panel_frame.to_dict(orient="records"):
        effect_scope = str(panel_row.get("effect_scope") or "")
        target_key = str(panel_row.get("target_key") or "")
        seen_roots: set[str] = set()
        seen_pdbs: set[str] = set()
        members: list[dict[str, Any]] = []

        base_sample_root = str(panel_row.get("sample_root") or "")
        base_sample_id = str(panel_row.get("representative_sample_id") or "")
        base_pdb_id = ""
        for candidate in _candidate_member_rows(
            root=root,
            case_id=case_id,
            effect_scope=effect_scope,
            target_key=target_key,
            mutation_pool=mutation_pool,
            seed_rank_map=seed_rank_map,
        ):
            sample_root = str(candidate["sample_root"])
            pdb_id = str(candidate["pdb_id"] or "")
            if sample_root in seen_roots or (pdb_id and pdb_id in seen_pdbs):
                continue
            members.append(
                {
                    "case_id": case_id,
                    "effect_scope": effect_scope,
                    "target_key": target_key,
                    "member_rank": int(len(members) + 1),
                    "member_id": str(candidate["sample_id"]),
                    "sample_id": str(candidate["sample_id"]),
                    "sample_root": sample_root,
                    "sample_source": str(candidate["sample_source"]),
                    "pdb_id": pdb_id,
                    "seed_rank": int(candidate["seed_rank"]),
                    "selection_bucket": str(candidate["selection_bucket"]),
                    "is_panel_representative": bool(base_sample_id and str(candidate["sample_id"]) == base_sample_id),
                }
            )
            seen_roots.add(sample_root)
            if pdb_id:
                seen_pdbs.add(pdb_id)
                if not base_pdb_id and base_sample_id and str(candidate["sample_id"]) == base_sample_id:
                    base_pdb_id = pdb_id
            if len(members) >= max_members:
                break

        if base_sample_root and sample_root_ready(Path(base_sample_root)) and base_sample_root not in seen_roots:
            members.insert(
                0,
                {
                    "case_id": case_id,
                    "effect_scope": effect_scope,
                    "target_key": target_key,
                    "member_rank": 1,
                    "member_id": base_sample_id or Path(base_sample_root).name,
                    "sample_id": base_sample_id,
                    "sample_root": base_sample_root,
                    "sample_source": str(panel_row.get("sample_source") or ""),
                    "pdb_id": base_pdb_id,
                    "seed_rank": int(seed_rank_map.get(base_pdb_id, 9999)) if base_pdb_id else 9999,
                    "selection_bucket": str(panel_row.get("stage5_selection_bucket") or ""),
                    "is_panel_representative": True,
                },
            )
            members = members[:max_members]

        if len(members) < min_members:
            for seed_row in list(case_entry.get("seed_pdb_candidates") or []):
                pdb_id = str(seed_row.get("pdb_id") or "").upper()
                if not pdb_id or pdb_id in seen_pdbs:
                    continue
                modeled_root = materialize_modeled_member(
                    root=root,
                    case_entry=case_entry,
                    effect_scope=effect_scope,
                    target_key=target_key,
                    pdb_id=pdb_id,
                )
                if modeled_root is None or not _ensemble_member_ready(modeled_root):
                    continue
                members.append(
                    {
                        "case_id": case_id,
                        "effect_scope": effect_scope,
                        "target_key": target_key,
                        "member_rank": int(len(members) + 1),
                        "member_id": str(pdb_id),
                        "sample_id": "",
                        "sample_root": str(modeled_root),
                        "sample_source": "receptor_ensemble_modeled",
                        "pdb_id": pdb_id,
                        "seed_rank": int(seed_row.get("seed_rank") or 9999),
                        "selection_bucket": "ensemble_modeled",
                        "is_panel_representative": False,
                    }
                )
                seen_roots.add(str(modeled_root))
                seen_pdbs.add(pdb_id)
                if len(members) >= max_members:
                    break

        for index, member in enumerate(members, start=1):
            member["member_rank"] = int(index)
            member["ensemble_size"] = int(len(members))
            member["ensemble_min_members"] = int(min_members)
            member["ensemble_meets_min"] = bool(len(members) >= min_members)
            rows.append(member)
    return pd.DataFrame.from_records(rows)


def write_receptor_ensemble_artifacts(
    *,
    root: Path,
    case_entry: dict[str, Any],
    panel_frame: pd.DataFrame,
    stage6: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(case_entry["case_id"])
    output_root = ensure_dir(root / "outputs" / case_id / "stage6_receptor_ensemble")
    mutation_pool = read_csv_optional(root / "outputs" / "case_manifests" / "mutation_pool.csv")
    members = build_receptor_ensemble_members(
        root=root,
        case_entry=case_entry,
        panel_frame=panel_frame,
        stage6=stage6,
        mutation_pool=mutation_pool,
    )
    members_path = output_root / "receptor_ensemble_members.csv"
    summary_path = output_root / "receptor_ensemble_summary.json"
    if members.empty:
        members.to_csv(members_path, index=False)
        summary = {
            "case_id": case_id,
            "generated_at": iso_now(),
            "enabled": False,
            "target_count": 0,
            "member_count": 0,
            "reason": "no_ensemble_members",
        }
        json_dump(summary_path, summary)
        return summary

    members.to_csv(members_path, index=False)
    target_sizes = members.groupby("target_key").size().astype(int)
    min_members = max(1, int(stage6.get("receptor_ensemble_min_members", 2)))
    summary = {
        "case_id": case_id,
        "generated_at": iso_now(),
        "enabled": True,
        "member_count": int(len(members)),
        "target_count": int(target_sizes.shape[0]),
        "targets_meeting_min_members": int((target_sizes >= min_members).sum()),
        "min_members": int(min_members),
        "max_members": int(stage6.get("receptor_ensemble_max_members", 4)),
        "aggregate": str(stage6.get("receptor_ensemble_aggregate", "median")),
        "members_csv": str(members_path),
        "target_size_distribution": {str(key): int(value) for key, value in target_sizes.to_dict().items()},
    }
    json_dump(summary_path, summary)
    return summary


def load_receptor_ensemble_members(case_root: Path) -> dict[str, list[dict[str, Any]]]:
    members_path = case_root / "stage6_receptor_ensemble" / "receptor_ensemble_members.csv"
    frame = read_csv_optional(members_path)
    if frame.empty:
        return {}
    mapping: dict[str, list[dict[str, Any]]] = {}
    for row in frame.sort_values(["target_key", "member_rank", "member_id"]).to_dict(orient="records"):
        mapping.setdefault(str(row["target_key"]), []).append(dict(row))
    return mapping


def aggregate_ensemble_value(values: list[float], mode: str) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    mode_value = str(mode or "median").lower()
    if mode_value == "cvar":
        tail_count = max(1, int(round(len(clean) * 0.5)))
        return float(sum(clean[:tail_count]) / tail_count)
    midpoint = len(clean) // 2
    if len(clean) % 2 == 1:
        return float(clean[midpoint])
    return float((clean[midpoint - 1] + clean[midpoint]) / 2.0)


def select_representative_member(
    members: list[dict[str, Any]],
    *,
    score_key: str,
    aggregate_value: float | None,
) -> dict[str, Any] | None:
    if not members:
        return None
    if aggregate_value is None:
        return dict(members[0])
    return min(
        (dict(member) for member in members),
        key=lambda member: (
            abs(float(member.get(score_key) or 0.0) - float(aggregate_value)),
            int(member.get("member_rank") or 9999),
            str(member.get("member_id") or ""),
        ),
    )
