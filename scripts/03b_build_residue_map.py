#!/usr/bin/env python3
"""Stage 3.2 residue mapping and mutation-site QC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import ensure_dir, load_yaml, project_root
from tools.stage3_utils import (
    case_uniprot_sequence,
    component_rows_for_sample,
    hiv_rt_domain_bounds,
    load_chain_residues,
    load_sifts_table,
    map_chain_residues,
    template_pocket_residues,
    write_csv_with_columns,
    sentinel_rows_for_sample,
    serialize_mapping_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def choose_chain(
    residues_by_chain: dict[str, list],
    preferred_chain: str | None,
    pdb_id: str,
    uniprot_id: str,
    sifts_chain_df: pd.DataFrame,
) -> tuple[str | None, str]:
    if preferred_chain and preferred_chain in residues_by_chain:
        return preferred_chain, "preferred_chain"
    if len(residues_by_chain) == 1:
        return next(iter(residues_by_chain)), "single_chain_fallback"

    best_chain = None
    best_score = None
    for chain_id, residues in residues_by_chain.items():
        segment_rows = sifts_chain_df[
            sifts_chain_df["PDB"].astype(str).str.lower().eq(str(pdb_id).lower())
            & sifts_chain_df["CHAIN"].astype(str).eq(str(chain_id))
            & sifts_chain_df["SP_PRIMARY"].astype(str).eq(str(uniprot_id))
        ]
        score = (0 if segment_rows.empty else 1, len(residues), str(chain_id))
        if best_score is None or score > best_score:
            best_score = score
            best_chain = str(chain_id)
    return best_chain, "best_sifts_chain"


def stage3_2_paths(root: Path, case_id: str) -> dict[str, Path]:
    case_root = ensure_dir(root / "outputs" / case_id / "stage3_2")
    return {
        "root": case_root,
        "residue_map": case_root / "residue_map.json",
        "residue_map_qc": case_root / "residue_map_qc.csv",
        "mutation_site_status": case_root / "mutation_site_status.csv",
        "component_mutation_status": case_root / "component_mutation_status.csv",
        "hiv_rt_numbering_qc": case_root / "hiv_rt_numbering_qc.csv",
        "excluded_samples": case_root / "excluded_samples.csv",
    }


def template_pocket_uniprot_positions(
    wt_structure_path: Path,
    wt_ligand_path: Path,
    resolved_chain_id: str,
    wt_mapping: dict[str, object],
    pocket_contact_distance_a: float,
) -> list[int]:
    mapping_lookup = {
        (int(row["pdb_resnum"]), str(row["insertion_code"])): int(row["uniprot_pos"])
        for row in wt_mapping.get("mapping_rows", [])
        if row.get("uniprot_pos") is not None
    }
    pocket_residues = template_pocket_residues(
        wt_structure_path,
        resolved_chain_id,
        wt_ligand_path,
        distance_cutoff_a=pocket_contact_distance_a,
    )
    return sorted(
        {
            mapping_lookup[(int(residue.pdb_resnum), str(residue.insertion_code))]
            for residue in pocket_residues
            if (int(residue.pdb_resnum), str(residue.insertion_code)) in mapping_lookup
        }
    )


def hiv_rt_relative_mapping(
    mapping: dict[str, object],
    full_sequence: str,
    polyprotein_start: int,
    polyprotein_end: int,
) -> tuple[dict[int, dict[str, object]], list[tuple[int, int]], list[tuple[int, int]], str, int | None]:
    if polyprotein_start < 1 or polyprotein_end < polyprotein_start:
        return {}, [], [], "", None
    offset = int(polyprotein_start - 1)
    relative_mapping = {
        int(position - offset): row
        for position, row in mapping["mapping_by_uniprot"].items()
        if polyprotein_start <= int(position) <= polyprotein_end and int(position - offset) >= 1
    }
    relative_chain_ranges = [
        (int(max(start, polyprotein_start) - offset), int(min(end, polyprotein_end) - offset))
        for start, end in mapping["chain_ranges"]
        if int(min(end, polyprotein_end) - offset) >= 1 and int(max(start, polyprotein_start)) <= int(min(end, polyprotein_end))
    ]
    relative_observed_ranges = [
        (int(max(start, polyprotein_start) - offset), int(min(end, polyprotein_end) - offset))
        for start, end in mapping["observed_ranges"]
        if int(min(end, polyprotein_end) - offset) >= 1 and int(max(start, polyprotein_start)) <= int(min(end, polyprotein_end))
    ]
    sequence_start = max(0, polyprotein_start - 1)
    sequence_end = min(len(full_sequence), polyprotein_end)
    reference_sequence = full_sequence[sequence_start:sequence_end] if sequence_start < len(full_sequence) else ""
    return relative_mapping, relative_chain_ranges, relative_observed_ranges, reference_sequence, offset


QC_COLUMNS = [
    "case_id",
    "sample_id",
    "pdb_id",
    "chain_id",
    "resolved_chain_id",
    "chain_resolution",
    "mapping_source",
    "alignment_identity",
    "alignment_query_coverage",
    "rt_numbering_offset",
    "rt_polyprotein_start",
    "rt_polyprotein_end",
    "unique_mapping_ok",
    "mapping_collision_count",
    "mapped_residue_count",
    "mapped_uniprot_count",
    "mapped_uniprot_min",
    "mapped_uniprot_max",
    "mutation_site_status",
    "all_components_mapped",
    "pocket_expected_count",
    "pocket_mapped_count",
    "pocket_coverage_fraction",
    "eligible_for_stage5",
]

MUTATION_COLUMNS = [
    "case_id",
    "sample_id",
    "pdb_id",
    "chain_id",
    "resolved_chain_id",
    "chain_resolution",
    "mapping_source",
    "alignment_identity",
    "alignment_query_coverage",
    "rt_numbering_offset",
    "rt_polyprotein_start",
    "rt_polyprotein_end",
    "unique_mapping_ok",
    "mapping_collision_count",
    "mutation_site_status",
    "all_components_mapped",
    "type",
    "evaluation_unit",
    "panel_names",
    "structure_path",
    "pocket_expected_count",
    "pocket_mapped_count",
    "pocket_coverage_fraction",
    "eligible_for_stage5",
]

COMPONENT_COLUMNS = [
    "case_id",
    "sample_id",
    "pdb_id",
    "chain_id",
    "component_mutation",
    "component_mutation_key",
    "mutation_class",
    "component_start_pos",
    "component_end_pos",
    "component_ref_aa",
    "component_alt_aa",
    "component_site_status",
    "mapped_positions",
    "observed_aa",
    "mapped_pdb_resnum",
    "mapped_insertion_code",
]

HIV_QC_COLUMNS = [
    "case_id",
    "sample_id",
    "pdb_id",
    "chain_id",
    "sentinel_pos",
    "expected_uniprot_aa",
    "mapped_pdb_resnum",
    "mapped_insertion_code",
    "mapped_pdb_aa",
    "status",
]

EXCLUDED_COLUMNS = ["case_id", "sample_id", "pdb_id", "chain_id", "reason"]


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage2 = config["stage2"]
    stage3 = config["stage3"]
    stage3_2 = config["stage3_2"]

    cases_frozen = load_yaml(root / stage2["cases_frozen_config"])
    extraction_report = pd.read_csv(root / stage3["extraction_source_report"])
    rt_domain_report = pd.read_csv(root / stage2["rt_domain_extraction_report"])
    sifts_chain_df = load_sifts_table(root / stage3_2["sifts_chain_uniprot_tsv"])
    sifts_observed_df = load_sifts_table(root / stage3_2["sifts_observed_segments_tsv"])
    alignment_min_identity = float(stage3_2["alignment_min_identity"])
    pocket_contact_distance_a = float(stage3_2["pocket_contact_distance_a"])
    pocket_coverage_threshold = float(stage3_2["pocket_coverage_threshold"])
    hiv_sentinel_positions = [int(value) for value in stage3_2["hiv_sentinel_positions"]]

    for case in cases_frozen.get("set_d", []):
        case_id = str(case["case_id"])
        output_paths = stage3_2_paths(root, case_id)
        case_sequence = case_uniprot_sequence(root, case_id)
        wt_template = case.get("wt_template", {})
        case_rows = extraction_report[
            extraction_report["case_id"].astype(str).eq(case_id)
            & ~extraction_report["is_excluded"].astype(bool)
        ].copy()

        residue_map_payload: dict[str, object] = {
            "case_id": case_id,
            "uniprot_id": case["uniprot_id"],
            "wt_template": {},
            "samples": [],
        }
        qc_rows: list[dict[str, object]] = []
        mutation_rows: list[dict[str, object]] = []
        component_rows: list[dict[str, object]] = []
        hiv_rows: list[dict[str, object]] = []
        excluded_rows: list[dict[str, object]] = []
        template_pocket_positions: list[int] = []

        wt_structure_path = root / str(wt_template["structure_path"]) if wt_template.get("structure_path") else None
        wt_ligand_path = root / "outputs" / case_id / "stage1_5" / "raw" / "ligand.sdf"
        wt_chain_id = wt_template.get("chain_id")
        if wt_structure_path and wt_structure_path.exists():
            wt_residues = load_chain_residues(wt_structure_path)
            resolved_chain, chain_resolution = choose_chain(
                wt_residues,
                None if wt_chain_id is None else str(wt_chain_id),
                str(wt_template.get("pdb_id") or ""),
                str(case["uniprot_id"]),
                sifts_chain_df,
            )
            if resolved_chain is not None:
                wt_mapping = map_chain_residues(
                    wt_residues[resolved_chain],
                    str(wt_template.get("pdb_id") or ""),
                    resolved_chain,
                    str(case["uniprot_id"]),
                    case_sequence,
                    sifts_chain_df,
                    sifts_observed_df,
                )
                wt_rt_offset = None
                wt_rt_bounds = None
                if case_id == "hiv_rt_rilpivirine":
                    template_pocket_positions = [int(value) for value in stage2["hiv_pocket_positions"]]
                    wt_rt_bounds = hiv_rt_domain_bounds(
                        rt_domain_report,
                        str(wt_template.get("pdb_id") or ""),
                        resolved_chain,
                    )
                    if wt_rt_bounds is not None:
                        _, _, _, _, wt_rt_offset = hiv_rt_relative_mapping(
                            wt_mapping,
                            case_sequence,
                            int(wt_rt_bounds["polyprotein_start"]),
                            int(wt_rt_bounds["polyprotein_end"]),
                        )
                elif wt_ligand_path.exists():
                    template_pocket_positions = template_pocket_uniprot_positions(
                        wt_structure_path,
                        wt_ligand_path,
                        resolved_chain,
                        wt_mapping,
                        pocket_contact_distance_a,
                    )
                residue_map_payload["wt_template"] = {
                    "pdb_id": wt_template.get("pdb_id"),
                    "preferred_chain_id": wt_chain_id,
                    "resolved_chain_id": resolved_chain,
                    "chain_resolution": chain_resolution,
                    "mapping_source": wt_mapping["mapping_source"],
                    "alignment_identity": wt_mapping["alignment_identity"],
                    "alignment_query_coverage": wt_mapping["alignment_query_coverage"],
                    "unique_mapping_ok": wt_mapping["unique_mapping_ok"],
                    "mapping_collision_count": len(wt_mapping["mapping_collisions"]),
                    "target_numbering_system": "rt_relative" if case_id == "hiv_rt_rilpivirine" else "uniprot",
                    "rt_numbering_offset": wt_rt_offset,
                    "rt_polyprotein_start": None if wt_rt_bounds is None else int(wt_rt_bounds["polyprotein_start"]),
                    "rt_polyprotein_end": None if wt_rt_bounds is None else int(wt_rt_bounds["polyprotein_end"]),
                    "template_pocket_positions": template_pocket_positions,
                    "residues": wt_mapping["mapping_rows"],
                }

        for row in case_rows.itertuples(index=False):
            source_map_path = root / str(row.source_map_path)
            if not source_map_path.exists():
                mutation_rows.append(
                    {
                        "case_id": case_id,
                        "sample_id": row.sample_id,
                        "pdb_id": row.pdb_id,
                        "chain_id": row.chain_id,
                        "resolved_chain_id": None,
                        "chain_resolution": "missing_source_map_json",
                        "mapping_source": None,
                        "alignment_identity": None,
                        "alignment_query_coverage": None,
                        "rt_numbering_offset": None,
                        "unique_mapping_ok": False,
                        "mapping_collision_count": 0,
                        "mutation_site_status": "UNMAPPED",
                        "all_components_mapped": False,
                        "type": row.type,
                        "evaluation_unit": row.evaluation_unit,
                        "panel_names": row.panel_names,
                        "structure_path": None,
                        "pocket_expected_count": len(template_pocket_positions),
                        "pocket_mapped_count": 0,
                        "pocket_coverage_fraction": 0.0,
                        "eligible_for_stage5": False,
                    }
                )
                excluded_rows.append(
                    {
                        "case_id": case_id,
                        "sample_id": row.sample_id,
                        "pdb_id": row.pdb_id,
                        "chain_id": row.chain_id,
                        "reason": "missing_source_map_json",
                    }
                )
                continue

            with source_map_path.open("r", encoding="utf-8") as handle:
                source_map = json.load(handle)

            mt_role = source_map.get("roles", {}).get("MT", {})
            mt_path_text = mt_role.get("standardized_path")
            mt_exists = bool(mt_role.get("exists"))
            if not mt_path_text or not mt_exists:
                component_rows_case, summary = component_rows_for_sample(
                    pd.Series(
                        {
                            "case_id": case_id,
                            "sample_id": row.sample_id,
                            "pdb_id": row.pdb_id,
                            "chain_id": row.chain_id,
                            "component_mutations": source_map.get("component_mutations", []),
                            "component_mutation_keys": source_map.get("component_mutation_keys", []),
                        }
                    ),
                    {},
                    [],
                    [],
                )
                component_rows.extend(component_rows_case)
                mutation_rows.append(
                    {
                        "case_id": case_id,
                        "sample_id": row.sample_id,
                        "pdb_id": row.pdb_id,
                        "chain_id": row.chain_id,
                        "resolved_chain_id": None,
                        "chain_resolution": "missing_mt_structure",
                        "mapping_source": None,
                        "alignment_identity": None,
                        "alignment_query_coverage": None,
                        "rt_numbering_offset": None,
                        "unique_mapping_ok": False,
                        "mapping_collision_count": 0,
                        "mutation_site_status": summary["mutation_site_status"],
                        "all_components_mapped": summary["all_components_mapped"],
                        "type": row.type,
                        "evaluation_unit": row.evaluation_unit,
                        "panel_names": row.panel_names,
                        "structure_path": mt_path_text,
                        "pocket_expected_count": len(template_pocket_positions),
                        "pocket_mapped_count": 0,
                        "pocket_coverage_fraction": 0.0,
                        "eligible_for_stage5": False,
                    }
                )
                excluded_rows.append(
                    {
                        "case_id": case_id,
                        "sample_id": row.sample_id,
                        "pdb_id": row.pdb_id,
                        "chain_id": row.chain_id,
                        "reason": "missing_mt_structure",
                    }
                )
                continue

            mt_path = root / str(mt_path_text)
            residues_by_chain = load_chain_residues(mt_path)
            resolved_chain, chain_resolution = choose_chain(
                residues_by_chain,
                None if pd.isna(row.chain_id) else str(row.chain_id),
                str(row.pdb_id),
                str(case["uniprot_id"]),
                sifts_chain_df,
            )
            if resolved_chain is None or resolved_chain not in residues_by_chain:
                mutation_rows.append(
                    {
                        "case_id": case_id,
                        "sample_id": row.sample_id,
                        "pdb_id": row.pdb_id,
                        "chain_id": row.chain_id,
                        "resolved_chain_id": None,
                        "chain_resolution": "unable_to_resolve_target_chain",
                        "mapping_source": None,
                        "alignment_identity": None,
                        "alignment_query_coverage": None,
                        "rt_numbering_offset": None,
                        "unique_mapping_ok": False,
                        "mapping_collision_count": 0,
                        "mutation_site_status": "UNMAPPED",
                        "all_components_mapped": False,
                        "type": row.type,
                        "evaluation_unit": row.evaluation_unit,
                        "panel_names": row.panel_names,
                        "structure_path": str(mt_path.relative_to(root)),
                        "pocket_expected_count": len(template_pocket_positions),
                        "pocket_mapped_count": 0,
                        "pocket_coverage_fraction": 0.0,
                        "eligible_for_stage5": False,
                    }
                )
                excluded_rows.append(
                    {
                        "case_id": case_id,
                        "sample_id": row.sample_id,
                        "pdb_id": row.pdb_id,
                        "chain_id": row.chain_id,
                        "reason": "unable_to_resolve_target_chain",
                    }
                )
                continue

            mapping = map_chain_residues(
                residues_by_chain[resolved_chain],
                str(row.pdb_id),
                resolved_chain,
                str(case["uniprot_id"]),
                case_sequence,
                sifts_chain_df,
                sifts_observed_df,
            )
            if mapping["mapping_source"] == "sequence_alignment" and mapping["alignment_identity"] is not None:
                if float(mapping["alignment_identity"]) < alignment_min_identity:
                    mapping["mapping_by_uniprot"] = {}
                    mapping["chain_ranges"] = []
                    mapping["observed_ranges"] = []
                    mapping["unique_mapping_ok"] = False

            site_mapping = mapping["mapping_by_uniprot"]
            site_chain_ranges = mapping["chain_ranges"]
            site_observed_ranges = mapping["observed_ranges"]
            site_reference_sequence = case_sequence
            rt_numbering_offset = None
            rt_polyprotein_start = None
            rt_polyprotein_end = None
            if case_id == "hiv_rt_rilpivirine":
                bounds = hiv_rt_domain_bounds(rt_domain_report, str(row.pdb_id), resolved_chain)
                if bounds is None:
                    mutation_rows.append(
                        {
                            "case_id": case_id,
                            "sample_id": row.sample_id,
                            "pdb_id": row.pdb_id,
                            "chain_id": row.chain_id,
                            "resolved_chain_id": resolved_chain,
                            "chain_resolution": chain_resolution,
                            "mapping_source": mapping["mapping_source"],
                            "alignment_identity": mapping["alignment_identity"],
                            "alignment_query_coverage": mapping["alignment_query_coverage"],
                            "rt_numbering_offset": None,
                            "rt_polyprotein_start": None,
                            "rt_polyprotein_end": None,
                            "unique_mapping_ok": False,
                            "mapping_collision_count": int(len(mapping["mapping_collisions"])),
                            "mutation_site_status": "UNMAPPED",
                            "all_components_mapped": False,
                            "type": row.type,
                            "evaluation_unit": row.evaluation_unit,
                            "panel_names": row.panel_names,
                            "structure_path": str(mt_path.relative_to(root)),
                            "pocket_expected_count": len(template_pocket_positions),
                            "pocket_mapped_count": 0,
                            "pocket_coverage_fraction": 0.0,
                            "eligible_for_stage5": False,
                        }
                    )
                    excluded_rows.append(
                        {
                            "case_id": case_id,
                            "sample_id": row.sample_id,
                            "pdb_id": row.pdb_id,
                            "chain_id": resolved_chain,
                            "reason": "missing_hiv_rt_domain_bounds",
                        }
                    )
                    continue
                rt_polyprotein_start = int(bounds["polyprotein_start"])
                rt_polyprotein_end = int(bounds["polyprotein_end"])
                (
                    site_mapping,
                    site_chain_ranges,
                    site_observed_ranges,
                    site_reference_sequence,
                    rt_numbering_offset,
                ) = hiv_rt_relative_mapping(
                    mapping,
                    case_sequence,
                    rt_polyprotein_start,
                    rt_polyprotein_end,
                )

            sample_series = pd.Series(
                {
                    "case_id": case_id,
                    "sample_id": row.sample_id,
                    "pdb_id": row.pdb_id,
                    "chain_id": resolved_chain,
                    "component_mutations": source_map.get("component_mutations", []),
                    "component_mutation_keys": source_map.get("component_mutation_keys", []),
                }
            )
            sample_component_rows, summary = component_rows_for_sample(
                sample_series,
                site_mapping,
                site_chain_ranges,
                site_observed_ranges,
            )
            component_rows.extend(sample_component_rows)
            pocket_expected_count = len(template_pocket_positions)
            pocket_mapped_count = sum(
                1 for position in template_pocket_positions if int(position) in site_mapping
            )
            pocket_coverage_fraction = (
                float(pocket_mapped_count / pocket_expected_count) if pocket_expected_count else 0.0
            )
            eligible_for_stage5 = bool(
                summary["mutation_site_status"] in {"PRESENT", "MUTATED_IN_PDB"}
                and summary["all_components_mapped"]
                and bool(mapping["unique_mapping_ok"])
                and pocket_expected_count > 0
                and pocket_coverage_fraction >= pocket_coverage_threshold
            )

            mutation_row = {
                "case_id": case_id,
                "sample_id": row.sample_id,
                "pdb_id": row.pdb_id,
                "chain_id": row.chain_id,
                "resolved_chain_id": resolved_chain,
                "chain_resolution": chain_resolution,
                "mapping_source": mapping["mapping_source"],
                "alignment_identity": mapping["alignment_identity"],
                "alignment_query_coverage": mapping["alignment_query_coverage"],
                "unique_mapping_ok": bool(mapping["unique_mapping_ok"]),
                "mapping_collision_count": int(len(mapping["mapping_collisions"])),
                "rt_numbering_offset": rt_numbering_offset,
                "rt_polyprotein_start": rt_polyprotein_start,
                "rt_polyprotein_end": rt_polyprotein_end,
                "mutation_site_status": summary["mutation_site_status"],
                "all_components_mapped": summary["all_components_mapped"],
                "type": row.type,
                "evaluation_unit": row.evaluation_unit,
                "panel_names": row.panel_names,
                "structure_path": str(mt_path.relative_to(root)),
                "pocket_expected_count": pocket_expected_count,
                "pocket_mapped_count": pocket_mapped_count,
                "pocket_coverage_fraction": pocket_coverage_fraction,
                "eligible_for_stage5": eligible_for_stage5,
            }
            mutation_rows.append(mutation_row)

            mapped_positions = [value["uniprot_pos"] for value in mapping["mapping_rows"] if value["uniprot_pos"] is not None]
            qc_rows.append(
                {
                    "case_id": case_id,
                    "sample_id": row.sample_id,
                    "pdb_id": row.pdb_id,
                    "chain_id": row.chain_id,
                    "resolved_chain_id": resolved_chain,
                    "chain_resolution": chain_resolution,
                    "mapping_source": mapping["mapping_source"],
                    "alignment_identity": mapping["alignment_identity"],
                    "alignment_query_coverage": mapping["alignment_query_coverage"],
                    "rt_numbering_offset": rt_numbering_offset,
                    "rt_polyprotein_start": rt_polyprotein_start,
                    "rt_polyprotein_end": rt_polyprotein_end,
                    "unique_mapping_ok": bool(mapping["unique_mapping_ok"]),
                    "mapping_collision_count": int(len(mapping["mapping_collisions"])),
                    "mapped_residue_count": len(mapping["mapping_rows"]),
                    "mapped_uniprot_count": len(mapping["mapping_by_uniprot"]),
                    "mapped_uniprot_min": None if not mapped_positions else int(min(mapped_positions)),
                    "mapped_uniprot_max": None if not mapped_positions else int(max(mapped_positions)),
                    "mutation_site_status": summary["mutation_site_status"],
                    "all_components_mapped": summary["all_components_mapped"],
                    "pocket_expected_count": pocket_expected_count,
                    "pocket_mapped_count": pocket_mapped_count,
                    "pocket_coverage_fraction": pocket_coverage_fraction,
                    "eligible_for_stage5": eligible_for_stage5,
                }
            )

            residue_map_payload["samples"].append(
                {
                    "sample_id": row.sample_id,
                    "pdb_id": row.pdb_id,
                    "preferred_chain_id": row.chain_id,
                    "resolved_chain_id": resolved_chain,
                    "chain_resolution": chain_resolution,
                    "mapping_source": mapping["mapping_source"],
                    "alignment_identity": mapping["alignment_identity"],
                    "alignment_query_coverage": mapping["alignment_query_coverage"],
                    "unique_mapping_ok": mapping["unique_mapping_ok"],
                    "mapping_collision_count": len(mapping["mapping_collisions"]),
                    "target_numbering_system": "rt_relative" if case_id == "hiv_rt_rilpivirine" else "uniprot",
                    "rt_numbering_offset": rt_numbering_offset,
                    "rt_polyprotein_start": rt_polyprotein_start,
                    "rt_polyprotein_end": rt_polyprotein_end,
                    "template_pocket_coverage_fraction": pocket_coverage_fraction,
                    "residues": mapping["mapping_rows"],
                }
            )

            exclusion_reasons: list[str] = []
            if not bool(mapping["unique_mapping_ok"]):
                exclusion_reasons.append("non_unique_residue_mapping")
            if summary["mutation_site_status"] not in {"PRESENT", "MUTATED_IN_PDB"}:
                exclusion_reasons.append(f"mutation_site_status={summary['mutation_site_status']}")
            if not summary["all_components_mapped"]:
                exclusion_reasons.append("all_components_mapped=false")
            if pocket_expected_count == 0:
                exclusion_reasons.append("template_pocket_residue_set_empty")
            if pocket_expected_count and pocket_coverage_fraction < pocket_coverage_threshold:
                exclusion_reasons.append("pocket_mapping_coverage_below_threshold")

            if case_id == "hiv_rt_rilpivirine":
                sentinel = sentinel_rows_for_sample(
                    pd.Series(
                        {
                            "case_id": case_id,
                            "sample_id": row.sample_id,
                            "pdb_id": row.pdb_id,
                            "chain_id": resolved_chain,
                        }
                    ),
                    site_mapping,
                    hiv_sentinel_positions,
                    site_reference_sequence,
                )
                hiv_rows.extend(sentinel)
                if not all(item["status"] == "PASS" for item in sentinel):
                    exclusion_reasons.append("hiv_rt_numbering_failed")

            if exclusion_reasons:
                excluded_rows.append(
                    {
                        "case_id": case_id,
                        "sample_id": row.sample_id,
                        "pdb_id": row.pdb_id,
                        "chain_id": resolved_chain,
                        "reason": "|".join(exclusion_reasons),
                    }
                )

        with output_paths["residue_map"].open("w", encoding="utf-8") as handle:
            json.dump(serialize_mapping_payload(residue_map_payload), handle, indent=2, ensure_ascii=True)
            handle.write("\n")

        write_csv_with_columns(pd.DataFrame.from_records(qc_rows, columns=QC_COLUMNS), output_paths["residue_map_qc"], QC_COLUMNS)
        write_csv_with_columns(
            pd.DataFrame.from_records(mutation_rows, columns=MUTATION_COLUMNS),
            output_paths["mutation_site_status"],
            MUTATION_COLUMNS,
        )
        write_csv_with_columns(
            pd.DataFrame.from_records(component_rows, columns=COMPONENT_COLUMNS),
            output_paths["component_mutation_status"],
            COMPONENT_COLUMNS,
        )
        write_csv_with_columns(
            pd.DataFrame.from_records(excluded_rows, columns=EXCLUDED_COLUMNS),
            output_paths["excluded_samples"],
            EXCLUDED_COLUMNS,
        )
        if case_id == "hiv_rt_rilpivirine":
            write_csv_with_columns(
                pd.DataFrame.from_records(hiv_rows, columns=HIV_QC_COLUMNS),
                output_paths["hiv_rt_numbering_qc"],
                HIV_QC_COLUMNS,
            )
        else:
            write_csv_with_columns(pd.DataFrame(columns=HIV_QC_COLUMNS), output_paths["hiv_rt_numbering_qc"], HIV_QC_COLUMNS)


if __name__ == "__main__":
    main()
