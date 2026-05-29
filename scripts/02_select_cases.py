#!/usr/bin/env python3
"""Stage 2 case freezing and HIV structure gating."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.public_data_utils import (
    evaluate_hiv_candidate_chains,
    fetch_rcsb_files,
    hiv_reference_domains,
    load_mmcif_dict,
    mmcif_entry_title,
    mmcif_resolution,
    nnrti_ligand_summary,
    request_session,
    select_best_template_chain,
)
from tools.runtime import ensure_dir, iso_now, json_dump, load_yaml, project_root, text_dump

HOTSPOT_MAP = {
    "egfr_erlotinib": {
        "EGFR:T790M",
        "EGFR:L858R",
        "EGFR:G719S",
        "EGFR:S768I",
        "EGFR:L861Q",
        "EGFR:E746_A750delELREA",
    },
    "abl1_nilotinib": {
        "ABL1:T315I",
        "ABL1:E255K",
        "ABL1:Y253H",
        "ABL1:F359V",
        "ABL1:M351T",
    },
    "hiv_rt_rilpivirine": {
        "GAG-POL_RT:K101E",
        "GAG-POL_RT:K101H",
        "GAG-POL_RT:E138A",
        "GAG-POL_RT:E138G",
        "GAG-POL_RT:E138K",
        "GAG-POL_RT:E138Q",
        "GAG-POL_RT:Y181C",
        "GAG-POL_RT:Y181I",
        "GAG-POL_RT:G190A",
        "GAG-POL_RT:G190S",
        "GAG-POL_RT:F227C",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def compute_p_appearance(background: float, selected: float) -> float:
    return 1.0 - (1.0 - float(background)) * (1.0 - float(selected))


def compute_log_enrichment(background: float, selected: float, eps: float = 1.0e-6) -> float:
    return math.log2((float(selected) + eps) / (float(background) + eps))


def python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return [value]


def build_case_master(master_table: pd.DataFrame, case: dict[str, Any]) -> pd.DataFrame:
    uniprot_id = str(case["uniprot_id"])
    drug_name = str(case["drug_name"])
    return master_table[
        master_table["UNIPROT_ID"].astype(str).eq(uniprot_id) & master_table["drug_name"].astype(str).eq(drug_name)
    ].copy()


def seed_candidate_counts(case_master: pd.DataFrame, limit: int) -> pd.DataFrame:
    counts = (
        case_master.dropna(subset=["PDB_ID"])
        .assign(PDB_ID=lambda frame: frame["PDB_ID"].astype(str))
        .groupby("PDB_ID", dropna=False)
        .agg(sample_count=("SAMPLE_ID", "nunique"))
        .reset_index()
        .sort_values(["sample_count", "PDB_ID"], ascending=[False, True])
        .head(limit)
        .reset_index(drop=True)
    )
    counts["seed_rank"] = counts.index + 1
    return counts


def build_site_pool(case_master: pd.DataFrame, global_prior: pd.DataFrame, case_id: str) -> pd.DataFrame:
    gene_symbol = str(case_master["gene_symbol"].dropna().astype(str).mode().iloc[0])
    drug_name = str(case_master["drug_name"].dropna().astype(str).mode().iloc[0])
    prior = global_prior[
        global_prior["gene_symbol"].astype(str).eq(gene_symbol) & global_prior["drug_name"].astype(str).eq(drug_name)
    ].copy()

    support_rows = []
    for row in case_master.itertuples(index=False):
        component_mutations = as_list(getattr(row, "component_mutations"))
        component_keys = as_list(getattr(row, "component_mutation_keys"))
        for mutation, mutation_key in zip(component_mutations, component_keys):
            support_rows.append(
                {
                    "SAMPLE_ID": row.SAMPLE_ID,
                    "mutation_key": mutation_key,
                    "component_mutation": mutation,
                    "PDB_ID": row.PDB_ID,
                    "combination_size": int(row.combination_size),
                    "evaluation_unit": row.evaluation_unit,
                    "TYPE": row.TYPE,
                    "DDG.EXP": row._asdict().get("DDG.EXP"),
                }
            )
    support = pd.DataFrame.from_records(support_rows)
    if support.empty:
        return prior.assign(
            sample_support_count=0,
            single_sample_support=0,
            combo_sample_support=0,
            representative_sample_id=None,
            representative_pdb_id=None,
            known_hotspot=False,
            P_appearance=lambda frame: frame.apply(
                lambda row: compute_p_appearance(row["P_background"], row["P_drug_selected"]), axis=1
            ),
        )

    seed_rank_lookup = {
        str(pdb_id): int(rank)
        for pdb_id, rank in support.dropna(subset=["PDB_ID"]).assign(PDB_ID=lambda frame: frame["PDB_ID"].astype(str))[
            ["PDB_ID"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(seed_rank=lambda frame: frame.index + 1)
        .itertuples(index=False)
    }
    representative_rows = []
    for mutation_key, group in support.groupby("mutation_key", dropna=False):
        ordered = group.assign(
            _single_pref=lambda frame: frame["combination_size"].eq(1).astype(int),
            _pdb_rank=lambda frame: frame["PDB_ID"].astype(str).map(seed_rank_lookup).fillna(10**6),
        ).sort_values(["_single_pref", "_pdb_rank", "SAMPLE_ID"], ascending=[False, True, True])
        best = ordered.iloc[0]
        representative_rows.append(
            {
                "mutation_key": mutation_key,
                "sample_support_count": int(group["SAMPLE_ID"].nunique()),
                "single_sample_support": int(group.loc[group["combination_size"].eq(1), "SAMPLE_ID"].nunique()),
                "combo_sample_support": int(group.loc[group["combination_size"].gt(1), "SAMPLE_ID"].nunique()),
                "representative_sample_id": str(best["SAMPLE_ID"]),
                "representative_pdb_id": None if pd.isna(best["PDB_ID"]) else str(best["PDB_ID"]),
            }
        )
    site_pool = prior.merge(pd.DataFrame.from_records(representative_rows), on="mutation_key", how="left")
    site_pool["sample_support_count"] = site_pool["sample_support_count"].fillna(0).astype(int)
    site_pool["single_sample_support"] = site_pool["single_sample_support"].fillna(0).astype(int)
    site_pool["combo_sample_support"] = site_pool["combo_sample_support"].fillna(0).astype(int)
    site_pool["known_hotspot"] = site_pool["mutation_key"].isin(HOTSPOT_MAP.get(case_id, set()))
    site_pool["curated_hotspot_flag"] = site_pool["known_hotspot"].astype(bool)
    site_pool["evaluation_hotspot_label"] = site_pool["known_hotspot"].astype(bool)
    site_pool["P_appearance"] = site_pool.apply(
        lambda row: compute_p_appearance(row["P_background"], row["P_drug_selected"]),
        axis=1,
    )
    site_pool["log_enrichment"] = site_pool.apply(
        lambda row: compute_log_enrichment(row["P_background"], row["P_drug_selected"]),
        axis=1,
    )
    site_pool = site_pool.sort_values(
        ["P_appearance", "log_enrichment", "sample_support_count", "mutation_key"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    site_pool["site_rank"] = site_pool.index + 1
    return site_pool


def build_mutation_pool(case_master: pd.DataFrame, site_pool: pd.DataFrame) -> pd.DataFrame:
    site_lookup = site_pool.set_index("mutation_key")[["P_appearance", "log_enrichment", "known_hotspot"]].to_dict("index")
    sample_rows = []
    for row in case_master.itertuples(index=False):
        component_keys = as_list(getattr(row, "component_mutation_keys"))
        site_metrics = [site_lookup[key] for key in component_keys if key in site_lookup]
        if site_metrics:
            p_appearance_max = max(metric["P_appearance"] for metric in site_metrics)
            p_appearance_mean = sum(metric["P_appearance"] for metric in site_metrics) / float(len(site_metrics))
            hotspot_count = sum(bool(metric["known_hotspot"]) for metric in site_metrics)
        else:
            p_appearance_max = 0.0
            p_appearance_mean = 0.0
            hotspot_count = 0
        sample_rows.append(
            {
                "case_id": None,
                "SAMPLE_ID": str(row.SAMPLE_ID),
                "TYPE": row.TYPE,
                "PDB_ID": None if pd.isna(row.PDB_ID) else str(row.PDB_ID),
                "MUTATION": row.MUTATION,
                "combination_size": int(row.combination_size),
                "evaluation_unit": row.evaluation_unit,
                "mutation_key": row.mutation_key,
                "combination_key": row.combination_key,
                "component_mutation_keys": list(component_keys),
                "sample_risk_proxy": float(p_appearance_max),
                "sample_risk_mean": float(p_appearance_mean),
                "hotspot_component_count": int(hotspot_count),
                "quality_ok": bool(row.mutation_parse_ok and not pd.isna(row.PDB_ID)),
            }
        )
    mutation_pool = pd.DataFrame.from_records(sample_rows)
    mutation_pool = mutation_pool.sort_values(
        ["sample_risk_proxy", "sample_risk_mean", "combination_size", "SAMPLE_ID"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return mutation_pool


def assign_panels(
    mutation_pool: pd.DataFrame,
    site_pool: pd.DataFrame,
    manifests_root: Path,
    case_id: str,
    top_risk_n: int,
    medium_n: int,
    tail_n: int,
) -> dict[str, Any]:
    selected_site_pool = site_pool.head(top_risk_n).copy()
    top_risk_sample_ids = [str(value) for value in selected_site_pool["representative_sample_id"].dropna().astype(str).tolist()]

    available = mutation_pool[mutation_pool["quality_ok"]].copy()
    available = available[~available["SAMPLE_ID"].isin(set(top_risk_sample_ids))].copy()

    medium_pool = available.sort_values(
        ["sample_risk_proxy", "sample_risk_mean", "SAMPLE_ID"],
        ascending=[False, False, True],
    ).head(medium_n)
    used_medium = set(medium_pool["SAMPLE_ID"].astype(str))

    tail_candidates = available[~available["SAMPLE_ID"].astype(str).isin(used_medium)].copy()
    tail_pool = tail_candidates.sort_values(
        ["sample_risk_proxy", "combination_size", "SAMPLE_ID"],
        ascending=[True, True, True],
    ).head(tail_n)

    panel_rows = {
        "TopRisk20": top_risk_sample_ids,
        "MediumRisk80": medium_pool["SAMPLE_ID"].astype(str).tolist(),
        "Tail100": tail_pool["SAMPLE_ID"].astype(str).tolist(),
    }
    for panel_name, sample_ids in panel_rows.items():
        text_dump(manifests_root / f"{case_id}_{panel_name}.sample_id_list.txt", "\n".join(sample_ids) + ("\n" if sample_ids else ""))
    return {
        "top_risk_site_pool": selected_site_pool,
        "medium_pool": medium_pool,
        "tail_pool": tail_pool,
        "panel_rows": panel_rows,
    }


def build_combo_panel(
    combo_prior: pd.DataFrame,
    combination_map: pd.DataFrame,
    case_master: pd.DataFrame,
    case_id: str,
    top_combo_n: int,
    combo_min_support: int,
) -> pd.DataFrame:
    gene_symbol = str(case_master["gene_symbol"].dropna().astype(str).mode().iloc[0])
    drug_name = str(case_master["drug_name"].dropna().astype(str).mode().iloc[0])
    combo_case = combo_prior[
        combo_prior["gene_symbol"].astype(str).eq(gene_symbol) & combo_prior["drug_name"].astype(str).eq(drug_name)
    ].copy()
    if combo_case.empty:
        return combo_case.assign(case_id=case_id)

    representative = (
        combination_map[
            combination_map["gene_symbol"].astype(str).eq(gene_symbol)
            & combination_map["drug_name"].astype(str).eq(drug_name)
        ][["sample_id", "combination_key"]]
        .drop_duplicates()
        .groupby("combination_key", dropna=False)
        .agg(representative_sample_id=("sample_id", "min"))
        .reset_index()
    )
    combo_case = combo_case.merge(representative, on="combination_key", how="left")
    combo_case["case_id"] = case_id
    combo_case["support_tier"] = combo_case["count"].ge(combo_min_support).map({True: "supported", False: "low_support"})
    combo_case = combo_case.sort_values(
        ["count", "freq", "combination_key"],
        ascending=[False, False, True],
    ).head(top_combo_n)
    combo_case["combo_rank"] = range(1, len(combo_case) + 1)
    return combo_case


def read_stage1_5_target(root: Path, case_id: str) -> dict[str, Any] | None:
    target_path = root / "outputs" / case_id / "stage1_5" / "meta" / "target.json"
    if not target_path.exists():
        return None
    return load_yaml(target_path) if target_path.suffix in {".yaml", ".yml"} else pd.read_json(target_path, typ="series").to_dict()


def load_target_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def wt_template_payload(target_payload: dict[str, Any] | None) -> dict[str, Any]:
    structure = (target_payload or {}).get("structure") or {}
    best_chain = structure.get("best_chain") or {}
    checks = (target_payload or {}).get("checks") or {}
    return {
        "available": bool(checks.get("overall_pass")),
        "structure_source": structure.get("source"),
        "pdb_id": structure.get("pdb_id"),
        "chain_id": best_chain.get("chain_id"),
        "resolution": structure.get("resolution"),
        "structure_path": structure.get("structure_path"),
        "structure_cif_path": structure.get("structure_cif_path"),
        "modeling_required": bool(checks.get("modeling_required")),
        "modeling_request_path": structure.get("modeling_request_path"),
    }


def build_frozen_cases_payload(
    seed_cases_config: dict[str, Any],
    frozen_rows: list[dict[str, Any]],
    seed_path: Path,
    frozen_path: Path,
) -> dict[str, Any]:
    set_n = seed_cases_config.get("set_n", [])
    return {
        "generated_by": "scripts/02_select_cases.py",
        "frozen_at": iso_now(),
        "source_seed_config": str(seed_path),
        "output_path": str(frozen_path),
        "set_d": frozen_rows,
        "set_n": set_n,
    }


def concat_record_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        records.extend(frame.to_dict(orient="records"))
    return pd.DataFrame.from_records(records)


def build_hiv_structure_reports(
    root: Path,
    case: dict[str, Any],
    candidate_counts: pd.DataFrame,
    stage2: dict[str, Any],
    target_payload: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    timeout = int(stage2["request_timeout_sec"])
    pocket_positions = [int(value) for value in stage2["hiv_pocket_positions"]]
    rt_keywords = [str(value).lower() for value in stage2["hiv_rt_keywords"]]
    exclude_keywords = [str(value).lower() for value in stage2["hiv_exclude_keywords"]]
    alias_terms = [str(value).lower() for value in stage2["hiv_nnrti_aliases"]]
    min_identity = float(stage2["hiv_domain_min_identity"])
    min_query_coverage = float(stage2["hiv_domain_min_query_coverage"])
    min_effective_score = float(stage2["hiv_domain_min_effective_score"])
    cache_root = root / stage2["rcsb_cache_root"]
    session = request_session()
    reference_domains = hiv_reference_domains(((target_payload or {}).get("sequence") or {}).get("record") or {})
    if not reference_domains:
        raise SystemExit("Unable to extract HIV RT domain references from Stage 1.5 target.json")

    template_files = fetch_rcsb_files(session, str(case["nnrti_template_pdb"]), cache_root, timeout)
    template_dict = load_mmcif_dict(template_files["cif"])
    template_chain = select_best_template_chain(template_dict, pocket_positions, rt_keywords, exclude_keywords)
    template_residues = template_chain["pocket_residues"]

    domain_rows = []
    rt_domain_rows = []
    pocket_rows = []
    structure_rows = []
    passed_seed_candidates: list[dict[str, Any]] = []
    rt_domain_report_path = stage2["rt_domain_extraction_report"]

    for row in candidate_counts.itertuples(index=False):
        pdb_id = str(row.PDB_ID)
        files = fetch_rcsb_files(session, pdb_id, cache_root, timeout)
        cif_dict = load_mmcif_dict(files["cif"])
        title = mmcif_entry_title(cif_dict)
        chain_rows = evaluate_hiv_candidate_chains(
            cif_dict,
            title,
            reference_domains,
            template_residues,
            pocket_positions,
            rt_keywords,
            exclude_keywords,
            min_identity,
            min_query_coverage,
            min_effective_score,
        )
        best_chain = max(
            chain_rows,
            key=lambda item: (
                1 if item["pass_rt_domain_gate"] else 0,
                1 if item["pass_sequence_domain_gate"] else 0,
                1 if item["rt_keyword_hit"] else 0,
                0 if item["excluded_keyword_hit"] else 1,
                float(item["sequence_effective_score"]),
                float(item["pocket_similarity"]),
                float(item["coverage_fraction"]),
                item["chain_id"],
            ),
        )
        title_lower = title.lower()
        title_alias_hit = any(alias in title_lower for alias in alias_terms)
        ligand_summary = nnrti_ligand_summary(cif_dict, alias_terms)
        pocket_similarity_pass = bool(best_chain["pocket_similarity"] > 0.9)
        domain_pass = bool(
            best_chain["pass_rt_domain_gate"]
            and pocket_similarity_pass
            and best_chain["coverage_count"] == len(pocket_positions)
        )
        is_holo_nnrti = bool(
            ligand_summary["has_nonpoly_ligand"]
            and (ligand_summary["is_holo_nnrti"] or title_alias_hit)
            and domain_pass
        )
        filter_reason = "PASS" if is_holo_nnrti else (
            "failed_domain_filter"
            if not domain_pass
            else "not_holo_nnrti"
        )
        for chain_row in chain_rows:
            rt_domain_rows.append(
                {
                    "case_id": case["case_id"],
                    "pdb_id": pdb_id,
                    "seed_rank": int(row.seed_rank),
                    "sample_count": int(row.sample_count),
                    "title": title,
                    "chain_id": chain_row["chain_id"],
                    "entity_description": chain_row["entity_description"],
                    "annotation_label": chain_row["annotation_label"],
                    "is_gag_pol_annotation": bool(chain_row["is_gag_pol_annotation"]),
                    "sequence_best_family": chain_row["sequence_best_family"],
                    "sequence_best_domain": chain_row["sequence_best_domain"],
                    "sequence_identity": float(chain_row["sequence_identity"]),
                    "sequence_query_coverage": float(chain_row["sequence_query_coverage"]),
                    "sequence_ref_coverage": float(chain_row["sequence_ref_coverage"]),
                    "sequence_effective_score": float(chain_row["sequence_effective_score"]),
                    "sequence_alignment_score": float(chain_row["sequence_alignment_score"]),
                    "sequence_runner_up_family": chain_row["sequence_runner_up_family"],
                    "sequence_runner_up_score": float(chain_row["sequence_runner_up_score"]),
                    "sequence_polyprotein_start": chain_row["sequence_polyprotein_start"],
                    "sequence_polyprotein_end": chain_row["sequence_polyprotein_end"],
                    "pass_sequence_domain_gate": bool(chain_row["pass_sequence_domain_gate"]),
                    "pass_rt_domain_gate": bool(chain_row["pass_rt_domain_gate"]),
                    "selected_best_chain": bool(chain_row["chain_id"] == best_chain["chain_id"]),
                }
            )
        domain_rows.append(
            {
                "case_id": case["case_id"],
                "pdb_id": pdb_id,
                "seed_rank": int(row.seed_rank),
                "sample_count": int(row.sample_count),
                "title": title,
                "resolution": mmcif_resolution(cif_dict),
                "chain_id": best_chain["chain_id"],
                "entity_description": best_chain["entity_description"],
                "annotation_label": best_chain["annotation_label"],
                "sequence_best_family": best_chain["sequence_best_family"],
                "sequence_best_domain": best_chain["sequence_best_domain"],
                "sequence_identity": float(best_chain["sequence_identity"]),
                "sequence_query_coverage": float(best_chain["sequence_query_coverage"]),
                "sequence_effective_score": float(best_chain["sequence_effective_score"]),
                "is_gag_pol_annotation": bool(best_chain["is_gag_pol_annotation"]),
                "pass_sequence_domain_gate": bool(best_chain["pass_sequence_domain_gate"]),
                "pass_domain_filter": domain_pass,
                "filter_reason": filter_reason,
                "rt_domain_report_path": rt_domain_report_path,
            }
        )
        pocket_rows.append(
            {
                "case_id": case["case_id"],
                "pdb_id": pdb_id,
                "template_pdb": str(case["nnrti_template_pdb"]).upper(),
                "template_chain": template_chain["chain_id"],
                "candidate_chain": best_chain["chain_id"],
                "coverage_count": int(best_chain["coverage_count"]),
                "coverage_fraction": float(best_chain["coverage_fraction"]),
                "pocket_similarity": float(best_chain["pocket_similarity"]),
                "pass_similarity_gate": pocket_similarity_pass,
                "sequence_best_domain": best_chain["sequence_best_domain"],
                "sequence_effective_score": float(best_chain["sequence_effective_score"]),
                "rt_domain_report_path": rt_domain_report_path,
            }
        )
        structure_rows.append(
            {
                "case_id": case["case_id"],
                "pdb_id": pdb_id,
                "chain_id": best_chain["chain_id"],
                "annotation_label": best_chain["annotation_label"],
                "sequence_best_family": best_chain["sequence_best_family"],
                "sequence_best_domain": best_chain["sequence_best_domain"],
                "is_gag_pol_annotation": bool(best_chain["is_gag_pol_annotation"]),
                "pass_sequence_domain_gate": bool(best_chain["pass_sequence_domain_gate"]),
                "pass_rt_domain_gate": bool(best_chain["pass_rt_domain_gate"]),
                "is_holo_nnrti": is_holo_nnrti,
                "has_nonpoly_ligand": bool(ligand_summary["has_nonpoly_ligand"]),
                "matched_nnrti_comp_ids": "|".join(
                    sorted({ligand["comp_id"] for ligand in ligand_summary["matched_nnrti_ligands"]})
                ),
                "all_nonpoly_comp_ids": "|".join(sorted({ligand["comp_id"] for ligand in ligand_summary["ligands"]})),
                "title_alias_hit": title_alias_hit,
                "pass_domain_filter": domain_pass,
                "pass_similarity_gate": pocket_similarity_pass,
                "filter_reason": filter_reason,
            }
        )
        if is_holo_nnrti:
            passed_seed_candidates.append(
                {
                    "pdb_id": pdb_id,
                    "seed_rank": int(row.seed_rank),
                    "sample_count": int(row.sample_count),
                    "chain_id": best_chain["chain_id"],
                    "annotation_label": best_chain["annotation_label"],
                    "sequence_best_domain": best_chain["sequence_best_domain"],
                    "pocket_similarity": float(best_chain["pocket_similarity"]),
                    "resolution": mmcif_resolution(cif_dict),
                }
            )
    return (
        pd.DataFrame.from_records(domain_rows),
        pd.DataFrame.from_records(rt_domain_rows),
        pd.DataFrame.from_records(pocket_rows),
        pd.DataFrame.from_records(structure_rows),
        passed_seed_candidates,
    )


def default_domain_rows(case: dict[str, Any], candidate_counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in candidate_counts.itertuples(index=False):
        rows.append(
            {
                "case_id": case["case_id"],
                "pdb_id": str(row.PDB_ID),
                "seed_rank": int(row.seed_rank),
                "sample_count": int(row.sample_count),
                "title": None,
                "resolution": None,
                "chain_id": None,
                "entity_description": case.get("target_name"),
                "annotation_label": None,
                "sequence_best_family": None,
                "sequence_best_domain": None,
                "sequence_identity": None,
                "sequence_query_coverage": None,
                "sequence_effective_score": None,
                "is_gag_pol_annotation": None,
                "pass_sequence_domain_gate": None,
                "pass_domain_filter": True,
                "filter_reason": "non_hiv_case",
                "rt_domain_report_path": None,
            }
        )
    return pd.DataFrame.from_records(rows)


def case_manifest_payload(
    root: Path,
    case: dict[str, Any],
    case_master: pd.DataFrame,
    site_pool: pd.DataFrame,
    panel_assignments: dict[str, Any],
    combo_panel: pd.DataFrame,
    seed_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    target_json_path = root / "outputs" / str(case["case_id"]) / "stage1_5" / "meta" / "target.json"
    target_payload = load_target_json(target_json_path)
    wt_template = wt_template_payload(target_payload)
    mutation_type_counts = case_master["TYPE"].astype(str).value_counts().to_dict()
    panel_rows = panel_assignments["panel_rows"]
    manifest = {
        "case_id": case["case_id"],
        "target_name": case["target_name"],
        "target_domain": case["target_domain"],
        "tissue_type": case["tissue_type"],
        "drug_name": case["drug_name"],
        "uniprot_id": case["uniprot_id"],
        "selection_query": f"UNIPROT_ID={case['uniprot_id']} AND DRUG={case['drug_name']}",
        "seed_pdb_candidates": seed_candidates,
        "public_stage1_5_target_json": None if target_payload is None else str(target_json_path.relative_to(root)),
        "wt_template_available": bool(wt_template["available"]),
        "wt_template": wt_template,
        "mutation_pool": {
            "rows": int(len(case_master)),
            "type_counts": {key: int(value) for key, value in mutation_type_counts.items()},
            "top_risk_sample_list": f"outputs/case_manifests/{case['case_id']}_TopRisk20.sample_id_list.txt",
            "medium_sample_list": f"outputs/case_manifests/{case['case_id']}_MediumRisk80.sample_id_list.txt",
            "tail_sample_list": f"outputs/case_manifests/{case['case_id']}_Tail100.sample_id_list.txt",
        },
        "site_pool": {
            "rows": int(len(site_pool)),
            "top_risk20": [
                {
                    "rank": int(row.site_rank),
                    "mutation_key": row.mutation_key,
                    "P_appearance": float(row.P_appearance),
                    "P_background": float(row.P_background),
                    "P_drug_selected": float(row.P_drug_selected),
                    "sample_support_count": int(row.sample_support_count),
                    "known_hotspot": bool(row.known_hotspot),
                    "representative_sample_id": row.representative_sample_id,
                    "representative_pdb_id": row.representative_pdb_id,
                }
                for row in panel_assignments["top_risk_site_pool"].itertuples(index=False)
            ],
        },
        "combo_panel": {
            "path": "outputs/case_manifests/combo_panel.csv",
            "rows": int(len(combo_panel)),
            "selected_combinations": combo_panel["combination_key"].astype(str).tolist(),
        },
        "panel_sizes": {key: len(value) for key, value in panel_rows.items()},
    }
    if str(case["case_id"]) == "hiv_rt_rilpivirine":
        manifest["hiv_binding_mode"] = case["hiv_binding_mode"]
        manifest["nnrti_template_pdb"] = case["nnrti_template_pdb"]
        manifest["rt_domain_extraction_report"] = "outputs/case_manifests/RT_domain_extraction_report.csv"
    return manifest


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage1 = config["stage1"]
    stage2 = config["stage2"]
    cases_seed_path = root / stage2["cases_seed_config"]
    cases_frozen_path = root / stage2["cases_frozen_config"]
    cases_config = load_yaml(cases_seed_path)

    manifests_root = ensure_dir(root / stage2["manifests_root"])

    master_table = pd.read_parquet(root / stage1["master_table"])
    global_prior = pd.read_parquet(root / stage1["global_prior"])
    observed_combo_prior = pd.read_parquet(root / stage1["observed_combo_prior"])
    mutation_combination_map = pd.read_parquet(root / stage1["mutation_combination_map"])

    domain_frames = []
    rt_domain_frames = []
    pocket_frames = []
    hiv_qc_frames = []
    combo_frames = []
    site_pool_frames = []
    mutation_pool_frames = []
    manifest_paths = []
    qc_rows = []
    frozen_case_rows = []

    for case in cases_config.get("set_d", []):
        case_id = str(case["case_id"])
        case_master = build_case_master(master_table, case)
        if case_master.empty:
            raise SystemExit(f"No rows found for case {case_id}")

        candidate_limit = int(stage2["hiv_seed_pdb_top_n"]) if case_id == "hiv_rt_rilpivirine" else int(stage2["seed_pdb_top_n"])
        candidate_counts = seed_candidate_counts(case_master, candidate_limit)
        site_pool = build_site_pool(case_master, global_prior, case_id)
        site_pool["case_id"] = case_id
        site_pool_frames.append(site_pool)

        mutation_pool = build_mutation_pool(case_master, site_pool)
        mutation_pool["case_id"] = case_id

        panel_assignments = assign_panels(
            mutation_pool,
            site_pool,
            manifests_root,
            case_id,
            int(stage2["top_risk_n"]),
            int(stage2["medium_risk_n"]),
            int(stage2["tail_n"]),
        )

        combo_panel = build_combo_panel(
            observed_combo_prior,
            mutation_combination_map,
            case_master,
            case_id,
            int(stage2["top_combo_n"]),
            int(stage2["combo_min_support"]),
        )
        combo_frames.append(combo_panel)
        if not combo_panel.empty:
            text_dump(
                manifests_root / f"{case_id}_TopCombo20.sample_id_list.txt",
                "\n".join(combo_panel["representative_sample_id"].dropna().astype(str).tolist()) + "\n",
            )

        mutation_pool["panel_name"] = "unassigned"
        mutation_pool.loc[
            mutation_pool["SAMPLE_ID"].astype(str).isin(panel_assignments["panel_rows"]["TopRisk20"]),
            "panel_name",
        ] = "TopRisk20"
        mutation_pool.loc[
            mutation_pool["SAMPLE_ID"].astype(str).isin(panel_assignments["panel_rows"]["MediumRisk80"]),
            "panel_name",
        ] = "MediumRisk80"
        mutation_pool.loc[
            mutation_pool["SAMPLE_ID"].astype(str).isin(panel_assignments["panel_rows"]["Tail100"]),
            "panel_name",
        ] = "Tail100"
        mutation_pool_frames.append(mutation_pool)

        target_json_path = root / "outputs" / case_id / "stage1_5" / "meta" / "target.json"
        target_payload = load_target_json(target_json_path)

        if case_id == "hiv_rt_rilpivirine":
            domain_report, rt_domain_report, pocket_report, hiv_qc, passed_seed_candidates = build_hiv_structure_reports(
                root,
                case,
                candidate_counts,
                stage2,
                target_payload,
            )
            domain_frames.append(domain_report)
            rt_domain_frames.append(rt_domain_report)
            pocket_frames.append(pocket_report)
            hiv_qc_frames.append(hiv_qc)
            final_seed_candidates = passed_seed_candidates
        else:
            domain_frames.append(default_domain_rows(case, candidate_counts))
            final_seed_candidates = [
                {
                    "pdb_id": str(row.PDB_ID),
                    "seed_rank": int(row.seed_rank),
                    "sample_count": int(row.sample_count),
                }
                for row in candidate_counts.itertuples(index=False)
            ]

        manifest = case_manifest_payload(
            root,
            case,
            case_master,
            site_pool,
            panel_assignments,
            combo_panel,
            final_seed_candidates,
        )
        manifest_path = manifests_root / f"{case_id}.yaml"
        with manifest_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=False)
        manifest_paths.append(str(manifest_path.relative_to(root)))

        frozen_case_rows.append(
            {
                key: value
                for key, value in {
                    **case,
                    "selection_query": manifest["selection_query"],
                    "manifest_path": str(manifest_path.relative_to(root)),
                    "public_stage1_5_target_json": manifest["public_stage1_5_target_json"],
                    "wt_template": manifest["wt_template"],
                    "seed_pdb_candidates": final_seed_candidates,
                    "panel_sizes": manifest["panel_sizes"],
                    "combo_panel_path": manifest["combo_panel"]["path"],
                }.items()
            }
        )

        top_hotspots = int(panel_assignments["top_risk_site_pool"]["known_hotspot"].sum())
        wt_target = target_payload
        qc_rows.append(
            {
                "case_id": case_id,
                "top_risk_hotspot_count": top_hotspots,
                "wt_template_available": bool(wt_target and wt_target.get("checks", {}).get("overall_pass")),
                "mt_extractable_count": int(mutation_pool["quality_ok"].sum()),
                "top_combo_rows": int(len(combo_panel)),
                "hiv_seed_pass_count": int(len(final_seed_candidates)) if case_id == "hiv_rt_rilpivirine" else None,
            }
        )

    domain_filter_report = concat_record_frames(domain_frames)
    domain_filter_report.to_csv(root / stage2["domain_filter_report"], index=False)
    rt_domain_all = concat_record_frames(rt_domain_frames)
    if not rt_domain_all.empty:
        rt_domain_all.to_csv(
            root / stage2["rt_domain_extraction_report"],
            index=False,
        )
    else:
        pd.DataFrame().to_csv(root / stage2["rt_domain_extraction_report"], index=False)
    pocket_all = concat_record_frames(pocket_frames)
    if not pocket_all.empty:
        pocket_all.to_csv(root / stage2["rt_pocket_similarity_report"], index=False)
    else:
        pd.DataFrame().to_csv(root / stage2["rt_pocket_similarity_report"], index=False)
    hiv_qc_all = concat_record_frames(hiv_qc_frames)
    if not hiv_qc_all.empty:
        hiv_qc_all.to_csv(root / stage2["hiv_nnrti_structure_qc"], index=False)
    else:
        pd.DataFrame().to_csv(root / stage2["hiv_nnrti_structure_qc"], index=False)

    combo_panel_all = concat_record_frames(combo_frames)
    combo_panel_all.to_csv(root / stage2["combo_panel"], index=False)
    concat_record_frames(site_pool_frames).to_csv(root / stage2["site_pool_report"], index=False)
    concat_record_frames(mutation_pool_frames).to_csv(root / stage2["mutation_pool_report"], index=False)

    hiv_selected = hiv_qc_all[hiv_qc_all["is_holo_nnrti"].astype(bool)].copy() if not hiv_qc_all.empty else pd.DataFrame()
    pocket_selected = pocket_all[
        pocket_all["pdb_id"].astype(str).isin(hiv_selected["pdb_id"].astype(str).tolist())
    ].copy() if not pocket_all.empty and not hiv_selected.empty else pd.DataFrame()
    hiv_has_selected_templates = bool(not hiv_selected.empty)
    qc = {
        "manifest_paths": manifest_paths,
        "case_rows": qc_rows,
        "acceptance": {
            "top_risk_hotspots_ok": all(row["top_risk_hotspot_count"] >= 2 for row in qc_rows),
            "wt_and_mt_extractable_ok": all(
                bool(row["wt_template_available"]) and int(row["mt_extractable_count"]) >= 20 for row in qc_rows
            ),
            "hiv_zero_wrong_target": bool(
                hiv_has_selected_templates and hiv_selected["pass_domain_filter"].astype(bool).all()
            ),
            "hiv_pocket_similarity_gt_90": bool(
                hiv_has_selected_templates and not pocket_selected.empty and pocket_selected["pocket_similarity"].fillna(0).gt(0.9).all()
            ),
            "hiv_holo_ratio_100": bool(
                hiv_has_selected_templates and hiv_selected["is_holo_nnrti"].astype(bool).all()
            ),
            "hiv_topcombo_non_empty": bool(
                combo_panel_all[combo_panel_all["case_id"].astype(str).eq("hiv_rt_rilpivirine")].shape[0] > 0
            ),
            "hiv_rt_domain_report_present": bool((root / stage2["rt_domain_extraction_report"]).exists() or rt_domain_frames),
            "hiv_selected_templates_non_empty": hiv_has_selected_templates,
        },
    }
    frozen_cases_payload = build_frozen_cases_payload(cases_config, frozen_case_rows, cases_seed_path, cases_frozen_path)
    with cases_frozen_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(frozen_cases_payload, handle, sort_keys=False, allow_unicode=False)
    json_dump(root / stage2["case_selection_qc"], qc)


if __name__ == "__main__":
    main()
