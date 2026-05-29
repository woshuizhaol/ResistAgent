#!/usr/bin/env python3
"""Assemble Stage 1 background and drug-selected priors."""

from __future__ import annotations

import argparse
import resource
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import json_dump, load_yaml, project_root
from tools.stage1_utils import noisy_or

FITNESS_FORMULA_VERSION = "resistagent_default_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def process_peak_rss_mb() -> float:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(max_rss) / (1024.0 * 1024.0)
    return float(max_rss) / 1024.0


def tuple_key(values: list[str], row: pd.Series) -> tuple:
    return tuple(row[column] for column in values)


def build_site_contexts(master_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in master_table.itertuples(index=False):
        for mutation, mutation_key in zip(row.component_mutations, row.component_mutation_keys):
            rows.append(
                {
                    "gene_symbol": row.gene_symbol,
                    "component_mutation": mutation,
                    "mutation_key": mutation_key,
                    "domain_type": row.domain_type,
                    "target_domain": row.target_domain,
                    "drug_name": row.drug_name,
                    "UNIPROT_ID": row.UNIPROT_ID,
                }
            )
    return pd.DataFrame.from_records(rows).drop_duplicates()


def aggregate_probability_table(
    df: pd.DataFrame,
    group_cols: list[str],
    probability_col: str,
    support_label: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + [probability_col, support_label, "count", "denominator", "source_list", "evidence_tier"])

    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_cols, keys)}
        row["count"] = int(group["count"].sum()) if "count" in group else int(len(group))
        row[probability_col] = noisy_or(group["freq"].tolist())
        row[support_label] = row["count"]
        if "denominator" in group and group["denominator"].notna().any():
            denominators = {int(value) for value in group["denominator"].dropna().astype(int).tolist()}
            row["denominator"] = next(iter(denominators)) if len(denominators) == 1 else None
        else:
            row["denominator"] = None
        row["source_list"] = "|".join(sorted({str(value) for value in group["source_db"].dropna()}))
        row["evidence_tier"] = "|".join(sorted({str(value) for value in group["evidence_tier"].dropna()}))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def build_case_tissue_lookup(master_table: pd.DataFrame, cases_config: dict) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    set_d = cases_config.get("set_d", [])
    for case in set_d:
        drug_name = case.get("drug_name")
        uniprot_id = case.get("uniprot_id")
        tissue_type = case.get("tissue_type")
        if not drug_name or not uniprot_id or not tissue_type or tissue_type == "NA":
            continue
        subset = master_table[
            master_table["UNIPROT_ID"].astype(str).eq(str(uniprot_id)) & master_table["drug_name"].astype(str).eq(str(drug_name))
        ]
        if subset.empty:
            subset = master_table[master_table["UNIPROT_ID"].astype(str).eq(str(uniprot_id))]
        if subset.empty:
            continue
        gene_symbol = subset["gene_symbol"].dropna().astype(str).value_counts().index[0]
        lookup[(gene_symbol, str(drug_name))] = str(tissue_type)
    return lookup


def normalize_tissue_aliases(raw_aliases: dict | None) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for label, candidates in (raw_aliases or {}).items():
        seen = set()
        ordered = []
        for value in [label] + list(candidates or []):
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                ordered.append(text)
        aliases[str(label)] = ordered
    return aliases


def build_observed_combo_prior(combination_map: pd.DataFrame) -> pd.DataFrame:
    if combination_map.empty:
        return pd.DataFrame(
            columns=[
                "gene_symbol",
                "drug_name",
                "combination_key",
                "combination_size",
                "domain_type",
                "target_domain",
                "count",
                "denominator",
                "freq",
                "source_db",
                "evidence_tier",
            ]
        )

    grouped = (
        combination_map.groupby(["gene_symbol", "drug_name", "combination_key", "combination_size", "domain_type", "target_domain"], dropna=False)
        .agg(count=("sample_id", "nunique"))
        .reset_index()
    )
    denominators = combination_map.groupby(["gene_symbol", "drug_name"], dropna=False).agg(denominator=("sample_id", "nunique")).reset_index()
    grouped = grouped.merge(denominators, on=["gene_symbol", "drug_name"], how="left")
    grouped["freq"] = grouped["count"] / grouped["denominator"]
    grouped["source_db"] = "MdrDB"
    grouped["evidence_tier"] = "observed_combo_frequency"
    return grouped


def aggregate_tissue_candidate_rows(
    background_prior_by_tissue: pd.DataFrame,
    gene_symbol: str,
    requested_tissue: str,
    tissue_aliases: dict[str, list[str]],
) -> pd.DataFrame | None:
    candidates = tissue_aliases.get(requested_tissue, [requested_tissue])
    subset = background_prior_by_tissue[
        background_prior_by_tissue["gene_symbol"].eq(gene_symbol)
        & background_prior_by_tissue["tissue_type"].isin(candidates)
    ].copy()
    if subset.empty:
        return None

    rows = []
    for keys, group in subset.groupby(["gene_symbol", "component_mutation", "mutation_key", "domain_type", "target_domain"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        count = int(group["count"].sum())
        denominator = int(group["denominator"].sum())
        rows.append(
            {
                "gene_symbol": keys[0],
                "component_mutation": keys[1],
                "mutation_key": keys[2],
                "domain_type": keys[3],
                "target_domain": keys[4],
                "tissue_type": requested_tissue,
                "P_background": (float(count) / float(denominator)) if denominator else 0.0,
                "background_support": count,
                "count": count,
                "denominator": denominator,
                "source_list": "|".join(sorted({str(value) for value in group["source_list"].dropna()})),
                "evidence_tier": "|".join(sorted({str(value) for value in group["evidence_tier"].dropna()})),
                "prior_scope": "tissue",
                "tissue_resolution": "exact" if candidates == [requested_tissue] else "alias_group",
            }
        )
    return pd.DataFrame.from_records(rows)


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage1 = config["stage1"]
    stage2 = config["stage2"]
    cases_config = load_yaml(root / stage2["cases_seed_config"])
    epsilon_background = float(stage1["epsilon_background"])
    tissue_min_denominator = int(stage1["tissue_min_denominator"])
    tissue_aliases = normalize_tissue_aliases(stage1.get("tissue_aliases"))

    master_table = pd.read_parquet(root / stage1["master_table"])
    mutation_frequency_map = pd.read_parquet(root / stage1["mutation_frequency_map"])
    mutation_combination_map = pd.read_parquet(root / stage1["mutation_combination_map"])
    hiv_prevalence = pd.read_parquet(root / stage1["hiv_prevalence_prior"])
    hiv_drug_selected = pd.read_parquet(root / stage1["hiv_drug_selected_prior"])

    site_contexts = build_site_contexts(master_table)
    case_tissue_lookup = build_case_tissue_lookup(master_table, cases_config)

    cancer_background_sources = mutation_frequency_map[
        mutation_frequency_map["source_db"].isin(["COSMIC", "DepMap"])
        & mutation_frequency_map["context_scope"].astype(str).eq("global")
        & mutation_frequency_map["drug_name"].isna()
    ].copy()
    cancer_background = aggregate_probability_table(
        cancer_background_sources,
        ["gene_symbol", "component_mutation", "mutation_key", "domain_type", "target_domain", "tissue_type"],
        "P_background",
        "background_support",
    )
    cancer_background_lookup = {
        tuple_key(["gene_symbol", "mutation_key"], row): row
        for _, row in cancer_background.iterrows()
    }

    depmap_tissue_sources = mutation_frequency_map[
        mutation_frequency_map["source_db"].eq("DepMap") & mutation_frequency_map["context_scope"].astype(str).eq("tissue")
    ].copy()
    background_prior_by_tissue = aggregate_probability_table(
        depmap_tissue_sources,
        ["gene_symbol", "component_mutation", "mutation_key", "domain_type", "target_domain", "tissue_type"],
        "P_background",
        "background_support",
    )
    background_prior_by_tissue["prior_scope"] = "tissue"
    background_prior_by_tissue["tissue_resolution"] = "exact"

    augmented_tissue_frames = [background_prior_by_tissue]
    for (gene_symbol, _drug_name), requested_tissue in sorted(case_tissue_lookup.items()):
        alias_frame = aggregate_tissue_candidate_rows(background_prior_by_tissue, gene_symbol, requested_tissue, tissue_aliases)
        if alias_frame is not None:
            augmented_tissue_frames.append(alias_frame)
    background_prior_by_tissue = (
        pd.concat(augmented_tissue_frames, ignore_index=True, sort=False)
        .drop_duplicates(subset=["gene_symbol", "mutation_key", "tissue_type"], keep="last")
    )

    tissue_background_lookup = {
        tuple_key(["gene_symbol", "mutation_key", "tissue_type"], row): row
        for _, row in background_prior_by_tissue.iterrows()
    }

    viral_background_lookup = {
        tuple_key(["gene_symbol", "mutation_key"], row): row
        for _, row in hiv_prevalence.iterrows()
    }

    background_rows = []
    for _, site in site_contexts.drop_duplicates(subset=["gene_symbol", "mutation_key"]).iterrows():
        site_key = (site["gene_symbol"], site["mutation_key"])
        if site["domain_type"] == "viral":
            matched = viral_background_lookup.get(site_key)
            if matched is not None:
                background_rows.append(
                    {
                        "gene_symbol": site["gene_symbol"],
                        "component_mutation": site["component_mutation"],
                        "mutation_key": site["mutation_key"],
                        "domain_type": site["domain_type"],
                        "target_domain": site["target_domain"],
                        "tissue_type": "NA",
                        "P_background": float(matched["freq"]),
                        "background_support": int(matched["count"]),
                        "source_list": str(matched["source_db"]),
                        "evidence_tier": str(matched["evidence_tier"]),
                        "prior_scope": "viral_background",
                    }
                )
            else:
                background_rows.append(
                    {
                        "gene_symbol": site["gene_symbol"],
                        "component_mutation": site["component_mutation"],
                        "mutation_key": site["mutation_key"],
                        "domain_type": site["domain_type"],
                        "target_domain": site["target_domain"],
                        "tissue_type": "NA",
                        "P_background": epsilon_background,
                        "background_support": 0,
                        "source_list": "epsilon_background",
                        "evidence_tier": "low_evidence",
                        "prior_scope": "epsilon_background",
                    }
                )
            continue

        matched = cancer_background_lookup.get(site_key)
        if matched is None:
            background_rows.append(
                {
                    "gene_symbol": site["gene_symbol"],
                    "component_mutation": site["component_mutation"],
                    "mutation_key": site["mutation_key"],
                    "domain_type": site["domain_type"],
                    "target_domain": site["target_domain"],
                    "tissue_type": "Global",
                    "P_background": 0.0,
                    "background_support": 0,
                    "source_list": "missing_external_background",
                    "evidence_tier": "missing_external_background",
                    "prior_scope": "missing_external_background",
                }
            )
            continue

        background_rows.append(
            {
                "gene_symbol": site["gene_symbol"],
                "component_mutation": site["component_mutation"],
                "mutation_key": site["mutation_key"],
                "domain_type": site["domain_type"],
                "target_domain": site["target_domain"],
                "tissue_type": "Global",
                "P_background": float(matched["P_background"]),
                "background_support": int(matched["background_support"]),
                "source_list": str(matched["source_list"]),
                "evidence_tier": str(matched["evidence_tier"]),
                "prior_scope": "global",
            }
        )

    background_prior = pd.DataFrame.from_records(background_rows)
    background_prior.to_parquet(root / stage1["background_prior"], index=False)
    background_prior_by_tissue.to_parquet(root / stage1["background_prior_by_tissue"], index=False)

    mdrdb_drug_sources = mutation_frequency_map[
        mutation_frequency_map["source_db"].eq("MdrDB") & mutation_frequency_map["context_scope"].astype(str).eq("gene_drug")
    ].copy()
    cosmic_drug_sources = mutation_frequency_map[
        mutation_frequency_map["source_db"].eq("COSMIC") & mutation_frequency_map["context_scope"].astype(str).eq("gene_drug")
    ].copy()

    drug_selected_rows = []
    benchmark_rows = []
    blind_source_tracker = set()

    for _, site in site_contexts.iterrows():
        if site["domain_type"] == "viral":
            production_sources = hiv_drug_selected[
                hiv_drug_selected["gene_symbol"].eq(site["gene_symbol"])
                & hiv_drug_selected["mutation_key"].eq(site["mutation_key"])
                & hiv_drug_selected["drug_name"].eq(site["drug_name"])
            ]
            mdrdb_sources = mdrdb_drug_sources[
                mdrdb_drug_sources["gene_symbol"].eq(site["gene_symbol"])
                & mdrdb_drug_sources["mutation_key"].eq(site["mutation_key"])
                & mdrdb_drug_sources["drug_name"].eq(site["drug_name"])
            ]
            production = pd.concat([production_sources, mdrdb_sources], ignore_index=True, sort=False)
            blind = production_sources.copy()
        else:
            production_sources = mdrdb_drug_sources[
                mdrdb_drug_sources["gene_symbol"].eq(site["gene_symbol"])
                & mdrdb_drug_sources["mutation_key"].eq(site["mutation_key"])
                & mdrdb_drug_sources["drug_name"].eq(site["drug_name"])
            ]
            cosmic_sources = cosmic_drug_sources[
                cosmic_drug_sources["gene_symbol"].eq(site["gene_symbol"])
                & cosmic_drug_sources["mutation_key"].eq(site["mutation_key"])
                & cosmic_drug_sources["drug_name"].eq(site["drug_name"])
            ]
            production = pd.concat([production_sources, cosmic_sources], ignore_index=True, sort=False)
            blind = cosmic_sources.copy()

        p_selected = noisy_or(production["freq"].tolist()) if not production.empty else 0.0
        p_selected_blind = noisy_or(blind["freq"].tolist()) if not blind.empty else 0.0
        source_list = "|".join(sorted({str(value) for value in production["source_db"].dropna()})) or "none"
        blind_source_list = "|".join(sorted({str(value) for value in blind["source_db"].dropna()})) or "none"
        blind_source_tracker.update(set(str(value) for value in blind["source_db"].dropna()))

        evidence_tier = "|".join(sorted({str(value) for value in production["evidence_tier"].dropna()})) or "no_same_drug_signal"
        blind_evidence_tier = "|".join(sorted({str(value) for value in blind["evidence_tier"].dropna()})) or "no_external_same_drug_signal"

        base_row = {
            "gene_symbol": site["gene_symbol"],
            "component_mutation": site["component_mutation"],
            "mutation_key": site["mutation_key"],
            "domain_type": site["domain_type"],
            "target_domain": site["target_domain"],
            "drug_name": site["drug_name"],
        }
        drug_selected_rows.append(
            {
                **base_row,
                "P_drug_selected": float(p_selected),
                "drug_selected_support": int(production["count"].sum()) if "count" in production else int(len(production)),
                "source_list": source_list,
                "evidence_tier": evidence_tier,
                "prior_scope": "gene_drug",
            }
        )
        benchmark_rows.append(
            {
                **base_row,
                "P_drug_selected": float(p_selected_blind),
                "drug_selected_support": int(blind["count"].sum()) if "count" in blind else int(len(blind)),
                "source_list": blind_source_list,
                "evidence_tier": blind_evidence_tier,
                "prior_scope": "blind_external_only",
            }
        )

    drug_selected_prior = pd.DataFrame.from_records(drug_selected_rows).drop_duplicates(
        subset=["gene_symbol", "mutation_key", "drug_name"]
    )
    benchmark_drug_prior = pd.DataFrame.from_records(benchmark_rows).drop_duplicates(
        subset=["gene_symbol", "mutation_key", "drug_name"]
    )
    drug_selected_prior.to_parquet(root / stage1["drug_selected_prior"], index=False)

    observed_combo_prior = build_observed_combo_prior(mutation_combination_map)
    observed_combo_prior.to_parquet(root / stage1["observed_combo_prior"], index=False)

    background_lookup = {
        tuple_key(["gene_symbol", "mutation_key"], row): row for _, row in background_prior.iterrows()
    }
    drug_selected_lookup = {
        tuple_key(["gene_symbol", "mutation_key", "drug_name"], row): row for _, row in drug_selected_prior.iterrows()
    }
    benchmark_drug_lookup = {
        tuple_key(["gene_symbol", "mutation_key", "drug_name"], row): row for _, row in benchmark_drug_prior.iterrows()
    }

    global_rows = []
    benchmark_prior_rows = []
    for _, site in site_contexts.iterrows():
        background_row = background_lookup.get((site["gene_symbol"], site["mutation_key"]))
        tissue_requested = case_tissue_lookup.get((site["gene_symbol"], site["drug_name"]))
        background_probability = float(background_row["P_background"]) if background_row is not None else 0.0
        background_scope = str(background_row["prior_scope"]) if background_row is not None else "missing_background"
        background_tissue = str(background_row["tissue_type"]) if background_row is not None else "Global"
        background_evidence = str(background_row["evidence_tier"]) if background_row is not None else "missing_background"
        background_sources = str(background_row["source_list"]) if background_row is not None else "missing_background"
        tissue_resolution = "global"
        tissue_denominator = None
        if tissue_requested:
            tissue_row = tissue_background_lookup.get((site["gene_symbol"], site["mutation_key"], tissue_requested))
            if tissue_row is not None and tissue_row.get("denominator") is not None and int(tissue_row["denominator"]) >= tissue_min_denominator:
                background_probability = float(tissue_row["P_background"])
                background_scope = "tissue"
                background_tissue = str(tissue_row["tissue_type"])
                background_evidence = str(tissue_row["evidence_tier"])
                background_sources = str(tissue_row["source_list"])
                tissue_resolution = str(tissue_row.get("tissue_resolution", "exact"))
                tissue_denominator = int(tissue_row["denominator"])
            else:
                background_scope = "global_fallback"
                tissue_resolution = "fallback_low_denominator" if tissue_row is not None else "fallback_missing_tissue"
                tissue_denominator = int(tissue_row["denominator"]) if tissue_row is not None and tissue_row.get("denominator") else None

        selected_row = drug_selected_lookup.get((site["gene_symbol"], site["mutation_key"], site["drug_name"]))
        blind_row = benchmark_drug_lookup.get((site["gene_symbol"], site["mutation_key"], site["drug_name"]))
        common = {
            "gene_symbol": site["gene_symbol"],
            "component_mutation": site["component_mutation"],
            "mutation_key": site["mutation_key"],
            "domain_type": site["domain_type"],
            "target_domain": site["target_domain"],
            "drug_name": site["drug_name"],
            "tissue_type": background_tissue,
            "P_background": background_probability,
            "prior_scope": background_scope,
            "background_source_list": background_sources,
            "background_evidence_tier": background_evidence,
            "fitness_formula_version": FITNESS_FORMULA_VERSION,
            "requested_tissue_type": tissue_requested,
            "tissue_resolution": tissue_resolution,
            "tissue_denominator": tissue_denominator,
            "tissue_prior_source": background_scope,
        }
        global_rows.append(
            {
                **common,
                "P_drug_selected": float(selected_row["P_drug_selected"]) if selected_row is not None else 0.0,
                "drug_selected_source_list": str(selected_row["source_list"]) if selected_row is not None else "none",
                "drug_selected_evidence_tier": str(selected_row["evidence_tier"]) if selected_row is not None else "none",
            }
        )
        benchmark_prior_rows.append(
            {
                **common,
                "P_drug_selected": float(blind_row["P_drug_selected"]) if blind_row is not None else 0.0,
                "drug_selected_source_list": str(blind_row["source_list"]) if blind_row is not None else "none",
                "drug_selected_evidence_tier": str(blind_row["evidence_tier"]) if blind_row is not None else "none",
            }
        )

    global_prior = pd.DataFrame.from_records(global_rows).drop_duplicates(subset=["gene_symbol", "mutation_key", "drug_name"])
    benchmark_prior_blind = pd.DataFrame.from_records(benchmark_prior_rows).drop_duplicates(
        subset=["gene_symbol", "mutation_key", "drug_name"]
    )
    global_prior.to_parquet(root / stage1["global_prior"], index=False)
    benchmark_prior_blind.to_parquet(root / stage1["benchmark_prior_blind"], index=False)

    qc = {
        "background_prior_rows": int(len(background_prior)),
        "background_prior_by_tissue_rows": int(len(background_prior_by_tissue)),
        "drug_selected_prior_rows": int(len(drug_selected_prior)),
        "observed_combo_prior_rows": int(len(observed_combo_prior)),
        "global_prior_rows": int(len(global_prior)),
        "benchmark_prior_blind_rows": int(len(benchmark_prior_blind)),
        "viral_epsilon_background_rows": int(background_prior["prior_scope"].eq("epsilon_background").sum()),
        "benchmark_blind_sources": sorted(blind_source_tracker),
        "benchmark_blind_has_mdrdb": "MdrDB" in blind_source_tracker,
        "case_tissue_lookup": {f"{gene}|{drug}": tissue for (gene, drug), tissue in case_tissue_lookup.items()},
        "tissue_min_denominator": tissue_min_denominator,
        "global_prior_source_counts": {
            key: int(value) for key, value in global_prior["tissue_prior_source"].value_counts().sort_index().items()
        },
        "global_prior_used_tissue_rows": int(global_prior["tissue_prior_source"].eq("tissue").sum()),
        "global_prior_used_tissue_rows_below_threshold": int(
            (
                global_prior["tissue_prior_source"].eq("tissue")
                & global_prior["tissue_denominator"].fillna(0).lt(tissue_min_denominator)
            ).sum()
        ),
        "case_tissue_resolution_counts": {
            f"{case_key[0]}|{case_key[1]}": {
                resolution: int(count)
                for resolution, count in subset["tissue_resolution"].value_counts().sort_index().items()
            }
            for case_key, subset in global_prior[
                global_prior["requested_tissue_type"].notna()
            ].groupby(["gene_symbol", "drug_name"], dropna=False)
        },
        "process_peak_rss_mb": round(process_peak_rss_mb(), 2),
    }
    json_dump(root / stage1["mutation_prior_qc"], qc)


if __name__ == "__main__":
    main()
