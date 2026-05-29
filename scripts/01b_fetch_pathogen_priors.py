#!/usr/bin/env python3
"""Build HIV background and drug-selected priors from local GenoRx tables."""

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

from tools.runtime import ensure_dir, json_dump, load_yaml, project_root
from tools.mutation_parser import parse_mutation
from tools.stage1_utils import (
    HIV_RULE_BASED_FALLBACKS,
    accumulate_hiv_mutation_counts,
    build_smiles_drug_alias_map,
    classify_domain_type,
    classify_target_domain,
    hiv_position_columns,
    infer_hiv_reference_map,
    normalize_drug_name,
    resolve_join_gene_symbol,
    split_hiv_treatment_tokens,
    write_long_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def process_peak_rss_mb() -> float:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(max_rss) / (1024.0 * 1024.0)
    return float(max_rss) / 1024.0


def counts_to_frame(
    counts: dict[str, int],
    denominator: int,
    gene_symbol: str,
    target_domain: str,
    evidence_tier: str,
    drug_name: str | None = None,
) -> pd.DataFrame:
    rows = []
    for mutation, count in sorted(counts.items()):
        rows.append(
            {
                "gene_symbol": gene_symbol,
                "component_mutation": mutation,
                "mutation_key": f"{gene_symbol}:{mutation}",
                "domain_type": "viral",
                "target_domain": target_domain,
                "tissue_type": "NA",
                "drug_name": drug_name,
                "count": int(count),
                "denominator": int(denominator),
                "freq": float(count) / float(denominator) if denominator else 0.0,
                "source_db": "HIVDB_GenoRx",
                "evidence_tier": evidence_tier,
            }
        )
    return pd.DataFrame.from_records(rows)


def apply_rule_based_fallback(df: pd.DataFrame, denominator: int, domain: str, drug_name: str) -> tuple[pd.DataFrame, list[str]]:
    fallback_mutations = HIV_RULE_BASED_FALLBACKS.get((domain, drug_name), [])
    if not fallback_mutations:
        return df, []
    existing = set(df["component_mutation"].dropna().astype(str)) if not df.empty else set()
    added = []
    rows = []
    effective_denominator = max(int(denominator), 1)
    for mutation in fallback_mutations:
        if mutation in existing:
            continue
        added.append(mutation)
        rows.append(
            {
                "gene_symbol": resolve_join_gene_symbol("GAG-POL", "viral", domain),
                "component_mutation": mutation,
                "mutation_key": f"{resolve_join_gene_symbol('GAG-POL', 'viral', domain)}:{mutation}",
                "domain_type": "viral",
                "target_domain": domain,
                "tissue_type": "NA",
                "drug_name": drug_name,
                "count": 1,
                "denominator": effective_denominator,
                "freq": max(1.0 / float(effective_denominator), 0.01),
                "source_db": "HIVDB_rule_fallback",
                "evidence_tier": "rule_based_fallback",
            }
        )
    if rows:
        df = pd.concat([df, pd.DataFrame.from_records(rows)], ignore_index=True, sort=False)
    return df, added


def build_mdrdb_reference_overrides(root: Path, stage1: dict) -> dict[str, dict[int, str]]:
    frame = pd.read_csv(
        root / stage1["mdrdb_main"],
        sep="\t",
        usecols=["DRUG", "SMILES", "PROTEIN_NAME", "FDA_MECHANISM", "DRUG_CLASSES", "MUTATION"],
        low_memory=False,
    )
    alias_map = build_smiles_drug_alias_map(frame[["SMILES", "DRUG"]])
    normalized = [normalize_drug_name(raw, smiles, alias_map) for raw, smiles in zip(frame["DRUG"], frame["SMILES"])]
    frame["drug_name"] = [item[0] for item in normalized]
    frame["domain_type"] = frame.apply(classify_domain_type, axis=1)
    frame = frame[frame["domain_type"].eq("viral")].copy()
    frame["target_domain"] = frame.apply(classify_target_domain, axis=1)

    counters: dict[str, dict[int, defaultdict[str, int]]] = {
        "RT": defaultdict(lambda: defaultdict(int)),
        "PR": defaultdict(lambda: defaultdict(int)),
        "IN": defaultdict(lambda: defaultdict(int)),
    }
    for mutation_text, domain in zip(frame["MUTATION"], frame["target_domain"]):
        if domain not in counters:
            continue
        parsed = parse_mutation(mutation_text)
        for component in parsed["parsed_components"]:
            if component.start_pos is None or not component.ref_aa:
                continue
            counters[domain][component.start_pos][component.ref_aa] += 1

    reference_overrides: dict[str, dict[int, str]] = {}
    for domain, by_position in counters.items():
        reference_overrides[domain] = {
            position: sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
            for position, counter in by_position.items()
            if counter
        }
    return reference_overrides


def build_domain_tables(
    path: Path,
    list_column: str,
    target_domain: str,
    reference_overrides: dict[str, dict[int, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(path, sep="\t", low_memory=False)
    position_columns = hiv_position_columns(raw)
    untreated = raw[raw[list_column].isna()].copy()
    reference_source = untreated if not untreated.empty else raw
    reference_map = infer_hiv_reference_map(reference_source, position_columns)
    reference_map.update(reference_overrides.get(target_domain, {}))
    gene_symbol = resolve_join_gene_symbol("GAG-POL", "viral", target_domain)

    background_counts = accumulate_hiv_mutation_counts(untreated if not untreated.empty else raw, position_columns, reference_map)
    background_tier = "genoRx_background" if not untreated.empty else "genoRx_background_all_isolates"
    background = counts_to_frame(background_counts, len(untreated if not untreated.empty else raw), gene_symbol, target_domain, background_tier)

    treatment_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, value in raw[list_column].dropna().items():
        for drug_name in split_hiv_treatment_tokens(value):
            treatment_to_indices[drug_name].append(index)

    selected_frames = []
    fallback_summary: dict[str, list[str]] = {}
    for drug_name, row_indices in sorted(treatment_to_indices.items()):
        selected = raw.loc[row_indices].copy()
        counts = accumulate_hiv_mutation_counts(selected, position_columns, reference_map)
        frame = counts_to_frame(counts, len(selected), gene_symbol, target_domain, "genoRx_drug_selected", drug_name=drug_name)
        frame, added = apply_rule_based_fallback(frame, len(selected), target_domain, drug_name)
        if added:
            fallback_summary[drug_name] = added
        selected_frames.append(frame)

    drug_selected = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    qc = {
        "rows": int(len(raw)),
        "background_rows": int(len(untreated)),
        "distinct_drug_contexts": int(len(treatment_to_indices)),
        "inferred_reference_positions": {
            str(position): reference_map[position]
            for position in sorted(reference_map)
            if position in {48, 54, 82, 90, 92, 101, 106, 138, 181, 188, 190, 227}
        },
        "fallback_summary": fallback_summary,
    }
    return background, drug_selected, qc


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage1 = config["stage1"]

    ensure_dir(root / stage1["tables_root"])
    reference_overrides = build_mdrdb_reference_overrides(root, stage1)

    domain_specs = [
        ("RT", root / stage1["hiv_rt_raw"], "RTIList"),
        ("PR", root / stage1["hiv_pr_raw"], "PIList"),
        ("IN", root / stage1["hiv_in_raw"], "INIList"),
    ]

    background_frames = []
    drug_selected_frames = []
    qc_payload = {}
    for domain, path, treatment_column in domain_specs:
        background, drug_selected, domain_qc = build_domain_tables(path, treatment_column, domain, reference_overrides)
        background_frames.append(background)
        drug_selected_frames.append(drug_selected)
        qc_payload[domain] = domain_qc

    hiv_prevalence = pd.concat(background_frames, ignore_index=True, sort=False)
    hiv_drug_selected = pd.concat(drug_selected_frames, ignore_index=True, sort=False)

    hiv_prevalence.to_parquet(root / stage1["hiv_prevalence_prior"], index=False)
    hiv_drug_selected.to_parquet(root / stage1["hiv_drug_selected_prior"], index=False)
    write_long_tsv(hiv_prevalence, root / stage1["hiv_prevalence_cache_tsv"])
    write_long_tsv(hiv_drug_selected, root / stage1["hiv_drug_selected_cache_tsv"])

    rpv_key_sites = {"K101E", "K101H", "E138A", "E138G", "E138K", "E138Q", "Y181C", "Y181I", "G190A", "G190S", "F227C"}
    observed_rpv = set(
        hiv_drug_selected[
            hiv_drug_selected["target_domain"].eq("RT") & hiv_drug_selected["drug_name"].eq("Rilpivirine")
        ]["component_mutation"].dropna().astype(str)
    )
    qc_payload["summary"] = {
        "background_row_count": int(len(hiv_prevalence)),
        "drug_selected_row_count": int(len(hiv_drug_selected)),
        "rilpivirine_key_site_coverage": sorted(observed_rpv & rpv_key_sites),
        "rilpivirine_missing_key_sites": sorted(rpv_key_sites - observed_rpv),
        "process_peak_rss_mb": round(process_peak_rss_mb(), 2),
    }
    json_dump(root / stage1["hiv_prior_qc"], qc_payload)


if __name__ == "__main__":
    main()
