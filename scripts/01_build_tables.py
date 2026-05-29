#!/usr/bin/env python3
"""Build the Stage 1 master mutation table and standard frequency maps."""

from __future__ import annotations

import argparse
import gzip
import resource
import sys
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mutation_parser import combination_key, component_to_mutation_key, parse_mutation
from tools.runtime import ensure_dir, json_dump, load_yaml, project_root
from tools.stage1_utils import (
    build_smiles_drug_alias_map,
    classify_domain_type,
    classify_target_domain,
    evaluation_unit_from_type,
    fetch_uniprot_gene_map,
    is_human_readable_drug,
    normalize_drug_name,
    normalize_protein_change,
    read_single_member_zip_tsv,
    resolve_join_gene_symbol,
    standardize_gene_symbol,
    write_schema_json,
)

FITNESS_FORMULA_VERSION = "resistagent_default_v1"
MDRDB_CORE_COLUMNS = {
    "SAMPLE_ID",
    "VERSION",
    "DATASET",
    "TYPE",
    "UNIPROT_ID",
    "PDB_ID",
    "MUTATION",
    "CHAIN_ID",
    "DRUG",
    "SMILES",
    "PROTEIN_NAME",
    "PROTEIN_FAMILY",
    "PROTEIN_SUPERFAMILY",
    "PROTEIN_DOMAIN",
    "CID",
    "FDA_MECHANISM",
    "MESH",
    "DRUG_CLASSES",
    "DDG.EXP",
    "SAMPLE_SOURCE",
    "MUTATION_SOURCE",
    "DRUG_POSE_SOURCE",
}
MDRDB_PREFIXES = ("PLIP.", "vina", "LIG.", "MUT.", "ENV.")
MDRDB_DTYPE = {
    "SAMPLE_ID": "string",
    "VERSION": "string",
    "DATASET": "string",
    "TYPE": "string",
    "UNIPROT_ID": "string",
    "PDB_ID": "string",
    "MUTATION": "string",
    "CHAIN_ID": "string",
    "DRUG": "string",
    "SMILES": "string",
    "PROTEIN_NAME": "string",
    "PROTEIN_FAMILY": "string",
    "PROTEIN_SUPERFAMILY": "string",
    "PROTEIN_DOMAIN": "string",
    "CID": "string",
    "FDA_MECHANISM": "string",
    "MESH": "string",
    "DRUG_CLASSES": "string",
    "SAMPLE_SOURCE": "string",
    "MUTATION_SOURCE": "string",
    "DRUG_POSE_SOURCE": "string",
}
COSMIC_USECOLS = ["SAMPLE_NAME", "GENE_SYMBOL", "DRUG_NAME", "MUTATION_AA", "HGVSP"]
COSMIC_DTYPE = {
    "SAMPLE_NAME": "string",
    "GENE_SYMBOL": "string",
    "DRUG_NAME": "string",
    "MUTATION_AA": "string",
    "HGVSP": "string",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def process_peak_rss_mb() -> float:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(max_rss) / (1024.0 * 1024.0)
    return float(max_rss) / 1024.0


def standardize_drug_for_output(value: str | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text == "nan":
        return None
    return text if is_human_readable_drug(text) else text.upper()


def mode_or_default(series: pd.Series, default: str = "unknown") -> str:
    counts = series.dropna().astype(str).value_counts()
    if counts.empty:
        return default
    return str(counts.index[0])


def selected_mdrdb_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    return [
        column
        for column in header
        if column in MDRDB_CORE_COLUMNS or any(column.startswith(prefix) for prefix in MDRDB_PREFIXES)
    ]


def load_mdrdb_main(path: Path, chunksize: int) -> pd.DataFrame:
    usecols = selected_mdrdb_columns(path)
    chunks = []
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        dtype={key: value for key, value in MDRDB_DTYPE.items() if key in usecols},
        low_memory=False,
        chunksize=chunksize,
    ):
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True)


def load_master_inputs(root: Path, stage1: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mdrdb = load_mdrdb_main(root / stage1["mdrdb_main"], int(stage1["mdrdb_chunksize"]))
    drug_annotation = read_single_member_zip_tsv(root / stage1["mdrdb_drug_annotation_zip"])
    protein_annotation = read_single_member_zip_tsv(root / stage1["mdrdb_protein_annotation_zip"])
    return mdrdb, drug_annotation, protein_annotation


def enrich_annotations(master: pd.DataFrame, drug_annotation: pd.DataFrame, protein_annotation: pd.DataFrame) -> pd.DataFrame:
    frame = master.copy()
    frame["__drug_key"] = frame["DRUG"].astype(str)
    frame["__smiles_key"] = frame["SMILES"].fillna("").astype(str)

    drug_ann = drug_annotation.copy()
    drug_ann["__drug_key"] = drug_ann["DRUG"].astype(str)
    drug_ann["__smiles_key"] = drug_ann["SMILES"].fillna("").astype(str)
    drug_ann = drug_ann.rename(columns={column: f"{column}_ann" for column in ["CID", "FDA_MECHANISM", "MESH", "DRUG_CLASSES"]})
    frame = frame.merge(
        drug_ann[["__drug_key", "__smiles_key", "CID_ann", "FDA_MECHANISM_ann", "MESH_ann", "DRUG_CLASSES_ann"]],
        on=["__drug_key", "__smiles_key"],
        how="left",
    )

    protein_ann = protein_annotation.rename(
        columns={
            "PROTEIN_NAME": "PROTEIN_NAME_ann",
            "PROTEIN_FAMILY": "PROTEIN_FAMILY_ann",
            "PROTEIN_SUPERFAMILY": "PROTEIN_SUPERFAMILY_ann",
            "PROTEIN_DOMAIN": "PROTEIN_DOMAIN_ann",
        }
    )
    frame = frame.merge(
        protein_ann[
            [
                "UNIPROT_ID",
                "PROTEIN_NAME_ann",
                "PROTEIN_FAMILY_ann",
                "PROTEIN_SUPERFAMILY_ann",
                "PROTEIN_DOMAIN_ann",
            ]
        ],
        on="UNIPROT_ID",
        how="left",
    )

    for column in ["CID", "FDA_MECHANISM", "MESH", "DRUG_CLASSES"]:
        frame[column] = frame[column].combine_first(frame[f"{column}_ann"])
        frame = frame.drop(columns=[f"{column}_ann"])
    for column in ["PROTEIN_NAME", "PROTEIN_FAMILY", "PROTEIN_SUPERFAMILY", "PROTEIN_DOMAIN"]:
        frame[column] = frame[column].combine_first(frame[f"{column}_ann"])
        frame = frame.drop(columns=[f"{column}_ann"])

    return frame.drop(columns=["__drug_key", "__smiles_key"])


def build_master_table(root: Path, stage1: dict) -> tuple[pd.DataFrame, dict]:
    mdrdb, drug_annotation, protein_annotation = load_master_inputs(root, stage1)
    mdrdb = enrich_annotations(mdrdb, drug_annotation, protein_annotation)

    alias_map = build_smiles_drug_alias_map(mdrdb[["SMILES", "DRUG"]])
    normalized_drugs = [normalize_drug_name(raw, smiles, alias_map) for raw, smiles in zip(mdrdb["DRUG"], mdrdb["SMILES"])]
    mdrdb["drug_name"] = [item[0] for item in normalized_drugs]
    mdrdb["drug_name_source"] = [item[1] for item in normalized_drugs]
    mdrdb["drug_name"] = mdrdb["drug_name"].map(standardize_drug_for_output)

    gene_map = fetch_uniprot_gene_map(mdrdb["UNIPROT_ID"].dropna().astype(str).unique(), root / stage1["uniprot_gene_cache"])
    mdrdb = mdrdb.merge(gene_map, left_on="UNIPROT_ID", right_on="uniprot_id", how="left")
    mdrdb["gene_symbol_base"] = mdrdb["gene_symbol"].map(standardize_gene_symbol)

    mdrdb["domain_type"] = mdrdb.apply(classify_domain_type, axis=1)
    mdrdb["target_domain"] = mdrdb.apply(classify_target_domain, axis=1)
    mdrdb["gene_symbol"] = [
        resolve_join_gene_symbol(symbol, domain_type, target_domain)
        for symbol, domain_type, target_domain in zip(mdrdb["gene_symbol_base"], mdrdb["domain_type"], mdrdb["target_domain"])
    ]

    parsed_mutation_types = []
    component_mutations = []
    component_mutation_keys = []
    combination_keys = []
    combination_sizes = []
    mutation_keys = []
    mutation_parse_ok = []

    for raw_mutation, gene_symbol in zip(mdrdb["MUTATION"], mdrdb["gene_symbol"]):
        parsed = parse_mutation(raw_mutation)
        parsed_mutation_types.append(parsed["parsed_mutation_type"])
        component_mutations.append(parsed["component_mutations"])
        parsed_components = parsed["parsed_components"]
        keys = [component_to_mutation_key(gene_symbol, component) for component in parsed_components]
        keys = [value for value in keys if value]
        component_mutation_keys.append(keys)
        combination_keys.append(combination_key(gene_symbol, parsed_components))
        combination_sizes.append(parsed["combination_size"])
        mutation_keys.append(keys[0] if len(keys) == 1 else None)
        mutation_parse_ok.append(bool(parsed["mutation_parse_ok"]))

    mdrdb["parsed_mutation_type"] = parsed_mutation_types
    mdrdb["component_mutations"] = component_mutations
    mdrdb["component_mutation_keys"] = component_mutation_keys
    mdrdb["combination_key"] = combination_keys
    mdrdb["combination_size"] = combination_sizes
    mdrdb["mutation_key"] = mutation_keys
    mdrdb["mutation_parse_ok"] = mutation_parse_ok
    mdrdb["evaluation_unit"] = mdrdb["TYPE"].map(evaluation_unit_from_type)
    mdrdb["has_observed_combo"] = (mdrdb["combination_size"] > 1) & mdrdb["evaluation_unit"].eq("observed_combo")
    mdrdb["fitness_formula_version"] = FITNESS_FORMULA_VERSION
    mdrdb["source_db"] = "MdrDB"
    mdrdb["tissue_type"] = mdrdb["domain_type"].map(lambda value: "NA" if value == "viral" else "Global")

    master_table = mdrdb.drop(columns=["uniprot_id"]).copy()
    qc = {
        "row_count": int(len(master_table)),
        "sample_id_unique": bool(master_table["SAMPLE_ID"].is_unique),
        "mutation_parse_ok_ratio": float(master_table["mutation_parse_ok"].mean()),
        "missing_gene_symbol_rows": int(master_table["gene_symbol"].isna().sum()),
        "raw_code_drug_rows": int(master_table["drug_name_source"].eq("raw_code").sum()),
        "domain_type_counts": {key: int(value) for key, value in master_table["domain_type"].value_counts().to_dict().items()},
        "evaluation_unit_counts": {key: int(value) for key, value in master_table["evaluation_unit"].value_counts().to_dict().items()},
    }
    return master_table, qc


def build_mdrdb_frequency_map(master_table: pd.DataFrame) -> pd.DataFrame:
    master_table = master_table[master_table["combination_size"].eq(1)].copy()
    records = []
    for row in master_table.itertuples(index=False):
        sample_id = row.SAMPLE_ID
        gene_symbol = row.gene_symbol
        for mutation, mutation_key in zip(row.component_mutations, row.component_mutation_keys):
            records.append(
                {
                    "sample_id": sample_id,
                    "gene_symbol": gene_symbol,
                    "component_mutation": mutation,
                    "mutation_key": mutation_key,
                    "domain_type": row.domain_type,
                    "target_domain": row.target_domain,
                    "tissue_type": row.tissue_type,
                    "drug_name": row.drug_name,
                    "observation_unit": row.evaluation_unit,
                }
            )
    exploded = pd.DataFrame.from_records(records)
    if exploded.empty:
        return exploded

    grouped = (
        exploded.groupby(
            [
                "gene_symbol",
                "component_mutation",
                "mutation_key",
                "domain_type",
                "target_domain",
                "tissue_type",
                "drug_name",
                "observation_unit",
            ],
            dropna=False,
        )
        .agg(count=("sample_id", "nunique"))
        .reset_index()
    )
    denominators = (
        exploded.groupby(["gene_symbol", "drug_name"], dropna=False)
        .agg(denominator=("sample_id", "nunique"))
        .reset_index()
    )
    grouped = grouped.merge(denominators, on=["gene_symbol", "drug_name"], how="left")
    grouped["freq"] = grouped["count"] / grouped["denominator"]
    grouped["source_db"] = "MdrDB"
    grouped["prior_scope"] = "gene_drug"
    grouped["context_scope"] = "gene_drug"
    grouped["evidence_tier"] = "observed_same_drug"
    return grouped


def build_cosmic_frequency_map(root: Path, stage1: dict, target_genes: set[str], target_domain_lookup: dict[str, str]) -> pd.DataFrame:
    path = root / stage1["cosmic_resistance_tar"]
    chunks = []
    with pd.option_context("mode.copy_on_write", True):
        with tarfile.open(path, "r") as archive:
            member = next(member for member in archive.getmembers() if member.isfile() and member.name.endswith(".tsv.gz"))
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Unable to extract {member.name} from {path}")
            with gzip.open(extracted, "rt", encoding="utf-8", errors="replace") as handle:
                for chunk in pd.read_csv(
                    handle,
                    sep="\t",
                    usecols=COSMIC_USECOLS,
                    dtype=COSMIC_DTYPE,
                    low_memory=False,
                    chunksize=int(stage1["cosmic_chunksize"]),
                ):
                    chunk["gene_symbol"] = chunk["GENE_SYMBOL"].map(standardize_gene_symbol)
                    chunk = chunk[chunk["gene_symbol"].isin(target_genes)].copy()
                    if chunk.empty:
                        continue
                    chunk["component_mutation"] = chunk["MUTATION_AA"].combine_first(chunk["HGVSP"]).map(normalize_protein_change)
                    chunk = chunk[chunk["component_mutation"].notna()].copy()
                    if chunk.empty:
                        continue
                    chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()
    cosmic = pd.concat(chunks, ignore_index=True)
    cosmic["gene_symbol"] = cosmic["GENE_SYMBOL"].map(standardize_gene_symbol)
    cosmic["mutation_key"] = cosmic["gene_symbol"] + ":" + cosmic["component_mutation"]
    cosmic["drug_name"] = cosmic["DRUG_NAME"].map(standardize_drug_for_output)
    cosmic["domain_type"] = "cancer"
    cosmic["target_domain"] = cosmic["gene_symbol"].map(target_domain_lookup).fillna("unknown")
    cosmic["tissue_type"] = "Global"

    global_counts = (
        cosmic.groupby(["gene_symbol", "component_mutation", "mutation_key", "domain_type", "target_domain", "tissue_type"])
        .agg(count=("SAMPLE_NAME", "nunique"))
        .reset_index()
    )
    global_denominators = cosmic.groupby("gene_symbol").agg(denominator=("SAMPLE_NAME", "nunique")).reset_index()
    global_counts = global_counts.merge(global_denominators, on="gene_symbol", how="left")
    global_counts["freq"] = global_counts["count"] / global_counts["denominator"]
    global_counts["drug_name"] = pd.NA
    global_counts["source_db"] = "COSMIC"
    global_counts["prior_scope"] = "global"
    global_counts["context_scope"] = "global"
    global_counts["evidence_tier"] = "clinical_resistance_catalog"

    drug_counts = (
        cosmic.dropna(subset=["drug_name"])
        .groupby(["gene_symbol", "component_mutation", "mutation_key", "domain_type", "target_domain", "tissue_type", "drug_name"])
        .agg(count=("SAMPLE_NAME", "nunique"))
        .reset_index()
    )
    if drug_counts.empty:
        return global_counts

    drug_denominators = (
        cosmic.dropna(subset=["drug_name"])
        .groupby(["gene_symbol", "drug_name"])
        .agg(denominator=("SAMPLE_NAME", "nunique"))
        .reset_index()
    )
    drug_counts = drug_counts.merge(drug_denominators, on=["gene_symbol", "drug_name"], how="left")
    drug_counts["freq"] = drug_counts["count"] / drug_counts["denominator"]
    drug_counts["source_db"] = "COSMIC"
    drug_counts["prior_scope"] = "gene_drug"
    drug_counts["context_scope"] = "gene_drug"
    drug_counts["evidence_tier"] = "clinical_drug_selected"

    return pd.concat([global_counts, drug_counts], ignore_index=True, sort=False)


def build_depmap_frequency_map(root: Path, stage1: dict, target_genes: set[str], target_domain_lookup: dict[str, str]) -> pd.DataFrame:
    model_metadata = pd.read_csv(
        root / stage1["depmap_model_csv"],
        usecols=["ModelID", "OncotreeLineage"],
        dtype={"ModelID": "string", "OncotreeLineage": "string"},
        low_memory=False,
    ).drop_duplicates(subset=["ModelID"])
    lineage_lookup = model_metadata.set_index("ModelID")["OncotreeLineage"]
    lineage_denominators = model_metadata.dropna(subset=["OncotreeLineage"]).groupby("OncotreeLineage").agg(denominator=("ModelID", "nunique"))
    global_denominator = int(model_metadata["ModelID"].nunique())

    chunks = []
    usecols = ["ModelID", "HugoSymbol", "ProteinChange"]
    for chunk in pd.read_csv(
        root / stage1["depmap_mutations_csv"],
        usecols=usecols,
        dtype={"ModelID": "string", "HugoSymbol": "string", "ProteinChange": "string"},
        low_memory=False,
        chunksize=int(stage1["depmap_chunksize"]),
    ):
        chunk = chunk[chunk["HugoSymbol"].isin(target_genes) & chunk["ProteinChange"].notna()].copy()
        if chunk.empty:
            continue
        chunk["gene_symbol"] = chunk["HugoSymbol"].map(standardize_gene_symbol)
        chunk["component_mutation"] = chunk["ProteinChange"].map(normalize_protein_change)
        chunk = chunk[chunk["component_mutation"].notna()].copy()
        if chunk.empty:
            continue
        chunk["mutation_key"] = chunk["gene_symbol"] + ":" + chunk["component_mutation"]
        chunk["tissue_type"] = chunk["ModelID"].map(lineage_lookup)
        chunk = chunk[["ModelID", "gene_symbol", "component_mutation", "mutation_key", "tissue_type"]].drop_duplicates()
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    depmap = pd.concat(chunks, ignore_index=True)
    depmap["domain_type"] = "cancer"
    depmap["target_domain"] = depmap["gene_symbol"].map(target_domain_lookup).fillna("unknown")

    global_counts = (
        depmap.groupby(["gene_symbol", "component_mutation", "mutation_key", "domain_type", "target_domain"])
        .agg(count=("ModelID", "nunique"))
        .reset_index()
    )
    global_counts["tissue_type"] = "Global"
    global_counts["denominator"] = global_denominator
    global_counts["freq"] = global_counts["count"] / global_counts["denominator"]
    global_counts["drug_name"] = pd.NA
    global_counts["source_db"] = "DepMap"
    global_counts["prior_scope"] = "global"
    global_counts["context_scope"] = "global"
    global_counts["evidence_tier"] = "cell_line_background"

    tissue_counts = (
        depmap.dropna(subset=["tissue_type"])
        .groupby(["gene_symbol", "component_mutation", "mutation_key", "domain_type", "target_domain", "tissue_type"])
        .agg(count=("ModelID", "nunique"))
        .reset_index()
    )
    if tissue_counts.empty:
        return global_counts

    tissue_counts = tissue_counts.merge(lineage_denominators, left_on="tissue_type", right_index=True, how="left")
    tissue_counts["freq"] = tissue_counts["count"] / tissue_counts["denominator"]
    tissue_counts["drug_name"] = pd.NA
    tissue_counts["source_db"] = "DepMap"
    tissue_counts["prior_scope"] = "tissue"
    tissue_counts["context_scope"] = "tissue"
    tissue_counts["evidence_tier"] = "cell_line_background"

    return pd.concat([global_counts, tissue_counts], ignore_index=True, sort=False)


def build_combination_map(master_table: pd.DataFrame) -> pd.DataFrame:
    combination_rows = master_table[master_table["combination_size"] > 1].copy()
    if combination_rows.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "gene_symbol",
                "drug_name",
                "combination_key",
                "component_mutations",
                "combination_size",
                "observed_count",
                "domain_type",
                "target_domain",
                "evaluation_unit",
            ]
        )

    observed_counts = (
        combination_rows.groupby(["gene_symbol", "drug_name", "combination_key"], dropna=False)
        .agg(observed_count=("SAMPLE_ID", "nunique"))
        .reset_index()
    )
    combination_map = combination_rows[
        [
            "SAMPLE_ID",
            "gene_symbol",
            "drug_name",
            "combination_key",
            "component_mutations",
            "combination_size",
            "domain_type",
            "target_domain",
            "evaluation_unit",
        ]
    ].rename(columns={"SAMPLE_ID": "sample_id"})
    combination_map = combination_map.merge(observed_counts, on=["gene_symbol", "drug_name", "combination_key"], how="left")
    return combination_map


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage1 = config["stage1"]

    ensure_dir(root / stage1["tables_root"])
    master_table, qc = build_master_table(root, stage1)
    master_table.to_parquet(root / stage1["master_table"], index=False)
    write_schema_json(master_table, root / stage1["master_schema"])

    combination_map = build_combination_map(master_table)
    combination_map.to_parquet(root / stage1["mutation_combination_map"], index=False)

    target_genes = {
        gene_symbol
        for gene_symbol, domain_type in zip(master_table["gene_symbol"], master_table["domain_type"])
        if gene_symbol and domain_type != "viral"
    }
    target_domain_lookup = (
        master_table.dropna(subset=["gene_symbol"])
        .groupby("gene_symbol")
        .agg(target_domain=("target_domain", lambda series: mode_or_default(series, "unknown")))
        ["target_domain"]
        .to_dict()
    )

    mdrdb_frequency = build_mdrdb_frequency_map(master_table)
    cosmic_frequency = build_cosmic_frequency_map(root, stage1, target_genes, target_domain_lookup)
    depmap_frequency = build_depmap_frequency_map(root, stage1, target_genes, target_domain_lookup)
    mutation_frequency_map = pd.concat([mdrdb_frequency, cosmic_frequency, depmap_frequency], ignore_index=True, sort=False)
    mutation_frequency_map.to_parquet(root / stage1["mutation_frequency_map"], index=False)

    rilpivirine_combo_rows = int(
        combination_map[
            combination_map["gene_symbol"].eq("GAG-POL_RT") & combination_map["drug_name"].eq("Rilpivirine")
        ].shape[0]
    )
    rilpivirine_combo_total = int(
        master_table[
            master_table["gene_symbol"].eq("GAG-POL_RT")
            & master_table["drug_name"].eq("Rilpivirine")
            & master_table["combination_size"].gt(1)
        ].shape[0]
    )
    qc.update(
        {
            "master_table_path": stage1["master_table"],
            "mutation_frequency_map_path": stage1["mutation_frequency_map"],
            "mutation_combination_map_path": stage1["mutation_combination_map"],
            "read_strategy": {
                "mdrdb_usecols_count": len(selected_mdrdb_columns(root / stage1["mdrdb_main"])),
                "mdrdb_chunksize": int(stage1["mdrdb_chunksize"]),
                "cosmic_chunksize": int(stage1["cosmic_chunksize"]),
                "depmap_chunksize": int(stage1["depmap_chunksize"]),
            },
            "frequency_rows_by_source": {
                key: int(value) for key, value in mutation_frequency_map["source_db"].value_counts().to_dict().items()
            },
            "mdrdb_gene_drug_site_rows": int(
                (
                    mutation_frequency_map["source_db"].eq("MdrDB")
                    & mutation_frequency_map["context_scope"].astype(str).eq("gene_drug")
                    & mutation_frequency_map["observation_unit"].astype(str).eq("site")
                ).sum()
            ),
            "mdrdb_gene_drug_observed_combo_rows": int(
                (
                    mutation_frequency_map["source_db"].eq("MdrDB")
                    & mutation_frequency_map["context_scope"].astype(str).eq("gene_drug")
                    & mutation_frequency_map["observation_unit"].astype(str).eq("observed_combo")
                ).sum()
            ),
            "mdrdb_gene_drug_indel_rows": int(
                (
                    mutation_frequency_map["source_db"].eq("MdrDB")
                    & mutation_frequency_map["context_scope"].astype(str).eq("gene_drug")
                    & mutation_frequency_map["observation_unit"].astype(str).eq("indel")
                ).sum()
            ),
            "combination_row_count": int(len(combination_map)),
            "rilpivirine_combo_rows": rilpivirine_combo_rows,
            "rilpivirine_combo_total": rilpivirine_combo_total,
            "rilpivirine_combo_coverage": float(rilpivirine_combo_rows / rilpivirine_combo_total)
            if rilpivirine_combo_total
            else None,
            "target_gene_count_for_depmap": int(len(target_genes)),
            "process_peak_rss_mb": round(process_peak_rss_mb(), 2),
        }
    )
    json_dump(root / stage1["stage1_build_qc"], qc)


if __name__ == "__main__":
    main()
