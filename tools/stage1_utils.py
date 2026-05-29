#!/usr/bin/env python3
"""Shared helpers for Stage 1 table and prior construction."""

from __future__ import annotations

import io
import json
import re
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from tools.runtime import ensure_dir

AA3_TO_1 = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Ter": "*",
}

HIV_GENERIC_TOKENS = {
    "NONE",
    "UNKNOWN",
    "RTI",
    "NRTI",
    "NNRTI",
    "PI",
    "INI",
}

HIV_DRUG_ALIASES = {
    "3TC": "Lamivudine",
    "ABC": "Abacavir",
    "ADV": "Adefovir",
    "APV": "Amprenavir",
    "AZT": "Zidovudine",
    "D4T": "Stavudine",
    "DDC": "Zalcitabine",
    "DDI": "Didanosine",
    "DLV": "Delavirdine",
    "DRV": "Darunavir",
    "DTG": "Dolutegravir",
    "EFV": "Efavirenz",
    "ETR": "Etravirine",
    "EVG": "Elvitegravir",
    "FTC": "Emtricitabine",
    "IDV": "Indinavir",
    "LPV": "Lopinavir",
    "NFV": "Nelfinavir",
    "NVP": "Nevirapine",
    "RAL": "Raltegravir",
    "RPV": "Rilpivirine",
    "RTV": "Ritonavir",
    "SQV": "Saquinavir",
    "TDF": "Tenofovir",
}

HIV_RULE_BASED_FALLBACKS = {
    ("RT", "Rilpivirine"): [
        "K101E",
        "K101H",
        "E138A",
        "E138G",
        "E138K",
        "E138Q",
        "V106M",
        "V179D",
        "Y181C",
        "Y181I",
        "G190A",
        "G190S",
        "F227C",
    ],
}

AA_TOKEN_RE = r"(?:[A-Z][a-z]{2}|[A-Z\*])"
SIMPLE_HGVSP_RE = re.compile(rf"^p\.({AA_TOKEN_RE})(\d+)({AA_TOKEN_RE}|=|\?)$")
DELINS_HGVSP_RE = re.compile(rf"^p\.({AA_TOKEN_RE})(\d+)(?:_({AA_TOKEN_RE})(\d+))?delins([A-Za-z\*]+)$")
DELETION_HGVSP_RE = re.compile(rf"^p\.({AA_TOKEN_RE})(\d+)(?:_({AA_TOKEN_RE})(\d+))?del([A-Za-z\*]+)?$")
INSERTION_HGVSP_RE = re.compile(rf"^p\.({AA_TOKEN_RE})(\d+)_({AA_TOKEN_RE})(\d+)ins([A-Za-z\*]+)$")


def read_single_member_zip_tsv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one file in {path}, found {members}")
        with archive.open(members[0]) as handle:
            return pd.read_csv(handle, sep="\t", low_memory=False)


def read_single_member_tar_tsvgz(path: Path) -> pd.DataFrame:
    with tarfile.open(path, "r") as archive:
        members = [member for member in archive.getmembers() if member.isfile() and member.name.endswith(".tsv.gz")]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one .tsv.gz file in {path}, found {[m.name for m in members]}")
        extracted = archive.extractfile(members[0])
        if extracted is None:
            raise ValueError(f"Unable to extract {members[0].name} from {path}")
        buffer = io.BytesIO(extracted.read())
    return pd.read_csv(buffer, sep="\t", compression="gzip", low_memory=False)


def write_schema_json(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    schema = {
        "row_count": int(len(df)),
        "columns": [
            {
                "name": column,
                "dtype": str(df[column].dtype),
                "non_null_count": int(df[column].notna().sum()),
                "null_count": int(df[column].isna().sum()),
            }
            for column in df.columns
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def is_human_readable_drug(token: str) -> bool:
    text = str(token)
    return any(char.islower() for char in text) or " " in text or "-" in text


def build_smiles_drug_alias_map(df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for _, row in df[["SMILES", "DRUG"]].dropna().iterrows():
        smiles = str(row["SMILES"]).strip()
        drug = str(row["DRUG"]).strip()
        if smiles and is_human_readable_drug(drug):
            grouped[smiles][drug] += 1
    for smiles, counter in grouped.items():
        mapping[smiles] = counter.most_common(1)[0][0]
    return mapping


def normalize_drug_name(raw_drug: str, smiles: str | None, alias_map: dict[str, str]) -> tuple[str, str]:
    drug = str(raw_drug).strip()
    if is_human_readable_drug(drug):
        return drug, "raw_name"
    if smiles:
        normalized = alias_map.get(str(smiles).strip())
        if normalized:
            return normalized, "smiles_alias"
    return drug, "raw_code"


def standardize_gene_symbol(symbol: str | None) -> str | None:
    if symbol is None or pd.isna(symbol):
        return None
    text = str(symbol).strip()
    if not text or text == "nan":
        return None
    return text.upper()


def resolve_join_gene_symbol(symbol: str | None, domain_type: str, target_domain: str) -> str | None:
    base_symbol = standardize_gene_symbol(symbol)
    if not base_symbol:
        return None
    normalized = base_symbol.replace("_", "-")
    if domain_type == "viral":
        viral_base = "GAG-POL" if normalized in {"GAG-POL", "GAGPOL", "POL"} else normalized
        if target_domain in {"RT", "PR", "IN"}:
            return f"{viral_base}_{target_domain}"
        return viral_base
    return normalized


def fetch_uniprot_gene_map(uniprot_ids: Iterable[str], cache_path: Path) -> pd.DataFrame:
    ids = sorted({value for value in uniprot_ids if value and value != "UNDEFINED"})
    ensure_dir(cache_path.parent)
    if cache_path.exists():
        cached = pd.read_csv(cache_path, sep="\t", dtype=str)
    else:
        cached = pd.DataFrame(columns=["Entry", "Gene Names (primary)", "Protein names"])
    cached_ids = set(cached.get("Entry", pd.Series(dtype=str)).dropna().astype(str))
    missing = [value for value in ids if value not in cached_ids]
    if missing:
        session = requests.Session()
        rows: list[pd.DataFrame] = []
        for start in range(0, len(missing), 100):
            chunk = missing[start : start + 100]
            query = " OR ".join(f"accession:{entry}" for entry in chunk)
            params = {
                "format": "tsv",
                "fields": "accession,gene_primary,protein_name",
                "query": f"({query})",
            }
            response = session.get("https://rest.uniprot.org/uniprotkb/stream", params=params, timeout=60)
            response.raise_for_status()
            rows.append(pd.read_csv(io.StringIO(response.text), sep="\t", dtype=str))
            time.sleep(0.1)
        if rows:
            fetched = pd.concat(rows, ignore_index=True)
            merged = pd.concat([cached, fetched], ignore_index=True).drop_duplicates(subset=["Entry"], keep="last")
            merged.to_csv(cache_path, sep="\t", index=False)
            cached = merged
    return cached.rename(
        columns={
            "Entry": "uniprot_id",
            "Gene Names (primary)": "gene_symbol",
            "Protein names": "uniprot_protein_name",
        }
    )


def classify_domain_type(row: pd.Series) -> str:
    drug_classes = str(row.get("DRUG_CLASSES", "") or "").lower()
    mechanism = str(row.get("FDA_MECHANISM", "") or "").lower()
    protein_name = str(row.get("PROTEIN_NAME", "") or "").lower()
    if "antiviral" in drug_classes or "polyprotein" in protein_name or "protease inhibitor" in mechanism or "integrase inhibitor" in mechanism:
        return "viral"
    if "antibacterial" in drug_classes or "antibiotic" in protein_name:
        return "bacterial"
    return "cancer"


def classify_target_domain(row: pd.Series) -> str:
    drug_name = str(row.get("drug_name", "") or "")
    protein_name = str(row.get("PROTEIN_NAME", "") or "").lower()
    mechanism = str(row.get("FDA_MECHANISM", "") or "").lower()
    if row.get("domain_type") == "viral":
        canonical = canonicalize_hiv_drug_name(drug_name)
        if canonical in {
            "Lamivudine",
            "Abacavir",
            "Adefovir",
            "Zidovudine",
            "Stavudine",
            "Zalcitabine",
            "Didanosine",
            "Delavirdine",
            "Efavirenz",
            "Etravirine",
            "Nevirapine",
            "Rilpivirine",
            "Tenofovir",
        } or "reverse transcriptase" in mechanism:
            return "RT"
        if canonical in {"Indinavir", "Lopinavir", "Nelfinavir", "Ritonavir", "Saquinavir", "Amprenavir", "Darunavir"} or "protease" in mechanism:
            return "PR"
        if canonical in {"Raltegravir", "Elvitegravir", "Dolutegravir"} or "integrase" in mechanism:
            return "IN"
        return "viral_other"
    if "kinase" in protein_name or "receptor" in protein_name:
        return "kinase"
    if "p53" in protein_name:
        return "p53"
    return "unknown"


def evaluation_unit_from_type(sample_type: str) -> str:
    if sample_type in {"Multiple Substitution", "Multiple Complex"}:
        return "observed_combo"
    if sample_type in {"Deletion", "Insertion", "Indel"}:
        return "indel"
    return "site"


def canonicalize_hiv_drug_name(token: str) -> str:
    raw = str(token or "").strip()
    upper = raw.upper()
    return HIV_DRUG_ALIASES.get(upper, raw)


def canonicalize_hiv_token(token: str) -> str | None:
    raw = str(token or "").strip()
    if not raw:
        return None
    upper = raw.upper()
    if upper in HIV_GENERIC_TOKENS:
        return None
    return canonicalize_hiv_drug_name(raw)


def split_hiv_treatment_tokens(value: str | None) -> list[str]:
    if value is None or pd.isna(value):
        return []
    tokens = []
    seen = set()
    for item in str(value).split(","):
        canonical = canonicalize_hiv_token(item)
        if canonical and canonical not in seen:
            seen.add(canonical)
            tokens.append(canonical)
    return tokens


def hiv_position_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in df.columns if re.fullmatch(r"P\d+", str(column))]
    return sorted(columns, key=lambda name: int(name[1:]))


def extract_unique_aas(value: str | None) -> list[str]:
    if value is None or pd.isna(value):
        return []
    token = re.sub(r"[^A-Za-z\*]", "", str(value).strip().upper())
    if not token or token == "NAN":
        return []
    ordered: list[str] = []
    seen = set()
    for char in token:
        if char not in seen:
            seen.add(char)
            ordered.append(char)
    return ordered


def infer_hiv_reference_map(df: pd.DataFrame, position_columns: list[str]) -> dict[int, str]:
    references: dict[int, str] = {}
    for column in position_columns:
        counts: Counter[str] = Counter()
        value_counts = df[column].dropna().astype(str).value_counts()
        for value, observed_count in value_counts.items():
            if value == "-":
                continue
            for aa in extract_unique_aas(value):
                counts[aa] += int(observed_count)
        if counts:
            references[int(column[1:])] = counts.most_common(1)[0][0]
    return references


def accumulate_hiv_mutation_counts(df: pd.DataFrame, position_columns: list[str], reference_map: dict[int, str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for column in position_columns:
        position = int(column[1:])
        reference = reference_map.get(position)
        if not reference:
            continue
        value_counts = df[column].dropna().astype(str).value_counts()
        for value, observed_count in value_counts.items():
            if value == "-":
                continue
            for aa in extract_unique_aas(value):
                if aa != reference:
                    counts[f"{reference}{position}{aa}"] += int(observed_count)
    return counts


def aa_token_to_one_letter_sequence(token: str | None) -> str | None:
    if token is None:
        return None
    raw = str(token).strip()
    if not raw or raw in {"?", "="}:
        return raw or None
    if raw == "*":
        return raw
    if re.fullmatch(r"[A-Z\*]+", raw):
        return raw
    if re.fullmatch(r"(?:[A-Z][a-z]{2}|\*)+", raw):
        pieces = re.findall(r"[A-Z][a-z]{2}|\*", raw)
        mapped = "".join(AA3_TO_1.get(piece, "") for piece in pieces)
        return mapped or None
    return None


def normalize_protein_change(raw_value: str | None) -> str | None:
    if raw_value is None or pd.isna(raw_value):
        return None
    text = str(raw_value).strip()
    if not text or text == "nan":
        return None
    text = text.replace("p.(", "p.").rstrip(")")

    match = SIMPLE_HGVSP_RE.match(text)
    if match:
        ref_token, position, alt_token = match.groups()
        ref = aa_token_to_one_letter_sequence(ref_token)
        alt = aa_token_to_one_letter_sequence(alt_token)
        if not ref or not alt or alt in {"?", "="} or len(ref) != 1 or len(alt) != 1:
            return None
        return f"{ref}{int(position)}{alt}"

    match = DELINS_HGVSP_RE.match(text)
    if match:
        ref_start, start_pos, ref_end, end_pos, alt_token = match.groups()
        ref_left = aa_token_to_one_letter_sequence(ref_start)
        ref_right = aa_token_to_one_letter_sequence(ref_end) if ref_end else None
        alt = aa_token_to_one_letter_sequence(alt_token)
        if not ref_left or not alt:
            return None
        start = int(start_pos)
        if ref_right and end_pos:
            return f"{ref_left}{start}_{ref_right}{int(end_pos)}delins{alt}"
        return f"{ref_left}{start}delins{alt}"

    match = DELETION_HGVSP_RE.match(text)
    if match:
        ref_start, start_pos, ref_end, end_pos, deleted_token = match.groups()
        ref_left = aa_token_to_one_letter_sequence(ref_start)
        ref_right = aa_token_to_one_letter_sequence(ref_end) if ref_end else None
        deleted = aa_token_to_one_letter_sequence(deleted_token) if deleted_token else ""
        if not ref_left:
            return None
        start = int(start_pos)
        if ref_right and end_pos:
            return f"{ref_left}{start}_{ref_right}{int(end_pos)}del{deleted}"
        return f"{ref_left}{start}del{deleted}"

    match = INSERTION_HGVSP_RE.match(text)
    if match:
        ref_start, start_pos, ref_end, end_pos, inserted_token = match.groups()
        ref_left = aa_token_to_one_letter_sequence(ref_start)
        ref_right = aa_token_to_one_letter_sequence(ref_end)
        inserted = aa_token_to_one_letter_sequence(inserted_token)
        if not ref_left or not ref_right or not inserted:
            return None
        return f"{ref_left}{int(start_pos)}_{ref_right}{int(end_pos)}ins{inserted}"

    return None


def noisy_or(values: Iterable[float]) -> float:
    probability = 1.0
    used = False
    for value in values:
        if value is None or pd.isna(value):
            continue
        probability *= 1.0 - float(value)
        used = True
    if not used:
        return 0.0
    return 1.0 - probability


def write_long_tsv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, sep="\t", index=False)
