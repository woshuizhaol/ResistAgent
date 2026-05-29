#!/usr/bin/env python3
"""Stage 7 benchmark helpers."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import urllib.request

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

from tools.runtime import iso_now, json_dump, load_yaml, project_root, sha256_file
from tools.stage4_utils import ndcg_at_k

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore[assignment]
    XGBRegressor = None  # type: ignore[assignment]

BENCHMARK_REQUIRED_PRIOR = "benchmark_prior_blind.parquet"
HTML_DATE_PATTERNS = [
    r'citation_online_date" content="([^"]+)"',
    r'citation_publication_date" content="([^"]+)"',
    r'article:published_time" content="([^"]+)"',
    r'datePublished" content="([^"]+)"',
    r'([A-Z][a-z]{2,9} \d{1,2}, \d{4})',
    r'(\d{4}/\d{2}/\d{2})',
    r'(\d{4}-\d{2}-\d{2})',
]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


def _safe_float_or(value: Any, default: float = 0.0) -> float:
    native = _safe_float(value)
    if native is None:
        return float(default)
    return float(native)


def _zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    std = float(array.std())
    if std <= 1.0e-12:
        return np.zeros_like(array, dtype=float)
    return (array - float(array.mean())) / std


def _logit01(value: float) -> float:
    clipped = min(1.0 - 1.0e-6, max(1.0e-6, float(value)))
    return float(math.log(clipped / (1.0 - clipped)))


def _parse_date_text(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = text.replace("Z", "").replace("T", " ").split()[0] if "T" in text else text
    for fmt in [
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%Y %b %d",
        "%Y %B %d",
        "%b %d, %Y",
        "%B %d, %Y",
        "%Y %b",
        "%Y %B",
        "%Y",
    ]:
        try:
            dt = datetime.strptime(candidate, fmt)
            return dt.date().isoformat()
        except Exception:
            continue
    return None


def _quarter_to_date(text: str | None) -> str | None:
    token = str(text or "").strip().upper()
    match = re.search(r"(?P<yy>\d{2})Q(?P<q>[1-4])", token)
    if not match:
        return None
    year = 2000 + int(match.group("yy"))
    month = 1 + (int(match.group("q")) - 1) * 3
    return f"{year:04d}-{month:02d}-01"


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(str(url), headers={"User-Agent": "ResistAgent/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="ignore")


def _extract_doi(text: str) -> str | None:
    match = re.search(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", str(text), flags=re.I)
    if not match:
        return None
    return str(match.group(1))


def _fetch_pubmed_sort_date(pmid: str) -> tuple[str | None, str]:
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
    payload = json.loads(_fetch_text(url))
    result = dict(payload.get("result") or {}).get(str(pmid)) or {}
    sortpubdate = _parse_date_text(result.get("sortpubdate"))
    if sortpubdate:
        return sortpubdate, "pmid_sortpubdate"
    pubdate = _parse_date_text(result.get("pubdate"))
    return pubdate, "pmid_pubdate"


def _fetch_doi_date(doi: str) -> tuple[str | None, str]:
    request = urllib.request.Request(
        f"https://doi.org/{doi}",
        headers={
            "Accept": "application/vnd.citationstyles.csl+json",
            "User-Agent": "ResistAgent/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    for key, label in [
        ("published-online", "doi_published_online"),
        ("published-print", "doi_published_print"),
        ("created", "doi_created"),
    ]:
        if key not in payload:
            continue
        parts = list(dict(payload.get(key) or {}).get("date-parts") or [])
        if not parts:
            continue
        fields = list(parts[0])
        year = int(fields[0])
        month = int(fields[1]) if len(fields) >= 2 else 1
        day = int(fields[2]) if len(fields) >= 3 else 1
        return f"{year:04d}-{month:02d}-{day:02d}", label
    for license_row in list(payload.get("license") or []):
        start = dict(license_row.get("start") or {})
        parts = list(start.get("date-parts") or [])
        if not parts:
            continue
        fields = list(parts[0])
        year = int(fields[0])
        month = int(fields[1]) if len(fields) >= 2 else 1
        day = int(fields[2]) if len(fields) >= 3 else 1
        return f"{year:04d}-{month:02d}-{day:02d}", "doi_license_start"
    return None, "doi_unresolved"


def _fetch_html_date(url: str) -> tuple[str | None, str]:
    text = _fetch_text(url)
    for pattern in HTML_DATE_PATTERNS:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        iso_date = _parse_date_text(match.group(1))
        if iso_date:
            return iso_date, f"html_pattern:{pattern}"
    return None, "html_unresolved"


def _gdsc_release_date(root: Path) -> tuple[str | None, str]:
    candidates = [
        root / "data" / "GDSC" / "ANOVA_results_GDSC2_27Oct23.xlsx",
        root / "data" / "GDSC" / "GDSC2_fitted_dose_response_27Oct23.xlsx",
    ]
    for path in candidates:
        match = re.search(r"(\d{2}[A-Z][a-z]{2}\d{2})", path.name)
        if match:
            dt = datetime.strptime(match.group(1), "%d%b%y").date().isoformat()
            return dt, f"local_release_filename:{path.name}"
    return None, "local_release_filename_missing"


def _depmap_release_date(root: Path) -> tuple[str | None, str]:
    model_path = root / "data" / "DepMap (Cancer Dependency Map)" / "Model.csv"
    if not model_path.exists():
        return None, "depmap_model_missing"
    with model_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            match = re.search(r"as of (\d{2}Q\d)", raw_line, flags=re.I)
            if match:
                return _quarter_to_date(match.group(1)), f"local_quarter_hint:{match.group(1).upper()}"
    return None, "depmap_quarter_hint_missing"


def build_source_time_metadata(root: Path, config: dict[str, Any]) -> pd.DataFrame:
    stage1 = dict(config["stage1"])
    stage7 = dict(config["stage7"])
    master = pd.read_parquet(root / str(stage1["master_table"]), columns=["DATASET", "SAMPLE_SOURCE"])
    sources = master.dropna(subset=["SAMPLE_SOURCE"]).drop_duplicates(subset=["DATASET", "SAMPLE_SOURCE"]).copy()
    rows: list[dict[str, Any]] = []
    for record in sources.to_dict(orient="records"):
        dataset = str(record["DATASET"])
        source_url = str(record["SAMPLE_SOURCE"])
        source_date = None
        evidence = "unresolved"
        try:
            pmid_match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", source_url, flags=re.I)
            if pmid_match:
                source_date, evidence = _fetch_pubmed_sort_date(str(pmid_match.group(1)))
            elif dataset == "GDSC":
                source_date, evidence = _gdsc_release_date(root)
            elif dataset == "Depmap":
                source_date, evidence = _depmap_release_date(root)
            elif "bioinfo.uth.edu/kmd" in source_url.lower():
                source_date, evidence = _fetch_html_date(source_url)
            else:
                doi = _extract_doi(source_url)
                if doi:
                    source_date, evidence = _fetch_doi_date(doi)
                else:
                    source_date, evidence = _fetch_html_date(source_url)
        except Exception as exc:
            source_date = None
            evidence = f"resolution_error:{type(exc).__name__}"
        rows.append(
            {
                "DATASET": dataset,
                "SAMPLE_SOURCE": source_url,
                "source_date": source_date,
                "source_year": None if source_date is None else int(str(source_date)[:4]),
                "time_evidence": evidence,
            }
        )
    frame = pd.DataFrame.from_records(rows).sort_values(["DATASET", "SAMPLE_SOURCE"]).reset_index(drop=True)
    output_csv = root / str(stage7["source_time_metadata_csv"])
    summary_json = root / str(stage7["source_time_metadata_summary_json"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    json_dump(
        summary_json,
        {
            "generated_at": iso_now(),
            "row_count": int(len(frame)),
            "resolved_rows": int(frame["source_date"].notna().sum()) if not frame.empty else 0,
            "evidence_counts": {str(key): int(value) for key, value in frame["time_evidence"].fillna("missing").value_counts().sort_index().items()},
        },
    )
    return frame


def ensure_blind_prior_path(prior_path: Path) -> None:
    name = prior_path.name
    if name != BENCHMARK_REQUIRED_PRIOR:
        raise RuntimeError(
            f"Stage 7 only allows {BENCHMARK_REQUIRED_PRIOR}; refusing to read {name}"
        )


def set_d_pairs(cases_config: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for case in list(cases_config.get("set_d", [])):
        pairs.add((str(case.get("uniprot_id") or ""), str(case.get("drug_name") or "")))
    return pairs


def load_benchmark_frame(
    *,
    root: Path,
    config: dict[str, Any],
    cases_config: dict[str, Any],
) -> tuple[pd.DataFrame, Path]:
    stage1 = dict(config["stage1"])
    stage7 = dict(config["stage7"])
    master_path = root / str(stage1["master_table"])
    prior_path = root / str(stage1["benchmark_prior_blind"])
    ensure_blind_prior_path(prior_path)

    master = pd.read_parquet(master_path)
    time_metadata = build_source_time_metadata(root, config)
    prior = pd.read_parquet(prior_path)
    frame = master.dropna(subset=["DDG.EXP", "gene_symbol", "mutation_key", "drug_name"]).copy()
    frame["drug_name"] = frame["drug_name"].astype(str)
    frame["gene_symbol"] = frame["gene_symbol"].astype(str)
    frame["mutation_key"] = frame["mutation_key"].astype(str)
    frame = frame.merge(
        prior,
        on=["gene_symbol", "mutation_key", "drug_name"],
        how="left",
        suffixes=("", "_prior"),
    )
    frame = frame.merge(
        time_metadata,
        on=["DATASET", "SAMPLE_SOURCE"],
        how="left",
    )
    frame["DDG.EXP"] = pd.to_numeric(frame["DDG.EXP"], errors="coerce")
    frame = frame.dropna(subset=["DDG.EXP"]).copy()
    frame["P_background"] = pd.to_numeric(frame.get("P_background"), errors="coerce").fillna(0.0)
    frame["P_drug_selected"] = pd.to_numeric(frame.get("P_drug_selected"), errors="coerce").fillna(0.0)
    frame["prior_logit_background"] = frame["P_background"].map(_logit01)
    frame["prior_logit_drug_selected"] = frame["P_drug_selected"].map(_logit01)
    frame["prior_delta"] = frame["P_drug_selected"] - frame["P_background"]
    frame["prior_ratio"] = frame["P_drug_selected"] / (frame["P_background"] + 1.0e-6)
    frame["combination_size"] = pd.to_numeric(frame.get("combination_size"), errors="coerce").fillna(1.0)
    frame["has_observed_combo"] = frame.get("has_observed_combo", False).fillna(False).astype(bool).astype(int)
    frame["is_single_substitution"] = frame["TYPE"].astype(str).eq("Single Substitution").astype(int)
    frame["is_multiple_substitution"] = frame["TYPE"].astype(str).eq("Multiple Substitution").astype(int)
    frame["is_deletion_like"] = frame["TYPE"].astype(str).isin(["Deletion", "Indel"]).astype(int)
    frame["fitness_proxy"] = (
        0.65 * frame["prior_logit_drug_selected"]
        - 0.35 * frame["prior_logit_background"]
        + 0.10 * frame["combination_size"]
    )
    frame["group_key"] = frame["UNIPROT_ID"].astype(str) + "||" + frame["drug_name"].astype(str)
    frame["source_date"] = pd.to_datetime(frame.get("source_date"), errors="coerce")
    frame["source_year"] = pd.to_numeric(frame.get("source_year"), errors="coerce")
    frame["set_bucket"] = np.where(
        frame[["UNIPROT_ID", "drug_name"]].apply(tuple, axis=1).isin(set_d_pairs(cases_config)),
        "Set-D",
        "MdrDB_holdout",
    )
    frame["benchmark_scope"] = str(stage7.get("benchmark_scope", "master_mutation_table"))
    return frame.reset_index(drop=True), prior_path


def feature_columns(stage7: dict[str, Any], feature_set: str, frame: pd.DataFrame) -> list[str]:
    configured = list(dict(stage7.get("feature_sets") or {}).get(feature_set, []))
    return [column for column in configured if column in frame.columns]


def model_variant_for_feature_set(stage7: dict[str, Any], feature_set: str) -> str:
    variants = dict(stage7.get("model_variants") or {})
    return str(variants.get(feature_set, variants.get("default", "histgb_regression")))


def build_split_manifest(frame: pd.DataFrame, config: dict[str, Any], cases_config: dict[str, Any]) -> dict[str, Any]:
    stage7 = dict(config["stage7"])
    min_group_size = int(stage7.get("min_group_size", 12))
    group_counts = frame["group_key"].value_counts()
    eligible_groups = sorted(group_counts[group_counts >= min_group_size].index.tolist())
    eligible_frame = frame[frame["group_key"].isin(eligible_groups)].copy()
    n_splits = max(2, min(int(stage7.get("groupkfold_splits", 5)), len(eligible_groups))) if eligible_groups else 0

    strategies: list[dict[str, Any]] = []
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        folds = []
        for fold_index, (_, test_index) in enumerate(
            gkf.split(eligible_frame, groups=eligible_frame["group_key"].astype(str)),
            start=1,
        ):
            test_groups = sorted(eligible_frame.iloc[test_index]["group_key"].astype(str).unique().tolist())
            test_mask = frame["group_key"].astype(str).isin(test_groups)
            folds.append(
                {
                    "fold_id": f"pair_groupkfold_{fold_index}",
                    "test_groups": test_groups,
                    "train_rows": int((~test_mask).sum()),
                    "test_rows": int(test_mask.sum()),
                }
            )
        strategies.append(
            {
                "strategy": "pair_groupkfold",
                "status": "ready",
                "grouping": "(UNIPROT_ID, DRUG)",
                "folds": folds,
            }
        )
    else:
        strategies.append(
            {
                "strategy": "pair_groupkfold",
                "status": "unavailable",
                "reason": "insufficient eligible groups",
                "folds": [],
            }
        )

    target_counts = frame["UNIPROT_ID"].astype(str).value_counts()
    target_entities = target_counts[target_counts >= min_group_size].head(int(stage7.get("leave_one_target_top_n", 12))).index.tolist()
    strategies.append(
        {
            "strategy": "leave_one_target_out",
            "status": "ready" if target_entities else "unavailable",
            "selection_rule": "top_n_by_row_count",
            "folds": [
                {
                    "fold_id": f"leave_target_{entity}",
                    "test_uniprot_id": str(entity),
                    "train_rows": int(frame["UNIPROT_ID"].astype(str).ne(str(entity)).sum()),
                    "test_rows": int(frame["UNIPROT_ID"].astype(str).eq(str(entity)).sum()),
                }
                for entity in target_entities
            ],
        }
    )

    drug_counts = frame["drug_name"].astype(str).value_counts()
    drug_entities = drug_counts[drug_counts >= min_group_size].head(int(stage7.get("leave_one_drug_top_n", 12))).index.tolist()
    strategies.append(
        {
            "strategy": "leave_one_drug_out",
            "status": "ready" if drug_entities else "unavailable",
            "selection_rule": "top_n_by_row_count",
            "folds": [
                {
                    "fold_id": f"leave_drug_{hashlib.sha1(str(entity).encode('utf-8')).hexdigest()[:8]}",
                    "test_drug_name": str(entity),
                    "train_rows": int(frame["drug_name"].astype(str).ne(str(entity)).sum()),
                    "test_rows": int(frame["drug_name"].astype(str).eq(str(entity)).sum()),
                }
                for entity in drug_entities
            ],
        }
    )

    external_min_rows = int(stage7.get("external_holdout_min_rows", 100))
    external_folds = []
    for dataset_name in list(stage7.get("external_holdout_datasets", [])):
        mask = frame["DATASET"].astype(str).eq(str(dataset_name))
        if int(mask.sum()) < external_min_rows:
            continue
        external_folds.append(
            {
                "fold_id": f"external_{str(dataset_name)}",
                "test_dataset": str(dataset_name),
                "train_rows": int((~mask).sum()),
                "test_rows": int(mask.sum()),
            }
        )
    strategies.append(
        {
            "strategy": "external_holdout",
            "status": "ready" if external_folds else "unavailable",
            "folds": external_folds,
        }
    )
    dated = frame.dropna(subset=["source_date"]).copy()
    time_split_folds: list[dict[str, Any]] = []
    min_train_rows = int(stage7.get("time_split_min_train_rows", 5000))
    min_test_rows = int(stage7.get("time_split_min_test_rows", 1000))
    if not dated.empty and dated["source_year"].dropna().nunique() >= 2:
        for year in sorted(int(value) for value in dated["source_year"].dropna().unique().tolist()):
            cutoff = pd.Timestamp(year=year, month=12, day=31)
            train_mask = frame["source_date"].notna() & frame["source_date"].le(cutoff)
            test_mask = frame["source_date"].notna() & frame["source_date"].gt(cutoff)
            train_rows = int(train_mask.sum())
            test_rows = int(test_mask.sum())
            if train_rows < min_train_rows or test_rows < min_test_rows:
                continue
            time_split_folds.append(
                {
                    "fold_id": f"time_after_{year}",
                    "cutoff_date": cutoff.date().isoformat(),
                    "train_rows": train_rows,
                    "test_rows": test_rows,
                    "train_years": sorted({int(value) for value in frame.loc[train_mask, "source_year"].dropna().astype(int).tolist()}),
                    "test_years": sorted({int(value) for value in frame.loc[test_mask, "source_year"].dropna().astype(int).tolist()}),
                }
            )
    strategies.append(
        {
            "strategy": "time_split",
            "status": "ready" if time_split_folds else "unavailable",
            "reason": None if time_split_folds else "insufficient resolved source_date coverage",
            "time_date_field": "source_date",
            "folds": time_split_folds,
        }
    )

    stage1 = dict(config.get("stage1") or {})
    prior_relpath = str(stage1.get("benchmark_prior_blind", BENCHMARK_REQUIRED_PRIOR))
    prior_abspath = project_root() / prior_relpath
    prior_sha = sha256_file(prior_abspath) if prior_abspath.exists() else None
    return {
        "generated_at": iso_now(),
        "prior_source": {
            "path": prior_relpath,
            "sha256": prior_sha,
            "global_prior_forbidden": True,
        },
        "set_boundary": {
            "set_n_external_included_in_main_benchmark": False,
            "set_d_pairs": sorted([list(pair) for pair in set_d_pairs(cases_config)]),
            "set_bucket_counts": {str(key): int(value) for key, value in frame["set_bucket"].value_counts().sort_index().items()},
        },
        "strategies": strategies,
    }


def iter_ready_folds(split_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    ready: list[dict[str, Any]] = []
    for strategy in list(split_manifest.get("strategies", [])):
        if str(strategy.get("status") or "") != "ready":
            continue
        for fold in list(strategy.get("folds", [])):
            ready.append({"strategy": str(strategy["strategy"]), **dict(fold)})
    return ready


def fold_test_mask(frame: pd.DataFrame, fold: dict[str, Any]) -> pd.Series:
    strategy = str(fold["strategy"])
    if strategy == "pair_groupkfold":
        return frame["group_key"].astype(str).isin(list(fold.get("test_groups", [])))
    if strategy == "leave_one_target_out":
        return frame["UNIPROT_ID"].astype(str).eq(str(fold["test_uniprot_id"]))
    if strategy == "leave_one_drug_out":
        return frame["drug_name"].astype(str).eq(str(fold["test_drug_name"]))
    if strategy == "external_holdout":
        return frame["DATASET"].astype(str).eq(str(fold["test_dataset"]))
    if strategy == "time_split":
        cutoff = pd.Timestamp(str(fold["cutoff_date"]))
        return frame["source_date"].notna() & frame["source_date"].gt(cutoff)
    raise ValueError(f"Unsupported Stage 7 split strategy: {strategy}")


def build_decoy_frame(frame: pd.DataFrame, test_frame: pd.DataFrame, stage7: dict[str, Any], fold: dict[str, Any]) -> pd.DataFrame:
    decoy = test_frame.copy()
    decoy["decoy_reason"] = ""
    mask = decoy["DDG.EXP"].fillna(np.inf).le(float(stage7.get("decoy_ddg_max", 0.25)))
    mask &= decoy["P_background"].fillna(np.inf).le(float(stage7.get("decoy_background_max", 0.01)))
    mask &= decoy["P_drug_selected"].fillna(np.inf).le(float(stage7.get("decoy_drug_selected_max", 0.01)))
    if "MUT.Volume" in decoy.columns:
        mask &= decoy["MUT.Volume"].fillna(0.0).abs().le(float(stage7.get("decoy_abs_mut_volume_max", 40.0)))
    if "MUT.NetCharge" in decoy.columns:
        mask &= decoy["MUT.NetCharge"].fillna(0.0).abs().le(float(stage7.get("decoy_abs_net_charge_max", 1.0)))
    decoy = decoy[mask].copy()
    if decoy.empty:
        return decoy
    decoy["decoy_reason"] = "low_ddg_low_prior_test_only"
    decoy["split_strategy"] = str(fold["strategy"])
    decoy["fold_id"] = str(fold["fold_id"])
    decoy["decoy_label"] = True
    return decoy


def fit_regressor(train_frame: pd.DataFrame, feature_cols: list[str], seed: int) -> Any:
    usable_cols = [column for column in feature_cols if column in train_frame.columns]
    if not usable_cols or len(train_frame) < 20:
        model = DummyRegressor(strategy="mean")
        model.fit(np.zeros((len(train_frame), 1), dtype=float), train_frame["DDG.EXP"].to_numpy())
        return model, usable_cols
    x_train = train_frame[usable_cols].apply(pd.to_numeric, errors="coerce")
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.06,
        max_iter=160,
        min_samples_leaf=20,
        random_state=int(seed),
    )
    model.fit(x_train, train_frame["DDG.EXP"].to_numpy(dtype=float))
    return model, usable_cols


def fit_feature_model_bundle(
    train_frame: pd.DataFrame,
    *,
    feature_set: str,
    feature_cols: list[str],
    stage7: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    usable_cols = [column for column in feature_cols if column in train_frame.columns]
    variant = model_variant_for_feature_set(stage7, feature_set)
    if variant == "xgb_dual_struct_ensemble":
        primary_cfg = dict(stage7.get("xgb_dual") or {})
        ensemble_cfg = dict(stage7.get("xgb_dual_struct_ensemble") or {})
        structure_cols = feature_columns(stage7, "mdrdb_structure_only", train_frame)
        if (
            XGBRegressor is not None
            and XGBClassifier is not None
            and usable_cols
            and structure_cols
            and len(train_frame) >= 20
        ):
            primary_x = train_frame[usable_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            structure_x = train_frame[structure_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            y_reg = train_frame["DDG.EXP"].to_numpy(dtype=float)
            threshold = float(stage7.get("resistant_ddg_threshold", 1.0))
            y_cls = train_frame["DDG.EXP"].ge(threshold).astype(int).to_numpy()
            regressor = XGBRegressor(
                n_estimators=int(primary_cfg.get("reg_n_estimators", 160)),
                max_depth=int(primary_cfg.get("max_depth", 6)),
                learning_rate=float(primary_cfg.get("learning_rate", 0.05)),
                subsample=float(primary_cfg.get("subsample", 0.8)),
                colsample_bytree=float(primary_cfg.get("colsample_bytree", 0.8)),
                reg_lambda=float(primary_cfg.get("reg_lambda", 1.0)),
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=int(primary_cfg.get("n_jobs", 8)),
                random_state=int(seed),
            )
            classifier = XGBClassifier(
                n_estimators=int(primary_cfg.get("clf_n_estimators", 140)),
                max_depth=int(primary_cfg.get("max_depth", 6)),
                learning_rate=float(primary_cfg.get("learning_rate", 0.05)),
                subsample=float(primary_cfg.get("subsample", 0.8)),
                colsample_bytree=float(primary_cfg.get("colsample_bytree", 0.8)),
                reg_lambda=float(primary_cfg.get("reg_lambda", 1.0)),
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=int(primary_cfg.get("n_jobs", 8)),
                random_state=int(seed),
            )
            structure_aux = HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.05,
                max_iter=220,
                min_samples_leaf=20,
                random_state=int(seed),
            )
            regressor.fit(primary_x, y_reg)
            classifier.fit(primary_x, y_cls)
            structure_aux.fit(structure_x, y_reg)
            return {
                "variant": "xgb_dual_struct_ensemble",
                "feature_cols": usable_cols,
                "structure_feature_cols": structure_cols,
                "regressor": regressor,
                "classifier": classifier,
                "structure_aux": structure_aux,
                "primary_rank_weight": float(ensemble_cfg.get("primary_rank_weight", 0.70)),
                "structure_aux_weight": float(ensemble_cfg.get("structure_aux_weight", 0.30)),
            }
    if variant == "xgb_dual" and XGBRegressor is not None and XGBClassifier is not None and usable_cols and len(train_frame) >= 20:
        cfg = dict(stage7.get("xgb_dual") or {})
        x_train = train_frame[usable_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y_reg = train_frame["DDG.EXP"].to_numpy(dtype=float)
        threshold = float(stage7.get("resistant_ddg_threshold", 1.0))
        y_cls = train_frame["DDG.EXP"].ge(threshold).astype(int).to_numpy()
        regressor = XGBRegressor(
            n_estimators=int(cfg.get("reg_n_estimators", 160)),
            max_depth=int(cfg.get("max_depth", 6)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            subsample=float(cfg.get("subsample", 0.8)),
            colsample_bytree=float(cfg.get("colsample_bytree", 0.8)),
            reg_lambda=float(cfg.get("reg_lambda", 1.0)),
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=int(cfg.get("n_jobs", 8)),
            random_state=int(seed),
        )
        classifier = XGBClassifier(
            n_estimators=int(cfg.get("clf_n_estimators", 140)),
            max_depth=int(cfg.get("max_depth", 6)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            subsample=float(cfg.get("subsample", 0.8)),
            colsample_bytree=float(cfg.get("colsample_bytree", 0.8)),
            reg_lambda=float(cfg.get("reg_lambda", 1.0)),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=int(cfg.get("n_jobs", 8)),
            random_state=int(seed),
        )
        regressor.fit(x_train, y_reg)
        classifier.fit(x_train, y_cls)
        return {
            "variant": "xgb_dual",
            "feature_cols": usable_cols,
            "regressor": regressor,
            "classifier": classifier,
        }
    regressor, usable_cols = fit_regressor(train_frame, usable_cols, seed)
    return {
        "variant": "regression_only",
        "feature_cols": usable_cols,
        "regressor": regressor,
    }


def predict_feature_model_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    stage7: dict[str, Any],
) -> dict[str, np.ndarray]:
    variant = str(bundle.get("variant") or "regression_only")
    feature_cols = [str(column) for column in list(bundle.get("feature_cols") or []) if str(column)]
    if not feature_cols:
        x_value = np.zeros((len(frame), 1), dtype=float)
    else:
        x_value = frame[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    regressor = bundle["regressor"]
    prediction = regressor.predict(x_value)
    payload: dict[str, np.ndarray] = {"prediction": np.asarray(prediction, dtype=float)}
    if variant == "xgb_dual_struct_ensemble":
        classifier = bundle["classifier"]
        classifier_weight = float(dict(dict(stage7.get("xgb_dual") or {}).get("rank_score") or {}).get("classifier_weight", 0.75))
        regression_z_weight = float(dict(dict(stage7.get("xgb_dual") or {}).get("rank_score") or {}).get("regression_z_weight", 0.15))
        prior_delta_weight = float(dict(dict(stage7.get("xgb_dual") or {}).get("rank_score") or {}).get("prior_delta_weight", 0.0))
        docking_z_weight = float(dict(dict(stage7.get("xgb_dual") or {}).get("rank_score") or {}).get("docking_z_weight", 0.0))
        resistant_prob = classifier.predict_proba(x_value)[:, 1]
        prior_delta_term = (
            prior_delta_weight * np.asarray(frame["prior_delta"].fillna(0.0).to_numpy(dtype=float), dtype=float)
            if prior_delta_weight != 0.0 and "prior_delta" in frame.columns
            else 0.0
        )
        docking_z_term = (
            docking_z_weight * _zscore(np.asarray(frame["vina"].fillna(0.0).to_numpy(dtype=float), dtype=float))
            if docking_z_weight != 0.0 and "vina" in frame.columns
            else 0.0
        )
        primary_rank = (
            classifier_weight * np.asarray(resistant_prob, dtype=float)
            + regression_z_weight * _zscore(np.asarray(prediction, dtype=float))
            + prior_delta_term
            + docking_z_term
        )
        structure_cols = [str(column) for column in list(bundle.get("structure_feature_cols") or []) if str(column)]
        structure_x = frame[structure_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0) if structure_cols else np.zeros((len(frame), 1), dtype=float)
        structure_pred = np.asarray(bundle["structure_aux"].predict(structure_x), dtype=float)
        payload["resistant_prob"] = np.asarray(resistant_prob, dtype=float)
        payload["rank_score"] = (
            float(bundle.get("primary_rank_weight", 0.70)) * primary_rank
            + float(bundle.get("structure_aux_weight", 0.30)) * _zscore(structure_pred)
        )
        payload["structure_aux_prediction"] = structure_pred
        return payload
    if variant == "xgb_dual":
        classifier = bundle["classifier"]
        resistant_prob = classifier.predict_proba(x_value)[:, 1]
        rank_cfg = dict(dict(stage7.get("xgb_dual") or {}).get("rank_score") or {})
        classifier_weight = float(rank_cfg.get("classifier_weight", 0.75))
        regression_z_weight = float(rank_cfg.get("regression_z_weight", 0.25))
        prior_delta_weight = float(rank_cfg.get("prior_delta_weight", 0.0))
        docking_z_weight = float(rank_cfg.get("docking_z_weight", 0.0))
        payload["resistant_prob"] = np.asarray(resistant_prob, dtype=float)
        prior_delta_term = (
            prior_delta_weight * np.asarray(frame["prior_delta"].fillna(0.0).to_numpy(dtype=float), dtype=float)
            if prior_delta_weight != 0.0 and "prior_delta" in frame.columns
            else 0.0
        )
        docking_z_term = (
            docking_z_weight * _zscore(np.asarray(frame["vina"].fillna(0.0).to_numpy(dtype=float), dtype=float))
            if docking_z_weight != 0.0 and "vina" in frame.columns
            else 0.0
        )
        payload["rank_score"] = (
            classifier_weight * np.asarray(resistant_prob, dtype=float)
            + regression_z_weight * _zscore(np.asarray(prediction, dtype=float))
            + prior_delta_term
            + docking_z_term
        )
    return payload


def predict_regressor(model: Any, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if not feature_cols:
        return model.predict(np.zeros((len(frame), 1), dtype=float))
    return model.predict(frame[feature_cols].apply(pd.to_numeric, errors="coerce"))


def regression_metrics(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
) -> dict[str, Any]:
    subset = frame.dropna(subset=[prediction_col, "DDG.EXP"]).copy()
    if subset.empty:
        return {
            "mae": None,
            "rmse": None,
            "pearson_r": None,
            "spearman_r": None,
            "row_count": 0,
        }
    actual = subset["DDG.EXP"].to_numpy(dtype=float)
    predicted = subset[prediction_col].to_numpy(dtype=float)
    mse = float(mean_squared_error(actual, predicted))
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(max(0.0, mse))),
        "pearson_r": _safe_float(subset["DDG.EXP"].corr(subset[prediction_col], method="pearson")),
        "spearman_r": _safe_float(subset["DDG.EXP"].corr(subset[prediction_col], method="spearman")),
        "row_count": int(len(subset)),
    }


def ranking_metrics(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    k_values: list[int],
    resistant_threshold: float,
) -> dict[str, Any]:
    groups = frame.groupby("group_key", dropna=False)
    ndcg_scores = {k: [] for k in k_values}
    precision_scores = {k: [] for k in k_values}
    group_count = 0
    for _, group in groups:
        if len(group) < 2:
            continue
        group_count += 1
        group = group.copy()
        group["relevance"] = group["DDG.EXP"].clip(lower=0.0)
        ranked = group.sort_values(prediction_col, ascending=False)
        positives = ranked["DDG.EXP"].ge(float(resistant_threshold)).astype(float)
        for k in k_values:
            ndcg_scores[k].append(ndcg_at_k(group, prediction_col, "relevance", int(k)))
            precision_scores[k].append(float(positives.head(int(k)).mean()))
    payload: dict[str, Any] = {"group_count": int(group_count)}
    for k in k_values:
        payload[f"ndcg@{k}"] = float(np.mean(ndcg_scores[k])) if ndcg_scores[k] else None
        payload[f"precision@{k}"] = float(np.mean(precision_scores[k])) if precision_scores[k] else None
    return payload


def evaluate_fold(
    frame: pd.DataFrame,
    *,
    fold: dict[str, Any],
    feature_set: str,
    feature_cols: list[str],
    stage7: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    test_mask = fold_test_mask(frame, fold)
    train = frame[~test_mask].copy()
    test = frame[test_mask].copy()
    model_bundle = fit_feature_model_bundle(
        train,
        feature_set=str(feature_set),
        feature_cols=feature_cols,
        stage7=stage7,
        seed=int(stage7.get("random_seed", 20260413)),
    )
    prediction_payload = predict_feature_model_bundle(model_bundle, test, stage7=stage7)
    scored = test.copy()
    for column, values in prediction_payload.items():
        scored[column] = values
    scored["feature_set"] = str(feature_set)
    scored["split_strategy"] = str(fold["strategy"])
    scored["fold_id"] = str(fold["fold_id"])
    regression = regression_metrics(scored, prediction_col="prediction")
    ranking_col = "rank_score" if "rank_score" in scored.columns else "prediction"
    ranking = ranking_metrics(
        scored,
        prediction_col=ranking_col,
        k_values=[int(value) for value in list(stage7.get("ndcg_at_k", [5, 10]))],
        resistant_threshold=float(stage7.get("resistant_ddg_threshold", 1.0)),
    )
    return scored, regression, ranking


def build_prior_usage_audit_rows(
    *,
    prior_path: Path,
    split_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in iter_ready_folds(split_manifest):
        rows.append(
            {
                "split_strategy": str(fold["strategy"]),
                "fold_id": str(fold["fold_id"]),
                "loaded_prior_path": str(prior_path),
                "allowed_prior_filename": BENCHMARK_REQUIRED_PRIOR,
                "global_prior_detected": prior_path.name == "global_prior.parquet",
                "prior_sha256": sha256_file(prior_path),
                "generated_at": iso_now(),
            }
        )
    return rows


def build_subgroup_metrics(
    predictions: pd.DataFrame,
    *,
    stage7: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return rows
    ranking_col = "rank_score" if "rank_score" in predictions.columns else "prediction"
    for (feature_set, split_strategy, mutation_type), group in predictions.groupby(
        ["feature_set", "split_strategy", "TYPE"],
        dropna=False,
    ):
        regression = regression_metrics(group, prediction_col="prediction")
        ranking = ranking_metrics(
            group,
            prediction_col=ranking_col,
            k_values=[int(value) for value in list(stage7.get("ndcg_at_k", [5, 10]))],
            resistant_threshold=float(stage7.get("resistant_ddg_threshold", 1.0)),
        )
        rows.append(
            {
                "feature_set": str(feature_set),
                "split_strategy": str(split_strategy),
                "mutation_type": str(mutation_type),
                **regression,
                **ranking,
            }
        )
    return rows


def build_decoy_metrics(
    predictions: pd.DataFrame,
    decoys: pd.DataFrame,
    *,
    stage7: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if predictions.empty or decoys.empty:
        return rows
    joined = predictions.merge(
        decoys[["SAMPLE_ID", "split_strategy", "fold_id", "decoy_label"]],
        on=["SAMPLE_ID", "split_strategy", "fold_id"],
        how="left",
    )
    if "decoy_label" in joined.columns:
        joined["decoy_label"] = pd.to_numeric(joined["decoy_label"], errors="coerce").fillna(0).astype(bool)
    else:
        joined["decoy_label"] = False
    ranking_col = "rank_score" if "rank_score" in joined.columns else "prediction"
    k_values = [int(value) for value in list(stage7.get("precision_at_k", [5, 10]))]
    for (feature_set, split_strategy), group in joined.groupby(["feature_set", "split_strategy"], dropna=False):
        for k in k_values:
            topk = (
                group.sort_values(["fold_id", "group_key", ranking_col], ascending=[True, True, False])
                .groupby(["fold_id", "group_key"], dropna=False)
                .head(int(k))
            )
            topk_total_slots = int(len(topk))
            rows.append(
                {
                    "feature_set": str(feature_set),
                    "split_strategy": str(split_strategy),
                    "k": int(k),
                    "decoy_count": int(group["decoy_label"].sum()),
                    "topk_decoy_hits": int(topk["decoy_label"].sum()),
                    "topk_total_slots": int(topk_total_slots),
                    "fpr_at_k": float(topk["decoy_label"].sum() / max(1, topk_total_slots)),
                }
            )
    random_rows = []
    seed = int(stage7.get("random_seed", 20260413))
    rng = np.random.default_rng(seed)
    for split_strategy, group in joined.groupby("split_strategy", dropna=False):
        random_group = group.copy()
        random_group["random_prediction"] = rng.random(len(random_group))
        for k in k_values:
            topk = (
                random_group.sort_values(["fold_id", "group_key", "random_prediction"], ascending=[True, True, False])
                .groupby(["fold_id", "group_key"], dropna=False)
                .head(int(k))
            )
            topk_total_slots = int(len(topk))
            random_rows.append(
                {
                    "feature_set": "random",
                    "split_strategy": str(split_strategy),
                    "k": int(k),
                    "decoy_count": int(random_group["decoy_label"].sum()),
                    "topk_decoy_hits": int(topk["decoy_label"].sum()),
                    "topk_total_slots": int(topk_total_slots),
                    "fpr_at_k": float(topk["decoy_label"].sum() / max(1, topk_total_slots)),
                }
            )
    rows.extend(random_rows)
    return rows


def build_objective_compare_rows(root: Path, stage7: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id in list(stage7.get("stage6_case_ids", [])):
        path = root / "outputs" / str(case_id) / "stage6" / "objective_ablation.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty or "objective_name" not in frame.columns:
            continue
        robust = frame[frame["objective_name"].astype(str).eq("robust")].head(1)
        naive = frame[frame["objective_name"].astype(str).eq("naive")].head(1)
        if robust.empty or naive.empty:
            continue
        robust_row = robust.iloc[0]
        naive_row = naive.iloc[0]
        rows.append(
            {
                "case_id": str(case_id),
                "robust_panel_passing_rate": _safe_float(robust_row.get("panel_passing_rate")),
                "naive_panel_passing_rate": _safe_float(naive_row.get("panel_passing_rate")),
                "robust_best_objective_reward": _safe_float(robust_row.get("best_objective_reward")),
                "naive_best_objective_reward": _safe_float(naive_row.get("best_objective_reward")),
                "robust_top20_dep_median": _safe_float(robust_row.get("top20_dep_median")),
                "naive_top20_dep_median": _safe_float(naive_row.get("top20_dep_median")),
                "robust_top50_scaffold_unique": _safe_float(robust_row.get("top50_scaffold_unique")),
                "naive_top50_scaffold_unique": _safe_float(naive_row.get("top50_scaffold_unique")),
                "panel_passing_rate_delta": _safe_float_or(robust_row.get("panel_passing_rate")) - _safe_float_or(naive_row.get("panel_passing_rate")),
                "best_objective_reward_delta": _safe_float_or(robust_row.get("best_objective_reward")) - _safe_float_or(naive_row.get("best_objective_reward")),
                "dep_median_delta": _safe_float_or(robust_row.get("top20_dep_median")) - _safe_float_or(naive_row.get("top20_dep_median")),
                "scaffold_unique_delta": _safe_float_or(robust_row.get("top50_scaffold_unique")) - _safe_float_or(naive_row.get("top50_scaffold_unique")),
            }
        )
    return rows


def write_split_manifest(path: Path, payload: dict[str, Any]) -> None:
    json_dump(path, payload)


def load_default_context() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = project_root()
    config = load_yaml(root / "configs" / "base.yaml")
    cases_config = load_yaml(root / "configs" / "cases.yaml")
    return root, config, cases_config
