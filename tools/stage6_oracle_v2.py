#!/usr/bin/env python3
"""counter-design step oracle v2 training and audit helpers."""

from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tools.runtime import ensure_dir, iso_now, json_dump

NUMERIC_FEATURES = [
    "delta_dock_kcal_mol",
    "delta_gnina_affinity_kcal_mol",
    "ifp_jaccard_loss",
    "ifp_occupancy_shift_mean_abs",
    "ifp_occupancy_anchor_loss",
    "anchor_loss_fraction",
    "pocket_volume_change_fraction",
    "solvent_proxy_shift",
    "stage4_local_rmsd_a",
]
CATEGORICAL_FEATURES = [
    "effect_scope",
    "sample_source",
    "target_domain",
]
BOOLEAN_FEATURES = [
    "is_partner_chain_sensitive",
    "used_synthetic_combo_model",
]
TARGET_COLUMN = "delta_mmgbsa_binding_kcal_mol"
META_COLUMNS = [
    "case_id",
    "target_name",
    "target_domain",
    "target_key",
    "effect_scope",
    "sample_source",
]


def _sigmoid(value: float, scale: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(value) / max(float(scale), 1.0e-6))))


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _safe_corr(lhs: pd.Series, rhs: pd.Series, method: str) -> float | None:
    valid = lhs.notna() & rhs.notna()
    if int(valid.sum()) < 3:
        return None
    lhs_valid = lhs[valid]
    rhs_valid = rhs[valid]
    if lhs_valid.nunique() <= 1 or rhs_valid.nunique() <= 1:
        return None
    return float(lhs_valid.corr(rhs_valid, method=method))


def _trust_score(
    *,
    pearson: float | None,
    spearman: float | None,
    train_rows: int,
    unique_targets: int,
    high_uncertainty_rate: float,
    mmgbsa_missing_rate: float,
) -> float:
    pearson_value = max(0.0, float(pearson or 0.0))
    spearman_value = max(0.0, float(spearman or 0.0))
    return float(
        0.35 * pearson_value
        + 0.15 * spearman_value
        + 0.15 * min(float(train_rows) / 20.0, 1.0)
        + 0.15 * min(float(unique_targets) / 10.0, 1.0)
        + 0.10 * max(0.0, 1.0 - float(high_uncertainty_rate))
        + 0.10 * max(0.0, 1.0 - float(mmgbsa_missing_rate))
    )


def resolve_stage5_root(
    *,
    root: Path,
    case_id: str,
    preferred_subdir: str = "stage5",
    fallback_subdirs: list[str] | None = None,
) -> Path | None:
    candidates = [preferred_subdir, *(fallback_subdirs or [])]
    for subdir in candidates:
        candidate = root / "outputs" / case_id / subdir
        if (candidate / "ifp_diff.csv").exists():
            return candidate
    return None


def _partner_chain_positions(root: Path, case_entry: dict[str, Any]) -> set[int]:
    case_id = str(case_entry["case_id"])
    payload_path = root / "outputs" / case_id / "stage3_5" / "wt_ifp_multichain.json"
    if not payload_path.exists():
        return set()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    primary_chain = str(case_entry.get("wt_template", {}).get("chain_id") or "")
    reference_template = dict(payload.get("reference_template") or {})
    reference_chain_ids = [str(value) for value in list(reference_template.get("reference_holo_chain_ids") or [])]
    partner_chain_ids = {chain_id for chain_id in reference_chain_ids if chain_id and chain_id != primary_chain}
    if not partner_chain_ids:
        return set()
    positions: set[int] = set()
    residue_labels = list(payload.get("partner_chain_residues") or [])
    if not residue_labels:
        residue_labels = [
            residue_label
            for residue_label in list((payload.get("pocket_residue_universe") or []))
            if str(residue_label).split(":", 1)[0] in partner_chain_ids
        ]
    for residue_label in residue_labels:
        match = re.match(r"(?P<chain>[^:]+):[A-Z]{3}(?P<position>\d+)", str(residue_label))
        if not match:
            continue
        if str(match.group("chain")) not in partner_chain_ids:
            continue
        positions.add(int(match.group("position")))
    return positions


def _target_positions(target_key: str) -> set[int]:
    return {int(value) for value in re.findall(r"(\d+)", str(target_key))}


def _frame_from_stage5(stage5_root: Path) -> pd.DataFrame:
    ifp_frame = read_csv_optional(stage5_root / "ifp_diff.csv")
    calibration_frame = read_csv_optional(stage5_root / "scoring_calibration.csv")
    if ifp_frame.empty and calibration_frame.empty:
        return pd.DataFrame()
    frame = ifp_frame.copy() if not ifp_frame.empty else calibration_frame.copy()
    if not calibration_frame.empty:
        keep_columns = [
            column
            for column in ["case_id", "effect_scope", "target_key", TARGET_COLUMN]
            if column in calibration_frame.columns
        ]
        if keep_columns:
            calibration_subset = calibration_frame.loc[:, keep_columns].drop_duplicates(
                [column for column in ["case_id", "effect_scope", "target_key"] if column in keep_columns]
            )
            merge_keys = [column for column in ["case_id", "effect_scope", "target_key"] if column in frame.columns and column in calibration_subset.columns]
            if merge_keys:
                frame = frame.merge(calibration_subset, on=merge_keys, how="left", suffixes=("", "_calibration"))
                calibration_target = f"{TARGET_COLUMN}_calibration"
                if calibration_target in frame.columns:
                    frame[TARGET_COLUMN] = frame[calibration_target].combine_first(frame.get(TARGET_COLUMN))
                    frame = frame.drop(columns=[calibration_target])
    return frame


def build_stage6_oracle_training_frame(
    *,
    root: Path,
    case_entries: list[dict[str, Any]],
    preferred_stage5_subdir: str = "stage5",
    fallback_stage5_subdirs: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for case_entry in case_entries:
        case_id = str(case_entry["case_id"])
        stage5_root = resolve_stage5_root(
            root=root,
            case_id=case_id,
            preferred_subdir=preferred_stage5_subdir,
            fallback_subdirs=fallback_stage5_subdirs,
        )
        if stage5_root is None:
            continue
        frame = _frame_from_stage5(stage5_root)
        if frame.empty:
            continue
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
        frame["case_id"] = str(case_id)
        frame["target_name"] = str(case_entry.get("target_name") or "")
        frame["target_domain"] = str(case_entry.get("target_domain") or "unknown").lower()
        partner_positions = _partner_chain_positions(root, case_entry)
        frame["is_partner_chain_sensitive"] = frame.get("target_key", pd.Series(dtype=str)).fillna("").astype(str).map(
            lambda value: bool(_target_positions(value) & partner_positions)
        )
        if "sample_source" not in frame.columns:
            frame["sample_source"] = "unknown"
        frame["sample_source"] = frame["sample_source"].fillna("unknown").astype(str)
        if "used_synthetic_combo_model" not in frame.columns:
            frame["used_synthetic_combo_model"] = False
        frame["used_synthetic_combo_model"] = frame["used_synthetic_combo_model"].fillna(False).astype(bool)
        if "high_uncertainty" not in frame.columns:
            frame["high_uncertainty"] = False
        frame["high_uncertainty"] = frame["high_uncertainty"].fillna(False).astype(bool)
        for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
            if column not in frame.columns:
                frame[column] = pd.Series(dtype=float)
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in CATEGORICAL_FEATURES:
            if column not in frame.columns:
                frame[column] = pd.Series(dtype="string")
            frame[column] = frame[column].astype("string").fillna("unknown")
        rows.append(frame)
    if not rows:
        columns = list(dict.fromkeys(META_COLUMNS + NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES + [TARGET_COLUMN]))
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)


def _model_pipeline(stage6: dict[str, Any]) -> Pipeline:
    numeric_columns = NUMERIC_FEATURES + BOOLEAN_FEATURES
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    regressor = HuberRegressor(
        epsilon=float(stage6.get("oracle_v2_huber_epsilon", 1.35)),
        alpha=float(stage6.get("oracle_v2_huber_alpha", 1.0e-4)),
        max_iter=400,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("regressor", regressor)])


@dataclass
class OracleRegressionModel:
    pipeline: Pipeline | None
    isotonic: IsotonicRegression | None
    metadata: dict[str, Any]
    residual_std: float | None

    @property
    def available(self) -> bool:
        return bool(self.pipeline is not None and bool(self.metadata.get("available", False)))

    def predict_ddg(self, features: dict[str, Any]) -> float | None:
        if not self.available or self.pipeline is None:
            return None
        row = {column: features.get(column) for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES}
        frame = pd.DataFrame.from_records([row])
        frame[BOOLEAN_FEATURES] = frame[BOOLEAN_FEATURES].fillna(False).astype(float)
        prediction = float(self.pipeline.predict(frame)[0])
        if self.isotonic is not None:
            prediction = float(self.isotonic.predict(np.asarray([prediction], dtype=float))[0])
        return prediction

    def predict_score(self, features: dict[str, Any], scale: float) -> float | None:
        prediction = self.predict_ddg(features)
        if prediction is None:
            return None
        return _sigmoid(-float(prediction), float(scale))


@dataclass
class Stage6OracleV2:
    case_model: OracleRegressionModel
    domain_model: OracleRegressionModel
    metadata: dict[str, Any]

    @property
    def available(self) -> bool:
        return bool(self.case_model.available or self.domain_model.available or self.metadata.get("available_models"))

    def predict(self, features: dict[str, Any], scale: float) -> dict[str, Any]:
        predictions: list[float] = []
        provenance: list[str] = []
        per_model: dict[str, dict[str, Any]] = {}
        for name, model in [("case", self.case_model), ("domain", self.domain_model)]:
            prediction = model.predict_ddg(features)
            if prediction is None:
                continue
            predictions.append(float(prediction))
            provenance.append(name)
            per_model[name] = {
                "pred_ddg": float(prediction),
                "pred_score": _sigmoid(-float(prediction), scale),
                "residual_std": None if model.residual_std is None else float(model.residual_std),
                "trust_score": float(model.metadata.get("trust_score") or 0.0),
                "label": str(model.metadata.get("label") or name),
            }
        if not predictions:
            return {
                "available": False,
                "pred_mean": None,
                "pred_std": None,
                "pred_conservative": None,
                "score_mean": None,
                "score_conservative": None,
                "trust_score": float(self.metadata.get("trust_score") or 0.0),
                "member_predictions": per_model,
                "provenance": provenance,
            }
        residual_terms = [float(value) for value in [self.case_model.residual_std, self.domain_model.residual_std] if value is not None]
        model_variance = float(np.var(predictions)) if len(predictions) > 1 else 0.0
        residual_variance = float(np.mean([value * value for value in residual_terms])) if residual_terms else 0.0
        pred_mean = float(np.mean(predictions))
        pred_std = float(math.sqrt(max(0.0, model_variance + residual_variance)))
        ensemble_metadata = dict(self.metadata.get("ensemble_model") or {})
        residual_weight = float(ensemble_metadata.get("residual_weight", 1.0))
        pred_conservative = float(max(predictions) + residual_weight * pred_std)
        return {
            "available": True,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "pred_conservative": pred_conservative,
            "score_mean": _sigmoid(-pred_mean, scale),
            "score_conservative": _sigmoid(-pred_conservative, scale),
            "trust_score": float(self.metadata.get("trust_score") or 0.0),
            "member_predictions": per_model,
            "provenance": provenance,
        }


def conservative_ensemble_metadata(
    *,
    case_model: OracleRegressionModel,
    domain_model: OracleRegressionModel,
    residual_weight: float,
) -> dict[str, Any]:
    available_models = [
        name
        for name, model in [("case", case_model), ("domain", domain_model)]
        if model.available
    ]
    available_scores = [
        float(model.metadata.get("trust_score") or 0.0)
        for model in [case_model, domain_model]
        if model.available
    ]
    return {
        "label": "conservative_ensemble",
        "available": bool(available_models),
        "strategy": "max_member_plus_residual_std",
        "component_models": available_models,
        "residual_weight": float(residual_weight),
        "training_row_count": int(
            max(
                int(case_model.metadata.get("training_row_count", 0) or 0),
                int(domain_model.metadata.get("training_row_count", 0) or 0),
            )
        ),
        "training_target_count": int(
            max(
                int(case_model.metadata.get("training_target_count", 0) or 0),
                int(domain_model.metadata.get("training_target_count", 0) or 0),
            )
        ),
        "trust_score": float(min(available_scores)) if available_scores else 0.0,
    }


def conservative_ensemble_holdout(
    *,
    case_metrics: dict[str, Any],
    domain_metrics: dict[str, Any],
    residual_weight: float,
) -> dict[str, Any]:
    available_labels = [
        label
        for label, payload in [("case", case_metrics), ("domain", domain_metrics)]
        if bool(payload.get("available", False))
    ]
    trust_scores = [
        float(payload.get("trust_score") or 0.0)
        for payload in [case_metrics, domain_metrics]
        if bool(payload.get("available", False))
    ]
    return {
        "label": "conservative_ensemble_holdout",
        "available": bool(available_labels),
        "strategy": "max_member_plus_residual_std",
        "component_models": available_labels,
        "residual_weight": float(residual_weight),
        "trust_score": float(min(trust_scores)) if trust_scores else 0.0,
    }


def _fit_single_model(
    *,
    frame: pd.DataFrame,
    source_frame: pd.DataFrame,
    stage6: dict[str, Any],
    label: str,
) -> OracleRegressionModel:
    min_rows = int(stage6.get("oracle_v2_min_rows", 6))
    min_targets = int(stage6.get("oracle_v2_min_unique_targets", 4))
    metadata: dict[str, Any] = {
        "label": label,
        "available": False,
        "training_row_count": int(len(frame)),
        "training_target_count": int(frame["target_key"].nunique()) if "target_key" in frame.columns else 0,
        "fitted_at": iso_now(),
        "reason": "",
    }
    if len(frame) < min_rows:
        metadata["reason"] = f"insufficient_rows<{min_rows}"
        metadata["trust_score"] = 0.0
        return OracleRegressionModel(None, None, metadata, None)
    if "target_key" in frame.columns and frame["target_key"].nunique() < min_targets:
        metadata["reason"] = f"insufficient_unique_targets<{min_targets}"
        metadata["trust_score"] = 0.0
        return OracleRegressionModel(None, None, metadata, None)
    train_x = frame.loc[:, NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES].copy()
    train_x[BOOLEAN_FEATURES] = train_x[BOOLEAN_FEATURES].fillna(False).astype(float)
    train_y = frame[TARGET_COLUMN].astype(float).copy()
    pipeline = _model_pipeline(stage6)
    pipeline.fit(train_x, train_y)
    predicted = pd.Series(pipeline.predict(train_x), index=train_y.index, dtype=float)
    isotonic = None
    min_rows_for_isotonic = int(stage6.get("oracle_v2_min_rows_for_isotonic", 8))
    if len(frame) >= min_rows_for_isotonic and predicted.nunique() >= 3:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(predicted.to_numpy(dtype=float), train_y.to_numpy(dtype=float))
        calibrated = pd.Series(isotonic.predict(predicted.to_numpy(dtype=float)), index=train_y.index, dtype=float)
    else:
        calibrated = predicted.copy()
    residual_std = float(np.sqrt(np.mean(np.square(calibrated - train_y)))) if len(train_y) else 0.0
    high_uncertainty_rate = float(source_frame.get("high_uncertainty", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()) if not source_frame.empty else 0.0
    mmgbsa_missing_rate = 1.0 - float(source_frame[TARGET_COLUMN].notna().mean()) if TARGET_COLUMN in source_frame.columns and not source_frame.empty else 1.0
    required_features = [str(value) for value in list(stage6.get("required_numeric_features", []))]
    required_feature_coverages = []
    missing_required_features: list[str] = []
    for column in required_features:
        if column not in source_frame.columns:
            required_feature_coverages.append(0.0)
            missing_required_features.append(column)
            continue
        coverage = float(pd.to_numeric(source_frame[column], errors="coerce").notna().mean()) if not source_frame.empty else 0.0
        required_feature_coverages.append(coverage)
        if coverage <= 0.0:
            missing_required_features.append(column)
    required_feature_coverage = float(np.mean(required_feature_coverages)) if required_feature_coverages else 1.0
    pearson = _safe_corr(calibrated, train_y, "pearson")
    spearman = _safe_corr(calibrated, train_y, "spearman")
    trust_score = _trust_score(
        pearson=pearson,
        spearman=spearman,
        train_rows=int(len(frame)),
        unique_targets=int(frame["target_key"].nunique()) if "target_key" in frame.columns else 0,
        high_uncertainty_rate=high_uncertainty_rate,
        mmgbsa_missing_rate=mmgbsa_missing_rate,
    ) * required_feature_coverage
    metadata.update(
        {
            "available": True,
            "reason": "ok" if not missing_required_features else "missing_required_features:" + ",".join(missing_required_features),
            "target_mean": float(train_y.mean()),
            "target_std": float(train_y.std(ddof=0)) if len(train_y) > 1 else 0.0,
            "fit_pearson_r": pearson,
            "fit_spearman_r": spearman,
            "high_uncertainty_rate": float(high_uncertainty_rate),
            "mmgbsa_missing_rate": float(mmgbsa_missing_rate),
            "required_feature_coverage": float(required_feature_coverage),
            "missing_required_features": missing_required_features,
            "residual_std": float(residual_std),
            "trust_score": float(trust_score),
        }
    )
    return OracleRegressionModel(pipeline, isotonic, metadata, residual_std)


def fit_stage6_oracle_v2(
    *,
    root: Path,
    case_entries: list[dict[str, Any]],
    focus_case_id: str,
    stage6: dict[str, Any],
    preferred_stage5_subdir: str = "stage5",
    fallback_stage5_subdirs: list[str] | None = None,
) -> dict[str, Any]:
    frame = build_stage6_oracle_training_frame(
        root=root,
        case_entries=case_entries,
        preferred_stage5_subdir=preferred_stage5_subdir,
        fallback_stage5_subdirs=fallback_stage5_subdirs,
    )
    focus_case = next(case for case in case_entries if str(case["case_id"]) == str(focus_case_id))
    focus_domain = str(focus_case.get("target_domain") or "unknown").lower()
    output_root = ensure_dir(root / "outputs" / str(focus_case_id) / str(stage6.get("oracle_v2_output_dirname", "stage6_oracle_v2")))
    if frame.empty:
        metadata = {
            "case_id": str(focus_case_id),
            "target_domain": focus_domain,
            "generated_at": iso_now(),
            "preferred_stage5_subdir": preferred_stage5_subdir,
            "fallback_stage5_subdirs": list(fallback_stage5_subdirs or []),
            "training_frame_path": str(output_root / "training_frame.csv"),
            "model_path": str(output_root / "stage6_oracle_v2.pkl"),
            "case_model": {"available": False, "reason": "missing_training_frame", "trust_score": 0.0},
            "domain_model": {"available": False, "reason": "missing_training_frame", "trust_score": 0.0},
            "ensemble_model": {
                "available": False,
                "reason": "missing_training_frame",
                "strategy": "max_member_plus_residual_std",
                "residual_weight": float(stage6.get("oracle_v2_conservative_residual_weight", 1.0)),
                "trust_score": 0.0,
            },
            "available_models": [],
            "trust_score": 0.0,
            "mode": "hypothesis_only",
        }
        frame.to_csv(output_root / "training_frame.csv", index=False)
        json_dump(output_root / "stage6_oracle_v2.json", metadata)
        return metadata
    case_source = frame[frame["case_id"].eq(str(focus_case_id))].copy()
    domain_source = frame[frame["target_domain"].eq(focus_domain)].copy()
    case_train = case_source.dropna(subset=[TARGET_COLUMN]).copy()
    domain_train = domain_source.dropna(subset=[TARGET_COLUMN]).copy()
    case_model = _fit_single_model(frame=case_train, source_frame=case_source, stage6=stage6, label="case")
    domain_model = _fit_single_model(frame=domain_train, source_frame=domain_source, stage6=stage6, label="domain")
    ensemble_model = conservative_ensemble_metadata(
        case_model=case_model,
        domain_model=domain_model,
        residual_weight=float(stage6.get("oracle_v2_conservative_residual_weight", 1.0)),
    )

    available_scores = [float(model.metadata.get("trust_score") or 0.0) for model in [case_model, domain_model] if model.available]
    trust_score = float(min(available_scores)) if available_scores else 0.0
    metadata = {
        "case_id": str(focus_case_id),
        "target_domain": focus_domain,
        "generated_at": iso_now(),
        "preferred_stage5_subdir": preferred_stage5_subdir,
        "fallback_stage5_subdirs": list(fallback_stage5_subdirs or []),
        "training_frame_path": str(output_root / "training_frame.csv"),
        "model_path": str(output_root / "stage6_oracle_v2.pkl"),
        "case_model": dict(case_model.metadata),
        "domain_model": dict(domain_model.metadata),
        "ensemble_model": ensemble_model,
        "available_models": [name for name, model in [("case", case_model), ("domain", domain_model)] if model.available],
        "trust_score": trust_score,
        "mode": (
            "normal_robust_optimization"
            if trust_score >= 0.60
            else "uncertainty_heavy"
            if trust_score >= 0.45
            else "hypothesis_only"
        ),
    }
    frame.to_csv(output_root / "training_frame.csv", index=False)
    payload = {
        "case_model": case_model,
        "domain_model": domain_model,
        "metadata": metadata,
    }
    with (output_root / "stage6_oracle_v2.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    json_dump(output_root / "stage6_oracle_v2.json", metadata)
    return metadata


@lru_cache(maxsize=16)
def load_stage6_oracle_v2(path_str: str) -> Stage6OracleV2:
    path = Path(path_str)
    if not path.exists():
        unavailable = OracleRegressionModel(None, None, {"available": False, "reason": "missing_model", "trust_score": 0.0}, None)
        return Stage6OracleV2(unavailable, unavailable, {"available_models": [], "trust_score": 0.0})
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return Stage6OracleV2(
        case_model=payload["case_model"],
        domain_model=payload["domain_model"],
        metadata=dict(payload.get("metadata") or {}),
    )


def evaluate_group_holdout(
    *,
    frame: pd.DataFrame,
    stage6: dict[str, Any],
    group_column: str,
    label: str,
) -> dict[str, Any]:
    observed = frame.dropna(subset=[TARGET_COLUMN]).copy()
    if observed.empty or group_column not in observed.columns:
        return {
            "label": label,
            "available": False,
            "reason": "missing_training_frame",
        }
    group_count = int(observed[group_column].nunique())
    if group_count < 3:
        return {
            "label": label,
            "available": False,
            "reason": "insufficient_groups",
            "group_count": group_count,
        }
    split_count = min(5, group_count)
    splitter = GroupKFold(n_splits=split_count)
    predictions: list[pd.DataFrame] = []
    for fold_index, (train_idx, test_idx) in enumerate(
        splitter.split(observed, groups=observed[group_column].astype(str)),
        start=1,
    ):
        train_frame = observed.iloc[train_idx].copy()
        test_frame = observed.iloc[test_idx].copy()
        model = _fit_single_model(frame=train_frame, source_frame=train_frame, stage6=stage6, label=f"{label}_fold_{fold_index}")
        if not model.available:
            continue
        test_x = test_frame.loc[:, NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES].copy()
        test_x[BOOLEAN_FEATURES] = test_x[BOOLEAN_FEATURES].fillna(False).astype(float)
        predicted = pd.Series(model.pipeline.predict(test_x), index=test_frame.index, dtype=float)
        if model.isotonic is not None:
            predicted = pd.Series(model.isotonic.predict(predicted.to_numpy(dtype=float)), index=test_frame.index, dtype=float)
        predictions.append(
            pd.DataFrame(
                {
                    "fold_index": fold_index,
                    "truth": test_frame[TARGET_COLUMN].astype(float),
                    "prediction": predicted.astype(float),
                }
            )
        )
    if not predictions:
        return {
            "label": label,
            "available": False,
            "reason": "no_fold_predictions",
        }
    prediction_frame = pd.concat(predictions, ignore_index=True)
    truth = prediction_frame["truth"].astype(float)
    prediction = prediction_frame["prediction"].astype(float)
    mae = float(np.mean(np.abs(prediction - truth)))
    rmse = float(np.sqrt(np.mean(np.square(prediction - truth))))
    pearson = _safe_corr(prediction, truth, "pearson")
    spearman = _safe_corr(prediction, truth, "spearman")
    high_uncertainty_rate = float(frame.get("high_uncertainty", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()) if not frame.empty else 0.0
    mmgbsa_missing_rate = 1.0 - float(frame[TARGET_COLUMN].notna().mean()) if TARGET_COLUMN in frame.columns and not frame.empty else 1.0
    required_features = [str(value) for value in list(stage6.get("required_numeric_features", []))]
    required_feature_coverages = []
    for column in required_features:
        if column not in frame.columns:
            required_feature_coverages.append(0.0)
            continue
        required_feature_coverages.append(float(pd.to_numeric(frame[column], errors="coerce").notna().mean()))
    required_feature_coverage = float(np.mean(required_feature_coverages)) if required_feature_coverages else 1.0
    return {
        "label": label,
        "available": True,
        "row_count": int(len(frame)),
        "observed_row_count": int(len(observed)),
        "group_count": group_count,
        "fold_count": split_count,
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "required_feature_coverage": float(required_feature_coverage),
        "trust_score": _trust_score(
            pearson=pearson,
            spearman=spearman,
            train_rows=int(len(observed)),
            unique_targets=int(observed["target_key"].nunique()) if "target_key" in observed.columns else 0,
            high_uncertainty_rate=high_uncertainty_rate,
            mmgbsa_missing_rate=mmgbsa_missing_rate,
        )
        * required_feature_coverage,
    }
