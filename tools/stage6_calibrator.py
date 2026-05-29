#!/usr/bin/env python3
"""counter-design step calibrated scoring built from mutation-effect step physics outputs."""

from __future__ import annotations

import json
import math
import pickle
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
CATEGORICAL_FEATURES = ["effect_scope"]
TARGET_COLUMN = "delta_mmgbsa_binding_kcal_mol"
KEY_COLUMNS = ["case_id", "effect_scope", "target_key", "representative_sample_id"]


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _sigmoid(value: float, scale: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(value) / max(float(scale), 1.0e-6))))


def _frame_from_stage5(stage5_root: Path) -> pd.DataFrame:
    ifp_frame = _read_csv_optional(stage5_root / "ifp_diff.csv")
    calibration_frame = _read_csv_optional(stage5_root / "scoring_calibration.csv")
    if ifp_frame.empty and calibration_frame.empty:
        return pd.DataFrame(columns=KEY_COLUMNS + NUMERIC_FEATURES + [TARGET_COLUMN])

    if not ifp_frame.empty:
        frame = ifp_frame.copy()
    else:
        frame = calibration_frame.copy()

    if not calibration_frame.empty:
        calibration_subset = calibration_frame.copy()
        if KEY_COLUMNS[3] not in calibration_subset.columns:
            calibration_subset[KEY_COLUMNS[3]] = pd.NA
        keep_columns = [column for column in KEY_COLUMNS + [TARGET_COLUMN] if column in calibration_subset.columns]
        calibration_subset = calibration_subset.loc[:, keep_columns].drop_duplicates(
            [column for column in KEY_COLUMNS if column in keep_columns]
        )
        merge_keys = [column for column in KEY_COLUMNS if column in frame.columns and column in calibration_subset.columns]
        if merge_keys:
            frame = frame.merge(
                calibration_subset,
                on=merge_keys,
                how="left",
                suffixes=("", "_calibration"),
            )
            calibration_target = f"{TARGET_COLUMN}_calibration"
            if calibration_target in frame.columns:
                frame[TARGET_COLUMN] = frame[calibration_target].combine_first(frame.get(TARGET_COLUMN))
                frame = frame.drop(columns=[calibration_target])

    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype=float)
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="string")
        frame[column] = frame[column].astype("string").fillna("unknown")
    for column in KEY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="string")
        frame[column] = frame[column].astype("string").fillna("")
    return frame


def _model_pipeline(huber_epsilon: float, huber_alpha: float) -> Pipeline:
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
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    regressor = HuberRegressor(
        epsilon=float(huber_epsilon),
        alpha=float(huber_alpha),
        max_iter=400,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("regressor", regressor)])


@dataclass
class Stage6Calibrator:
    pipeline: Pipeline | None
    isotonic: IsotonicRegression | None
    metadata: dict[str, Any]

    @property
    def available(self) -> bool:
        return bool(self.pipeline is not None and bool(self.metadata.get("available", False)))

    def predict_ddg(self, features: dict[str, Any]) -> float | None:
        if not self.available or self.pipeline is None:
            return None
        row = {column: features.get(column) for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
        frame = pd.DataFrame.from_records([row])
        prediction = float(self.pipeline.predict(frame)[0])
        if self.isotonic is not None:
            prediction = float(self.isotonic.predict(np.asarray([prediction], dtype=float))[0])
        return prediction

    def predict_score(self, features: dict[str, Any], scale: float) -> float | None:
        prediction = self.predict_ddg(features)
        if prediction is None:
            return None
        return _sigmoid(-float(prediction), float(scale))


def fit_case_calibrator(
    *,
    stage5_root: Path,
    stage6_root: Path,
    case_id: str,
    stage6: dict[str, Any],
    stage5_qc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    calibrator_root = ensure_dir(stage6_root / "calibrator")
    metadata_path = calibrator_root / "stage6_calibrator.json"
    model_path = calibrator_root / "stage6_calibrator.pkl"

    frame = _frame_from_stage5(stage5_root)
    train_frame = frame.dropna(subset=[TARGET_COLUMN]).copy()
    min_rows = int(stage6.get("calibrator_min_rows", 4))
    min_targets = int(stage6.get("calibrator_min_unique_targets", 3))

    metadata: dict[str, Any] = {
        "case_id": str(case_id),
        "available": False,
        "fitted_at": iso_now(),
        "stage5_root": str(stage5_root),
        "model_path": str(model_path),
        "feature_columns": list(NUMERIC_FEATURES + CATEGORICAL_FEATURES),
        "target_column": TARGET_COLUMN,
        "training_row_count": int(len(train_frame)),
        "training_target_count": int(train_frame["target_key"].nunique()) if "target_key" in train_frame.columns else 0,
        "dock_vs_mmgbsa_pearson_r": None if not stage5_qc else stage5_qc.get("dock_vs_mmgbsa_pearson_r"),
        "source_ifp_diff_csv": str(stage5_root / "ifp_diff.csv"),
        "source_scoring_calibration_csv": str(stage5_root / "scoring_calibration.csv"),
        "reason": "",
    }

    if len(train_frame) < min_rows:
        metadata["reason"] = f"insufficient_rows<{min_rows}"
        json_dump(metadata_path, metadata)
        return metadata
    if "target_key" in train_frame.columns and train_frame["target_key"].nunique() < min_targets:
        metadata["reason"] = f"insufficient_unique_targets<{min_targets}"
        json_dump(metadata_path, metadata)
        return metadata
    required_numeric_features = [str(value) for value in list(stage6.get("required_numeric_features", []))]
    missing_required_features = [
        column
        for column in required_numeric_features
        if column not in train_frame.columns or int(train_frame[column].notna().sum()) <= 0
    ]
    if missing_required_features:
        metadata["reason"] = "missing_required_features:" + ",".join(missing_required_features)
        json_dump(metadata_path, metadata)
        raise RuntimeError(
            "Stage6 calibrator missing required observed features: " + ", ".join(missing_required_features)
        )

    pipeline = _model_pipeline(
        float(stage6.get("calibrator_huber_epsilon", 1.35)),
        float(stage6.get("calibrator_huber_alpha", 1.0e-4)),
    )
    train_x = train_frame.loc[:, NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    train_y = train_frame[TARGET_COLUMN].astype(float).copy()
    pipeline.fit(train_x, train_y)
    predicted = pipeline.predict(train_x)

    isotonic = None
    if len(train_frame) >= int(stage6.get("calibrator_min_rows_for_isotonic", 6)) and np.unique(predicted).size >= 3:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(predicted, train_y.to_numpy(dtype=float))

    raw_series = pd.Series(predicted, dtype=float)
    calibrated_predicted = isotonic.predict(predicted) if isotonic is not None else predicted
    calibrated_series = pd.Series(calibrated_predicted, dtype=float)
    metadata.update(
        {
            "available": True,
            "reason": "ok",
            "target_mean": float(train_y.mean()),
            "target_std": float(train_y.std(ddof=0)) if len(train_y) > 1 else 0.0,
            "raw_fit_pearson_r": None if raw_series.nunique() <= 1 else float(raw_series.corr(train_y, method="pearson")),
            "calibrated_fit_pearson_r": None
            if calibrated_series.nunique() <= 1
            else float(calibrated_series.corr(train_y, method="pearson")),
        }
    )
    payload = {
        "pipeline": pipeline,
        "isotonic": isotonic,
        "metadata": metadata,
    }
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle)
    json_dump(metadata_path, metadata)
    return metadata


@lru_cache(maxsize=16)
def load_case_calibrator(path_str: str) -> Stage6Calibrator:
    path = Path(path_str)
    if not path.exists():
        return Stage6Calibrator(pipeline=None, isotonic=None, metadata={"available": False, "reason": "missing_model"})
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return Stage6Calibrator(
        pipeline=payload.get("pipeline"),
        isotonic=payload.get("isotonic"),
        metadata=dict(payload.get("metadata") or {}),
    )
