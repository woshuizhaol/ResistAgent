#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.stage7_utils import (
    BENCHMARK_REQUIRED_PRIOR,
    build_decoy_frame,
    build_decoy_metrics,
    build_split_manifest,
    ensure_blind_prior_path,
    fold_test_mask,
    iter_ready_folds,
)


def _benchmark_frame() -> pd.DataFrame:
    rows = []
    for pair_index in range(4):
        uniprot = f"P{pair_index:05d}"
        drug = f"Drug{pair_index}"
        for row_index in range(6):
            rows.append(
                {
                    "SAMPLE_ID": f"S{pair_index}_{row_index}",
                    "UNIPROT_ID": uniprot,
                    "drug_name": drug,
                    "group_key": f"{uniprot}||{drug}",
                    "DATASET": "Platinum" if pair_index == 0 else "GDSC",
                    "TYPE": "Single Substitution" if row_index < 5 else "Multiple Substitution",
                    "DDG.EXP": 0.05 if row_index == 0 else (1.5 + row_index),
                    "P_background": 0.001 if row_index == 0 else 0.2,
                    "P_drug_selected": 0.001 if row_index == 0 else 0.3,
                    "MUT.Volume": 10.0 if row_index == 0 else 80.0,
                    "MUT.NetCharge": 0.0,
                    "set_bucket": "MdrDB_holdout",
                    "source_date": pd.Timestamp("2020-01-01") if pair_index < 2 else pd.Timestamp("2023-10-27"),
                    "source_year": 2020 if pair_index < 2 else 2023,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_ensure_blind_prior_path_rejects_global_prior() -> None:
    ensure_blind_prior_path(Path(BENCHMARK_REQUIRED_PRIOR))
    with pytest.raises(RuntimeError):
        ensure_blind_prior_path(Path("global_prior.parquet"))


def test_build_split_manifest_includes_pair_target_drug_and_time_entries() -> None:
    frame = _benchmark_frame()
    manifest = build_split_manifest(
        frame,
        {"stage7": {
            "groupkfold_splits": 3,
            "min_group_size": 4,
            "leave_one_target_top_n": 2,
            "leave_one_drug_top_n": 2,
            "external_holdout_datasets": ["Platinum"],
            "external_holdout_min_rows": 2,
            "time_split_min_train_rows": 4,
            "time_split_min_test_rows": 4,
        }},
        {"set_d": [{"uniprot_id": "P00000", "drug_name": "Drug0"}]},
    )

    strategies = {row["strategy"]: row for row in manifest["strategies"]}
    assert strategies["pair_groupkfold"]["status"] == "ready"
    assert len(strategies["pair_groupkfold"]["folds"]) == 3
    assert strategies["leave_one_target_out"]["status"] == "ready"
    assert strategies["leave_one_drug_out"]["status"] == "ready"
    assert strategies["external_holdout"]["status"] == "ready"
    assert strategies["time_split"]["status"] == "ready"
    assert len(strategies["time_split"]["folds"]) >= 1
    assert manifest["set_boundary"]["set_n_external_included_in_main_benchmark"] is False


def test_fold_test_mask_and_decoy_selection_are_test_only() -> None:
    frame = _benchmark_frame()
    fold = {
        "strategy": "leave_one_target_out",
        "fold_id": "leave_target_P00000",
        "test_uniprot_id": "P00000",
    }
    mask = fold_test_mask(frame, fold)
    test_frame = frame[mask].copy()
    decoy = build_decoy_frame(
        frame,
        test_frame,
        {
            "decoy_ddg_max": 0.25,
            "decoy_background_max": 0.01,
            "decoy_drug_selected_max": 0.01,
            "decoy_abs_mut_volume_max": 40.0,
            "decoy_abs_net_charge_max": 1.0,
        },
        fold,
    )

    assert set(decoy["UNIPROT_ID"].unique()) == {"P00000"}
    assert set(decoy["SAMPLE_ID"].tolist()) == {"S0_0"}
    assert decoy["decoy_label"].all()


def test_fold_test_mask_supports_time_split() -> None:
    frame = _benchmark_frame()
    fold = {
        "strategy": "time_split",
        "fold_id": "time_after_2020",
        "cutoff_date": "2020-12-31",
    }
    mask = fold_test_mask(frame, fold)
    assert int(mask.sum()) == 12
    assert set(frame.loc[mask, "UNIPROT_ID"].unique()) == {"P00002", "P00003"}


def test_build_decoy_metrics_reports_model_and_random_rows() -> None:
    predictions = pd.DataFrame.from_records(
        [
            {"SAMPLE_ID": "S0", "group_key": "G1", "feature_set": "ours", "split_strategy": "pair_groupkfold", "fold_id": "f1", "prediction": 0.9},
            {"SAMPLE_ID": "S1", "group_key": "G1", "feature_set": "ours", "split_strategy": "pair_groupkfold", "fold_id": "f1", "prediction": 0.1},
            {"SAMPLE_ID": "S0", "group_key": "G1", "feature_set": "frequency_only", "split_strategy": "pair_groupkfold", "fold_id": "f1", "prediction": 0.8},
            {"SAMPLE_ID": "S1", "group_key": "G1", "feature_set": "frequency_only", "split_strategy": "pair_groupkfold", "fold_id": "f1", "prediction": 0.2},
        ]
    )
    decoys = pd.DataFrame.from_records(
        [
            {"SAMPLE_ID": "S0", "split_strategy": "pair_groupkfold", "fold_id": "f1", "decoy_label": True},
        ]
    )

    rows = build_decoy_metrics(predictions, decoys, stage7={"precision_at_k": [1], "random_seed": 7})

    feature_sets = {(row["feature_set"], row["k"]) for row in rows}
    assert ("ours", 1) in feature_sets
    assert ("frequency_only", 1) in feature_sets
    assert ("random", 1) in feature_sets
    ours = [row for row in rows if row["feature_set"] == "ours"][0]
    assert ours["topk_decoy_hits"] == 1
