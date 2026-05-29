#!/usr/bin/env python3
"""Repair Stage 7 decoy specificity metrics with group-level top-k burden."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import ensure_dir, iso_now, json_dump, load_yaml, project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--predictions", default="outputs/benchmark/benchmark_predictions.parquet")
    parser.add_argument("--decoys", default="outputs/benchmark/decoy_mutations.parquet")
    parser.add_argument("--output", default="outputs/benchmark/decoy_metrics_repaired.csv")
    parser.add_argument("--group-output", default="outputs/benchmark/decoy_group_topk_repaired.csv")
    parser.add_argument("--summary-json", default="outputs/benchmark/decoy_metrics_repaired_summary.json")
    parser.add_argument("--k", type=int, nargs="*", default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=None)
    return parser.parse_args()


def _effective_rank_score(frame: pd.DataFrame) -> pd.Series:
    rank_score = pd.to_numeric(frame.get("rank_score"), errors="coerce")
    prediction = pd.to_numeric(frame.get("prediction"), errors="coerce")
    return rank_score.where(rank_score.notna(), prediction)


def _ci(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def _bootstrap_group_mean(values: np.ndarray, rng: np.random.Generator, n_iter: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1 or n_iter <= 0:
        value = float(values[0])
        return value, value
    draws = rng.integers(0, values.size, size=(int(n_iter), values.size))
    means = values[draws].mean(axis=1)
    return _ci(means)


def _simulate_random_topk(
    group_sizes: np.ndarray,
    decoy_counts: np.ndarray,
    topk_sizes: np.ndarray,
    rng: np.random.Generator,
    n_iter: int,
) -> dict[str, Any]:
    n_iter = max(1, int(n_iter))
    frac_values = np.empty(n_iter, dtype=float)
    any_values = np.empty(n_iter, dtype=float)
    for iteration in range(n_iter):
        hits = rng.hypergeometric(
            ngood=decoy_counts.astype(int),
            nbad=(group_sizes - decoy_counts).astype(int),
            nsample=topk_sizes.astype(int),
        )
        frac_values[iteration] = np.mean(hits / np.maximum(1, topk_sizes))
        any_values[iteration] = np.mean(hits > 0)
    frac_low, frac_high = _ci(frac_values)
    any_low, any_high = _ci(any_values)
    return {
        "random_group_topk_decoy_fraction_mean": float(frac_values.mean()),
        "random_group_topk_decoy_fraction_ci_low": frac_low,
        "random_group_topk_decoy_fraction_ci_high": frac_high,
        "random_group_topk_any_decoy_rate_mean": float(any_values.mean()),
        "random_group_topk_any_decoy_rate_ci_low": any_low,
        "random_group_topk_any_decoy_rate_ci_high": any_high,
        "random_fraction_draws": frac_values,
        "random_any_draws": any_values,
    }


def repaired_decoy_metrics(
    predictions: pd.DataFrame,
    decoys: pd.DataFrame,
    *,
    k_values: list[int],
    seed: int,
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_prediction_cols = [
        "SAMPLE_ID",
        "feature_set",
        "split_strategy",
        "fold_id",
        "group_key",
        "prediction",
    ]
    missing_prediction = [col for col in required_prediction_cols if col not in predictions.columns]
    if missing_prediction:
        raise ValueError(f"Missing prediction columns: {missing_prediction}")
    required_decoy_cols = ["SAMPLE_ID", "split_strategy", "fold_id", "decoy_label"]
    missing_decoy = [col for col in required_decoy_cols if col not in decoys.columns]
    if missing_decoy:
        raise ValueError(f"Missing decoy columns: {missing_decoy}")

    decoy_labels = decoys[required_decoy_cols].drop_duplicates()
    joined = predictions.merge(decoy_labels, on=["SAMPLE_ID", "split_strategy", "fold_id"], how="left")
    joined["decoy_label"] = pd.to_numeric(joined["decoy_label"], errors="coerce").fillna(0).astype(bool)
    joined["effective_rank_score"] = _effective_rank_score(joined)
    joined = joined[joined["effective_rank_score"].notna()].copy()

    group_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    sort_cols = ["feature_set", "split_strategy", "fold_id", "group_key", "effective_rank_score", "SAMPLE_ID"]
    sorted_frame = joined.sort_values(sort_cols, ascending=[True, True, True, True, False, True])

    for (feature_set, split_strategy), split_frame in sorted_frame.groupby(["feature_set", "split_strategy"], dropna=False):
        grouped = split_frame.groupby(["fold_id", "group_key"], dropna=False, sort=False)
        group_size_series = grouped.size().rename("group_size")
        decoy_count_series = grouped["decoy_label"].sum().rename("group_decoy_count")
        base_group = pd.concat([group_size_series, decoy_count_series], axis=1).reset_index()
        base_group["feature_set"] = str(feature_set)
        base_group["split_strategy"] = str(split_strategy)
        base_group["group_decoy_prevalence"] = (
            pd.to_numeric(base_group["group_decoy_count"], errors="coerce")
            / pd.to_numeric(base_group["group_size"], errors="coerce").replace(0, np.nan)
        )

        for k in k_values:
            topk = grouped.head(int(k))
            topk_grouped = topk.groupby(["fold_id", "group_key"], dropna=False, sort=False)
            topk_size = topk_grouped.size().rename("topk_size")
            topk_decoy = topk_grouped["decoy_label"].sum().rename("topk_decoy_count")
            topk_frame = base_group.merge(
                pd.concat([topk_size, topk_decoy], axis=1).reset_index(),
                on=["fold_id", "group_key"],
                how="left",
            )
            topk_frame["k"] = int(k)
            topk_frame["topk_size"] = pd.to_numeric(topk_frame["topk_size"], errors="coerce").fillna(0).astype(int)
            topk_frame["topk_decoy_count"] = (
                pd.to_numeric(topk_frame["topk_decoy_count"], errors="coerce").fillna(0).astype(int)
            )
            topk_frame["group_topk_decoy_fraction"] = (
                topk_frame["topk_decoy_count"] / topk_frame["topk_size"].replace(0, np.nan)
            )
            topk_frame["group_topk_any_decoy"] = topk_frame["topk_decoy_count"].gt(0)
            group_rows.extend(topk_frame.to_dict(orient="records"))

            fraction_values = topk_frame["group_topk_decoy_fraction"].to_numpy(dtype=float)
            any_values = topk_frame["group_topk_any_decoy"].astype(float).to_numpy(dtype=float)
            prevalence_values = topk_frame["group_decoy_prevalence"].to_numpy(dtype=float)
            fraction_ci_low, fraction_ci_high = _bootstrap_group_mean(
                fraction_values,
                rng=rng,
                n_iter=int(bootstrap_iterations),
            )
            any_ci_low, any_ci_high = _bootstrap_group_mean(
                any_values,
                rng=rng,
                n_iter=int(bootstrap_iterations),
            )
            random_stats = _simulate_random_topk(
                group_sizes=topk_frame["group_size"].to_numpy(dtype=int),
                decoy_counts=topk_frame["group_decoy_count"].to_numpy(dtype=int),
                topk_sizes=topk_frame["topk_size"].to_numpy(dtype=int),
                rng=rng,
                n_iter=int(bootstrap_iterations),
            )
            observed_fraction = float(np.nanmean(fraction_values)) if fraction_values.size else math.nan
            observed_any = float(np.nanmean(any_values)) if any_values.size else math.nan
            random_fraction_draws = np.asarray(random_stats.pop("random_fraction_draws"), dtype=float)
            random_any_draws = np.asarray(random_stats.pop("random_any_draws"), dtype=float)
            lower_tail_fraction_p = float((1 + np.sum(random_fraction_draws <= observed_fraction)) / (len(random_fraction_draws) + 1))
            lower_tail_any_p = float((1 + np.sum(random_any_draws <= observed_any)) / (len(random_any_draws) + 1))
            prevalence_group_mean = float(np.nanmean(prevalence_values)) if prevalence_values.size else math.nan
            metric_rows.append(
                {
                    "feature_set": str(feature_set),
                    "split_strategy": str(split_strategy),
                    "k": int(k),
                    "metric_definition": "group_mean_topk_decoy_burden",
                    "group_count": int(len(topk_frame)),
                    "row_count": int(len(split_frame)),
                    "decoy_count": int(split_frame["decoy_label"].sum()),
                    "weighted_decoy_prevalence": float(split_frame["decoy_label"].mean()) if len(split_frame) else math.nan,
                    "group_decoy_prevalence_mean": prevalence_group_mean,
                    "group_topk_decoy_fraction": observed_fraction,
                    "group_topk_decoy_fraction_ci_low": fraction_ci_low,
                    "group_topk_decoy_fraction_ci_high": fraction_ci_high,
                    "group_topk_any_decoy_rate": observed_any,
                    "group_topk_any_decoy_rate_ci_low": any_ci_low,
                    "group_topk_any_decoy_rate_ci_high": any_ci_high,
                    "specificity_gain_vs_group_prevalence": float(prevalence_group_mean - observed_fraction),
                    "specificity_enrichment_vs_group_prevalence": float(observed_fraction / prevalence_group_mean)
                    if prevalence_group_mean and math.isfinite(prevalence_group_mean)
                    else math.nan,
                    "legacy_slot_topk_decoy_hits": int(topk_frame["topk_decoy_count"].sum()),
                    "legacy_slot_topk_total_slots": int(topk_frame["topk_size"].sum()),
                    "legacy_slot_fpr_at_k": float(
                        topk_frame["topk_decoy_count"].sum() / max(1, topk_frame["topk_size"].sum())
                    ),
                    **random_stats,
                    "specificity_gain_vs_random_fraction": float(
                        random_stats["random_group_topk_decoy_fraction_mean"] - observed_fraction
                    ),
                    "specificity_gain_vs_random_any": float(
                        random_stats["random_group_topk_any_decoy_rate_mean"] - observed_any
                    ),
                    "random_lower_tail_p_fraction": lower_tail_fraction_p,
                    "random_lower_tail_p_any": lower_tail_any_p,
                    "generated_at": iso_now(),
                }
            )

    return pd.DataFrame.from_records(metric_rows), pd.DataFrame.from_records(group_rows)


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage7 = dict(config.get("stage7", {}))
    k_values = [int(value) for value in (args.k or stage7.get("precision_at_k", [5, 10]))]
    seed = int(args.random_seed if args.random_seed is not None else stage7.get("random_seed", 20260413))

    predictions_path = root / str(args.predictions)
    decoys_path = root / str(args.decoys)
    predictions = pd.read_parquet(predictions_path)
    decoys = pd.read_parquet(decoys_path)
    metrics, group_metrics = repaired_decoy_metrics(
        predictions,
        decoys,
        k_values=k_values,
        seed=seed,
        bootstrap_iterations=int(args.bootstrap_iterations),
    )
    output_path = root / str(args.output)
    group_output_path = root / str(args.group_output)
    summary_path = root / str(args.summary_json)
    ensure_dir(output_path.parent)
    metrics.to_csv(output_path, index=False)
    group_metrics.to_csv(group_output_path, index=False)
    summary = {
        "generated_at": iso_now(),
        "predictions_path": str(predictions_path.relative_to(root)),
        "decoys_path": str(decoys_path.relative_to(root)),
        "output": str(output_path.relative_to(root)),
        "group_output": str(group_output_path.relative_to(root)),
        "k_values": k_values,
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "random_seed": seed,
        "prediction_rows": int(len(predictions)),
        "decoy_rows": int(len(decoys)),
        "metric_rows": int(len(metrics)),
        "group_metric_rows": int(len(group_metrics)),
        "primary_metric": "group_topk_decoy_fraction",
        "primary_interpretation": "lower values indicate fewer decoys in each group's top-k set",
    }
    json_dump(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
