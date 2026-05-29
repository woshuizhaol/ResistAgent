#!/usr/bin/env python3
"""Run the Stage 7 benchmark and audits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import ensure_dir, json_dump, load_yaml, project_root
from tools.stage7_utils import (
    build_decoy_frame,
    build_decoy_metrics,
    build_objective_compare_rows,
    build_prior_usage_audit_rows,
    build_split_manifest,
    build_subgroup_metrics,
    evaluate_fold,
    feature_columns,
    fold_test_mask,
    iter_ready_folds,
    load_benchmark_frame,
    write_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def aggregate_rows(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    numeric_cols = [column for column in frame.columns if column not in set(group_cols + ["fold_id"])]
    aggregated = frame.groupby(group_cols, dropna=False)[numeric_cols].mean(numeric_only=True).reset_index()
    aggregated["fold_id"] = "mean"
    return aggregated


def write_stage7_checkpoint(
    *,
    output_root: Path,
    regression_rows: list[dict[str, object]],
    ranking_rows: list[dict[str, object]],
    completed_items: int,
    total_items: int,
    current_fold: dict[str, object] | None,
    current_feature_set: str | None,
    started_at: float,
) -> None:
    progress_payload = {
        "completed_items": int(completed_items),
        "total_items": int(total_items),
        "fraction_complete": 0.0 if total_items <= 0 else float(completed_items / float(total_items)),
        "current_feature_set": None if current_feature_set is None else str(current_feature_set),
        "current_fold_id": None if current_fold is None else str(current_fold.get("fold_id") or ""),
        "current_split_strategy": None if current_fold is None else str(current_fold.get("strategy") or ""),
        "elapsed_seconds": float(max(0.0, time.perf_counter() - started_at)),
    }
    json_dump(output_root / "benchmark_progress.json", progress_payload)
    if regression_rows:
        pd.DataFrame.from_records(regression_rows).to_csv(output_root / "regression_metrics.partial.csv", index=False)
    if ranking_rows:
        pd.DataFrame.from_records(ranking_rows).to_csv(output_root / "ranking_metrics.partial.csv", index=False)


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    cases_config = load_yaml(root / "configs" / "cases.yaml")
    stage7 = dict(config["stage7"])
    output_root = ensure_dir(Path(args.output_dir).resolve() if args.output_dir else root / str(stage7.get("output_dir", "outputs/benchmark")))

    frame, prior_path = load_benchmark_frame(root=root, config=config, cases_config=cases_config)
    split_manifest = build_split_manifest(frame, config, cases_config)
    write_split_manifest(output_root / "split_manifest.json", split_manifest)

    feature_set_names = list(dict(stage7.get("feature_sets") or {}).keys())
    ready_folds = list(iter_ready_folds(split_manifest))
    total_items = int(len(ready_folds) * len(feature_set_names))
    started_at = time.perf_counter()
    print(
        f"[stage7] benchmark_rows={len(frame)} ready_folds={len(ready_folds)} "
        f"feature_sets={len(feature_set_names)} total_items={total_items}",
        flush=True,
    )
    regression_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    decoy_frames: list[pd.DataFrame] = []
    completed_items = 0

    for fold in ready_folds:
        test_mask = fold_test_mask(frame, fold)
        test_frame = frame[test_mask].copy()
        decoy = build_decoy_frame(frame, test_frame, stage7, fold)
        if not decoy.empty:
            decoy_frames.append(decoy)
        for feature_set in feature_set_names:
            print(
                f"[stage7] start split={fold['strategy']} fold={fold['fold_id']} feature_set={feature_set} "
                f"train_rows={(~test_mask).sum()} test_rows={test_mask.sum()}",
                flush=True,
            )
            columns = feature_columns(stage7, feature_set, frame)
            scored, regression, ranking = evaluate_fold(
                frame,
                fold=fold,
                feature_set=feature_set,
                feature_cols=columns,
                stage7=stage7,
            )
            prediction_frames.append(
                scored[
                    [
                        column
                        for column in [
                            "SAMPLE_ID",
                            "UNIPROT_ID",
                            "drug_name",
                            "group_key",
                            "TYPE",
                            "DDG.EXP",
                            "prediction",
                            "rank_score",
                            "resistant_prob",
                            "feature_set",
                            "split_strategy",
                            "fold_id",
                        ]
                        if column in scored.columns
                    ]
                ].copy()
            )
            regression_rows.append(
                {
                    "feature_set": str(feature_set),
                    "split_strategy": str(fold["strategy"]),
                    "fold_id": str(fold["fold_id"]),
                    **regression,
                }
            )
            ranking_rows.append(
                {
                    "feature_set": str(feature_set),
                    "split_strategy": str(fold["strategy"]),
                    "fold_id": str(fold["fold_id"]),
                    **ranking,
                }
            )
            completed_items += 1
            write_stage7_checkpoint(
                output_root=output_root,
                regression_rows=regression_rows,
                ranking_rows=ranking_rows,
                completed_items=completed_items,
                total_items=total_items,
                current_fold=fold,
                current_feature_set=str(feature_set),
                started_at=started_at,
            )
            print(
                f"[stage7] done split={fold['strategy']} fold={fold['fold_id']} feature_set={feature_set} "
                f"completed={completed_items}/{total_items}",
                flush=True,
            )

    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    decoys = pd.concat(decoy_frames, ignore_index=True, sort=False) if decoy_frames else pd.DataFrame()
    regression_frame = pd.DataFrame.from_records(regression_rows)
    ranking_frame = pd.DataFrame.from_records(ranking_rows)
    decoy_metrics = pd.DataFrame.from_records(build_decoy_metrics(predictions, decoys, stage7=stage7))
    subgroup_metrics = pd.DataFrame.from_records(build_subgroup_metrics(predictions, stage7=stage7))
    objective_compare = pd.DataFrame.from_records(build_objective_compare_rows(root, stage7))
    prior_usage_audit = pd.DataFrame.from_records(build_prior_usage_audit_rows(prior_path=prior_path, split_manifest=split_manifest))

    regression_output = pd.concat(
        [regression_frame, aggregate_rows(regression_frame, ["feature_set", "split_strategy"])],
        ignore_index=True,
        sort=False,
    ) if not regression_frame.empty else regression_frame
    ranking_output = pd.concat(
        [ranking_frame, aggregate_rows(ranking_frame, ["feature_set", "split_strategy"])],
        ignore_index=True,
        sort=False,
    ) if not ranking_frame.empty else ranking_frame
    ablation = ranking_output.merge(
        regression_output,
        on=["feature_set", "split_strategy", "fold_id"],
        how="outer",
        suffixes=("_ranking", "_regression"),
    ) if not ranking_output.empty or not regression_output.empty else pd.DataFrame()

    regression_output.to_csv(output_root / "regression_metrics.csv", index=False)
    ranking_output.to_csv(output_root / "ranking_metrics.csv", index=False)
    ablation.to_csv(output_root / "ablation.csv", index=False)
    objective_compare.to_csv(output_root / "objective_compare.csv", index=False)
    prior_usage_audit.to_csv(output_root / "prior_usage_audit.csv", index=False)
    decoy_metrics.to_csv(output_root / "decoy_metrics.csv", index=False)
    subgroup_metrics.to_csv(output_root / "subgroup_metrics_by_mutation_type.csv", index=False)
    if not decoys.empty:
        decoys.to_parquet(output_root / "decoy_mutations.parquet", index=False)
    if not predictions.empty:
        predictions.to_parquet(output_root / "benchmark_predictions.parquet", index=False)

    summary = {
        "benchmark_rows": int(len(frame)),
        "ready_fold_count": int(len(ready_folds)),
        "prediction_rows": int(len(predictions)),
        "decoy_rows": int(len(decoys)),
        "feature_sets": feature_set_names,
    }
    json_dump(output_root / "benchmark_run_summary.json", summary)
    print(
        f"benchmark_rows={summary['benchmark_rows']} ready_folds={summary['ready_fold_count']} "
        f"prediction_rows={summary['prediction_rows']}"
    )


if __name__ == "__main__":
    main()
