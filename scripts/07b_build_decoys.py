#!/usr/bin/env python3
"""Build Stage 7 decoy mutations from test-side folds only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import ensure_dir, load_yaml, project_root
from tools.stage7_utils import (
    build_decoy_frame,
    build_split_manifest,
    fold_test_mask,
    iter_ready_folds,
    load_benchmark_frame,
    write_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    cases_config = load_yaml(root / "configs" / "cases.yaml")
    stage7 = dict(config["stage7"])
    output_root = ensure_dir(root / str(stage7.get("output_dir", "outputs/benchmark")))
    output_path = Path(args.output).resolve() if args.output else output_root / "decoy_mutations.parquet"
    split_manifest_path = Path(args.split_manifest).resolve() if args.split_manifest else output_root / "split_manifest.json"

    frame, _ = load_benchmark_frame(root=root, config=config, cases_config=cases_config)
    if split_manifest_path.exists():
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    else:
        split_manifest = build_split_manifest(frame, config, cases_config)
        write_split_manifest(split_manifest_path, split_manifest)

    decoy_frames = []
    for fold in iter_ready_folds(split_manifest):
        test_mask = fold_test_mask(frame, fold)
        decoy = build_decoy_frame(frame, frame[test_mask].copy(), stage7, fold)
        if not decoy.empty:
            decoy_frames.append(decoy)
    decoy_frame = pd.concat(decoy_frames, ignore_index=True, sort=False) if decoy_frames else pd.DataFrame()
    ensure_dir(output_path.parent)
    decoy_frame.to_parquet(output_path, index=False)
    print(f"decoy_rows={len(decoy_frame)} output={output_path}")


if __name__ == "__main__":
    main()
