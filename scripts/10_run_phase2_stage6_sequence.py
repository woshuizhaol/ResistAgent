#!/usr/bin/env python3
"""Run the Phase 2 counter-design step remote sequence in a deterministic order."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENV_PYTHON = "/data/workplace/miniconda/envs/resistagent/bin/python"
ENV_BIN = "/data/workplace/miniconda/envs/resistagent/bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-stage5", default="1")
    parser.add_argument("--gpu-stage6", default="1")
    parser.add_argument("--hiv-max-rounds", type=int, default=1)
    parser.add_argument("--hiv-proposal-count", type=int, default=10)
    parser.add_argument("--hiv-beam-width", type=int, default=8)
    parser.add_argument("--abl-max-parallel-candidates", type=int, default=8)
    parser.add_argument("--egfr-max-parallel-candidates", type=int, default=8)
    parser.add_argument("--stamp", default=None)
    return parser.parse_args()


def log(message: str) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def snapshot_stage6(case_id: str, stamp: str) -> None:
    stage6_root = ROOT / "outputs" / case_id / "stage6"
    snapshot_root = ROOT / "outputs" / case_id / f"stage6_key_artifacts_phase2seq_{stamp}"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    for filename in [
        "stage6_qc.json",
        "objective_ablation.csv",
        "leaderboard.csv",
        "designed_top200.sdf",
        "robust_sar_rules.md",
        "search_trajectory.csv",
    ]:
        source = stage6_root / filename
        if source.exists():
            shutil.copy2(source, snapshot_root / filename)
    postmortem_root = stage6_root / "postmortem"
    if postmortem_root.exists():
        destination = snapshot_root / "postmortem"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(postmortem_root, destination)
    log(f"snapshot {case_id} -> {snapshot_root}")


def _base_env(env_updates: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{ENV_BIN}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(ROOT)
    if env_updates:
        env.update(env_updates)
    return env


def run_python(arguments: list[str], *, env_updates: dict[str, str] | None = None) -> None:
    log("run " + " ".join(arguments))
    completed = subprocess.run(
        [ENV_PYTHON, *arguments],
        cwd=str(ROOT),
        env=_base_env(env_updates),
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    args = parse_args()
    stamp = str(args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S"))

    snapshot_stage6("abl1_nilotinib", stamp)
    run_python(
        [
            "scripts/09_counter_design.py",
            "--config",
            "configs/base.yaml",
            "--case-id",
            "abl1_nilotinib",
            "--max-parallel-candidates",
            str(int(args.abl_max_parallel_candidates)),
        ],
        env_updates={"CUDA_VISIBLE_DEVICES": str(args.gpu_stage6)},
    )
    run_python(
        [
            "scripts/09b_stage6_postmortem.py",
            "--config",
            "configs/base.yaml",
            "--case-id",
            "abl1_nilotinib",
            "--update-qc",
        ]
    )

    snapshot_stage6("hiv_rt_rilpivirine", stamp)
    run_python(
        ["scripts/07_compute_ifp.py", "--config", "configs/base.yaml", "--case-id", "hiv_rt_rilpivirine"],
        env_updates={"RESISTGPT_STAGE5_GPU_IDS": str(args.gpu_stage5)},
    )
    run_python(
        ["scripts/08b_calibrate_scoring.py", "--config", "configs/base.yaml", "--case-id", "hiv_rt_rilpivirine"],
        env_updates={"RESISTGPT_STAGE5_GPU_IDS": str(args.gpu_stage5)},
    )
    run_python(["scripts/08c_train_stage6_oracle.py", "--config", "configs/base.yaml", "--case-id", "hiv_rt_rilpivirine"])
    run_python(["scripts/08d_oracle_holdout_eval.py", "--config", "configs/base.yaml", "--case-id", "hiv_rt_rilpivirine"])
    run_python(
        [
            "scripts/09_counter_design.py",
            "--config",
            "configs/base.yaml",
            "--case-id",
            "hiv_rt_rilpivirine",
            "--max-rounds",
            str(int(args.hiv_max_rounds)),
            "--proposal-count",
            str(int(args.hiv_proposal_count)),
            "--beam-width",
            str(int(args.hiv_beam_width)),
        ],
        env_updates={"CUDA_VISIBLE_DEVICES": str(args.gpu_stage6)},
    )
    run_python(
        [
            "scripts/09b_stage6_postmortem.py",
            "--config",
            "configs/base.yaml",
            "--case-id",
            "hiv_rt_rilpivirine",
            "--update-qc",
        ]
    )

    snapshot_stage6("egfr_erlotinib", stamp)
    run_python(
        [
            "scripts/09_counter_design.py",
            "--config",
            "configs/base.yaml",
            "--case-id",
            "egfr_erlotinib",
            "--max-parallel-candidates",
            str(int(args.egfr_max_parallel_candidates)),
        ],
        env_updates={"CUDA_VISIBLE_DEVICES": str(args.gpu_stage6)},
    )
    run_python(
        [
            "scripts/09b_stage6_postmortem.py",
            "--config",
            "configs/base.yaml",
            "--case-id",
            "egfr_erlotinib",
            "--update-qc",
        ]
    )


if __name__ == "__main__":
    main()
