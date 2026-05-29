#!/usr/bin/env python3
"""counter-design step.5 supplementary validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import load_yaml, project_root
from tools.stage6_5_utils import finalize_existing_stage6_5_case, run_stage6_5_case, selected_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--finalize-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    cases_config = load_yaml(root / "configs" / "cases.yaml")
    cases = selected_cases(cases_config, args.case_id)
    if not cases:
        raise SystemExit(f"No counter-design step.5 case matched --case-id={args.case_id}")
    for case_entry in cases:
        if args.finalize_existing:
            qc = finalize_existing_stage6_5_case(
                root=root,
                config=config,
                case_entry=dict(case_entry),
            )
        else:
            qc = run_stage6_5_case(
                root=root,
                config=config,
                case_entry=dict(case_entry),
            )
        print(
            f"{case_entry['case_id']}: md_pair_success_count={qc['md_pair_success_count']}, "
            f"mmgbsa_available_count={qc['mmgbsa_available_count']}"
        )


if __name__ == "__main__":
    main()
