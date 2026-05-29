#!/usr/bin/env python3
"""Generate Stage 0 data audit outputs without touching the raw dataset."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import json_dump, load_yaml, project_root, text_dump
from tools.stage0_common import candidate_archive, detect_archive_type, is_ignored_path, relative_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage0 = config["stage0"]
    data_root = root / stage0["data_root"]
    ignore_prefixes = stage0.get("ignore_prefixes", ["._"])
    compute_md5 = bool(stage0.get("compute_md5", False))
    md5_max_bytes = int(stage0.get("md5_max_bytes", 0))

    files = []
    manifest_lines = []
    skipped_ignored = 0

    for path in sorted(data_root.rglob("*")):
        if not path.is_file():
            continue
        if is_ignored_path(path.relative_to(root), ignore_prefixes):
            skipped_ignored += 1
            continue
        readable = True
        try:
            with path.open("rb") as handle:
                handle.read(1)
        except OSError:
            readable = False
        archive_type, _ = detect_archive_type(path) if candidate_archive(path) else (None, None)
        size_bytes = path.stat().st_size
        md5_value = None
        md5_status = "skipped"
        if compute_md5 and size_bytes <= md5_max_bytes and readable:
            md5_value = file_md5(path)
            md5_status = "computed"
        elif readable:
            md5_status = "disabled" if not compute_md5 else "skipped_large_file"
        relative_path = path.relative_to(root)
        manifest_lines.append(str(relative_path))
        files.append(
            {
                "path": str(relative_path),
                "size_bytes": size_bytes,
                "format": archive_type or relative_format(path),
                "readable": readable,
                "md5": md5_value,
                "md5_status": md5_status,
            }
        )

    total_bytes = sum(item["size_bytes"] for item in files)
    archive_count = sum(1 for item in files if item["format"] in {"tar", "tar.gz", "zip", "gzip"})
    compression_only_count = sum(1 for item in files if item["format"] == "gzip")
    payload = {
        "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "data_root": str(data_root.relative_to(root)),
        "summary": {
            "file_count": len(files),
            "total_bytes": total_bytes,
            "archive_count": archive_count,
            "compression_only_count": compression_only_count,
            "skipped_ignored_files": skipped_ignored,
        },
        "files": files,
    }
    text_dump(root / stage0["file_manifest"], "\n".join(manifest_lines) + ("\n" if manifest_lines else ""))
    json_dump(root / stage0["data_audit"], payload)


if __name__ == "__main__":
    main()
