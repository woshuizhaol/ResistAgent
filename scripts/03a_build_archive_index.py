#!/usr/bin/env python3
"""Create Stage 0 alias, archive magic, and archive member index outputs."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import ensure_dir, load_yaml, project_root
from tools.stage0_common import (
    canonical_alias_entries,
    candidate_archive,
    classify_file_role,
    detect_archive_type,
    infer_sample_id,
    is_ignored_path,
    iter_archive_members,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def init_db(path: Path) -> sqlite3.Connection:
    ensure_dir(path.parent)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE archive_members (
            archive_name TEXT NOT NULL,
            archive_type TEXT NOT NULL,
            sample_id TEXT,
            member_path TEXT NOT NULL,
            file_role TEXT NOT NULL,
            size_bytes INTEGER NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX idx_archive_members_sample_id ON archive_members(sample_id)")
    connection.execute("CREATE INDEX idx_archive_members_role ON archive_members(file_role)")
    connection.execute("CREATE INDEX idx_archive_members_archive_name ON archive_members(archive_name)")
    return connection


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage0 = config["stage0"]
    data_root = root / stage0["data_root"]
    ignore_prefixes = stage0.get("ignore_prefixes", ["._"])

    alias_entries = canonical_alias_entries(root)
    alias_map_path = root / stage0["alias_map"]
    ensure_dir(alias_map_path.parent)
    with alias_map_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"aliases": alias_entries}, handle, sort_keys=False)

    archive_rows: list[tuple[str, str, str]] = []
    connection = init_db(root / stage0["archive_index"])

    try:
        for path in sorted(data_root.rglob("*")):
            if not path.is_file():
                continue
            relative_path = path.relative_to(root)
            if is_ignored_path(relative_path, ignore_prefixes):
                continue
            if not candidate_archive(path):
                continue
            archive_type, command = detect_archive_type(path)
            if not archive_type or not command:
                continue
            archive_name = str(relative_path)
            archive_rows.append((archive_name, archive_type, command))
            for member_path, size_bytes in iter_archive_members(path, archive_type):
                connection.execute(
                    """
                    INSERT INTO archive_members (
                        archive_name, archive_type, sample_id, member_path, file_role, size_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        archive_name,
                        archive_type,
                        infer_sample_id(member_path),
                        member_path,
                        classify_file_role(member_path),
                        size_bytes,
                    ),
                )
        connection.commit()
    finally:
        connection.close()

    magic_report_path = root / stage0["archive_magic_report"]
    ensure_dir(magic_report_path.parent)
    with magic_report_path.open("w", encoding="utf-8") as handle:
        handle.write("archive_name\tarchive_type\trecommended_command\n")
        for archive_name, archive_type, command in archive_rows:
            handle.write(f"{archive_name}\t{archive_type}\t{command}\n")


if __name__ == "__main__":
    main()
