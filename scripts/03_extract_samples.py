#!/usr/bin/env python3
"""Stage 3 sample extraction and source normalization."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.public_data_utils import fetch_rcsb_files, request_session
from tools.runtime import ensure_dir, json_dump, load_yaml, project_root
from tools.stage3_utils import (
    ROLE_TO_FILENAME,
    apply_hiv_stage2_gate,
    build_case_selection_table,
    load_case_manifests,
    bulk_extract_archive_hits,
    hardlink_or_copy,
    index_local_structure_roots,
    plan_set_n_stage3_actions,
    query_archive_hits,
    relative_or_absolute,
    to_jsonable,
    validate_standardized_role,
    write_csv_with_columns,
    write_extract_lists,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def panel_string(value: list[str]) -> str:
    return "|".join(sorted({str(item) for item in value}))


def role_source_tier(source_kind: str) -> int | None:
    if source_kind == "samples":
        return 1
    if source_kind == "extracted_full":
        return 2
    if source_kind == "archive_extract":
        return 3
    return None


REPORT_COLUMNS = [
    "case_id",
    "sample_id",
    "pdb_id",
    "chain_id",
    "selection_origin",
    "set_n_case_id",
    "type",
    "evaluation_unit",
    "panel_names",
    "is_excluded",
    "exclusion_reason",
    "is_modeling_queue",
    "eligible_for_missing_gate",
    "core_roles_complete",
    "present_required_roles",
    "present_optional_roles",
    "missing_required_roles",
    "missing_optional_roles",
    "invalid_required_roles",
    "invalid_optional_roles",
    "source_kinds",
    "source_map_path",
    "standardized_dir",
    "hiv_holo_whitelist_ok",
    "hiv_rt_chain_ok",
    "wt_read_ok",
    "mt_read_ok",
    "ligand_read_ok",
    "wt_complex_read_ok",
    "mt_complex_read_ok",
]

EXCLUDED_COLUMNS = ["case_id", "sample_id", "pdb_id", "chain_id", "selection_origin", "set_n_case_id", "type", "reason"]
MISSING_COLUMNS = [
    "case_id",
    "sample_id",
    "pdb_id",
    "chain_id",
    "selection_origin",
    "set_n_case_id",
    "type",
    "evaluation_unit",
    "panel_names",
    "missing_required_roles",
    "missing_optional_roles",
    "invalid_required_roles",
    "invalid_optional_roles",
    "reason",
]
MODELING_COLUMNS = [
    "case_id",
    "sample_id",
    "pdb_id",
    "chain_id",
    "selection_origin",
    "set_n_case_id",
    "type",
    "evaluation_unit",
    "panel_names",
    "core_roles_complete",
    "missing_required_roles",
    "invalid_required_roles",
    "reason",
]

SET_N_REPORT_COLUMNS = [
    "set_n_case_id",
    "pdb_id",
    "target_name",
    "drug_name",
    "matching_master_rows",
    "matching_exact_drug_rows",
    "bridge_case_id",
    "bridge_sample_id",
    "status",
    "status_note",
    "external_asset_status",
    "external_asset_error",
    "external_structure_path",
    "external_structure_cif_path",
    "external_asset_manifest",
]


def materialize_set_n_external_assets(
    root: Path,
    stage1_5: dict[str, object],
    stage3: dict[str, object],
    set_n_report_rows: list[dict[str, object]],
) -> None:
    timeout = int(stage1_5["request_timeout_sec"])
    cache_root = root / str(stage1_5["rcsb_cache_root"])
    external_root = root / str(stage3["set_n_external_root"])
    session = request_session()
    for row in set_n_report_rows:
        row["external_asset_status"] = ""
        row["external_asset_error"] = ""
        row["external_structure_path"] = ""
        row["external_structure_cif_path"] = ""
        row["external_asset_manifest"] = ""
        if str(row["status"]) != "external_test_set":
            continue
        case_root = external_root / str(row["set_n_case_id"])
        raw_root = ensure_dir(case_root / "raw")
        meta_root = ensure_dir(case_root / "meta")
        try:
            rcsb_files = fetch_rcsb_files(session, str(row["pdb_id"]), cache_root, timeout)
            structure_path = raw_root / "structure.pdb"
            structure_cif_path = raw_root / "structure.cif"
            shutil.copyfile(rcsb_files["pdb"], structure_path)
            shutil.copyfile(rcsb_files["cif"], structure_cif_path)
            asset_manifest = {
                "case_id": row["set_n_case_id"],
                "pdb_id": row["pdb_id"],
                "target_name": row["target_name"],
                "drug_name": row["drug_name"],
                "source": "rcsb_pdb",
                "structure_path": relative_or_absolute(root, structure_path),
                "structure_cif_path": relative_or_absolute(root, structure_cif_path),
                "status": "external_test_set_template_ready",
            }
            asset_manifest_path = meta_root / "asset_manifest.json"
            json_dump(asset_manifest_path, asset_manifest)
            row["external_asset_status"] = "template_ready"
            row["external_structure_path"] = relative_or_absolute(root, structure_path)
            row["external_structure_cif_path"] = relative_or_absolute(root, structure_cif_path)
            row["external_asset_manifest"] = relative_or_absolute(root, asset_manifest_path)
        except Exception as exc:  # pragma: no cover - network and remote cache vary
            row["external_asset_status"] = "fetch_failed"
            row["external_asset_error"] = f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    stage1_5 = config["stage1_5"]
    stage2 = config["stage2"]
    stage3 = config["stage3"]

    cases_frozen = load_yaml(root / stage2["cases_frozen_config"])
    manifests_root = root / stage2["manifests_root"]
    extract_lists_root = root / stage3["extract_lists_root"]
    structures_root = root / stage3["structures_root"]
    sample_raw_root = root / stage3["sample_raw_root"]
    search_roots = [root / value for value in stage3["sample_search_roots"]]
    required_roles = [str(value) for value in stage3["required_roles"]]
    optional_roles = [str(value) for value in stage3["optional_roles"]]
    archive_extractable_types = {str(value) for value in stage3["archive_extractable_types"]}
    modeling_queue_types = {str(value) for value in stage3["modeling_queue_types"]}
    missing_rate_threshold = float(stage3["missing_rate_threshold"])

    master_cols = [
        "SAMPLE_ID",
        "PDB_ID",
        "CHAIN_ID",
        "TYPE",
        "MUTATION",
        "evaluation_unit",
        "combination_size",
        "combination_key",
        "component_mutations",
        "component_mutation_keys",
        "UNIPROT_ID",
        "gene_symbol",
        "drug_name",
        "domain_type",
        "target_domain",
    ]
    master_table = pd.read_parquet(root / config["stage1"]["master_table"], columns=master_cols)
    hiv_qc = pd.read_csv(root / stage2["hiv_nnrti_structure_qc"])
    rt_domain_report = pd.read_csv(root / stage2["rt_domain_extraction_report"])
    case_manifests = load_case_manifests(root, cases_frozen)
    selection_df = build_case_selection_table(root, case_manifests, manifests_root, master_table)
    supplemental_rows, set_n_report_rows = plan_set_n_stage3_actions(cases_frozen, case_manifests, selection_df, master_table)
    if supplemental_rows:
        selection_df = build_case_selection_table(
            root,
            case_manifests,
            manifests_root,
            master_table,
            supplemental_rows=supplemental_rows,
        )
    selection_df = apply_hiv_stage2_gate(selection_df, hiv_qc, rt_domain_report)
    materialize_set_n_external_assets(root, stage1_5, stage3, set_n_report_rows)

    extract_list_selection = selection_df[~selection_df["is_excluded"].astype(bool)].copy()
    extract_lists = write_extract_lists(
        extract_list_selection,
        extract_lists_root,
        {str(key): str(value) for key, value in stage3.get("extract_list_names", {}).items()},
    )

    local_index = index_local_structure_roots(search_roots)
    needed_archive_pairs = {
        (str(row.sample_id), role)
        for row in extract_list_selection.itertuples(index=False)
        for role in required_roles + optional_roles
        if local_index.get((str(row.sample_id), role)) is None
    }
    archive_hits = query_archive_hits(
        root / config["stage0"]["archive_index"],
        [sample_id for sample_id, _ in sorted(needed_archive_pairs)],
    )
    archive_hits = {
        key: value
        for key, value in archive_hits.items()
        if key in needed_archive_pairs
    }
    archive_raw_paths = bulk_extract_archive_hits(root, archive_hits, sample_raw_root)

    report_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    modeling_rows: list[dict[str, object]] = []

    for row in selection_df.itertuples(index=False):
        case_required_roles = list(required_roles)
        case_optional_roles = list(optional_roles)
        source_role_rows: dict[str, dict[str, object]] = {}
        source_kinds: set[str] = set()
        missing_required_roles: list[str] = []
        missing_optional_roles: list[str] = []
        invalid_required_roles: list[str] = []
        invalid_optional_roles: list[str] = []
        is_modeling_queue = str(row.type) in modeling_queue_types
        eligible_for_missing_gate = str(row.type) in archive_extractable_types and not bool(row.is_excluded)
        selection_error = getattr(row, "selection_error", None)

        if selection_error is not None and not pd.isna(selection_error) and str(selection_error).strip():
            excluded_rows.append(
                {
                    "case_id": row.case_id,
                    "sample_id": row.sample_id,
                    "pdb_id": row.pdb_id,
                    "chain_id": row.chain_id,
                    "selection_origin": row.selection_origin,
                    "set_n_case_id": row.set_n_case_id,
                    "type": row.type,
                    "reason": str(selection_error).strip(),
                }
            )
            continue

        if bool(row.is_excluded):
            excluded_rows.append(
                {
                    "case_id": row.case_id,
                    "sample_id": row.sample_id,
                    "pdb_id": row.pdb_id,
                    "chain_id": row.chain_id,
                    "selection_origin": row.selection_origin,
                    "set_n_case_id": row.set_n_case_id,
                    "type": row.type,
                    "reason": row.exclusion_reason,
                }
            )
            report_rows.append(
                {
                    "case_id": row.case_id,
                    "sample_id": row.sample_id,
                    "pdb_id": row.pdb_id,
                    "chain_id": row.chain_id,
                    "selection_origin": row.selection_origin,
                    "set_n_case_id": row.set_n_case_id,
                    "type": row.type,
                    "evaluation_unit": row.evaluation_unit,
                    "panel_names": panel_string(row.panel_names),
                    "is_excluded": True,
                    "exclusion_reason": row.exclusion_reason,
                    "is_modeling_queue": is_modeling_queue,
                    "eligible_for_missing_gate": False,
                    "core_roles_complete": False,
                    "present_required_roles": 0,
                    "present_optional_roles": 0,
                    "missing_required_roles": "|".join(case_required_roles),
                    "missing_optional_roles": "|".join(case_optional_roles),
                    "invalid_required_roles": "",
                    "invalid_optional_roles": "",
                    "source_kinds": "",
                    "source_map_path": None,
                    "standardized_dir": None,
                    "hiv_holo_whitelist_ok": bool(row.hiv_holo_whitelist_ok),
                    "hiv_rt_chain_ok": bool(row.hiv_rt_chain_ok),
                    "wt_read_ok": False,
                    "mt_read_ok": False,
                    "ligand_read_ok": False,
                    "wt_complex_read_ok": False,
                    "mt_complex_read_ok": False,
                }
            )
            continue

        sample_dir = ensure_dir(structures_root / str(row.sample_id))

        for role in case_required_roles + case_optional_roles:
            standardized_path = sample_dir / ROLE_TO_FILENAME[role]
            local_hit = local_index.get((str(row.sample_id), role))
            archive_hit = archive_hits.get((str(row.sample_id), role))
            src_path: Path | None = None
            source_kind = ""
            source_path = None
            archive_name = None
            member_path = None
            raw_cache_path = None
            size_bytes = None

            if local_hit is not None:
                src_path = Path(local_hit.path)
                source_kind = str(local_hit.source_kind)
                source_path = str(src_path)
                size_bytes = src_path.stat().st_size
            elif archive_hit is not None and (str(row.sample_id), role) in archive_raw_paths:
                src_path = Path(archive_raw_paths[(str(row.sample_id), role)])
                source_kind = "archive_extract"
                source_path = archive_hit.member_path
                archive_name = archive_hit.archive_name
                member_path = archive_hit.member_path
                raw_cache_path = str(src_path.relative_to(root)) if src_path.is_relative_to(root) else str(src_path)
                size_bytes = archive_hit.size_bytes

            exists = src_path is not None and src_path.exists() and src_path.stat().st_size > 0
            validation = {"read_ok": False, "exists": False, "size_bytes": 0, "error": "missing"}
            if exists and src_path is not None:
                hardlink_or_copy(src_path, standardized_path)
                validation = validate_standardized_role(standardized_path, role)
                source_kinds.add(source_kind)
                if not bool(validation["read_ok"]):
                    if role in case_required_roles:
                        invalid_required_roles.append(role)
                    else:
                        invalid_optional_roles.append(role)
            else:
                if role in case_required_roles:
                    missing_required_roles.append(role)
                else:
                    missing_optional_roles.append(role)

            source_role_rows[role] = {
                "role": role,
                "source_kind": source_kind or "missing",
                "source_tier": role_source_tier(source_kind),
                "source_path": source_path,
                "archive_name": archive_name,
                "member_path": member_path,
                "raw_cache_path": raw_cache_path,
                "standardized_path": relative_or_absolute(root, standardized_path),
                "exists": bool(validation["exists"]),
                "size_bytes": int(validation["size_bytes"]) if validation["size_bytes"] is not None else size_bytes,
                "read_ok": bool(validation["read_ok"]),
                "read_error": validation["error"],
            }

        core_roles_complete = not missing_required_roles and not invalid_required_roles
        if (missing_required_roles or invalid_required_roles) and eligible_for_missing_gate:
            missing_rows.append(
                {
                    "case_id": row.case_id,
                        "sample_id": row.sample_id,
                        "pdb_id": row.pdb_id,
                        "chain_id": row.chain_id,
                        "selection_origin": row.selection_origin,
                        "set_n_case_id": row.set_n_case_id,
                        "type": row.type,
                        "evaluation_unit": row.evaluation_unit,
                        "panel_names": panel_string(row.panel_names),
                        "missing_required_roles": "|".join(sorted(missing_required_roles)),
                        "missing_optional_roles": "|".join(sorted(missing_optional_roles)),
                    "invalid_required_roles": "|".join(sorted(invalid_required_roles)),
                    "invalid_optional_roles": "|".join(sorted(invalid_optional_roles)),
                    "reason": "missing_or_invalid_required_structure_roles",
                }
            )

        if is_modeling_queue:
            modeling_rows.append(
                {
                    "case_id": row.case_id,
                    "sample_id": row.sample_id,
                    "pdb_id": row.pdb_id,
                    "chain_id": row.chain_id,
                    "selection_origin": row.selection_origin,
                    "set_n_case_id": row.set_n_case_id,
                    "type": row.type,
                    "evaluation_unit": row.evaluation_unit,
                    "panel_names": panel_string(row.panel_names),
                    "core_roles_complete": core_roles_complete,
                    "missing_required_roles": "|".join(sorted(missing_required_roles)),
                    "invalid_required_roles": "|".join(sorted(invalid_required_roles)),
                    "reason": "type_requires_modeling_queue",
                }
            )

        source_map = {
            "case_id": row.case_id,
            "sample_id": row.sample_id,
            "pdb_id": row.pdb_id,
            "chain_id": row.chain_id,
            "type": row.type,
            "mutation": row.mutation,
            "evaluation_unit": row.evaluation_unit,
            "component_mutations": list(row.component_mutations),
            "component_mutation_keys": list(row.component_mutation_keys),
            "combination_key": row.combination_key,
            "panel_names": list(row.panel_names),
            "selection_origin": row.selection_origin,
            "set_n_case_id": row.set_n_case_id,
            "hiv_stage2_gate": {
                "hiv_holo_whitelist_ok": bool(row.hiv_holo_whitelist_ok),
                "hiv_rt_chain_ok": bool(row.hiv_rt_chain_ok),
                "rt_domain_extraction_report": stage2["rt_domain_extraction_report"],
                "hiv_structure_qc": stage2["hiv_nnrti_structure_qc"],
            },
            "roles": source_role_rows,
        }
        source_map_path = sample_dir / "source_map.json"
        json_dump(source_map_path, to_jsonable(source_map))

        report_rows.append(
            {
                    "case_id": row.case_id,
                    "sample_id": row.sample_id,
                    "pdb_id": row.pdb_id,
                    "chain_id": row.chain_id,
                    "selection_origin": row.selection_origin,
                    "set_n_case_id": row.set_n_case_id,
                    "type": row.type,
                    "evaluation_unit": row.evaluation_unit,
                    "panel_names": panel_string(row.panel_names),
                    "is_excluded": False,
                    "exclusion_reason": "",
                "is_modeling_queue": is_modeling_queue,
                "eligible_for_missing_gate": eligible_for_missing_gate,
                "core_roles_complete": core_roles_complete,
                "present_required_roles": len(case_required_roles) - len(missing_required_roles),
                "present_optional_roles": len(case_optional_roles) - len(missing_optional_roles),
                "missing_required_roles": "|".join(sorted(missing_required_roles)),
                "missing_optional_roles": "|".join(sorted(missing_optional_roles)),
                "invalid_required_roles": "|".join(sorted(invalid_required_roles)),
                "invalid_optional_roles": "|".join(sorted(invalid_optional_roles)),
                "source_kinds": "|".join(sorted(source_kinds)),
                "source_map_path": relative_or_absolute(root, source_map_path),
                "standardized_dir": relative_or_absolute(root, sample_dir),
                "hiv_holo_whitelist_ok": bool(row.hiv_holo_whitelist_ok),
                "hiv_rt_chain_ok": bool(row.hiv_rt_chain_ok),
                "wt_read_ok": bool(source_role_rows["WT"]["read_ok"]),
                "mt_read_ok": bool(source_role_rows["MT"]["read_ok"]),
                "ligand_read_ok": bool(source_role_rows["ligand"]["read_ok"]),
                "wt_complex_read_ok": bool(source_role_rows["WT_complex"]["read_ok"]),
                "mt_complex_read_ok": bool(source_role_rows["MT_complex"]["read_ok"]),
            }
        )

    report_df = pd.DataFrame.from_records(report_rows, columns=REPORT_COLUMNS)
    if not report_df.empty:
        report_df = report_df.sort_values(["case_id", "sample_id"]).reset_index(drop=True)
    excluded_df = pd.DataFrame.from_records(excluded_rows, columns=EXCLUDED_COLUMNS)
    if not excluded_df.empty:
        excluded_df = excluded_df.sort_values(["case_id", "sample_id"]).reset_index(drop=True)
    missing_df = pd.DataFrame.from_records(missing_rows, columns=MISSING_COLUMNS)
    if not missing_df.empty:
        missing_df = missing_df.sort_values(["case_id", "sample_id"]).reset_index(drop=True)
    modeling_df = pd.DataFrame.from_records(modeling_rows, columns=MODELING_COLUMNS)
    if not modeling_df.empty:
        modeling_df = modeling_df.sort_values(["case_id", "sample_id"]).reset_index(drop=True)

    ensure_dir((root / stage3["extraction_source_report"]).parent)
    write_csv_with_columns(report_df, root / stage3["extraction_source_report"], REPORT_COLUMNS)
    write_csv_with_columns(excluded_df, root / stage3["excluded_samples_report"], EXCLUDED_COLUMNS)
    write_csv_with_columns(missing_df, root / stage3["missing_samples_report"], MISSING_COLUMNS)
    write_csv_with_columns(modeling_df, root / stage3["modeling_queue_report"], MODELING_COLUMNS)
    write_csv_with_columns(
        pd.DataFrame.from_records(set_n_report_rows, columns=SET_N_REPORT_COLUMNS),
        root / stage3["set_n_attempt_report"],
        SET_N_REPORT_COLUMNS,
    )

    included_report_df = report_df[~report_df["is_excluded"].fillna(False)].copy()
    readability_gate_df = included_report_df[~included_report_df["is_modeling_queue"].fillna(False)].copy()
    gate_df = report_df[report_df["eligible_for_missing_gate"].astype(bool)].copy()
    missing_gate_count = int((~gate_df["core_roles_complete"].astype(bool)).sum()) if not gate_df.empty else 0
    missing_rate = float(missing_gate_count / len(gate_df)) if len(gate_df) else 0.0
    required_readable_ok = bool(
        readability_gate_df[["wt_read_ok", "mt_read_ok", "ligand_read_ok"]].fillna(False).all(axis=1).all()
    ) if not readability_gate_df.empty else True
    qc_payload = {
        "extract_lists": {key: str(Path(value).relative_to(root)) if Path(value).is_relative_to(root) else str(value) for key, value in extract_lists.items()},
        "search_root_exists": {str(path): path.exists() for path in search_roots},
        "total_selected_samples": int(len(selection_df)),
        "included_samples": int(len(included_report_df)),
        "excluded_samples": int(len(excluded_df)),
        "modeling_queue_samples": int(len(modeling_df)),
        "archive_extract_samples": int(included_report_df["source_kinds"].fillna("").str.contains("archive_extract").sum()) if not included_report_df.empty else 0,
        "core_complete_samples": int(included_report_df["core_roles_complete"].astype(bool).sum()) if not included_report_df.empty else 0,
        "readability_gate_denominator": int(len(readability_gate_df)),
        "required_readable_samples": int(
            readability_gate_df[["wt_read_ok", "mt_read_ok", "ligand_read_ok"]].fillna(False).all(axis=1).sum()
        ) if not readability_gate_df.empty else 0,
        "missing_gate_denominator": int(len(gate_df)),
        "missing_gate_fail_count": missing_gate_count,
        "missing_rate": missing_rate,
        "missing_rate_threshold": missing_rate_threshold,
        "acceptance": {
            "missing_rate_within_threshold": bool(missing_rate <= missing_rate_threshold),
            "all_included_samples_have_source_map": bool(included_report_df["source_map_path"].notna().all()) if not included_report_df.empty else True,
            "all_included_samples_have_readable_core_files": required_readable_ok,
            "set_n_attempts_recorded": bool(len(set_n_report_rows) == len(cases_frozen.get("set_n", []))),
            "set_n_external_assets_ready": bool(
                all(
                    row["status"] != "external_test_set" or row["external_asset_status"] == "template_ready"
                    for row in set_n_report_rows
                )
            ),
            "all_reports_written": True,
        },
    }
    json_dump(root / stage3["extraction_qc"], qc_payload)

    if missing_rate > missing_rate_threshold:
        raise SystemExit(
            f"Stage 3 missing rate {missing_rate:.4f} exceeded threshold {missing_rate_threshold:.4f}; see {stage3['missing_samples_report']}"
        )


if __name__ == "__main__":
    main()
