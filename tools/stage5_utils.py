#!/usr/bin/env python3
"""mutation-effect step docking, interaction, and calibration helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import shutil
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from Bio.PDB import PDBIO, PDBParser, Select
from rdkit import Chem

from tools.mutation_parser import ParsedComponent, parse_mutation
from tools.runtime import command_exists, ensure_dir, iso_now, json_dump, load_yaml, run_command, sha256_file
from tools.stage35_utils import (
    build_box_from_ligand_coords,
    classify_hiv_pose,
    choose_chain_ligand,
    crystal_ligand_from_template,
    extract_ligand_to_sdf,
    load_structure,
    load_rdkit_molecule,
    merge_pdb_fragments,
    mol_coordinates,
    plip_ifp,
    pose_pdb_from_sdf,
    prepare_ligand_pdbqt,
    prepare_receptor_pdbqt,
    run_pdbfixer,
    run_vina_redocking,
    save_chain_protein,
    save_residue_subset,
    select_best_pose,
    standardize_reference_ligand,
    suppress_rdkit_logs,
    top_pose_per_seed,
    ifp_frequency,
)
from tools.stage4_utils import (
    _hetero_residue_ids_for_chain,
    build_case_context,
    build_hiv_pose_reference,
    materialize_synthetic_combo_sample,
    position_lookup_from_rows,
)
from tools.stage5_physics import (
    fpocket_top_pocket_metrics,
    gnina_score_only,
    multi_score_consensus,
    prepare_amber_complex,
    run_amber_relaxation,
    run_mmgbsa,
)

POLAR_ATOMIC_NUMBERS = {7, 8, 15, 16}
CHARGE_CLASS = {
    "D": "negative",
    "E": "negative",
    "H": "positive",
    "K": "positive",
    "R": "positive",
}
PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
PLIP_INTERACTION_TYPES = (
    "hydrogen_bond",
    "salt_bridge",
    "hydrophobic",
    "pi_stacking",
    "pi_cation",
    "metal_complex",
)
SPECTATOR_METAL_ATOMIC_NUMBERS = {3, 4, 11, 12, 19, 20, 25, 26, 27, 28, 29, 30}
STAGE5_MODEL_VERSION = "stage5_modeled_sample_v1"


class ChainProteinExcludingResiduesSelect(Select):
    def __init__(self, chain_id: str, excluded_residue_ids: set[tuple[str, int, str]]) -> None:
        self.chain_id = str(chain_id)
        self.excluded_residue_ids = excluded_residue_ids

    def accept_chain(self, chain) -> bool:
        return str(chain.id) == self.chain_id

    def accept_residue(self, residue) -> bool:
        residue_key = (
            str(residue.id[0]),
            int(residue.id[1]),
            str(residue.id[2]).strip(),
        )
        return residue.id[0] == " " and residue_key not in self.excluded_residue_ids


def package_version(name: str) -> str | None:
    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:
        return None


def stage5_software_versions() -> dict[str, str | None]:
    versions = {"python": platform.python_version()}
    for binary, args, key in [
        ("python3", ["--version"], "python3"),
        ("conda", ["--version"], "conda"),
        ("snakemake", ["--version"], "snakemake_cli"),
        ("vina", ["--version"], "vina"),
        ("obabel", ["-V"], "obabel"),
        ("plip", ["-h"], "plip_cli"),
        ("fpocket", ["-h"], "fpocket"),
    ]:
        if not command_exists(binary):
            versions[key] = None
            continue
        result = run_command([binary] + args)
        versions[key] = (result.stdout or result.stderr).strip().splitlines()[0] if result.returncode == 0 else None
    versions["openai"] = package_version("openai")
    versions["pandas"] = package_version("pandas")
    versions["matplotlib"] = package_version("matplotlib")
    versions["rdkit"] = package_version("rdkit")
    versions["biopython"] = package_version("biopython")
    return versions


def merge_nested_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_nested_dicts(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def stage5_for_case(stage5: dict[str, Any], case_id: str) -> dict[str, Any]:
    overrides = dict(stage5.get("case_overrides", {}))
    case_override = dict(overrides.get(case_id, {}))
    if not case_override:
        merged = copy.deepcopy(stage5)
        merged.pop("case_overrides", None)
        return merged
    merged = merge_nested_dicts(stage5, case_override)
    merged.pop("case_overrides", None)
    return merged


def _pdb_has_atom_records(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith(("ATOM", "HETATM")):
                    return True
    except OSError:
        return False
    return False


def _sdf_has_atoms(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        molecule = load_rdkit_molecule(path, sanitize=False)
    except Exception:
        return False
    return molecule is not None and int(molecule.GetNumAtoms()) > 0


def _invalid_relaxation_reason(relaxation_payload: dict[str, Any]) -> str | None:
    reasons: list[str] = []
    if not _pdb_has_atom_records(Path(str(relaxation_payload.get("refined_complex_pdb") or ""))):
        reasons.append("refined_complex_missing_atoms")
    if not _pdb_has_atom_records(Path(str(relaxation_payload.get("refined_receptor_pdb") or ""))):
        reasons.append("refined_receptor_missing_atoms")
    if not _sdf_has_atoms(Path(str(relaxation_payload.get("refined_ligand_sdf") or ""))):
        reasons.append("refined_ligand_missing_atoms")
    if not reasons:
        return None
    return "|".join(reasons)


def _validated_relaxation_artifacts(relaxation_payload: dict[str, Any]) -> dict[str, Path]:
    invalid_reason = _invalid_relaxation_reason(relaxation_payload)
    if invalid_reason is not None:
        raise RuntimeError(f"Invalid Stage5 relaxation outputs: {invalid_reason}")
    return {
        "complex_pdb": Path(str(relaxation_payload["refined_complex_pdb"])),
        "receptor_pdb": Path(str(relaxation_payload["refined_receptor_pdb"])),
        "ligand_sdf": Path(str(relaxation_payload["refined_ligand_sdf"])),
    }


def _reset_incomplete_stage5_workdir(work_root: Path, manifest_path: Path) -> None:
    if manifest_path.exists() or not work_root.exists():
        return
    try:
        has_contents = any(work_root.iterdir())
    except OSError:
        has_contents = False
    if has_contents:
        shutil.rmtree(work_root)
        work_root.mkdir(parents=True, exist_ok=True)


def native_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return native_value(value.item())
    return value


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def stable_target_slug(effect_scope: str, target_key: str) -> str:
    digest = hashlib.sha256(f"{effect_scope}:{target_key}".encode("utf-8")).hexdigest()[:12]
    return f"{effect_scope}_{digest}"


def first_protein_chain_id(pdb_path: Path) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    for model in structure:
        for chain in model:
            has_protein = False
            for residue in chain:
                hetflag = str(residue.id[0]).strip()
                residue_name = str(residue.resname).strip().upper()
                if not hetflag and residue_name in PROTEIN_RESIDUES:
                    has_protein = True
                    break
            if has_protein:
                return str(chain.id)
    raise RuntimeError(f"Unable to infer a protein chain from {pdb_path}")


def sample_root_for_target(root: Path, case_id: str, sample_id: str) -> tuple[Path, str]:
    if str(sample_id).startswith("combo_model_"):
        return root / "outputs" / case_id / "stage4" / "synthetic_combo_samples" / sample_id, "synthetic_combo"
    return root / "outputs" / "structures" / sample_id, "observed_sample"


def synthetic_combo_root_for_key(root: Path, case_id: str, combination_key: str) -> Path | None:
    synthetic_root = root / "outputs" / case_id / "stage4" / "synthetic_combo_samples"
    if not synthetic_root.exists():
        return None
    for candidate in sorted(synthetic_root.glob("combo_model_*")):
        manifest = candidate / "model_manifest.json"
        if not manifest.exists():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("combination_key") or "") == str(combination_key):
            return candidate
    return None


def sample_root_ready(sample_root: Path) -> bool:
    required = [
        sample_root / "WT.pdb",
        sample_root / "MT.pdb",
        sample_root / "ligand.sdf",
        sample_root / "WT_complex.pdb",
        sample_root / "MT_complex.pdb",
    ]
    return all(path.exists() for path in required)


def stage5_modeled_sample_root(root: Path, case_id: str, effect_scope: str, target_key: str) -> Path:
    return root / "outputs" / case_id / "stage5" / "modeled_samples" / stable_target_slug(effect_scope, target_key)


def _base_config(root: Path) -> dict[str, Any]:
    return load_yaml(root / "configs" / "base.yaml")


@lru_cache(maxsize=None)
def stage5_case_context(root: Path, case_id: str) -> dict[str, Any]:
    config = _base_config(root)
    stage3_5 = config["stage3_5"]
    kwargs: dict[str, Any] = {}
    if str(case_id) == "hiv_rt_rilpivirine":
        kwargs.update(
            {
                "root": root,
                "hiv_reference_holo_pdb": str(stage3_5["hiv_reference_holo_pdb"]),
                "hiv_reference_holo_chain": str(stage3_5["hiv_reference_holo_chain"]),
            }
        )
    return build_case_context(
        case_id=case_id,
        wt_complex_path=root / "outputs" / case_id / "stage3_5" / "wt_complex.pdb",
        residue_map_path=root / "outputs" / case_id / "stage3_2" / "residue_map.json",
        anchor_path=root / "outputs" / case_id / "stage3_5" / "wt_anchor_residues.txt",
        wt_ifp_path=root / "outputs" / case_id / "stage3_5" / "wt_ifp.json",
        pocket_cutoff_a=float(config["stage4"]["pocket_distance_a"]),
        second_shell_cutoff_a=float(config["stage4"]["second_shell_distance_a"]),
        **kwargs,
    )


def _parsed_component_deletion_positions(component: ParsedComponent) -> list[int]:
    if component.mutation_class != "deletion":
        return []
    if component.start_pos is None:
        return []
    end_pos = component.end_pos if component.end_pos is not None else component.start_pos
    return list(range(int(component.start_pos), int(end_pos) + 1))


def _substitution_specs_for_components(
    components: list[ParsedComponent],
    residue_rows: list[dict[str, Any]],
    numbering_system: str,
    rt_offset: int | None,
) -> list[str]:
    by_position, _, _ = position_lookup_from_rows(residue_rows, numbering_system, rt_offset)
    aa1_to_3 = {
        "A": "ALA",
        "C": "CYS",
        "D": "ASP",
        "E": "GLU",
        "F": "PHE",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "K": "LYS",
        "L": "LEU",
        "M": "MET",
        "N": "ASN",
        "P": "PRO",
        "Q": "GLN",
        "R": "ARG",
        "S": "SER",
        "T": "THR",
        "V": "VAL",
        "W": "TRP",
        "Y": "TYR",
    }
    specs: list[str] = []
    for component in components:
        if component.mutation_class != "single_substitution" or component.start_pos is None or not component.alt_aa:
            continue
        row = by_position.get(int(component.start_pos))
        if row is None:
            raise RuntimeError(f"Unable to map mutation position {component.start_pos} onto WT template residues")
        ref_aa = str(row.get("uniprot_aa") or row.get("pdb_aa") or component.ref_aa or "").upper()
        alt_aa = str(component.alt_aa).upper()
        ref_aa3 = aa1_to_3.get(ref_aa)
        alt_aa3 = aa1_to_3.get(alt_aa)
        if ref_aa3 is None or alt_aa3 is None:
            raise RuntimeError(f"Unsupported mutation code for mutation-effect step modeling: {ref_aa}->{alt_aa}")
        specs.append(f"{ref_aa3}-{int(row['pdb_resnum'])}-{alt_aa3}")
    return specs


def _deletion_residue_ids(
    components: list[ParsedComponent],
    residue_rows: list[dict[str, Any]],
    numbering_system: str,
    rt_offset: int | None,
) -> set[tuple[str, int, str]]:
    by_position, _, _ = position_lookup_from_rows(residue_rows, numbering_system, rt_offset)
    residue_ids: set[tuple[str, int, str]] = set()
    for component in components:
        for position in _parsed_component_deletion_positions(component):
            row = by_position.get(int(position))
            if row is None:
                raise RuntimeError(f"Unable to map deletion position {position} onto WT template residues")
            residue_ids.add((" ", int(row["pdb_resnum"]), str(row.get("insertion_code") or "").strip()))
    return residue_ids


def save_chain_without_residues(
    source_pdb: Path,
    chain_id: str,
    excluded_residue_ids: set[tuple[str, int, str]],
    output_pdb: Path,
) -> None:
    structure = load_structure(source_pdb)
    io = PDBIO()
    io.set_structure(structure)
    ensure_dir(output_pdb.parent)
    io.save(str(output_pdb), select=ChainProteinExcludingResiduesSelect(chain_id, excluded_residue_ids))


def _ligand_from_case_template(
    *,
    wt_complex_path: Path,
    ligand_input_path: Path,
    chain_id: str,
    sample_root: Path,
) -> Path:
    ligand_output = sample_root / "ligand.sdf"
    ligand = choose_chain_ligand(wt_complex_path, chain_id)
    if ligand is None:
        shutil.copyfile(ligand_input_path, ligand_output)
        return ligand_output
    try:
        crystal_ligand_from_template(
            wt_complex_path,
            chain_id,
            ligand,
            ligand_input_path,
            ligand_output,
            sample_root,
        )
    except Exception:
        extract_ligand_to_sdf(
            wt_complex_path,
            chain_id,
            ligand,
            ligand_output,
            sample_root,
        )
    return ligand_output


def materialize_stage5_modeled_sample(
    *,
    root: Path,
    case_id: str,
    effect_scope: str,
    target_key: str,
    sample_root: Path,
) -> tuple[str, dict[str, Any]]:
    context = stage5_case_context(root, case_id)
    wt_complex_path = root / "outputs" / case_id / "stage3_5" / "wt_complex.pdb"
    ligand_input_path = root / "outputs" / case_id / "stage1_5" / "raw" / "ligand.sdf"
    chain_id = str(context["chain_id"])
    residue_rows = list(context["residue_rows"])
    numbering_system = str(context["numbering_system"])
    rt_offset = None if context.get("rt_offset") is None else int(context["rt_offset"])
    components = target_components(target_key)
    if not components:
        raise RuntimeError(f"No parsed mutation component is available for mutation-effect step modeling: {target_key}")
    mutation_classes = {str(component.mutation_class) for component in components}

    if mutation_classes == {"single_substitution"}:
        materialize_synthetic_combo_sample(
            combo_key=target_key,
            components=[
                SimpleNamespace(
                    raw=str(component.raw),
                    position=int(component.start_pos) if component.start_pos is not None else None,
                    ref_aa=component.ref_aa,
                    alt_aa=component.alt_aa,
                    mutation_class=component.mutation_class,
                )
                for component in components
            ],
            sample_root=sample_root,
            wt_complex_path=wt_complex_path,
            ligand_input_path=ligand_input_path,
            chain_id=chain_id,
            residue_rows=residue_rows,
            numbering_system=numbering_system,
            rt_offset=rt_offset,
        )
        model_kind = "single_substitution" if len(components) == 1 else "multi_substitution"
    elif mutation_classes.issubset({"single_substitution", "deletion"}) and "deletion" in mutation_classes:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile

        ensure_dir(sample_root)
        wt_pdb = sample_root / "WT.pdb"
        mt_pdb = sample_root / "MT.pdb"
        wt_complex = sample_root / "WT_complex.pdb"
        mt_complex = sample_root / "MT_complex.pdb"
        save_chain_protein(wt_complex_path, chain_id, wt_pdb)
        shutil.copyfile(wt_complex_path, wt_complex)
        _ligand_from_case_template(
            wt_complex_path=wt_complex_path,
            ligand_input_path=ligand_input_path,
            chain_id=chain_id,
            sample_root=sample_root,
        )
        deletion_ids = _deletion_residue_ids(components, residue_rows, numbering_system, rt_offset)
        mt_seed = sample_root / "MT_seed.pdb"
        save_chain_without_residues(wt_complex_path, chain_id, deletion_ids, mt_seed)
        substitution_specs = _substitution_specs_for_components(components, residue_rows, numbering_system, rt_offset)
        if substitution_specs:
            fixer = PDBFixer(filename=str(mt_seed))
            fixer.applyMutations(substitution_specs, str(chain_id))
            fixer.findMissingResidues()
            fixer.missingResidues = {}
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            with mt_pdb.open("w", encoding="utf-8") as handle:
                PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
        else:
            shutil.copyfile(mt_seed, mt_pdb)
        hetero_ids = _hetero_residue_ids_for_chain(wt_complex_path, chain_id)
        hetero_pdb = sample_root / "hetero_fragment.pdb"
        save_residue_subset(wt_complex_path, chain_id, hetero_ids, hetero_pdb)
        merge_pdb_fragments([mt_pdb, hetero_pdb], mt_complex)
        model_kind = "mixed_deletion_substitution" if substitution_specs else "deletion"
    else:
        raise RuntimeError(
            f"mutation-effect step modeling does not support component classes={sorted(mutation_classes)} for {target_key}"
        )

    manifest = {
        "stage5_model_version": STAGE5_MODEL_VERSION,
        "case_id": str(case_id),
        "effect_scope": str(effect_scope),
        "target_key": str(target_key),
        "model_kind": model_kind,
        "mutation_classes": sorted(mutation_classes),
        "component_mutations": [str(component.raw) for component in components],
        "template_complex": str(wt_complex_path),
        "template_ligand": str(ligand_input_path),
    }
    json_dump(sample_root / "model_manifest.json", manifest)
    return model_kind, manifest


def resolve_stage5_target_sample(
    *,
    root: Path,
    case_id: str,
    effect_scope: str,
    target_key: str,
    sample_id: str,
    sample_root: Path,
    sample_source: str,
    allow_modeling: bool,
    force_modeling: bool = False,
) -> tuple[Path, str, bool, str]:
    if sample_root_ready(sample_root) and not force_modeling:
        return sample_root, sample_source, False, ""
    modeled_root = stage5_modeled_sample_root(root, case_id, effect_scope, target_key)
    if sample_root_ready(modeled_root):
        manifest_path = modeled_root / "model_manifest.json"
        model_kind = ""
        if manifest_path.exists():
            try:
                model_kind = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("model_kind") or "")
            except Exception:
                model_kind = ""
        return modeled_root, "stage5_modeled", True, model_kind
    if not allow_modeling:
        return sample_root, sample_source, False, ""
    materialize_stage5_modeled_sample(
        root=root,
        case_id=case_id,
        effect_scope=effect_scope,
        target_key=target_key,
        sample_root=modeled_root,
    )
    manifest = json.loads((modeled_root / "model_manifest.json").read_text(encoding="utf-8"))
    return modeled_root, "stage5_modeled", True, str(manifest.get("model_kind") or "")


def stage5_reference_ligand_input(sample_root: Path, work_root: Path) -> Path:
    ligand_input = sample_root / "ligand.sdf"
    wt_complex = sample_root / "WT_complex.pdb"
    wt_pdb = sample_root / "WT.pdb"
    if not (ligand_input.exists() and wt_complex.exists() and wt_pdb.exists()):
        return ligand_input
    try:
        chain_id = first_protein_chain_id(wt_pdb)
        ligand = choose_chain_ligand(wt_complex, chain_id)
        if ligand is None:
            return ligand_input
        output_sdf = work_root / "ligand_from_wt_complex.sdf"
        try:
            crystal_ligand_from_template(
                wt_complex,
                chain_id,
                ligand,
                ligand_input,
                output_sdf,
                work_root,
            )
        except Exception:
            extract_ligand_to_sdf(
                wt_complex,
                chain_id,
                ligand,
                output_sdf,
                work_root,
            )
        if not output_sdf.exists():
            return ligand_input
        template = load_rdkit_molecule(ligand_input, sanitize=False)
        candidate = load_rdkit_molecule(output_sdf, sanitize=False)
        template_atoms = non_spectator_heavy_atom_count(template)
        candidate_atoms = non_spectator_heavy_atom_count(candidate)
        # Guard against extracting a wrong hetero ligand from multi-ligand WT complexes.
        if template_atoms > 0 and candidate_atoms < max(4, math.ceil(template_atoms * 0.6)):
            return ligand_input
        return output_sdf
    except Exception:
        return ligand_input


def effect_target_key(row: pd.Series | dict[str, Any]) -> str:
    if isinstance(row, pd.Series):
        series = row
    else:
        series = pd.Series(row)
    return str(series.get("mutation_key") or series.get("combination_key") or "")


def effect_stage4_rank(row: pd.Series | dict[str, Any]) -> int | None:
    if isinstance(row, pd.Series):
        series = row
    else:
        series = pd.Series(row)
    for field in ["mutation_rank", "combo_rank"]:
        value = series.get(field)
        if value is None or pd.isna(value):
            continue
        return int(value)
    return None


def _build_site_panel(
    *,
    root: Path,
    case_id: str,
    site_rank: pd.DataFrame,
    stage5: dict[str, Any],
    mutation_status: pd.DataFrame,
) -> list[dict[str, Any]]:
    if site_rank.empty:
        return []
    status_lookup = mutation_status.drop_duplicates("sample_id").set_index("sample_id").to_dict(orient="index")
    sorted_frame = site_rank.sort_values("mutation_rank").reset_index(drop=True)
    rows: list[dict[str, Any]] = []

    def record_from_row(row: pd.Series, selection_bucket: str) -> dict[str, Any]:
        sample_id = str(row.get("representative_sample_id") or "")
        sample_root, sample_source = sample_root_for_target(root, case_id, sample_id) if sample_id else (Path(), "missing")
        status_row = status_lookup.get(sample_id, {})
        allow_modeling = bool(stage5.get("model_missing_structures", True))
        used_stage5_modeled_sample = False
        stage5_model_kind = ""
        if allow_modeling and (not sample_id or not sample_root_ready(sample_root) or not bool(status_row.get("eligible_for_stage5", False))):
            try:
                sample_root, sample_source, used_stage5_modeled_sample, stage5_model_kind = resolve_stage5_target_sample(
                    root=root,
                    case_id=case_id,
                    effect_scope="site",
                    target_key=str(row["mutation_key"]),
                    sample_id=sample_id,
                    sample_root=sample_root,
                    sample_source=sample_source,
                    allow_modeling=allow_modeling,
                    force_modeling=bool(sample_id and bool(status_row) and not bool(status_row.get("eligible_for_stage5", False))),
                )
            except Exception:
                used_stage5_modeled_sample = False
                stage5_model_kind = ""
        eligible = True if sample_source == "stage5_modeled" else bool(status_row.get("eligible_for_stage5", False)) if status_row else False
        ready = sample_root_ready(sample_root) and eligible
        if not sample_id and sample_source != "stage5_modeled":
            skip_reason = "missing_representative_sample"
        elif not sample_root.exists():
            skip_reason = "missing_sample_root"
        elif not sample_root_ready(sample_root):
            skip_reason = "sample_root_incomplete"
        elif not eligible:
            skip_reason = "sample_not_eligible_for_stage5"
        else:
            skip_reason = ""
        return {
            "effect_scope": "site",
            "target_key": str(row["mutation_key"]),
            "stage4_rank": effect_stage4_rank(row),
            "risk_score": native_value(row.get("risk_calibrated", row.get("risk_score"))),
            "impact_evidence_tier": str(row.get("impact_evidence_tier") or ""),
            "proxy_status": str(row.get("proxy_status") or ""),
            "representative_sample_id": sample_id,
            "sample_source": sample_source,
            "sample_root": str(sample_root) if sample_root else "",
            "stage5_ready": bool(ready),
            "stage5_skip_reason": skip_reason,
            "stage5_selection_bucket": selection_bucket,
            "used_synthetic_combo_model": False,
            "used_stage5_modeled_sample": bool(used_stage5_modeled_sample),
            "stage5_model_kind": stage5_model_kind,
            "component_positions": str(row.get("component_positions") or "[]"),
            "component_count": int(row.get("component_count") or 1),
            "stage4_delta_dock_proxy": native_value(row.get("delta_dock_proxy")),
            "stage4_delta_ifp_proxy": native_value(row.get("delta_ifp_proxy")),
            "stage4_anchor_loss_fraction": native_value(row.get("anchor_loss_fraction")),
            "stage4_local_rmsd_a": native_value(
                row.get("local_backbone_rmsd_a") if "local_backbone_rmsd_a" in row else row.get("local_rmsd_a")
            ),
            "stage4_ddg_fold_surrogate": native_value(row.get("ddg_fold_surrogate")),
        }

    top_n = int(stage5["site_top_n"])
    for _, row in sorted_frame.head(top_n).iterrows():
        rows.append(record_from_row(row, "primary_topk"))
    ready_count = sum(1 for row in rows if bool(row["stage5_ready"]))
    if ready_count < top_n:
        for _, row in sorted_frame.iloc[top_n:].iterrows():
            record = record_from_row(row, "ready_backfill")
            if not bool(record["stage5_ready"]):
                continue
            rows.append(record)
            ready_count += 1
            if ready_count >= top_n:
                break
    return rows


def _build_combo_panel(
    *,
    root: Path,
    case_id: str,
    combo_rank: pd.DataFrame,
    stage5: dict[str, Any],
    mutation_status: pd.DataFrame,
) -> list[dict[str, Any]]:
    if combo_rank.empty:
        return []
    status_lookup = mutation_status.drop_duplicates("sample_id").set_index("sample_id").to_dict(orient="index")
    sorted_frame = combo_rank.sort_values("combo_rank").reset_index(drop=True)
    rows: list[dict[str, Any]] = []

    def record_from_row(row: pd.Series, selection_bucket: str) -> dict[str, Any]:
        sample_id = str(row.get("representative_sample_id") or "")
        sample_root, sample_source = sample_root_for_target(root, case_id, sample_id) if sample_id else (Path(), "missing")
        observed_sample = status_lookup.get(sample_id, {})
        synthetic_model = bool(row.get("used_synthetic_combo_model", False))
        used_stage5_modeled_sample = False
        stage5_model_kind = ""
        if synthetic_model and not sample_root.exists():
            synthetic_root = synthetic_combo_root_for_key(root, case_id, str(row["combination_key"]))
            if synthetic_root is not None:
                sample_root = synthetic_root
                sample_source = "synthetic_combo"
        allow_modeling = bool(stage5.get("model_missing_structures", True))
        if allow_modeling and (
            not sample_id
            or not sample_root_ready(sample_root)
            or not bool(observed_sample.get("eligible_for_stage5", False))
            or not bool(row.get("all_components_mapped", True))
        ):
            try:
                sample_root, sample_source, used_stage5_modeled_sample, stage5_model_kind = resolve_stage5_target_sample(
                    root=root,
                    case_id=case_id,
                    effect_scope="combo",
                    target_key=str(row["combination_key"]),
                    sample_id=sample_id,
                    sample_root=sample_root,
                    sample_source=sample_source,
                    allow_modeling=allow_modeling,
                    force_modeling=bool(
                        sample_root_ready(sample_root)
                        and (
                            not bool(observed_sample.get("eligible_for_stage5", False))
                            or not bool(row.get("all_components_mapped", True))
                        )
                    ),
                )
            except Exception:
                used_stage5_modeled_sample = False
                stage5_model_kind = ""
        eligible = True if str(sample_source) in {"synthetic_combo", "stage5_modeled"} else bool(observed_sample.get("eligible_for_stage5", False))
        all_components_mapped = True if str(sample_source) in {"synthetic_combo", "stage5_modeled"} else bool(row.get("all_components_mapped", eligible))
        ready = sample_root_ready(sample_root) and eligible and all_components_mapped
        if not sample_id and sample_source not in {"synthetic_combo", "stage5_modeled"}:
            skip_reason = "missing_representative_sample"
        elif not sample_root.exists():
            skip_reason = "missing_sample_root"
        elif not sample_root_ready(sample_root):
            skip_reason = "sample_root_incomplete"
        elif not eligible:
            skip_reason = "sample_not_eligible_for_stage5"
        elif not all_components_mapped:
            skip_reason = "components_not_fully_mapped"
        else:
            skip_reason = ""
        return {
            "effect_scope": "combo",
            "target_key": str(row["combination_key"]),
            "stage4_rank": effect_stage4_rank(row),
            "risk_score": native_value(row.get("risk_calibrated", row.get("risk_score"))),
            "impact_evidence_tier": str(row.get("impact_evidence_tier") or ""),
            "proxy_status": str(row.get("proxy_status") or ""),
            "representative_sample_id": sample_id,
            "sample_source": sample_source,
            "sample_root": str(sample_root) if sample_root else "",
            "stage5_ready": bool(ready),
            "stage5_skip_reason": skip_reason,
            "stage5_selection_bucket": selection_bucket,
            "used_synthetic_combo_model": bool(row.get("used_synthetic_combo_model", sample_source == "synthetic_combo")),
            "used_stage5_modeled_sample": bool(used_stage5_modeled_sample),
            "stage5_model_kind": stage5_model_kind,
            "component_positions": str(row.get("component_positions") or "[]"),
            "component_count": int(row.get("combination_size") or row.get("component_count") or 0),
            "stage4_delta_dock_proxy": native_value(row.get("delta_dock_proxy")),
            "stage4_delta_ifp_proxy": native_value(row.get("delta_ifp_proxy")),
            "stage4_anchor_loss_fraction": native_value(row.get("anchor_loss_fraction")),
            "stage4_local_rmsd_a": native_value(
                row.get("local_backbone_rmsd_a") if "local_backbone_rmsd_a" in row else row.get("local_rmsd_a")
            ),
            "stage4_ddg_fold_surrogate": native_value(row.get("ddg_fold_surrogate")),
        }

    top_n = int(stage5["combo_top_n"])
    for _, row in sorted_frame.head(top_n).iterrows():
        rows.append(record_from_row(row, "primary_topk"))
    ready_count = sum(1 for row in rows if bool(row["stage5_ready"]))
    if ready_count < top_n:
        for _, row in sorted_frame.iloc[top_n:].iterrows():
            record = record_from_row(row, "ready_backfill")
            if not bool(record["stage5_ready"]):
                continue
            rows.append(record)
            ready_count += 1
            if ready_count >= top_n:
                break
    return rows


def build_stage5_target_panel(
    *,
    root: Path,
    case_id: str,
    site_rank: pd.DataFrame,
    combo_rank: pd.DataFrame,
    mutation_status: pd.DataFrame,
    stage5: dict[str, Any],
) -> pd.DataFrame:
    rows = _build_site_panel(
        root=root,
        case_id=case_id,
        site_rank=site_rank,
        stage5=stage5,
        mutation_status=mutation_status,
    )
    rows.extend(
        _build_combo_panel(
            root=root,
            case_id=case_id,
            combo_rank=combo_rank,
            stage5=stage5,
            mutation_status=mutation_status,
        )
    )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "effect_scope",
                "target_key",
                "stage4_rank",
                "risk_score",
                "impact_evidence_tier",
                "proxy_status",
                "representative_sample_id",
                "sample_source",
                "sample_root",
                "stage5_ready",
                "stage5_skip_reason",
                "stage5_selection_bucket",
                "used_synthetic_combo_model",
                "used_stage5_modeled_sample",
                "stage5_model_kind",
                "component_positions",
                "component_count",
                "stage4_delta_dock_proxy",
                "stage4_delta_ifp_proxy",
                "stage4_anchor_loss_fraction",
                "stage4_local_rmsd_a",
                "stage4_ddg_fold_surrogate",
            ]
        )
    return frame.sort_values(["effect_scope", "stage4_rank", "target_key"]).reset_index(drop=True)


def build_hiv_reference(
    *,
    root: Path,
    case_entry: dict[str, Any],
    stage2: dict[str, Any],
    stage3_5: dict[str, Any],
) -> dict[str, Any] | None:
    if str(case_entry.get("target_domain") or "").lower() != "rt":
        return None
    return build_hiv_pose_reference(
        root=root,
        case_entry=case_entry,
        pocket_positions=list(stage2["hiv_pocket_positions"]),
        reference_holo_pdb=str(stage3_5["hiv_reference_holo_pdb"]),
        reference_holo_chain=str(stage3_5["hiv_reference_holo_chain"]),
        contact_cutoff_a=float(stage3_5["hiv_pose_contact_cutoff_a"]),
    )


def _expand_box(docking_box: dict[str, Any], delta_a: float, attempt_index: int) -> dict[str, Any]:
    expanded = dict(docking_box)
    if delta_a > 0.0:
        expanded["size_x"] = float(expanded["size_x"]) + delta_a
        expanded["size_y"] = float(expanded["size_y"]) + delta_a
        expanded["size_z"] = float(expanded["size_z"]) + delta_a
    expanded["source"] = f"{docking_box['source']}_retry_{attempt_index}"
    expanded["retry_expansion_delta_a"] = float(delta_a)
    expanded["retry_attempt_index"] = int(attempt_index)
    return expanded


def write_table(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def write_json(path: Path, payload: Any) -> None:
    json_dump(path, payload)


def _copy_to_stable_path(source: Path, destination: Path) -> Path:
    ensure_dir(destination.parent)
    destination.write_bytes(source.read_bytes())
    return destination


def run_stage5_pair_docking(
    *,
    root: Path,
    case_id: str,
    target_row: dict[str, Any],
    stage5: dict[str, Any],
    hiv_reference: dict[str, Any] | None,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_id = str(target_row.get("representative_sample_id") or "")
    target_key = str(target_row["target_key"])
    target_slug = stable_target_slug(str(target_row["effect_scope"]), target_key)
    run_root = ensure_dir(output_root / target_slug)
    pose_rows: list[dict[str, Any]] = []
    started_at = iso_now()

    base_record = {
        **target_row,
        "case_id": case_id,
        "target_slug": target_slug,
        "stage5_run_root": relative_path(run_root, root),
        "stage5_status": "skipped",
        "stage5_error": None,
        "stage5_attempt_count": 0,
        "stage5_attempt_history_json": "[]",
        "wt_best_affinity_kcal_mol": None,
        "mt_best_affinity_kcal_mol": None,
        "delta_dock_kcal_mol": None,
        "wt_pose_count": 0,
        "mt_pose_count": 0,
        "wt_pose_sdf": "",
        "mt_pose_sdf": "",
        "wt_receptor_pdb": "",
        "mt_receptor_pdb": "",
        "wt_complex_docked_pdb": "",
        "mt_complex_docked_pdb": "",
        "docking_box_source": "",
        "started_at": started_at,
        "finished_at": None,
        "run_summary_json": relative_path(run_root / "docking_run.json", root),
    }
    if not bool(target_row.get("stage5_ready", False)):
        base_record["stage5_error"] = str(target_row.get("stage5_skip_reason") or "target_not_ready")
        base_record["finished_at"] = iso_now()
        write_json(run_root / "docking_run.json", base_record)
        return base_record, pose_rows

    try:
        sample_root = Path(str(target_row["sample_root"]))
        chain_id = first_protein_chain_id(sample_root / "WT.pdb")
        ligand_reference_input = stage5_reference_ligand_input(sample_root, run_root)
        ligand_standardized = run_root / "ligand_standardized.sdf"
        standardize_reference_ligand(ligand_reference_input, ligand_standardized)
        ligand = load_rdkit_molecule(ligand_standardized, sanitize=True)
        ligand_coords = mol_coordinates(ligand)
        base_box = build_box_from_ligand_coords(
            ligand_coords,
            default_box_size_a=float(stage5["default_box_size_a"]),
            ligand_padding_a=float(stage5["ligand_box_padding_a"]),
            source="sample_ligand_centroid",
        )

        max_attempts = int(stage5["max_redocking_attempts"])
        expansion_step_a = float(stage5["box_expansion_step_a"])
        attempt_history: list[dict[str, Any]] = []
        final_record = dict(base_record)
        last_error = "unknown_error"

        for attempt_index in range(1, max_attempts + 1):
            attempt_root = ensure_dir(run_root / f"attempt_{attempt_index}")
            current_box = _expand_box(base_box, expansion_step_a * float(attempt_index - 1), attempt_index)
            try:
                wt_receptor_input = sample_root / "WT.pdb"
                mt_receptor_input = sample_root / "MT.pdb"
                if bool(stage5.get("use_pdbfixer", False)):
                    wt_fixed = attempt_root / "WT_fixed.pdb"
                    mt_fixed = attempt_root / "MT_fixed.pdb"
                    run_pdbfixer(sample_root / "WT.pdb", wt_fixed, float(stage5["protein_prep_ph"]))
                    run_pdbfixer(sample_root / "MT.pdb", mt_fixed, float(stage5["protein_prep_ph"]))
                    wt_receptor_input = wt_fixed
                    mt_receptor_input = mt_fixed

                wt_receptor_pdbqt = attempt_root / "WT.pdbqt"
                mt_receptor_pdbqt = attempt_root / "MT.pdbqt"
                ligand_pdbqt = attempt_root / "ligand.pdbqt"
                prepare_receptor_pdbqt(wt_receptor_input, wt_receptor_pdbqt, float(stage5["protein_prep_ph"]))
                prepare_receptor_pdbqt(mt_receptor_input, mt_receptor_pdbqt, float(stage5["protein_prep_ph"]))
                prepare_ligand_pdbqt(ligand_standardized, ligand_pdbqt, float(stage5["protein_prep_ph"]))

                wt_rows = run_vina_redocking(
                    receptor_pdbqt=wt_receptor_pdbqt,
                    ligand_pdbqt=ligand_pdbqt,
                    docking_box=current_box,
                    output_root=attempt_root / "wt_docking",
                    seeds=list(stage5["vina_seeds"]),
                    exhaustiveness=int(stage5["vina_exhaustiveness"]),
                    num_modes=int(stage5["vina_num_modes"]),
                    energy_range=int(stage5["vina_energy_range"]),
                    cpu_threads=int(stage5["vina_cpu_threads"]),
                )
                mt_rows = run_vina_redocking(
                    receptor_pdbqt=mt_receptor_pdbqt,
                    ligand_pdbqt=ligand_pdbqt,
                    docking_box=current_box,
                    output_root=attempt_root / "mt_docking",
                    seeds=list(stage5["vina_seeds"]),
                    exhaustiveness=int(stage5["vina_exhaustiveness"]),
                    num_modes=int(stage5["vina_num_modes"]),
                    energy_range=int(stage5["vina_energy_range"]),
                    cpu_threads=int(stage5["vina_cpu_threads"]),
                )
                if hiv_reference is not None:
                    for kind, rows in [("wt", wt_rows), ("mt", mt_rows)]:
                        for row in rows:
                            row.update(
                                classify_hiv_pose(
                                    Path(str(row["pose_sdf"])),
                                    hiv_reference["nnrti_residue_coords"],
                                    hiv_reference["active_site_residue_coords"],
                                    float(stage5["hiv_pose_contact_cutoff_a"]),
                                )
                            )

                best_wt = select_best_pose(wt_rows, hiv_mode=hiv_reference is not None)
                best_mt = select_best_pose(mt_rows, hiv_mode=hiv_reference is not None)
                if best_wt is None or best_mt is None:
                    raise RuntimeError("No valid WT/MT pose selected after mutation-effect step gating")
                if hiv_reference is not None:
                    required_label = str(stage5["hiv_required_pose_label"])
                    global_best_wt = select_best_pose(wt_rows, hiv_mode=False)
                    global_best_mt = select_best_pose(mt_rows, hiv_mode=False)
                    if global_best_wt is not None and str(global_best_wt.get("pose_label") or "other") != required_label:
                        raise RuntimeError(f"WT global-best pose left HIV NNRTI pocket: {global_best_wt.get('pose_label')}")
                    if global_best_mt is not None and str(global_best_mt.get("pose_label") or "other") != required_label:
                        raise RuntimeError(f"MT global-best pose left HIV NNRTI pocket: {global_best_mt.get('pose_label')}")

                for kind, rows, selected in [("wt", wt_rows, best_wt), ("mt", mt_rows, best_mt)]:
                    selected_seed = int(selected["seed"])
                    selected_mode_rank = int(selected["mode_rank"])
                    for row in rows:
                        pose_rows.append(
                            {
                                "case_id": case_id,
                                "target_key": target_key,
                                "target_slug": target_slug,
                                "effect_scope": str(target_row["effect_scope"]),
                                "sample_id": sample_id,
                                "pose_set": kind,
                                "attempt_index": int(attempt_index),
                                "seed": int(row["seed"]),
                                "mode_rank": int(row["mode_rank"]),
                                "affinity_kcal_mol": float(row["affinity_kcal_mol"]),
                                "selected_for_stage5": bool(
                                    int(row["seed"]) == selected_seed and int(row["mode_rank"]) == selected_mode_rank
                                ),
                                "pose_label": str(row.get("pose_label") or ""),
                                "nnrti_min_distance_a": native_value(row.get("nnrti_min_distance_a")),
                                "nnrti_coverage_count": native_value(row.get("nnrti_coverage_count")),
                                "nnrti_coverage_fraction": native_value(row.get("nnrti_coverage_fraction")),
                                "active_site_min_distance_a": native_value(row.get("active_site_min_distance_a")),
                                "active_site_coverage_count": native_value(row.get("active_site_coverage_count")),
                                "active_site_coverage_fraction": native_value(row.get("active_site_coverage_fraction")),
                                "pose_sdf": relative_path(Path(str(row["pose_sdf"])), root),
                            }
                        )

                wt_pose_sdf = _copy_to_stable_path(Path(str(best_wt["pose_sdf"])), run_root / "wt_best_pose.sdf")
                mt_pose_sdf = _copy_to_stable_path(Path(str(best_mt["pose_sdf"])), run_root / "mt_best_pose.sdf")
                wt_receptor_pdb = _copy_to_stable_path(wt_receptor_input, run_root / "wt_receptor_stage5.pdb")
                mt_receptor_pdb = _copy_to_stable_path(mt_receptor_input, run_root / "mt_receptor_stage5.pdb")
                wt_pose_pdb = run_root / "wt_best_pose.pdb"
                mt_pose_pdb = run_root / "mt_best_pose.pdb"
                pose_pdb_from_sdf(wt_pose_sdf, wt_pose_pdb)
                pose_pdb_from_sdf(mt_pose_sdf, mt_pose_pdb)
                wt_complex_docked = run_root / "wt_complex_docked.pdb"
                mt_complex_docked = run_root / "mt_complex_docked.pdb"
                merge_pdb_fragments([wt_receptor_pdb, wt_pose_pdb], wt_complex_docked)
                merge_pdb_fragments([mt_receptor_pdb, mt_pose_pdb], mt_complex_docked)

                final_record.update(
                    {
                        "stage5_status": "ok",
                        "stage5_error": None,
                        "stage5_attempt_count": int(attempt_index),
                        "stage5_attempt_history_json": json.dumps(attempt_history, ensure_ascii=True),
                        "wt_best_affinity_kcal_mol": float(best_wt["affinity_kcal_mol"]),
                        "mt_best_affinity_kcal_mol": float(best_mt["affinity_kcal_mol"]),
                        "delta_dock_kcal_mol": float(best_mt["affinity_kcal_mol"] - best_wt["affinity_kcal_mol"]),
                        "wt_pose_count": int(len(wt_rows)),
                        "mt_pose_count": int(len(mt_rows)),
                        "wt_pose_sdf": relative_path(wt_pose_sdf, root),
                        "mt_pose_sdf": relative_path(mt_pose_sdf, root),
                        "wt_receptor_pdb": relative_path(wt_receptor_pdb, root),
                        "mt_receptor_pdb": relative_path(mt_receptor_pdb, root),
                        "wt_complex_docked_pdb": relative_path(wt_complex_docked, root),
                        "mt_complex_docked_pdb": relative_path(mt_complex_docked, root),
                        "docking_box_source": str(current_box["source"]),
                        "finished_at": iso_now(),
                    }
                )
                write_json(run_root / "docking_run.json", final_record)
                return final_record, pose_rows
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                attempt_history.append(
                    {
                        "attempt_index": int(attempt_index),
                        "docking_box_source": str(current_box["source"]),
                        "retry_expansion_delta_a": float(expansion_step_a * float(attempt_index - 1)),
                        "error": last_error,
                    }
                )
    except Exception as exc:
        failure_record = dict(base_record)
        failure_record.update(
            {
                "stage5_status": "failed",
                "stage5_error": f"{type(exc).__name__}: {exc}",
                "stage5_attempt_count": 0,
                "stage5_attempt_history_json": "[]",
                "finished_at": iso_now(),
            }
        )
        write_json(run_root / "docking_run.json", failure_record)
        return failure_record, pose_rows

    final_record.update(
        {
            "stage5_status": "failed",
            "stage5_error": last_error,
            "stage5_attempt_count": int(max_attempts),
            "stage5_attempt_history_json": json.dumps(attempt_history, ensure_ascii=True),
            "finished_at": iso_now(),
        }
    )
    write_json(run_root / "docking_run.json", final_record)
    return final_record, pose_rows


def load_anchor_labels(anchor_path: Path) -> set[str]:
    if not anchor_path.exists():
        return set()
    return {
        line.strip()
        for line in anchor_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def ifp_interaction_type_counts(ifp_payload: dict[str, Any]) -> dict[str, int]:
    counts = ifp_payload.get("interaction_type_counts", {})
    return {key: int(counts.get(key, 0)) for key in PLIP_INTERACTION_TYPES}


def jaccard_loss(lhs: set[str], rhs: set[str]) -> float:
    union = lhs | rhs
    if not union:
        return 0.0
    return float(1.0 - (len(lhs & rhs) / float(len(union))))


def anchor_loss_fraction(anchor_labels: set[str], mt_labels: set[str]) -> float:
    if not anchor_labels:
        return 0.0
    return float(1.0 - (len(anchor_labels & mt_labels) / float(len(anchor_labels))))


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _min_distance_set(point: tuple[float, float, float], coords: list[tuple[float, float, float]]) -> float | None:
    if not coords:
        return None
    return min(_distance(point, coord) for coord in coords)


def _protein_heavy_atom_coords(pdb_path: Path) -> list[tuple[float, float, float]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    coords: list[tuple[float, float, float]] = []
    for atom in structure.get_atoms():
        residue = atom.get_parent()
        hetflag = str(residue.id[0]).strip()
        residue_name = str(residue.resname).strip().upper()
        element = str(getattr(atom, "element", "")).strip().upper()
        if hetflag or residue_name not in PROTEIN_RESIDUES:
            continue
        if element in {"", "H", "D"}:
            continue
        coord = atom.get_coord()
        coords.append((float(coord[0]), float(coord[1]), float(coord[2])))
    return coords


def non_spectator_heavy_atom_count(molecule: Chem.Mol) -> int:
    return int(
        sum(
            1
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() > 1 and int(atom.GetAtomicNum()) not in SPECTATOR_METAL_ATOMIC_NUMBERS
        )
    )


def local_pocket_metrics(
    *,
    receptor_pdb: Path,
    ligand_sdf: Path,
    pocket_contact_distance_a: float,
    polar_contact_distance_a: float,
    solvent_exposure_neighbor_threshold: int,
) -> dict[str, Any]:
    try:
        with suppress_rdkit_logs():
            ligand = load_rdkit_molecule(ligand_sdf, sanitize=True)
    except ValueError:
        ligand = load_rdkit_molecule(ligand_sdf, sanitize=False)
    ligand_coords = mol_coordinates(ligand)
    protein_coords = _protein_heavy_atom_coords(receptor_pdb)
    near_coords = [
        coord
        for coord in protein_coords
        if _min_distance_set(coord, ligand_coords) is not None and _min_distance_set(coord, ligand_coords) <= float(pocket_contact_distance_a)
    ]
    if near_coords:
        xs = [coord[0] for coord in near_coords]
        ys = [coord[1] for coord in near_coords]
        zs = [coord[2] for coord in near_coords]
        volume = float(max(xs) - min(xs)) * float(max(ys) - min(ys)) * float(max(zs) - min(zs))
    else:
        volume = 0.0
    tight_contacts = sum(
        1
        for coord in protein_coords
        if _min_distance_set(coord, ligand_coords) is not None and _min_distance_set(coord, ligand_coords) <= 4.0
    )
    contact_density = float(tight_contacts / max(1, ligand.GetNumHeavyAtoms()))
    polar_atom_indices = [
        atom.GetIdx()
        for atom in ligand.GetAtoms()
        if int(atom.GetAtomicNum()) in POLAR_ATOMIC_NUMBERS and atom.GetAtomicNum() > 1
    ]
    exposed_count = 0
    for atom_idx in polar_atom_indices:
        coord = ligand_coords[atom_idx]
        neighbors = sum(
            1
            for protein_coord in protein_coords
            if _distance(coord, protein_coord) <= float(polar_contact_distance_a)
        )
        if neighbors < int(solvent_exposure_neighbor_threshold):
            exposed_count += 1
    polar_exposed_fraction = float(exposed_count / max(1, len(polar_atom_indices))) if polar_atom_indices else 0.0
    return {
        "pocket_volume_proxy_a3": float(volume),
        "contact_density": float(contact_density),
        "polar_atom_count": int(len(polar_atom_indices)),
        "polar_exposed_fraction": float(polar_exposed_fraction),
        "pocket_contact_atom_count": int(len(near_coords)),
    }


def should_run_local_sampling(docking_row: dict[str, Any], stage5: dict[str, Any]) -> bool:
    if not bool(stage5.get("local_sampling_enabled", True)):
        return False
    if int(docking_row.get("component_count") or 0) >= int(stage5.get("local_sampling_component_count_trigger", 2)):
        return True
    local_rmsd = docking_row.get("stage4_local_rmsd_a")
    if local_rmsd is not None and float(local_rmsd) >= float(stage5.get("local_sampling_rmsd_trigger_a", 0.8)):
        return True
    model_kind = str(docking_row.get("stage5_model_kind") or "")
    return "deletion" in model_kind or "indel" in model_kind


def ensure_stage5_relaxation(
    *,
    root: Path,
    docking_row: dict[str, Any],
    stage5: dict[str, Any],
    gpu_id: int | str | None = None,
) -> dict[str, dict[str, Any]]:
    run_root = root / str(docking_row["stage5_run_root"])
    local_sampling = should_run_local_sampling(docking_row, stage5)
    payload: dict[str, dict[str, Any]] = {}
    for pose_set in ["wt", "mt"]:
        receptor_pdb = root / str(docking_row[f"{pose_set}_receptor_pdb"])
        pose_sdf = root / str(docking_row[f"{pose_set}_pose_sdf"])
        physics_root = ensure_dir(run_root / f"{pose_set}_physics")
        ligand_template_sdf = run_root / "ligand_standardized.sdf"
        amber_prep_root = ensure_dir(physics_root / "amber_prep")
        prep_manifest_path = amber_prep_root / "amber_prep_manifest.json"
        _reset_incomplete_stage5_workdir(amber_prep_root, prep_manifest_path)
        if prep_manifest_path.exists():
            amber_prep = json.loads(prep_manifest_path.read_text(encoding="utf-8"))
        else:
            amber_prep = prepare_amber_complex(
                receptor_pdb=receptor_pdb,
                ligand_pose_sdf=pose_sdf,
                work_root=amber_prep_root,
                ligand_template_sdf=ligand_template_sdf if ligand_template_sdf.exists() else None,
                ligand_parameter_cache_root=run_root.parent.parent / "ligand_param_cache",
            )
        relaxation_root = ensure_dir(physics_root / "relaxation")
        relaxation_manifest_path = relaxation_root / "relaxation_manifest.json"
        _reset_incomplete_stage5_workdir(relaxation_root, relaxation_manifest_path)
        if relaxation_manifest_path.exists():
            relaxation_payload = json.loads(relaxation_manifest_path.read_text(encoding="utf-8"))
            if _invalid_relaxation_reason(relaxation_payload) is not None:
                relaxation_payload = run_amber_relaxation(
                    amber_prep=amber_prep,
                    work_root=relaxation_root,
                    stage5=stage5,
                    run_local_sampling=local_sampling,
                    cuda_visible_devices=gpu_id,
                )
        else:
            relaxation_payload = run_amber_relaxation(
                amber_prep=amber_prep,
                work_root=relaxation_root,
                stage5=stage5,
                run_local_sampling=local_sampling,
                cuda_visible_devices=gpu_id,
            )
        payload[pose_set] = {
            "amber_prep": amber_prep,
            "relaxation": relaxation_payload,
        }
    return payload


def pose_ensemble_rows_for_target(
    pose_ensemble: pd.DataFrame,
    *,
    target_slug: str,
    pose_set: str,
    attempt_index: int,
) -> list[dict[str, Any]]:
    if pose_ensemble.empty:
        return []
    subset = pose_ensemble[
        pose_ensemble["target_slug"].astype(str).eq(str(target_slug))
        & pose_ensemble["pose_set"].astype(str).eq(str(pose_set))
        & pose_ensemble["attempt_index"].fillna(-1).astype(int).eq(int(attempt_index))
    ].copy()
    if subset.empty:
        return []
    return subset.sort_values(["seed", "mode_rank"]).to_dict(orient="records")


def occupancy_frequency_payload(
    *,
    root: Path,
    run_root: Path,
    receptor_pdb: Path,
    pose_rows: list[dict[str, Any]],
    pose_set: str,
) -> dict[str, Any]:
    top_rows = top_pose_per_seed(pose_rows)
    if not top_rows:
        return {"top_seed_count": 0, "occupancy_map": {}}
    top_pose_ifps: list[dict[str, Any]] = []
    occupancy_root = ensure_dir(run_root / "occupancy" / pose_set)
    for row in top_rows:
        pose_sdf = root / str(row["pose_sdf"])
        pose_pdb = occupancy_root / f"seed_{int(row['seed'])}_pose.pdb"
        pose_pdb_from_sdf(pose_sdf, pose_pdb)
        complex_pdb = occupancy_root / f"seed_{int(row['seed'])}_complex.pdb"
        merge_pdb_fragments([receptor_pdb, pose_pdb], complex_pdb)
        top_pose_ifps.append(plip_ifp(complex_pdb))
    return {
        "top_seed_count": int(len(top_rows)),
        "occupancy_map": ifp_frequency(top_pose_ifps),
    }


def occupancy_shift_metrics(
    *,
    wt_payload: dict[str, Any],
    mt_payload: dict[str, Any],
    anchor_labels: set[str],
) -> dict[str, Any]:
    wt_map = {str(key): float(value) for key, value in wt_payload.get("occupancy_map", {}).items()}
    mt_map = {str(key): float(value) for key, value in mt_payload.get("occupancy_map", {}).items()}
    union = sorted(set(wt_map) | set(mt_map))
    if not union:
        return {
            "wt_top_seed_count": int(wt_payload.get("top_seed_count", 0)),
            "mt_top_seed_count": int(mt_payload.get("top_seed_count", 0)),
            "ifp_occupancy_shift_mean_abs": 0.0,
            "ifp_occupancy_anchor_loss": 0.0,
            "wt_ifp_occupancy_json": "{}",
            "mt_ifp_occupancy_json": "{}",
        }
    diffs = [abs(float(mt_map.get(label, 0.0)) - float(wt_map.get(label, 0.0))) for label in union]
    anchor_diffs = [
        max(0.0, float(wt_map.get(label, 0.0)) - float(mt_map.get(label, 0.0)))
        for label in sorted(anchor_labels)
        if label in wt_map or label in mt_map
    ]
    return {
        "wt_top_seed_count": int(wt_payload.get("top_seed_count", 0)),
        "mt_top_seed_count": int(mt_payload.get("top_seed_count", 0)),
        "ifp_occupancy_shift_mean_abs": float(sum(diffs) / len(diffs)),
        "ifp_occupancy_anchor_loss": float(sum(anchor_diffs) / len(anchor_diffs)) if anchor_diffs else 0.0,
        "wt_ifp_occupancy_json": json.dumps(wt_map, ensure_ascii=True, sort_keys=True),
        "mt_ifp_occupancy_json": json.dumps(mt_map, ensure_ascii=True, sort_keys=True),
    }


def ensure_stage5_scoring_payload(
    *,
    root: Path,
    docking_row: dict[str, Any],
    stage5: dict[str, Any],
    gpu_id: int | str | None = None,
) -> dict[str, Any]:
    def gnina_payload_usable(payload: dict[str, Any]) -> bool:
        if not bool(payload.get("available", False)):
            return False
        affinity = native_value(payload.get("affinity_kcal_mol"))
        return affinity is not None and math.isfinite(float(affinity))

    def mmgbsa_unavailable_payload(exc: Exception) -> dict[str, Any]:
        return {
            "mmgbsa_at": iso_now(),
            "available": False,
            "delta_total_kcal_mol": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    run_root = root / str(docking_row["stage5_run_root"])
    require_gnina = bool(stage5.get("require_gnina_for_stage5_scoring", False))
    gnina_binary_available = bool(command_exists("gnina"))
    relaxation = ensure_stage5_relaxation(
        root=root,
        docking_row=docking_row,
        stage5=stage5,
        gpu_id=gpu_id,
    )
    payload: dict[str, Any] = {}
    for pose_set in ["wt", "mt"]:
        physics_root = ensure_dir(run_root / f"{pose_set}_physics")
        amber_prep = relaxation[pose_set]["amber_prep"]
        relaxation_payload = relaxation[pose_set]["relaxation"]
        mmgbsa_manifest_path = physics_root / "mmgbsa" / "mmgbsa_manifest.json"
        if mmgbsa_manifest_path.exists():
            mmgbsa_payload = json.loads(mmgbsa_manifest_path.read_text(encoding="utf-8"))
        else:
            try:
                mmgbsa_payload = run_mmgbsa(
                    amber_prep=amber_prep,
                    relaxation_payload=relaxation_payload,
                    work_root=physics_root / "mmgbsa",
                    stage5=stage5,
                )
            except Exception as exc:
                mmgbsa_payload = mmgbsa_unavailable_payload(exc)
                json_dump(mmgbsa_manifest_path, mmgbsa_payload)
        gnina_root = ensure_dir(physics_root / "gnina")
        gnina_payload_path = gnina_root / "gnina_manifest.json"
        gnina_inputs = _validated_relaxation_artifacts(relaxation_payload)
        if gnina_payload_path.exists():
            gnina_payload = json.loads(gnina_payload_path.read_text(encoding="utf-8"))
            if not gnina_payload_usable(gnina_payload) and gnina_binary_available:
                gnina_payload = gnina_score_only(
                    receptor_pdb=gnina_inputs["receptor_pdb"],
                    ligand_sdf=gnina_inputs["ligand_sdf"],
                    work_root=gnina_root,
                    cuda_visible_devices=gpu_id,
                )
                json_dump(gnina_payload_path, gnina_payload)
        else:
            gnina_payload = gnina_score_only(
                receptor_pdb=gnina_inputs["receptor_pdb"],
                ligand_sdf=gnina_inputs["ligand_sdf"],
                work_root=gnina_root,
                cuda_visible_devices=gpu_id,
            )
            json_dump(gnina_payload_path, gnina_payload)
        if require_gnina and not gnina_payload_usable(gnina_payload):
            raise RuntimeError(f"gnina scoring unavailable for strict Stage5 scoring at {gnina_payload_path}")
        payload[pose_set] = {
            "mmgbsa": mmgbsa_payload,
            "gnina": gnina_payload,
        }

    wt_mmgbsa = native_value(payload["wt"]["mmgbsa"].get("delta_total_kcal_mol"))
    mt_mmgbsa = native_value(payload["mt"]["mmgbsa"].get("delta_total_kcal_mol"))
    wt_gnina = native_value(payload["wt"]["gnina"].get("affinity_kcal_mol"))
    mt_gnina = native_value(payload["mt"]["gnina"].get("affinity_kcal_mol"))
    delta_mmgbsa = None if wt_mmgbsa is None or mt_mmgbsa is None else float(mt_mmgbsa) - float(wt_mmgbsa)
    delta_gnina = None if wt_gnina is None or mt_gnina is None else float(mt_gnina) - float(wt_gnina)
    consensus = multi_score_consensus(
        {
            "vina": native_value(docking_row.get("delta_dock_kcal_mol")),
            "mmgbsa": delta_mmgbsa,
            "gnina": delta_gnina,
        },
        neutral_threshold=float(stage5.get("multi_score_neutral_threshold_kcal_mol", 0.25)),
        consensus_threshold=float(stage5.get("multi_score_consensus_threshold", 0.70)),
    )
    if delta_mmgbsa is None:
        consensus["high_uncertainty"] = True
    return {
        "wt_mmgbsa_binding_kcal_mol": wt_mmgbsa,
        "mt_mmgbsa_binding_kcal_mol": mt_mmgbsa,
        "delta_mmgbsa_binding_kcal_mol": delta_mmgbsa,
        "wt_gnina_affinity_kcal_mol": wt_gnina,
        "mt_gnina_affinity_kcal_mol": mt_gnina,
        "delta_gnina_affinity_kcal_mol": delta_gnina,
        **consensus,
    }


def target_components(target_key: str) -> list[ParsedComponent]:
    token = str(target_key)
    if ":" in token:
        token = token.split(":", 1)[1]
    parsed = parse_mutation(token)
    return list(parsed.get("parsed_components", []))


def has_charge_shift(target_key: str) -> bool:
    for component in target_components(target_key):
        ref = CHARGE_CLASS.get(str(component.ref_aa or "").upper(), "neutral")
        alt = CHARGE_CLASS.get(str(component.alt_aa or "").upper(), "neutral")
        if ref != alt:
            return True
    return False


def deterministic_mechanism_labels(
    *,
    target_key: str,
    delta_dock_kcal_mol: float,
    ifp_jaccard_loss_value: float,
    anchor_loss_fraction_value: float,
    hbond_delta_count: int,
    salt_bridge_delta_count: int,
    local_rmsd_a: float | None,
    pocket_volume_change_fraction: float,
    solvent_proxy_shift: float,
    stage5: dict[str, Any],
) -> list[str]:
    labels: list[str] = []
    if anchor_loss_fraction_value >= float(stage5["anchor_loss_threshold"]) or hbond_delta_count < 0 or salt_bridge_delta_count < 0:
        labels.append("anchor_loss")
    if (
        delta_dock_kcal_mol >= float(stage5["steric_clash_delta_dock_threshold"])
        and pocket_volume_change_fraction <= -float(stage5["pocket_volume_change_threshold"])
    ):
        labels.append("steric_clash")
    if has_charge_shift(target_key) and (
        salt_bridge_delta_count < 0
        or hbond_delta_count < 0
        or solvent_proxy_shift >= float(stage5["solvent_proxy_shift_threshold"])
    ):
        labels.append("electrostatic_shift")
    if (
        (local_rmsd_a is not None and float(local_rmsd_a) >= float(stage5["local_rmsd_threshold_a"]))
        or abs(pocket_volume_change_fraction) >= float(stage5["pocket_volume_change_threshold"])
        or abs(solvent_proxy_shift) >= float(stage5["solvent_proxy_shift_threshold"])
    ):
        labels.append("pocket_rearrangement")
    if not labels:
        if ifp_jaccard_loss_value >= 0.25:
            labels.append("anchor_loss")
        elif delta_dock_kcal_mol >= 0.75:
            labels.append("steric_clash")
        else:
            labels.append("pocket_rearrangement")
    return sorted(set(labels))


def mechanism_signature(labels: list[str]) -> str:
    if not labels:
        return "none"
    return "|".join(sorted(set(str(label) for label in labels)))


def compute_ifp_effect_row(
    *,
    root: Path,
    case_id: str,
    docking_row: dict[str, Any],
    anchor_labels: set[str],
    stage5: dict[str, Any],
    pose_ensemble: pd.DataFrame,
    gpu_id: int | str | None = None,
) -> dict[str, Any]:
    base = {
        "case_id": case_id,
        "effect_scope": str(docking_row["effect_scope"]),
        "target_key": str(docking_row["target_key"]),
        "target_slug": str(docking_row["target_slug"]),
        "representative_sample_id": str(docking_row.get("representative_sample_id") or ""),
        "stage5_status": str(docking_row.get("stage5_status") or ""),
        "impact_evidence_tier": str(docking_row.get("impact_evidence_tier") or ""),
        "sample_source": str(docking_row.get("sample_source") or ""),
        "used_synthetic_combo_model": bool(docking_row.get("used_synthetic_combo_model", False)),
        "used_stage5_modeled_sample": bool(docking_row.get("used_stage5_modeled_sample", False)),
        "stage5_model_kind": str(docking_row.get("stage5_model_kind") or ""),
        "stage4_rank": native_value(docking_row.get("stage4_rank")),
        "risk_score": native_value(docking_row.get("risk_score")),
        "delta_dock_kcal_mol": native_value(docking_row.get("delta_dock_kcal_mol")),
        "stage4_local_rmsd_a": native_value(docking_row.get("stage4_local_rmsd_a")),
    }
    if str(docking_row.get("stage5_status") or "") != "ok":
        return {
            **base,
            "ifp_status": "skipped",
            "ifp_error": str(docking_row.get("stage5_error") or "docking_not_available"),
            "ifp_jaccard_loss": None,
            "anchor_loss_fraction": None,
            "refinement_status": "skipped",
            "refinement_error": str(docking_row.get("stage5_error") or "docking_not_available"),
            "mechanism_labels_json": "[]",
        }

    wt_complex = root / str(docking_row["wt_complex_docked_pdb"])
    mt_complex = root / str(docking_row["mt_complex_docked_pdb"])
    wt_receptor = root / str(docking_row["wt_receptor_pdb"])
    mt_receptor = root / str(docking_row["mt_receptor_pdb"])
    wt_pose_sdf = root / str(docking_row["wt_pose_sdf"])
    mt_pose_sdf = root / str(docking_row["mt_pose_sdf"])
    refinement_status = "ok"
    refinement_error = None
    local_sampling_applied = should_run_local_sampling(docking_row, stage5)
    relaxation = ensure_stage5_relaxation(
        root=root,
        docking_row=docking_row,
        stage5=stage5,
        gpu_id=gpu_id,
    )
    wt_artifacts = _validated_relaxation_artifacts(dict(relaxation["wt"]["relaxation"]))
    mt_artifacts = _validated_relaxation_artifacts(dict(relaxation["mt"]["relaxation"]))
    wt_complex = wt_artifacts["complex_pdb"]
    mt_complex = mt_artifacts["complex_pdb"]
    wt_receptor = wt_artifacts["receptor_pdb"]
    mt_receptor = mt_artifacts["receptor_pdb"]
    wt_pose_sdf = wt_artifacts["ligand_sdf"]
    mt_pose_sdf = mt_artifacts["ligand_sdf"]

    wt_ifp = plip_ifp(wt_complex)
    mt_ifp = plip_ifp(mt_complex)
    wt_labels = set(str(value) for value in wt_ifp.get("residue_set", []))
    mt_labels = set(str(value) for value in mt_ifp.get("residue_set", []))
    baseline_anchor_labels = set(anchor_labels) & wt_labels if anchor_labels else wt_labels
    wt_counts = ifp_interaction_type_counts(wt_ifp)
    mt_counts = ifp_interaction_type_counts(mt_ifp)
    wt_pocket = local_pocket_metrics(
        receptor_pdb=wt_receptor,
        ligand_sdf=wt_pose_sdf,
        pocket_contact_distance_a=float(stage5["pocket_contact_distance_a"]),
        polar_contact_distance_a=float(stage5["polar_contact_distance_a"]),
        solvent_exposure_neighbor_threshold=int(stage5["solvent_exposure_neighbor_threshold"]),
    )
    mt_pocket = local_pocket_metrics(
        receptor_pdb=mt_receptor,
        ligand_sdf=mt_pose_sdf,
        pocket_contact_distance_a=float(stage5["pocket_contact_distance_a"]),
        polar_contact_distance_a=float(stage5["polar_contact_distance_a"]),
        solvent_exposure_neighbor_threshold=int(stage5["solvent_exposure_neighbor_threshold"]),
    )
    volume_change_fraction = 0.0
    if float(wt_pocket["pocket_volume_proxy_a3"]) > 0.0:
        volume_change_fraction = float(
            (float(mt_pocket["pocket_volume_proxy_a3"]) - float(wt_pocket["pocket_volume_proxy_a3"]))
            / float(wt_pocket["pocket_volume_proxy_a3"])
        )
    wt_fpocket_volume_a3 = None
    mt_fpocket_volume_a3 = None
    fpocket_error = None
    try:
        wt_fpocket = fpocket_top_pocket_metrics(wt_receptor)
        mt_fpocket = fpocket_top_pocket_metrics(mt_receptor)
        wt_fpocket_volume_a3 = native_value(wt_fpocket.get("volume_a3"))
        mt_fpocket_volume_a3 = native_value(mt_fpocket.get("volume_a3"))
        if wt_fpocket_volume_a3 not in {None, 0.0}:
            volume_change_fraction = float((float(mt_fpocket_volume_a3 or 0.0) - float(wt_fpocket_volume_a3)) / float(wt_fpocket_volume_a3))
    except Exception as exc:
        fpocket_error = f"{type(exc).__name__}: {exc}"
    solvent_proxy_shift = float(mt_pocket["polar_exposed_fraction"] - wt_pocket["polar_exposed_fraction"])
    ifp_jaccard_loss_value = jaccard_loss(wt_labels, mt_labels)
    anchor_loss_value = anchor_loss_fraction(baseline_anchor_labels, mt_labels)
    target_slug = str(docking_row["target_slug"])
    selected_attempt = int(docking_row.get("stage5_attempt_count") or 1)
    wt_occupancy = occupancy_frequency_payload(
        root=root,
        run_root=root / str(docking_row["stage5_run_root"]),
        receptor_pdb=root / str(docking_row["wt_receptor_pdb"]),
        pose_rows=pose_ensemble_rows_for_target(
            pose_ensemble,
            target_slug=target_slug,
            pose_set="wt",
            attempt_index=selected_attempt,
        ),
        pose_set="wt",
    )
    mt_occupancy = occupancy_frequency_payload(
        root=root,
        run_root=root / str(docking_row["stage5_run_root"]),
        receptor_pdb=root / str(docking_row["mt_receptor_pdb"]),
        pose_rows=pose_ensemble_rows_for_target(
            pose_ensemble,
            target_slug=target_slug,
            pose_set="mt",
            attempt_index=selected_attempt,
        ),
        pose_set="mt",
    )
    occupancy_payload = occupancy_shift_metrics(
        wt_payload=wt_occupancy,
        mt_payload=mt_occupancy,
        anchor_labels=baseline_anchor_labels,
    )
    labels = deterministic_mechanism_labels(
        target_key=str(docking_row["target_key"]),
        delta_dock_kcal_mol=float(docking_row.get("delta_dock_kcal_mol") or 0.0),
        ifp_jaccard_loss_value=float(ifp_jaccard_loss_value),
        anchor_loss_fraction_value=float(anchor_loss_value),
        hbond_delta_count=int(mt_counts["hydrogen_bond"] - wt_counts["hydrogen_bond"]),
        salt_bridge_delta_count=int(mt_counts["salt_bridge"] - wt_counts["salt_bridge"]),
        local_rmsd_a=None if docking_row.get("stage4_local_rmsd_a") is None else float(docking_row["stage4_local_rmsd_a"]),
        pocket_volume_change_fraction=float(volume_change_fraction),
        solvent_proxy_shift=float(solvent_proxy_shift),
        stage5=stage5,
    )
    return {
        **base,
        "ifp_status": "ok",
        "ifp_error": None,
        "refinement_status": refinement_status,
        "refinement_error": refinement_error,
        "local_sampling_applied": bool(local_sampling_applied),
        "ifp_jaccard_loss": float(ifp_jaccard_loss_value),
        "anchor_loss_fraction": float(anchor_loss_value),
        "wt_residue_count": int(len(wt_labels)),
        "mt_residue_count": int(len(mt_labels)),
        "lost_residue_labels_json": json.dumps(sorted(wt_labels - mt_labels), ensure_ascii=True),
        "gained_residue_labels_json": json.dumps(sorted(mt_labels - wt_labels), ensure_ascii=True),
        "lost_anchor_labels_json": json.dumps(sorted(baseline_anchor_labels - mt_labels), ensure_ascii=True),
        "hydrogen_bond_delta_count": int(mt_counts["hydrogen_bond"] - wt_counts["hydrogen_bond"]),
        "salt_bridge_delta_count": int(mt_counts["salt_bridge"] - wt_counts["salt_bridge"]),
        "hydrophobic_delta_count": int(mt_counts["hydrophobic"] - wt_counts["hydrophobic"]),
        "pi_stacking_delta_count": int(mt_counts["pi_stacking"] - wt_counts["pi_stacking"]),
        "pi_cation_delta_count": int(mt_counts["pi_cation"] - wt_counts["pi_cation"]),
        "metal_complex_delta_count": int(mt_counts["metal_complex"] - wt_counts["metal_complex"]),
        "wt_pocket_volume_proxy_a3": float(wt_pocket["pocket_volume_proxy_a3"]),
        "mt_pocket_volume_proxy_a3": float(mt_pocket["pocket_volume_proxy_a3"]),
        "wt_fpocket_volume_a3": native_value(wt_fpocket_volume_a3),
        "mt_fpocket_volume_a3": native_value(mt_fpocket_volume_a3),
        "fpocket_error": fpocket_error,
        "pocket_volume_change_fraction": float(volume_change_fraction),
        "wt_contact_density": float(wt_pocket["contact_density"]),
        "mt_contact_density": float(mt_pocket["contact_density"]),
        "contact_density_change": float(mt_pocket["contact_density"] - wt_pocket["contact_density"]),
        "wt_polar_exposed_fraction": float(wt_pocket["polar_exposed_fraction"]),
        "mt_polar_exposed_fraction": float(mt_pocket["polar_exposed_fraction"]),
        "solvent_proxy_shift": float(solvent_proxy_shift),
        **occupancy_payload,
        "mechanism_labels_json": json.dumps(labels, ensure_ascii=True),
        "mechanism_signature": mechanism_signature(labels),
    }


def parse_json_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def predicted_effect_score(row: pd.Series | dict[str, Any]) -> float:
    if isinstance(row, dict):
        series = pd.Series(row)
    else:
        series = row
    return float(
        max(0.0, float(series.get("delta_dock_kcal_mol") or 0.0))
        + 0.5 * max(0.0, float(series.get("ifp_jaccard_loss") or 0.0))
        + 0.5 * max(0.0, float(series.get("anchor_loss_fraction") or 0.0))
        + 0.2 * max(0.0, abs(float(series.get("pocket_volume_change_fraction") or 0.0)))
        + 0.2 * max(0.0, float(series.get("solvent_proxy_shift") or 0.0))
    )


def assign_epistasis_flag(combo_row: pd.Series, site_lookup: dict[str, pd.Series]) -> str:
    target_key = str(combo_row["target_key"])
    components = target_components(target_key)
    if len(components) <= 1:
        return "unresolved"
    gene_prefix = target_key.split(":", 1)[0] if ":" in target_key else ""
    component_keys = [f"{gene_prefix}:{component.raw}" if gene_prefix else component.raw for component in components]
    site_rows: list[pd.Series] = []
    for key in component_keys:
        row = site_lookup.get(key)
        if row is None or str(row.get("ifp_status") or "") != "ok":
            return "unresolved"
        site_rows.append(row)
    observed = predicted_effect_score(combo_row)
    expected = float(sum(predicted_effect_score(row) for row in site_rows))
    tolerance = max(0.35, 0.25 * expected)
    if math.isclose(observed, expected, abs_tol=tolerance, rel_tol=0.25):
        return "additive_like"
    return "non_additive"


def mechanism_cluster_frame(effect_frame: pd.DataFrame) -> pd.DataFrame:
    if effect_frame.empty:
        return pd.DataFrame(
            columns=["effect_scope", "mechanism_signature", "target_count", "case_count", "case_ids", "target_keys"]
        )
    grouped = (
        effect_frame.groupby(["effect_scope", "mechanism_signature"], dropna=False)
        .agg(
            target_count=("target_key", "size"),
            case_count=("case_id", pd.Series.nunique),
            case_ids=("case_id", lambda values: "|".join(sorted({str(value) for value in values}))),
            target_keys=("target_key", lambda values: "|".join(sorted({str(value) for value in values}))),
        )
        .reset_index()
        .sort_values(["effect_scope", "target_count", "mechanism_signature"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return grouped


def load_empirical_ddg_lookup(root: Path, stage1: dict[str, Any], sample_ids: list[str]) -> pd.DataFrame:
    requested_ids = sorted({str(value) for value in sample_ids if str(value).startswith("MdrDB")})
    if not requested_ids:
        return pd.DataFrame(columns=["SAMPLE_ID", "DDG.EXP"])
    main_path = root / str(stage1["mdrdb_main"])
    frame = pd.read_csv(
        main_path,
        sep="\t",
        usecols=["SAMPLE_ID", "DDG.EXP"],
        dtype={"SAMPLE_ID": "string"},
        low_memory=False,
    )
    frame = frame[frame["SAMPLE_ID"].isin(requested_ids)].copy()
    frame["DDG.EXP"] = pd.to_numeric(frame["DDG.EXP"], errors="coerce")
    return frame.dropna(subset=["DDG.EXP"]).drop_duplicates("SAMPLE_ID").reset_index(drop=True)


def calibration_frame(
    *,
    effect_frame: pd.DataFrame,
    ddg_lookup: pd.DataFrame,
) -> pd.DataFrame:
    output_columns = [
        "case_id",
        "effect_scope",
        "target_key",
        "representative_sample_id",
        "predicted_effect_score",
        "delta_dock_kcal_mol",
        "delta_mmgbsa_binding_kcal_mol",
        "delta_gnina_affinity_kcal_mol",
        "ifp_jaccard_loss",
        "anchor_loss_fraction",
        "pocket_volume_change_fraction",
        "solvent_proxy_shift",
        "experimental_ddg_exp",
    ]
    if effect_frame.empty:
        return pd.DataFrame(columns=output_columns)
    merged = effect_frame.copy()
    if not ddg_lookup.empty:
        merged = merged.merge(
            ddg_lookup.rename(columns={"SAMPLE_ID": "representative_sample_id", "DDG.EXP": "experimental_ddg_exp"}),
            on="representative_sample_id",
            how="left",
        )
    else:
        merged["experimental_ddg_exp"] = pd.Series(dtype=float)
    for column in output_columns:
        if column not in merged.columns:
            merged[column] = pd.Series(dtype=float if column == "experimental_ddg_exp" else object)
    merged["predicted_effect_score"] = merged.apply(predicted_effect_score, axis=1)
    merged = merged[merged["delta_mmgbsa_binding_kcal_mol"].notna()].copy()
    return merged[output_columns].sort_values(["case_id", "effect_scope", "target_key"]).reset_index(drop=True)


def calibration_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or len(frame) < 2:
        return {
            "calibration_sample_count": int(len(frame)),
            "dock_vs_mmgbsa_pearson_r": None,
            "dock_vs_mmgbsa_spearman_r": None,
            "predicted_vs_mmgbsa_pearson_r": None,
            "predicted_vs_mmgbsa_spearman_r": None,
            "experimental_alignment_pearson_r": None,
            "experimental_alignment_spearman_r": None,
        }
    experimental_frame = frame.dropna(subset=["experimental_ddg_exp"]).copy()
    return {
        "calibration_sample_count": int(len(frame)),
        "dock_vs_mmgbsa_pearson_r": float(frame["delta_dock_kcal_mol"].corr(frame["delta_mmgbsa_binding_kcal_mol"], method="pearson")),
        "dock_vs_mmgbsa_spearman_r": float(
            frame["delta_dock_kcal_mol"].corr(frame["delta_mmgbsa_binding_kcal_mol"], method="spearman")
        ),
        "predicted_vs_mmgbsa_pearson_r": float(
            frame["predicted_effect_score"].corr(frame["delta_mmgbsa_binding_kcal_mol"], method="pearson")
        ),
        "predicted_vs_mmgbsa_spearman_r": float(
            frame["predicted_effect_score"].corr(frame["delta_mmgbsa_binding_kcal_mol"], method="spearman")
        ),
        "experimental_alignment_pearson_r": None
        if experimental_frame.empty or len(experimental_frame) < 2
        else float(
            experimental_frame["delta_mmgbsa_binding_kcal_mol"].corr(experimental_frame["experimental_ddg_exp"], method="pearson")
        ),
        "experimental_alignment_spearman_r": None
        if experimental_frame.empty or len(experimental_frame) < 2
        else float(
            experimental_frame["delta_mmgbsa_binding_kcal_mol"].corr(experimental_frame["experimental_ddg_exp"], method="spearman")
        ),
    }


def hashed_inputs(paths: list[Path], root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        if path.exists():
            hashes[relative_path(path, root)] = sha256_file(path)
    return hashes
