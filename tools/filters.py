#!/usr/bin/env python3
"""counter-design step prefilters and descriptor-driven QC gates."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _native_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        if not math.isfinite(value):
            return None
        return float(value)
    return None


def _sa_score_fallback(mol: Chem.Mol) -> float:
    heavy_atoms = float(mol.GetNumHeavyAtoms())
    ring_count = float(rdMolDescriptors.CalcNumRings(mol))
    stereo_count = float(rdMolDescriptors.CalcNumAtomStereoCenters(mol))
    hetero_count = float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6}))
    spiro_count = float(rdMolDescriptors.CalcNumSpiroAtoms(mol))
    bridge_count = float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    estimate = 1.0 + 0.08 * heavy_atoms + 0.35 * ring_count + 0.30 * stereo_count + 0.06 * hetero_count
    estimate += 0.45 * spiro_count + 0.35 * bridge_count
    return _clamp(estimate, 1.0, 10.0)


def synthetic_accessibility_score(mol: Chem.Mol) -> float:
    try:
        from rdkit.Contrib.SA_Score import sascorer  # type: ignore

        return float(sascorer.calculateScore(mol))
    except Exception:
        return _sa_score_fallback(mol)


@lru_cache(maxsize=1)
def pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


def pains_matches(mol: Chem.Mol) -> list[str]:
    catalog = pains_catalog()
    matches = catalog.GetMatches(mol)
    labels: list[str] = []
    for match in matches:
        label = str(match.GetDescription() or "").strip()
        if label:
            labels.append(label)
    return sorted(set(labels))


def _smarts_catalog(patterns: dict[str, str]) -> tuple[tuple[str, Chem.Mol], ...]:
    catalog: list[tuple[str, Chem.Mol]] = []
    for label, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            catalog.append((str(label), pattern))
    return tuple(catalog)


@lru_cache(maxsize=1)
def medchem_blacklist_catalog() -> tuple[tuple[str, Chem.Mol], ...]:
    return _smarts_catalog(
        {
            "acyl_halide": "[CX3](=[OX1])[F,Cl,Br,I]",
            "sulfonyl_halide": "[SX4](=[OX1])(=[OX1])[F,Cl,Br,I]",
            "acid_anhydride": "[CX3](=O)O[CX3](=O)",
            "isocyanate": "[NX2]=[CX2]=[OX1]",
            "isothiocyanate": "[NX2]=[CX2]=[SX1]",
            "peroxide": "[OX2][OX2]",
        }
    )


@lru_cache(maxsize=1)
def unstable_motif_catalog() -> tuple[tuple[str, Chem.Mol], ...]:
    return _smarts_catalog(
        {
            "aldehyde": "[CX3H1](=O)[#6]",
            "michael_acceptor": "[CX3]=[CX3][CX3](=O)[#6,#7,#8]",
            "epoxide": "[O;r3]1[C;r3][C;r3]1",
            "imine": "[CX3]=[NX2]",
        }
    )


def _pattern_matches(mol: Chem.Mol, catalog: tuple[tuple[str, Chem.Mol], ...]) -> list[str]:
    labels: list[str] = []
    for label, pattern in catalog:
        if mol.HasSubstructMatch(pattern):
            labels.append(str(label))
    return sorted(set(labels))


def medchem_blacklist_matches(mol: Chem.Mol) -> list[str]:
    return _pattern_matches(mol, medchem_blacklist_catalog())


def unstable_motif_matches(mol: Chem.Mol) -> list[str]:
    return _pattern_matches(mol, unstable_motif_catalog())


def _morgan_fp(mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
    return AllChem.GetMorganFingerprintAsBitVect(Chem.RemoveHs(Chem.Mol(mol)), 2, nBits=2048)


def _safe_scaffold_smiles(mol: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(Chem.RemoveHs(Chem.Mol(mol)))
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold, canonical=True)


def reaction_accessibility_score(
    mol: Chem.Mol,
    *,
    sa_score: float,
    pains_count: int,
    medchem_blacklist_count: int,
    unstable_motif_count: int,
) -> float:
    heavy_atoms = float(mol.GetNumHeavyAtoms())
    ring_count = float(rdMolDescriptors.CalcNumRings(mol))
    stereo_count = float(rdMolDescriptors.CalcNumAtomStereoCenters(mol))
    spiro_count = float(rdMolDescriptors.CalcNumSpiroAtoms(mol))
    bridge_count = float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    fraction_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
    score = 1.15
    score -= 0.10 * max(0.0, float(sa_score) - 1.0)
    score -= 0.01 * max(0.0, heavy_atoms - 22.0)
    score -= 0.03 * max(0.0, ring_count - 3.0)
    score -= 0.04 * stereo_count
    score -= 0.05 * spiro_count
    score -= 0.05 * bridge_count
    score -= 0.08 * float(pains_count)
    score -= 0.12 * float(medchem_blacklist_count)
    score -= 0.06 * float(unstable_motif_count)
    score += 0.10 * fraction_csp3
    return float(_clamp01(score))


def scscore_heuristic(
    mol: Chem.Mol,
    *,
    sa_score: float,
    pains_count: int,
    medchem_blacklist_count: int,
    unstable_motif_count: int,
) -> float:
    heavy_atoms = float(mol.GetNumHeavyAtoms())
    ring_count = float(rdMolDescriptors.CalcNumRings(mol))
    stereo_count = float(rdMolDescriptors.CalcNumAtomStereoCenters(mol))
    spiro_count = float(rdMolDescriptors.CalcNumSpiroAtoms(mol))
    bridge_count = float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    score = 1.0
    score += 0.45 * max(0.0, float(sa_score) - 1.0)
    score += 0.015 * heavy_atoms
    score += 0.08 * ring_count
    score += 0.12 * stereo_count
    score += 0.12 * spiro_count
    score += 0.10 * bridge_count
    score += 0.15 * float(pains_count + medchem_blacklist_count + unstable_motif_count)
    return float(_clamp(score, 1.0, 5.0))


def retrosynthesis_plausibility_score(
    *,
    reaction_accessibility: float,
    scscore: float,
    medchem_blacklist_count: int,
    unstable_motif_count: int,
) -> float:
    sc_penalty = _clamp01((float(scscore) - 1.0) / 4.0)
    motif_penalty = _clamp01(0.35 * float(medchem_blacklist_count) + 0.15 * float(unstable_motif_count))
    score = (
        0.55 * float(reaction_accessibility)
        + 0.30 * float(1.0 - sc_penalty)
        + 0.15 * float(1.0 - motif_penalty)
    )
    return float(_clamp01(score))


def series_likeness_metrics(
    mol: Chem.Mol,
    *,
    baseline_smiles: str | None,
) -> dict[str, Any]:
    baseline = Chem.MolFromSmiles(str(baseline_smiles)) if baseline_smiles else None
    if baseline is None:
        return {
            "series_similarity": 1.0,
            "series_scaffold_similarity": 1.0,
            "series_scaffold_match": True,
            "series_likeness_score": 1.0,
        }
    candidate_fp = _morgan_fp(mol)
    baseline_fp = _morgan_fp(baseline)
    series_similarity = float(DataStructs.TanimotoSimilarity(candidate_fp, baseline_fp))
    candidate_scaffold = _safe_scaffold_smiles(mol)
    baseline_scaffold = _safe_scaffold_smiles(baseline)
    if not candidate_scaffold and not baseline_scaffold:
        scaffold_similarity = 1.0
        scaffold_match = True
    elif candidate_scaffold and baseline_scaffold:
        scaffold_match = bool(candidate_scaffold == baseline_scaffold)
        scaffold_similarity = 1.0 if scaffold_match else float(
            DataStructs.TanimotoSimilarity(
                _morgan_fp(Chem.MolFromSmiles(candidate_scaffold)),
                _morgan_fp(Chem.MolFromSmiles(baseline_scaffold)),
            )
        )
    else:
        scaffold_similarity = 0.0
        scaffold_match = False
    heavy_delta = abs(float(mol.GetNumHeavyAtoms()) - float(baseline.GetNumHeavyAtoms()))
    heavy_proximity = math.exp(-heavy_delta / 8.0)
    likeness = 0.55 * series_similarity + 0.30 * scaffold_similarity + 0.15 * heavy_proximity
    return {
        "series_similarity": float(series_similarity),
        "series_scaffold_similarity": float(scaffold_similarity),
        "series_scaffold_match": bool(scaffold_match),
        "series_likeness_score": float(_clamp01(likeness)),
    }


def _prefilter_thresholds(config: dict[str, Any]) -> dict[str, float]:
    thresholds = dict(config.get("prefilter", {}))
    return {
        "qed_min": float(thresholds.get("qed_min", 0.35)),
        "sa_max": float(thresholds.get("sa_max", 6.5)),
        "ra_score_min": float(thresholds.get("ra_score_min", 0.30)),
        "scscore_max": float(thresholds.get("scscore_max", 4.5)),
        "retrosynthesis_plausibility_min": float(thresholds.get("retrosynthesis_plausibility_min", 0.30)),
        "series_likeness_min": float(thresholds.get("series_likeness_min", 0.25)),
        "clogp_min": float(thresholds.get("clogp_min", -0.5)),
        "clogp_max": float(thresholds.get("clogp_max", 5.5)),
        "hbd_max": float(thresholds.get("hbd_max", 6)),
        "hba_max": float(thresholds.get("hba_max", 12)),
        "tpsa_max": float(thresholds.get("tpsa_max", 160.0)),
        "mw_max": float(thresholds.get("mw_max", 650.0)),
        "rotatable_bonds_max": float(thresholds.get("rotatable_bonds_max", 14)),
    }


def _baseline_tolerances(config: dict[str, Any]) -> dict[str, float]:
    tolerances = dict(config.get("prefilter", {}).get("baseline_tolerance", {}))
    return {
        "qed_drop": float(tolerances.get("qed_drop", 0.03)),
        "sa_increase": float(tolerances.get("sa_increase", 0.5)),
        "clogp_decrease": float(tolerances.get("clogp_decrease", 0.4)),
        "clogp_increase": float(tolerances.get("clogp_increase", 0.4)),
        "hbd_increase": float(tolerances.get("hbd_increase", 1.0)),
        "hba_increase": float(tolerances.get("hba_increase", 1.0)),
        "tpsa_increase": float(tolerances.get("tpsa_increase", 15.0)),
        "mw_increase": float(tolerances.get("mw_increase", 60.0)),
        "rotatable_bonds_increase": float(tolerances.get("rotatable_bonds_increase", 2.0)),
    }


def _native_descriptor(value: Any) -> float | None:
    return _native_float(value)


def _passes_lower_bound(
    *,
    value: float | None,
    threshold: float,
    baseline: float | None,
    tolerance: float,
) -> tuple[bool, bool]:
    if value is None:
        return False, False
    if float(value) >= float(threshold):
        return True, False
    if baseline is not None and float(baseline) < float(threshold) and float(value) >= float(baseline) - float(tolerance):
        return True, True
    return False, False


def _passes_upper_bound(
    *,
    value: float | None,
    threshold: float,
    baseline: float | None,
    tolerance: float,
) -> tuple[bool, bool]:
    if value is None:
        return False, False
    if float(value) <= float(threshold):
        return True, False
    if baseline is not None and float(baseline) > float(threshold) and float(value) <= float(baseline) + float(tolerance):
        return True, True
    return False, False


def descriptor_payload(mol: Chem.Mol) -> dict[str, Any]:
    pains_alerts = pains_matches(mol)
    medchem_blacklist_labels = medchem_blacklist_matches(mol)
    unstable_motif_labels = unstable_motif_matches(mol)
    sa_score = float(synthetic_accessibility_score(mol))
    ra_score = reaction_accessibility_score(
        mol,
        sa_score=sa_score,
        pains_count=len(pains_alerts),
        medchem_blacklist_count=len(medchem_blacklist_labels),
        unstable_motif_count=len(unstable_motif_labels),
    )
    scscore = scscore_heuristic(
        mol,
        sa_score=sa_score,
        pains_count=len(pains_alerts),
        medchem_blacklist_count=len(medchem_blacklist_labels),
        unstable_motif_count=len(unstable_motif_labels),
    )
    retrosynthesis_score = retrosynthesis_plausibility_score(
        reaction_accessibility=ra_score,
        scscore=scscore,
        medchem_blacklist_count=len(medchem_blacklist_labels),
        unstable_motif_count=len(unstable_motif_labels),
    )
    return {
        "canonical_smiles": Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), canonical=True),
        "qed": float(QED.qed(mol)),
        "sa_score": sa_score,
        "ra_score": float(ra_score),
        "scscore": float(scscore),
        "retrosynthesis_plausibility": float(retrosynthesis_score),
        "clogp": float(Crippen.MolLogP(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "mw": float(Descriptors.MolWt(mol)),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "stereo_count": int(rdMolDescriptors.CalcNumAtomStereoCenters(mol)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "pains_alert_count": int(len(pains_alerts)),
        "pains_alerts": pains_alerts,
        "medchem_blacklist_count": int(len(medchem_blacklist_labels)),
        "medchem_blacklist_labels": medchem_blacklist_labels,
        "unstable_motif_count": int(len(unstable_motif_labels)),
        "unstable_motif_labels": unstable_motif_labels,
    }


def admet_penalty(descriptors: dict[str, Any], config: dict[str, Any]) -> float:
    thresholds = _prefilter_thresholds(config)
    penalty = 0.0
    qed = _native_float(descriptors.get("qed"))
    sa_score = _native_float(descriptors.get("sa_score"))
    clogp = _native_float(descriptors.get("clogp"))
    hbd = _native_float(descriptors.get("hbd"))
    hba = _native_float(descriptors.get("hba"))
    tpsa = _native_float(descriptors.get("tpsa"))
    mw = _native_float(descriptors.get("mw"))
    rotatable = _native_float(descriptors.get("rotatable_bonds"))
    pains_count = int(descriptors.get("pains_alert_count") or 0)

    if qed is not None and qed < thresholds["qed_min"]:
        penalty += (thresholds["qed_min"] - qed) / max(thresholds["qed_min"], 1e-6)
    if sa_score is not None and sa_score > thresholds["sa_max"]:
        penalty += (sa_score - thresholds["sa_max"]) / max(thresholds["sa_max"], 1e-6)
    if clogp is not None and clogp < thresholds["clogp_min"]:
        penalty += (thresholds["clogp_min"] - clogp) / max(abs(thresholds["clogp_min"]), 1.0)
    if clogp is not None and clogp > thresholds["clogp_max"]:
        penalty += (clogp - thresholds["clogp_max"]) / max(thresholds["clogp_max"], 1.0)
    if hbd is not None and hbd > thresholds["hbd_max"]:
        penalty += (hbd - thresholds["hbd_max"]) / max(thresholds["hbd_max"], 1.0)
    if hba is not None and hba > thresholds["hba_max"]:
        penalty += (hba - thresholds["hba_max"]) / max(thresholds["hba_max"], 1.0)
    if tpsa is not None and tpsa > thresholds["tpsa_max"]:
        penalty += (tpsa - thresholds["tpsa_max"]) / max(thresholds["tpsa_max"], 1.0)
    if mw is not None and mw > thresholds["mw_max"]:
        penalty += (mw - thresholds["mw_max"]) / max(thresholds["mw_max"], 1.0)
    if rotatable is not None and rotatable > thresholds["rotatable_bonds_max"]:
        penalty += (rotatable - thresholds["rotatable_bonds_max"]) / max(thresholds["rotatable_bonds_max"], 1.0)
    penalty += float(pains_count)
    return float(_clamp(penalty, 0.0, 5.0))


def synthesis_penalty(descriptors: dict[str, Any], config: dict[str, Any]) -> float:
    thresholds = _prefilter_thresholds(config)
    penalty = 0.0
    ra_score = _native_float(descriptors.get("ra_score"))
    scscore = _native_float(descriptors.get("scscore"))
    retrosynthesis = _native_float(descriptors.get("retrosynthesis_plausibility"))
    series_likeness = _native_float(descriptors.get("series_likeness_score"))
    medchem_blacklist_count = int(descriptors.get("medchem_blacklist_count") or 0)
    unstable_motif_count = int(descriptors.get("unstable_motif_count") or 0)

    if ra_score is not None and ra_score < thresholds["ra_score_min"]:
        penalty += (thresholds["ra_score_min"] - ra_score) / max(thresholds["ra_score_min"], 1.0e-6)
    if scscore is not None and scscore > thresholds["scscore_max"]:
        penalty += (scscore - thresholds["scscore_max"]) / max(thresholds["scscore_max"], 1.0)
    if retrosynthesis is not None and retrosynthesis < thresholds["retrosynthesis_plausibility_min"]:
        penalty += (thresholds["retrosynthesis_plausibility_min"] - retrosynthesis) / max(
            thresholds["retrosynthesis_plausibility_min"], 1.0e-6
        )
    if series_likeness is not None and series_likeness < thresholds["series_likeness_min"]:
        penalty += (thresholds["series_likeness_min"] - series_likeness) / max(
            thresholds["series_likeness_min"], 1.0e-6
        )
    penalty += 0.8 * float(medchem_blacklist_count)
    penalty += 0.25 * float(unstable_motif_count)
    return float(_clamp(penalty, 0.0, 5.0))


def apply_prefilters(
    mol: Chem.Mol | str,
    config: dict[str, Any],
    *,
    baseline_descriptors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(str(mol)) if isinstance(mol, str) else Chem.Mol(mol)
    if molecule is None:
        return {
            "prefilter_pass": False,
            "prefilter_fail_reasons": ["invalid_molecule"],
            "prefilter_fail_reason": "invalid_molecule",
            "prefilter_warning_reasons": [],
            "prefilter_warning_reason": "",
            "docking_skipped": True,
            "admet_penalty": 1.0,
            "synthesis_penalty": 1.0,
            "total_penalty": 2.0,
            "prefilter_score": 0.0,
        }
    descriptors = descriptor_payload(molecule)
    thresholds = _prefilter_thresholds(config)
    tolerances = _baseline_tolerances(config)
    baseline = baseline_descriptors or {}
    descriptors.update(
        series_likeness_metrics(
            molecule,
            baseline_smiles=str(baseline.get("canonical_smiles") or ""),
        )
    )
    fail_reasons: list[str] = []
    warning_reasons: list[str] = []

    def register_bound_check(
        *,
        reason: str,
        passed: bool,
        baseline_allowed: bool,
        should_warn: bool,
    ) -> None:
        if should_warn:
            warning_reasons.append(reason)
        if not passed and not baseline_allowed:
            fail_reasons.append(reason)

    qed_passed, qed_baseline_allowed = _passes_lower_bound(
        value=_native_descriptor(descriptors.get("qed")),
        threshold=thresholds["qed_min"],
        baseline=_native_descriptor(baseline.get("qed")),
        tolerance=tolerances["qed_drop"],
    )
    register_bound_check(
        reason="qed_below_min",
        passed=qed_passed,
        baseline_allowed=qed_baseline_allowed,
        should_warn=float(descriptors["qed"]) < thresholds["qed_min"],
    )

    sa_passed, sa_baseline_allowed = _passes_upper_bound(
        value=_native_descriptor(descriptors.get("sa_score")),
        threshold=thresholds["sa_max"],
        baseline=_native_descriptor(baseline.get("sa_score")),
        tolerance=tolerances["sa_increase"],
    )
    register_bound_check(
        reason="sa_above_max",
        passed=sa_passed,
        baseline_allowed=sa_baseline_allowed,
        should_warn=float(descriptors["sa_score"]) > thresholds["sa_max"],
    )

    clogp_low_passed, clogp_low_baseline_allowed = _passes_lower_bound(
        value=_native_descriptor(descriptors.get("clogp")),
        threshold=thresholds["clogp_min"],
        baseline=_native_descriptor(baseline.get("clogp")),
        tolerance=tolerances["clogp_decrease"],
    )
    register_bound_check(
        reason="clogp_below_min",
        passed=clogp_low_passed,
        baseline_allowed=clogp_low_baseline_allowed,
        should_warn=float(descriptors["clogp"]) < thresholds["clogp_min"],
    )

    clogp_high_passed, clogp_high_baseline_allowed = _passes_upper_bound(
        value=_native_descriptor(descriptors.get("clogp")),
        threshold=thresholds["clogp_max"],
        baseline=_native_descriptor(baseline.get("clogp")),
        tolerance=tolerances["clogp_increase"],
    )
    register_bound_check(
        reason="clogp_above_max",
        passed=clogp_high_passed,
        baseline_allowed=clogp_high_baseline_allowed,
        should_warn=float(descriptors["clogp"]) > thresholds["clogp_max"],
    )

    hbd_passed, hbd_baseline_allowed = _passes_upper_bound(
        value=_native_descriptor(descriptors.get("hbd")),
        threshold=thresholds["hbd_max"],
        baseline=_native_descriptor(baseline.get("hbd")),
        tolerance=tolerances["hbd_increase"],
    )
    register_bound_check(
        reason="hbd_above_max",
        passed=hbd_passed,
        baseline_allowed=hbd_baseline_allowed,
        should_warn=int(descriptors["hbd"]) > int(thresholds["hbd_max"]),
    )

    hba_passed, hba_baseline_allowed = _passes_upper_bound(
        value=_native_descriptor(descriptors.get("hba")),
        threshold=thresholds["hba_max"],
        baseline=_native_descriptor(baseline.get("hba")),
        tolerance=tolerances["hba_increase"],
    )
    register_bound_check(
        reason="hba_above_max",
        passed=hba_passed,
        baseline_allowed=hba_baseline_allowed,
        should_warn=int(descriptors["hba"]) > int(thresholds["hba_max"]),
    )

    tpsa_passed, tpsa_baseline_allowed = _passes_upper_bound(
        value=_native_descriptor(descriptors.get("tpsa")),
        threshold=thresholds["tpsa_max"],
        baseline=_native_descriptor(baseline.get("tpsa")),
        tolerance=tolerances["tpsa_increase"],
    )
    register_bound_check(
        reason="tpsa_above_max",
        passed=tpsa_passed,
        baseline_allowed=tpsa_baseline_allowed,
        should_warn=float(descriptors["tpsa"]) > thresholds["tpsa_max"],
    )

    mw_passed, mw_baseline_allowed = _passes_upper_bound(
        value=_native_descriptor(descriptors.get("mw")),
        threshold=thresholds["mw_max"],
        baseline=_native_descriptor(baseline.get("mw")),
        tolerance=tolerances["mw_increase"],
    )
    register_bound_check(
        reason="mw_above_max",
        passed=mw_passed,
        baseline_allowed=mw_baseline_allowed,
        should_warn=float(descriptors["mw"]) > thresholds["mw_max"],
    )

    rotatable_passed, rotatable_baseline_allowed = _passes_upper_bound(
        value=_native_descriptor(descriptors.get("rotatable_bonds")),
        threshold=thresholds["rotatable_bonds_max"],
        baseline=_native_descriptor(baseline.get("rotatable_bonds")),
        tolerance=tolerances["rotatable_bonds_increase"],
    )
    register_bound_check(
        reason="rotatable_bonds_above_max",
        passed=rotatable_passed,
        baseline_allowed=rotatable_baseline_allowed,
        should_warn=int(descriptors["rotatable_bonds"]) > int(thresholds["rotatable_bonds_max"]),
    )

    if float(descriptors["ra_score"]) < thresholds["ra_score_min"]:
        fail_reasons.append("ra_score_below_min")
    if float(descriptors["scscore"]) > thresholds["scscore_max"]:
        fail_reasons.append("scscore_above_max")
    if float(descriptors["retrosynthesis_plausibility"]) < thresholds["retrosynthesis_plausibility_min"]:
        fail_reasons.append("retrosynthesis_plausibility_below_min")
    if float(descriptors["series_likeness_score"]) < thresholds["series_likeness_min"]:
        fail_reasons.append("series_likeness_below_min")
    if int(descriptors["pains_alert_count"]) > 0:
        fail_reasons.append("pains_alert")
    if int(descriptors["medchem_blacklist_count"]) > 0 and bool(config.get("prefilter", {}).get("medchem_blacklist_hard_fail", True)):
        fail_reasons.append("medchem_blacklist")
    if int(descriptors["unstable_motif_count"]) > 0:
        warning_reasons.append("unstable_motif_present")
        if bool(config.get("prefilter", {}).get("unstable_motif_hard_fail", False)):
            fail_reasons.append("unstable_motif")
    admet = admet_penalty(descriptors, config)
    synth = synthesis_penalty(descriptors, config)
    total_penalty = float(admet + synth)
    passed = not fail_reasons
    return {
        **descriptors,
        "prefilter_pass": bool(passed),
        "prefilter_fail_reasons": fail_reasons,
        "prefilter_fail_reason": ";".join(fail_reasons),
        "prefilter_warning_reasons": warning_reasons,
        "prefilter_warning_reason": ";".join(warning_reasons),
        "docking_skipped": not bool(passed),
        "admet_penalty": float(admet),
        "synthesis_penalty": float(synth),
        "total_penalty": float(total_penalty),
        "prefilter_score": 1.0 if passed else 0.0,
    }
