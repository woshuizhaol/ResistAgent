#!/usr/bin/env python3
"""counter-design step reward-v2 helpers."""

from __future__ import annotations

import math
from typing import Any


LAYER_NAMES = (
    "keep_ifp_anchor",
    "keep_ifp_backbone",
    "keep_ifp_partner_chain",
    "keep_ifp_nonhotspot",
)


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def normalize_positive(value: float | None, scale: float) -> float:
    if value is None:
        return 0.0
    return _clamp01(1.0 - math.exp(-max(0.0, float(value)) / max(float(scale), 1.0e-6)))


def reward_v2_layer_weights(
    *,
    case_id: str,
    target_domain: str,
    stage6: dict[str, Any],
    layer_available: dict[str, bool] | None = None,
) -> dict[str, float]:
    layer_available = dict(layer_available or {})
    base = {
        "keep_ifp_anchor": 0.40,
        "keep_ifp_backbone": 0.25,
        "keep_ifp_partner_chain": 0.10,
        "keep_ifp_nonhotspot": 0.25,
    }
    case_key = str(case_id)
    domain_key = str(target_domain or "").lower()
    if case_key == "egfr_erlotinib":
        base = {
            "keep_ifp_anchor": 0.55,
            "keep_ifp_backbone": 0.20,
            "keep_ifp_partner_chain": 0.05,
            "keep_ifp_nonhotspot": 0.20,
        }
    elif case_key == "hiv_rt_rilpivirine" or domain_key == "rt":
        base = {
            "keep_ifp_anchor": 0.20,
            "keep_ifp_backbone": 0.10,
            "keep_ifp_partner_chain": 0.35,
            "keep_ifp_nonhotspot": 0.35,
        }
    elif case_key == "abl1_nilotinib":
        base = {
            "keep_ifp_anchor": 0.20,
            "keep_ifp_backbone": 0.45,
            "keep_ifp_partner_chain": 0.05,
            "keep_ifp_nonhotspot": 0.30,
        }
    configured = dict(stage6.get("reward_v2_keep_ifp_weights", {}))
    weights = {name: float(configured.get(name, base[name])) for name in LAYER_NAMES}
    available_weights = {
        name: float(weight)
        for name, weight in weights.items()
        if bool(layer_available.get(name, True)) and float(weight) > 0.0
    }
    total = float(sum(available_weights.values()))
    if total <= 0.0:
        fallback = [name for name in LAYER_NAMES if bool(layer_available.get(name, True))]
        if not fallback:
            return {name: 0.0 for name in LAYER_NAMES}
        equal = float(1.0 / len(fallback))
        return {name: (equal if name in fallback else 0.0) for name in LAYER_NAMES}
    return {name: float(available_weights.get(name, 0.0) / total) for name in LAYER_NAMES}


def weighted_keep_ifp(layer_scores: dict[str, float], layer_weights: dict[str, float]) -> float:
    return float(
        sum(float(layer_scores.get(name, 0.0)) * float(layer_weights.get(name, 0.0)) for name in LAYER_NAMES)
    )


def oracle_uncertainty_score(
    *,
    effective_trust_score: float | None,
    pred_std_mean: float | None,
    stage6: dict[str, Any],
) -> dict[str, float]:
    trust_penalty = _clamp01(1.0 - float(effective_trust_score)) if effective_trust_score is not None else 1.0
    pred_std_penalty = normalize_positive(
        pred_std_mean,
        float(stage6.get("reward_v2_oracle_uncertainty_scale", 2.0)),
    )
    oracle_uncertainty = float(
        0.5 * trust_penalty + 0.5 * pred_std_penalty
    )
    return {
        "oracle_uncertainty": float(oracle_uncertainty),
        "oracle_uncertainty_trust_penalty": float(trust_penalty),
        "oracle_uncertainty_pred_std_penalty": float(pred_std_penalty),
    }


def alt_anchor_score(
    *,
    layer_scores: dict[str, float],
    new_nonhotspot_score: float,
    compensation_gain: float,
    hotspot_fraction: float,
    stage6: dict[str, Any],
) -> dict[str, float]:
    new_nonhotspot_norm = normalize_positive(
        new_nonhotspot_score,
        float(stage6.get("reward_v2_new_nonhotspot_scale", 4.0)),
    )
    compensation_norm = normalize_positive(
        compensation_gain,
        float(stage6.get("reward_v2_compensation_scale", 4.0)),
    )
    support_score = float(
        0.45 * float(layer_scores.get("keep_ifp_nonhotspot", 0.0))
        + 0.35 * float(layer_scores.get("keep_ifp_partner_chain", 0.0))
        + 0.20 * float(layer_scores.get("keep_ifp_backbone", 0.0))
    )
    alternative_anchor = _clamp01(
        0.60 * support_score
        + 0.25 * max(new_nonhotspot_norm, compensation_norm)
        + 0.15 * float(max(0.0, 1.0 - float(hotspot_fraction)))
    )
    return {
        "alt_anchor_score": float(alternative_anchor),
        "new_nonhotspot_score": float(new_nonhotspot_norm),
        "compensation_gain_score": float(compensation_norm),
    }


def reward_v2_components(
    *,
    s_wt: float,
    robust_site_core: float,
    robust_combo_core: float | None,
    combo_dense_case: bool,
    alt_anchor_score_value: float,
    new_nonhotspot_score: float,
    hotspot_fraction: float,
    oracle_uncertainty: float,
    synth_penalty: float,
) -> dict[str, float]:
    combo_weight = 1.2 if bool(combo_dense_case) and robust_combo_core is not None else 0.0
    reward = float(
        0.8 * float(s_wt)
        + 1.0 * float(robust_site_core)
        + combo_weight * float(robust_combo_core or 0.0)
        + 0.5 * float(alt_anchor_score_value)
        + 0.35 * float(new_nonhotspot_score)
        - 0.25 * float(hotspot_fraction)
        - 0.25 * float(oracle_uncertainty)
        - 0.25 * float(synth_penalty)
    )
    return {
        "reward_v2_wt_score": float(s_wt),
        "reward_v2_robust_site_core": float(robust_site_core),
        "reward_v2_robust_combo_core": float(robust_combo_core or 0.0),
        "reward_v2_alt_anchor_score": float(alt_anchor_score_value),
        "reward_v2_new_nonhotspot_score": float(new_nonhotspot_score),
        "reward_v2_hotspot_fraction": float(hotspot_fraction),
        "reward_v2_oracle_uncertainty": float(oracle_uncertainty),
        "reward_v2_synth_penalty": float(synth_penalty),
        "reward_v2_combo_weight": float(combo_weight),
        "reward_v2_raw": float(reward),
    }
