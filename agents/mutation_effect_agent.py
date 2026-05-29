#!/usr/bin/env python3
"""mutation-effect step mutation effect interpretation agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from agents.glm_client import GLMCallRecord, GLMClient
from tools.stage5_utils import native_value, parse_json_list


def _records(frame: pd.DataFrame, columns: list[str], limit: int) -> list[dict[str, Any]]:
    if frame.empty or limit <= 0:
        return []
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit).copy()
    return [
        {column: native_value(value) for column, value in row.items()}
        for row in subset.to_dict(orient="records")
    ]


def _canonical_effect_key(value: str, allowed_keys: set[str]) -> str:
    text = str(value or "").strip()
    if text in allowed_keys:
        return text
    suffix_matches = [key for key in allowed_keys if key.split(":", 1)[-1] == text]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return text


@dataclass
class MutationEffectAgentConfig:
    temperature: float = 0.2
    max_tokens: int = 2400
    thinking: dict[str, str] | None = None
    top_site_n: int = 6
    top_combo_n: int = 5


class MutationEffectAgent:
    """Translate deterministic mutation-effect step evidence into structured mechanism narratives."""

    def __init__(self, client: GLMClient | None = None, config: MutationEffectAgentConfig | None = None) -> None:
        self.client = client or GLMClient()
        self.config = config or MutationEffectAgentConfig(thinking={"type": "enabled"})
        if self.config.thinking is None:
            self.config.thinking = {"type": "enabled"}

    @staticmethod
    def output_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "required": [
                "executive_summary",
                "site_effects",
                "combo_effects",
                "global_caveats",
            ],
            "properties": {
                "executive_summary": {
                    "type": "object",
                    "required": [
                        "overall_effect_pattern",
                        "calibration_note",
                        "principal_uncertainty",
                    ],
                    "properties": {
                        "overall_effect_pattern": {"type": "string"},
                        "calibration_note": {"type": "string"},
                        "principal_uncertainty": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                "site_effects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "mutation_key",
                            "mechanism_labels",
                            "reasoning",
                            "supporting_signals",
                            "confidence",
                            "caveat",
                        ],
                        "properties": {
                            "mutation_key": {"type": "string"},
                            "mechanism_labels": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "steric_clash",
                                        "anchor_loss",
                                        "electrostatic_shift",
                                        "pocket_rearrangement",
                                    ],
                                },
                            },
                            "reasoning": {"type": "string"},
                            "supporting_signals": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                            "caveat": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                },
                "combo_effects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "combination_key",
                            "mechanism_labels",
                            "epistasis_flag",
                            "reasoning",
                            "supporting_signals",
                            "confidence",
                            "caveat",
                        ],
                        "properties": {
                            "combination_key": {"type": "string"},
                            "mechanism_labels": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "steric_clash",
                                        "anchor_loss",
                                        "electrostatic_shift",
                                        "pocket_rearrangement",
                                    ],
                                },
                            },
                            "epistasis_flag": {
                                "type": "string",
                                "enum": ["additive_like", "non_additive", "unresolved"],
                            },
                            "reasoning": {"type": "string"},
                            "supporting_signals": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                            "caveat": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                },
                "global_caveats": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "additionalProperties": False,
        }

    def build_prompt_input(
        self,
        *,
        case_entry: dict[str, Any],
        site_effects: pd.DataFrame,
        combo_effects: pd.DataFrame,
        qc_payload: dict[str, Any],
        calibration_summary: dict[str, Any],
    ) -> dict[str, Any]:
        site_columns = [
            "stage4_rank",
            "target_key",
            "impact_evidence_tier",
            "delta_dock_kcal_mol",
            "delta_mmgbsa_binding_kcal_mol",
            "delta_gnina_affinity_kcal_mol",
            "ifp_jaccard_loss",
            "ifp_occupancy_shift_mean_abs",
            "ifp_occupancy_anchor_loss",
            "anchor_loss_fraction",
            "stage4_local_rmsd_a",
            "pocket_volume_change_fraction",
            "solvent_proxy_shift",
            "consensus_fraction",
            "consensus_direction",
            "high_uncertainty",
            "mechanism_labels_json",
            "lost_anchor_labels_json",
            "hydrogen_bond_delta_count",
            "salt_bridge_delta_count",
            "sample_source",
        ]
        combo_columns = [
            "stage4_rank",
            "target_key",
            "impact_evidence_tier",
            "delta_dock_kcal_mol",
            "delta_mmgbsa_binding_kcal_mol",
            "delta_gnina_affinity_kcal_mol",
            "ifp_jaccard_loss",
            "ifp_occupancy_shift_mean_abs",
            "ifp_occupancy_anchor_loss",
            "anchor_loss_fraction",
            "stage4_local_rmsd_a",
            "pocket_volume_change_fraction",
            "solvent_proxy_shift",
            "consensus_fraction",
            "consensus_direction",
            "high_uncertainty",
            "mechanism_labels_json",
            "lost_anchor_labels_json",
            "epistasis_flag",
            "sample_source",
            "used_synthetic_combo_model",
        ]
        site_candidates = _records(site_effects, site_columns, self.config.top_site_n)
        combo_candidates = _records(combo_effects, combo_columns, self.config.top_combo_n)
        for row in site_candidates:
            row["mutation_key"] = str(row.pop("target_key"))
        for row in combo_candidates:
            row["combination_key"] = str(row.pop("target_key"))
        return {
            "case_meta": {
                "case_id": str(case_entry["case_id"]),
                "target_name": str(case_entry["target_name"]),
                "drug_name": str(case_entry["drug_name"]),
                "target_domain": str(case_entry.get("target_domain") or ""),
                "is_viral": bool(str(case_entry.get("target_domain") or "").lower() == "rt"),
            },
            "qc_summary": {
                "selected_site_targets": int(qc_payload.get("selected_site_targets", 0)),
                "selected_combo_targets": int(qc_payload.get("selected_combo_targets", 0)),
                "docking_success_count": int(qc_payload.get("docking_success_count", 0)),
                "docking_failure_count": int(qc_payload.get("docking_failure_count", 0)),
                "ifp_success_count": int(qc_payload.get("ifp_success_count", 0)),
                "skipped_unready_count": int(qc_payload.get("skipped_unready_count", 0)),
            },
            "calibration_summary": {
                "calibration_sample_count": int(calibration_summary.get("calibration_sample_count", 0)),
                "dock_vs_mmgbsa_pearson_r": native_value(calibration_summary.get("dock_vs_mmgbsa_pearson_r")),
                "dock_vs_mmgbsa_spearman_r": native_value(calibration_summary.get("dock_vs_mmgbsa_spearman_r")),
                "predicted_vs_mmgbsa_pearson_r": native_value(calibration_summary.get("predicted_vs_mmgbsa_pearson_r")),
                "predicted_vs_mmgbsa_spearman_r": native_value(calibration_summary.get("predicted_vs_mmgbsa_spearman_r")),
                "experimental_alignment_pearson_r": native_value(calibration_summary.get("experimental_alignment_pearson_r")),
                "experimental_alignment_spearman_r": native_value(calibration_summary.get("experimental_alignment_spearman_r")),
            },
            "site_effect_candidates": site_candidates,
            "combo_effect_candidates": combo_candidates,
        }

    def run(
        self,
        *,
        case_entry: dict[str, Any],
        site_effects: pd.DataFrame,
        combo_effects: pd.DataFrame,
        qc_payload: dict[str, Any],
        calibration_summary: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], GLMCallRecord]:
        prompt_input = self.build_prompt_input(
            case_entry=case_entry,
            site_effects=site_effects,
            combo_effects=combo_effects,
            qc_payload=qc_payload,
            calibration_summary=calibration_summary,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are ResistAgent Mutation Effect Agent for mutation-effect step. "
                    "You only translate deterministic docking and interaction evidence into mechanism rationale. "
                    "Do not invent labels, residues, scores, calibration values, or epistasis outcomes. "
                    "Use only mechanism labels already present in the payload. "
                    "If evidence is weak or synthetic-only, say so explicitly. "
                    "For viral cases, do not output the strings COSMIC or DepMap. "
                    "Return JSON only and match the schema exactly."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize the mutation-effect step mutation-effect evidence for this case. "
                    "Explain the provided mechanism labels, cite the deterministic signals that support them, "
                    "and keep combo epistasis separate from site effects.\n\n"
                    f"{json.dumps(prompt_input, indent=2, sort_keys=True, ensure_ascii=True)}"
                ),
            },
        ]
        payload, record = self.client.chat_json(
            messages,
            self.output_schema(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            thinking=self.config.thinking,
        )
        if prompt_input["case_meta"]["is_viral"]:
            payload = self._sanitize_viral_payload(payload)
        payload = self._canonicalize_payload_keys(payload, prompt_input)
        payload = self._constrain_payload_to_deterministic_values(payload, prompt_input)
        self._validate_response(payload, prompt_input)
        return prompt_input, payload, record

    def _validate_response(self, payload: dict[str, Any], prompt_input: dict[str, Any]) -> None:
        site_lookup = {
            str(row["mutation_key"]): set(parse_json_list(row.get("mechanism_labels_json")))
            for row in prompt_input["site_effect_candidates"]
        }
        combo_lookup = {
            str(row["combination_key"]): {
                "labels": set(parse_json_list(row.get("mechanism_labels_json"))),
                "epistasis_flag": str(row.get("epistasis_flag") or "unresolved"),
            }
            for row in prompt_input["combo_effect_candidates"]
        }
        for row in payload.get("site_effects", []):
            mutation_key = str(row.get("mutation_key") or "")
            if mutation_key not in site_lookup:
                raise ValueError(f"MutationEffectAgent returned unknown mutation_key: {mutation_key}")
            returned_labels = set(str(value) for value in row.get("mechanism_labels", []))
            if not returned_labels.issubset(site_lookup[mutation_key]):
                raise ValueError(f"MutationEffectAgent returned unsupported mechanism label(s) for {mutation_key}")
        for row in payload.get("combo_effects", []):
            combination_key = str(row.get("combination_key") or "")
            if combination_key not in combo_lookup:
                raise ValueError(f"MutationEffectAgent returned unknown combination_key: {combination_key}")
            returned_labels = set(str(value) for value in row.get("mechanism_labels", []))
            if not returned_labels.issubset(combo_lookup[combination_key]["labels"]):
                raise ValueError(f"MutationEffectAgent returned unsupported mechanism label(s) for {combination_key}")
            if str(row.get("epistasis_flag") or "") != combo_lookup[combination_key]["epistasis_flag"]:
                raise ValueError(f"MutationEffectAgent changed deterministic epistasis flag for {combination_key}")
        if prompt_input["case_meta"]["is_viral"]:
            rendered = json.dumps(payload, ensure_ascii=True).lower()
            if "cosmic" in rendered or "depmap" in rendered:
                raise ValueError("MutationEffectAgent leaked COSMIC/DepMap terms into a viral case output.")

    def _canonicalize_payload_keys(self, payload: dict[str, Any], prompt_input: dict[str, Any]) -> dict[str, Any]:
        site_keys = {str(row["mutation_key"]) for row in prompt_input["site_effect_candidates"]}
        combo_keys = {str(row["combination_key"]) for row in prompt_input["combo_effect_candidates"]}
        for row in payload.get("site_effects", []):
            row["mutation_key"] = _canonical_effect_key(str(row.get("mutation_key") or ""), site_keys)
        for row in payload.get("combo_effects", []):
            row["combination_key"] = _canonical_effect_key(str(row.get("combination_key") or ""), combo_keys)
        return payload

    def _constrain_payload_to_deterministic_values(self, payload: dict[str, Any], prompt_input: dict[str, Any]) -> dict[str, Any]:
        site_lookup = {
            str(row["mutation_key"]): list(dict.fromkeys(str(label) for label in parse_json_list(row.get("mechanism_labels_json"))))
            for row in prompt_input["site_effect_candidates"]
        }
        combo_lookup = {
            str(row["combination_key"]): {
                "labels": list(
                    dict.fromkeys(str(label) for label in parse_json_list(row.get("mechanism_labels_json")))
                ),
                "epistasis_flag": str(row.get("epistasis_flag") or "unresolved"),
            }
            for row in prompt_input["combo_effect_candidates"]
        }
        for row in payload.get("site_effects", []):
            allowed_labels = site_lookup.get(str(row.get("mutation_key") or ""), [])
            returned_labels = [str(label) for label in row.get("mechanism_labels", [])]
            constrained_labels = [label for label in returned_labels if label in allowed_labels]
            row["mechanism_labels"] = constrained_labels or allowed_labels
        for row in payload.get("combo_effects", []):
            combo_payload = combo_lookup.get(str(row.get("combination_key") or ""), {})
            allowed_labels = list(combo_payload.get("labels", []))
            returned_labels = [str(label) for label in row.get("mechanism_labels", [])]
            constrained_labels = [label for label in returned_labels if label in allowed_labels]
            row["mechanism_labels"] = constrained_labels or allowed_labels
            if combo_payload:
                row["epistasis_flag"] = str(combo_payload["epistasis_flag"])
        return payload

    def _sanitize_viral_payload(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            return {key: self._sanitize_viral_payload(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [self._sanitize_viral_payload(value) for value in payload]
        if isinstance(payload, str):
            text = re.sub(r"cosmic\s*/\s*depmap|depmap\s*/\s*cosmic", "tumor priors", payload, flags=re.IGNORECASE)
            text = re.sub(r"cosmic", "tumor prior", text, flags=re.IGNORECASE)
            text = re.sub(r"depmap", "tumor prior", text, flags=re.IGNORECASE)
            return text
        return payload
