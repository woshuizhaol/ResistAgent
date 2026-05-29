#!/usr/bin/env python3
"""mutation-proposal step mutation proposal reasoning layer."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from agents.glm_client import GLMCallRecord, GLMClient


def _native_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(float(value), 6)
    if hasattr(value, "item"):
        return _native_value(value.item())
    return value


def _records(frame: pd.DataFrame, columns: list[str], limit: int) -> list[dict[str, Any]]:
    if frame.empty or limit <= 0:
        return []
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit).copy()
    return [
        {column: _native_value(value) for column, value in row.items()}
        for row in subset.to_dict(orient="records")
    ]


@dataclass
class MutationProposalAgentConfig:
    temperature: float = 0.2
    max_tokens: int = 2200
    thinking: dict[str, str] | None = None
    top_site_n: int = 8
    top_combo_n: int = 5
    top_residue_n: int = 6


class MutationProposalAgent:
    """Calls GLM to translate deterministic mutation-proposal step rankings into structured rationale."""

    def __init__(self, client: GLMClient | None = None, config: MutationProposalAgentConfig | None = None) -> None:
        self.client = client or GLMClient()
        self.config = config or MutationProposalAgentConfig(thinking={"type": "enabled"})
        if self.config.thinking is None:
            self.config.thinking = {"type": "enabled"}

    @staticmethod
    def output_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "required": [
                "executive_summary",
                "site_interpretations",
                "combo_interpretations",
                "residue_watchlist",
                "global_caveats",
            ],
            "properties": {
                "executive_summary": {
                    "type": "object",
                    "required": [
                        "overall_risk_pattern",
                        "ranking_logic",
                        "coverage_note",
                        "principal_caveat",
                    ],
                    "properties": {
                        "overall_risk_pattern": {"type": "string"},
                        "ranking_logic": {"type": "string"},
                        "coverage_note": {"type": "string"},
                        "principal_caveat": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                "site_interpretations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "mutation_key",
                            "target_position",
                            "mechanism_hypothesis",
                            "reasoning",
                            "supporting_signals",
                            "confidence",
                            "caveat",
                        ],
                        "properties": {
                            "mutation_key": {"type": "string"},
                            "target_position": {"type": ["integer", "null"]},
                            "mechanism_hypothesis": {"type": "string"},
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
                "combo_interpretations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "combination_key",
                            "mechanism_hypothesis",
                            "reasoning",
                            "supporting_signals",
                            "confidence",
                            "caveat",
                        ],
                        "properties": {
                            "combination_key": {"type": "string"},
                            "mechanism_hypothesis": {"type": "string"},
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
                "residue_watchlist": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "target_position",
                            "best_mutation_key",
                            "why_it_matters",
                        ],
                        "properties": {
                            "target_position": {"type": "integer"},
                            "best_mutation_key": {"type": "string"},
                            "why_it_matters": {"type": "string"},
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
        site_rank: pd.DataFrame,
        residue_risk: pd.DataFrame,
        combo_rank: pd.DataFrame,
        qc_payload: dict[str, Any],
        structural_context_summary: dict[str, Any],
    ) -> dict[str, Any]:
        site_columns = [
            "mutation_rank",
            "mutation_key",
            "target_position",
            "risk_score",
            "P_appearance",
            "impact_score",
            "FitnessWeight",
            "structure_layer",
            "impact_evidence_tier",
            "proxy_status",
            "known_hotspot",
            "evaluation_hotspot_label",
            "functional_site_flag",
            "delta_dock_proxy",
            "delta_ifp_proxy",
            "local_backbone_rmsd_a",
            "position_entropy_mean",
            "conservation_source",
            "uses_rule_based_fallback",
            "scope_multiplier",
            "evidence_sources",
        ]
        residue_columns = [
            "target_position",
            "best_mutation_key",
            "max_risk_score",
            "mean_risk_score",
            "candidate_count",
            "structure_layer",
            "known_hotspot_any",
            "functional_site_flag",
            "min_ligand_distance_a",
            "mean_position_entropy",
            "conservation_source",
            "solvent_accessibility",
            "secondary_structure_context",
        ]
        combo_columns = [
            "combo_rank",
            "combination_key",
            "risk_score",
            "P_appearance_combo",
            "impact_score",
            "FitnessWeight",
            "structure_layer",
            "impact_evidence_tier",
            "proxy_status",
            "delta_dock_proxy",
            "delta_ifp_proxy",
            "position_entropy_mean",
            "conservation_source",
            "evidence_sources",
        ]
        return {
            "case_meta": {
                "case_id": str(case_entry["case_id"]),
                "target_name": str(case_entry["target_name"]),
                "drug_name": str(case_entry["drug_name"]),
                "target_domain": str(case_entry.get("target_domain") or ""),
                "evaluation_unit": str(case_entry.get("evaluation_unit") or ""),
                "tissue_type": str(case_entry.get("tissue_type") or ""),
                "is_viral": bool(str(case_entry.get("target_domain") or "").lower() == "rt" or site_rank.get("domain_type", pd.Series(dtype=str)).fillna("").eq("viral").any()),
            },
            "qc_summary": {
                "site_candidate_count": int(qc_payload.get("site_candidate_count", 0)),
                "site_structure_backed_count": int(qc_payload.get("site_structure_backed_count", 0)),
                "site_fallback_count": int(qc_payload.get("site_fallback_count", 0)),
                "site_in_scope_candidate_count": int(qc_payload.get("site_in_scope_candidate_count", 0)),
                "site_rule_based_fraction": _native_value(qc_payload.get("site_rule_based_fraction")),
                "site_proxy_job_count": int(qc_payload.get("site_proxy_job_count", 0)),
                "site_docking_failure_rate": _native_value(qc_payload.get("site_docking_failure_rate")),
                "site_top20_hotspot_recall": _native_value(qc_payload.get("site_top20_hotspot_recall")),
                "site_ndcg_risk": _native_value(qc_payload.get("site_ndcg_risk")),
                "site_ndcg_frequency": _native_value(qc_payload.get("site_ndcg_frequency")),
                "site_ndcg_background_only": _native_value(qc_payload.get("site_ndcg_background_only")),
                "site_ndcg_random_mean": _native_value(qc_payload.get("site_ndcg_random_mean")),
                "combo_candidate_count": int(qc_payload.get("combo_candidate_count", 0)),
                "site_combo_split_ok": bool(qc_payload.get("site_combo_split_ok", False)),
                "viral_no_cosmic_depmap_leakage": bool(qc_payload.get("viral_no_cosmic_depmap_leakage", True)),
                "combo_projection_leakage_count": int(qc_payload.get("combo_projection_leakage_count", 0)),
                "site_proxy_total_retries": int(qc_payload.get("site_proxy_total_retries", 0)),
                "fitness_conservation_primary_source": str(qc_payload.get("fitness_conservation_primary_source") or ""),
            },
            "structural_context_summary": structural_context_summary,
            "site_candidates": _records(site_rank, site_columns, self.config.top_site_n),
            "residue_candidates": _records(residue_risk, residue_columns, self.config.top_residue_n),
            "combo_candidates": _records(combo_rank, combo_columns, self.config.top_combo_n),
        }

    def run(
        self,
        *,
        case_entry: dict[str, Any],
        site_rank: pd.DataFrame,
        residue_risk: pd.DataFrame,
        combo_rank: pd.DataFrame,
        qc_payload: dict[str, Any],
        structural_context_summary: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], GLMCallRecord]:
        prompt_input = self.build_prompt_input(
            case_entry=case_entry,
            site_rank=site_rank,
            residue_risk=residue_risk,
            combo_rank=combo_rank,
            qc_payload=qc_payload,
            structural_context_summary=structural_context_summary,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are ResistAgent Mutation Proposal Agent for mutation-proposal step. "
                    "You only translate deterministic ranking outputs into scientific rationale. "
                    "Do not invent mutations, combos, residues, scores, sources, or physical calculations. "
                    "Do not alter ranking order or recompute risk. "
                    "Treat combo_rank as a separate table from single-site ranking. "
                    "If evidence tier is fallback, say so explicitly. "
                    "For viral branches, do not output the strings COSMIC or DepMap at all, even in negative statements. "
                    "Return JSON only and match the schema exactly."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize the mutation-proposal step mutation ranking for this case. "
                    "Use only identifiers and evidence present in the payload. "
                    "Prioritize: top site candidates, separate combo interpretations, residue watchlist, and caveats.\n\n"
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
        self._validate_response(payload, prompt_input)
        return prompt_input, payload, record

    def _validate_response(self, payload: dict[str, Any], prompt_input: dict[str, Any]) -> None:
        site_ids = {str(row["mutation_key"]) for row in prompt_input["site_candidates"]}
        combo_ids = {str(row["combination_key"]) for row in prompt_input["combo_candidates"]}
        residue_ids = {int(row["target_position"]) for row in prompt_input["residue_candidates"] if row.get("target_position") is not None}

        for row in payload.get("site_interpretations", []):
            mutation_key = str(row.get("mutation_key") or "")
            if mutation_key not in site_ids:
                raise ValueError(f"MutationProposalAgent returned unknown mutation_key: {mutation_key}")
        for row in payload.get("combo_interpretations", []):
            combination_key = str(row.get("combination_key") or "")
            if combination_key not in combo_ids:
                raise ValueError(f"MutationProposalAgent returned unknown combination_key: {combination_key}")
        for row in payload.get("residue_watchlist", []):
            position = row.get("target_position")
            if position is None or int(position) not in residue_ids:
                raise ValueError(f"MutationProposalAgent returned unknown residue position: {position}")

        if not combo_ids and payload.get("combo_interpretations"):
            raise ValueError("MutationProposalAgent returned combo_interpretations for a case without combo candidates.")
        if prompt_input["case_meta"]["is_viral"]:
            rendered = json.dumps(payload, ensure_ascii=True).lower()
            if "cosmic" in rendered or "depmap" in rendered:
                raise ValueError("MutationProposalAgent leaked COSMIC/DepMap terms into a viral case output.")

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
