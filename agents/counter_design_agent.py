#!/usr/bin/env python3
"""counter-design step counter-design planning agent."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from agents.glm_client import GLMCallRecord, GLMClient

SUPPORTED_EDIT_FAMILIES = (
    "ADD",
    "REPLACE",
    "DELETE",
    "HETERO_SWAP",
    "RING_EDIT",
    "RIGIDIFY_LINKER",
    "VECTOR_FLIP",
    "BIOISOSTERE_SWAP",
    "RING_EXPANSION",
    "RING_CONTRACTION",
    "HETEROARYL_SWAP",
    "BACKBONE_SEEKER",
    "WATER_BRIDGE_PROBE",
    "HALOGEN_SCAN",
    "SMALL_POLAR_SCAN",
    "LINKER_HETERO_SCAN",
    "N_METHYL_SCAN",
    "CONSTRAINED_TAIL_TRIM",
)
SUPPORTED_PATTERNS = (
    "add_polar_cap",
    "add_hydrophobic_cap",
    "add_hbond_acceptor",
    "replace_leaf_with_polar",
    "replace_leaf_with_hydrophobe",
    "trim_flexible_tail",
    "trim_steric_bulk",
    "hetero_swap_linker",
    "aza_scan_ring",
    "ring_hetero_tune",
    "rigidify_linker",
    "vector_flip_adjacent",
    "bioisostere_swap",
    "ring_expansion",
    "ring_contraction",
    "heteroaryl_swap",
    "backbone_seeker",
    "water_bridge_probe",
    "halogen_scan",
    "small_polar_scan",
    "linker_hetero_scan",
    "n_methyl_scan",
    "constrained_tail_trim",
)


def _native_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(float(value), 6)
    if hasattr(value, "item"):
        return _native_value(value.item())
    if isinstance(value, list):
        return [_native_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _native_value(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    return value


def _beam_records(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if frame.empty or limit <= 0:
        return []
    columns = [
        "candidate_id",
        "smiles",
        "round_index",
        "objective_name",
        "objective_reward",
        "robust_score",
        "naive_mean_affinity",
        "keep_ifp",
        "dep",
        "compensation_gain",
        "wt_hard_constraint_pass",
        "prefilter_pass",
        "mechanism_risk_focus",
        "action_sequence_json",
    ]
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit).copy()
    return [{column: _native_value(value) for column, value in row.items()} for row in subset.to_dict(orient="records")]


def _parse_target_scores(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        payload = value
    else:
        try:
            payload = json.loads(str(value))
        except Exception:
            return []
    if not isinstance(payload, dict):
        return []
    return [dict(item) for item in payload.values() if isinstance(item, dict)]


def _worst_targets(frame: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    if frame.empty or "target_scores_json" not in frame.columns:
        return []
    focus_row = frame.sort_values("objective_reward", ascending=False).iloc[0]
    rows = _parse_target_scores(focus_row.get("target_scores_json"))
    rows.sort(
        key=lambda row: (
            str(row.get("docking_status") or "") == "ok",
            float(row.get("score") if row.get("score") is not None else -1.0),
            float(row.get("keep_ifp") if row.get("keep_ifp") is not None else -1.0),
        )
    )
    output: list[dict[str, Any]] = []
    for row in rows[:limit]:
        output.append(
            {
                "target_key": _native_value(row.get("target_key")),
                "effect_scope": _native_value(row.get("effect_scope")),
                "score": _native_value(row.get("score")),
                "keep_ifp": _native_value(row.get("keep_ifp")),
                "lost_anchor_labels": _native_value(row.get("lost_anchor_labels")),
                "mechanism_labels": _native_value(row.get("mechanism_labels")),
                "target_uncertain": _native_value(row.get("target_uncertain")),
                "is_partner_chain_sensitive": _native_value(row.get("is_partner_chain_sensitive")),
            }
        )
    return output


def _constraint_breakdown(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {
            "wt_fail_count": 0,
            "coverage_fail_count": 0,
            "keep_ifp_fail_count": 0,
            "compensation_fail_count": 0,
        }
    return {
        "wt_fail_count": int(frame.get("wt_hard_constraint_pass", pd.Series(dtype=bool)).fillna(False).eq(False).sum()),
        "coverage_fail_count": int(frame.get("coverage_pass", pd.Series(dtype=bool)).fillna(False).eq(False).sum()),
        "keep_ifp_fail_count": int(frame.get("keep_ifp_constraint_pass", pd.Series(dtype=bool)).fillna(False).eq(False).sum()),
        "compensation_fail_count": int(frame.get("compensation_constraint_pass", pd.Series(dtype=bool)).fillna(False).eq(False).sum()),
    }


@dataclass
class CounterDesignAgentConfig:
    temperature: float = 0.7
    max_tokens: int = 2600
    thinking: dict[str, str] | None = None
    proposal_count: int = 32


class CounterDesignAgent:
    """Prioritizes supported medicinal-chemistry edit templates for counter-design step search."""

    def __init__(self, client: GLMClient | None = None, config: CounterDesignAgentConfig | None = None) -> None:
        self.client = client or GLMClient()
        self.config = config or CounterDesignAgentConfig(thinking={"type": "enabled"})
        if self.config.thinking is None:
            self.config.thinking = {"type": "enabled"}

    @staticmethod
    def output_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["round_strategy", "exploration_mode", "action_templates", "global_caveats"],
            "properties": {
                "round_strategy": {"type": "string"},
                "exploration_mode": {"type": "string", "enum": ["focused", "balanced", "exploratory"]},
                "action_templates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["edit_family", "pattern", "fragment", "priority", "mechanistic_goal", "rationale"],
                        "properties": {
                            "edit_family": {"type": "string", "enum": list(SUPPORTED_EDIT_FAMILIES)},
                            "pattern": {"type": "string", "enum": list(SUPPORTED_PATTERNS)},
                            "fragment": {"type": "string"},
                            "priority": {"type": "integer", "minimum": 1},
                            "mechanistic_goal": {"type": "string"},
                            "rationale": {"type": "string"},
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
        objective_name: str,
        round_index: int,
        mechanism_summary: dict[str, Any],
        hotspot_residues: list[str],
        beam_frame: pd.DataFrame,
        action_space: dict[str, Any],
        qc_payload: dict[str, Any],
        lead_descriptors: dict[str, Any],
    ) -> dict[str, Any]:
        beam_focus = {}
        worst_targets = _worst_targets(beam_frame, limit=5)
        constraint_breakdown = _constraint_breakdown(beam_frame)
        if not beam_frame.empty:
            focus_row = beam_frame.sort_values("objective_reward", ascending=False).iloc[0].to_dict()
            beam_focus = {
                "candidate_id": _native_value(focus_row.get("candidate_id")),
                "objective_reward": _native_value(focus_row.get("objective_reward")),
                "robust_score": _native_value(focus_row.get("robust_score")),
                "keep_ifp": _native_value(focus_row.get("keep_ifp")),
                "dep": _native_value(focus_row.get("dep")),
                "panel_coverage": _native_value(focus_row.get("panel_coverage")),
                "new_nonhotspot_contact_score": _native_value(focus_row.get("new_nonhotspot_contact_score")),
                "hotspot_fraction": _native_value(focus_row.get("hotspot_fraction")),
                "mechanism_risk_focus": _native_value(focus_row.get("mechanism_risk_focus")),
            }
        failure_mechanism_counts: dict[str, int] = {}
        for target in worst_targets:
            for label in list(target.get("mechanism_labels") or []):
                label_text = str(label)
                failure_mechanism_counts[label_text] = int(failure_mechanism_counts.get(label_text, 0) + 1)
        failure_context = {
            "focus_candidate_id": _native_value(beam_focus.get("candidate_id")),
            "worst_targets": worst_targets,
            "worst_site_targets": [row for row in worst_targets if str(row.get("effect_scope") or "") == "site"],
            "worst_combo_targets": [row for row in worst_targets if str(row.get("effect_scope") or "") == "combo"],
            "mechanism_counts": failure_mechanism_counts,
            "partner_chain_sensitive_count": int(sum(bool(row.get("is_partner_chain_sensitive")) for row in worst_targets)),
            "uncertain_target_count": int(sum(bool(row.get("target_uncertain")) for row in worst_targets)),
        }
        pocket_context = {
            "partner_chain_residues": _native_value(action_space.get("partner_chain_residues")),
            "partner_chain_positions": _native_value(action_space.get("partner_chain_positions")),
            "pocket_profile": _native_value(action_space.get("pocket_profile")),
            "case_specific_action_hints": _native_value(action_space.get("case_specific_action_hints")),
        }
        return {
            "case_meta": {
                "case_id": str(case_entry["case_id"]),
                "target_name": str(case_entry["target_name"]),
                "drug_name": str(case_entry["drug_name"]),
                "target_domain": str(case_entry.get("target_domain") or ""),
                "evaluation_unit": str(case_entry.get("evaluation_unit") or ""),
                "objective_name": str(objective_name),
                "is_viral": bool(str(case_entry.get("target_domain") or "").lower() == "rt"),
            },
            "search_state": {
                "round_index": int(round_index),
                "proposal_count": int(self.config.proposal_count),
                "beam_size": int(min(len(beam_frame), 8)),
                "current_beam": _beam_records(beam_frame, limit=8),
                "beam_focus": beam_focus,
            },
            "mechanism_summary": _native_value(mechanism_summary),
            "hotspot_residues": [str(value) for value in hotspot_residues],
            "worst_targets": worst_targets,
            "constraint_breakdown": constraint_breakdown,
            "failure_context": _native_value(failure_context),
            "pocket_context": _native_value(pocket_context),
            "new_nonhotspot_contact_score": _native_value(beam_focus.get("new_nonhotspot_contact_score")),
            "hotspot_fraction": _native_value(beam_focus.get("hotspot_fraction")),
            "lead_profile": _native_value(lead_descriptors),
            "action_space": _native_value(action_space),
            "qc_summary": _native_value(qc_payload),
        }

    def _fallback_payload(
        self,
        *,
        objective_name: str,
        mechanism_summary: dict[str, Any],
        action_space: dict[str, Any],
        lead_descriptors: dict[str, Any],
        prompt_input: dict[str, Any],
    ) -> dict[str, Any]:
        mechanism_counts = dict(mechanism_summary.get("mechanism_label_counts") or {})
        ranked: list[tuple[str, str, str, str]] = []
        lead_qed = float(lead_descriptors.get("qed") or 0.0)
        lead_sa = float(lead_descriptors.get("sa_score") or 0.0)
        lead_clogp = float(lead_descriptors.get("clogp") or 0.0)
        lead_rotatable = int(lead_descriptors.get("rotatable_bonds") or 0)
        failure_context = dict(prompt_input.get("failure_context") or {})
        pocket_context = dict(prompt_input.get("pocket_context") or {})
        worst_targets = [dict(row) for row in list(failure_context.get("worst_targets") or []) if isinstance(row, dict)]
        if bool(failure_context.get("partner_chain_sensitive_count", 0)):
            ranked.extend(
                [
                    ("SMALL_POLAR_SCAN", "small_polar_scan", "OC", "Partner-chain-sensitive failures need peripheral polar reach."),
                    ("LINKER_HETERO_SCAN", "linker_hetero_scan", "N", "Retune linker polarity for partner-chain contact recovery."),
                    ("RING_EDIT", "aza_scan_ring", "N", "Use aza scans to shift vectoring away from brittle partner-chain loss."),
                ]
            )
        if any("steric_clash" in list(row.get("mechanism_labels") or []) for row in worst_targets):
            ranked.append(("CONSTRAINED_TAIL_TRIM", "constrained_tail_trim", "", "Worst targets indicate steric clash; trim before expanding."))
        if any("electrostatic_shift" in list(row.get("mechanism_labels") or []) for row in worst_targets):
            ranked.append(("LINKER_HETERO_SCAN", "linker_hetero_scan", "N", "Worst targets indicate electrostatic shift; retune local hetero pattern."))
        for hint in list(pocket_context.get("case_specific_action_hints") or []):
            if not isinstance(hint, dict):
                continue
            ranked.append(
                (
                    str(hint.get("edit_family") or ""),
                    str(hint.get("pattern") or ""),
                    str(hint.get("fragment") or ""),
                    str(hint.get("rationale") or "Case-specific pocket prior."),
                )
            )
        if lead_qed < 0.35 or lead_clogp > 5.5 or lead_sa > 6.0:
            ranked.extend(
                [
                    ("DELETE", "trim_steric_bulk", "", "Reduce lipophilic or bulky peripheral groups before adding new bulk."),
                    ("REPLACE", "replace_leaf_with_polar", "N", "Swap terminal leaves to more polar alternatives."),
                    ("REPLACE", "replace_leaf_with_polar", "OC", "Recover polarity without overextending the scaffold."),
                    ("RING_EDIT", "aza_scan_ring", "N", "Use aza-scans to reduce hotspot dependence and cLogP."),
                ]
            )
        if lead_rotatable > 8:
            ranked.append(("DELETE", "trim_flexible_tail", "", "Trim flexible tails before pursuing new affinity gains."))
        if int(mechanism_counts.get("anchor_loss", 0)) > 0:
            ranked.extend(
                [
                    ("ADD", "add_hbond_acceptor", "OC", "Recover anchor-compatible polar contacts."),
                    ("REPLACE", "replace_leaf_with_polar", "N", "Replace terminal groups with polar alternatives."),
                ]
            )
        if int(mechanism_counts.get("steric_clash", 0)) > 0:
            ranked.extend(
                [
                    ("CONSTRAINED_TAIL_TRIM", "constrained_tail_trim", "", "Reduce flexible tail bulk before expanding into strained pockets."),
                    ("DELETE", "trim_steric_bulk", "", "Reduce peripheral steric bulk."),
                    ("HALOGEN_SCAN", "halogen_scan", "F", "Swap bulky leaves for compact halogen probes."),
                ]
            )
        if int(mechanism_counts.get("electrostatic_shift", 0)) > 0:
            ranked.extend(
                [
                    ("LINKER_HETERO_SCAN", "linker_hetero_scan", "N", "Tune local polarity through linker hetero edits."),
                    ("SMALL_POLAR_SCAN", "small_polar_scan", "O", "Introduce compact polar atoms near the periphery."),
                ]
            )
        ranked.extend(
            [
                ("RING_EDIT", "aza_scan_ring", "N", "Reduce hotspot dependence with ring aza-scans."),
                ("SMALL_POLAR_SCAN", "small_polar_scan", "O", "Probe compact polar compensation around the lead periphery."),
                ("HALOGEN_SCAN", "halogen_scan", "F", "Probe shallow hydrophobic compensation."),
            ]
        )
        if objective_name == "naive" and not (lead_qed < 0.35 or lead_clogp > 5.5):
            ranked.insert(0, ("HALOGEN_SCAN", "halogen_scan", "Cl", "Prioritize average affinity gains with compact hydrophobes."))

        allowed_fragments = {str(value) for value in action_space.get("fragments", [])}
        payload_rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        priority = 1
        for edit_family, pattern, fragment, rationale in ranked:
            fragment_value = (
                fragment
                if fragment in allowed_fragments or fragment in {"", "-C"}
                else next(iter(sorted(allowed_fragments)), "")
            )
            signature = (edit_family, pattern, fragment_value)
            if signature in seen:
                continue
            seen.add(signature)
            payload_rows.append(
                {
                    "edit_family": edit_family,
                    "pattern": pattern,
                    "fragment": fragment_value,
                    "priority": priority,
                    "mechanistic_goal": "robust_anchor_preservation" if objective_name == "robust" else "mean_affinity_gain",
                    "rationale": rationale,
                }
            )
            priority += 1
            if len(payload_rows) >= int(self.config.proposal_count):
                break
        return {
            "round_strategy": (
                "Focus on edits that preserve anchors, reduce hotspot over-dependence, and add non-hotspot compensation."
                if objective_name == "robust"
                else "Focus on compact affinity-improving edits while keeping the scaffold editable and chemically valid."
            ),
            "exploration_mode": "balanced" if objective_name == "robust" else "focused",
            "action_templates": payload_rows,
            "global_caveats": [
                "Only supported edit templates are emitted; atom-level placement remains deterministic in code.",
                "Fallback strategy was used instead of a live GLM plan." if True else "",
            ],
        }

    def _sanitize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        priorities: list[int] = []
        for row in list(payload.get("action_templates") or []):
            edit_family = str(row.get("edit_family") or "").upper()
            pattern = str(row.get("pattern") or "")
            fragment = str(row.get("fragment") or "")
            if edit_family not in SUPPORTED_EDIT_FAMILIES or pattern not in SUPPORTED_PATTERNS:
                continue
            signature = (edit_family, pattern, fragment)
            if signature in seen:
                continue
            seen.add(signature)
            rows.append(
                {
                    "edit_family": edit_family,
                    "pattern": pattern,
                    "fragment": fragment,
                    "priority": int(row.get("priority") or len(rows) + 1),
                    "mechanistic_goal": str(row.get("mechanistic_goal") or ""),
                    "rationale": str(row.get("rationale") or ""),
                }
            )
            priorities.append(int(rows[-1]["priority"]))
            if len(rows) >= int(self.config.proposal_count):
                break
        rows.sort(key=lambda item: (int(item["priority"]), item["edit_family"], item["pattern"], item["fragment"]))
        for index, row in enumerate(rows, start=1):
            row["priority"] = index
        payload["action_templates"] = rows
        if str(payload.get("exploration_mode") or "") not in {"focused", "balanced", "exploratory"}:
            payload["exploration_mode"] = "balanced"
        payload["global_caveats"] = [str(item) for item in list(payload.get("global_caveats") or []) if str(item).strip()]
        payload["round_strategy"] = str(payload.get("round_strategy") or "")
        return payload

    def run(
        self,
        *,
        case_entry: dict[str, Any],
        objective_name: str,
        round_index: int,
        mechanism_summary: dict[str, Any],
        hotspot_residues: list[str],
        beam_frame: pd.DataFrame,
        action_space: dict[str, Any],
        qc_payload: dict[str, Any],
        lead_descriptors: dict[str, Any],
        temperature: float,
        disable_llm: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any], GLMCallRecord | None]:
        prompt_input = self.build_prompt_input(
            case_entry=case_entry,
            objective_name=objective_name,
            round_index=round_index,
            mechanism_summary=mechanism_summary,
            hotspot_residues=hotspot_residues,
            beam_frame=beam_frame,
            action_space=action_space,
            qc_payload=qc_payload,
            lead_descriptors=lead_descriptors,
        )
        if disable_llm:
            payload = self._sanitize_payload(
                self._fallback_payload(
                    objective_name=objective_name,
                    mechanism_summary=mechanism_summary,
                    action_space=action_space,
                    lead_descriptors=lead_descriptors,
                    prompt_input=prompt_input,
                )
            )
            return prompt_input, payload, None

        system_message = (
            "You are the ResistAgent Counter-Design Agent. "
            "You do not invent de novo scaffolds. You only prioritize supported edit templates for the current lead molecule. "
            "Return strict JSON. Use only the listed edit families, patterns, and fragments. "
            "For robust objective, favor anchor retention, non-hotspot compensation, and lower hotspot dependence. "
            "For naive objective, favor average affinity gains but keep chemistry compact and plausible. "
        )
        if bool(prompt_input["case_meta"].get("is_viral")):
            system_message += "For viral cases, do not output the strings COSMIC or DepMap anywhere. "
        user_message = (
            "Prioritize the next-round medicinal-chemistry edit templates for counter-design step beam search.\n"
            "Only emit template-level plans; atom placement is handled deterministically by code.\n"
            f"{prompt_input}"
        )
        try:
            payload, record = self.client.chat_json(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                self.output_schema(),
                temperature=float(temperature),
                max_tokens=int(self.config.max_tokens),
                thinking=self.config.thinking,
            )
            payload = self._sanitize_payload(payload)
            return prompt_input, payload, record
        except Exception:
            payload = self._sanitize_payload(
                self._fallback_payload(
                    objective_name=objective_name,
                    mechanism_summary=mechanism_summary,
                    action_space=action_space,
                    lead_descriptors=lead_descriptors,
                    prompt_input=prompt_input,
                )
            )
            return prompt_input, payload, None
