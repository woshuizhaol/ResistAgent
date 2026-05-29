#!/usr/bin/env python3
"""Minimal LangGraph-compatible orchestration scaffold for ResistAgent."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tools.runtime import MAX_WORKERS, ensure_dir, iso_now


@dataclass
class DecisionRecord:
    agent_name: str
    decision_type: str
    input_artifacts: list[str]
    tool_calls: list[dict[str, Any]]
    decision_rationale: str
    output_artifacts: list[str]
    decided_at: str = field(default_factory=iso_now)


@dataclass
class OrchestratorState:
    project_id: str
    stage: str
    inputs: dict[str, Any]
    artifacts: dict[str, Any]
    qc: dict[str, Any]
    software_versions: dict[str, Any]
    seeds: dict[str, Any]
    commands: list[str]
    llm_decisions: list[dict[str, Any]] = field(default_factory=list)


class StageOrchestrator:
    """Records routing decisions and tool calls without mutating tool outputs."""

    def __init__(self, state_path: str | Path, max_workers: int = MAX_WORKERS) -> None:
        self.state_path = Path(state_path)
        self.max_workers = max_workers
        self._tools: dict[str, Callable[..., Any]] = {}

    def register_tool(self, name: str, func: Callable[..., Any]) -> None:
        self._tools[name] = func

    def call_tool(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name](**kwargs)

    def append_decision(self, decision: DecisionRecord) -> None:
        ensure_dir(self.state_path.parent)
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        else:
            state = {"llm_decisions": []}
        state.setdefault("llm_decisions", []).append(decision.__dict__)
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")


def build_langgraph_placeholder() -> str:
    """Return a stable placeholder until stage-specific nodes are added."""
    try:
        from langgraph.graph import StateGraph  # type: ignore
    except ImportError:
        return "langgraph_not_installed"
    graph = StateGraph(dict)
    graph.add_node("noop", lambda state: state)
    graph.set_entry_point("noop")
    graph.set_finish_point("noop")
    return "langgraph_placeholder_ready"
