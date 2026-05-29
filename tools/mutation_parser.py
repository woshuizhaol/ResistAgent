#!/usr/bin/env python3
"""Mutation parsing helpers used by Stage 1 and later stages."""

from __future__ import annotations

import re
from dataclasses import dataclass

SINGLE_SUBSTITUTION_RE = re.compile(r"^([A-Z\*])(\d+)([A-Z\*])$")
INDEL_RE = re.compile(r"^([A-Z\*])(\d+)(?:_([A-Z\*])(\d+))?delins([A-Z\*\.]+)$", re.IGNORECASE)
DELETION_RE = re.compile(r"^([A-Z\*])(\d+)(?:_([A-Z\*])(\d+))?del([A-Z\*\.]+)?$", re.IGNORECASE)
INSERTION_RE = re.compile(r"^([A-Z\*])(\d+)_([A-Z\*])(\d+)ins([A-Z\*\.]+)$", re.IGNORECASE)


@dataclass
class ParsedComponent:
    raw: str
    mutation_class: str
    start_pos: int | None
    end_pos: int | None
    ref_aa: str | None
    alt_aa: str | None
    sort_key: tuple[int, str]


def _normalize_token(token: str) -> str:
    return token.strip().replace(" ", "")


def parse_component(component: str) -> ParsedComponent:
    token = _normalize_token(component)
    match = SINGLE_SUBSTITUTION_RE.match(token)
    if match:
        ref_aa, position, alt_aa = match.groups()
        pos = int(position)
        return ParsedComponent(
            raw=f"{ref_aa}{pos}{alt_aa}",
            mutation_class="single_substitution",
            start_pos=pos,
            end_pos=pos,
            ref_aa=ref_aa,
            alt_aa=alt_aa,
            sort_key=(pos, f"{ref_aa}{alt_aa}"),
        )

    match = INDEL_RE.match(token)
    if match:
        ref_start, start_pos, ref_end, end_pos, alt = match.groups()
        start = int(start_pos)
        end = int(end_pos) if end_pos else start
        return ParsedComponent(
            raw=f"{ref_start}{start}" + (f"_{ref_end}{end}" if ref_end and end_pos else "") + f"delins{alt}",
            mutation_class="indel",
            start_pos=start,
            end_pos=end,
            ref_aa=ref_start,
            alt_aa=alt,
            sort_key=(start, "indel"),
        )

    match = DELETION_RE.match(token)
    if match:
        ref_start, start_pos, ref_end, end_pos, deleted = match.groups()
        start = int(start_pos)
        end = int(end_pos) if end_pos else start
        return ParsedComponent(
            raw=f"{ref_start}{start}" + (f"_{ref_end}{end}" if ref_end and end_pos else "") + f"del{deleted or ''}",
            mutation_class="deletion",
            start_pos=start,
            end_pos=end,
            ref_aa=ref_start,
            alt_aa=deleted or "",
            sort_key=(start, "deletion"),
        )

    match = INSERTION_RE.match(token)
    if match:
        ref_start, start_pos, ref_end, end_pos, inserted = match.groups()
        start = int(start_pos)
        end = int(end_pos)
        return ParsedComponent(
            raw=f"{ref_start}{start}_{ref_end}{end}ins{inserted}",
            mutation_class="insertion",
            start_pos=start,
            end_pos=end,
            ref_aa=ref_start,
            alt_aa=inserted,
            sort_key=(start, "insertion"),
        )

    return ParsedComponent(
        raw=token,
        mutation_class="other",
        start_pos=None,
        end_pos=None,
        ref_aa=None,
        alt_aa=None,
        sort_key=(10**9, token),
    )


def parse_mutation(text: str | None) -> dict[str, object]:
    value = (text or "").strip()
    if not value:
        return {
            "raw_mutation": text,
            "parsed_mutation_type": "missing",
            "component_mutations": [],
            "component_mutation_keys": [],
            "combination_size": 0,
            "mutation_parse_ok": False,
        }

    raw_components = [_normalize_token(token) for token in value.split("+") if _normalize_token(token)]
    parsed_components = [parse_component(component) for component in raw_components]
    ordered_components = sorted(parsed_components, key=lambda item: item.sort_key)
    mutation_classes = {component.mutation_class for component in ordered_components}

    if len(ordered_components) == 1:
        parsed_mutation_type = ordered_components[0].mutation_class
    elif mutation_classes == {"single_substitution"}:
        parsed_mutation_type = "multi_substitution"
    else:
        parsed_mutation_type = "complex_multi_mutation"

    return {
        "raw_mutation": value,
        "parsed_mutation_type": parsed_mutation_type,
        "component_mutations": [component.raw for component in ordered_components],
        "parsed_components": ordered_components,
        "component_mutation_keys": [component.raw for component in ordered_components],
        "combination_size": len(ordered_components),
        "mutation_parse_ok": all(component.mutation_class != "other" for component in ordered_components),
    }


def component_to_mutation_key(gene_symbol: str | None, component: ParsedComponent) -> str | None:
    if not gene_symbol:
        return None
    return f"{gene_symbol}:{component.raw}"


def combination_key(gene_symbol: str | None, parsed_components: list[ParsedComponent]) -> str | None:
    if not gene_symbol or len(parsed_components) <= 1:
        return None
    ordered = sorted(parsed_components, key=lambda item: item.sort_key)
    return f"{gene_symbol}:{'+'.join(component.raw for component in ordered)}"


def first_position(component: ParsedComponent) -> int | None:
    return component.start_pos
