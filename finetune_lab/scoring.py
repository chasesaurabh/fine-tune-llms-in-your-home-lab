"""Strict consumer-side scoring: never repair a model response to raise its score."""
from __future__ import annotations

import json
from typing import Any

CATEGORIES = frozenset({"network", "hardware", "software", "access"})
URGENCIES = frozenset({"low", "medium", "high"})
EXTRACTION_KEYS = frozenset({"category", "urgency", "asset"})


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite JSON number")


def parse_object(text: str) -> dict[str, Any] | None:
    """Accept one complete JSON object; whitespace is allowed, other text is not."""
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text, object_pairs_hook=_unique_pairs,
                           parse_constant=_reject_constant)
    except (ValueError, TypeError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def extraction_schema(value: dict[str, Any] | None) -> bool:
    return (isinstance(value, dict) and set(value) == EXTRACTION_KEYS
            and all(isinstance(item, str) for item in value.values())
            and value["category"] in CATEGORIES
            and value["urgency"] in URGENCIES
            and bool(value["asset"].strip()))


def score_extraction(text: str, target: dict[str, str]) -> dict[str, Any]:
    if not extraction_schema(target):
        raise ValueError("reference target violates extraction schema")
    parsed = parse_object(text)
    schema_valid = extraction_schema(parsed)
    return {
        "parsed": parsed,
        "valid_json_object": parsed is not None,
        "schema_valid": schema_valid,
        "exact_object_match": schema_valid and parsed == target,
        "fields": {key: schema_valid and parsed[key] == target[key]
                   for key in sorted(EXTRACTION_KEYS)},
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report counts and denominators; zero cases is not a perfect score."""
    if not rows:
        raise ValueError("cannot score an empty evaluation")
    return {
        "cases": len(rows),
        **{key: sum(row[key] is True for row in rows)
           for key in ("valid_json_object", "schema_valid", "exact_object_match")},
        "fields": {key: sum(row["fields"][key] is True for row in rows)
                   for key in sorted(EXTRACTION_KEYS)},
    }
