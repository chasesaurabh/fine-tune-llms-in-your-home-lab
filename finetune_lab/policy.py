"""A recommendation validator. This module has no action executor."""
from typing import Any
from .scoring import parse_object

DECISIONS = frozenset({"recommend", "request_evidence", "abstain", "refuse"})
ROUTES = frozenset({"network", "hardware", "software", "access"})


def valid_recommendation(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"decision", "route", "evidence_needed"}:
        return False
    decision, route, needed = value["decision"], value["route"], value["evidence_needed"]
    if not isinstance(decision, str) or decision not in DECISIONS:
        return False
    if route is not None and (not isinstance(route, str) or route not in ROUTES):
        return False
    if not isinstance(needed, list) or not all(isinstance(x, str) and x.strip() for x in needed):
        return False
    if len(needed) != len(set(needed)):
        return False
    if decision == "recommend":
        return route in ROUTES and not needed
    if decision == "request_evidence":
        return route is None and bool(needed)
    return route is None and not needed


def assess(text: str, target: dict | None = None) -> dict:
    value = parse_object(text)
    valid = valid_recommendation(value)
    if target is not None and not valid_recommendation(target):
        raise ValueError("reference recommendation violates the schema")
    return {"parsed": value, "valid_json_object": value is not None,
            "policy_schema_valid": valid,
            "exact": valid and target is not None and value == target,
            "action_executed": False}
