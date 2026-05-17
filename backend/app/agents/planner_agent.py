from __future__ import annotations

import json
from typing import Any

from ..llm.gemini_client import generate_json

REQUIRED_KEYS = {
    "policy_query",
    "case_query",
    "focus_areas",
    "suspected_risk_categories",
}


def _ensure_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise RuntimeError(f"Planner output field '{field_name}' must be a list of strings.")
    normalized = [str(item).strip() for item in value if str(item).strip()]
    return normalized


def _validate_planner_output(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise RuntimeError(f"Planner output is missing required keys: {', '.join(missing)}")

    policy_query = str(payload.get("policy_query", "")).strip()
    case_query = str(payload.get("case_query", "")).strip()
    if not policy_query:
        raise RuntimeError("Planner output field 'policy_query' is empty.")
    if not case_query:
        raise RuntimeError("Planner output field 'case_query' is empty.")

    focus_areas = _ensure_string_list(payload.get("focus_areas"), "focus_areas")
    suspected_risk_categories = _ensure_string_list(
        payload.get("suspected_risk_categories"), "suspected_risk_categories"
    )

    return {
        "policy_query": policy_query,
        "case_query": case_query,
        "focus_areas": focus_areas,
        "suspected_risk_categories": suspected_risk_categories,
    }


def build_investigation_plan(
    transaction: dict[str, Any],
    customer: dict[str, Any],
    features: dict[str, Any],
) -> dict[str, Any]:
    prompt = f"""
You are a financial risk investigation planner.

Create retrieval planning JSON for an agentic investigation workflow.

Return ONLY valid JSON with exactly these keys:
- policy_query (string)
- case_query (string)
- focus_areas (array of strings)
- suspected_risk_categories (array of strings)

Use the inputs below:

transaction:
{json.dumps(transaction, indent=2, default=str)}

customer:
{json.dumps(customer, indent=2, default=str)}

extracted_features:
{json.dumps(features, indent=2, default=str)}

Constraints:
- Make queries specific and retrieval-ready.
- Focus on AML/risk policy language and historical case similarity language.
- Keep arrays concise and practical.
- Do not include markdown or explanation outside JSON.
""".strip()

    planner_output = generate_json(prompt)
    return _validate_planner_output(planner_output)
