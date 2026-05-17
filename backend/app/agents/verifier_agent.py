from __future__ import annotations

import json
from typing import Any

from ..llm.gemini_client import generate_json

REQUIRED_KEYS = {
    "supported",
    "unsupported_claims",
    "needs_more_evidence",
    "suggested_query",
    "confidence_adjustment",
}


def _to_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise RuntimeError(f"Verifier output field '{field_name}' must be a boolean.")


def _to_string(value: Any, field_name: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if field_name == "suggested_query":
        return text
    if not text:
        raise RuntimeError(f"Verifier output field '{field_name}' cannot be empty.")
    return text


def _to_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise RuntimeError(f"Verifier output field '{field_name}' must be a list.")
    return [str(item).strip() for item in value if str(item).strip()]


def _to_float(value: Any, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Verifier output field '{field_name}' must be numeric.") from exc
    return max(-1.0, min(1.0, numeric))


def _validate_verifier_output(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise RuntimeError(f"Verifier output is missing required keys: {', '.join(missing)}")

    normalized = {
        "supported": _to_bool(payload.get("supported"), "supported"),
        "unsupported_claims": _to_string_list(payload.get("unsupported_claims"), "unsupported_claims"),
        "needs_more_evidence": _to_bool(payload.get("needs_more_evidence"), "needs_more_evidence"),
        "suggested_query": _to_string(payload.get("suggested_query"), "suggested_query"),
        "confidence_adjustment": _to_float(payload.get("confidence_adjustment"), "confidence_adjustment"),
    }

    if normalized["needs_more_evidence"] and not normalized["suggested_query"]:
        raise RuntimeError("Verifier indicated more evidence is needed but suggested_query is empty.")

    return normalized


def verify_investigation_report(
    draft_report: dict[str, Any],
    policy_evidence: list[dict[str, Any]],
    case_evidence: list[dict[str, Any]],
    anomaly_features: dict[str, Any],
) -> dict[str, Any]:
    prompt = f"""
You are a verification agent for financial risk investigation reports.

Review the draft report against the retrieved evidence and anomaly features.
You must verify evidence support quality and recommend whether retrieval should be repeated.

Return ONLY valid JSON with exactly these keys:
- supported (boolean)
- unsupported_claims (array of strings)
- needs_more_evidence (boolean)
- suggested_query (string)
- confidence_adjustment (number between -1.0 and 1.0)

Inputs:
draft_report:
{json.dumps(draft_report, indent=2, default=str)}

retrieved_policy_evidence:
{json.dumps(policy_evidence, indent=2, default=str)}

retrieved_case_evidence:
{json.dumps(case_evidence, indent=2, default=str)}

anomaly_features:
{json.dumps(anomaly_features, indent=2, default=str)}

Validation goals:
1) Are all risk claims supported by evidence?
2) Which claims are unsupported or weakly supported?
3) Is additional retrieval needed?
4) If needed, what single improved retrieval query should be used?
5) Should confidence be adjusted up/down?

Rules:
- If evidence is weak, set needs_more_evidence=true and provide a concrete suggested_query.
- Keep unsupported_claims short and specific.
- If report is well-supported, set needs_more_evidence=false and suggested_query="".
- Output JSON only, no markdown and no extra explanation.
""".strip()

    verifier_payload = generate_json(prompt)
    return _validate_verifier_output(verifier_payload)
