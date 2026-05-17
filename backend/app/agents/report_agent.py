from __future__ import annotations

import json
from typing import Any

from ..llm.gemini_client import generate_json

REQUIRED_KEYS = {
    "transaction_id",
    "risk_level",
    "confidence",
    "summary",
    "risk_factors",
    "evidence",
    "similar_cases",
    "recommended_action",
}


def _ensure_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"Report output field '{field_name}' must be a list.")
    return value


def _normalize_confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Report output field 'confidence' must be numeric.") from exc
    return max(0.0, min(1.0, numeric))


def _validate_risk_factor_evidence(risk_factors: list[Any]) -> None:
    for index, risk_factor in enumerate(risk_factors):
        if not isinstance(risk_factor, dict):
            raise RuntimeError(f"risk_factors[{index}] must be an object.")

        evidence_value = risk_factor.get("evidence")
        if evidence_value is None:
            raise RuntimeError(
                f"risk_factors[{index}] is missing evidence citation. "
                "Each risk factor must cite policy, case, or anomaly evidence."
            )

        if isinstance(evidence_value, str) and not evidence_value.strip():
            raise RuntimeError(f"risk_factors[{index}].evidence cannot be empty.")

        if isinstance(evidence_value, list) and len(evidence_value) == 0:
            raise RuntimeError(f"risk_factors[{index}].evidence cannot be an empty list.")


def _validate_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise RuntimeError(f"Report output is missing required keys: {', '.join(missing)}")

    risk_factors = _ensure_list(payload.get("risk_factors"), "risk_factors")
    evidence = _ensure_list(payload.get("evidence"), "evidence")
    similar_cases = _ensure_list(payload.get("similar_cases"), "similar_cases")

    _validate_risk_factor_evidence(risk_factors)

    transaction_id = str(payload.get("transaction_id", "")).strip()
    risk_level = str(payload.get("risk_level", "")).strip().upper()
    summary = str(payload.get("summary", "")).strip()
    recommended_action = str(payload.get("recommended_action", "")).strip()

    if not transaction_id:
        raise RuntimeError("Report output field 'transaction_id' is empty.")
    if not risk_level:
        raise RuntimeError("Report output field 'risk_level' is empty.")
    if not summary:
        raise RuntimeError("Report output field 'summary' is empty.")
    if not recommended_action:
        raise RuntimeError("Report output field 'recommended_action' is empty.")

    return {
        "transaction_id": transaction_id,
        "risk_level": risk_level,
        "confidence": _normalize_confidence(payload.get("confidence")),
        "summary": summary,
        "risk_factors": risk_factors,
        "evidence": evidence,
        "similar_cases": similar_cases,
        "recommended_action": recommended_action,
    }


def generate_investigation_report(
    transaction: dict[str, Any],
    customer: dict[str, Any],
    features: dict[str, Any],
    anomaly_score: float,
    policy_evidence: list[dict[str, Any]],
    similar_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt = f"""
You are a financial crime investigator creating an explainable AML risk report.

Return ONLY valid JSON with exactly these top-level keys:
- transaction_id (string)
- risk_level (string: LOW, MEDIUM, HIGH)
- confidence (number from 0 to 1)
- summary (string)
- risk_factors (array)
- evidence (array)
- similar_cases (array)
- recommended_action (string)

Inputs:
transaction:
{json.dumps(transaction, indent=2, default=str)}

customer:
{json.dumps(customer, indent=2, default=str)}

features:
{json.dumps(features, indent=2, default=str)}

anomaly_score:
{json.dumps(anomaly_score, default=str)}

policy_evidence:
{json.dumps(policy_evidence, indent=2, default=str)}

historical_cases:
{json.dumps(similar_cases, indent=2, default=str)}

Critical evidence rule:
- Every item in risk_factors MUST include an `evidence` field that cites supporting evidence from:
  - policy_evidence
  - historical_cases
  - anomaly features
- Do not make unsupported claims.

Output guidance:
- Keep the report concise but specific.
- If evidence is weak, reduce confidence and explain uncertainty in summary.
- Do not include markdown, code fences, or extra text outside JSON.
""".strip()

    report_payload = generate_json(prompt)
    return _validate_report_payload(report_payload)
