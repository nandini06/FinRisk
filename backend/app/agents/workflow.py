from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from ..db.database import SessionLocal
from ..db.repositories import create_investigation_report, get_transaction_with_customer
from .anomaly_agent import run_anomaly_analysis
from .case_agent import retrieve_similar_cases
from .planner_agent import build_investigation_plan
from .policy_agent import retrieve_policy_evidence
from .report_agent import generate_investigation_report
from .verifier_agent import verify_investigation_report


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _model_to_dict(model: Any) -> dict[str, Any]:
    return {
        column.name: _serialize_value(getattr(model, column.name))
        for column in model.__table__.columns
    }


def _clamp_0_1(value: float) -> float:
    return max(0.0, min(1.0, value))


def _adjust_report_confidence(report: dict[str, Any], confidence_adjustment: float) -> dict[str, Any]:
    confidence = report.get("confidence", 0.0)
    try:
        base_confidence = float(confidence)
    except (TypeError, ValueError):
        base_confidence = 0.0

    adjusted_report = dict(report)
    adjusted_report["confidence"] = round(_clamp_0_1(base_confidence + confidence_adjustment), 6)
    return adjusted_report


def _dedupe_evidence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in rows:
        metadata = row.get("metadata", {})
        key = f"{row.get('text', '')}|{metadata}|{row.get('similarity_score', 0.0)}"
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    unique_rows.sort(key=lambda item: float(item.get("similarity_score", 0.0)), reverse=True)
    return unique_rows


def run_investigation(transaction_id: str) -> dict[str, Any]:
    with SessionLocal() as db:
        record = get_transaction_with_customer(db, transaction_id=transaction_id)
        if record is None:
            raise LookupError(f"Transaction '{transaction_id}' not found.")

        transaction_model, customer_model = record
        transaction = _model_to_dict(transaction_model)
        customer = _model_to_dict(customer_model)

        anomaly_result = run_anomaly_analysis(transaction=transaction, customer=customer)
        features = anomaly_result["features"]
        anomaly_score = anomaly_result["anomaly_score"]

        plan = build_investigation_plan(
            transaction=transaction,
            customer=customer,
            features=features,
        )

        policy_evidence = retrieve_policy_evidence(policy_query=plan["policy_query"], top_k=3)
        case_evidence = retrieve_similar_cases(case_query=plan["case_query"], top_k=3)

        draft_report = generate_investigation_report(
            transaction=transaction,
            customer=customer,
            features=features,
            anomaly_score=anomaly_score,
            policy_evidence=policy_evidence,
            similar_cases=case_evidence,
        )

        verifier_result = verify_investigation_report(
            draft_report=draft_report,
            policy_evidence=policy_evidence,
            case_evidence=case_evidence,
            anomaly_features=features,
        )

        final_policy_evidence = list(policy_evidence)
        final_case_evidence = list(case_evidence)
        final_report = dict(draft_report)

        if verifier_result["needs_more_evidence"]:
            suggested_query = verifier_result.get("suggested_query", "").strip()
            if suggested_query:
                extra_policy_evidence = retrieve_policy_evidence(policy_query=suggested_query, top_k=3)
                extra_case_evidence = retrieve_similar_cases(case_query=suggested_query, top_k=3)

                final_policy_evidence = _dedupe_evidence_rows(policy_evidence + extra_policy_evidence)[:3]
                final_case_evidence = _dedupe_evidence_rows(case_evidence + extra_case_evidence)[:3]

                # One self-correction retry only.
                final_report = generate_investigation_report(
                    transaction=transaction,
                    customer=customer,
                    features=features,
                    anomaly_score=anomaly_score,
                    policy_evidence=final_policy_evidence,
                    similar_cases=final_case_evidence,
                )

        final_report = _adjust_report_confidence(
            report=final_report,
            confidence_adjustment=float(verifier_result.get("confidence_adjustment", 0.0)),
        )
        final_report["transaction_id"] = transaction_id

        create_investigation_report(
            db=db,
            transaction_id=transaction_id,
            report_json=final_report,
            risk_level=str(final_report.get("risk_level", "")).upper() or None,
            confidence=float(final_report.get("confidence", 0.0)),
            summary=str(final_report.get("summary", "")) or None,
        )

    return final_report
