from __future__ import annotations

import math
import re
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session

from ..db.database import SessionLocal
from ..db.repositories import create_investigation_report, get_transaction_with_customer
from .anomaly_agent import run_anomaly_analysis
from .case_retriever import retrieve_similar_cases
from .evidence import build_evidence_registry, merge_evidence_rows, registry_records
from .inference_inputs import build_inference_inputs
from .planner import build_investigation_plan
from .policy_retriever import retrieve_policy_evidence
from .report_generator import generate_investigation_report
from .verifier import verify_investigation_report


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


def _verdict_passes(verdict: dict[str, Any]) -> bool:
    return (
        verdict.get("supported") is True
        and verdict.get("unsupported_claims") == []
        and verdict.get("needs_more_evidence") is False
    )


def _adjust_confidence(report: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
    confidence = float(report["confidence"])
    adjustment = float(verdict["confidence_adjustment"])
    if not math.isfinite(confidence) or not math.isfinite(adjustment):
        raise RuntimeError("Final confidence values must be finite.")
    adjusted = dict(report)
    adjusted["confidence"] = round(max(0.0, min(1.0, confidence + adjustment)), 6)
    return adjusted


def _registry(
    transaction: dict[str, Any],
    customer: dict[str, Any],
    features: dict[str, Any],
    anomaly_score: float,
    policies: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return build_evidence_registry(
        transaction=transaction,
        customer=customer,
        features=features,
        anomaly_score=anomaly_score,
        policy_evidence=policies,
        case_evidence=cases,
    )


def _exclude_potential_case_overlap(
    cases: list[dict[str, Any]], transaction: dict[str, Any]
) -> list[dict[str, Any]]:
    """Conservatively exclude direct IDs and potential customer-plus-amount overlaps."""
    transaction_id = str(transaction.get("transaction_id") or "").lower()
    customer_id = str(transaction.get("customer_id") or "").lower()
    amount = transaction.get("amount")
    amount_tokens: set[str] = set()
    try:
        numeric_amount = float(amount)
        amount_tokens = {f"{numeric_amount:.2f}", f"{numeric_amount:g}"}
    except (TypeError, ValueError):
        pass

    output = []
    for row in cases:
        text = str(row.get("text") or "").lower()
        if transaction_id and transaction_id in text:
            continue
        normalized_text = re.sub(r"[^a-z0-9.\-]+", " ", text)
        if customer_id and customer_id in normalized_text and any(
            token in normalized_text for token in amount_tokens
        ):
            continue
        output.append(row)
    return output


def _run_investigation(db: Session, transaction_id: str) -> dict[str, Any]:
    record = get_transaction_with_customer(db, transaction_id=transaction_id)
    if record is None:
        raise LookupError(f"Transaction '{transaction_id}' not found.")

    transaction_model, customer_model = record
    raw_transaction = _model_to_dict(transaction_model)
    raw_customer = _model_to_dict(customer_model)
    transaction, customer = build_inference_inputs(raw_transaction, raw_customer)

    anomaly_result = run_anomaly_analysis(transaction=transaction, customer=customer)
    features = anomaly_result["features"]
    anomaly_score = anomaly_result["anomaly_score"]
    plan = build_investigation_plan(transaction=transaction, customer=customer, features=features)

    policies = retrieve_policy_evidence(plan["policy_query"], top_k=3)
    cases = _exclude_potential_case_overlap(
        retrieve_similar_cases(plan["case_query"], top_k=3), transaction
    )
    evidence_registry = _registry(
        transaction, customer, features, anomaly_score, policies, cases
    )

    draft = generate_investigation_report(
        transaction, customer, features, anomaly_score, evidence_registry
    )
    initial_verdict = verify_investigation_report(
        draft, transaction, customer, features, evidence_registry
    )

    trace: dict[str, Any] = {
        "policy_query": plan["policy_query"],
        "case_query": plan["case_query"],
        "passes": [
            {
                "pass": 1,
                "evidence_ids": list(evidence_registry),
                "verification": initial_verdict,
            }
        ],
        "extra_retrieval_performed": False,
        "report_generation_passes": 1,
        "verification_passes": 1,
    }

    final_report = draft
    final_verdict = initial_verdict
    suggested_query = str(initial_verdict.get("suggested_query") or "").strip()

    if not _verdict_passes(initial_verdict) and suggested_query:
        extra_policies = retrieve_policy_evidence(suggested_query, top_k=3)
        extra_cases = _exclude_potential_case_overlap(
            retrieve_similar_cases(suggested_query, top_k=3), transaction
        )
        policies = merge_evidence_rows(policies, extra_policies)
        cases = merge_evidence_rows(cases, extra_cases)
        evidence_registry = _registry(
            transaction, customer, features, anomaly_score, policies, cases
        )
        final_report = generate_investigation_report(
            transaction, customer, features, anomaly_score, evidence_registry
        )
        final_verdict = verify_investigation_report(
            final_report, transaction, customer, features, evidence_registry
        )
        trace["extra_retrieval_performed"] = True
        trace["additional_query"] = suggested_query
        trace["report_generation_passes"] = 2
        trace["verification_passes"] = 2
        trace["passes"].append(
            {
                "pass": 2,
                "evidence_ids": list(evidence_registry),
                "verification": final_verdict,
            }
        )

    final_report = _adjust_confidence(final_report, final_verdict)
    passed = _verdict_passes(final_verdict)
    final_report["status"] = "supported" if passed else "needs_review"
    final_report["evidence"] = registry_records(evidence_registry)
    final_report["verification"] = final_verdict
    final_report["trace"] = trace

    if not passed:
        final_report["unresolved_claims"] = list(final_verdict["unsupported_claims"])
        final_report["remaining_evidence_request"] = str(
            final_verdict.get("suggested_query") or ""
        ).strip()
        action = str(final_report.get("recommended_action") or "").strip()
        if "human review" not in action.lower():
            final_report["recommended_action"] = (
                f"{action} Requires human review before any decision."
            ).strip()

    create_investigation_report(
        db=db,
        transaction_id=transaction_id,
        report_json=final_report,
        risk_level=final_report["risk_level"],
        confidence=final_report["confidence"],
        summary=final_report["summary"],
    )
    return final_report


def run_investigation(transaction_id: str) -> dict[str, Any]:
    with SessionLocal() as db:
        return _run_investigation(db, transaction_id)
