from types import SimpleNamespace

import pytest

from backend.app.agents import workflow
from backend.app.db.models import Customer, Transaction


def _models():
    transaction = Transaction(
        transaction_id="TXN-1",
        customer_id="C-1",
        amount=100,
        currency="USD",
        transaction_type="WIRE",
        metadata_json={
            "country_risk_score": 2,
            "risk_label": "suspicious",
            "risk_score": 0.99,
            "rule_hits": "R-SECRET",
            "unknown": {"gold_explanation": "SECRET GOLD"},
        },
    )
    customer = Customer(
        customer_id="C-1",
        country="US",
        kyc_level="Retail",
        kyc_risk_score=0.2,
        avg_transaction_amount=50,
    )
    return transaction, customer


def _report(confidence=0.5):
    return {
        "transaction_id": "TXN-1",
        "risk_level": "MEDIUM",
        "confidence": confidence,
        "summary": "Observed facts",
        "risk_factors": [],
        "similar_cases": [],
        "recommended_action": "Review transaction",
    }


def _verdict(supported=True, claims=None, more=False, query="", adjustment=0.1):
    return {
        "supported": supported,
        "unsupported_claims": [] if claims is None else claims,
        "needs_more_evidence": more,
        "suggested_query": query,
        "confidence_adjustment": adjustment,
    }


def _setup(monkeypatch, verdicts):
    captured = SimpleNamespace(
        planner=[], reports=[], verifiers=[], saved=[], policy_calls=[], case_calls=[]
    )
    monkeypatch.setattr(workflow, "get_transaction_with_customer", lambda db, transaction_id: _models())
    monkeypatch.setattr(
        workflow,
        "run_anomaly_analysis",
        lambda transaction, customer: {
            "features": {"amount_vs_customer_avg": 2.0},
            "anomaly_score": 0.25,
            "top_contributing_features": [],
        },
    )

    def planner(transaction, customer, features):
        captured.planner.append((transaction, customer, features))
        return {"policy_query": "p", "case_query": "c"}

    monkeypatch.setattr(workflow, "build_investigation_plan", planner)

    def policies(query, top_k=3):
        captured.policy_calls.append(query)
        suffix = "2" if len(captured.policy_calls) > 1 else "1"
        return [{"source_id": f"policy:P{suffix}", "text": "policy", "metadata": {"policy_id": f"P{suffix}"}}]

    def cases(query, top_k=3):
        captured.case_calls.append(query)
        suffix = "2" if len(captured.case_calls) > 1 else "1"
        return [{"source_id": f"case:C{suffix}", "text": "case", "metadata": {"case_id": f"C{suffix}"}}]

    monkeypatch.setattr(workflow, "retrieve_policy_evidence", policies)
    monkeypatch.setattr(workflow, "retrieve_similar_cases", cases)

    def report(transaction, customer, features, anomaly_score, evidence_registry):
        captured.reports.append((transaction, customer, evidence_registry))
        return _report(confidence=0.5 if len(captured.reports) == 1 else 0.7)

    monkeypatch.setattr(workflow, "generate_investigation_report", report)
    verdict_iter = iter(verdicts)

    def verifier(draft, transaction, customer, features, registry):
        captured.verifiers.append((draft, transaction, customer, registry))
        return next(verdict_iter)

    monkeypatch.setattr(workflow, "verify_investigation_report", verifier)
    monkeypatch.setattr(
        workflow,
        "create_investigation_report",
        lambda **kwargs: captured.saved.append(kwargs),
    )
    return captured


def test_supported_first_draft_and_prompts_use_clean_inputs(monkeypatch):
    captured = _setup(monkeypatch, [_verdict(adjustment=0.1)])
    result = workflow._run_investigation(object(), "TXN-1")
    assert result["status"] == "supported"
    assert result["confidence"] == 0.6
    assert result["trace"]["report_generation_passes"] == 1
    assert result["trace"]["verification_passes"] == 1
    assert result["trace"]["extra_retrieval_performed"] is False
    assert len(captured.saved) == 1
    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, (list, tuple)):
            return set().union(*(keys(item) for item in value)) if value else set()
        return set()

    captured_values = captured.planner + captured.reports + captured.verifiers
    assert {"risk_label", "risk_score", "rule_hits", "gold_explanation"}.isdisjoint(
        keys(captured_values)
    )
    assert "SECRET GOLD" not in repr(captured_values)


def test_corrected_report_is_reverified_and_uses_final_adjustment(monkeypatch):
    captured = _setup(
        monkeypatch,
        [
            _verdict(False, ["weak"], True, "new evidence", adjustment=-0.4),
            _verdict(True, [], False, "", adjustment=0.05),
        ],
    )
    result = workflow._run_investigation(object(), "TXN-1")
    assert result["status"] == "supported"
    assert result["confidence"] == 0.75
    assert result["trace"]["report_generation_passes"] == 2
    assert result["trace"]["verification_passes"] == 2
    assert captured.policy_calls == ["p", "new evidence"]
    assert "policy:P1" in captured.reports[1][2]
    assert "policy:P2" in captured.reports[1][2]
    assert captured.saved[0]["report_json"]["verification"]["supported"] is True


def test_still_unsupported_is_bounded_and_requires_human_review(monkeypatch):
    captured = _setup(
        monkeypatch,
        [
            _verdict(False, ["weak"], True, "new evidence"),
            _verdict(False, ["still weak"], True, "third query", adjustment=-0.2),
        ],
    )
    result = workflow._run_investigation(object(), "TXN-1")
    assert result["status"] == "needs_review"
    assert result["unresolved_claims"] == ["still weak"]
    assert "human review" in result["recommended_action"].lower()
    assert len(captured.reports) == len(captured.verifiers) == 2
    assert len(captured.policy_calls) == 2


def test_unsupported_without_query_does_not_retrieve_again(monkeypatch):
    captured = _setup(monkeypatch, [_verdict(False, ["unsupported"], True, "")])
    result = workflow._run_investigation(object(), "TXN-1")
    assert result["status"] == "needs_review"
    assert len(captured.policy_calls) == 1
    assert len(captured.reports) == 1


def test_inconsistent_verdict_never_passes(monkeypatch):
    _setup(monkeypatch, [_verdict(True, ["contradiction"], False, "")])
    result = workflow._run_investigation(object(), "TXN-1")
    assert result["status"] == "needs_review"


def test_verifier_exception_is_not_saved_or_approved(monkeypatch):
    captured = _setup(monkeypatch, [_verdict()])

    def fail(*args, **kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(workflow, "verify_investigation_report", fail)
    with pytest.raises(RuntimeError, match="provider failed"):
        workflow._run_investigation(object(), "TXN-1")
    assert captured.saved == []


def test_potential_historical_case_overlap_is_conservatively_filtered():
    cases = [
        {
            "source_id": "case:SAME",
            "text": "CUST-1 initiated a WIRE of USD 100.00 to Canada.",
            "metadata": {"case_id": "SAME"},
        },
        {
            "source_id": "case:OTHER",
            "text": "A different synthetic case.",
            "metadata": {"case_id": "OTHER"},
        },
    ]
    filtered = workflow._exclude_potential_case_overlap(
        cases, {"transaction_id": "TXN-1", "customer_id": "CUST-1", "amount": 100}
    )
    assert [case["source_id"] for case in filtered] == ["case:OTHER"]
