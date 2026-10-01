import math

import pytest
from pydantic import ValidationError

from backend.app.agents import report_agent
from backend.app.llm.gemini_client import GeminiProviderError, StructuredOutputError
from backend.app.schemas.investigation import ReportOutput, VerifierOutput


def valid_report(level="LOW", confidence=0.0):
    return {
        "transaction_id": "TXN-1",
        "risk_level": level,
        "risk_level_rationale": "Model assessment based on the supplied amount feature.",
        "risk_level_source_ids": ["feature:amount"],
        "confidence": confidence,
        "summary": "Summary",
        "risk_factors": [
            {
                "factor": "Amount",
                "severity": level,
                "policy_severities": [],
                "severity_rationale": "Assessed from the supplied amount feature.",
                "explanation": "Observed amount",
                "source_ids": ["feature:amount"],
            }
        ],
        "similar_cases": [],
        "recommended_action": "Suggested for human review: Review the amount.",
        "recommended_action_basis": "DISCRETIONARY_HUMAN_REVIEW",
        "recommended_action_source_ids": ["feature:amount"],
        "recommended_action_rationale": "A human should assess the observed amount.",
    }


@pytest.mark.parametrize("level", ["LOW", "MEDIUM", "HIGH"])
def test_risk_levels_and_zero_confidence_are_valid(level):
    parsed = ReportOutput.model_validate(valid_report(level=level, confidence=0.0))
    assert parsed.risk_level == level
    assert parsed.confidence == 0.0


@pytest.mark.parametrize("bad", ["CRITICAL", "low", math.nan, math.inf])
def test_invalid_level_or_nonfinite_confidence_rejected(bad):
    payload = valid_report()
    if isinstance(bad, str):
        payload["risk_level"] = bad
    else:
        payload["confidence"] = bad
    with pytest.raises(ValidationError):
        ReportOutput.model_validate(payload)


def test_verifier_requires_actual_booleans_and_finite_adjustment():
    payload = {
        "supported": "true",
        "unsupported_claims": [],
        "needs_more_evidence": False,
        "suggested_query": "",
        "confidence_adjustment": 0.0,
    }
    with pytest.raises(ValidationError):
        VerifierOutput.model_validate(payload)
    payload["supported"] = True
    payload["confidence_adjustment"] = math.nan
    with pytest.raises(ValidationError):
        VerifierOutput.model_validate(payload)


def test_transaction_id_consistency():
    with pytest.raises(ValueError, match="transaction_id"):
        report_agent._validate_report(valid_report() | {"transaction_id": "WRONG"}, "TXN-1", {"feature:amount": {}})


def test_policy_severity_is_distinct_from_higher_model_assessment():
    registry = {
        "policy:AML-003": {"metadata": {"severity": "MEDIUM"}},
        "feature:amount": {"value": 54.0},
    }
    payload = valid_report(level="HIGH")
    payload["risk_factors"][0].update(
        policy_severities=[{"policy_id": "AML-003", "severity": "MEDIUM"}],
        severity_rationale=(
            "The model assesses HIGH because the supplied amount feature is 54x; "
            "AML-003 itself is MEDIUM."
        ),
        source_ids=["policy:AML-003", "feature:amount"],
    )

    result = report_agent._validate_report(payload, "TXN-1", registry)
    assert result["risk_factors"][0]["severity"] == "HIGH"
    assert result["risk_factors"][0]["policy_severities"] == [
        {"policy_id": "AML-003", "severity": "MEDIUM"}
    ]


def test_higher_model_severity_requires_fact_or_feature_support():
    registry = {"policy:AML-003": {"metadata": {"severity": "MEDIUM"}}}
    payload = valid_report(level="HIGH")
    payload["risk_level_source_ids"] = ["policy:AML-003"]
    payload["recommended_action_source_ids"] = ["policy:AML-003"]
    payload["risk_factors"][0].update(
        policy_severities=[{"policy_id": "AML-003", "severity": "MEDIUM"}],
        severity_rationale="Model elevation without factual support.",
        source_ids=["policy:AML-003"],
    )

    with pytest.raises(ValueError, match="without fact/feature evidence"):
        report_agent._validate_report(payload, "TXN-1", registry)


def test_policy_required_and_discretionary_actions_are_labeled_and_sourced():
    registry = {
        "policy:AML-003": {"metadata": {"severity": "MEDIUM"}},
        "feature:amount": {"value": 54.0},
    }
    discretionary = valid_report()
    assert report_agent._validate_report(discretionary, "TXN-1", registry)[
        "recommended_action_basis"
    ] == "DISCRETIONARY_HUMAN_REVIEW"

    policy_required = valid_report()
    policy_required.update(
        recommended_action="Policy-required: Investigate the unusual activity.",
        recommended_action_basis="POLICY_REQUIRED",
        recommended_action_source_ids=["policy:AML-003"],
        recommended_action_rationale="AML-003 explicitly says the activity should be investigated.",
    )
    assert report_agent._validate_report(policy_required, "TXN-1", registry)[
        "recommended_action_basis"
    ] == "POLICY_REQUIRED"

    mislabeled = valid_report()
    mislabeled["recommended_action"] = "Hold the transaction."
    with pytest.raises(ValueError, match="Suggested for human review"):
        report_agent._validate_report(mislabeled, "TXN-1", registry)


def test_report_repair_once_then_success(monkeypatch):
    calls = []

    def fake_generate(prompt, response_schema=None):
        calls.append(prompt)
        if len(calls) == 1:
            raise StructuredOutputError("malformed JSON")
        return valid_report()

    monkeypatch.setattr(report_agent, "generate_json", fake_generate)
    result = report_agent.generate_investigation_report(
        {"transaction_id": "TXN-1"}, {}, {}, 0.0, {"feature:amount": {}}
    )
    assert result["transaction_id"] == "TXN-1"
    assert len(calls) == 2


def test_provider_failure_is_clear_and_not_repaired(monkeypatch):
    calls = []

    def fail(prompt, response_schema=None):
        calls.append(prompt)
        raise GeminiProviderError("private provider details")

    monkeypatch.setattr(report_agent, "generate_json", fail)
    with pytest.raises(RuntimeError, match="provider request failed"):
        report_agent.generate_investigation_report(
            {"transaction_id": "TXN-1"}, {}, {}, 0.0, {"feature:amount": {}}
        )
    assert len(calls) == 1


def test_report_repair_is_exhausted_after_two_calls(monkeypatch):
    calls = []

    def fail(prompt, response_schema=None):
        calls.append(prompt)
        raise RuntimeError("malformed JSON")

    monkeypatch.setattr(report_agent, "generate_json", fail)
    with pytest.raises(RuntimeError, match="after one repair"):
        report_agent.generate_investigation_report(
            {"transaction_id": "TXN-1"}, {}, {}, 0.0, {"feature:amount": {}}
        )
    assert len(calls) == 2
