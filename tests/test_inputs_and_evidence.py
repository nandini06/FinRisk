from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from backend.app.components import planner, report_generator, verifier
from backend.app.components.evidence import (
    build_evidence_registry,
    merge_evidence_rows,
    registry_records,
    validate_report_citations,
)
from backend.app.components.inference_inputs import build_inference_inputs
from backend.app.ml.features import extract_features


def test_answer_fields_excluded_without_mutation_and_unknown_nested_blocked():
    transaction = {
        "transaction_id": "TXN-1",
        "amount": 10,
        "risk_label": "suspicious",
        "metadata_json": {
            "merchant_category": "retail",
            "risk_score": 0.99,
            "rule_hits": "secret",
            "nested": {"gold_explanation": "leak"},
        },
    }
    customer = {"customer_id": "C-1", "kyc_risk_score": 0.2, "unknown": "leak"}
    original = deepcopy((transaction, customer))

    clean_tx, clean_customer = build_inference_inputs(transaction, customer)

    assert clean_tx == {
        "transaction_id": "TXN-1",
        "amount": 10,
        "metadata_json": {"merchant_category": "retail"},
    }
    assert clean_customer == {"customer_id": "C-1", "kyc_risk_score": 0.2}
    assert (transaction, customer) == original


def test_legacy_customer_average_is_exposed_with_monthly_semantics():
    _, clean_customer = build_inference_inputs(
        {}, {"customer_id": "C-1", "avg_transaction_amount": 1234.56}
    )

    assert clean_customer["avg_monthly_transaction_amount"] == 1234.56
    assert "avg_transaction_amount" not in clean_customer


def test_committed_transaction_ratios_use_customer_monthly_average():
    root = Path(__file__).resolve().parents[1]
    transactions = pd.read_csv(root / "data" / "transactions.csv")
    customers = pd.read_csv(root / "data" / "customers.csv")
    joined = transactions.merge(
        customers[["customer_id", "avg_monthly_transaction_amount"]],
        on="customer_id",
        validate="many_to_one",
    )

    calculated = (joined["amount"] / joined["avg_monthly_transaction_amount"]).round(2)
    assert (calculated - joined["amount_vs_customer_avg"]).abs().max() < 1e-9


@pytest.mark.parametrize(
    ("transaction_id", "feature_name", "expected"),
    [
        ("TXN-000001", "new_destination_country_flag", False),
        ("TXN-000002", "new_destination_country_flag", True),
        ("TXN-000003", "new_transaction_type_flag", False),
        ("TXN-000001", "new_transaction_type_flag", True),
    ],
)
def test_real_dataset_behavior_flags_survive_sanitization(
    transaction_id, feature_name, expected
):
    root = Path(__file__).resolve().parents[1]
    transactions = pd.read_csv(root / "data" / "transactions.csv").set_index("transaction_id")
    customers = pd.read_csv(root / "data" / "customers.csv").set_index("customer_id")
    row = transactions.loc[transaction_id]
    customer_row = customers.loc[row["customer_id"]]
    transaction = {
        "transaction_id": transaction_id,
        "customer_id": row["customer_id"],
        "amount": row["amount"],
        "transaction_type": row["transaction_type"],
        "destination_country": row["destination_country"],
        "channel": row["channel"],
        "metadata_json": {
            "country_risk_score": int(row["country_risk_score"]),
            "new_destination_country_flag": int(row["new_destination_country_flag"]),
            "new_transaction_type_flag": int(row["new_transaction_type_flag"]),
            "risk_label": row["risk_label"],
            "risk_score": row["risk_score"],
            "rule_hits": row["rule_hits"],
        },
    }
    customer = {
        "customer_id": row["customer_id"],
        "kyc_risk_score": customer_row["kyc_risk_score"],
        "avg_transaction_amount": customer_row["avg_monthly_transaction_amount"],
    }

    clean_transaction, clean_customer = build_inference_inputs(transaction, customer)
    features = extract_features(clean_transaction, clean_customer)

    assert bool(clean_transaction["metadata_json"][feature_name]) is expected
    assert features[feature_name] is expected
    assert {"risk_label", "risk_score", "rule_hits"}.isdisjoint(
        clean_transaction["metadata_json"]
    )


def _registry():
    return build_evidence_registry(
        transaction={"transaction_id": "TXN-1", "amount": 10},
        customer={"customer_id": "C-1"},
        features={"amount_vs_customer_avg": 2.0},
        anomaly_score=0.2,
        policy_evidence=[
            {
                "source_id": "policy:AML-001",
                "text": "Actual policy text",
                "metadata": {"policy_id": "AML-001"},
            }
        ],
        case_evidence=[
            {
                "source_id": "case:CASE-0001",
                "text": "Actual case text",
                "metadata": {"case_id": "CASE-0001"},
            }
        ],
    )


def test_citations_resolve_and_content_is_application_owned():
    registry = _registry()
    report = {
        "risk_level_source_ids": ["feature:amount_vs_customer_avg"],
        "risk_factors": [
            {
                "severity": "HIGH",
                "policy_severities": [{"policy_id": "AML-001", "severity": ""}],
                "source_ids": ["policy:AML-001", "feature:amount_vs_customer_avg"],
            }
        ],
        "similar_cases": [{"case_id": "CASE-0001"}],
        "recommended_action": "Suggested for human review: Review.",
        "recommended_action_basis": "DISCRETIONARY_HUMAN_REVIEW",
        "recommended_action_source_ids": ["feature:amount_vs_customer_avg"],
    }
    registry["policy:AML-001"]["metadata"]["severity"] = "HIGH"
    report["risk_factors"][0]["policy_severities"][0]["severity"] = "HIGH"
    validate_report_citations(report, registry)
    records = {item["source_id"]: item for item in registry_records(registry)}
    assert records["policy:AML-001"]["text"] == "Actual policy text"
    assert records["feature:amount_vs_customer_avg"]["value"] == 2.0


@pytest.mark.parametrize(
    "source_ids",
    [[], ["policy:AML-999"], ["url:https://invented.example"], "policy:AML-001"],
)
def test_empty_unknown_and_wrong_type_citations_rejected(source_ids):
    with pytest.raises(ValueError):
        validate_report_citations(
            {
                "risk_level_source_ids": ["feature:amount_vs_customer_avg"],
                "risk_factors": [
                    {"severity": "LOW", "policy_severities": [], "source_ids": source_ids}
                ],
                "similar_cases": [],
                "recommended_action": "Suggested for human review: Review.",
                "recommended_action_basis": "DISCRETIONARY_HUMAN_REVIEW",
                "recommended_action_source_ids": ["feature:amount_vs_customer_avg"],
            },
            _registry(),
        )


def test_real_but_unavailable_policy_is_rejected():
    with pytest.raises(ValueError, match="unavailable"):
        validate_report_citations(
            {
                "risk_level_source_ids": ["feature:amount_vs_customer_avg"],
                "risk_factors": [
                    {
                        "severity": "HIGH",
                        "policy_severities": [
                            {"policy_id": "AML-002", "severity": "HIGH"}
                        ],
                        "source_ids": ["policy:AML-002"],
                    }
                ],
                "similar_cases": [],
                "recommended_action": "Suggested for human review: Review.",
                "recommended_action_basis": "DISCRETIONARY_HUMAN_REVIEW",
                "recommended_action_source_ids": ["feature:amount_vs_customer_avg"],
            },
            _registry(),
        )


def test_different_stored_answers_produce_identical_inference_inputs():
    base = {
        "transaction_id": "TXN-1",
        "amount": 10,
        "metadata_json": {"merchant_category": "retail"},
    }
    first = deepcopy(base)
    first["metadata_json"].update(risk_label="normal", risk_score=0.1, rule_hits="")
    second = deepcopy(base)
    second["metadata_json"].update(
        risk_label="suspicious", risk_score=0.99, rule_hits="R-1"
    )
    assert build_inference_inputs(first, {}) == build_inference_inputs(second, {})


def test_merge_uses_source_id_and_retains_new_evidence():
    initial = [{"source_id": "policy:A", "similarity_score": 0.9}]
    extra = [
        {"source_id": "policy:A", "similarity_score": 0.1},
        {"source_id": "policy:B", "similarity_score": 0.2},
    ]
    assert [row["source_id"] for row in merge_evidence_rows(initial, extra)] == [
        "policy:A",
        "policy:B",
    ]


def test_captured_llm_prompts_exclude_answer_fields_and_gold_text(monkeypatch):
    captured = []
    clean_transaction, clean_customer = build_inference_inputs(
        {
            "transaction_id": "TXN-1",
            "amount": 10,
            "risk_label": "suspicious",
            "metadata_json": {
                "risk_score": 0.9,
                "rule_hits": "R-1",
                "country_risk_score": 2,
                "nested": {"gold_explanation": "DO NOT LEAK"},
            },
        },
        {"customer_id": "C-1", "kyc_risk_score": 0.2},
    )
    registry = build_evidence_registry(
        clean_transaction,
        clean_customer,
        {"amount_vs_customer_avg": 1.0},
        0.1,
        [],
        [],
    )

    responses = iter(
        [
            {
                "policy_query": "policy",
                "case_query": "case",
                "focus_areas": [],
                "suspected_risk_categories": [],
            },
            {
                "transaction_id": "TXN-1",
                "risk_level": "LOW",
                "risk_level_rationale": "Facts support a low model assessment.",
                "risk_level_source_ids": ["feature:amount_vs_customer_avg"],
                "confidence": 0.0,
                "summary": "Observed facts",
                "risk_factors": [],
                "similar_cases": [],
                "recommended_action": "Suggested for human review: No automated action.",
                "recommended_action_basis": "DISCRETIONARY_HUMAN_REVIEW",
                "recommended_action_source_ids": ["feature:amount_vs_customer_avg"],
                "recommended_action_rationale": "A human may close after reviewing the facts.",
            },
            {
                "supported": True,
                "unsupported_claims": [],
                "needs_more_evidence": False,
                "suggested_query": "",
                "confidence_adjustment": 0.0,
            },
        ]
    )

    def fake_generate(prompt, response_schema=None):
        captured.append(prompt)
        return next(responses)

    monkeypatch.setattr(planner, "generate_json", fake_generate)
    monkeypatch.setattr(report_generator, "generate_json", fake_generate)
    monkeypatch.setattr(verifier, "generate_json", fake_generate)
    features = {"amount_vs_customer_avg": 1.0}
    planner.build_investigation_plan(clean_transaction, clean_customer, features)
    report = report_generator.generate_investigation_report(
        clean_transaction, clean_customer, features, 0.1, registry
    )
    verifier.verify_investigation_report(
        report, clean_transaction, clean_customer, features, registry
    )

    prompts = "\n".join(captured)
    for prohibited in (
        '"risk_label"',
        '"risk_score"',
        '"rule_hits"',
        '"gold_explanation"',
        "DO NOT LEAK",
    ):
        assert prohibited not in prompts
    assert "dataset-provided customer monthly-average" in prompts
    assert "customer.country and transaction.origin_country are different facts" in prompts
