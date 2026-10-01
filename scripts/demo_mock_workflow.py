from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.agents import workflow  # noqa: E402
from backend.app.db.models import Customer, Transaction  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Deterministic mocked workflow demonstration.")
    parser.add_argument(
        "--scenario", choices=["supported", "corrected", "unresolved", "all"], default="all"
    )
    return parser.parse_args()


def _records():
    return (
        Transaction(
            transaction_id="TXN-MOCK-001",
            customer_id="CUST-MOCK",
            amount=15000,
            currency="USD",
            transaction_type="WIRE",
            destination_country="Exampleland",
            metadata_json={"country_risk_score": 3},
        ),
        Customer(
            customer_id="CUST-MOCK",
            country="US",
            kyc_level="Retail",
            kyc_risk_score=0.3,
            avg_transaction_amount=1000,
        ),
    )


def _report(pass_number):
    return {
        "transaction_id": "TXN-MOCK-001",
        "risk_level": "MEDIUM",
        "risk_level_rationale": "The amount deviation warrants model-assessed review.",
        "risk_level_source_ids": ["feature:amount_vs_customer_avg"],
        "confidence": 0.55 + (0.05 * (pass_number - 1)),
        "summary": "Mock report grounded in the supplied amount feature.",
        "risk_factors": [
            {
                "factor": "Amount deviation",
                "severity": "MEDIUM",
                "policy_severities": [],
                "severity_rationale": "The supplied feature shows a substantial amount deviation.",
                "explanation": "The amount is above the customer average.",
                "source_ids": ["feature:amount_vs_customer_avg"],
            }
        ],
        "similar_cases": [],
        "recommended_action": "Suggested for human review: review the transaction.",
        "recommended_action_basis": "DISCRETIONARY_HUMAN_REVIEW",
        "recommended_action_source_ids": ["feature:amount_vs_customer_avg"],
        "recommended_action_rationale": "No supplied policy explicitly mandates an action.",
    }


def _verdict(supported, claims, more, query, adjustment=0.0):
    return {
        "supported": supported,
        "unsupported_claims": claims,
        "needs_more_evidence": more,
        "suggested_query": query,
        "confidence_adjustment": adjustment,
    }


def run_scenario(name):
    verdicts = {
        "supported": [_verdict(True, [], False, "", 0.05)],
        "corrected": [
            _verdict(False, ["Policy support is weak"], True, "wire review policy"),
            _verdict(True, [], False, "", 0.02),
        ],
        "unresolved": [
            _verdict(False, ["Purpose is unknown"], True, "transaction purpose evidence"),
            _verdict(False, ["Purpose remains unknown"], False, "", -0.1),
        ],
    }[name]
    report_pass = {"count": 0}
    policy_pass = {"count": 0}

    def report(*args, **kwargs):
        report_pass["count"] += 1
        return _report(report_pass["count"])

    def policy(*args, **kwargs):
        policy_pass["count"] += 1
        identifier = f"MOCK-{policy_pass['count']}"
        return [
            {
                "source_id": f"policy:{identifier}",
                "text": "Mock policy evidence.",
                "metadata": {"policy_id": identifier},
            }
        ]

    verdict_iter = iter(verdicts)
    with (
        patch.object(workflow, "get_transaction_with_customer", return_value=_records()),
        patch.object(
            workflow,
            "run_anomaly_analysis",
            return_value={
                "features": {"amount_vs_customer_avg": 15.0},
                "anomaly_score": 0.35,
                "top_contributing_features": [],
            },
        ),
        patch.object(
            workflow,
            "build_investigation_plan",
            return_value={"policy_query": "wire policy", "case_query": "similar wire"},
        ),
        patch.object(workflow, "retrieve_policy_evidence", side_effect=policy),
        patch.object(workflow, "retrieve_similar_cases", return_value=[]),
        patch.object(workflow, "generate_investigation_report", side_effect=report),
        patch.object(
            workflow, "verify_investigation_report", side_effect=lambda *a, **k: next(verdict_iter)
        ),
        patch.object(workflow, "create_investigation_report"),
    ):
        result = workflow._run_investigation(object(), "TXN-MOCK-001")
    return {"mode": "DETERMINISTIC_MOCK", "scenario": name, "report": result}


def main():
    requested = parse_args().scenario
    scenarios = ["supported", "corrected", "unresolved"] if requested == "all" else [requested]
    print(json.dumps([run_scenario(name) for name in scenarios], indent=2))


if __name__ == "__main__":
    main()
