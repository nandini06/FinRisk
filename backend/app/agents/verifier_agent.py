from __future__ import annotations

import json
from typing import Any

from ..llm.gemini_client import GeminiProviderError, generate_json
from ..schemas.investigation import VerifierOutput


def verify_investigation_report(
    draft_report: dict[str, Any],
    transaction: dict[str, Any],
    customer: dict[str, Any],
    anomaly_features: dict[str, Any],
    evidence_registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    prompt = f"""
You verify a financial-risk report against the exact supplied facts and evidence registry.
Check every material claim, including the summary and recommended action, not merely whether
citation IDs exist. A valid source ID establishes existence, not semantic support.

Return JSON with exactly:
- supported: boolean
- unsupported_claims: array of short strings
- needs_more_evidence: boolean
- suggested_query: string (empty when no useful retrieval query exists)
- confidence_adjustment: finite number from -1 to 1

A clean pass requires supported=true, no unsupported claims, and needs_more_evidence=false.

Verification rules for this dataset:
- Treat risk_level and each risk-factor severity as model assessments, distinct from the severity
  recorded in cited policy metadata. Verify every policy_severities entry exactly against the
  registry. If the assessed severity is higher, require a clear rationale grounded in cited facts
  or features and reject wording that attributes the elevation to policy.
- Do not judge agreement with an expected/reference label; no such label is an inference input.
- For recommended_action_basis=POLICY_REQUIRED, confirm that cited policy text explicitly requires
  the complete action. Investigating activity does not by itself mandate holding a transaction,
  EDD, filing, or a particular document request.
- Treat an action that goes beyond explicit policy wording as DISCRETIONARY_HUMAN_REVIEW and require
  the "Suggested for human review:" label. Reject any implication that a discretionary action is
  policy-mandated.
- customer.avg_monthly_transaction_amount is a dataset-provided synthetic monthly-average
  amount baseline, not an average individual transaction amount. The repository does not include
  its source aggregation window or generation code. A report may compare the transaction to the
  provided baseline, but must not claim that the baseline was independently reconstructed.
- If a baseline has a different or unknown meaning, reject any claim that AML-003's 10x monthly
  threshold was confirmed; the numerical deviation may remain as a separate factual observation.
- customer.country and transaction.origin_country are distinct. Reject wording that conflates
  them or presents them as a slash-separated customer origin. Require explicit role labels when
  both are mentioned.

draft_report:
{json.dumps(draft_report, indent=2, default=str)}
transaction:
{json.dumps(transaction, indent=2, default=str)}
customer:
{json.dumps(customer, indent=2, default=str)}
computed_features:
{json.dumps(anomaly_features, indent=2, default=str)}
exact_evidence_registry:
{json.dumps(evidence_registry, indent=2, default=str)}
""".strip()

    error = ""
    for attempt in range(2):
        current_prompt = prompt
        if attempt:
            current_prompt += (
                "\n\nThe previous verdict failed validation. Correct the JSON only, using the same "
                f"facts and evidence. Validation feedback: {error}"
            )
        try:
            payload = generate_json(current_prompt, response_schema=VerifierOutput)
            return VerifierOutput.model_validate(payload).model_dump()
        except GeminiProviderError as exc:
            raise RuntimeError("Verifier provider request failed.") from exc
        except Exception as exc:
            error = str(exc)[:500]
    raise RuntimeError(f"Verifier output validation failed after one repair: {error}")
