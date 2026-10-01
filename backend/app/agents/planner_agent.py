from __future__ import annotations

import json
from typing import Any

from ..llm.gemini_client import GeminiProviderError, generate_json
from ..schemas.investigation import PlannerOutput


def _generate_with_one_repair(prompt: str) -> dict[str, Any]:
    error = ""
    for attempt in range(2):
        current_prompt = prompt
        if attempt:
            current_prompt += (
                "\n\nThe previous output failed validation. Correct only the output format. "
                f"Validation feedback: {error}"
            )
        try:
            payload = generate_json(current_prompt, response_schema=PlannerOutput)
            return PlannerOutput.model_validate(payload).model_dump()
        except GeminiProviderError as exc:
            raise RuntimeError("Planner provider request failed.") from exc
        except Exception as exc:
            error = str(exc)[:500]
    raise RuntimeError(f"Planner output validation failed after one repair: {error}")


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

    return _generate_with_one_repair(prompt)
