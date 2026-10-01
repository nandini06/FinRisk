from __future__ import annotations

from types import SimpleNamespace

from backend.app.llm import gemini_client
from backend.app.schemas.investigation import PlannerOutput, ReportOutput


class _FakeModels:
    def __init__(self, captured: dict) -> None:
        self.captured = captured

    def generate_content(self, *, model, contents, config):
        self.captured.update(model=model, contents=contents, config=config)
        return SimpleNamespace(
            text=(
                '{"policy_query":"policy","case_query":"case",'
                '"focus_areas":[],"suspected_risk_categories":[]}'
            ),
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                cached_content_token_count=0,
                candidates_token_count=20,
                thoughts_token_count=30,
                tool_use_prompt_token_count=0,
                total_token_count=150,
            ),
        )


class _FakeClient:
    def __init__(self, captured: dict) -> None:
        self.models = _FakeModels(captured)


def test_generate_text_sends_strict_pydantic_contract_as_json_schema(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        gemini_client,
        "_build_client",
        lambda: (_FakeClient(captured), "gemini-3.8-flash"),
    )

    gemini_client.generate_text("plan", response_schema=PlannerOutput)

    config = captured["config"]
    assert config.response_schema is None
    assert config.response_json_schema["additionalProperties"] is False
    assert set(config.response_json_schema["required"]) == {
        "policy_query",
        "case_query",
        "focus_areas",
        "suspected_risk_categories",
    }


def test_nested_strict_contract_keeps_additional_properties_protection(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        gemini_client,
        "_build_client",
        lambda: (_FakeClient(captured), "gemini-3.8-flash"),
    )

    gemini_client.generate_text("report", response_schema=ReportOutput)

    schema = captured["config"].response_json_schema
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["RiskFactor"]["additionalProperties"] is False
    assert schema["$defs"]["SimilarCaseReference"]["additionalProperties"] is False


def test_capture_usage_records_provider_reported_tokens(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        gemini_client,
        "_build_client",
        lambda: (_FakeClient(captured), "gemini-3.8-flash"),
    )

    with gemini_client.capture_usage() as usage_events:
        gemini_client.generate_text("plan", response_schema=PlannerOutput)

    assert usage_events == [
        {
            "model": "gemini-3.8-flash",
            "usage_metadata_available": True,
            "prompt_token_count": 100,
            "cached_content_token_count": 0,
            "candidates_token_count": 20,
            "thoughts_token_count": 30,
            "tool_use_prompt_token_count": 0,
            "total_token_count": 150,
        }
    ]
