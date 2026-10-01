from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PlannerOutput(StrictModel):
    policy_query: str = Field(min_length=1)
    case_query: str = Field(min_length=1)
    focus_areas: list[str]
    suspected_risk_categories: list[str]


class PolicySeverityReference(StrictModel):
    policy_id: str = Field(min_length=1)
    severity: Literal["LOW", "MEDIUM", "HIGH"]


class RiskFactor(StrictModel):
    factor: str = Field(min_length=1)
    # This is the model's evidence-based assessment, not a policy-defined value.
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    policy_severities: list[PolicySeverityReference]
    severity_rationale: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)

    @field_validator("source_ids")
    @classmethod
    def source_ids_must_be_nonempty(cls, value: list[str]) -> list[str]:
        if any(not source_id.strip() for source_id in value):
            raise ValueError("source IDs cannot be blank")
        return value


class SimilarCaseReference(StrictModel):
    case_id: str = Field(min_length=1)
    similarity_reason: str = Field(min_length=1)
    decision: str = Field(min_length=1)


class ReportOutput(StrictModel):
    transaction_id: str = Field(min_length=1)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    risk_level_rationale: str = Field(min_length=1)
    risk_level_source_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1)
    risk_factors: list[RiskFactor]
    similar_cases: list[SimilarCaseReference]
    recommended_action: str = Field(min_length=1)
    recommended_action_basis: Literal["POLICY_REQUIRED", "DISCRETIONARY_HUMAN_REVIEW"]
    recommended_action_source_ids: list[str] = Field(min_length=1)
    recommended_action_rationale: str = Field(min_length=1)

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("confidence must be finite")
        return value

    @field_validator("risk_level_source_ids", "recommended_action_source_ids")
    @classmethod
    def report_source_ids_must_be_nonempty(cls, value: list[str]) -> list[str]:
        if any(not source_id.strip() for source_id in value):
            raise ValueError("source IDs cannot be blank")
        return value


class VerifierOutput(StrictModel):
    supported: bool
    unsupported_claims: list[str]
    needs_more_evidence: bool
    suggested_query: str
    confidence_adjustment: float = Field(ge=-1.0, le=1.0)

    @field_validator("confidence_adjustment")
    @classmethod
    def adjustment_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("confidence adjustment must be finite")
        return value
