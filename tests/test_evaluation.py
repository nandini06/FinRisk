from copy import deepcopy
from decimal import Decimal
import json
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.evaluate as evaluate
from scripts.evaluate import (
    EstimatedCostLimitReached,
    UsageAccountingUnavailable,
    _RequestCostGuard,
    _assert_resume_compatible,
    _case_overlap_status,
    _ensure_usage_accounting_available,
    _load_case_artifacts,
    _pending_manifest_items,
    _save_case_attempt,
    _source_metadata,
    _usage_summary,
)


def _report_with_case(text):
    return {"evidence": [{"source_type": "case", "text": text}]}


def test_exact_transaction_id_is_direct_overlap():
    transaction = pd.Series(
        {"customer_id": "CUST-1", "amount": 100.0}, name="TXN-123"
    )
    assert (
        _case_overlap_status(_report_with_case("Prior TXN-123 investigation"), transaction)
        == "direct_overlap_exact_transaction_id"
    )


def test_customer_and_amount_is_only_potential_overlap():
    transaction = pd.Series(
        {"customer_id": "CUST-1", "amount": 100.0}, name="TXN-123"
    )
    assert (
        _case_overlap_status(
            _report_with_case("CUST-1 initiated a transfer of USD 100.00"), transaction
        )
        == "potential_overlap_customer_id_and_amount"
    )


def test_evaluation_source_metadata_identifies_working_tree_state():
    metadata = _source_metadata()
    assert metadata["head_revision"]
    assert metadata["working_tree_state"] in {"clean", "uncommitted_changes_present"}
    assert len(metadata["source_fingerprint_sha256"]) == 64
    assert isinstance(metadata["changed_source_files"], list)
    assert isinstance(metadata["changed_generated_files"], list)


def test_usage_summary_prices_provider_reported_tokens_including_thoughts():
    summary = _usage_summary(
        [
            {
                "usage_metadata_available": True,
                "prompt_token_count": 100,
                "cached_content_token_count": 10,
                "candidates_token_count": 20,
                "thoughts_token_count": 30,
                "tool_use_prompt_token_count": 0,
                "total_token_count": 150,
            }
        ]
    )

    assert summary["uncached_input_token_count"] == 90
    assert summary["billed_output_token_count"] == 50
    assert summary["estimated_total_cost_usd"] == pytest.approx(0.00025575)


def _attempt(status="error", attempt_number=1):
    usage_event = {
        "usage_metadata_available": True,
        "prompt_token_count": 100,
        "cached_content_token_count": 0,
        "candidates_token_count": 20,
        "thoughts_token_count": 30,
        "tool_use_prompt_token_count": 0,
        "total_token_count": 150,
    }
    usage = _usage_summary([usage_event])
    return {
        "attempt_number": attempt_number,
        "started_at": "2026-09-30T00:00:00+00:00",
        "completed_at": "2026-09-30T00:00:01+00:00",
        "case": {
            "transaction_id": "TXN-1",
            "status": status,
            "wall_clock_seconds": 1.0,
            "estimated_cost_usd": usage["estimated_total_cost_usd"],
        },
        "usage": usage,
        "usage_events": [usage_event],
        "report": None if status == "error" else {"transaction_id": "TXN-1"},
    }


def test_case_attempt_is_atomically_saved_and_retry_history_is_retained(tmp_path):
    artifacts = {}
    first = _attempt()
    _save_case_attempt(tmp_path, artifacts, "TXN-1", first)

    saved = _load_case_artifacts(tmp_path, {"TXN-1"})["TXN-1"]
    assert saved["latest"] == first
    assert saved["latest"]["case"]["wall_clock_seconds"] == 1.0
    assert saved["latest"]["usage_events"]
    assert saved["latest"]["usage"]["estimated_total_cost_usd"] > 0

    second = _attempt(status="supported", attempt_number=2)
    _save_case_attempt(tmp_path, artifacts, "TXN-1", second)
    saved = _load_case_artifacts(tmp_path, {"TXN-1"})["TXN-1"]
    assert [item["attempt_number"] for item in saved["attempts"]] == [1, 2]
    assert saved["latest"]["case"]["status"] == "supported"
    assert not list((tmp_path / "cases").glob("*.tmp"))


def test_resume_skips_recorded_errors_unless_retry_is_explicit():
    manifest = {
        "transactions": [
            {"transaction_id": "TXN-1"},
            {"transaction_id": "TXN-2"},
        ]
    }
    error_attempt = _attempt()
    artifacts = {
        "TXN-1": {
            "transaction_id": "TXN-1",
            "attempts": [error_attempt],
            "latest": error_attempt,
        }
    }

    assert [item["transaction_id"] for item in _pending_manifest_items(
        manifest, artifacts, retry_errors=False
    )] == ["TXN-2"]
    assert [item["transaction_id"] for item in _pending_manifest_items(
        manifest, artifacts, retry_errors=True
    )] == ["TXN-1", "TXN-2"]


@pytest.mark.parametrize(
    "changed_field",
    ["manifest_sha256", "source_state", "model", "generation_settings"],
)
def test_resume_refuses_changed_run_signature(changed_field):
    saved = {
        "manifest_sha256": "a" * 64,
        "source_state": {"head_revision": "abc", "source_fingerprint_sha256": "b" * 64},
        "model": "gemini-3.8-flash",
        "generation_settings": {"timeout_ms": 60000, "temperature": "provider_default"},
    }
    current = deepcopy(saved)
    current[changed_field] = "changed"

    with pytest.raises(RuntimeError, match=changed_field):
        _assert_resume_compatible(saved, current)


def test_cost_guard_checks_before_each_request_including_repairs():
    events = []
    calls = {"count": 0}
    guard = _RequestCostGuard(Decimal("0.000075"), [])
    guard.begin_case(events)

    def fake_generate(*args, **kwargs):
        calls["count"] += 1
        events.append(
            {
                "usage_metadata_available": True,
                "prompt_token_count": 100,
                "cached_content_token_count": 0,
                "candidates_token_count": 0,
                "thoughts_token_count": 0,
                "tool_use_prompt_token_count": 0,
                "total_token_count": 100,
            }
        )
        return "{}"

    guarded = guard.wrap(fake_generate)
    assert guarded("initial") == "{}"
    with pytest.raises(EstimatedCostLimitReached):
        guarded("repair")
    assert calls["count"] == 1


def test_cost_guard_stops_when_provider_usage_is_unavailable():
    events = []
    guard = _RequestCostGuard(Decimal("1.00"), [])
    guard.begin_case(events)

    def fake_generate(*args, **kwargs):
        events.append({"usage_metadata_available": False})
        return "{}"

    with pytest.raises(UsageAccountingUnavailable):
        guard.wrap(fake_generate)("request")


def test_failed_request_without_usage_leaves_non_resumable_marker():
    events = []
    guard = _RequestCostGuard(Decimal("1.00"), [])
    guard.begin_case(events)

    def failed_generate(*args, **kwargs):
        raise RuntimeError("provider failed before returning usage")

    with pytest.raises(UsageAccountingUnavailable):
        guard.wrap(failed_generate)("request")
    assert events == [
        {
            "provider_response_received": False,
            "usage_metadata_available": False,
            "accounting_error": "request_failed_without_one_usage_event",
        }
    ]
    with pytest.raises(RuntimeError, match="no further paid requests"):
        _ensure_usage_accounting_available(events)


def _mock_report(transaction_id):
    return {
        "transaction_id": transaction_id,
        "risk_level": "LOW",
        "status": "supported",
        "risk_factors": [],
        "evidence": [],
        "trace": {"extra_retrieval_performed": False},
    }


def _runner_args(output_dir, *, resume=False, retry_errors=False):
    return SimpleNamespace(
        live=True,
        confirm_paid_requests=True,
        output_dir=output_dir,
        estimated_cost_limit_usd=Decimal("1.00"),
        resume=resume,
        retry_errors=retry_errors,
    )


def test_runner_persists_stop_resumes_and_retries_errors_explicitly(tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    initial_calls = []
    manifest_ids = {
        item["transaction_id"]
        for item in json.loads(
            (evaluate.ROOT_DIR / "evaluation" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )["transactions"]
    }

    def stop_on_second_case(transaction_id):
        initial_calls.append(transaction_id)
        if len(initial_calls) == 2:
            raise EstimatedCostLimitReached(
                "estimated_cost_limit_reached", "offline simulated cost stop"
            )
        return _mock_report(transaction_id)

    monkeypatch.setattr(evaluate, "parse_args", lambda: _runner_args(output_dir))
    monkeypatch.setattr(evaluate, "run_investigation", stop_on_second_case)
    with pytest.raises(SystemExit, match="stopped safely"):
        evaluate.main()

    artifacts = _load_case_artifacts(output_dir, set(initial_calls))
    assert len(artifacts) == 2
    assert artifacts[initial_calls[0]]["latest"]["report"] is not None
    assert artifacts[initial_calls[1]]["latest"]["case"]["status"] == "error"
    assert (output_dir / "results.json").is_file()

    resumed_calls = []
    monkeypatch.setattr(
        evaluate,
        "parse_args",
        lambda: _runner_args(output_dir, resume=True),
    )
    monkeypatch.setattr(
        evaluate,
        "run_investigation",
        lambda transaction_id: resumed_calls.append(transaction_id)
        or _mock_report(transaction_id),
    )
    evaluate.main()
    assert initial_calls[0] not in resumed_calls
    assert initial_calls[1] not in resumed_calls
    assert len(_load_case_artifacts(output_dir, manifest_ids)) == 15

    retry_calls = []
    monkeypatch.setattr(
        evaluate,
        "parse_args",
        lambda: _runner_args(output_dir, resume=True, retry_errors=True),
    )
    monkeypatch.setattr(
        evaluate,
        "run_investigation",
        lambda transaction_id: retry_calls.append(transaction_id)
        or _mock_report(transaction_id),
    )
    evaluate.main()
    assert retry_calls == [initial_calls[1]]
    retried = _load_case_artifacts(output_dir, manifest_ids)[initial_calls[1]]
    assert len(retried["attempts"]) == 2
    assert retried["latest"]["case"]["status"] == "supported"
