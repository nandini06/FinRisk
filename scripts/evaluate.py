from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.components.workflow import run_investigation  # noqa: E402
from backend.app.llm import gemini_client  # noqa: E402
from backend.app.llm.gemini_client import capture_usage  # noqa: E402

LABEL_MAP = {"normal": "LOW", "review": "MEDIUM", "suspicious": "HIGH"}
GEMINI_38_STANDARD_INPUT_USD_PER_MILLION = 0.75
GEMINI_38_STANDARD_CACHED_INPUT_USD_PER_MILLION = 0.075
GEMINI_38_STANDARD_OUTPUT_USD_PER_MILLION = 3.75
PRICING_EFFECTIVE_NOTE = "Gemini 3.8 Flash standard introductory pricing through 2026-12-31"
RUN_STATE_FILENAME = "run_state.json"


class EvaluationStop(BaseException):
    """Stop the evaluation without being swallowed by workflow repair loops."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class EstimatedCostLimitReached(EvaluationStop):
    pass


class UsageAccountingUnavailable(EvaluationStop):
    pass


def _positive_cost_limit(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("cost limit must be a decimal number") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise argparse.ArgumentTypeError("cost limit must be finite and greater than zero")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Run the fixed FinRisk live evaluation sample.")
    parser.add_argument("--live", action="store_true", help="Required: execute the real workflow.")
    parser.add_argument(
        "--confirm-paid-requests",
        action="store_true",
        help="Acknowledge that Gemini calls may incur cost.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "evaluation" / "results")
    parser.add_argument(
        "--estimated-cost-limit-usd",
        type=_positive_cost_limit,
        required=True,
        help=(
            "Estimated accumulated Gemini cost at which no subsequent model request is started. "
            "This is not a guaranteed billing cap; leave budget headroom."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible run from atomically saved per-case artifacts.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="With --resume, retry cases whose latest recorded attempt is an error.",
    )
    return parser.parse_args()


def _source_metadata(*, excluded_roots: tuple[Path, ...] = ()) -> dict:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
        raw_status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=ROOT_DIR,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {
            "head_revision": "unknown",
            "working_tree_state": "unknown",
            "source_fingerprint_sha256": "unknown",
            "changed_source_files": [],
            "changed_generated_files": [],
        }

    excluded_relative = []
    for excluded in excluded_roots:
        try:
            excluded_relative.append(excluded.resolve().relative_to(ROOT_DIR.resolve()).as_posix())
        except ValueError:
            continue

    entries = []
    for raw_entry in raw_status.split(b"\0"):
        if not raw_entry:
            continue
        decoded = raw_entry.decode("utf-8", errors="replace")
        if len(decoded) < 4:
            continue
        path = decoded[3:].replace("\\", "/")
        if any(path == root or path.startswith(f"{root}/") for root in excluded_relative):
            continue
        entries.append({"status": decoded[:2], "path": path})

    def is_generated(path: str) -> bool:
        return (
            path.startswith("chroma_db/")
            or "/__pycache__/" in f"/{path}"
            or path.endswith((".pyc", ".pyo"))
            or path.startswith(".pytest_cache/")
            or path.startswith("evaluation/results/")
        )

    generated = [entry for entry in entries if is_generated(entry["path"])]
    source = [entry for entry in entries if not is_generated(entry["path"])]
    fingerprint = hashlib.sha256()
    for entry in sorted(source, key=lambda item: item["path"]):
        fingerprint.update(f"{entry['status']} {entry['path']}\0".encode())
        path = ROOT_DIR / entry["path"]
        if path.is_file():
            fingerprint.update(path.read_bytes())

    return {
        "head_revision": revision,
        "working_tree_state": "uncommitted_changes_present" if entries else "clean",
        "source_fingerprint_sha256": fingerprint.hexdigest(),
        "changed_source_files": source,
        "changed_generated_files": generated,
    }


def _citation_stats(report: dict) -> tuple[int, int]:
    available = {item.get("source_id") for item in report.get("evidence", [])}
    citations = [
        source_id
        for factor in report.get("risk_factors", [])
        for source_id in factor.get("source_ids", [])
    ]
    return sum(source_id in available for source_id in citations), len(citations)


def _case_overlap_status(report: dict, transaction: pd.Series) -> str:
    transaction_id = str(transaction.name).lower()
    customer_id = str(transaction["customer_id"]).lower()
    amount = float(transaction["amount"])
    amount_tokens = {f"{amount:.2f}", f"{amount:g}"}
    case_texts = [
        str(item.get("text") or "").lower()
        for item in report.get("evidence", [])
        if item.get("source_type") == "case"
    ]
    if any(transaction_id in text for text in case_texts):
        return "direct_overlap_exact_transaction_id"
    if any(
        customer_id in text and any(token in text for token in amount_tokens)
        for text in case_texts
    ):
        return "potential_overlap_customer_id_and_amount"
    return "uncertain_provenance_no_overlap_detected"


def _safe_error(exc: Exception) -> str:
    message = str(exc)[:500]
    message = re.sub(r"([a-z][a-z0-9+.-]*://)[^\s/@]+(?::[^\s/@]*)?@", r"\1<redacted>@", message)
    message = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "<redacted-api-key>", message)
    return f"{type(exc).__name__}: {message[:240]}"


def _usage_summary(events: list[dict]) -> dict:
    totals = {
        "provider_response_count": sum(
            bool(event.get("provider_response_received", True)) for event in events
        ),
        "responses_with_usage_metadata": sum(
            bool(event.get("usage_metadata_available")) for event in events
        ),
        "prompt_token_count": sum(int(event.get("prompt_token_count", 0)) for event in events),
        "cached_content_token_count": sum(
            int(event.get("cached_content_token_count", 0)) for event in events
        ),
        "candidates_token_count": sum(
            int(event.get("candidates_token_count", 0)) for event in events
        ),
        "thoughts_token_count": sum(
            int(event.get("thoughts_token_count", 0)) for event in events
        ),
        "tool_use_prompt_token_count": sum(
            int(event.get("tool_use_prompt_token_count", 0)) for event in events
        ),
        "total_token_count": sum(int(event.get("total_token_count", 0)) for event in events),
    }
    uncached_input = max(
        totals["prompt_token_count"] - totals["cached_content_token_count"], 0
    )
    billed_output = totals["candidates_token_count"] + totals["thoughts_token_count"]
    input_cost = (
        uncached_input * GEMINI_38_STANDARD_INPUT_USD_PER_MILLION / 1_000_000
    )
    cached_input_cost = (
        totals["cached_content_token_count"]
        * GEMINI_38_STANDARD_CACHED_INPUT_USD_PER_MILLION
        / 1_000_000
    )
    output_cost = (
        billed_output * GEMINI_38_STANDARD_OUTPUT_USD_PER_MILLION / 1_000_000
    )
    totals.update(
        uncached_input_token_count=uncached_input,
        billed_output_token_count=billed_output,
        estimated_input_cost_usd=round(input_cost, 8),
        estimated_cached_input_cost_usd=round(cached_input_cost, 8),
        estimated_output_cost_usd=round(output_cost, 8),
        estimated_total_cost_usd=round(input_cost + cached_input_cost + output_cost, 8),
    )
    return totals


def _estimated_cost_decimal(events: list[dict[str, Any]]) -> Decimal:
    prompt = sum(Decimal(int(event.get("prompt_token_count", 0))) for event in events)
    cached = sum(
        Decimal(int(event.get("cached_content_token_count", 0))) for event in events
    )
    output = sum(
        Decimal(int(event.get("candidates_token_count", 0)))
        + Decimal(int(event.get("thoughts_token_count", 0)))
        for event in events
    )
    uncached = max(prompt - cached, Decimal(0))
    million = Decimal(1_000_000)
    return (
        uncached * Decimal(str(GEMINI_38_STANDARD_INPUT_USD_PER_MILLION)) / million
        + cached
        * Decimal(str(GEMINI_38_STANDARD_CACHED_INPUT_USD_PER_MILLION))
        / million
        + output * Decimal(str(GEMINI_38_STANDARD_OUTPUT_USD_PER_MILLION)) / million
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, content: str) -> None:
    """Durably replace one file without exposing a partially written destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2))


def _generation_settings() -> dict[str, Any]:
    try:
        sdk_version = importlib.metadata.version("google-genai")
    except importlib.metadata.PackageNotFoundError:
        sdk_version = "not-installed"
    return {
        "model": os.getenv("GEMINI_MODEL", gemini_client.DEFAULT_MODEL),
        "timeout_ms": int(
            os.getenv("GEMINI_TIMEOUT_MS", str(gemini_client.DEFAULT_TIMEOUT_MS))
        ),
        "api_method": "client.models.generate_content",
        "structured_response_mime_type": "application/json",
        "structured_schema_transport": "response_json_schema",
        "temperature": "provider_default",
        "top_p": "provider_default",
        "top_k": "provider_default",
        "max_output_tokens": "provider_default",
        "seed": "provider_default",
        "thinking_configuration": "provider_default",
        "sdk_retry_options": None,
        "google_genai_version": sdk_version,
        "structured_output_repair_limit": 1,
        "investigation_retrieval_retry_limit": 1,
    }


def _run_signature(manifest_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    source = _source_metadata(excluded_roots=(output_dir,) if output_dir else ())
    settings = _generation_settings()
    return {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_state": {
            "head_revision": source["head_revision"],
            "source_fingerprint_sha256": source["source_fingerprint_sha256"],
        },
        "model": settings["model"],
        "generation_settings": settings,
    }


def _assert_resume_compatible(saved: dict[str, Any], current: dict[str, Any]) -> None:
    fields = ("manifest_sha256", "source_state", "model", "generation_settings")
    mismatches = [field for field in fields if saved.get(field) != current.get(field)]
    if mismatches:
        raise RuntimeError(
            "Resume refused because the saved run differs from the current "
            + ", ".join(mismatches)
            + ". Start a new output directory for a new run."
        )


def _latest_attempt(artifact: dict[str, Any]) -> dict[str, Any]:
    attempts = artifact.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise RuntimeError("Case artifact has no completed attempts.")
    latest = artifact.get("latest")
    if latest != attempts[-1]:
        raise RuntimeError("Case artifact latest attempt does not match its attempt history.")
    return latest


def _load_case_artifacts(
    output_dir: Path, allowed_transaction_ids: set[str]
) -> dict[str, dict[str, Any]]:
    cases_dir = output_dir / "cases"
    if not cases_dir.exists():
        return {}
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(cases_dir.glob("*.json")):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot resume from invalid case artifact {path.name}.") from exc
        transaction_id = str(artifact.get("transaction_id") or "")
        if transaction_id not in allowed_transaction_ids or path.stem != transaction_id:
            raise RuntimeError(f"Unexpected case artifact {path.name}; resume refused.")
        _latest_attempt(artifact)
        artifacts[transaction_id] = artifact
    return artifacts


def _all_usage_events(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for artifact in artifacts.values():
        for attempt in artifact["attempts"]:
            events.extend(deepcopy(attempt.get("usage_events") or []))
    return events


def _ensure_usage_accounting_available(events: list[dict[str, Any]]) -> None:
    if any(not event.get("usage_metadata_available") for event in events):
        raise RuntimeError(
            "Usage accounting is unavailable in a saved attempt; no further paid requests "
            "may be started from this run."
        )


class _RequestCostGuard:
    def __init__(self, limit_usd: Decimal, prior_events: list[dict[str, Any]]):
        self.limit_usd = limit_usd
        self.prior_events = deepcopy(prior_events)
        self.current_events: list[dict[str, Any]] | None = None

    def begin_case(self, events: list[dict[str, Any]]) -> None:
        if self.current_events is not None:
            raise RuntimeError("A usage-capture case is already active.")
        self.current_events = events

    def finish_case(self) -> None:
        if self.current_events is None:
            raise RuntimeError("No usage-capture case is active.")
        self.prior_events.extend(deepcopy(self.current_events))
        self.current_events = None

    def events(self) -> list[dict[str, Any]]:
        return self.prior_events + list(self.current_events or [])

    def before_request(self) -> None:
        if self.current_events is None:
            raise RuntimeError("Model request occurred outside evaluation usage capture.")
        if any(not event.get("usage_metadata_available") for event in self.events()):
            raise UsageAccountingUnavailable(
                "usage_accounting_unavailable",
                "Provider usage metadata is unavailable; stopping before another paid request.",
            )
        estimated = _estimated_cost_decimal(self.events())
        if estimated >= self.limit_usd:
            raise EstimatedCostLimitReached(
                "estimated_cost_limit_reached",
                (
                    f"Estimated accumulated cost ${estimated} has reached the configured "
                    f"${self.limit_usd} threshold. This estimate is not a guaranteed billing cap."
                ),
            )

    def wrap(self, original):
        def guarded_generate_text(*args, **kwargs):
            self.before_request()
            assert self.current_events is not None
            event_count_before = len(self.current_events)
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                new_events = self.current_events[event_count_before:]
                if len(new_events) != 1 or not new_events[0].get(
                    "usage_metadata_available"
                ):
                    if not new_events or len(new_events) != 1:
                        self.current_events.append(
                            {
                                "provider_response_received": False,
                                "usage_metadata_available": False,
                                "accounting_error": "request_failed_without_one_usage_event",
                            }
                        )
                    raise UsageAccountingUnavailable(
                        "usage_accounting_unavailable",
                        "A model request completed without usable provider token accounting; "
                        "the evaluation has stopped.",
                    ) from exc
                raise
            new_events = self.current_events[event_count_before:]
            if len(new_events) != 1 or not new_events[0].get("usage_metadata_available"):
                if not new_events or len(new_events) != 1:
                    self.current_events.append(
                        {
                            "provider_response_received": False,
                            "usage_metadata_available": False,
                            "accounting_error": "response_missing_one_usage_event",
                        }
                    )
                raise UsageAccountingUnavailable(
                    "usage_accounting_unavailable",
                    "A model response lacked usable provider token accounting; the evaluation "
                    "has stopped.",
                )
            return result

        return guarded_generate_text


@contextmanager
def _guard_model_requests(guard: _RequestCostGuard) -> Iterator[None]:
    original = gemini_client.generate_text
    gemini_client.generate_text = guard.wrap(original)
    try:
        yield
    finally:
        gemini_client.generate_text = original


def _validate_manifest(manifest: dict) -> None:
    tx = pd.read_csv(ROOT_DIR / "data" / "transactions.csv", usecols=["transaction_id", "risk_label"])
    refs = pd.read_csv(
        ROOT_DIR / "data" / "investigation_examples.csv",
        usecols=["transaction_id", "expected_risk_level"],
    )
    truth = tx.merge(refs, on="transaction_id").set_index("transaction_id")
    conflicts = []
    for item in manifest["transactions"]:
        transaction_id = item["transaction_id"]
        row = truth.loc[transaction_id]
        mapped = LABEL_MAP.get(row["risk_label"])
        if mapped != row["expected_risk_level"] or mapped != item["expected_risk_level"]:
            conflicts.append(transaction_id)
    if conflicts:
        raise RuntimeError(f"Reference-label conflicts in manifest: {conflicts}")


def _build_payload(
    manifest: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    run_state: dict[str, Any],
) -> dict[str, Any]:
    ordered_ids = [item["transaction_id"] for item in manifest["transactions"]]
    latest_attempts = [
        _latest_attempt(artifacts[transaction_id])
        for transaction_id in ordered_ids
        if transaction_id in artifacts
    ]
    cases = [deepcopy(attempt["case"]) for attempt in latest_attempts]
    generated_reports = {
        attempt["case"]["transaction_id"]: deepcopy(attempt["report"])
        for attempt in latest_attempts
        if attempt.get("report") is not None
    }
    total = len(ordered_ids)
    completed = len(cases)
    valid = [row for row in cases if row["status"] != "error"]
    matches = sum(bool(row["agreement"]) for row in valid)
    citations = sum(int(row["citation_count"]) for row in cases)
    valid_citations = sum(int(row["citation_valid_count"]) for row in cases)
    supported = sum(row["status"] == "supported" for row in cases)
    unresolved = sum(row["status"] == "needs_review" for row in cases)
    corrected = sum(bool(row["correction_occurred"]) for row in cases)
    failures = sum(row["status"] == "error" for row in cases)
    aggregate_usage = _usage_summary(_all_usage_events(artifacts))
    aggregate = {
        "selected_cases": total,
        "completed_cases": completed,
        "pending_cases": total - completed,
        "status_counts": dict(Counter(row["status"] for row in cases)),
        "valid_report_agreement": f"{matches}/{len(valid)}" if valid else "0/0",
        "all_selected_agreement": f"{matches}/{total}",
        "citation_validity": f"{valid_citations}/{citations}" if citations else "0/0",
        "report_validation_success": f"{len(valid)}/{total}",
        "verification_pass_rate": f"{supported}/{total}",
        "unresolved_rate": f"{unresolved}/{total}",
        "correction_frequency": f"{corrected}/{total}",
        "failures": f"{failures}/{total}",
        "usage": aggregate_usage,
    }
    settings = run_state["signature"]["generation_settings"]
    return {
        "run_at": run_state["created_at"],
        "updated_at": run_state["updated_at"],
        "mode": "live",
        "run_status": run_state["status"],
        "stop_reason": run_state.get("stop_reason", ""),
        "model": run_state["signature"]["model"],
        "configuration": {
            **settings,
            "estimated_cost_limit_usd": run_state["estimated_cost_limit_usd"],
            "cost_limit_warning": (
                "Estimated stop only, not a guaranteed billing cap; budget headroom is required."
            ),
            "pricing": {
                "note": PRICING_EFFECTIVE_NOTE,
                "standard_input_usd_per_million_tokens": GEMINI_38_STANDARD_INPUT_USD_PER_MILLION,
                "standard_cached_input_usd_per_million_tokens": GEMINI_38_STANDARD_CACHED_INPUT_USD_PER_MILLION,
                "standard_output_including_thinking_usd_per_million_tokens": GEMINI_38_STANDARD_OUTPUT_USD_PER_MILLION,
                "cost_method": "provider-reported prompt/cached/candidate/thinking tokens",
            },
        },
        "source": run_state["source_metadata"],
        "signature": run_state["signature"],
        "manifest_description": manifest["description"],
        "cases": cases,
        "generated_reports": generated_reports,
        "case_results": deepcopy(artifacts),
        "aggregate": aggregate,
    }


def _write_outputs(output_dir: Path, payload: dict, *, final: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / "results.json", payload)
    fields = list(payload["cases"][0]) if payload["cases"] else []
    csv_buffer = io.StringIO(newline="")
    if fields:
        writer = csv.DictWriter(csv_buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["cases"])
    _atomic_write_text(output_dir / "results.csv", csv_buffer.getvalue())
    summary = payload["aggregate"]
    lines = [
        "# FinRisk live evaluation summary",
        "",
        "This fixed synthetic sample is deliberately selected and is not representative population accuracy.",
        "Synthetic reference labels are not independently validated truth.",
        "",
        f"- Run status: {payload['run_status']}",
        f"- Completed cases: {summary['completed_cases']}/{summary['selected_cases']}",
        f"- Valid-report agreement: {summary['valid_report_agreement']}",
        f"- All-selected agreement (failures count as nonmatches): {summary['all_selected_agreement']}",
        f"- Status counts: {summary['status_counts']}",
        f"- Citation validity: {summary['citation_validity']}",
        f"- Report-validation success: {summary['report_validation_success']}",
        f"- Final verification pass rate: {summary['verification_pass_rate']}",
        f"- Unresolved rate: {summary['unresolved_rate']}",
        f"- Correction frequency: {summary['correction_frequency']}",
        f"- Failures: {summary['failures']}",
        f"- Provider responses with token metadata: {summary['usage']['responses_with_usage_metadata']}",
        f"- Prompt tokens: {summary['usage']['prompt_token_count']}",
        f"- Candidate output tokens: {summary['usage']['candidates_token_count']}",
        f"- Thinking tokens: {summary['usage']['thoughts_token_count']}",
        f"- Estimated Gemini cost (USD): ${summary['usage']['estimated_total_cost_usd']:.8f}",
        f"- Configured estimated-cost stop (USD): ${payload['configuration']['estimated_cost_limit_usd']}",
        "",
        "The cost threshold is an estimated stop, not a guaranteed billing cap; budget headroom is required.",
        "Verifier approval is not a measured factual-accuracy score.",
    ]
    _atomic_write_text(output_dir / "SUMMARY.md", "\n".join(lines))

    review_path = output_dir / "manual_review.csv"
    if final and not review_path.exists():
        review_fields = [
            "transaction_id",
            "factual_consistency",
            "citation_existence",
            "cited_sources_support_claims",
            "unsupported_claims",
            "action_appropriateness",
            "reviewer_notes",
            "review_status",
        ]
        review_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(review_buffer, fieldnames=review_fields)
        writer.writeheader()
        for case in payload["cases"][:5]:
            writer.writerow(
                {"transaction_id": case["transaction_id"], "review_status": "UNREVIEWED"}
            )
        _atomic_write_text(review_path, review_buffer.getvalue())


def _save_case_attempt(
    output_dir: Path,
    artifacts: dict[str, dict[str, Any]],
    transaction_id: str,
    attempt: dict[str, Any],
) -> None:
    existing = deepcopy(
        artifacts.get(transaction_id, {"transaction_id": transaction_id, "attempts": []})
    )
    existing["attempts"].append(deepcopy(attempt))
    existing["latest"] = deepcopy(attempt)
    _atomic_write_json(output_dir / "cases" / f"{transaction_id}.json", existing)
    artifacts[transaction_id] = existing


def _pending_manifest_items(
    manifest: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    *,
    retry_errors: bool,
) -> list[dict[str, Any]]:
    pending = []
    for item in manifest["transactions"]:
        artifact = artifacts.get(item["transaction_id"])
        if artifact is None:
            pending.append(item)
        elif retry_errors and _latest_attempt(artifact)["case"]["status"] == "error":
            pending.append(item)
    return pending


def _write_run_state(output_dir: Path, state: dict[str, Any]) -> None:
    _atomic_write_json(output_dir / RUN_STATE_FILENAME, state)


def main() -> None:
    args = parse_args()
    if args.retry_errors and not args.resume:
        raise SystemExit("--retry-errors requires --resume.")
    if not args.live or not args.confirm_paid_requests:
        raise SystemExit(
            "Live evaluation not started. Pass both --live and --confirm-paid-requests after "
            "checking local services, credentials, and API cost authorization."
        )
    manifest_path = ROOT_DIR / "evaluation" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    output_dir = args.output_dir.resolve()
    allowed_ids = {item["transaction_id"] for item in manifest["transactions"]}
    current_signature = _run_signature(manifest_path, output_dir)
    state_path = output_dir / RUN_STATE_FILENAME

    if args.resume:
        if not state_path.is_file():
            raise SystemExit(f"Resume refused: {state_path} does not exist.")
        try:
            run_state = json.loads(state_path.read_text(encoding="utf-8"))
            _assert_resume_compatible(run_state["signature"], current_signature)
        except (OSError, json.JSONDecodeError, KeyError, RuntimeError) as exc:
            raise SystemExit(str(exc)) from exc
        artifacts = _load_case_artifacts(output_dir, allowed_ids)
        try:
            _ensure_usage_accounting_available(_all_usage_events(artifacts))
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        run_state["status"] = "running"
        run_state["stop_reason"] = ""
        run_state["estimated_cost_limit_usd"] = float(args.estimated_cost_limit_usd)
        run_state["updated_at"] = _utc_now()
        _write_run_state(output_dir, run_state)
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise SystemExit(
                f"New run refused because {output_dir} is not empty. Use --resume only for a "
                "compatible saved run, or choose a new output directory."
            )
        source_metadata = _source_metadata(excluded_roots=(output_dir,))
        run_state = {
            "version": 1,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "status": "running",
            "stop_reason": "",
            "estimated_cost_limit_usd": float(args.estimated_cost_limit_usd),
            "signature": current_signature,
            "source_metadata": source_metadata,
            "completed_cases": 0,
        }
        artifacts = {}
        _write_run_state(output_dir, run_state)

    transaction_facts = pd.read_csv(
        ROOT_DIR / "data" / "transactions.csv",
        usecols=["transaction_id", "customer_id", "amount"],
    ).set_index("transaction_id")
    pending = _pending_manifest_items(
        manifest, artifacts, retry_errors=bool(args.retry_errors)
    )
    prior_usage_events = _all_usage_events(artifacts)
    prior_estimated_cost = _estimated_cost_decimal(prior_usage_events)
    if pending and prior_estimated_cost >= args.estimated_cost_limit_usd:
        run_state["status"] = "stopped"
        run_state["stop_reason"] = "estimated_cost_limit_reached"
        run_state["completed_cases"] = len(artifacts)
        run_state["updated_at"] = _utc_now()
        _write_run_state(output_dir, run_state)
        payload = _build_payload(manifest, artifacts, run_state)
        _write_outputs(output_dir, payload)
        raise SystemExit(
            "Evaluation stopped safely before another case: accumulated estimated cost "
            f"${prior_estimated_cost} has reached the configured "
            f"${args.estimated_cost_limit_usd} threshold. Increase the threshold only with "
            "additional authorized budget headroom."
        )

    guard = _RequestCostGuard(args.estimated_cost_limit_usd, prior_usage_events)
    stopped: EvaluationStop | None = None

    with _guard_model_requests(guard):
        for item in pending:
            started = time.perf_counter()
            started_at = _utc_now()
            report = None
            transaction_id = item["transaction_id"]
            attempt_number = len(artifacts.get(transaction_id, {}).get("attempts", [])) + 1
            row = {
                "transaction_id": transaction_id,
                "expected_risk_level": item["expected_risk_level"],
                "attempt_number": attempt_number,
                "predicted_risk_level": "",
                "status": "error",
                "citation_valid_count": 0,
                "citation_count": 0,
                "correction_occurred": False,
                "historical_case_overlap": "not_checked_due_to_error",
                "agreement": False,
                "error": "",
                "stop_reason": "",
            }
            with capture_usage() as usage_events:
                guard.begin_case(usage_events)
                try:
                    report = run_investigation(transaction_id)
                    valid_citations, citation_count = _citation_stats(report)
                    row.update(
                        predicted_risk_level=report.get("risk_level"),
                        status=report.get("status"),
                        citation_valid_count=valid_citations,
                        citation_count=citation_count,
                        correction_occurred=bool(
                            report.get("trace", {}).get("extra_retrieval_performed")
                        ),
                        historical_case_overlap=_case_overlap_status(
                            report, transaction_facts.loc[transaction_id]
                        ),
                        agreement=report.get("risk_level") == item["expected_risk_level"],
                    )
                except EvaluationStop as exc:
                    stopped = exc
                    row["error"] = f"{exc.code}: {exc.message}"
                    row["stop_reason"] = exc.code
                except Exception as exc:
                    row["error"] = _safe_error(exc)

            usage = _usage_summary(usage_events)
            row.update(
                provider_response_count=usage["provider_response_count"],
                responses_with_usage_metadata=usage["responses_with_usage_metadata"],
                prompt_token_count=usage["prompt_token_count"],
                cached_content_token_count=usage["cached_content_token_count"],
                candidates_token_count=usage["candidates_token_count"],
                thoughts_token_count=usage["thoughts_token_count"],
                total_token_count=usage["total_token_count"],
                estimated_cost_usd=usage["estimated_total_cost_usd"],
                wall_clock_seconds=round(time.perf_counter() - started, 3),
            )
            attempt = {
                "attempt_number": attempt_number,
                "started_at": started_at,
                "completed_at": _utc_now(),
                "case": deepcopy(row),
                "usage": usage,
                "usage_events": deepcopy(usage_events),
                "report": deepcopy(report),
            }
            _save_case_attempt(output_dir, artifacts, transaction_id, attempt)
            guard.finish_case()
            run_state["completed_cases"] = len(artifacts)
            run_state["updated_at"] = _utc_now()
            if stopped is not None:
                run_state["status"] = "stopped"
                run_state["stop_reason"] = stopped.code
            else:
                run_state["status"] = "running"
                run_state["stop_reason"] = ""
            _write_run_state(output_dir, run_state)
            _write_outputs(output_dir, _build_payload(manifest, artifacts, run_state))
            if stopped is not None:
                break

    if stopped is None:
        run_state["status"] = "completed"
        run_state["stop_reason"] = ""
    run_state["completed_cases"] = len(artifacts)
    run_state["updated_at"] = _utc_now()
    _write_run_state(output_dir, run_state)
    payload = _build_payload(manifest, artifacts, run_state)
    _write_outputs(output_dir, payload, final=run_state["status"] == "completed")
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"Results written to {output_dir}")
    if stopped is not None:
        raise SystemExit(f"Evaluation stopped safely: {stopped.message}")


if __name__ == "__main__":
    main()
