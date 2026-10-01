from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_TIMEOUT_MS = 60_000

_usage_events: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "gemini_usage_events", default=None
)


class GeminiProviderError(RuntimeError):
    pass


class StructuredOutputError(RuntimeError):
    pass


@contextmanager
def capture_usage() -> Iterator[list[dict[str, Any]]]:
    """Capture provider-reported token usage for calls in the current context."""
    events: list[dict[str, Any]] = []
    token = _usage_events.set(events)
    try:
        yield events
    finally:
        _usage_events.reset(token)


def _record_usage(response: Any, model: str) -> None:
    events = _usage_events.get()
    if events is None:
        return
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        events.append({"model": model, "usage_metadata_available": False})
        return

    def count(name: str) -> int:
        return int(getattr(usage, name, None) or 0)

    events.append(
        {
            "model": model,
            "usage_metadata_available": True,
            "prompt_token_count": count("prompt_token_count"),
            "cached_content_token_count": count("cached_content_token_count"),
            "candidates_token_count": count("candidates_token_count"),
            "thoughts_token_count": count("thoughts_token_count"),
            "tool_use_prompt_token_count": count("tool_use_prompt_token_count"),
            "total_token_count": count("total_token_count"),
        }
    )


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to your .env file.")
    return value


def _build_client() -> tuple[genai.Client, str]:
    api_key = _get_env("GEMINI_API_KEY")
    model = _get_env("GEMINI_MODEL", DEFAULT_MODEL)
    timeout_ms = int(os.getenv("GEMINI_TIMEOUT_MS", str(DEFAULT_TIMEOUT_MS)))
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms)), model


def _extract_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text and str(text).strip():
        return str(text).strip()

    candidates = getattr(response, "candidates", None) or []
    parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            maybe_text = getattr(part, "text", None)
            if maybe_text:
                parts.append(str(maybe_text))

    merged = "\n".join(parts).strip()
    if merged:
        return merged
    raise RuntimeError("Gemini returned an empty response.")


def _strip_markdown_code_fences(text: str) -> str:
    stripped = text.strip()

    # Handle fenced blocks like ```json ... ``` or ``` ... ```
    fenced_match = re.match(r"^```(?:json|JSON)?\s*([\s\S]*?)\s*```$", stripped)
    if fenced_match:
        return fenced_match.group(1).strip()

    # Fallback: remove standalone fence lines while preserving content.
    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def generate_text(prompt: str, response_schema: type[BaseModel] | None = None) -> str:
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    client, model = _build_client()
    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json" if response_schema else None,
            # Passing a strict Pydantic model through response_schema makes the
            # SDK translate JSON Schema's `additionalProperties` keyword to the
            # unsupported OpenAPI field `additional_properties`. Send the raw
            # JSON Schema instead so Gemini receives the supported keyword while
            # local Pydantic validation can continue to forbid extra fields.
            response_json_schema=(
                response_schema.model_json_schema() if response_schema else None
            ),
        )
        response = client.models.generate_content(model=model, contents=prompt, config=config)
        _record_usage(response, model)
        return _extract_text(response)
    except Exception as exc:
        raise GeminiProviderError(f"Gemini generation failed ({type(exc).__name__}).") from exc


def _extract_json_candidate(text: str) -> str:
    cleaned = _strip_markdown_code_fences(text)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]

    return cleaned


def generate_json(
    prompt: str, response_schema: type[BaseModel] | None = None
) -> dict[str, Any]:
    text = generate_text(prompt, response_schema=response_schema)
    candidate = _extract_json_candidate(text)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            "Gemini returned malformed JSON. "
            f"JSON error: {exc.msg} at line {exc.lineno}, column {exc.colno}."
        ) from exc

    if not isinstance(parsed, dict):
        raise StructuredOutputError(
            f"Gemini returned valid JSON but not a JSON object (got {type(parsed).__name__})."
        )
    return parsed
