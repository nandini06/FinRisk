from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from google import genai

load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to your .env file.")
    return value


def _build_client() -> tuple[genai.Client, str]:
    api_key = _get_env("GEMINI_API_KEY")
    model = _get_env("GEMINI_MODEL", DEFAULT_MODEL)
    return genai.Client(api_key=api_key), model


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


def generate_text(prompt: str) -> str:
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    client, model = _build_client()
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        return _extract_text(response)
    except Exception as exc:
        raise RuntimeError(f"Gemini text generation failed: {exc}") from exc


def _extract_json_candidate(text: str) -> str:
    cleaned = _strip_markdown_code_fences(text)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]

    return cleaned


def generate_json(prompt: str) -> dict[str, Any]:
    text = generate_text(prompt)
    candidate = _extract_json_candidate(text)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        preview = candidate[:400].replace("\n", " ")
        raise RuntimeError(
            "Gemini returned malformed JSON. "
            f"JSON error: {exc.msg} at line {exc.lineno}, column {exc.colno}. "
            f"Response preview: {preview}"
        ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"Gemini returned valid JSON but not a JSON object (got {type(parsed).__name__})."
        )
    return parsed
