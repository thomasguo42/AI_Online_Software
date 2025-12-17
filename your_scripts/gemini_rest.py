#!/usr/bin/env python3
"""
Lightweight REST client for Gemini generateContent API.

Avoids gRPC/SRV DNS issues by using plain HTTPS requests with backoff.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Dict, Optional, Tuple

import requests

_logger = logging.getLogger(__name__)


def _extract_candidate_payload(data: Dict[str, Any]) -> Tuple[str, Any]:
    """Extract concatenated text and first JSON part from REST response."""
    if not isinstance(data, dict):
        return "", None

    candidates = data.get("candidates") or []
    if not candidates:
        return "", None

    text_parts = []
    json_part: Any = None

    for candidate in candidates:
        content_obj = candidate.get("content")
        parts = []
        if isinstance(content_obj, dict):
            parts = content_obj.get("parts") or []
        elif isinstance(content_obj, list):
            parts = content_obj

        for part in parts:
            if isinstance(part, dict):
                if "text" in part and isinstance(part["text"], str):
                    text_parts.append(part["text"])
                elif "inline_data" in part and json_part is None:
                    json_part = part["inline_data"]

    return "\n".join(p.strip() for p in text_parts if isinstance(p, str) and p.strip()), json_part


def generate_text(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    top_k: int = 1,
    top_p: float = 0.8,
    max_output_tokens: int = 8192,
    timeout_seconds: int = 45,
    max_attempts: int = 6,
    response_mime_type: str = "text/plain",
) -> str:
    """Call Gemini generateContent REST API and return concatenated text.

    Implements exponential backoff with jitter for 429/503 and honors any
    retry hint found within the response body (e.g., "retry in Xs").
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        _logger.error("GEMINI_API_KEY not set; cannot call Gemini REST API")
        return ""

    model_id = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"

    headers = {"Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}],
        }],
        "generationConfig": {
            "temperature": temperature,
            "topK": top_k,
            "topP": top_p,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": response_mime_type,
        },
    }

    base = float(os.getenv("GEMINI_BACKOFF_BASE_S", "2.0"))
    cap = float(os.getenv("GEMINI_BACKOFF_MAX_S", "60"))

    last_exc: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            resp = requests.post(url, headers=headers, params={"key": api_key}, json=payload, timeout=timeout_seconds)
            if resp.status_code in (429, 503):
                # Respect server backoff hints if present
                reason = resp.reason or "Rate limited"
                txt = resp.text or ""
                retry_after: Optional[float] = None
                try:
                    import re as _re
                    m = _re.search(r"retry in\s*([0-9]+\.?[0-9]*)s", txt, flags=_re.IGNORECASE)
                    if m:
                        retry_after = float(m.group(1))
                except Exception:
                    retry_after = None

                delay = min(cap, base * (2 ** attempt))
                delay = max(delay * (1 + 0.25 * random.random()), retry_after or 0.0)
                delay = min(delay, cap)
                _logger.warning(
                    "Gemini REST backoff %s/%s: %s; sleep %.2fs",
                    attempt + 1,
                    max_attempts,
                    reason,
                    delay,
                )
                if attempt < max_attempts - 1:
                    time.sleep(delay)
                    continue
            resp.raise_for_status()
            data = resp.json()
            text, _json_part = _extract_candidate_payload(data)
            return (text or "").strip()
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            # DNS/network/backoff handling
            txt = str(exc).lower()
            is_retryable = any(
                key in txt
                for key in (
                    "timeout",
                    "temporarily unavailable",
                    "connection reset",
                    "connection aborted",
                    "dns",
                    "name or service not known",
                    "temporary failure in name resolution",
                    "could not contact dns servers",
                )
            )
            if attempt < max_attempts - 1 and is_retryable:
                delay = min(cap, base * (2 ** attempt))
                delay = delay * (1 + 0.25 * random.random())
                _logger.warning("Gemini REST network retry %s/%s; sleep %.2fs", attempt + 1, max_attempts, delay)
                time.sleep(delay)
                continue
            break
        except Exception as exc:  # Non-HTTP parsing or other errors
            last_exc = exc
            _logger.error("Gemini REST unexpected error (attempt %s): %s", attempt + 1, exc)
            break

    if last_exc is not None:
        _logger.error("Gemini REST call failed after %s attempts: %s", max_attempts, last_exc)
    return ""

