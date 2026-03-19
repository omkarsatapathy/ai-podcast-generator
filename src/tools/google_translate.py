"""Google Translate helper with retry support."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from config.settings import settings

_TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"


def _resolve_translate_api_key(explicit_key: str | None = None) -> str:
    key = (
        explicit_key
        or settings.GOOGLE_TRANSLATE_API_KEY
        or os.environ.get("GOOGLE_TRANSLATE_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
        or os.environ.get("GOOGLE_SEARCH_API_KEY", "")
    )
    if not key:
        raise ValueError(
            "Google Translate API key not found. Set GOOGLE_TRANSLATE_API_KEY (preferred)."
        )
    return key


def translate_text(
    text: str,
    target_language: str,
    source_language: str = "en",
    api_key: str | None = None,
    timeout_seconds: int | None = None,
    max_retries: int = 3,
) -> str:
    """Translate text with Google Translate v2 REST API."""

    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    key = _resolve_translate_api_key(api_key)
    timeout = timeout_seconds or settings.PHASE4_TRANSLATION_TIMEOUT_SECONDS

    payload: dict[str, Any] = {
        "q": cleaned,
        "target": target_language,
        "source": source_language,
        "format": "text",
    }

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                _TRANSLATE_URL,
                params={"key": key},
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            translations = (
                data.get("data", {}).get("translations", [])
                if isinstance(data, dict)
                else []
            )
            if not translations:
                raise ValueError("Google Translate returned no translations")
            translated = translations[0].get("translatedText", "")
            if not isinstance(translated, str) or not translated.strip():
                raise ValueError("Google Translate returned empty translatedText")
            return translated.strip()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(0.25 * (2 ** (attempt - 1)), 2.0))

    raise RuntimeError(f"Google Translate failed after {max_retries} attempts: {last_error}")
