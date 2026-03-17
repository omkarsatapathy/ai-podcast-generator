"""Google Gemini TTS invocation helpers."""

from __future__ import annotations

import base64
from typing import Any, Dict

import requests


def synthesize_gemini_speech(
    prompt: str,
    voice_name: str,
    model: str,
    api_key: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    """Invoke the Gemini TTS API and return raw PCM bytes."""

    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for Google TTS synthesis")

    response = requests.post(
        url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice_name,
                        }
                    }
                },
            },
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()

    candidates = payload.get("candidates", [])
    for candidate in candidates:
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            inline_data = part.get("inlineData") or part.get("inline_data") or {}
            encoded = inline_data.get("data")
            if encoded:
                return {
                    "audio_bytes": base64.b64decode(encoded),
                    "sample_rate": 24000,
                    "channels": 1,
                    "sample_width": 2,
                    "response_payload": payload,
                }

    raise ValueError("Google TTS response did not include audio data")
