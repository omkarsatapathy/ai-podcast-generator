"""OpenAI TTS invocation helpers (gpt-4o-mini-tts)."""

from __future__ import annotations

from typing import Any, Dict

import requests


def synthesize_openai_speech(
    text: str,
    voice: str,
    model: str,
    api_key: str,
    instructions: str = "",
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """Invoke the OpenAI TTS API and return raw PCM-like audio bytes.

    The API returns audio in the requested format.  We ask for ``pcm``
    (24 kHz, 16-bit mono, little-endian) so the output is byte-compatible
    with the Google and ElevenLabs providers.
    """

    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI TTS synthesis")
    if not voice:
        raise ValueError("A valid OpenAI voice name is required for synthesis")

    payload: Dict[str, Any] = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "pcm",
    }
    if instructions:
        payload["instructions"] = instructions

    response = requests.post(
        url="https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    return {
        "audio_bytes": response.content,
        "sample_rate": 24000,
        "channels": 1,
        "sample_width": 2,
        "response_headers": dict(response.headers),
    }
