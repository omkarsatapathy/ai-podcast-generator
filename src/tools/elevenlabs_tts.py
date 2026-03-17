"""ElevenLabs TTS invocation helpers."""

from __future__ import annotations

from typing import Any, Dict

import requests


def synthesize_elevenlabs_speech(
    text: str,
    voice_id: str,
    model: str,
    api_key: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    """Invoke the ElevenLabs TTS API and return provider audio bytes."""

    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY is required for ElevenLabs synthesis")
    if not voice_id:
        raise ValueError("A valid ElevenLabs voice_id is required for synthesis")

    response = requests.post(
        url=f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        params={"output_format": "pcm_24000"},
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": model,
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.8,
            },
        },
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
