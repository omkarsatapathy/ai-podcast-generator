"""Voice/TTS provider factory — uniform interface for all TTS providers.

All providers return: {"audio_bytes", "sample_rate", "channels", "sample_width", ...}
Audio format: PCM 24kHz, mono, 16-bit (consistent across all providers).
"""

from __future__ import annotations

import os
from typing import Any, Dict

import requests

from config.settings import settings


def synthesize_speech(provider: str, **kwargs: Any) -> Dict[str, Any]:
    """Route TTS synthesis to the correct provider.

    Args:
        provider: "google", "openai", "elevenlabs", or "sarvam"
        **kwargs: Provider-specific parameters (see individual functions)

    Returns:
        Dict with audio_bytes, sample_rate, channels, sample_width
    """
    if provider == "google":
        from src.tools.gemini_tts import synthesize_gemini_speech

        return synthesize_gemini_speech(**kwargs)

    if provider == "openai":
        from src.tools.openai_tts import synthesize_openai_speech

        return synthesize_openai_speech(**kwargs)

    if provider == "elevenlabs":
        from src.tools.elevenlabs_tts import synthesize_elevenlabs_speech

        return synthesize_elevenlabs_speech(**kwargs)

    if provider == "sarvam":
        return _synthesize_sarvam(**kwargs)

    raise ValueError(f"Unsupported TTS provider: '{provider}'")


def _synthesize_sarvam(
    text: str,
    voice: str = "meera",
    model: str = "bulbul:v2",
    api_key: str = "",
    target_language: str = "en-IN",
    timeout_seconds: int = 60,
    **_: Any,
) -> Dict[str, Any]:
    """Invoke Sarvam Bulbul TTS API.

    Docs: https://docs.sarvam.ai/api-reference-docs/getting-started/models/bulbul
    """
    api_key = api_key or os.environ.get("SARVAM_API_KEY", "")
    if not api_key:
        raise ValueError("SARVAM_API_KEY is required for Sarvam TTS")

    resp = requests.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "inputs": [text],
            "target_language_code": target_language,
            "speaker": voice,
            "model": model,
        },
        timeout=timeout_seconds,
    )
    resp.raise_for_status()

    data = resp.json()
    # Sarvam returns base64 audio — decode to raw bytes
    import base64

    audio_b64 = data.get("audios", [None])[0]
    if not audio_b64:
        raise ValueError("Sarvam TTS response did not include audio data")

    return {
        "audio_bytes": base64.b64decode(audio_b64),
        "sample_rate": 22050,  # Sarvam Bulbul returns WAV at 22050 Hz
        "channels": 1,
        "sample_width": 2,
    }
