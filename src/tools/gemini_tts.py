"""Vertex AI Gemini TTS invocation helpers."""

from __future__ import annotations
from typing import Any, Dict
from google import genai
from google.genai import types

def synthesize_gemini_speech(
    prompt: str,
    voice_name: str,
    model: str,  # e.g., "gemini-2.5-pro-tts"
    project_id: str,
    location: str = "us-central1",
) -> Dict[str, Any]:
    """Invoke the Vertex AI Gemini TTS API and return raw PCM bytes."""

    # 1. Initialize Vertex AI Client
    # Authentication is handled via Application Default Credentials (ADC)
    # in production (Service Accounts)
    client = genai.Client(
        vertexai=True, 
        project=project_id, 
        location=location
    )

    # 2. Configure the TTS Payload
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name,
                )
            )
        ),
    )

    # 3. Call Vertex AI
    # Vertex Model IDs: "gemini-2.5-pro-tts" or "gemini-2.5-flash-tts"
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )

    # 4. Extract Audio Part
    if not response.candidates or not response.candidates[0].content.parts:
        raise ValueError("Google TTS response did not include audio data")

    audio_part = response.candidates[0].content.parts[0]
    
    # The SDK returns bytes directly if available in inline_data
    if audio_part.inline_data:
        return {
            "audio_bytes": audio_part.inline_data.data,
            "sample_rate": 24000,
            "channels": 1,
            "sample_width": 2,
        }

    raise ValueError("Audio data was not found in the response parts")

# --- Example Usage for Production ---
if __name__ == "__main__":
    import wave
    import os

    result = synthesize_gemini_speech(
        prompt="Hello production world!",
        voice_name="Kore",
        model="gemini-2.5-pro-tts",
        project_id="effortless-lock-329115"
    )

    output_path = os.path.join(os.getcwd(), "output_audio.wav")
    print(f"Saving audio to {output_path}...")
    
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(result["channels"])
        wf.setsampwidth(result["sample_width"])
        wf.setframerate(result["sample_rate"])
        wf.writeframes(result["audio_bytes"])
        
    print(f"Audio successfully saved to: {output_path}")