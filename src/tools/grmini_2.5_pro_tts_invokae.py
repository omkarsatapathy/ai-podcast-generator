import os
import wave
from google import genai
from google.genai import types

# Initialize client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Use the specific TTS model variant
model_id = "gemini-2.5-pro-preview-tts" 

def save_as_wav(filename, pcm_data, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)

print(f"Invoking {model_id} for TTS...")

# Generate content with audio modality
response = client.models.generate_content(
    model=model_id,
    contents="Say in a calm, professional tone: Welcome to the Podcast Creator tool. We are now testing the high-fidelity Gemini TTS voices.",
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Aoede"  # Choose from the dictionary provided earlier
                )
            )
        )
    )
)

# Extract and save the audio
audio_found = False
for part in response.candidates[0].content.parts:
    if part.inline_data:
        # Gemini returns raw PCM bytes
        pcm_bytes = part.inline_data.data
        output_file = "podcast_intro.wav"
        save_as_wav(output_file, pcm_bytes)
        print(f"Success! Audio saved to {output_file}")
        audio_found = True
        break

if not audio_found:
    print("Error: No audio data returned. Check your model permissions or API region.")