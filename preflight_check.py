import base64
import wave
import io
from sarvamai import SarvamAI

client = SarvamAI(api_subscription_key="sk_z7xqmb7q_lE6GUdAiu7OMaspkPBdrE0PI")

response = client.text_to_speech.convert(
    text="मैं ऑफिस जा रहा हूँ I am going to the office",
    target_language_code="hi-IN",
    model="bulbul:v3",
    speaker="priya"
)

# Decode and combine all audio chunks
audio_data = b"".join(base64.b64decode(chunk) for chunk in response.audios)

# Save to .wav
with open("output.wav", "wb") as f:
    f.write(audio_data)

print("Saved to output.wav")