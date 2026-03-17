"""Integration test for the Phase 4 graph with mocked synthesis."""

from __future__ import annotations

from array import array
from pathlib import Path

from config.settings import settings
from src.agents.phase4 import tts_router
from src.pipeline.phases.phase4_graph import create_phase4_graph


def _sample_state() -> dict:
    return {
        "topic": "AI in healthcare",
        "character_personas": [
            {
                "name": "Ava Hart",
                "role": "host",
                "tts_voice_id": "Aoede",
            },
            {
                "name": "Dr. Miles Chen",
                "role": "expert",
                "tts_voice_id": "Charon",
            },
        ],
        "chapter_dialogues": [
            {
                "chapter_number": 1,
                "utterances": [
                    {
                        "utterance_id": "ch1_u001",
                        "speaker": "Ava Hart",
                        "role": "host",
                        "beat": 1,
                        "text_clean": "Welcome back to the show.",
                        "text_with_naturalness": "[PAUSE:short] Welcome back to the show.",
                        "text_ssml": "<speak><break time=\"400ms\"/> Welcome back to the show.</speak>",
                        "estimated_duration_seconds": 1.4,
                        "tts_voice_id": "Aoede",
                        "audio_metadata": {},
                    },
                    {
                        "utterance_id": "ch1_u002",
                        "speaker": "Dr. Miles Chen",
                        "role": "expert",
                        "beat": 1,
                        "text_clean": "AI can help doctors surface patterns earlier.",
                        "text_with_naturalness": "AI can help doctors surface patterns earlier.",
                        "text_ssml": "<speak>AI can help doctors surface patterns earlier.</speak>",
                        "estimated_duration_seconds": 2.0,
                        "tts_voice_id": "Charon",
                        "audio_metadata": {
                            "interrupt_duration": "0.3s",
                            "backchannel_speaker": "host",
                        },
                    },
                ],
            }
        ],
    }


def _tone_bytes(frame_count: int = 6000) -> bytes:
    samples = array("h", [1200 if index % 2 == 0 else -1200 for index in range(frame_count)])
    return samples.tobytes()


def test_phase4_graph_packages_audio_and_manifests(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "TTS_PROVIDER", "google")
    monkeypatch.setattr(settings, "TTS_FALLBACK_PROVIDER", "")
    monkeypatch.setattr(settings, "PHASE4_RAW_AUDIO_DIR", tmp_path / "raw")
    monkeypatch.setattr(settings, "PHASE4_MAX_WORKERS", 1)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")

    def fake_gemini_call(**_: str) -> dict:
        return {
            "audio_bytes": _tone_bytes(),
            "sample_rate": 24000,
            "channels": 1,
            "sample_width": 2,
        }

    monkeypatch.setattr(tts_router, "synthesize_gemini_speech", fake_gemini_call)

    graph = create_phase4_graph()
    result = graph.invoke(_sample_state())

    assert result["ready_for_phase5"] is True
    assert len(result["audio_files"]) == 2
    assert len(result["chapter_audio_manifests"]) == 1
    assert result["phase4_output"]["ready_for_phase5"] is True

    for audio_file in result["audio_files"]:
        assert Path(audio_file["path"]).exists()
