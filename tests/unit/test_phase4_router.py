"""Unit tests for Phase 4 routing and validation."""

from __future__ import annotations

from config.settings import settings
from src.agents.phase4.tts_router import (
    plan_tts_jobs,
    resolve_voice_policy,
    route_tts_jobs,
    validate_phase4_input_contract,
)


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
                        },
                    },
                ],
            }
        ],
    }


def test_validate_phase4_input_contract_blocks_duplicate_utterance_ids():
    state = _sample_state()
    state["chapter_dialogues"][0]["utterances"][1]["utterance_id"] = "ch1_u001"

    _, report, blocked = validate_phase4_input_contract(state)

    assert blocked is True
    assert any("Duplicate utterance_id" in error for error in report["errors"])


def test_resolve_voice_policy_uses_elevenlabs_role_defaults(monkeypatch):
    state = _sample_state()
    validated_scripts, _, blocked = validate_phase4_input_contract(state)

    assert blocked is False

    monkeypatch.setattr(settings, "TTS_PROVIDER", "elevenlabs")
    monkeypatch.setattr(settings, "ELEVENLABS_HOST_VOICE_ID", "host_voice")
    monkeypatch.setattr(settings, "ELEVENLABS_EXPERT_VOICE_ID", "expert_voice")
    monkeypatch.setattr(settings, "ELEVENLABS_SKEPTIC_VOICE_ID", "skeptic_voice")

    speaker_voice_map, provider_policy, _ = resolve_voice_policy(
        validated_scripts,
        state["character_personas"],
    )

    assert provider_policy["primary_provider"] == "elevenlabs"
    assert speaker_voice_map["Ava Hart"]["voice_id"] == "host_voice"
    assert speaker_voice_map["Dr. Miles Chen"]["voice_id"] == "expert_voice"


def test_route_tts_jobs_builds_google_plain_text_payload(monkeypatch):
    state = _sample_state()
    validated_scripts, _, blocked = validate_phase4_input_contract(state)

    assert blocked is False

    monkeypatch.setattr(settings, "TTS_PROVIDER", "google")
    speaker_voice_map, _, _ = resolve_voice_policy(validated_scripts, state["character_personas"])
    jobs, _, _ = plan_tts_jobs(validated_scripts, speaker_voice_map, "episode-test")
    routed_jobs, _, _ = route_tts_jobs(jobs)

    first_payload = routed_jobs[0]["payload"]
    assert first_payload["provider"] == "google"
    assert "<speak>" not in first_payload["text"]
    assert "Welcome back to the show." in first_payload["prompt"]
