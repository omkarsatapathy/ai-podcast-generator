"""Phase 3 Agent: Dialogue Engine + Expert Content Expander.

Generates beat-by-beat dialogue for each chapter and selectively expands
brief expert utterances for depth and authority.
"""

import re
from typing import List, Dict
from langchain_openai import ChatOpenAI

from config.settings import settings
from src.llm.prompts import (
    DIALOGUE_BEAT_PROMPT, OPENING_BEAT_PROMPT, EXPERT_EXPANSION_PROMPT,
    EXPERT_EXPAND_WITH_NATURALNESS_PROMPT, BEAT_OBJECTIVES,
)
from src.models.dialogue import BeatDialogue
from src.utils.cost_tracker import cost_tracker
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Target metrics per beat (~2.5 words/second speaking rate)
BEAT_CONFIG = {
    0: {"name": "OPENING", "target_words": 120, "target_utterances": 8},
    1: {"name": "HOOK", "target_words": 50, "target_utterances": 4},
    2: {"name": "CONTEXT", "target_words": 180, "target_utterances": 10},
    3: {"name": "DEEP_DIVE", "target_words": 250, "target_utterances": 15},
    4: {"name": "AHA_MOMENT", "target_words": 100, "target_utterances": 6},
    5: {"name": "WRAP", "target_words": 50, "target_utterances": 4},
}


def _get_llm(model: str = None, temperature: float = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.DIALOGUE_ENGINE_MODEL,
        temperature=temperature if temperature is not None else settings.DIALOGUE_ENGINE_TEMPERATURE,
        callbacks=[cost_tracker],
    )


def _build_characters_text(personas: List[Dict]) -> str:
    return "\n".join(
        f"- {p['name']} ({p['role']}): {p['speaking_style']} "
        f"Vocabulary: {p['vocabulary_level']}."
        for p in personas
    )


def _get_chapter_source_chunks(chapter: Dict, ranked_chunks: List[Dict]) -> List[Dict]:
    """Map chapter source_chunk_ids to actual chunk data."""
    chunk_map = {str(c["chunk_id"]): c for c in ranked_chunks}
    return [
        chunk_map[str(cid)]
        for cid in chapter.get("source_chunk_ids", [])
        if str(cid) in chunk_map
    ]


def _build_source_text(chunks: List[Dict]) -> str:
    if not chunks:
        return "No source material available."
    return "\n---\n".join(
        f"[Chunk {c['chunk_id']}] (relevance: {c.get('relevance_score', 'N/A')}):\n"
        f"{c['text'][:500]}\nSource: {c.get('source_url', 'unknown')}"
        for c in chunks[:10]  # cap to prevent token overflow
    )


def _build_previous_beats_text(beat_history: List[List[Dict]]) -> str:
    if not beat_history:
        return ""
    lines = ["## Previous Beats (for context continuity)"]
    for beat_num, utterances in enumerate(beat_history, 1):
        lines.append(f"\n### Beat {beat_num}")
        for u in utterances:
            lines.append(f"**{u['speaker']}:** {u['text']}")
    return "\n".join(lines)


def _generate_beat(
    chapter: Dict, personas: List[Dict], beat_num: int,
    beat_history: List[List[Dict]], source_chunks: List[Dict],
) -> List[Dict]:
    """Generate dialogue for a single beat with retry."""
    config = BEAT_CONFIG[beat_num]
    beat_objective = BEAT_OBJECTIVES[beat_num]
    if beat_num == 5:
        beat_objective = beat_objective.format(
            transition_hook=chapter.get("transition_hook", "Stay tuned.")
        )

    prompt = DIALOGUE_BEAT_PROMPT.format(
        beat_number=beat_num,
        beat_name=config["name"],
        chapter_number=chapter["chapter_number"],
        chapter_title=chapter["title"],
        act=chapter["act"],
        energy_level=chapter["energy_level"],
        key_points=", ".join(chapter["key_points"]),
        target_words=config["target_words"],
        target_utterances=config["target_utterances"],
        characters_text=_build_characters_text(personas),
        beat_objective=beat_objective,
        previous_beats_text=_build_previous_beats_text(beat_history),
        source_chunks_text=_build_source_text(source_chunks),
    )

    llm = _get_llm().with_structured_output(BeatDialogue, method="json_schema")
    persona_map = {p["name"]: p for p in personas}

    for attempt in range(3):
        try:
            result: BeatDialogue = llm.invoke(prompt)
            utterances = []
            for i, u in enumerate(result.utterances):
                persona = persona_map.get(u.speaker, {})
                utterances.append({
                    "utterance_id": f"ch{chapter['chapter_number']}_b{beat_num}_u{i+1:03d}",
                    "speaker": u.speaker,
                    "role": persona.get("role", "unknown"),
                    "beat": beat_num,
                    "text_clean": u.text,
                    "text_with_naturalness": "",
                    "text_ssml": "",
                    "intent": u.intent,
                    "emotion": u.emotion,
                    "grounding_chunk_ids": u.grounding_chunk_ids,
                    "estimated_duration_seconds": len(u.text.split()) / 2.5,
                    "tts_voice_id": persona.get("tts_voice_id", ""),
                    "audio_metadata": {},
                })
            return utterances
        except Exception as e:
            logger.warning(f"Beat {beat_num} attempt {attempt+1} failed: {e}")

    # Fallback: minimal transition dialogue
    host = next((p for p in personas if p["role"] == "host"), personas[0])
    return [{
        "utterance_id": f"ch{chapter['chapter_number']}_b{beat_num}_u001",
        "speaker": host["name"], "role": "host", "beat": beat_num,
        "text_clean": "Let's continue exploring this topic.",
        "text_with_naturalness": "", "text_ssml": "",
        "intent": "transition", "emotion": "neutral",
        "grounding_chunk_ids": [], "estimated_duration_seconds": 3.0,
        "tts_voice_id": host.get("tts_voice_id", ""), "audio_metadata": {},
    }]


def _generate_opening_beat(
    chapter: Dict, personas: List[Dict], topic: str,
) -> List[Dict]:
    """Generate Beat 0 (OPENING) for Chapter 1: host welcome, guest intros, warm-up."""
    config = BEAT_CONFIG[0]
    characters_text = _build_characters_text(personas)

    # Build a persona summary for the prompt
    persona_intros = []
    for p in personas:
        persona_intros.append(
            f"- {p['name']} ({p['role']}): expertise in {p.get('expertise_area', 'the topic')}. "
            f"Style: {p['speaking_style']}"
        )
    personas_detail = "\n".join(persona_intros)

    prompt = OPENING_BEAT_PROMPT.format(
        topic=topic,
        chapter_title=chapter["title"],
        characters_text=characters_text,
        personas_detail=personas_detail,
        target_words=config["target_words"],
        target_utterances=config["target_utterances"],
    )

    llm = _get_llm().with_structured_output(BeatDialogue, method="json_schema")
    persona_map = {p["name"]: p for p in personas}

    for attempt in range(3):
        try:
            result: BeatDialogue = llm.invoke(prompt)
            utterances = []
            for i, u in enumerate(result.utterances):
                persona = persona_map.get(u.speaker, {})
                utterances.append({
                    "utterance_id": f"ch{chapter['chapter_number']}_b0_u{i+1:03d}",
                    "speaker": u.speaker,
                    "role": persona.get("role", "unknown"),
                    "beat": 0,
                    "text_clean": u.text,
                    "text_with_naturalness": "",
                    "text_ssml": "",
                    "intent": u.intent,
                    "emotion": u.emotion,
                    "grounding_chunk_ids": [],
                    "estimated_duration_seconds": len(u.text.split()) / 2.5,
                    "tts_voice_id": persona.get("tts_voice_id", ""),
                    "audio_metadata": {},
                })
            return utterances
        except Exception as e:
            logger.warning(f"Opening beat attempt {attempt+1} failed: {e}")

    # Fallback: minimal welcome
    host = next((p for p in personas if p["role"] == "host"), personas[0])
    return [{
        "utterance_id": f"ch{chapter['chapter_number']}_b0_u001",
        "speaker": host["name"], "role": "host", "beat": 0,
        "text_clean": f"Welcome everyone! Today we're diving into {topic}. Let's get started.",
        "text_with_naturalness": "", "text_ssml": "",
        "intent": "summary", "emotion": "excited",
        "grounding_chunk_ids": [], "estimated_duration_seconds": 5.0,
        "tts_voice_id": host.get("tts_voice_id", ""), "audio_metadata": {},
    }]


def generate_chapter_dialogue(
    chapter: Dict, personas: List[Dict], ranked_chunks: List[Dict],
    topic: str = "",
) -> List[Dict]:
    """Generate all beats of dialogue for a chapter.

    For Chapter 1, prepends Beat 0 (OPENING) with host welcome,
    guest introductions, and warm-up conversation.
    """
    source_chunks = _get_chapter_source_chunks(chapter, ranked_chunks)
    all_utterances = []
    beat_history = []

    # Beat 0 (OPENING) — only for the first chapter
    if chapter.get("chapter_number") == 1:
        logger.info(f"Chapter {chapter['chapter_number']} - Beat 0 (OPENING)")
        opening = _generate_opening_beat(chapter, personas, topic)
        all_utterances.extend(opening)
        beat_history.append([
            {"speaker": u["speaker"], "text": u["text_clean"]} for u in opening
        ])

    for beat_num in range(1, 6):
        logger.info(f"Chapter {chapter['chapter_number']} - Beat {beat_num}")
        utterances = _generate_beat(chapter, personas, beat_num, beat_history, source_chunks)
        all_utterances.extend(utterances)
        beat_history.append([
            {"speaker": u["speaker"], "text": u["text_clean"]} for u in utterances
        ])

    logger.info(f"Chapter {chapter['chapter_number']}: {len(all_utterances)} utterances generated")
    return all_utterances


def expand_expert_utterances(
    utterances: List[Dict], chapter: Dict,
    personas: List[Dict], ranked_chunks: List[Dict],
) -> List[Dict]:
    """Selectively expand short expert utterances for depth.

    Qualifying utterances get a combined expand + naturalness pass
    (single LLM call) so the naturalness injector can skip them later.
    """
    source_chunks = _get_chapter_source_chunks(chapter, ranked_chunks)
    chunk_map = {str(c["chunk_id"]): c for c in source_chunks}

    expert = next((p for p in personas if p["role"] == "expert"), None)
    if not expert:
        return utterances

    llm = _get_llm(settings.EXPERT_EXPANDER_MODEL, settings.EXPERT_EXPANDER_TEMPERATURE)

    for i, utt in enumerate(utterances):
        word_count = len(utt["text_clean"].split())
        # Only expand expert answers/challenges in beats 2-4 that are brief
        if (utt["role"] != "expert"
                or utt["intent"] not in ("answer", "challenge")
                or utt["beat"] not in (2, 3, 4)
                or word_count >= 50):
            continue

        prev_text = utterances[i - 1]["text_clean"] if i > 0 else ""
        next_text = utterances[i + 1]["text_clean"] if i < len(utterances) - 1 else ""

        source_text = "\n".join(
            chunk_map[str(cid)]["text"][:300]
            for cid in utt["grounding_chunk_ids"]
            if str(cid) in chunk_map
        ) or "No source material."

        target_words = min(int(word_count * 2.5), 150)

        # Combined expand + naturalness in a single LLM call
        prompt = EXPERT_EXPAND_WITH_NATURALNESS_PROMPT.format(
            speaker_name=utt["speaker"], original_text=utt["text_clean"],
            current_words=word_count, target_words=target_words,
            speaking_style=expert["speaking_style"],
            vocabulary_level=expert["vocabulary_level"],
            intent=utt["intent"], emotion=utt["emotion"],
            beat=utt["beat"],
            energy_level=chapter.get("energy_level", "medium"),
            previous_text=prev_text[:200], next_text=next_text[:200],
            source_text=source_text, key_points=", ".join(chapter["key_points"]),
        )

        try:
            result = llm.invoke(prompt).content.strip().strip('"')
            # Strip markers to get clean text for word-count validation
            clean_result = re.sub(r'\[([A-Z_]+)(?::([a-z0-9.]+))?\]\s*', '', result)
            expanded_words = len(clean_result.split())
            if word_count < expanded_words <= target_words * 1.5:
                utt["text_clean"] = clean_result
                utt["text_with_naturalness"] = result
                utt["naturalness_applied"] = True
                utt["estimated_duration_seconds"] = expanded_words / 2.5
                logger.info(
                    f"Expanded+naturalness {utt['utterance_id']}: "
                    f"{word_count} → {expanded_words} words (combined pass)"
                )
        except Exception as e:
            logger.warning(f"Expansion failed for {utt['utterance_id']}: {e}")

    return utterances
