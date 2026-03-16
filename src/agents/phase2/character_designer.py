"""Phase 2 Agent 2: Character Designer.

Creates rich, distinct speaker personas using a single LLM call.
Input: topic + chapter_outlines + num_speakers
Output: List of CharacterPersona dicts saved to phase2_results.json
"""

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI

from config.settings import settings
from src.llm.prompts import CHARACTER_DESIGNER_PROMPT
from src.models.character import CharacterRoster, CharacterPersona
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Gemini 2.5 Pro Preview TTS voice bank
VOICE_BANK = [
    {"name": "Aoede", "gender": "female", "profile": "Breezy, clear, great for hosting."},
    {"name": "Charon", "gender": "male", "profile": "Deep, informative, authoritative narrator."},
    {"name": "Puck", "gender": "male", "profile": "Upbeat, youthful, very energetic."},
    {"name": "Kore", "gender": "female", "profile": "Firm, professional, news anchor style."},
    {"name": "Fenrir", "gender": "male", "profile": "Excitable, high-energy."},
    {"name": "Leda", "gender": "female", "profile": "Youthful, conversational."},
    {"name": "Enceladus", "gender": "male", "profile": "Breathy, intimate, good for storytelling."},
    {"name": "Vindemiatrix", "gender": "female", "profile": "Gentle, calm, ASMR-lite vibes."},
    {"name": "Zubenelgenubi", "gender": "male", "profile": "Casual, relaxed, everyman voice."},
    {"name": "Zephyr", "gender": "female", "profile": "Bright, friendly, commercial style."},
]


def _build_chapters_context(chapter_outlines: List[Dict[str, Any]]) -> str:
    """Format chapter outlines into a readable context string for the LLM."""
    lines = []
    for ch in chapter_outlines:
        lines.append(
            f"Chapter {ch['chapter_number']}: \"{ch['title']}\" "
            f"(Act: {ch['act']}, Energy: {ch['energy_level']})"
        )
        for i, kp in enumerate(ch["key_points"], 1):
            lines.append(f"  {i}. {kp}")
        lines.append(f"  Transition: {ch['transition_hook']}")
        lines.append("")
    return "\n".join(lines)


def _build_voices_list() -> str:
    """Format voice bank into a readable list for the LLM."""
    lines = []
    for v in VOICE_BANK:
        lines.append(f"- Name: {v['name']}, Gender: {v['gender']}, Profile: {v['profile']}")
    return "\n".join(lines)


def _get_role_rules(num_speakers: int) -> str:
    """Return role assignment rules based on speaker count."""
    if num_speakers == 2:
        return (
            "You must create exactly 2 characters:\n"
            "1. The Host (Curious Generalist): Listener's proxy. Simple language, asks clarifying "
            "questions, summarises complex points, expresses surprise and curiosity. vocabulary_level=casual.\n"
            "2. The Expert (Domain Authority): Deep topic knowledge. Uses technical terminology "
            "(which the Host asks to explain). Provides substance. vocabulary_level=technical."
        )
    return (
        "You must create exactly 3 characters:\n"
        "1. The Host (Curious Generalist): Listener's proxy. Simple language, asks clarifying "
        "questions, summarises complex points, expresses surprise and curiosity. vocabulary_level=casual.\n"
        "2. The Expert (Domain Authority): Deep topic knowledge. Uses technical terminology "
        "(which the Host asks to explain). Provides substance. vocabulary_level=technical.\n"
        "3. The Skeptic/Contrarian (Devil's Advocate): Challenges assumptions, brings up risks "
        "and downsides, creates tension and debate. vocabulary_level=moderate."
    )


def _get_gender_rule(num_speakers: int) -> str:
    """Return gender diversity rule based on speaker count."""
    if num_speakers == 3:
        return "- For 3 speakers, at least one character MUST be female. Pick a female voice for them."
    return "- For 2 speakers, choose any gender mix that fits the voices best."


def _get_llm() -> ChatOpenAI:
    """Get configured LLM for character designer."""
    return ChatOpenAI(
        model=settings.CHARACTER_DESIGNER_MODEL,
        temperature=settings.CHARACTER_DESIGNER_TEMPERATURE,
    )


def design_characters(
    topic: str,
    chapter_outlines: List[Dict[str, Any]],
    num_speakers: int,
) -> List[Dict[str, Any]]:
    """Design speaker personas using a single LLM call with structured output.

    Args:
        topic: Podcast topic string
        chapter_outlines: List of chapter outline dicts from Chapter Planner
        num_speakers: Number of speakers (2 or 3)

    Returns:
        List of CharacterPersona dicts
    """
    num_speakers = max(2, min(3, num_speakers))  # clamp to 2-3

    prompt = CHARACTER_DESIGNER_PROMPT.format(
        num_speakers=num_speakers,
        topic=topic,
        chapters_context=_build_chapters_context(chapter_outlines),
        role_rules=_get_role_rules(num_speakers),
        voices_list=_build_voices_list(),
        gender_rule=_get_gender_rule(num_speakers),
    )

    llm = _get_llm().with_structured_output(CharacterRoster)
    logger.info(f"Generating {num_speakers} character personas for topic: '{topic}'")

    roster: CharacterRoster = llm.invoke(prompt)

    # Validate basics
    roles = [c.role for c in roster.characters]
    assert "host" in roles, "Must have exactly one host"
    assert len(roster.characters) == num_speakers, f"Expected {num_speakers} characters"

    # Validate voice uniqueness
    voices = [c.tts_voice_id for c in roster.characters]
    assert len(set(voices)) == len(voices), "Each character must have a unique voice"

    personas = [c.model_dump() for c in roster.characters]

    for p in personas:
        logger.info(f"  {p['role'].upper()}: {p['name']} — voice={p['tts_voice_id']}, "
                     f"vocab={p['vocabulary_level']}")

    return personas
