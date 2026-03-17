"""Phase 3 Agent: Naturalness Injector.

Adds speech patterns, hesitations, and conversational markers using
a hybrid rule-based + LLM approach.
"""

import random
import re
from typing import List, Dict

from langchain_openai import ChatOpenAI

from config.settings import settings
from src.llm.prompts import NATURALNESS_INJECTION_PROMPT
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _apply_rule_based_markers(utterances: List[Dict], energy_level: str) -> List[Dict]:
    """Apply deterministic structural markers."""
    for i, utt in enumerate(utterances):
        text = utt["text_clean"]
        word_count = len(text.split())

        # Pause after questions
        if text.strip().endswith("?") and utt["intent"] == "question":
            text += " [PAUSE:short]"

        # Self-continuation filler
        if i > 0 and utterances[i - 1]["speaker"] == utt["speaker"] and random.random() < 0.4:
            text = "[FILLER:thinking] " + text

        # High energy + long utterance → fast pacing for first third
        if energy_level == "high" and word_count > 30:
            words = text.split()
            third = len(words) // 3
            text = "[PACE:fast] " + " ".join(words[:third]) + " " + " ".join(words[third:])

        # Slow pacing for beat 5 summaries
        if utt["beat"] == 5 and utt["intent"] == "summary":
            text = "[PACE:slow] " + text

        utt["text_with_naturalness"] = text
    return utterances


def inject_naturalness(
    utterances: List[Dict], personas: List[Dict], energy_level: str,
) -> List[Dict]:
    """Apply naturalness markers: rule-based first, then LLM-enhanced."""
    utterances = _apply_rule_based_markers(utterances, energy_level)

    persona_map = {p["name"]: p for p in personas}
    llm = ChatOpenAI(
        model=settings.NATURALNESS_MODEL,
        temperature=settings.NATURALNESS_TEMPERATURE,
    )

    for i, utt in enumerate(utterances):
        persona = persona_map.get(utt["speaker"], {})
        prev_text = utterances[i - 1]["text_clean"][:200] if i > 0 else ""
        next_text = utterances[i + 1]["text_clean"][:200] if i < len(utterances) - 1 else ""

        prompt = NATURALNESS_INJECTION_PROMPT.format(
            speaker_name=utt["speaker"], role=utt["role"],
            text=utt["text_with_naturalness"],
            intent=utt["intent"], emotion=utt["emotion"],
            beat=utt["beat"], energy_level=energy_level,
            speaking_style=persona.get("speaking_style", ""),
            vocabulary_level=persona.get("vocabulary_level", "casual"),
            previous_text=prev_text, next_text=next_text,
        )

        try:
            enhanced = llm.invoke(prompt).content.strip().strip('"')
            markers = re.findall(r'\[([A-Z_]+)(?::([a-z0-9.]+))?\]', enhanced)
            if len(markers) <= 6:  # respect per-utterance cap
                utt["text_with_naturalness"] = enhanced
        except Exception as e:
            logger.warning(f"Naturalness LLM failed for {utt['utterance_id']}: {e}")

    total = sum(
        len(re.findall(r'\[', u["text_with_naturalness"])) for u in utterances
    )
    logger.info(f"Naturalness complete: {total} markers across {len(utterances)} utterances")
    return utterances
