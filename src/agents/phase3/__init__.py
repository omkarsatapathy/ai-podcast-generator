"""Phase 3 agents - Dialogue generation and annotation."""

from src.agents.phase3.dialogue_engine import generate_chapter_dialogue, expand_expert_utterances
from src.agents.phase3.naturalness_injector import inject_naturalness
from src.agents.phase3.fact_checker import check_facts
from src.agents.phase3.qa_reviewer import review_chapter
from src.agents.phase3.ssml_annotator import annotate_chapter

__all__ = [
    "generate_chapter_dialogue",
    "expand_expert_utterances",
    "inject_naturalness",
    "check_facts",
    "review_chapter",
    "annotate_chapter",
]
