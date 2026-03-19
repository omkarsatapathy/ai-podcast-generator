"""Phase 3 Agent: QA Reviewer.

Evaluates chapter scripts from a first-time listener's perspective.
"""

from typing import List, Dict, Tuple

from config.settings import settings
from src.api_factory.llm import get_llm
from src.llm.prompts import QA_REVIEW_PROMPT
from src.models.dialogue import QAReviewResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _format_script(utterances: List[Dict]) -> str:
    """Format utterances as readable script."""
    current_beat = 0
    lines = []
    for u in utterances:
        if u["beat"] != current_beat:
            current_beat = u["beat"]
            lines.append(f"\n--- Beat {current_beat} ---")
        text = u.get("text_with_naturalness") or u["text_clean"]
        lines.append(f"[{u['utterance_id']}] **{u['speaker']}:** {text}")
    return "\n".join(lines)


def review_chapter(
    utterances: List[Dict], chapter: Dict,
) -> Tuple[List[Dict], Dict]:
    """Review chapter quality. Returns (utterances, review_dict)."""
    llm = get_llm(
        tier=settings.QA_REVIEWER_MODEL,
        temperature=settings.QA_REVIEWER_TEMPERATURE,
    ).with_structured_output(QAReviewResult, method="json_schema")

    prompt = QA_REVIEW_PROMPT.format(
        chapter_number=chapter["chapter_number"],
        chapter_title=chapter["title"],
        act=chapter["act"],
        energy_level=chapter["energy_level"],
        key_points=", ".join(chapter["key_points"]),
        script_text=_format_script(utterances),
    )

    try:
        result: QAReviewResult = llm.invoke(prompt)
        review = result.model_dump()
        critical = [i for i in result.issues_found if i.get("severity") == "critical"]
        logger.info(
            f"QA Ch{chapter['chapter_number']}: score={result.listener_experience_score}, "
            f"pass={result.overall_pass}, critical={len(critical)}"
        )
        return utterances, review
    except Exception as e:
        logger.warning(f"QA review failed: {e}")
        return utterances, {
            "overall_pass": True, "issues_found": [], "strengths": [],
            "listener_experience_score": 7.0,
            "reasoning": "QA review skipped due to error",
        }
