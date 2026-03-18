"""Phase 3: Dialogue Generation Subgraph.

Transforms chapter outlines and character personas into SSML-annotated
dialogue scripts ready for TTS synthesis.

Flow: dialogue_engine → expert_expander → naturalness_injector
      → fact_checker → qa_reviewer → ssml_annotator → END
"""

import re
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Phase3State(TypedDict):
    """State for Phase 3: Dialogue Generation."""
    # Inputs from Phase 1 & 2
    topic: str
    chapter_outlines: List[Dict[str, Any]]
    character_personas: List[Dict[str, Any]]
    ranked_chunks: List[Dict[str, Any]]
    # Output
    chapter_dialogues: List[Dict[str, Any]]


def dialogue_engine_node(state: Phase3State) -> Phase3State:
    """Generate beat-by-beat dialogue for all chapters."""
    from src.agents.phase3.dialogue_engine import generate_chapter_dialogue

    print("\n🎤 PHASE 3 - DIALOGUE ENGINE")
    chapter_dialogues = []
    for ch in state["chapter_outlines"]:
        print(f"   Generating Chapter {ch['chapter_number']}: {ch['title']}")
        utts = generate_chapter_dialogue(
            ch, state["character_personas"], state["ranked_chunks"],
            topic=state.get("topic", ""),
        )
        chapter_dialogues.append({
            "chapter_number": ch["chapter_number"],
            "chapter_outline": ch,
            "utterances": utts,
            "fact_check_issues": [],
            "qa_review": {},
            "quality_checks_passed": False,
            "validation_metadata": {},
        })
        print(f"   ✅ Chapter {ch['chapter_number']}: {len(utts)} utterances")

    state["chapter_dialogues"] = chapter_dialogues
    return state


def expert_expander_node(state: Phase3State) -> Phase3State:
    """Expand short expert utterances across all chapters."""
    from src.agents.phase3.dialogue_engine import expand_expert_utterances

    print("\n📝 EXPERT CONTENT EXPANSION")
    for cd in state["chapter_dialogues"]:
        cd["utterances"] = expand_expert_utterances(
            cd["utterances"], cd["chapter_outline"],
            state["character_personas"], state["ranked_chunks"],
        )
    print("   ✅ Done\n")
    return state


def naturalness_node(state: Phase3State) -> Phase3State:
    """Inject naturalness markers into all chapters."""
    from src.agents.phase3.naturalness_injector import inject_naturalness

    print("🎭 NATURALNESS INJECTION")
    for cd in state["chapter_dialogues"]:
        cd["utterances"] = inject_naturalness(
            cd["utterances"], state["character_personas"],
            cd["chapter_outline"]["energy_level"],
        )
    print("   ✅ Done\n")
    return state


def fact_checker_node(state: Phase3State) -> Phase3State:
    """Fact-check claims across all chapters."""
    from src.agents.phase3.fact_checker import check_facts

    print("🔍 FACT CHECKING")
    for cd in state["chapter_dialogues"]:
        cd["utterances"], issues = check_facts(
            cd["utterances"], state["ranked_chunks"],
        )
        cd["fact_check_issues"] = issues
        status = f"⚠️  {len(issues)} issues" if issues else "✅ verified"
        print(f"   Chapter {cd['chapter_number']}: {status}")
    print()
    return state


def qa_reviewer_node(state: Phase3State) -> Phase3State:
    """QA review all chapters from listener perspective."""
    from src.agents.phase3.qa_reviewer import review_chapter

    print("📋 QA REVIEW")
    for cd in state["chapter_dialogues"]:
        cd["utterances"], review = review_chapter(
            cd["utterances"], cd["chapter_outline"],
        )
        cd["qa_review"] = review
        score = review.get("listener_experience_score", 0)
        passed = review.get("overall_pass", False)
        print(f"   {'✅' if passed else '⚠️ '} Chapter {cd['chapter_number']}: score={score}")
    print()
    return state


def ssml_annotator_node(state: Phase3State) -> Phase3State:
    """Convert naturalness markers to TTS-ready output."""
    from src.agents.phase3.ssml_annotator import annotate_chapter

    print(f"🔊 SSML ANNOTATION (provider: {settings.TTS_PROVIDER})")
    for cd in state["chapter_dialogues"]:
        cd["utterances"] = annotate_chapter(
            cd["utterances"], state["character_personas"],
        )
        # Compile final metadata
        total_dur = sum(u["estimated_duration_seconds"] for u in cd["utterances"])
        cd["total_utterances"] = len(cd["utterances"])
        cd["estimated_chapter_duration"] = total_dur / 60
        cd["quality_checks_passed"] = cd.get("qa_review", {}).get("overall_pass", True)
        cd["validation_metadata"] = {
            "fact_check_claims_verified": len(cd["utterances"]) - len(cd.get("fact_check_issues", [])),
            "fact_check_unsupported": len(cd.get("fact_check_issues", [])),
            "qa_review_score": cd.get("qa_review", {}).get("listener_experience_score", 0),
            "naturalness_markers": sum(
                len(re.findall(r'\[', u.get("text_with_naturalness", "")))
                for u in cd["utterances"]
            ),
        }
        print(
            f"   ✅ Chapter {cd['chapter_number']}: "
            f"{cd['total_utterances']} utts, {cd['estimated_chapter_duration']:.1f} min"
        )

    print("\n✅ Phase 3 complete!\n")
    return state


def create_phase3_graph():
    """Create and compile the Phase 3 (Dialogue Generation) subgraph.

    Flow: dialogue_engine → expert_expander → naturalness_injector
          → [fact_checker] → [qa_reviewer] → ssml_annotator → END

    fact_checker and qa_reviewer are optional, controlled by
    settings.PHASE3_ENABLE_FACT_CHECKER and settings.PHASE3_ENABLE_QA_REVIEWER.
    """
    workflow = StateGraph(Phase3State)

    workflow.add_node("dialogue_engine", dialogue_engine_node)
    workflow.add_node("expert_expander", expert_expander_node)
    workflow.add_node("naturalness_injector", naturalness_node)
    workflow.add_node("ssml_annotator", ssml_annotator_node)

    workflow.set_entry_point("dialogue_engine")
    workflow.add_edge("dialogue_engine", "expert_expander")
    workflow.add_edge("expert_expander", "naturalness_injector")

    # Build the optional middle chain: naturalness → [fact_checker] → [qa_reviewer] → ssml
    prev = "naturalness_injector"

    if settings.PHASE3_ENABLE_FACT_CHECKER:
        workflow.add_node("fact_checker", fact_checker_node)
        workflow.add_edge(prev, "fact_checker")
        prev = "fact_checker"
        logger.info("Fact checker ENABLED")
    else:
        logger.info("Fact checker DISABLED — skipping")

    if settings.PHASE3_ENABLE_QA_REVIEWER:
        workflow.add_node("qa_reviewer", qa_reviewer_node)
        workflow.add_edge(prev, "qa_reviewer")
        prev = "qa_reviewer"
        logger.info("QA reviewer ENABLED")
    else:
        logger.info("QA reviewer DISABLED — skipping")

    workflow.add_edge(prev, "ssml_annotator")
    workflow.add_edge("ssml_annotator", END)

    return workflow.compile()
