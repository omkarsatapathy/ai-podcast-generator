"""Phase 2: Content Planning Subgraph.

Workflow:
1. chapter_planner → Analyze, cluster, sequence, and outline chapters
2. character_designer → Create speaker personas via single LLM call
"""

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END


class Phase2State(TypedDict):
    """State for Phase 2: Content Planning."""
    # Phase 1 inputs
    topic: str
    ranked_chunks: List[Dict[str, Any]]
    num_speakers: int

    # Chapter Planner outputs
    chapter_outlines: List[Dict[str, Any]]
    analyzed_chunks: List[Dict[str, Any]]
    total_estimated_duration: float

    # Character Designer outputs
    character_personas: List[Dict[str, Any]]


def chapter_planner_node(state: Phase2State) -> Phase2State:
    """Execute Chapter Planner agent."""
    from src.agents.phase2.chapter_planner import process

    print("\n📖 PHASE 2 - CHAPTER PLANNER")
    print(f"   Input: {len(state['ranked_chunks'])} ranked chunks\n")

    result = process(
        ranked_chunks=state["ranked_chunks"],
        topic=state["topic"],
    )

    state["chapter_outlines"] = result["chapter_outlines"]
    state["analyzed_chunks"] = result["analyzed_chunks"]
    state["total_estimated_duration"] = result["stats"]["total_duration_minutes"]

    print(f"   ✅ {result['stats']['num_chapters']} chapters planned")
    print(f"   ⏱️  Total duration: {result['stats']['total_duration_minutes']:.1f} min\n")

    return state


def character_designer_node(state: Phase2State) -> Phase2State:
    """Execute Character Designer agent."""
    from src.agents.phase2.character_designer import design_characters

    print("\n🎭 PHASE 2 - CHARACTER DESIGNER")
    print(f"   Designing {state['num_speakers']} speaker personas\n")

    personas = design_characters(
        topic=state["topic"],
        chapter_outlines=state["chapter_outlines"],
        num_speakers=state["num_speakers"],
    )

    state["character_personas"] = personas

    for p in personas:
        print(f"   ✅ {p['role'].upper()}: {p['name']} (voice: {p['tts_voice_id']})")
    print()

    return state


def create_phase2_graph():
    """Create and compile the Phase 2 (Content Planning) subgraph.

    Flow: chapter_planner → character_designer → END
    """
    workflow = StateGraph(Phase2State)

    workflow.add_node("chapter_planner", chapter_planner_node)
    workflow.add_node("character_designer", character_designer_node)

    workflow.set_entry_point("chapter_planner")
    workflow.add_edge("chapter_planner", "character_designer")
    workflow.add_edge("character_designer", END)

    return workflow.compile()
