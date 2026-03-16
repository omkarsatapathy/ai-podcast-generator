"""Phase 2: Content Planning Subgraph.

Workflow:
1. chapter_planner → Analyze, cluster, sequence, and outline chapters
2. character_designer → Create speaker personas (placeholder)
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

    # Character Designer outputs (placeholder)
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


def character_designer_placeholder(state: Phase2State) -> Phase2State:
    """Placeholder for Character Designer agent."""
    print("\n⏳ PHASE 2 - CHARACTER DESIGNER (Not yet implemented)\n")
    state["character_personas"] = []
    return state


def create_phase2_graph():
    """Create and compile the Phase 2 (Content Planning) subgraph.

    Flow: chapter_planner → character_designer → END
    """
    workflow = StateGraph(Phase2State)

    workflow.add_node("chapter_planner", chapter_planner_node)
    workflow.add_node("character_designer", character_designer_placeholder)

    workflow.set_entry_point("chapter_planner")
    workflow.add_edge("chapter_planner", "character_designer")
    workflow.add_edge("character_designer", END)

    return workflow.compile()
