"""Main Orchestrator: Composes all 5 phase subgraphs into a complete pipeline.

This module imports each phase subgraph and chains them together to create
the full podcast generation pipeline:

Phase 1 (Research & Ingestion)
    ↓
Phase 2 (Content Planning)
    ↓
Phase 3 (Dialogue Generation)
    ↓
Phase 4 (Voice Synthesis)
    ↓
Phase 5 (Audio Post-Processing)
    ↓
Final Podcast MP3

Each phase subgraph can also be imported independently:
- query_producer.py imports create_phase1_graph() for standalone use
- Other agents import their respective phase subgraphs
"""

from typing import Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage

from src.pipeline.phases import (
    create_phase1_graph,
    create_phase2_graph,
    create_phase3_graph,
    create_phase4_graph,
    create_phase5_graph,
)
from src.utils.cost_tracker import cost_tracker


# ==================== GLOBAL STATE ====================

class GlobalState:
    """Global state that flows through all 5 phases.

    Each phase transforms/enriches this state with its outputs.
    """
    pass  # Will be expanded as phases are implemented


# ==================== ORCHESTRATOR NODES ====================

def phase1_wrapper(state: Any) -> Any:
    """Phase 1: Research & Ingestion wrapper."""
    print("\n" + "="*80)
    print("📊 PHASE 1: RESEARCH & INGESTION")
    print("="*80)
    graph = create_phase1_graph()
    result = graph.invoke(state)
    cost_tracker.print_summary()
    return result


def phase2_wrapper(state: Any) -> Any:
    """Phase 2: Content Planning wrapper."""
    print("\n" + "="*80)
    print("📋 PHASE 2: CONTENT PLANNING")
    print("="*80)
    graph = create_phase2_graph()
    result = graph.invoke(state)
    cost_tracker.print_summary()
    return result


def phase3_wrapper(state: Any) -> Any:
    """Phase 3: Dialogue Generation wrapper."""
    print("\n" + "="*80)
    print("💬 PHASE 3: DIALOGUE GENERATION")
    print("="*80)
    graph = create_phase3_graph()
    result = graph.invoke(state)
    cost_tracker.print_summary()
    return result


def phase4_wrapper(state: Any) -> Any:
    """Phase 4: Voice Synthesis wrapper."""
    print("\n" + "="*80)
    print("🎤 PHASE 4: VOICE SYNTHESIS")
    print("="*80)
    graph = create_phase4_graph()
    result = graph.invoke(state)
    cost_tracker.print_summary()
    return result


def phase5_wrapper(state: Any) -> Any:
    """Phase 5: Audio Post-Processing wrapper."""
    print("\n" + "="*80)
    print("🎵 PHASE 5: AUDIO POST-PROCESSING")
    print("="*80)
    graph = create_phase5_graph()
    result = graph.invoke(state)

    # Final cumulative summary + inject into state
    cost_tracker.print_summary()
    result["cost_summary"] = cost_tracker.get_summary()
    return result


# ==================== GRAPH CONSTRUCTION ====================

def create_main_orchestrator_graph():
    """Create the main orchestrator that chains all 5 phase subgraphs.

    Structure:
    - Each phase is added as a node
    - Nodes are connected in sequence: Phase1 → Phase2 → ... → Phase5
    - State flows through all phases, accumulating results

    Returns:
        CompiledGraph: Full podcast generation pipeline
    """
    workflow = StateGraph(dict)  # Using dict for flexibility during development

    # Add all 5 phase nodes
    workflow.add_node("phase1", phase1_wrapper)
    workflow.add_node("phase2", phase2_wrapper)
    workflow.add_node("phase3", phase3_wrapper)
    workflow.add_node("phase4", phase4_wrapper)
    workflow.add_node("phase5", phase5_wrapper)

    # Chain phases in sequence
    workflow.set_entry_point("phase1")
    workflow.add_edge("phase1", "phase2")
    workflow.add_edge("phase2", "phase3")
    workflow.add_edge("phase3", "phase4")
    workflow.add_edge("phase4", "phase5")
    workflow.add_edge("phase5", END)

    return workflow.compile()


# For backward compatibility, keep old function name
def create_query_producer_graph():
    """Backward compatibility wrapper.

    DEPRECATED: Use create_phase1_graph() from src.pipeline.phases instead.
    """
    return create_phase1_graph()
