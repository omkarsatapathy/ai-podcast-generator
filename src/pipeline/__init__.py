"""Pipeline orchestration and graph workflows.

Main exports:
- create_main_orchestrator_graph: Full 5-phase pipeline
- Phase subgraphs: Available from src.pipeline.phases

For backward compatibility:
- create_query_producer_graph: Wrapper for Phase 1
- QueryProducerState: Imported from Phase 1 subgraph
"""

# Main orchestrator
from src.pipeline.graph import create_main_orchestrator_graph

# Phase subgraphs
from src.pipeline.phases import (
    create_phase1_graph,
    create_phase2_graph,
    create_phase3_graph,
    create_phase4_graph,
    create_phase5_graph,
)

# Phase 1 state and backward compatibility
from src.pipeline.phases.phase1_graph import Phase1State as QueryProducerState

# Backward compatibility wrapper
def create_query_producer_graph():
    """Deprecated: Use create_phase1_graph() instead."""
    return create_phase1_graph()


__all__ = [
    # Main orchestrator
    "create_main_orchestrator_graph",
    # Phase subgraphs
    "create_phase1_graph",
    "create_phase2_graph",
    "create_phase3_graph",
    "create_phase4_graph",
    "create_phase5_graph",
    # Backward compatibility
    "create_query_producer_graph",
    "QueryProducerState",
]
