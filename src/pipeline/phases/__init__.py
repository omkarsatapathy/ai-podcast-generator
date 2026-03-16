"""Phase subgraphs for modular podcast generation pipeline.

Each phase is an independent, composable subgraph:
- Phase 1: Research & Ingestion
- Phase 2: Content Planning
- Phase 3: Dialogue Generation
- Phase 4: Voice Synthesis
- Phase 5: Audio Post-Processing

These subgraphs can be:
1. Imported independently (e.g., query_producer imports Phase 1 only)
2. Composed into a main orchestrator graph for full pipeline execution
"""

from .phase1_graph import create_phase1_graph
from .phase2_graph import create_phase2_graph
from .phase3_graph import create_phase3_graph
from .phase4_graph import create_phase4_graph
from .phase5_graph import create_phase5_graph

__all__ = [
    "create_phase1_graph",
    "create_phase2_graph",
    "create_phase3_graph",
    "create_phase4_graph",
    "create_phase5_graph",
]
