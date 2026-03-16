# Modular Pipeline Architecture ✅

## Overview

Your podcast generation pipeline has been refactored into a **modular, composable architecture** where each phase is an independent, self-contained subgraph.

### Previous Architecture (Monolithic)
```
src/pipeline/graph.py
└── One large file with all 7 Query Producer nodes
    └── Difficult to extend when adding Phase 2, 3, 4, 5
```

### New Architecture (Modular)
```
src/pipeline/
├── graph.py                          ← Main orchestrator (chains all phases)
└── phases/
    ├── __init__.py                   ← Phase exports
    ├── phase1_graph.py               ← Research & Ingestion subgraph
    ├── phase2_graph.py               ← Content Planning subgraph
    ├── phase3_graph.py               ← Dialogue Generation subgraph
    ├── phase4_graph.py               ← Voice Synthesis subgraph
    └── phase5_graph.py               ← Audio Post-Processing subgraph
```

---

## How It Works

### 1. Independent Subgraphs

Each phase is a **complete, compiled LangGraph** with its own:
- State definition (e.g., `Phase1State`)
- Nodes (e.g., 7 nodes for Phase 1)
- Edges and routing logic
- Entry/exit points

**Example: Phase 1**
```python
# src/pipeline/phases/phase1_graph.py

class Phase1State(TypedDict):
    topic: str
    freshness: str
    queries: List[Dict[str, Any]]
    # ... other fields

def create_phase1_graph():
    workflow = StateGraph(Phase1State)
    # Add 7 nodes: initialize, classify_freshness, seed_search, etc.
    return workflow.compile()  # ✅ Fully compiled subgraph
```

### 2. Standalone Usage (Phase 1)

The `QueryProducerAgent` imports **only Phase 1 subgraph**:

```python
# src/agents/phase1/query_producer.py

from src.pipeline.phases.phase1_graph import create_phase1_graph

class QueryProducerAgent:
    def __init__(self):
        self.graph = create_phase1_graph()  # ✅ Only Phase 1

    def run(self, topic: str) -> QueryProducerOutput:
        initial_state = {"topic": topic, "freshness": "", ...}
        final_state = self.graph.invoke(initial_state)
        return QueryProducerOutput(...)
```

**Benefits:**
- Phase 1 doesn't "see" Phase 2, 3, 4, 5 code
- Can run and test Phase 1 independently
- Adding Phase 6 won't break query_producer.py

### 3. Full Pipeline (Main Orchestrator)

The orchestrator imports all phases and chains them:

```python
# src/pipeline/graph.py

def create_main_orchestrator_graph():
    workflow = StateGraph(dict)

    # Add each phase as a node
    workflow.add_node("phase1", phase1_wrapper)
    workflow.add_node("phase2", phase2_wrapper)
    # ... etc

    # Chain them in sequence
    workflow.set_entry_point("phase1")
    workflow.add_edge("phase1", "phase2")
    workflow.add_edge("phase2", "phase3")
    # ... etc

    return workflow.compile()  # ✅ Full 5-phase pipeline
```

**Data Flow:**
```
User Input (topic: "AI in healthcare")
    ↓
Phase 1 (Research)
    → Classifies freshness
    → Generates 10 queries
    → Executes searches
    → Outputs: {queries: [...], results: [...]}
    ↓
Phase 2 (Planning) [Placeholder]
    → Takes Phase 1 output
    → Plans chapters
    → Designs characters
    → Outputs: {chapter_outlines: [...], personas: [...]}
    ↓
Phase 3 (Dialogue) [Placeholder]
    → Takes Phase 2 output
    → Generates dialogue
    → Adds naturalness
    → Outputs: {scripts: [...]}
    ↓
Phase 4 (TTS) [Placeholder]
    → Takes Phase 3 output
    → Synthesizes audio
    → Outputs: {audio_files: [...]}
    ↓
Phase 5 (Post-Processing) [Placeholder]
    → Takes Phase 4 output
    → Overlays, mixes, stitches
    → Outputs: {final_podcast_mp3: "podcast.mp3"}
```

---

## Directory Structure

```
src/
├── pipeline/
│   ├── __init__.py
│   │   ├── create_main_orchestrator_graph()  ← Full pipeline
│   │   ├── create_phase1_graph()             ← Phase 1
│   │   ├── create_phase2_graph()             ← Phase 2
│   │   ├── create_phase3_graph()             ← Phase 3
│   │   ├── create_phase4_graph()             ← Phase 4
│   │   ├── create_phase5_graph()             ← Phase 5
│   │   └── create_query_producer_graph()     ← Backward compat
│   │
│   ├── graph.py
│   │   └── create_main_orchestrator_graph()  ← Orchestrator
│   │
│   └── phases/
│       ├── __init__.py                        ← Phase exports
│       ├── phase1_graph.py                    ← 7 nodes (COMPLETE)
│       ├── phase2_graph.py                    ← Placeholder
│       ├── phase3_graph.py                    ← Placeholder
│       ├── phase4_graph.py                    ← Placeholder
│       └── phase5_graph.py                    ← Placeholder
│
├── agents/
│   └── phase1/
│       └── query_producer.py
│           └── Imports: create_phase1_graph() ✅
```

---

## How to Use

### Option 1: Query Producer (Phase 1 Only)

```python
from src.agents.phase1.query_producer import QueryProducerAgent

agent = QueryProducerAgent()
result = agent.run("SpaceX Starship launch 2024")

# Output: QueryProducerOutput with 10 queries and search results
```

### Option 2: Full Pipeline (All 5 Phases)

```python
from src.pipeline.graph import create_main_orchestrator_graph

graph = create_main_orchestrator_graph()
initial_state = {"topic": "AI in healthcare"}
final_result = graph.invoke(initial_state)

# Output: Final podcast MP3 (when all phases implemented)
```

### Option 3: Individual Phase (Future)

```python
from src.pipeline.phases.phase2_graph import create_phase2_graph
from src.pipeline.phases.phase3_graph import create_phase3_graph

# Use only Phase 2 and 3 for testing
```

---

## Backward Compatibility

Old code still works:

```python
# This still works (but issues deprecation warning)
from src.pipeline.graph import create_query_producer_graph
graph = create_query_producer_graph()  # ← Calls create_phase1_graph()

# This also works (recommended)
from src.pipeline import QueryProducerState
```

---

## Adding a New Phase

When you implement Phase 2 (Content Planning), follow this pattern:

### Step 1: Create Phase 2 Graph
```python
# src/pipeline/phases/phase2_graph.py

class Phase2State(TypedDict):
    # Phase 1 output becomes Phase 2 input
    queries: List[Dict[str, Any]]

    # Phase 2 specific
    chapter_outlines: List[Dict[str, Any]]
    character_personas: List[Dict[str, Any]]

def create_phase2_graph():
    workflow = StateGraph(Phase2State)

    # Add Chapter Planner node
    workflow.add_node("chapter_planner", chapter_planner_node)

    # Add Character Designer node
    workflow.add_node("character_designer", character_designer_node)

    # ... connect nodes ...

    return workflow.compile()
```

### Step 2: Update phases/__init__.py
```python
from .phase2_graph import create_phase2_graph

__all__ = ["create_phase2_graph", ...]
```

### Step 3: Create Agent if Needed
```python
# src/agents/phase2/chapter_planner.py

from src.pipeline.phases.phase2_graph import create_phase2_graph

class ChapterPlannerAgent:
    def __init__(self):
        self.graph = create_phase2_graph()
```

That's it! The orchestrator and Phase 2 subgraph remain independent.

---

## Testing

Run the test suite:

```bash
python test_modular_pipeline.py
```

**Tests:**
1. ✅ **Phase 1 Standalone** - Query Producer runs independently
2. ✅ **Full Pipeline** - All 5 phases chain correctly
3. ✅ **Phase Imports** - Each phase can be imported separately

All tests pass! ✅

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Orchestrator                          │
│              create_main_orchestrator_graph()                │
└─────────────────────────────────────────────────────────────┘
            ↓           ↓           ↓           ↓           ↓
    ┌─────────────┬──────────┬──────────┬──────────┬────────────┐
    │   Phase 1   │  Phase 2 │  Phase 3 │  Phase 4 │  Phase 5   │
    │ (Research)  │ (Planning│(Dialogue)│ (TTS)    │(Post-Proc) │
    │     ✅      │    🔄    │    🔄    │    🔄    │     🔄     │
    └─────────────┴──────────┴──────────┴──────────┴────────────┘
         ↑            ↑            ↑            ↑            ↑
         │            │            │            │            │
    QueryProducer  ChapterPlanner  Dialogue   TTSRouter   Overlap
    (standalone)      Agent        Engine     Agent       Engine

    ✅ = Complete & tested
    🔄 = Placeholder (to be implemented)
```

---

## Key Points ✨

1. **Independence:** Each phase is a completely compiled LangGraph
   - Query Producer imports only Phase 1
   - Adding Phase 6 won't affect existing code

2. **Composability:** Phases can be combined in any order
   - Full pipeline: Phase 1 → 2 → 3 → 4 → 5
   - Custom pipeline: Phase 1 → 3 → 5 (if needed)

3. **Testability:** Each phase can be tested independently
   - Test Phase 1 without implementing Phases 2-5
   - Test Phase 3 in isolation

4. **Scalability:** Adding nodes is easy
   - Implement Phase 2 without touching other phases
   - Refactor Phase 3 internally without affecting orchestrator

5. **Maintainability:** Code is organized and modular
   - Clear separation of concerns
   - Easy to locate and modify phase logic

---

## Next Steps 🚀

1. ✅ Phase 1: Research & Ingestion (COMPLETE)
2. 🔄 Phase 2: Content Planning (Chapter Planner, Character Designer)
3. 🔄 Phase 3: Dialogue Generation (Dialogue Engine, Naturalness Injector, etc.)
4. 🔄 Phase 4: Voice Synthesis (TTS Router)
5. 🔄 Phase 5: Audio Post-Processing (Overlap Engine, Post-Processor, etc.)

For each phase, simply:
1. Implement the agents in `src/agents/phase{N}/`
2. Update `src/pipeline/phases/phase{N}_graph.py`
3. Update the phase state definition if needed
4. Everything else stays compatible! ✨
