# Phase 2: Content Planning - Implementation Plan

## Overview

Phase 2 transforms ranked source chunks from Phase 1 into a structured podcast plan with chapters and character personas. It consists of two sequential agents:
1. **Chapter Planner** - Clusters and sequences content into chapters with narrative arc
2. **Character Designer** - Creates distinct speaker personas

---

## Architecture & Data Flow

```
Phase 1 Output (50-80 ranked chunks)
    ↓
Chapter Planner Agent
    - Cluster similar chunks
    - Sequence into narrative arc (Setup → Exploration → Resolution)
    - Create chapter outlines with metadata
    ↓ (Chapter outlines)
Character Designer Agent
    - Analyze chapter content
    - Create 2-3 distinct character personas
    - Ensure complementary roles (Host, Expert, Skeptic)
    ↓
Phase 2 Output (Chapters + Personas)
    → Phase 3 (Dialogue Generation)
```

---

## Phase 2 State Definition

```python
class Phase2State(TypedDict):
    """State for Phase 2: Content Planning."""
    # Phase 1 inputs
    topic: str
    ranked_chunks: List[Dict[str, Any]]  # {content, source_url, relevance_score}

    # Chapter Planner outputs
    chapter_outlines: List[Dict[str, Any]]

    # Character Designer outputs
    character_personas: List[Dict[str, Any]]

    # Metadata
    num_speakers: int
    total_estimated_duration: float
```

---

## 1. Data Models

### Location: `src/models/chapter.py`

**Chapter Outline Schema:**
```python
@dataclass
class ChapterOutline:
    chapter_number: int          # 1-indexed
    title: str                   # Engaging chapter title
    act: str                     # "setup" | "explore" | "resolve"
    energy_level: str            # "high" | "medium" | "low"
    key_points: List[str]        # 3-5 key points to cover
    source_chunk_ids: List[int]  # Indices of chunks used
    transition_hook: str         # Sentence that teases next chapter
    estimated_duration_minutes: float
```

### Location: `src/models/character.py`

**Character Persona Schema:**
```python
@dataclass
class CharacterPersona:
    name: str                    # "Dr. Sarah Chen"
    role: str                    # "host" | "expert" | "skeptic"
    expertise_area: str          # Domain knowledge
    speaking_style: str          # Description of speaking patterns
    vocabulary_level: str        # "casual" | "moderate" | "technical"
    filler_patterns: List[str]   # ["you know", "like", "basically"]
    reaction_patterns: List[str] # ["Oh interesting!", "Wait really?"]
    disagreement_style: str      # How they challenge ideas
    laugh_frequency: str         # "rare" | "moderate" | "frequent"
    catchphrases: List[str]      # Signature phrases
    emotional_range: str         # How expressive they are
    tts_voice_id: str            # ElevenLabs voice ID
```

---

## 2. Chapter Planner Agent

### Location: `src/agents/phase2/chapter_planner.py`

### Core Algorithm

**Step 1: Analyze Content Landscape**
- Summarize each chunk (extract main topic)
- Identify natural clusters/themes
- Detect potential chapter boundaries

**Step 2: Apply 3-Act Structure**
- **Act 1 (Setup):** 2 chapters - Introduction, background, context
- **Act 2 (Exploration):** 3-5 chapters - Deep dives, different angles, debate
- **Act 3 (Resolution):** 1 chapter - Synthesis, future outlook, takeaways

**Step 3: Create Chapter Outline**
- Assign chunks to chapters
- Name chapters with engaging titles
- Set energy levels for pacing control
- Extract key points from chunks
- Create transition hooks between chapters

**Step 4: Validate & Optimize**
- Ensure all chunks are assigned
- Check time budget (target 25-28 min total)
- Verify energy level variation
- Adjust chapter boundaries if needed

### Implementation Details

```python
class ChapterPlannerAgent(BaseAgent):
    """
    Transforms ranked source chunks into chapter outlines.

    Input: 50-80 ranked chunks with topics and relevance scores
    Output: List[ChapterOutline] with 6-8 chapters
    """

    def __init__(self, llm_model: str = "claude-sonnet"):
        self.llm = get_llm(llm_model)

    def run(self, ranked_chunks: List[Dict], topic: str) -> List[ChapterOutline]:
        """
        Main execution method.

        Args:
            ranked_chunks: Output from Phase 1 Dedup + Relevance Scorer
            topic: Original podcast topic

        Returns:
            List of ChapterOutline objects
        """

        # Step 1: Analyze chunks
        chunk_analysis = self._analyze_chunks(ranked_chunks)

        # Step 2: Cluster into chapters
        chapter_clusters = self._cluster_into_chapters(chunk_analysis, topic)

        # Step 3: Apply narrative structure
        chapters_with_arc = self._apply_narrative_arc(chapter_clusters)

        # Step 4: Generate detailed outlines
        chapter_outlines = self._generate_outlines(chapters_with_arc, ranked_chunks)

        # Step 5: Validate
        self._validate_chapters(chapter_outlines)

        return chapter_outlines

    def _analyze_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Use LLM to extract topic and summary from each chunk.

        Prompt: "Analyze this content chunk and extract:
                 - Main topic (1-2 words)
                 - Summary (1-2 sentences)
                 - Key subtopics (list)
                 Format as JSON"
        """
        # Batch process chunks (5-10 at a time to manage tokens)
        pass

    def _cluster_into_chapters(self, analysis: List[Dict], topic: str) -> List[List[Dict]]:
        """
        Use embedding similarity and LLM to group chunks naturally.

        Approach:
        1. Compute embeddings for each chunk summary
        2. Use FAISS to find natural clusters
        3. Use LLM to validate cluster coherence
        4. Generate cluster titles
        """
        pass

    def _apply_narrative_arc(self, clusters: List[List[Dict]]) -> List[List[Dict]]:
        """
        Reorganize clusters into 3-act structure.

        - Acts 1: Setup (2 chapters) - Introduction, background
        - Act 2: Exploration (3-5 chapters) - Deep dives, perspectives
        - Act 3: Resolution (1 chapter) - Synthesis, outlook
        """
        pass

    def _generate_outlines(self, chapters: List[List[Dict]],
                          ranked_chunks: List[Dict]) -> List[ChapterOutline]:
        """
        Use LLM to generate detailed chapter outlines.

        Prompt: "Create a chapter outline for a podcast chapter.
                 Content topics: [...]

                 Generate:
                 - Engaging title (5-8 words)
                 - 3-5 key points to cover
                 - Energy level: high/medium/low
                 - Transition hook to next chapter
                 - Estimated duration in minutes

                 Format as JSON"
        """
        pass

    def _validate_chapters(self, chapters: List[ChapterOutline]):
        """
        Validation checks:
        - All chunks assigned
        - Total duration ~25-28 min
        - Energy levels vary (not all high/low)
        - Each chapter has 3-5 key points
        - No chapter exceeds 5 minutes or is below 2 minutes
        """
        pass
```

### LLM Configuration

- **Model:** Claude Sonnet (fast tier)
- **Temperature:** 0.7 (balanced creativity for titles/hooks)
- **Context Window:** Use summaries, not full chunks (token management)

### Key Prompts

**Prompt 1: Chunk Analysis**
```
You are analyzing source material for a podcast episode about: [TOPIC]

Analyze this content chunk:
[CHUNK_TEXT]

Extract:
1. Main topic (1-2 words)
2. Summary (1-2 sentences max)
3. Key subtopics (list, 3-5 items)
4. Tone (factual/opinion/debate/etc)

Format as JSON with keys: topic, summary, subtopics, tone
```

**Prompt 2: Chapter Outline**
```
You are creating a podcast chapter outline.

Topic: [MAIN_TOPIC]
Chapter Content Areas: [CLUSTER_TOPICS]

Create an engaging podcast chapter with:
- Title (5-8 words, engaging, specific to this chapter)
- Energy level: "high" (debate/tension), "medium" (explanation), or "low" (summary/synthesis)
- Key points to cover: 3-5 specific points
- Transition hook: 1 sentence that teases the next chapter
- Estimated duration: X minutes (usually 4-5 min per chapter)

Format as JSON:
{
  "title": "...",
  "energy_level": "...",
  "key_points": [...],
  "transition_hook": "...",
  "estimated_duration_minutes": X
}
```

---

## 3. Character Designer Agent

### Location: `src/agents/phase2/character_designer.py`

### Core Algorithm

**Step 1: Analyze Content Complexity**
- Number of topics/subtopics
- Debate potential (controversial angles?)
- Technical depth required
- Required expertise areas

**Step 2: Define Role Archetypes**
- **Host (Curious Generalist):** Asks questions, connects ideas, represents listener
- **Expert (Domain Authority):** Deep knowledge, uses jargon, provides substance
- **Skeptic (Devil's Advocate):** Challenges assumptions, raises concerns, debates

**Step 3: Create Distinct Personas**
- Generate unique speaking styles
- Assign complementary vocabularies
- Create personality quirks (fillers, reactions, catchphrases)
- Assign appropriate TTS voice

**Step 4: Ensure Consistency**
- Personas persist across all chapters
- Roles remain consistent
- Speaking patterns are distinct and recognizable

### Implementation Details

```python
class CharacterDesignerAgent(BaseAgent):
    """
    Creates distinct character personas for podcast speakers.

    Input: Chapter outlines and topic
    Output: List[CharacterPersona] - typically 2-3 distinct speakers
    """

    def __init__(self, num_speakers: int = 3, llm_model: str = "claude-sonnet"):
        self.num_speakers = num_speakers
        self.llm = get_llm(llm_model)
        self.available_voices = get_elevenlab_voices()

    def run(self, topic: str, chapters: List[ChapterOutline]) -> List[CharacterPersona]:
        """
        Main execution method.

        Args:
            topic: Original podcast topic
            chapters: ChapterOutline objects from Chapter Planner

        Returns:
            List of CharacterPersona objects (typically 2-3)
        """

        # Step 1: Analyze content requirements
        content_analysis = self._analyze_content(topic, chapters)

        # Step 2: Define role configuration
        roles = self._assign_roles(content_analysis, self.num_speakers)

        # Step 3: Generate persona profiles
        personas = self._generate_personas(topic, roles, content_analysis)

        # Step 4: Assign TTS voices
        personas = self._assign_voices(personas)

        # Step 5: Validate personas
        self._validate_personas(personas)

        return personas

    def _analyze_content(self, topic: str, chapters: List[ChapterOutline]) -> Dict:
        """
        Analyze chapters to determine required expertise and debate potential.

        Returns: {
            "expertise_areas": [...],
            "debate_potential": bool,
            "technical_depth": "high/medium/low",
            "complexity_score": float
        }
        """
        pass

    def _assign_roles(self, analysis: Dict, num_speakers: int) -> List[str]:
        """
        Determine which role archetypes to use.

        - 2 speakers: Host + Expert
        - 3 speakers: Host + Expert + Skeptic (recommended)

        The Host is always present, the others vary by content.
        """
        pass

    def _generate_personas(self, topic: str, roles: List[str],
                          analysis: Dict) -> List[CharacterPersona]:
        """
        Use LLM to generate detailed personas for each role.

        Prompt: "Create a podcast character persona for the role: [ROLE]

                 Podcast topic: [TOPIC]
                 Required expertise: [AREAS]
                 Content complexity: [LEVEL]

                 Generate a unique character with:
                 - Full name (realistic, diverse backgrounds)
                 - Specific expertise/background
                 - Distinct speaking style (2-3 sentences describing how they talk)
                 - Vocabulary level: casual/moderate/technical
                 - Filler words they use (5-7 examples)
                 - Reaction patterns (how they respond, 5-7 examples)
                 - Disagreement style (how they politely challenge)
                 - Laugh frequency and style
                 - 2-3 catchphrases
                 - Emotional range (how expressive they are)

                 Create interesting, realistic characters that work well together.
                 Format as JSON"
        """
        pass

    def _assign_voices(self, personas: List[CharacterPersona]) -> List[CharacterPersona]:
        """
        Assign ElevenLabs voice IDs to each persona.

        Criteria:
        - Different genders/accents for distinctness
        - Voice matches character background (e.g., Dr. Chen = Asian accent option)
        - Variety in age/tone

        Returns personas with tts_voice_id populated
        """
        pass

    def _validate_personas(self, personas: List[CharacterPersona]):
        """
        Validation checks:
        - All required fields populated
        - Distinct speaking styles
        - Voice IDs exist in ElevenLabs API
        - Names are unique
        - Roles are non-overlapping
        """
        pass
```

### LLM Configuration

- **Model:** Claude Sonnet (creative task - needs nuance)
- **Temperature:** 0.8 (more creative for character development)
- **System Prompt:** Emphasis on creating realistic, distinct, professionally-voiced characters

### Key Prompts

**Prompt 1: Persona Generation**
```
You are creating a podcast character for a sophisticated audience.

Podcast topic: [TOPIC]
Character role: [HOST/EXPERT/SKEPTIC]
Required expertise: [AREAS]
Content complexity: [LEVEL]

Create a unique, realistic character with:

1. Name: Full name that reflects expertise/background
2. Background: 1-2 sentence professional background
3. Speaking style: How they naturally speak (2-3 sentences)
4. Expertise area: Specific to this topic
5. Vocabulary level: "casual" (simple words, friendly), "moderate" (professional), or "technical" (jargon-heavy)
6. Filler patterns: Words they say when thinking (e.g., "you know", "like", "basically")
7. Reaction patterns: How they respond to ideas (e.g., "Oh interesting!", "That's brilliant!")
8. Disagreement style: How they politely challenge ideas
9. Laugh frequency: "rare", "moderate", or "frequent"
10. Laugh style: When and how they laugh
11. Catchphrases: 2-3 signature phrases unique to them
12. Emotional range: How expressive are they (1-10 scale with examples)

Create interesting characters that complement each other (if multi-speaker).

Output: JSON with all fields above
```

**Prompt 2: Character Ensemble Validation**
```
Review this ensemble of podcast characters:

[CHARACTER 1]
[CHARACTER 2]
[CHARACTER 3]

Ensure:
1. Each has a distinct speaking style and voice
2. Roles complement each other (not redundant)
3. Vocabulary levels vary appropriately
4. Names are realistic and diverse
5. Expertise areas don't completely overlap

Suggest any adjustments to increase distinctness and dynamic interaction.
```

---

## 4. Phase 2 Graph Implementation

### Location: `src/pipeline/phases/phase2_graph.py`

```python
"""Phase 2: Content Planning Subgraph."""

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from src.agents.phase2.chapter_planner import ChapterPlannerAgent
from src.agents.phase2.character_designer import CharacterDesignerAgent


class Phase2State(TypedDict):
    """State for Phase 2: Content Planning."""
    # Phase 1 outputs
    topic: str
    ranked_chunks: List[Dict[str, Any]]
    num_speakers: int

    # Phase 2 outputs
    chapter_outlines: List[Dict[str, Any]]
    character_personas: List[Dict[str, Any]]

    # Metadata
    total_estimated_duration: float


def chapter_planner_node(state: Phase2State) -> Phase2State:
    """Execute Chapter Planner agent."""
    agent = ChapterPlannerAgent()

    chapter_outlines = agent.run(
        ranked_chunks=state["ranked_chunks"],
        topic=state["topic"]
    )

    # Calculate total duration
    total_duration = sum(ch.estimated_duration_minutes for ch in chapter_outlines)

    state["chapter_outlines"] = [ch.to_dict() for ch in chapter_outlines]
    state["total_estimated_duration"] = total_duration

    return state


def character_designer_node(state: Phase2State) -> Phase2State:
    """Execute Character Designer agent."""
    agent = CharacterDesignerAgent(num_speakers=state["num_speakers"])

    personas = agent.run(
        topic=state["topic"],
        chapters=state["chapter_outlines"]
    )

    state["character_personas"] = [p.to_dict() for p in personas]

    return state


def create_phase2_graph():
    """Create and compile the Phase 2 (Content Planning) subgraph."""
    workflow = StateGraph(Phase2State)

    # Add nodes
    workflow.add_node("chapter_planner", chapter_planner_node)
    workflow.add_node("character_designer", character_designer_node)

    # Set flow: Chapter Planner → Character Designer
    workflow.set_entry_point("chapter_planner")
    workflow.add_edge("chapter_planner", "character_designer")
    workflow.add_edge("character_designer", END)

    return workflow.compile()
```

---

## 5. LLM Prompts Module

### Location: `src/llm/prompts.py`

Create a centralized prompts module with:

```python
CHAPTER_PLANNER_PROMPTS = {
    "analyze_chunks": "...",
    "cluster_validation": "...",
    "generate_outline": "...",
}

CHARACTER_DESIGNER_PROMPTS = {
    "analyze_content": "...",
    "generate_persona": "...",
    "ensemble_validation": "...",
}
```

---

## 6. Testing Strategy

### Location: `tests/test_phase2/`

**Test Files:**
1. `test_chapter_planner.py`
   - Test chunk analysis
   - Test narrative arc structure
   - Test duration calculation
   - Test chapter outline generation

2. `test_character_designer.py`
   - Test persona generation
   - Test role assignment
   - Test voice assignment
   - Test ensemble compatibility

3. `test_phase2_graph.py`
   - Test full Phase 2 graph execution
   - Test state flow between agents
   - Test output validation

**Test Data:**
- Sample ranked chunks from Phase 1
- Expected chapter structures
- Expected persona formats

---

## 7. Configuration & Settings

### Location: `config/phase2_settings.py`

```python
CHAPTER_PLANNER_CONFIG = {
    "min_chapters": 6,
    "max_chapters": 8,
    "target_duration_minutes": 26,  # 25-28 min podcast
    "min_chapter_duration": 2.0,
    "max_chapter_duration": 5.0,
    "clustering_method": "embedding_faiss",  # or "llm_based"
}

CHARACTER_DESIGNER_CONFIG = {
    "default_num_speakers": 3,
    "voice_provider": "elevenlabs",
    "available_voices": [...],
    "voice_filter_criteria": {
        "exclude_genders": [],
        "preferred_accents": ["neutral", "american"],
    }
}
```

---

## 8. Integration Points

### With Phase 1 (Input)
- Receives: `ranked_chunks`, `topic`, `num_speakers`
- Requires: Chunks have `{content, source_url, relevance_score}`

### With Phase 3 (Output)
- Provides: `chapter_outlines`, `character_personas`
- Phase 3 uses these to generate dialogue per chapter

---

## 9. Implementation Schedule

### Week 1: Foundation
- [ ] Create data models (Chapter, Character)
- [ ] Set up test fixtures and sample data
- [ ] Create LLM prompts module

### Week 2: Chapter Planner
- [ ] Implement `_analyze_chunks()`
- [ ] Implement `_cluster_into_chapters()`
- [ ] Implement `_apply_narrative_arc()`
- [ ] Unit tests

### Week 3: Character Designer
- [ ] Implement `_analyze_content()`
- [ ] Implement `_generate_personas()`
- [ ] Implement `_assign_voices()`
- [ ] Unit tests

### Week 4: Integration & Polish
- [ ] Build Phase 2 graph
- [ ] Integration tests
- [ ] Performance optimization
- [ ] Documentation

---

## 10. Success Criteria

✅ **Functional Requirements:**
- Chapter Planner produces 6-8 chapters with proper narrative arc
- Character Designer creates 2-3 distinct personas
- All chapters assigned with chunks
- Total duration within 25-28 minutes
- Chapters have energy level variation

✅ **Quality Requirements:**
- Chapter titles are engaging and topic-specific
- Character personas are realistic and distinct
- Transition hooks create flow between chapters
- Personas work well together (no role overlap)

✅ **Technical Requirements:**
- Phase 2 graph integrates with Phase 1 and Phase 3
- All LLM calls stay within token budgets
- Tests achieve >90% code coverage
- Documentation is complete and clear

---

## Notes & Considerations

### Embedding Models for Clustering
- Use `sentence-transformers/all-MiniLM-L6-v2` (lightweight, accurate)
- Alternative: Amazon Titan Embeddings (managed service)

### LLM Cost Optimization
- Use Sonnet for fast operations (chapter titles, persona generation)
- Batch chunk analysis (5-10 chunks per API call)
- Cache prompts for repeat operations

### Future Enhancements
- **Pacing Controller (v2):** Dynamically adjust energy levels to optimize listener retention
- **Character Consistency Checker:** Validate personas remain consistent across chapters
- **Multi-language Support:** Generate personas for non-English podcasts

---

## References

- **Design Document:** Pages 11-14 (Phase 2: Content Planning)
- **Architecture:** Modular pipeline with independent subgraphs
- **Data Flow:** Phase 1 Output → Chapter Planner → Character Designer → Phase 3 Input
