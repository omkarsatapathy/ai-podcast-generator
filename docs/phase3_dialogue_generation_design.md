# Phase 3: Dialogue Generation - Implementation Design Document

**Version:** 2.0
**Date:** March 2026
**TTS Provider:** Google Cloud Text-to-Speech (WaveNet/Neural2)
**Architecture:** LangGraph Multi-Node Pipeline
**Author:** Omkar

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Graph Node Structure](#graph-node-structure)
4. [Node 1: Dialogue Engine](#node-1-dialogue-engine)
5. [Node 2: Expert Content Expander](#node-2-expert-content-expander)
6. [Node 3: Naturalness Injector](#node-3-naturalness-injector)
7. [Node 4: Fact-Checker Agent](#node-4-fact-checker-agent)
8. [Node 5: QA Reviewer Agent](#node-5-qa-reviewer-agent)
9. [Node 6: SSML Annotator (Google TTS)](#node-6-ssml-annotator-google-tts)
10. [Data Schemas](#data-schemas)
11. [Graph Routing Logic](#graph-routing-logic)
12. [Retry and Fallback Strategies](#retry-and-fallback-strategies)
13. [Quality Gates and Validation](#quality-gates-and-validation)
14. [Google TTS Configuration](#google-tts-configuration)
15. [Testing and Validation](#testing-and-validation)

---

## Executive Summary

Phase 3 transforms chapter outlines and character personas into SSML-annotated dialogue scripts ready for Text-to-Speech synthesis. This phase is the **creative heart** of the podcast generation pipeline, where factual content becomes conversational storytelling.

### Key Design Principles

1. **Multi-Node Architecture:** Phase 3 is implemented as **6+ distinct LangGraph nodes** (not a monolithic agent)
2. **Beat-by-Beat Generation:** Each chapter's 5 beats generated sequentially with context continuity
3. **Expert Content Expansion:** Dedicated node enriches expert utterances with detailed explanations while maintaining flow
4. **Graph-Based Retry Logic:** All fallbacks, retries, and quality checks handled via conditional edges in LangGraph
5. **Google TTS Optimization:** SSML output specifically formatted for Google Cloud TTS WaveNet/Neural2 voices

### Phase 3 Input
```json
{
  "chapter_number": 1,
  "title": "The AI Revolution Begins",
  "act": "setup",
  "energy_level": "high",
  "key_points": ["What is AI?", "Brief history", "Why it matters now"],
  "source_chunk_ids": [12, 45, 67, 89, 103],
  "transition_hook": "But how did we get here?",
  "estimated_duration_minutes": 4.5,
  "characters": {
    "host": {
      "name": "Dr. Sarah Chen",
      "role": "host",
      "speaking_style": "Uses analogies and metaphors heavily...",
      "vocabulary_level": "casual",
      "tts_voice_id": "en-US-Neural2-F"
    },
    "expert": {
      "name": "Prof. Marcus Thompson",
      "role": "expert",
      "speaking_style": "Technical but accessible...",
      "vocabulary_level": "moderate",
      "tts_voice_id": "en-US-Neural2-D"
    },
    "skeptic": {
      "name": "Jamie Rodriguez",
      "role": "skeptic",
      "speaking_style": "Probing questions...",
      "vocabulary_level": "casual",
      "tts_voice_id": "en-US-Neural2-C"
    }
  },
  "source_chunks": [
    {
      "chunk_id": 12,
      "content": "Artificial Intelligence refers to...",
      "source_url": "https://example.com/ai-basics",
      "relevance_score": 0.94
    }
  ]
}
```

### Phase 3 Output
```json
{
  "chapter_number": 1,
  "utterances": [
    {
      "utterance_id": "ch1_u001",
      "speaker": "Dr. Sarah Chen",
      "beat": 1,
      "text_clean": "Welcome back! Today we're diving into something that's reshaping our world faster than most people realize.",
      "text_with_naturalness": "[PACE:fast] Welcome back! [PAUSE:short] Today we're diving into something that's reshaping our world faster than most people realize.",
      "text_ssml": "<prosody rate=\"fast\">Welcome back!</prosody> <break time=\"400ms\"/> Today we're diving into something that's reshaping our world faster than most people realize.",
      "intent": "hook",
      "emotion": "excited",
      "grounding_chunk_ids": [],
      "tts_voice_id": "en-US-Neural2-F",
      "estimated_duration_seconds": 5.2
    }
  ],
  "total_utterances": 42,
  "estimated_chapter_duration": 4.7,
  "quality_checks_passed": true,
  "validation_metadata": {
    "fact_check_claims_verified": 8,
    "qa_review_iterations": 1,
    "naturalness_markers_added": 15
  }
}
```

---

## System Architecture Overview

### Phase 3 Graph Flow

```
                                    ┌─────────────────────┐
                                    │  PHASE 3 ENTRY      │
                                    │  (Chapter Input)    │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │  NODE 1:            │
                                    │  Dialogue Engine    │
                                    │  (Beat Generator)   │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │  NODE 2:            │
                                    │  Expert Content     │
                                    │  Expander           │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │  NODE 3:            │
                                    │  Naturalness        │
                                    │  Injector           │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │  NODE 4:            │
                                    │  Fact-Checker       │
                                    └──────────┬──────────┘
                                               │
                                ┌──────────────┴──────────────┐
                                │                             │
                          [Claims Valid?]              [Claims Invalid?]
                                │                             │
                                ▼                             ▼
                    ┌─────────────────────┐       ┌─────────────────────┐
                    │  NODE 5:            │       │  RETRY: Route back  │
                    │  QA Reviewer        │       │  to Dialogue Engine │
                    └──────────┬──────────┘       │  with corrections   │
                               │                  └─────────────────────┘
                ┌──────────────┴──────────────┐
                │                             │
          [QA Pass?]                    [QA Issues?]
                │                             │
                ▼                             ▼
    ┌─────────────────────┐       ┌─────────────────────┐
    │  NODE 6:            │       │  RETRY: Route back  │
    │  SSML Annotator     │       │  to Dialogue Engine │
    │  (Google TTS)       │       │  with QA feedback   │
    └──────────┬──────────┘       └─────────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │  PHASE 3 OUTPUT     │
    │  (SSML Script)      │
    └─────────────────────┘
```

### Graph State Schema

```python
class Phase3GraphState(TypedDict):
    # Input from Phase 2
    chapter_input: ChapterInput

    # Working state across nodes
    current_beat: int  # 1-5
    beat_history: List[Dict]  # Previous beats for context
    utterances: List[Utterance]

    # Quality tracking
    retry_count: int
    max_retries: int  # Default: 2
    fact_check_issues: List[str]
    qa_review_issues: List[str]

    # Node completion flags
    dialogue_complete: bool
    expansion_complete: bool
    naturalness_complete: bool
    fact_check_complete: bool
    qa_review_complete: bool
    ssml_complete: bool

    # Final output
    final_output: ChapterDialogueOutput
```

---

## Graph Node Structure

### Node Responsibilities

| Node | Input | Output | Retry Logic | Max Duration |
|------|-------|--------|-------------|--------------|
| **Node 1: Dialogue Engine** | Chapter outline, personas, beat number | Clean dialogue for current beat | Graph routes back if fact-check fails | 30s per beat |
| **Node 2: Expert Content Expander** | Beat dialogue | Expanded expert utterances | Graph routes back if too verbose (>150 words/utterance) | 20s |
| **Node 3: Naturalness Injector** | Clean dialogue | Dialogue with naturalness markers | None (deterministic rules + LLM) | 15s |
| **Node 4: Fact-Checker** | Dialogue with markers | Validation report | Graph routes to Node 1 if claims unsupported | 45s |
| **Node 5: QA Reviewer** | Complete chapter script | Quality assessment | Graph routes to Node 1 if issues found | 30s |
| **Node 6: SSML Annotator** | Validated script | SSML-formatted output | None (rule-based conversion) | 10s |

### Graph Routing Strategy

```python
# Conditional edge logic (pseudocode)

def route_after_dialogue_engine(state):
    """After Node 1 completes"""
    if state.current_beat < 5:
        return "dialogue_engine"  # Continue to next beat
    else:
        return "expert_expander"  # All beats done, move to expansion

def route_after_fact_checker(state):
    """After Node 4 completes"""
    if state.fact_check_issues:
        if state.retry_count < state.max_retries:
            state.retry_count += 1
            return "dialogue_engine"  # Retry with corrections
        else:
            return "qa_reviewer"  # Give up, move forward with warnings
    else:
        return "qa_reviewer"  # All facts verified

def route_after_qa_reviewer(state):
    """After Node 5 completes"""
    if state.qa_review_issues and state.retry_count < state.max_retries:
        state.retry_count += 1
        return "dialogue_engine"  # Targeted rewrite
    else:
        return "ssml_annotator"  # Move to final step
```

---

## Node 1: Dialogue Engine

### Purpose
Generate clean, multi-turn dialogue for a single beat within the chapter, following the 5-beat narrative structure (Hook → Context → Deep Dive → Aha Moment → Wrap).

### Beat-by-Beat Architecture

**Strategy:** Instead of generating the entire chapter in one LLM call, generate each beat sequentially:
- Beat 1 → Beat 2 → Beat 3 → Beat 4 → Beat 5
- Each beat generation sees the previous beats' dialogue as context
- This prevents token limit issues and improves coherence

### System Prompt Template

```markdown
# ROLE
You are an expert podcast dialogue writer specializing in educational multi-speaker conversations. You write natural, engaging dialogue that sounds like real people discussing a topic, not scripted interviews.

# TASK
Generate dialogue for **Beat {beat_number}** of Chapter {chapter_number}: "{chapter_title}".

# CONTEXT
## Chapter Information
- **Act:** {act} (setup/explore/resolve)
- **Energy Level:** {energy_level} (high/medium/low)
- **Key Points to Cover:** {key_points}
- **Target Duration:** {estimated_duration_minutes} minutes total (this beat: ~{beat_duration} seconds)

## Characters
{for each character:}
- **{character.name}** ({character.role}):
  - Speaking Style: {character.speaking_style}
  - Vocabulary: {character.vocabulary_level}
  - Personality Traits: {character.personality_traits}

## Beat {beat_number} Objective
{if beat == 1:}
**HOOK (15-30 seconds):** Open with a provocative statement, surprising fact, or compelling question that grabs attention. The {host_name} should make the listener want to keep listening. Create immediate curiosity.

{if beat == 2:}
**CONTEXT BUILDING (60-90 seconds):** Frame the topic. The {expert_name} provides necessary background. The {host_name} asks clarifying questions on behalf of the listener. Ensure the listener has enough context to follow the deep dive.

{if beat == 3:}
**DEEP DIVE + TENSION (90-120 seconds):** This is the meatiest part. The {expert_name} explains the core content in detail. The {skeptic_name} challenges claims and brings up risks/downsides/alternative viewpoints. Create genuine intellectual tension through respectful disagreement. This should NOT be a monologue—it's a dynamic conversation.

{if beat == 4:}
**ANALOGY / AHA MOMENT (30-60 seconds):** Someone (usually {expert_name}) offers an analogy or simplification that crystallizes the concept. The {host_name} should have a visible "aha" reaction. This is the memorable takeaway the listener can explain to someone else.

{if beat == 5:}
**WRAP + TRANSITION (15-30 seconds):** The {host_name} summarizes the key point of this chapter in 1-2 sentences. Then tease the next chapter with this hook: "{transition_hook}". Create forward momentum.

## Previous Beats (for context continuity)
{if beat > 1:}
{render previous beats' dialogue here so the LLM can reference it}

## Source Material
You MUST ground your dialogue in these source chunks. Every factual claim must trace back to these sources:
{for each source_chunk:}
**Chunk {chunk_id}** (Relevance: {relevance_score}):
{chunk_content}
Source: {source_url}

# OUTPUT FORMAT
Return a JSON array of utterances. Each utterance must follow this schema:

[
  {
    "speaker": "character name exactly as provided",
    "text": "the actual spoken dialogue",
    "intent": "question | answer | reaction | challenge | summary | transition",
    "emotion": "curious | excited | skeptical | thoughtful | amused | neutral",
    "grounding_chunk_ids": [list of chunk IDs this utterance draws from, empty if opinion/reaction]
  }
]

# CRITICAL RULES
1. **Grounding:** Every factual claim MUST reference source chunks via grounding_chunk_ids. Do NOT make up statistics, dates, or facts.
2. **Natural Flow:** This is a conversation, not an interview. Characters interrupt each other (politely), build on each other's points, and reference what was said earlier.
3. **Show, Don't Tell:** Instead of "{expert} explained the concept", write the actual explanation as dialogue.
4. **Length Target:** Aim for {target_word_count} words total for this beat (~{target_utterances} utterances).
5. **Character Consistency:** Each character must speak in their unique style. The {host_name} uses simple language and analogies. The {expert_name} uses technical terms but explains them. The {skeptic_name} asks probing questions.
6. **No Meta-Commentary:** Don't write "(laughs)" or "[pauses]". Just write the words they speak. Naturalness markers will be added later.
7. **Energy Level:** This chapter's energy is {energy_level}. Reflect this in the pacing and enthusiasm of the dialogue.

# EXAMPLE OUTPUT
[
  {
    "speaker": "Dr. Sarah Chen",
    "text": "Okay, here's something that blew my mind when I first learned about it. Ready?",
    "intent": "question",
    "emotion": "excited",
    "grounding_chunk_ids": []
  },
  {
    "speaker": "Jamie Rodriguez",
    "text": "Hit me with it.",
    "intent": "reaction",
    "emotion": "curious",
    "grounding_chunk_ids": []
  },
  {
    "speaker": "Prof. Marcus Thompson",
    "text": "In 2024, AI models surpassed human performance on 78% of benchmark tasks, up from just 12% in 2020. That's not incremental progress—that's exponential.",
    "intent": "answer",
    "emotion": "neutral",
    "grounding_chunk_ids": [12, 45]
  }
]

Now generate Beat {beat_number} dialogue.
```

### LLM Configuration

```python
{
  "model": "claude-sonnet-4-5",  # Or gpt-4-turbo
  "temperature": 0.8,  # Higher creativity for dialogue
  "max_tokens": 2000,  # ~400-500 words
  "response_format": {"type": "json_object"},
  "timeout": 30
}
```

### Node Implementation Logic

```
1. Load chapter input and character personas
2. Initialize beat counter (beat = 1)
3. FOR each beat from 1 to 5:
   a. Construct beat-specific system prompt
   b. Include previous beats as context (if beat > 1)
   c. Calculate target word count for this beat
   d. Call LLM with structured output
   e. Parse JSON response
   f. Validate utterance schema
   g. Add utterances to state.utterances list
   h. Store beat in state.beat_history
   i. Increment beat counter
4. Mark state.dialogue_complete = True
5. Return state
```

### Quality Checks (Within Node)

- **Schema Validation:** Every utterance has required fields
- **Speaker Validation:** Speaker names match provided characters exactly
- **Length Check:** Beat dialogue within 50-150% of target word count
- **Grounding Check:** At least 60% of utterances reference source chunks

### Failure Handling

If LLM returns invalid JSON or violates schema:
- Retry with explicit correction prompt (max 2 retries within node)
- If still fails, log error and use fallback: generic transition dialogue

---

## Node 2: Expert Content Expander

### Purpose
Take the expert character's utterances and expand them with more detailed explanations, examples, and technical depth while maintaining the conversational flow and topic coherence.

### Why This Node Exists

**Problem:** Initial dialogue generation often produces expert responses that are too brief:
- "AI uses neural networks to learn patterns." (10 words)

**Solution:** Expand expert content for depth and authority:
- "AI uses neural networks to learn patterns. Think of it like this: a neural network is organized in layers, kind of like how your brain processes information. The first layer might recognize basic shapes, the second layer combines those into patterns, and deeper layers understand complex concepts. The network learns by adjusting billions of tiny connections between these layers based on the training data it sees." (68 words)

### Expansion Strategy

1. **Selective Expansion:** Only expand utterances where:
   - speaker.role == "expert"
   - intent == "answer" or "challenge"
   - text length < 50 words
   - Beat 2, 3, or 4 (not Hook or Wrap)

2. **Maintain Flow:** Expansion must not break conversation coherence:
   - If next utterance is an interruption, keep expansion brief
   - If next utterance references "you just said", ensure expansion doesn't change the core point

### System Prompt Template

```markdown
# ROLE
You are an expert content expander specialized in educational podcast dialogue. Your job is to take brief expert explanations and expand them with rich detail, examples, and technical depth while keeping the conversational tone intact.

# TASK
Expand the following expert utterance to be more detailed and informative, targeting approximately {target_expansion_words} words (currently {current_words} words).

# CONTEXT
## Original Utterance
**Speaker:** {speaker_name} ({role})
**Current Text:** "{original_text}"
**Current Word Count:** {current_words}
**Intent:** {intent}
**Emotion:** {emotion}

## Character Profile
{speaker.speaking_style}
**Vocabulary Level:** {speaker.vocabulary_level}
**Expertise Area:** {speaker.expertise_area}

## Conversation Context
**What was said immediately before:**
{previous_utterance_text}

**What comes immediately after:**
{next_utterance_text}

**Chapter Key Points:**
{chapter.key_points}

## Source Material (for grounding)
{for each chunk in grounding_chunk_ids:}
**Chunk {chunk_id}:**
{chunk_content}

# EXPANSION GUIDELINES

## 1. Depth Enhancement Techniques
- **Add Examples:** Include concrete, real-world examples that illustrate the concept
- **Explain Technical Terms:** If using jargon, briefly explain it in accessible language
- **Provide Context:** Add historical context, comparisons, or relevant statistics
- **Include Nuance:** Mention exceptions, caveats, or alternative perspectives
- **Layer Information:** Start simple, then add complexity progressively

## 2. Conversational Tone Preservation
- Keep it natural and spoken, not academic
- Use rhetorical devices: "Think of it like this...", "Here's why that matters...", "The interesting part is..."
- Maintain the character's speaking style (technical but accessible)
- Use transitions: "And here's the key thing...", "But wait, there's more to it..."

## 3. Flow Integration
- The expansion must NOT contradict what comes before or after
- If the next speaker references this utterance, ensure core point remains clear
- Respect conversation dynamics—don't turn dialogue into monologue
- If energy_level is "high", keep sentences shorter and punchier
- If energy_level is "low", allow more contemplative, detailed explanations

## 4. Factual Grounding
- Every expanded claim must trace to provided source chunks
- Do NOT invent statistics, dates, or facts not in sources
- If adding examples, make them clearly hypothetical: "Imagine if...", "For instance, consider..."

## 5. Length Targets
- **Target Expansion:** {target_expansion_words} words
- **Minimum:** {min_expansion_words} words
- **Maximum:** {max_expansion_words} words (don't create monologues)

# WHAT NOT TO DO
❌ Don't break the fourth wall: No "(pauses)" or "[gestures]"
❌ Don't change the core message: Expand, don't replace
❌ Don't create disconnection: Next utterance must still make sense
❌ Don't add fabricated facts: Stay grounded in source material
❌ Don't lose the character's voice: Maintain speaking style

# OUTPUT FORMAT
Return ONLY the expanded text as a clean string. No JSON, no explanations.

# EXAMPLE

**Original (12 words):**
"Deep learning revolutionized AI by enabling computers to learn from raw data."

**Expanded (65 words):**
"Deep learning revolutionized AI by enabling computers to learn from raw data. Here's what makes it special: instead of humans manually programming every rule, deep learning models discover patterns on their own. You feed them millions of examples—say, images of cats—and they figure out what makes a cat a cat. The 'deep' part refers to the many layers of processing, where each layer learns increasingly abstract features. It's like how a child learns to recognize faces without anyone explaining facial geometry."

Now expand the original utterance.
```

### LLM Configuration

```python
{
  "model": "claude-sonnet-4-5",
  "temperature": 0.7,  # Balanced creativity
  "max_tokens": 800,  # Room for expansion
  "timeout": 20
}
```

### Expansion Logic

```
1. Load all utterances from state
2. Filter utterances for expansion candidates:
   - speaker.role == "expert"
   - beat in [2, 3, 4]
   - word_count < 50
   - intent in ["answer", "challenge"]
3. FOR each candidate utterance:
   a. Get previous and next utterance for context
   b. Calculate target expansion (current_words * 2.5, max 150 words)
   c. Retrieve source chunks from grounding_chunk_ids
   d. Construct expansion prompt
   e. Call LLM
   f. Validate expanded text:
      - Word count within range
      - No broken sentences
      - Character consistency check
   g. Replace utterance text with expanded version
   h. Update estimated_duration_seconds (word_count / 2.5)
4. Mark state.expansion_complete = True
5. Return state
```

### Quality Checks

- **Length Validation:** Expanded text between 1.5x-3x original length
- **Coherence Check:** Next utterance still makes sense given expansion
- **Source Grounding:** No new claims added without source references
- **Character Voice:** Expanded text matches character.speaking_style

### Fallback Strategy

If expansion fails or produces invalid output:
- Keep original text unchanged
- Log warning
- Continue to next node (don't block pipeline)

---

## Node 3: Naturalness Injector

### Purpose
Transform clean dialogue into human-like speech by injecting imperfections, speech patterns, and conversational markers that make it sound like real people talking, not AI reading scripts.

### Naturalness Markers Taxonomy

| Marker | Meaning | Frequency | Placement Rules |
|--------|---------|-----------|-----------------|
| `[FILLER:thinking]` | Thinking filler (um, so, well) | 15-20% of utterances | Before complex statements |
| `[FILLER:agreement]` | Agreement filler (yeah, right) | 10% of reactions | Start of response utterances |
| `[INTERRUPT:0.3s]` | Next speaker overlaps by 300ms | 10-15% of transitions | High-energy beats, skeptic role |
| `[BACKCHANNEL:host]` | Background "mm-hm" from host | Once per 50+ word expert monologue | Mid-sentence during explanations |
| `[FALSE_START]` | Speaker restarts sentence | Max 2 per chapter | Before nuanced/controversial points |
| `[LAUGH:light]` | Light chuckle | After analogies, self-deprecation | Never during serious content |
| `[LAUGH:medium]` | Genuine laugh | After funny moments, absurd facts | When multiple speakers amused |
| `[EMPHASIS:word]` | Stress specific word | 5-8 per chapter | Key revelations, statistics |
| `[PAUSE:short]` | Brief pause (400ms) | 20-30% of utterances | After rhetorical questions |
| `[PAUSE:long]` | Dramatic pause (800ms) | 3-5 per chapter | Before big reveals |
| `[PACE:fast]` | Speed up delivery | When excited or listing | High-energy beats, enthusiasm |
| `[PACE:slow]` | Slow down for emphasis | Key conclusions, gravitas | Important takeaways |

### Injection Strategy: Rule-Based + LLM Hybrid

**Approach:**
1. **Rule-Based (Deterministic):** Apply structural markers
   - Interrupts based on speaker role (skeptic interrupts more)
   - Backchannels during long monologues (>50 words)
   - Pauses after questions (every utterance ending with "?")

2. **LLM-Based (Contextual):** Apply nuanced markers
   - Fillers based on content complexity and character
   - Emphasis on contextually important words
   - Pacing changes based on emotional content
   - Laugh placement based on humor detection

### System Prompt Template (LLM Pass)

```markdown
# ROLE
You are a naturalness injection specialist for podcast dialogue. Your job is to make scripted dialogue sound like spontaneous human conversation by adding speech patterns, hesitations, and emotional cues.

# TASK
Analyze the following utterance and inject appropriate naturalness markers to make it sound like a real person speaking naturally.

# UTTERANCE TO ENHANCE
**Speaker:** {speaker_name} ({role})
**Text:** "{utterance_text}"
**Intent:** {intent}
**Emotion:** {emotion}
**Beat:** {beat}
**Word Count:** {word_count}

# CHARACTER PROFILE
**Speaking Style:** {character.speaking_style}
**Vocabulary Level:** {character.vocabulary_level}
**Filler Patterns:** {character.filler_patterns}
**Pace Preference:** {character.pace_preference}

# CONVERSATION CONTEXT
**Previous Utterance:** "{previous_text}"
**Next Utterance:** "{next_text}"
**Energy Level:** {chapter.energy_level}

# AVAILABLE NATURALNESS MARKERS

## Fillers
- `[FILLER:thinking]` - Insert before complex explanations → renders as "um," "so," "well,"
- `[FILLER:agreement]` - Insert at start of agreeing responses → renders as "yeah," "right,"

## Pauses
- `[PAUSE:short]` - 400ms pause → Use after rhetorical questions, before lists
- `[PAUSE:long]` - 800ms pause → Use before major revelations, dramatic moments

## Emphasis
- `[EMPHASIS:word]` - Stress a specific word → Use on statistics, key terms, surprising facts
  Example: "In [EMPHASIS:2024] alone" or "That's [EMPHASIS:exactly] right"

## Pacing
- `[PACE:fast]` - Speed up delivery → Use when excited, listing things, high energy
- `[PACE:slow]` - Slow down → Use for gravitas, important conclusions

## Emotional
- `[LAUGH:light]` - Soft chuckle → After self-deprecation, gentle humor
- `[LAUGH:medium]` - Genuine laugh → After funny moments, absurd juxtapositions

## Restarts
- `[FALSE_START]` - Speaker restarts sentence → Use when making nuanced/controversial point
  Example: "The thing is— well, let me put it differently"

# INJECTION RULES

## 1. Frequency Limits (Per Utterance)
- Maximum 1-2 markers for utterances < 20 words
- Maximum 2-4 markers for utterances 20-50 words
- Maximum 4-6 markers for utterances > 50 words

## 2. Character Consistency
- **Casual vocabulary characters:** More fillers, more informal markers
- **Technical vocabulary characters:** Fewer fillers, more pauses for precision
- **Host role:** More emphatic markers, more pacing changes
- **Expert role:** More pauses, strategic emphasis on technical terms
- **Skeptic role:** More questioning pauses, emphasis on challenges

## 3. Context-Driven Placement
- **If previous speaker was interrupted:** Don't start with filler (jump right in)
- **If complex technical content:** Add thinking filler at start
- **If rhetorical question:** Add short pause after
- **If surprising statistic:** Add emphasis marker on the number
- **If energy_level is "high":** Favor [PACE:fast], fewer long pauses
- **If emotion is "excited":** More pacing changes, potential light laugh
- **If intent is "challenge":** Emphasis on key contrasting words

## 4. Natural Flow
- Don't cluster markers: Space them out across the utterance
- Don't over-inject: The dialogue should still feel natural, not gimmicky
- Maintain readability: Markers should enhance, not obscure the content

## 5. Forbidden Patterns
- ❌ Never add laugh during serious/sensitive content
- ❌ Never add filler immediately after another filler
- ❌ Never add false start more than once per utterance
- ❌ Never add emphasis to every word (defeats the purpose)

# OUTPUT FORMAT
Return the utterance text with markers inserted. Return ONLY the enhanced text, no explanations.

# EXAMPLES

**Original:**
"That's a really important point. In 2024, AI systems processed more data than the entire internet contained in 2000."

**Enhanced:**
"[PAUSE:short] That's a [EMPHASIS:really] important point. [FILLER:thinking] In [EMPHASIS:2024], AI systems processed more data than the entire internet contained in 2000."

**Original:**
"Wait, are you saying it learned that on its own? No human programmed those rules?"

**Enhanced:**
"Wait, [PAUSE:short] are you saying it learned that [EMPHASIS:on its own]? No human programmed those rules?"

**Original:**
"Exactly! And here's why that matters. When the model can self-correct, it becomes exponentially more powerful. We're not talking about linear improvement here."

**Enhanced:**
"[PACE:fast] Exactly! [PAUSE:short] And here's why that matters. [PACE:slow] When the model can [EMPHASIS:self-correct], it becomes exponentially more powerful. We're not talking about linear improvement here."

Now enhance the utterance.
```

### LLM Configuration

```python
{
  "model": "claude-sonnet-4-5",  # Or gpt-4-turbo
  "temperature": 0.6,  # Moderate creativity
  "max_tokens": 500,
  "timeout": 15
}
```

### Rule-Based Injection Logic (Applied First)

```
1. FOR each utterance in state.utterances:

   # Structural Rules
   a. IF utterance ends with "?" AND intent == "question":
      - Add [PAUSE:short] after the question

   b. IF speaker.role == "skeptic" AND next_speaker != current_speaker:
      - 30% chance: Add [INTERRUPT:0.3s] marker to NEXT utterance

   c. IF word_count > 50 AND speaker.role == "expert":
      - Find midpoint of utterance
      - Insert [BACKCHANNEL:host] marker
      - Tag this as metadata (not in text, for audio mixer)

   d. IF previous_utterance.speaker == current_utterance.speaker:
      - 40% chance: Add [FILLER:thinking] at start (self-correction pattern)

   e. IF energy_level == "high" AND word_count > 30:
      - Wrap first 1/3 of utterance in [PACE:fast]

   f. IF beat == 5 AND intent == "summary":
      - Wrap key sentence in [PACE:slow]

2. Store rule-based enhanced text in temporary field

3. Pass to LLM for contextual enhancement

4. Merge rule-based + LLM markers (deduplication)

5. Update utterance.text_with_naturalness
```

### Node Implementation

```
1. Load all utterances from state
2. Apply rule-based markers (structural)
3. FOR each utterance:
   a. Construct naturalness injection prompt
   b. Include previous/next utterance context
   c. Call LLM
   d. Parse enhanced text
   e. Validate marker syntax
   f. Merge with rule-based markers
   g. Store in utterance.text_with_naturalness
4. Validate overall marker frequency (not too many)
5. Mark state.naturalness_complete = True
6. Return state
```

### Quality Checks

- **Marker Syntax:** All markers match regex: `\[([A-Z_]+)(?::([a-z0-9.]+))?\]`
- **Frequency Limits:** No utterance has >6 markers
- **Character Consistency:** Casual characters have more fillers than technical ones
- **Laugh Placement:** No laughs during serious content (check emotion field)

---

## Node 4: Fact-Checker Agent

### Purpose
Verify that every factual claim in the dialogue can be traced back to original source chunks from Phase 1, preventing LLM hallucinations and ensuring research-grounded content.

### Verification Strategy

**Three-Step Process:**
1. **Claim Extraction:** Identify factual claims (numbers, dates, entities, assertions)
2. **Source Matching:** For each claim, retrieve grounding source chunks
3. **Verification:** LLM determines if source supports the claim

### Claim Types to Check

| Claim Type | Example | Verification Method |
|------------|---------|---------------------|
| **Quantitative** | "78% of tasks" | Exact match in source |
| **Temporal** | "In 2024" | Date range match in source |
| **Named Entity** | "GPT-4 achieved" | Entity mentioned in source |
| **Causal** | "X caused Y" | Source explicitly states relationship |
| **Comparative** | "faster than" | Source provides comparison data |

### System Prompt Template

```markdown
# ROLE
You are a rigorous fact-checker for podcast content. Your job is to verify that factual claims are supported by source material and flag unsupported assertions.

# TASK
Verify if the following claim from the podcast dialogue is supported by the provided source chunks.

# CLAIM TO VERIFY
**Utterance ID:** {utterance_id}
**Speaker:** {speaker_name}
**Claim:** "{extracted_claim}"
**Claim Type:** {claim_type} (quantitative | temporal | entity | causal | comparative)

# SOURCE CHUNKS
The speaker referenced these source chunks (via grounding_chunk_ids):

{for each chunk_id in utterance.grounding_chunk_ids:}
**Chunk {chunk_id}:**
{chunk.content}
**Source URL:** {chunk.source_url}
---

# VERIFICATION CRITERIA

## SUPPORTED
A claim is SUPPORTED if:
- The source explicitly states the claim OR
- The claim is a reasonable inference from the source (within one logical step) OR
- The numbers/dates match exactly OR
- The entity/event is clearly mentioned in context

## PARTIALLY SUPPORTED
A claim is PARTIALLY SUPPORTED if:
- The source discusses the general topic but lacks specific details OR
- The numbers are close but not exact (e.g., source says "around 75%", claim says "78%") OR
- The claim is technically true but missing important nuance from the source

## UNSUPPORTED
A claim is UNSUPPORTED if:
- The source does not mention the claim at all OR
- The numbers/dates contradict the source OR
- The claim extrapolates beyond what the source states OR
- The claim is plausible but not grounded in provided sources (likely hallucination)

# OUTPUT FORMAT
Return a JSON object:

{
  "verdict": "supported | partially_supported | unsupported",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this verdict was reached",
  "source_quote": "Exact quote from source that supports (or contradicts) the claim",
  "correction": "If unsupported/partial, suggest a corrected claim that IS supported, or null"
}

# EXAMPLES

## Example 1: SUPPORTED
**Claim:** "In 2024, AI models surpassed human performance on 78% of benchmark tasks."
**Source:** "According to the 2024 AI Index Report, AI systems now exceed human-level performance on 78 percent of standard benchmark evaluations."
**Output:**
{
  "verdict": "supported",
  "confidence": 1.0,
  "reasoning": "The source explicitly states the exact statistic with matching numbers and timeframe.",
  "source_quote": "AI systems now exceed human-level performance on 78 percent of standard benchmark evaluations",
  "correction": null
}

## Example 2: UNSUPPORTED
**Claim:** "By 2030, AI will replace 40% of all jobs."
**Source:** "Some economists predict significant labor market disruption from AI, though exact figures vary widely from 10% to 50% depending on the study."
**Output:**
{
  "verdict": "unsupported",
  "confidence": 0.9,
  "reasoning": "The source mentions a range (10-50%) but does not specifically state 40%. This appears to be a specific number not grounded in the source.",
  "source_quote": "exact figures vary widely from 10% to 50%",
  "correction": "Economists' predictions for AI's labor market impact by 2030 vary widely, ranging from 10% to 50% of jobs affected."
}

## Example 3: PARTIALLY SUPPORTED
**Claim:** "Deep learning revolutionized computer vision in the early 2010s."
**Source:** "The breakthrough moment came in 2012 when AlexNet won the ImageNet competition, demonstrating deep learning's potential for image recognition tasks."
**Output:**
{
  "verdict": "partially_supported",
  "confidence": 0.8,
  "reasoning": "The source confirms a major breakthrough in 2012 (early 2010s) in computer vision using deep learning. However, 'revolutionized' is a strong characterization—the source says 'breakthrough' which is supportive but slightly less dramatic.",
  "source_quote": "The breakthrough moment came in 2012 when AlexNet won the ImageNet competition",
  "correction": null
}

Now verify the claim.
```

### LLM Configuration

```python
{
  "model": "claude-haiku",  # Fast, cheap model for verification
  "temperature": 0.1,  # Low creativity, high precision
  "max_tokens": 400,
  "response_format": {"type": "json_object"},
  "timeout": 45
}
```

### Claim Extraction Logic (Pre-Processing)

```
1. FOR each utterance in state.utterances:
   a. IF utterance.grounding_chunk_ids is empty:
      - Skip (opinion/reaction, not factual)

   b. Use regex + NLP to extract potential claims:
      - Numbers + units: r'\d+\.?\d*\s*%|million|billion|dollars'
      - Dates: r'\b(19|20)\d{2}\b|January|February|...'
      - Named entities: NER extraction (GPT-4, Claude, etc.)
      - Causal words: "caused", "led to", "resulted in"
      - Comparative: "more than", "faster than", "better than"

   c. Store extracted claims in list:
      claims_to_verify.append({
        "utterance_id": utterance.utterance_id,
        "claim": extracted_text,
        "claim_type": type,
        "grounding_chunk_ids": utterance.grounding_chunk_ids
      })

2. Batch process claims (5-10 at a time for efficiency)
```

### Node Implementation

```
1. Extract claims from all utterances
2. IF no claims found:
   - Log warning ("No factual claims detected")
   - Mark state.fact_check_complete = True
   - Return state (proceed to next node)

3. FOR each claim:
   a. Retrieve source chunks from chunk_ids
   b. Construct fact-check prompt
   c. Call LLM
   d. Parse verification result
   e. IF verdict == "unsupported":
      - Add to state.fact_check_issues
      - Store correction suggestion
   f. IF verdict == "partially_supported" AND confidence < 0.7:
      - Add to state.fact_check_issues (warning level)

4. IF state.fact_check_issues is not empty:
   a. Check retry count
   b. IF retry_count < max_retries:
      - Prepare correction instructions for Dialogue Engine
      - Graph will route back to Node 1
   c. ELSE:
      - Log issues as warnings
      - Mark state.fact_check_complete = True
      - Proceed to QA Reviewer (give up on retries)

5. ELSE:
   - Mark state.fact_check_complete = True
   - Return state
```

### Correction Instruction Format

When routing back to Dialogue Engine after failed fact-check:

```json
{
  "retry_type": "fact_correction",
  "issues": [
    {
      "utterance_id": "ch1_u015",
      "speaker": "Prof. Marcus Thompson",
      "problematic_claim": "By 2030, AI will replace 40% of all jobs",
      "verdict": "unsupported",
      "suggested_correction": "Economists' predictions for AI's labor market impact by 2030 vary widely, ranging from 10% to 50% of jobs affected.",
      "source_reference": "Chunk 45"
    }
  ],
  "instruction": "Revise the specified utterances to replace unsupported claims with the suggested corrections. Maintain the conversational flow and character voice."
}
```

### Quality Gates

- **Claim Coverage:** At least 60% of utterances with grounding_chunk_ids checked
- **Verification Confidence:** Average confidence score > 0.75
- **Critical Claims:** Any quantitative claim must be "supported" (not partial)

---

## Node 5: QA Reviewer Agent

### Purpose
Evaluate the complete chapter script from a first-time listener's perspective, catching issues that would cause listener drop-off: repetition, monologue traps, unclear explanations, broken transitions, and energy flow problems.

### QA Dimensions

| Dimension | What It Checks | Pass Criteria | Fix Strategy |
|-----------|----------------|---------------|--------------|
| **Repetition Detection** | Same concept explained twice | No significant overlap between chapters | Remove redundant chapter or merge |
| **Engagement Assessment** | Monologue trap detection | No stretch >90s with single speaker | Break monologue with interruption/question |
| **Clarity Check** | Terms used before explanation | All jargon introduced before use | Reorder utterances or add definition |
| **Transition Quality** | Chapter connections | Each chapter end flows to next | Rewrite transition_hook |
| **Energy Arc Validation** | Pacing rhythm | Matches intended energy pattern | Flag for re-planning (rare) |

### System Prompt Template

```markdown
# ROLE
You are a podcast quality assurance expert who evaluates scripts from the listener's perspective. You catch issues that cause listener drop-off and ensure the podcast maintains engagement from start to finish.

# TASK
Review the complete script for Chapter {chapter_number} and assess it across five quality dimensions.

# COMPLETE CHAPTER SCRIPT

{render all utterances in order, formatted as:}
**[Beat {beat}] {Speaker}:** {text_with_naturalness}

---
(repeat for all utterances)
---

# CHAPTER CONTEXT
- **Title:** {chapter_title}
- **Act:** {act}
- **Energy Level:** {energy_level}
- **Key Points:** {key_points}
- **Target Duration:** {estimated_duration_minutes} minutes
- **Previous Chapter:** {previous_chapter_title} (if applicable)
- **Next Chapter:** {next_chapter_title} (if applicable)

# EVALUATION DIMENSIONS

## 1. REPETITION DETECTION
**Question:** Does this chapter repeat content from previous chapters?

**Check for:**
- Same examples used in multiple chapters
- Same statistic mentioned twice
- Same concept explained in different words
- Overlapping key points

**If found:** Note the specific utterances and which previous chapter it duplicates.

## 2. ENGAGEMENT ASSESSMENT
**Question:** Are there monologue traps where one speaker dominates too long?

**Check for:**
- Any single speaker talking for >90 seconds continuously
- More than 3 consecutive utterances from the same speaker without interruption
- Lack of back-and-forth dynamic (especially in Beat 3)

**If found:** Identify the monotonous section and suggest where to add interruption/question.

## 3. CLARITY CHECK
**Question:** Will a first-time listener understand everything in sequence?

**Check for:**
- Technical terms used before they're defined
- Acronyms not spelled out on first use
- Concepts referenced before they're introduced
- Assumed knowledge that wasn't established

**If found:** Note the confusing term and where it should be explained.

## 4. TRANSITION QUALITY
**Question:** Does the chapter flow naturally from the previous and to the next?

**Check for:**
- Beat 1 (Hook): Does it connect to the previous chapter's ending?
- Beat 5 (Wrap): Does it set up the next chapter effectively?
- Abrupt topic switches within the chapter

**If found:** Suggest how to smooth the transition.

## 5. ENERGY ARC VALIDATION
**Question:** Does the energy flow match the intended level and create rhythm?

**Check for:**
- High-energy chapters should have: short utterances, excitement, fast pacing markers
- Low-energy chapters should have: longer explanations, pauses, contemplative tone
- The actual energy matches the specified {energy_level}

**If found:** Note if the chapter feels mismatched to its energy label.

# OUTPUT FORMAT
Return a JSON object:

{
  "overall_pass": true/false,
  "issues_found": [
    {
      "dimension": "repetition | engagement | clarity | transition | energy",
      "severity": "critical | warning | minor",
      "description": "Clear description of the issue",
      "location": "Specific utterance ID(s) or beat number",
      "suggested_fix": "Actionable recommendation"
    }
  ],
  "strengths": ["List 2-3 things the chapter does well"],
  "listener_experience_score": 1-10,
  "reasoning": "Brief overall assessment"
}

# SEVERITY DEFINITIONS
- **Critical:** Will cause listener confusion or drop-off. Must fix before TTS.
- **Warning:** Degrades quality but not fatal. Fix if possible.
- **Minor:** Stylistic improvement. Optional fix.

# EXAMPLE OUTPUT

{
  "overall_pass": false,
  "issues_found": [
    {
      "dimension": "repetition",
      "severity": "critical",
      "description": "Utterance ch3_u012 explains neural network layers using the same 'brain layers' analogy already used in Chapter 1 (ch1_u008).",
      "location": "ch3_u012",
      "suggested_fix": "Use a different analogy here (e.g., assembly line, orchestra) or reference the earlier explanation: 'Remember the brain layers analogy from earlier? Here's why that matters...'"
    },
    {
      "dimension": "engagement",
      "severity": "warning",
      "description": "Prof. Marcus Thompson speaks continuously for 95 seconds (utterances ch3_u018 through ch3_u022) without interruption.",
      "location": "Beat 3, utterances ch3_u018-u022",
      "suggested_fix": "Add a clarifying question from host or skeptic after ch3_u020 to break up the monologue."
    }
  ],
  "strengths": [
    "Excellent transition from Chapter 2—the hook directly addresses the question left hanging",
    "Beat 4 analogy (comparing AI training to teaching a child) is memorable and effective",
    "Energy level is appropriately high with good pacing markers"
  ],
  "listener_experience_score": 7,
  "reasoning": "Solid chapter with good flow and energy, but the repetition issue is critical and the monologue in Beat 3 risks losing engagement."
}

Now evaluate the chapter.
```

### LLM Configuration

```python
{
  "model": "claude-sonnet-4-5",  # Needs strong reasoning
  "temperature": 0.3,  # Analytical, not creative
  "max_tokens": 1500,
  "response_format": {"type": "json_object"},
  "timeout": 30
}
```

### Node Implementation

```
1. Load complete chapter dialogue from state
2. Retrieve previous chapter context (if not chapter 1)
3. Construct QA review prompt with full script
4. Call LLM
5. Parse QA results
6. Categorize issues by severity:
   - critical_issues = [issues where severity == "critical"]
   - warning_issues = [issues where severity == "warning"]
   - minor_issues = [issues where severity == "minor"]

7. IF critical_issues:
   a. Add to state.qa_review_issues
   b. IF retry_count < max_retries:
      - Prepare targeted rewrite instructions
      - Graph will route back to Node 1 (Dialogue Engine)
   c. ELSE:
      - Log issues as errors
      - Proceed to SSML Annotator (give up on retries)

8. ELSE IF warning_issues AND listener_experience_score < 7:
   a. Add to state.qa_review_issues (lower priority)
   b. IF retry_count < max_retries:
      - Prepare targeted rewrite instructions
   c. ELSE:
      - Proceed to SSML Annotator

9. ELSE:
   - Mark state.qa_review_complete = True
   - Proceed to SSML Annotator

10. Return state
```

### Targeted Rewrite Instructions

When routing back to Dialogue Engine after QA issues:

```json
{
  "retry_type": "qa_improvement",
  "focus_areas": ["engagement", "repetition"],
  "specific_fixes": [
    {
      "utterance_ids": ["ch3_u018", "ch3_u019", "ch3_u020"],
      "issue": "Monologue trap—95 seconds continuous",
      "fix_instruction": "Break this section by having Jamie Rodriguez interrupt after ch3_u020 with a clarifying question about the training process."
    },
    {
      "utterance_id": "ch3_u012",
      "issue": "Repetition—brain layers analogy already used in Chapter 1",
      "fix_instruction": "Replace this analogy with a different one (e.g., assembly line, orchestra) or reference the earlier explanation explicitly."
    }
  ],
  "instruction": "Apply these targeted fixes without regenerating the entire chapter. Maintain the flow and only modify the specified sections."
}
```

### Quality Gates

- **Pass Threshold:** overall_pass == true OR listener_experience_score >= 7
- **Critical Blocker:** Any "critical" severity issue blocks pipeline unless max retries exceeded
- **Warning Threshold:** More than 3 "warning" issues triggers retry

---

## Node 6: SSML Annotator (Google TTS)

### Purpose
Convert naturalness markers into Google Cloud TTS-compatible SSML tags and configure per-speaker voice characteristics for optimal prosody control.

### Google TTS SSML Support

Google Cloud TTS (WaveNet/Neural2) supports these SSML elements:

| SSML Tag | Purpose | Google TTS Support |
|----------|---------|-------------------|
| `<break time="Xms"/>` | Insert pause | ✅ Full support (10ms-10000ms) |
| `<emphasis level="X">` | Stress word/phrase | ✅ Supports: strong, moderate, reduced |
| `<prosody rate="X">` | Change speaking speed | ✅ Supports: x-slow, slow, medium, fast, x-fast, or % |
| `<prosody pitch="X">` | Change pitch | ✅ Supports: x-low, low, medium, high, x-high, or semitones |
| `<prosody volume="X">` | Change volume | ✅ Supports: silent, x-soft, soft, medium, loud, x-loud, or dB |
| `<say-as interpret-as="X">` | Number/date formatting | ✅ Supports: cardinal, ordinal, date, time, etc. |
| `<sub alias="X">` | Pronunciation override | ✅ Supported |

**Note:** Google TTS does NOT support:
- `<voice>` tag (voice selected via API parameter, not SSML)
- `<audio>` tag (can't embed audio clips)
- `<phoneme>` (limited IPA support)

### Marker-to-SSML Conversion Table

| Naturalness Marker | Google TTS SSML Output | Notes |
|-------------------|------------------------|-------|
| `[FILLER:thinking]` | `<break time="300ms"/>um,` | Render as literal "um," with pause before |
| `[FILLER:agreement]` | `yeah,` | Just text, natural intonation |
| `[PAUSE:short]` | `<break time="400ms"/>` | Standard short pause |
| `[PAUSE:long]` | `<break time="800ms"/>` | Dramatic pause |
| `[EMPHASIS:word]` | `<emphasis level="strong">word</emphasis>` | Strong emphasis |
| `[PACE:fast]` | `<prosody rate="fast">...text...</prosody>` | Speed up (or rate="120%") |
| `[PACE:slow]` | `<prosody rate="slow">...text...</prosody>` | Slow down (or rate="80%") |
| `[LAUGH:light]` | `<break time="200ms"/>` + "heh" text | Simulate chuckle |
| `[LAUGH:medium]` | `<break time="300ms"/>` | Just pause (TTS can't laugh) |
| `[FALSE_START]` | `The thing is—<break time="200ms"/> well,` | Em dash + pause |
| `[INTERRUPT:0.3s]` | **Stripped** (metadata only) | Handled by Phase 5 audio mixer |
| `[BACKCHANNEL:host]` | **Stripped** (metadata only) | Handled by Phase 5 audio mixer |

### SSML Annotation Logic

```
1. FOR each utterance in state.utterances:

   a. Start with utterance.text_with_naturalness

   b. Extract and store audio mixer metadata:
      - IF "[INTERRUPT:" in text:
         * Extract timing value (e.g., "0.3s")
         * Store in utterance.audio_metadata.interrupt_duration
         * Remove marker from text
      - IF "[BACKCHANNEL:" in text:
         * Extract speaker (e.g., "host")
         * Store in utterance.audio_metadata.backchannel_speaker
         * Remove marker from text

   c. Convert remaining markers to SSML:

      # Pauses
      text = text.replace("[PAUSE:short]", "<break time=\"400ms\"/>")
      text = text.replace("[PAUSE:long]", "<break time=\"800ms\"/>")

      # Fillers (with pauses)
      text = text.replace("[FILLER:thinking]", "<break time=\"300ms\"/>um,")
      text = text.replace("[FILLER:agreement]", "yeah,")

      # Emphasis (word-level)
      text = re.sub(
        r'\[EMPHASIS:(\w+)\]',
        r'<emphasis level="strong">\1</emphasis>',
        text
      )

      # Pacing (wrap sections)
      text = re.sub(
        r'\[PACE:fast\](.*?)\[/PACE\]',
        r'<prosody rate="fast">\1</prosody>',
        text
      )
      text = re.sub(
        r'\[PACE:slow\](.*?)\[/PACE\]',
        r'<prosody rate="slow">\1</prosody>',
        text
      )

      # Laughs (convert to pauses + text)
      text = text.replace("[LAUGH:light]", "<break time=\"200ms\"/>heh")
      text = text.replace("[LAUGH:medium]", "<break time=\"300ms\"/>")

      # False starts (em dash + pause)
      text = re.sub(
        r'\[FALSE_START\](.*?)—',
        r'\1—<break time=\"200ms\"/>',
        text
      )

   d. Wrap in root SSML structure:
      ssml_text = f'<speak>{text}</speak>'

   e. Add voice-specific prosody wrapper:
      base_rate = character.default_speaking_rate  # e.g., "medium", "fast"
      base_pitch = character.default_pitch  # e.g., "+2st", "-1st"

      IF base_rate != "medium" OR base_pitch != "0st":
        ssml_text = f'<speak><prosody rate="{base_rate}" pitch="{base_pitch}">{text}</prosody></speak>'

   f. Store in utterance.text_ssml

   g. Store TTS voice configuration:
      utterance.tts_config = {
        "voice_id": character.tts_voice_id,  # e.g., "en-US-Neural2-F"
        "language_code": "en-US",
        "ssml_gender": character.ssml_gender,  # "FEMALE", "MALE", "NEUTRAL"
        "speaking_rate": 1.0,  # Default, SSML overrides
        "pitch": 0.0  # Default, SSML overrides
      }

2. Validate SSML syntax:
   - Parse with XML parser to ensure well-formed
   - Check for unclosed tags
   - Verify break times are valid (10ms-10000ms)

3. Mark state.ssml_complete = True
4. Compile final output
5. Return state
```

### Per-Character Voice Configuration

**Host (Dr. Sarah Chen):**
```json
{
  "tts_voice_id": "en-US-Neural2-F",
  "ssml_gender": "FEMALE",
  "default_speaking_rate": "medium",
  "default_pitch": "+1st",
  "base_prosody": "<prosody rate=\"medium\" pitch=\"+1st\">"
}
```

**Expert (Prof. Marcus Thompson):**
```json
{
  "tts_voice_id": "en-US-Neural2-D",
  "ssml_gender": "MALE",
  "default_speaking_rate": "slow",
  "default_pitch": "-2st",
  "base_prosody": "<prosody rate=\"slow\" pitch=\"-2st\">"
}
```

**Skeptic (Jamie Rodriguez):**
```json
{
  "tts_voice_id": "en-US-Neural2-C",
  "ssml_gender": "NEUTRAL",
  "default_speaking_rate": "fast",
  "default_pitch": "0st",
  "base_prosody": "<prosody rate=\"fast\">"
}
```

### SSML Output Example

**Input (with naturalness markers):**
```
"[PACE:fast] Welcome back! [PAUSE:short] Today we're diving into something that's reshaping our world faster than most people realize."
```

**Output (Google TTS SSML):**
```xml
<speak>
  <prosody rate="medium" pitch="+1st">
    <prosody rate="fast">Welcome back!</prosody>
    <break time="400ms"/>
    Today we're diving into something that's reshaping our world faster than most people realize.
  </prosody>
</speak>
```

### Node Implementation

```
1. Load all utterances from state
2. Load character voice configurations
3. FOR each utterance:
   a. Extract audio mixer metadata (interrupts, backchannels)
   b. Convert naturalness markers to SSML
   c. Wrap in character-specific prosody
   d. Validate SSML syntax
   e. Store in utterance.text_ssml
   f. Store TTS config parameters
4. Compile final chapter output
5. Calculate total estimated duration (sum of utterance durations)
6. Mark state.ssml_complete = True
7. Return final output
```

### Quality Checks

- **SSML Validation:** All utterances parse as valid XML
- **Tag Nesting:** Prosody tags properly nested, no overlapping
- **Break Times:** All break values between 10ms-10000ms
- **Voice Assignment:** Every utterance has valid tts_voice_id

---

## Data Schemas

### ChapterInput (Phase 2 → Phase 3)

```python
from typing import List, Dict, Literal
from pydantic import BaseModel, Field

class SourceChunk(BaseModel):
    chunk_id: int
    content: str
    source_url: str
    relevance_score: float

class CharacterPersona(BaseModel):
    name: str
    role: Literal["host", "expert", "skeptic"]
    expertise_area: str
    speaking_style: str
    vocabulary_level: Literal["casual", "moderate", "technical"]
    filler_patterns: List[str] = ["um", "so", "well"]
    reaction_patterns: List[str] = ["interesting", "wait, really?"]
    disagreement_style: str
    laugh_frequency: Literal["rare", "moderate", "frequent"]
    catchphrases: List[str] = []
    emotional_range: str
    tts_voice_id: str  # Google TTS voice ID
    ssml_gender: Literal["FEMALE", "MALE", "NEUTRAL"]
    default_speaking_rate: Literal["x-slow", "slow", "medium", "fast", "x-fast"] = "medium"
    default_pitch: str = "0st"  # Semitones, e.g., "+2st", "-1st"

class ChapterInput(BaseModel):
    chapter_number: int
    title: str
    act: Literal["setup", "explore", "resolve"]
    energy_level: Literal["high", "medium", "low"]
    key_points: List[str]
    source_chunk_ids: List[int]
    transition_hook: str
    estimated_duration_minutes: float
    characters: Dict[str, CharacterPersona]
    source_chunks: List[SourceChunk]
```

### Utterance (Working Schema)

```python
class AudioMetadata(BaseModel):
    interrupt_duration: Optional[str] = None  # e.g., "0.3s"
    backchannel_speaker: Optional[str] = None  # e.g., "host"

class TTSConfig(BaseModel):
    voice_id: str  # e.g., "en-US-Neural2-F"
    language_code: str = "en-US"
    ssml_gender: Literal["FEMALE", "MALE", "NEUTRAL"]
    speaking_rate: float = 1.0
    pitch: float = 0.0

class Utterance(BaseModel):
    utterance_id: str  # e.g., "ch1_u001"
    speaker: str  # Character name
    beat: int  # 1-5
    text_clean: str  # Original dialogue without markers
    text_with_naturalness: str  # With [PAUSE], [EMPHASIS], etc.
    text_ssml: str  # Final SSML for Google TTS
    intent: Literal["question", "answer", "reaction", "challenge", "summary", "transition"]
    emotion: Literal["curious", "excited", "skeptical", "thoughtful", "amused", "neutral"]
    grounding_chunk_ids: List[int]
    estimated_duration_seconds: float
    audio_metadata: AudioMetadata = AudioMetadata()
    tts_config: TTSConfig
```

### ChapterDialogueOutput (Phase 3 Output)

```python
class ValidationMetadata(BaseModel):
    fact_check_claims_verified: int
    fact_check_unsupported: int
    qa_review_iterations: int
    naturalness_markers_added: int
    listener_experience_score: float

class ChapterDialogueOutput(BaseModel):
    chapter_number: int
    utterances: List[Utterance]
    total_utterances: int
    estimated_chapter_duration: float  # minutes
    quality_checks_passed: bool
    validation_metadata: ValidationMetadata
```

---

## Graph Routing Logic

### Conditional Edges

```python
from langgraph.graph import StateGraph, END

# Define the graph
graph = StateGraph(Phase3GraphState)

# Add nodes
graph.add_node("dialogue_engine", dialogue_engine_node)
graph.add_node("expert_expander", expert_expander_node)
graph.add_node("naturalness_injector", naturalness_injector_node)
graph.add_node("fact_checker", fact_checker_node)
graph.add_node("qa_reviewer", qa_reviewer_node)
graph.add_node("ssml_annotator", ssml_annotator_node)

# Set entry point
graph.set_entry_point("dialogue_engine")

# Conditional routing
def route_after_dialogue(state: Phase3GraphState) -> str:
    """Route after Dialogue Engine completes a beat"""
    if state.current_beat < 5:
        # Continue generating remaining beats
        state.current_beat += 1
        return "dialogue_engine"
    else:
        # All beats complete, move to expansion
        return "expert_expander"

def route_after_fact_checker(state: Phase3GraphState) -> str:
    """Route after Fact-Checker validation"""
    if state.fact_check_issues:
        if state.retry_count < state.max_retries:
            state.retry_count += 1
            state.current_beat = 1  # Restart from beat 1 with corrections
            return "dialogue_engine"
        else:
            # Max retries exceeded, log warnings and proceed
            logger.warning(f"Fact-check issues remain after {state.max_retries} retries")
            return "qa_reviewer"
    else:
        return "qa_reviewer"

def route_after_qa_reviewer(state: Phase3GraphState) -> str:
    """Route after QA Review"""
    critical_issues = [i for i in state.qa_review_issues if i["severity"] == "critical"]

    if critical_issues and state.retry_count < state.max_retries:
        state.retry_count += 1
        state.current_beat = 1  # Targeted rewrite
        return "dialogue_engine"
    else:
        return "ssml_annotator"

# Add conditional edges
graph.add_conditional_edges(
    "dialogue_engine",
    route_after_dialogue,
    {
        "dialogue_engine": "dialogue_engine",  # Loop for more beats
        "expert_expander": "expert_expander"   # Move to next phase
    }
)

graph.add_edge("expert_expander", "naturalness_injector")
graph.add_edge("naturalness_injector", "fact_checker")

graph.add_conditional_edges(
    "fact_checker",
    route_after_fact_checker,
    {
        "dialogue_engine": "dialogue_engine",  # Retry with corrections
        "qa_reviewer": "qa_reviewer"          # Proceed
    }
)

graph.add_conditional_edges(
    "qa_reviewer",
    route_after_qa_reviewer,
    {
        "dialogue_engine": "dialogue_engine",  # Retry with QA fixes
        "ssml_annotator": "ssml_annotator"    # Proceed to final step
    }
)

graph.add_edge("ssml_annotator", END)

# Compile
phase3_graph = graph.compile()
```

---

## Retry and Fallback Strategies

### Retry Logic Summary

| Node | Retry Trigger | Max Retries | Retry Strategy | Fallback |
|------|--------------|-------------|----------------|----------|
| **Dialogue Engine** | Invalid JSON, schema violation | 2 (per beat) | Re-prompt with error message | Generic transition dialogue |
| **Expert Expander** | Expansion too verbose | 1 | Reduce target word count | Keep original text |
| **Naturalness Injector** | Marker syntax error | 1 | Rule-based only (skip LLM) | Use rule-based markers |
| **Fact-Checker** | Unsupported claims detected | 2 (graph-level) | Route back to Dialogue Engine | Proceed with warnings |
| **QA Reviewer** | Critical issues found | 2 (graph-level) | Route back to Dialogue Engine | Proceed with warnings |
| **SSML Annotator** | Invalid SSML syntax | 1 | Strip problematic tags | Plain text output |

### Graph-Level Retry Management

```python
class Phase3GraphState(TypedDict):
    retry_count: int  # Global retry counter
    max_retries: int  # Default: 2
    retry_history: List[Dict]  # Track what was retried

def increment_retry(state: Phase3GraphState, reason: str):
    """Increment retry counter with logging"""
    state.retry_count += 1
    state.retry_history.append({
        "retry_number": state.retry_count,
        "reason": reason,
        "timestamp": datetime.now().isoformat()
    })

    if state.retry_count >= state.max_retries:
        logger.warning(f"Max retries ({state.max_retries}) reached for: {reason}")
```

### Fact-Check Retry Example

```python
def fact_checker_node(state: Phase3GraphState) -> Phase3GraphState:
    # ... verification logic ...

    if unsupported_claims:
        state.fact_check_issues = [
            {
                "utterance_id": claim.utterance_id,
                "issue": claim.verdict,
                "correction": claim.suggested_correction
            }
            for claim in unsupported_claims
        ]

        # Don't retry here—let graph routing decide
        return state

    state.fact_check_complete = True
    return state

def route_after_fact_checker(state: Phase3GraphState) -> str:
    if state.fact_check_issues:
        if state.retry_count < state.max_retries:
            increment_retry(state, "fact_check_failures")

            # Prepare correction instructions for Dialogue Engine
            state.retry_instructions = {
                "retry_type": "fact_correction",
                "issues": state.fact_check_issues
            }

            return "dialogue_engine"  # Graph routes back
        else:
            # Give up, proceed with warnings
            logger.warning("Proceeding with fact-check warnings")
            return "qa_reviewer"

    return "qa_reviewer"
```

### Fallback Strategies

**Dialogue Engine Fallback:**
```python
def dialogue_engine_fallback(beat: int, chapter_title: str) -> List[Utterance]:
    """Generic fallback dialogue if LLM fails"""
    return [
        Utterance(
            utterance_id=f"fallback_b{beat}_u001",
            speaker="Host",
            beat=beat,
            text_clean="Let's explore this topic further.",
            intent="transition",
            emotion="neutral",
            grounding_chunk_ids=[]
        )
    ]
```

**Expert Expander Fallback:**
```python
def expert_expander_fallback(utterance: Utterance) -> Utterance:
    """If expansion fails, keep original"""
    logger.warning(f"Expansion failed for {utterance.utterance_id}, keeping original")
    utterance.text_with_naturalness = utterance.text_clean
    return utterance
```

---

## Quality Gates and Validation

### Pre-Node Validation

Before entering Phase 3 graph:

```python
def validate_phase3_input(chapter_input: ChapterInput) -> bool:
    """Validate input before processing"""

    # Required fields
    assert chapter_input.chapter_number > 0
    assert len(chapter_input.key_points) > 0
    assert len(chapter_input.source_chunks) > 0
    assert len(chapter_input.characters) == 3  # Host, Expert, Skeptic

    # Character validation
    for char in chapter_input.characters.values():
        assert char.tts_voice_id.startswith("en-US-")
        assert char.role in ["host", "expert", "skeptic"]

    # Duration sanity check
    assert 2.0 <= chapter_input.estimated_duration_minutes <= 8.0

    return True
```

### Post-Node Validation

After each node completes:

```python
def validate_dialogue_output(state: Phase3GraphState) -> bool:
    """Validate Dialogue Engine output"""
    for utt in state.utterances:
        # Schema compliance
        assert utt.speaker in state.chapter_input.characters.keys()
        assert 1 <= utt.beat <= 5
        assert utt.intent in ["question", "answer", "reaction", "challenge", "summary", "transition"]

        # Grounding check
        if utt.intent in ["answer", "challenge"]:
            # Factual claims should have grounding
            if len(utt.text_clean.split()) > 10:  # Non-trivial utterance
                assert len(utt.grounding_chunk_ids) > 0, f"Missing grounding: {utt.utterance_id}"

    return True

def validate_ssml_output(utterance: Utterance) -> bool:
    """Validate SSML syntax"""
    import xml.etree.ElementTree as ET

    try:
        ET.fromstring(utterance.text_ssml)
    except ET.ParseError as e:
        logger.error(f"Invalid SSML in {utterance.utterance_id}: {e}")
        return False

    # Check break times
    breaks = re.findall(r'<break time="(\d+)ms"/>', utterance.text_ssml)
    for break_time in breaks:
        assert 10 <= int(break_time) <= 10000, f"Invalid break time: {break_time}ms"

    return True
```

### Final Output Validation

Before returning Phase 3 output:

```python
def validate_final_output(output: ChapterDialogueOutput) -> bool:
    """Comprehensive final validation"""

    # Duration check
    total_duration = sum(utt.estimated_duration_seconds for utt in output.utterances)
    expected_duration = output.estimated_chapter_duration * 60
    assert 0.8 * expected_duration <= total_duration <= 1.2 * expected_duration

    # Quality threshold
    assert output.validation_metadata.listener_experience_score >= 6.0

    # SSML completeness
    for utt in output.utterances:
        assert utt.text_ssml is not None
        assert utt.tts_config is not None
        assert validate_ssml_output(utt)

    # Beat coverage (all 5 beats present)
    beats_covered = set(utt.beat for utt in output.utterances)
    assert beats_covered == {1, 2, 3, 4, 5}

    return True
```

---

## Google TTS Configuration

### Voice Selection Guidelines

**Google Cloud TTS Neural2 Voices (Recommended):**

| Voice ID | Gender | Description | Best For |
|----------|--------|-------------|----------|
| `en-US-Neural2-A` | Male | Warm, friendly | Host (casual) |
| `en-US-Neural2-C` | Female | Professional, clear | Host (authoritative) |
| `en-US-Neural2-D` | Male | Deep, authoritative | Expert (technical) |
| `en-US-Neural2-F` | Female | Energetic, expressive | Host (energetic) |
| `en-US-Neural2-G` | Female | Calm, measured | Expert (contemplative) |
| `en-US-Neural2-J` | Male | Young, casual | Skeptic (informal) |

### TTS API Call Structure

```python
from google.cloud import texttospeech

def synthesize_utterance(utterance: Utterance) -> bytes:
    """Convert SSML to audio using Google TTS"""

    client = texttospeech.TextToSpeechClient()

    # Configure synthesis input
    synthesis_input = texttospeech.SynthesisInput(ssml=utterance.text_ssml)

    # Configure voice parameters
    voice = texttospeech.VoiceSelectionParams(
        language_code=utterance.tts_config.language_code,
        name=utterance.tts_config.voice_id,
        ssml_gender=getattr(
            texttospeech.SsmlVoiceGender,
            utterance.tts_config.ssml_gender
        )
    )

    # Configure audio format
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,  # WAV
        speaking_rate=utterance.tts_config.speaking_rate,
        pitch=utterance.tts_config.pitch,
        sample_rate_hertz=24000  # High quality
    )

    # Synthesize
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    return response.audio_content  # WAV bytes
```

### SSML Best Practices for Google TTS

1. **Nested Prosody:** Avoid deep nesting (max 2 levels)
   ```xml
   <!-- Good -->
   <prosody rate="fast">Fast section</prosody>

   <!-- Avoid -->
   <prosody rate="fast"><prosody pitch="+2st"><prosody volume="loud">Too nested</prosody></prosody></prosody>
   ```

2. **Break Time Limits:** Stay within 10ms-10000ms (10 seconds)
   ```xml
   <!-- Good -->
   <break time="800ms"/>

   <!-- Invalid -->
   <break time="15000ms"/>  <!-- Exceeds 10s limit -->
   ```

3. **Emphasis Levels:** Use sparingly (max 3-4 per utterance)
   ```xml
   <!-- Good -->
   That's <emphasis level="strong">exactly</emphasis> right.

   <!-- Over-emphasized (loses impact) -->
   That's <emphasis>exactly</emphasis> <emphasis>right</emphasis> <emphasis>now</emphasis>.
   ```

4. **Rate/Pitch Ranges:** Stay within reasonable bounds
   ```xml
   <!-- Good -->
   <prosody rate="120%" pitch="+2st">Slightly faster and higher</prosody>

   <!-- Too extreme (sounds unnatural) -->
   <prosody rate="200%" pitch="+10st">Chipmunk voice</prosody>
   ```

---

## Testing and Validation

### Unit Tests (Per Node)

**Dialogue Engine Test:**
```python
def test_dialogue_engine_beat_generation():
    """Test single beat generation"""
    state = Phase3GraphState(
        chapter_input=sample_chapter_input,
        current_beat=1,
        beat_history=[],
        utterances=[]
    )

    result = dialogue_engine_node(state)

    # Assertions
    assert len(result.utterances) > 0
    assert all(utt.beat == 1 for utt in result.utterances)
    assert all(utt.speaker in ["Dr. Sarah Chen", "Prof. Marcus Thompson", "Jamie Rodriguez"]
               for utt in result.utterances)
    assert result.current_beat == 1  # Still on beat 1
```

**Expert Expander Test:**
```python
def test_expert_content_expansion():
    """Test expert utterance expansion"""
    short_utterance = Utterance(
        utterance_id="test_u001",
        speaker="Prof. Marcus Thompson",
        role="expert",
        text_clean="Neural networks learn from data.",
        word_count=5,
        intent="answer"
    )

    state = Phase3GraphState(utterances=[short_utterance])
    result = expert_expander_node(state)

    expanded = result.utterances[0]
    assert len(expanded.text_clean.split()) > 30  # Significantly expanded
    assert "Neural networks" in expanded.text_clean  # Core concept preserved
```

**Fact-Checker Test:**
```python
def test_fact_checker_validation():
    """Test fact verification"""
    state = Phase3GraphState(
        utterances=[
            Utterance(
                utterance_id="test_u001",
                text_clean="In 2024, AI surpassed humans on 78% of benchmark tasks.",
                grounding_chunk_ids=[12],
                intent="answer"
            )
        ],
        chapter_input=ChapterInput(
            source_chunks=[
                SourceChunk(
                    chunk_id=12,
                    content="The 2024 AI Index Report found AI systems now exceed human performance on 78% of benchmarks.",
                    source_url="https://example.com"
                )
            ]
        )
    )

    result = fact_checker_node(state)

    assert len(result.fact_check_issues) == 0  # Should pass
```

### Integration Tests (Full Graph)

```python
def test_phase3_full_pipeline():
    """End-to-end Phase 3 test"""
    chapter_input = ChapterInput(
        chapter_number=1,
        title="The AI Revolution",
        act="setup",
        energy_level="high",
        key_points=["What is AI?", "Brief history", "Why now?"],
        # ... full input
    )

    # Run graph
    result = phase3_graph.invoke({
        "chapter_input": chapter_input,
        "current_beat": 1,
        "retry_count": 0,
        "max_retries": 2,
        "utterances": []
    })

    # Assertions
    assert result["ssml_complete"] == True
    assert len(result["final_output"].utterances) >= 20  # Reasonable dialogue length
    assert result["final_output"].quality_checks_passed == True

    # Validate all utterances have SSML
    for utt in result["final_output"].utterances:
        assert utt.text_ssml.startswith("<speak>")
        assert utt.tts_config.voice_id.startswith("en-US-")
```

### Performance Benchmarks

| Metric | Target | Notes |
|--------|--------|-------|
| **Total Phase 3 Duration** | < 3 minutes | For single chapter, all nodes |
| **Dialogue Engine (per beat)** | < 30 seconds | LLM generation time |
| **Expert Expansion** | < 20 seconds | 3-5 utterances typically |
| **Naturalness Injection** | < 15 seconds | Hybrid rule + LLM |
| **Fact-Checking** | < 45 seconds | 8-12 claims on average |
| **QA Review** | < 30 seconds | Full chapter analysis |
| **SSML Annotation** | < 10 seconds | Mostly rule-based |

### Quality Metrics

Track these across chapters:

```python
class Phase3Metrics:
    avg_listener_experience_score: float  # Target: > 7.0
    fact_check_pass_rate: float  # Target: > 90%
    retry_rate: float  # Target: < 20%
    avg_utterances_per_chapter: int  # Target: 30-45
    avg_naturalness_markers_per_chapter: int  # Target: 15-25
    ssml_validation_pass_rate: float  # Target: 100%
```

---

## Summary

This design document provides a complete implementation blueprint for Phase 3: Dialogue Generation. Key takeaways:

### Architecture Decisions
- ✅ **6-node graph structure** (not monolithic)
- ✅ **Beat-by-beat generation** for manageability
- ✅ **Dedicated Expert Expander node** for content depth
- ✅ **Graph-level retry logic** via conditional edges
- ✅ **Google TTS SSML optimization** throughout

### Critical Success Factors
1. **Naturalness Injector** is the differentiator (chatbot → conversation)
2. **Fact-Checker** prevents expensive TTS regeneration from hallucinations
3. **QA Reviewer** catches listener experience issues early
4. **Retry limits** prevent cost explosion (max 2 retries)

### Next Steps
1. Implement each node as a separate Python module
2. Define LangGraph state schema and edges
3. Create comprehensive test suite
4. Integrate with Phase 2 (upstream) and Phase 4 (downstream)
5. Monitor quality metrics across production chapters

---

**Document Version:** 2.0
**Last Updated:** March 17, 2026
**Next Review:** After Phase 3 prototype implementation
