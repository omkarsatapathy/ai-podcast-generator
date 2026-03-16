"""Centralized LLM prompts for all phases."""


# ==================== PHASE 2: CHAPTER PLANNER ====================

CHUNK_ANALYSIS_PROMPT = """You are analyzing source material for a podcast episode about: {topic}

Analyze each content chunk below and extract structured metadata.

STRICT RULES:
- "topic": The main topic in EXACTLY 1-5 words. No more.
- "subtopics": 3-5 subtopics, each in EXACTLY 1-8 words. No more.
- "summary": A summary in EXACTLY 1-15 words. No more.
- "tone": One of: factual, opinion, debate, technical, narrative

CHUNKS TO ANALYZE:
{chunks_text}

Return a structured analysis for each chunk. Match chunk_id exactly as provided."""


NARRATIVE_SEQUENCE_PROMPT = """You are a podcast content architect designing the narrative flow for a podcast about: {topic}

You have {num_clusters} content clusters. Each cluster represents a group of related content.
Your job is to organize these into a compelling 3-act podcast structure.

CLUSTERS:
{clusters_summary}

STRUCTURE RULES:
1. Act 1 "setup" (2 chapters): Introduce the topic, provide background. Accessible to newcomers. The host "discovers" the topic alongside the listener.
2. Act 2 "explore" (3-5 chapters): Deep dives into sub-topics. Different angles, debates, expert perspectives. Each chapter should feel distinct.
3. Act 3 "resolve" (1 chapter): Synthesis, future outlook, takeaways. End on a forward-looking, thought-provoking note.

PACING RULES:
- Energy levels MUST alternate. Never place 3 consecutive chapters with the same energy level.
- "high" = debate, tension, surprising revelations
- "medium" = explanation, examples, context building
- "low" = reflection, synthesis, contemplation
- Recommended pattern: medium -> high -> medium -> high -> low -> medium -> low

DURATION RULES:
- Total duration must be between 25-28 minutes
- Each chapter: 2-5 minutes
- Setup chapters: ~3-4 min each
- Exploration chapters: ~4-5 min each
- Resolution chapter: ~3-4 min

Each cluster can appear in exactly one chapter. A chapter can contain 1-3 clusters.
Assign every cluster to a chapter. Do not leave any cluster unassigned.

Return the chapter sequence with act labels, energy levels, cluster assignments, and duration estimates."""


CHAPTER_OUTLINE_PROMPT = """You are creating detailed chapter outlines for a podcast about: {topic}

Below are the planned chapters with their content clusters and source material summaries.

{chapters_detail}

For EACH chapter, generate:
1. "title": An engaging, specific title (5-8 words). Make it catchy and topic-relevant. Avoid generic titles like "Introduction" or "Deep Dive".
2. "key_points": 3-5 specific discussion points the speakers should cover. Each point should be a concrete, actionable talking point (not vague).
3. "transition_hook": One compelling sentence that teases the next chapter and creates forward momentum. For the last chapter, use a thought-provoking closing hook.

QUALITY RULES:
- Titles must be specific to the content, not generic. BAD: "Getting Started". GOOD: "When Algorithms Learn to Diagnose".
- Key points must be concrete. BAD: "Discuss the technology". GOOD: "Compare accuracy rates of AI vs human radiologists in breast cancer screening".
- Transition hooks must create curiosity. BAD: "Next we'll talk about X". GOOD: "But not everyone is celebrating these breakthroughs..."

Return outlines for all {num_chapters} chapters."""
