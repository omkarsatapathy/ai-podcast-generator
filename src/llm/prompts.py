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


# ==================== PHASE 2: CHARACTER DESIGNER ====================

CHARACTER_DESIGNER_PROMPT = """You are a podcast character designer. Create {num_speakers} distinct speaker personas for a podcast about: {topic}

CHAPTER CONTEXT (use this to tailor expertise and personality):
{chapters_context}

ROLE ASSIGNMENT RULES:
{role_rules}

AVAILABLE TTS VOICES (you MUST pick from this list):
{voices_list}

VOICE SELECTION RULES:
- Each character must use a DIFFERENT voice from the list above.
- Match voice personality to character role (e.g. authoritative voice for expert, energetic for host).
- The "tts_voice_id" field must be the exact voice Name from the list.
- The "gender" field must match the voice's gender.
{gender_rule}

PERSONA DESIGN RULES:
- Characters must CONTRAST each other — different vocabulary levels, different speaking styles.
- No two characters should have the same vocabulary_level.
- filler_patterns: 2-4 speech fillers unique to each character (e.g. "you know", "right?", "basically").
- reaction_patterns: 2-4 reactions unique to each character (e.g. "Oh interesting!", "Wait, really?").
- catchphrases: 2-3 signature phrases each character uses repeatedly.
- speaking_style: Describe HOW they talk, not WHAT they say.
- expertise_area: Must be specific to the podcast topic, not generic.
- emotional_range: What makes them excited, skeptical, or amused — specific to this topic.

HOST-SPECIFIC RULES (role=host):
The host is the engine of the conversation — not a passive moderator. Their persona must reflect ALL of the following active behaviors:
- Asking questions: Opens each chapter beat with a question, surfaces "what the listener at home is thinking".
- Interrupting: Jumps in mid-sentence when something is surprising or unclear — not rudely, but with genuine curiosity (e.g. "Wait — hold on — you're saying that…?").
- Counter-questioning: When an expert gives an answer, the host pushes back with "but what about…" or "okay but doesn't that mean…" to deepen the conversation.
- Plain-language recaps: After a complex explanation, the host summarises it in one sentence to confirm they understood and help the listener follow.
- Admitting confusion: Comfortable saying "I'm not fully with you yet — can you give me a real-world example?" — this is an asset, not a weakness.
- speaking_style must explicitly mention: asking follow-up questions, interrupting to clarify, recapping in plain terms, and playing devil's advocate.
- reaction_patterns must include at least one interruption-style reaction (e.g. "Wait, hold on —", "Hang on, say that again.").
- catchphrases must include at least one question-driven phrase (e.g. "So what does that actually mean for a normal person?", "Bottom line it for us.").
- disagreement_style must describe HOW the host challenges without being combative — e.g. playing confused listener, asking for concrete examples.

Make each character feel like a real person a listener would recognize and remember."""
