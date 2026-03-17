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


# ==================== PHASE 3: DIALOGUE GENERATION ====================

BEAT_OBJECTIVES = {
    0: "OPENING (30-50s): The host warmly welcomes listeners, introduces each guest by name and expertise, briefly previews the topic, and has a short warm-up exchange. This sets the tone and lets the audience know who they're listening to.",
    1: "HOOK (15-30s): Open with a provocative statement, surprising fact, or compelling question. The host grabs attention and creates immediate curiosity.",
    2: "CONTEXT BUILDING (60-90s): Frame the topic. Expert provides necessary background. Host asks clarifying questions on behalf of the listener.",
    3: "DEEP DIVE + TENSION (90-120s): Core content. Expert explains in detail. Skeptic challenges claims and raises counterpoints. Create intellectual tension through respectful disagreement. Dynamic multi-speaker conversation, NOT a monologue.",
    4: "AHA MOMENT (30-60s): Someone offers an analogy or simplification that crystallizes the concept. Host reacts with a clear 'aha' moment. This is the memorable takeaway.",
    5: 'WRAP + TRANSITION (15-30s): Host summarizes the key point in 1-2 sentences. Tease next chapter: "{transition_hook}". Create forward momentum.',
}

DIALOGUE_BEAT_PROMPT = """You are an expert podcast dialogue writer. Generate natural, engaging multi-speaker dialogue.

Generate dialogue for **Beat {beat_number}** ({beat_name}) of Chapter {chapter_number}: "{chapter_title}".

## Chapter Info
- Act: {act} | Energy Level: {energy_level}
- Key Points: {key_points}

## Characters
{characters_text}

## Beat Objective
{beat_objective}

{previous_beats_text}

## Source Material (ground ALL factual claims here)
{source_chunks_text}

## Rules
- Target ~{target_words} words across ~{target_utterances} utterances
- Every factual claim MUST reference source chunks via grounding_chunk_ids
- Natural conversation flow — characters build on each other's points
- Each character speaks in their unique style consistently
- No meta-commentary like "(laughs)" or "[pauses]" — just spoken words
- Match the {energy_level} energy level in pacing and enthusiasm"""


OPENING_BEAT_PROMPT = """You are an expert podcast dialogue writer. Generate the OPENING segment for a podcast episode.

This is **Beat 0 (OPENING)** — the very first thing listeners hear after the intro music. It must:
1. The host warmly WELCOMES the audience to the show.
2. The host INTRODUCES each guest by name and briefly mentions their expertise/background.
3. The host gives a SHORT PREVIEW of today's topic — what they'll explore and why it matters.
4. A brief WARM-UP exchange: guests react to the topic preview, share initial excitement or a quick personal anecdote related to the topic. This should feel natural and conversational, like real people settling into a discussion.

## Topic
{topic}

## First Chapter
"{chapter_title}"

## Characters
{characters_text}

## Character Details (for introductions)
{personas_detail}

## Rules
- Target ~{target_words} words across ~{target_utterances} utterances
- The host MUST speak first — welcoming listeners
- The host MUST introduce each non-host character by name and role/expertise
- After introductions, have a 2-3 utterance warm-up exchange where characters react naturally
- Keep it warm, energetic, and conversational — NOT scripted or stiff
- No meta-commentary like "(laughs)" or "[pauses]" — just spoken words
- Every character should speak at least once during the opening
- Do NOT dive into the actual content yet — this is just the welcome and setup
- The opening should make a new listener feel oriented: who are these people, and what are we about to discuss?"""


EXPERT_EXPANSION_PROMPT = """Expand this brief expert podcast utterance to ~{target_words} words with more depth, examples, or context. Keep the conversational tone.

Speaker: {speaker_name} ({vocabulary_level} vocabulary)
Style: {speaking_style}
Original ({current_words} words): "{original_text}"

Context before: "{previous_text}"
Context after: "{next_text}"
Chapter key points: {key_points}

Source material for grounding:
{source_text}

Rules:
- Add depth without changing the core message
- Keep it spoken and conversational, not academic
- The next utterance must still make sense after expansion
- Stay grounded in source material

Return ONLY the expanded text."""


NATURALNESS_INJECTION_PROMPT = """Add naturalness markers to this podcast utterance to make it sound like spontaneous speech.

Speaker: {speaker_name} ({role}, {vocabulary_level})
Style: {speaking_style}
Text: "{text}"
Intent: {intent} | Emotion: {emotion} | Beat: {beat} | Energy: {energy_level}

Previous: "{previous_text}"
Next: "{next_text}"

Available markers:
- [FILLER:thinking] — "um," before complex statements
- [FILLER:agreement] — "yeah," at start of responses
- [PAUSE:short] — 400ms after questions, before lists
- [PAUSE:long] — 800ms before revelations
- [EMPHASIS:word] — stress key terms, stats
- [PACE:fast] — excited, listing things
- [PACE:slow] — important conclusions
- [LAUGH:light] — after humor
- [FALSE_START] — before nuanced points

Rules: Max 1-2 markers for <20 words, 2-4 for 20-50, 4-6 for >50. Use at most 1 filler ([FILLER:thinking] or [FILLER:agreement]) per utterance — never two fillers in a row. Casual speakers get occasional fillers; technical speakers get more pauses. No laugh during serious content.

Return ONLY the enhanced text with markers inserted."""


BATCH_FACT_CHECK_PROMPT = """Verify these podcast claims against the source material below.

## Claims
{claims_text}

## Source Material
{source_chunks_text}

For each claim, determine:
- "supported": source explicitly states or clearly implies the claim
- "partially_supported": source discusses topic but lacks specifics
- "unsupported": source doesn't mention or contradicts the claim"""


QA_REVIEW_PROMPT = """Review this podcast chapter script from a first-time listener's perspective.

## Chapter {chapter_number}: "{chapter_title}"
Act: {act} | Energy: {energy_level} | Key Points: {key_points}

## Script
{script_text}

## Evaluate
1. Repetition: Same concept/example repeated?
2. Engagement: Monologue traps (>3 consecutive same-speaker utterances)?
3. Clarity: Jargon used before defined?
4. Transition: Natural flow? Ending sets up next chapter?
5. Energy: Matches intended {energy_level} level?

Severity: critical (causes confusion/drop-off), warning (degrades quality), minor (stylistic)"""
