# Character Designer Agent — Phase 2.2

## Purpose
Creates distinct speaker personas for the podcast via a single LLM call. Each persona persists across all chapters in Phase 3 (Dialogue Generation).

## Pipeline Position
```
Phase 1 Output (ranked chunks)
    → Chapter Planner (4.1)
    → Character Designer (4.2)  ← this agent
    → Phase 3 (Dialogue Engine)
```

## Input
| Field | Source |
|---|---|
| `topic` | User input |
| `chapter_outlines` | Chapter Planner output (titles, key_points, acts, energy levels) |
| `num_speakers` | Settings (`NUM_SPEAKERS`, 2 or 3) |

## Output
`List[CharacterPersona]` — serialized into `phase2_results.json` alongside chapter outlines.

## Speaker Roles
- **2 speakers**: 1 Host (casual) + 1 Expert (technical)
- **3 speakers**: 1 Host (casual) + 1 Expert (technical) + 1 Skeptic (moderate). At least one must be female.

## Voice Bank
Uses **Gemini 2.5 Pro Preview TTS** voices. The LLM picks from 10 available voices based on character fit:

| Voice | Gender | Profile |
|---|---|---|
| Aoede | F | Breezy, clear, great for hosting |
| Charon | M | Deep, authoritative narrator |
| Puck | M | Upbeat, youthful, energetic |
| Kore | F | Firm, professional, news anchor |
| Fenrir | M | Excitable, high-energy |
| Leda | F | Youthful, conversational |
| Enceladus | M | Breathy, intimate, storytelling |
| Vindemiatrix | F | Gentle, calm |
| Zubenelgenubi | M | Casual, relaxed, everyman |
| Zephyr | F | Bright, friendly |

## Persona Schema Fields
`name`, `role`, `expertise_area`, `speaking_style`, `vocabulary_level`, `filler_patterns`, `reaction_patterns`, `disagreement_style`, `laugh_frequency`, `catchphrases`, `emotional_range`, `tts_voice_id`, `gender`

## Key Files
- Agent: `src/agents/phase2/character_designer.py`
- Model: `src/models/character.py`
- Prompt: `src/llm/prompts.py` → `CHARACTER_DESIGNER_PROMPT`
- Settings: `config/settings.py` → `CHARACTER_DESIGNER_MODEL`, `CHARACTER_DESIGNER_TEMPERATURE`
- Graph: `src/pipeline/phases/phase2_graph.py` → `character_designer_node`

## Design Decisions
1. **Single LLM call** — all characters generated together so the LLM contrasts them against each other.
2. **Chapter context fed to LLM** — expertise areas are tailored to the specific topic, not generic.
3. **LLM chooses voices** — voice selection is based on character personality fit, not random assignment.
4. **Validation** — asserts one host, correct speaker count, and unique voices.
