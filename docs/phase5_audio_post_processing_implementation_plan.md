# Phase 5: Audio Post-Processing — Detailed Implementation Plan

**Version:** 1.0
**Date:** March 17, 2026
**Primary Audio Libraries:** `pydub`, `ffmpeg`
**LLM Usage:** Claude Haiku (Cold Open script scan only — single node)
**Architecture Target:** LangGraph subgraph (`phase5_graph.py`)
**Authoring Context:** Built from `design_docs/design_doc.pdf` + current repository state (Phase 4 output contract fully known)

---

## 1. Executive Summary

Phase 5 is the final transformation stage of the entire pipeline. It receives raw per-utterance WAV files from Phase 4 and converts them into a single polished, publish-ready podcast MP3. This phase is almost entirely pure audio DSP (Digital Signal Processing) — no LLMs are involved except for one lightweight script-scan call inside the Cold Open Generator.

The four agents in this phase are:

1. **Audio Overlap Engine** — Takes individual WAV clips and `timing_directives` (INTERRUPT, BACKCHANNEL, LAUGH markers) from Phase 4 and mixes them into a single per-chapter audio timeline with natural conversational overlaps.
2. **Audio Post-Processor** — Applies professional mastering to each chapter's mixed audio: noise gating, EQ, dynamic compression, loudness normalization to -16 LUFS, and optional ambient room tone.
3. **Cold Open Generator** — Uses an LLM to identify the single most compelling 15–30 second moment from the episode script and extracts the corresponding audio slice from the already-rendered chapter files.
4. **Chapter Stitcher** — Assembles every component (cold open, intro music, all chapters, host outro, outro music) into the final podcast MP3 with proper ID3 metadata tags.

Phase 5 is complete when `podcast_episode_final.mp3` exists at the configured output path with correct loudness, metadata, and all content assembled in order.

---

## 2. Why This Phase Exists

Phase 4 produces a set of raw isolated WAV clips — one per utterance, per chapter. These clips are:

- **Sequential, not natural.** Each speaker waits for the previous to finish. No real podcast sounds like this.
- **Not professionally mixed.** Individual TTS clips have inconsistent volume, TTS-specific silence artifacts, and no room ambience.
- **Scattered.** There is no structure tying them into a single podcast episode. There is no cold open, no intro, no transitions, no outro.

Phase 5 exists to fix all three problems:

- The **Overlap Engine** solves the sequential problem by physically layering audio tracks at the timestamps where naturalness markers indicate overlaps should occur.
- The **Post-Processor** solves the mixing quality problem by applying the same mastering chain a human audio engineer would apply.
- The **Cold Open Generator** solves the first-impression problem by creating the hook that prevents listeners from skipping.
- The **Chapter Stitcher** solves the structure problem by assembling the complete episode in the correct narrative order with transitions and music.

Without this phase, the pipeline produces audio that is technically correct but unconvincing. Phase 5 is what makes the output sound like a real podcast rather than a read-aloud text file.

---

## 3. Total Information Flow

```
Phase 4 Output Contract
    chapter_audio_manifests[]        ← ordered WAV clips per chapter
    timing_metadata{}                ← INTERRUPT / BACKCHANNEL / LAUGH directives per utterance
    audio_files[]                    ← flat clip registry with paths, durations, speaker metadata
    voice_metadata{}                 ← speaker-to-voice mapping (needed for backchannel TTS)
    episode_id                       ← used for output file naming
    ready_for_phase5: true
          │
          ▼
[Step 1: Validate Phase 5 Input Contract]
    Ensure manifests are complete, all WAV files exist and are readable
    Resolve output directory structure
          │
          ▼
[Step 2: Audio Overlap Engine]   ← one pass per chapter
    Read ChapterAudioManifest clips in order_index order
    Build sequential timeline with 200–400ms gaps
    Apply INTERRUPT directives → shift clip start times, mix overlapping segments
    Apply BACKCHANNEL directives → generate/fetch short TTS clip, place at -8dB on secondary track
    Apply LAUGH directives → mix laughter clip at -4dB
    Apply 50–100ms cross-fades between speaker turns
    Emit: chapter_N_overlap_mixed.wav + utterance_timestamp_map{}
          │
          ▼
[Step 3: Audio Post-Processor]   ← one pass per chapter
    Noise gate: strip silence artifacts below -40dB
    EQ: boost 2–5 kHz presence, cut below 100 Hz
    Dynamic compression: 2:1 ratio, -20dB threshold
    Loudness normalisation: -16 LUFS (Apple / Spotify standard)
    Ambient room tone: overlay at -30dB to -35dB
    Emit: chapter_N_mastered.wav
          │
          ▼
[Step 4: Cold Open Generator]    ← one LLM call across all chapters
    LLM reads full final script → identifies top 2–3 compelling moments
    Select best moment (2–4 utterances, ~15–20 seconds audio)
    Prepend framing line: "Later in this episode..."
    Extract audio slice from mastered chapter WAV using utterance_timestamp_map
    Append transition sound (brief silence or sting)
    Emit: cold_open.wav
          │
          ▼
[Step 5: Chapter Stitcher]
    Assemble episode in order:
        cold_open.wav
        → intro_music.wav (fade in, cross-fade under cold open tail)
        → host_intro.wav (generated or static)
        → chapter_1_mastered.wav → transition_sting.wav
        → chapter_2_mastered.wav → transition_sting.wav
        → ... → chapter_N_mastered.wav
        → host_outro.wav
        → outro_music.wav (fade in under outro tail)
    Final loudness normalisation on complete assembled file
    Export as MP3 128kbps CBR (or 192kbps)
    Embed ID3 tags: title, episode number, description, chapter markers
    Emit: podcast_episode_final.mp3
          │
          ▼
[Step 6: Phase 5 Output Packaging]
    Verify MP3 exists and is decodable
    Record final duration, file size, chapter timestamps
    Emit: phase5_output contract
```

---

## 4. Cross-Cutting Implementation Rules

These rules apply throughout all Phase 5 nodes and must be respected by every function written.

### 4.1 Audio format consistency

All intermediate WAV files produced within Phase 5 must share a consistent format: **44100 Hz sample rate, 16-bit PCM, stereo (2 channels)**. If Phase 4 WAV files come in as mono or at a different sample rate (e.g., 24000 Hz from Gemini), a conversion step must be applied at the very start of the Overlap Engine before any mixing begins. `pydub` handles this via `AudioSegment.set_frame_rate()`, `set_sample_width()`, and `set_channels()`. This consistency requirement is non-negotiable because `pydub`'s `overlay()` and `append()` operations silently produce corrupted output when formats do not match.

### 4.2 Deterministic output paths

All intermediate and final output files must follow a fixed naming convention tied to `episode_id`:

```
data/audio/phase5/{episode_id}/overlap/chapter_{N}_overlap_mixed.wav
data/audio/phase5/{episode_id}/mastered/chapter_{N}_mastered.wav
data/audio/phase5/{episode_id}/cold_open.wav
data/audio/phase5/{episode_id}/final/podcast_episode_final.mp3
```

All paths must be created with `Path.mkdir(parents=True, exist_ok=True)` before writing. File writes must be atomic (write to `.tmp`, then `os.replace()` to final path) to prevent partial files from being consumed downstream.

### 4.3 Timestamp map is a first-class artifact

The `utterance_timestamp_map` — a dictionary mapping `utterance_id` to its absolute start and end timestamp in the final chapter audio (in milliseconds) — must be computed during the Overlap Engine and preserved in state. The Cold Open Generator cannot function without it. Every operation that shifts timing (interrupt overlap, backchannel insertion) must update this map accordingly.

### 4.4 pydub vs ffmpeg responsibilities

- **pydub** is used for: timeline construction, audio segment operations (overlay, concatenate, fade, slice), amplitude manipulation (volume adjustments in dB).
- **ffmpeg** is used for: EQ filtering, dynamic compression, loudness normalization (`loudnorm`), noise gating (`silenceremove`), MP3 encoding. These are invoked via Python `subprocess` with `ffmpeg` CLI commands. Do not use `pydub`'s export format conversion as a substitute for `ffmpeg` mastering.

### 4.5 No LLMs except in Cold Open Generator

Phase 5 is a pure audio processing phase. The only LLM call in the entire phase is inside the Cold Open Generator's script-scan step. No other node should make any LLM calls. This is critical for cost control — a 30-minute podcast should cost less than $0.10 in LLM calls for this entire phase.

### 4.6 Graceful degradation

If any single chapter fails during overlap mixing or post-processing, the pipeline must not abort. Mark that chapter as `degraded` in state, log the error, and continue with the remaining chapters. The Chapter Stitcher must be able to assemble an episode that excludes or uses a fallback version of a failed chapter rather than producing no output at all. This is aligned with the same philosophy used in Phase 4.

---

## 5. Phase 5 Node Plan (6 Steps)

---

### Step 1: Input Contract Validation Node

**Node name in graph:** `validate_phase5_input`

#### Objective
Guarantee that Phase 5 receives a complete, internally consistent, and physically present Phase 4 handoff before any audio processing begins. Fail fast with a detailed report if the input is broken.

#### Why we implement this
Phase 4 produces many intermediate artifacts. It is entirely possible that some WAV files were marked as `qc_passed` in the manifest but were later deleted, moved, or are zero-byte. Any audio processing node that tries to open a missing file will throw a cryptic exception mid-run. This validation node catches those issues at the boundary with a clear, actionable report.

#### Input
- `phase4_output` dict from Phase 4 state (containing `chapter_audio_manifests`, `timing_metadata`, `audio_files`, `voice_metadata`, `episode_id`, `ready_for_phase5`)
- Configured output directory root

#### Detailed process

**Sub-step 1.1 — Parse the Phase 4 contract:**
Extract `chapter_audio_manifests`, `timing_metadata`, `audio_files`, and `voice_metadata` from the incoming `phase4_output` dict. If `ready_for_phase5` is `false`, emit a blocking validation error immediately with the message "Phase 4 did not signal readiness for Phase 5." Do not attempt to work around this — if Phase 4 itself determined it was incomplete, Phase 5 must respect that.

**Sub-step 1.2 — Validate manifest completeness:**
For each `ChapterAudioManifest` in `chapter_audio_manifests`, verify:
- `manifest_version` field is present and non-empty.
- `complete` flag is `true`. If any manifest has `complete=false`, log a warning but do not block (allow degraded chapter handling).
- `clips` list is non-empty and each clip dict contains at minimum: `utterance_id`, `path`, `order_index`, `speaker`, `duration_seconds`.
- `timing_directives` list exists (may be empty if the chapter had no naturalness markers).

**Sub-step 1.3 — Validate physical file existence:**
For every clip in every manifest's `clips` list, check:
- `path` is a non-empty string.
- `Path(path).exists()` is `True`.
- `Path(path).stat().st_size > 0` (non-zero bytes).
- The file is a readable WAV: try `wave.open(path, "rb")` and read the header. If it raises an exception, mark this clip as `invalid`.

Build a `missing_clips` list of all clips that fail these checks. If more than a configurable threshold of clips are missing (default: if any chapter has more than 20% missing clips), set `phase5_blocked = True`. If below threshold, mark those individual chapters as `degraded` and allow the run to continue.

**Sub-step 1.4 — Validate timing directives:**
For each timing directive in `timing_metadata` (or inside each manifest's `timing_directives`), verify:
- `type` field is one of: `INTERRUPT`, `BACKCHANNEL`, `LAUGH`.
- `utterance_id` reference resolves to an actual clip in the same chapter's manifest.
- For `INTERRUPT` type: `duration_ms` field is present and is a positive integer.
- For `BACKCHANNEL` type: `speaker` field is present and maps to a known speaker in `voice_metadata`.
- For `LAUGH` type: `speaker` field is present.

Any directive referencing a non-existent utterance is silently dropped (logged as a warning) — do not block on orphaned timing directives.

**Sub-step 1.5 — Resolve and create output directory structure:**
Using the `episode_id`, create all required output directories:
- `data/audio/phase5/{episode_id}/overlap/`
- `data/audio/phase5/{episode_id}/mastered/`
- `data/audio/phase5/{episode_id}/final/`

If any directory cannot be created due to permissions, block immediately.

**Sub-step 1.6 — Detect format of incoming WAV clips:**
Open the first available WAV clip and read its technical parameters: `sample_rate`, `channels`, `sample_width`. Store these as `input_format` in state. This will be used by the Overlap Engine to decide whether format conversion is needed. All clips from Phase 4 should have the same format (they all came from the same TTS provider), so sampling one is sufficient — but log a warning if any clip differs.

#### Output
- `validated_manifests` — list of manifests confirmed to be processable
- `degraded_chapter_numbers` — list of chapter numbers with partial clip failures
- `validated_timing_directives` — cleaned timing directives with resolved clip references
- `input_audio_format` — dict with `sample_rate`, `channels`, `sample_width` of source clips
- `phase5_output_paths` — resolved output path dict keyed by artifact name
- `phase5_blocked` — boolean; true only on catastrophic input failure
- `phase5_validation_report` — full report with counts, warnings, and error details

#### Implementation guidance
- Use Pydantic model validation where possible (existing `ChapterAudioManifest` model from `src/models/phase4.py` is already defined and can be reused).
- Do not mutate `phase4_output` in place. Copy what is needed into new Phase 5 state keys.
- Keep this node pure — no audio processing, no file writes, no external calls.
- Unit-testable with fixture manifests pointing to real or mock WAV paths.

---

### Step 2: Audio Overlap Engine Node

**Node name in graph:** `run_overlap_engine`

**Agent file:** `src/agents/phase5/overlap_engine.py`

#### Objective
Transform the raw sequential utterance clips for each chapter into a single mixed stereo WAV file where the conversational timing markers (interruptions, backchannels, laughter) are physically baked into the audio timeline. This is what makes the podcast sound like a real conversation rather than a series of turns.

#### Why we implement this
Without overlap, every speaker waits for the previous speaker to finish before starting. Real human conversations never work this way. The overlap engine is the most perceptually impactful component in Phase 5 — it is the difference between "two chatbots reading turns" and "two people actually talking."

#### Input
- `validated_manifests` — from Step 1
- `validated_timing_directives` — from Step 1
- `input_audio_format` — from Step 1
- `voice_metadata` — needed for backchannel TTS generation (speaker voices)
- `phase5_output_paths` — to know where to write output files

#### Detailed process (per chapter — this entire process runs once per chapter)

**Sub-step 2.1 — Load and normalize all clips for the chapter:**

Load every clip from the chapter manifest in `order_index` order. Use `pydub.AudioSegment.from_wav(path)`. After loading, normalize each clip's format to the pipeline standard (44100 Hz, stereo, 16-bit PCM):
- If `sample_rate != 44100`: call `clip.set_frame_rate(44100)`.
- If `channels == 1`: call `clip.set_channels(2)` (convert mono to stereo by duplicating the channel).
- If `sample_width != 2`: call `clip.set_sample_width(2)`.

Store each normalized clip in a dict keyed by `utterance_id`. Also store the actual measured duration in milliseconds (use `len(clip)` in pydub, which returns milliseconds) — this is the ground truth duration, more reliable than the `estimated_duration_seconds` from Phase 4.

**Sub-step 2.2 — Build the sequential baseline timeline:**

Create an initially empty `AudioSegment` (zero duration). Iterate through clips in `order_index` order and append each one with a configurable gap between speakers. The gap value should come from config (default: 300ms). This is the "robotic" baseline before any overlaps are applied.

Simultaneously, build the `utterance_timestamp_map` for this chapter. For each utterance appended, record:
- `start_ms`: the absolute millisecond offset in the timeline where this utterance starts
- `end_ms`: `start_ms + len(clip)`

This map is critical — every downstream step (Cold Open Generator, Chapter Stitcher for chapter markers) depends on it.

**Sub-step 2.3 — Resolve timing directives into timeline operations:**

Parse the timing directives for this chapter and classify them into three operation lists:
- `interrupt_ops`: list of `{target_utterance_id, next_utterance_id, overlap_ms}` — these shift the start of `next_utterance_id` backwards in the timeline.
- `backchannel_ops`: list of `{listener_speaker, during_utterance_id, insertion_offset_ms}` — these inject a short background clip at a specific timestamp while another speaker is talking.
- `laugh_ops`: list of `{speaker, after_utterance_id}` — these insert a laughter clip immediately after an utterance.

Sort all ops by their target timestamp in the current timeline to ensure they are applied in chronological order. Apply them one at a time; each application must update the `utterance_timestamp_map` to reflect the new positions.

**Sub-step 2.4 — Apply INTERRUPT operations:**

For each `interrupt_op`:

1. Look up the current `start_ms` of `next_utterance_id` in `utterance_timestamp_map`.
2. Compute `new_start_ms = current_start_ms - overlap_ms` (shift backwards).
3. Verify `new_start_ms > start_ms_of_target_utterance` — the interrupt must start after the interrupted speaker began talking, not before.
4. Extract the tail segment of the current timeline from `new_start_ms` onwards. Extract the `next_utterance_id` clip.
5. Apply `-3dB` to the tail portion of the interrupted speaker's clip: `tail_segment = tail_segment - 3`. This volume reduction makes the interrupter audible over the person being interrupted.
6. Mix the interrupt clip over the tail: `mixed_tail = tail_segment.overlay(interrupt_clip)`.
7. Reconstruct the timeline: `timeline = timeline[:new_start_ms] + mixed_tail`.
8. Update `utterance_timestamp_map` for `next_utterance_id`: set `start_ms = new_start_ms`.
9. Update all utterances that came after `next_utterance_id` in the map: their positions may have shifted if `mixed_tail` changed total duration. Recompute them by summing sequential durations from this point forward.

**Sub-step 2.5 — Generate backchannel clips:**

For each `backchannel_op`:

The backchannel sound ("mm-hm", "yeah", "right", "uh-huh") is a very short audio clip (typically 0.5–1.5 seconds) spoken by the listening speaker. Two strategies are acceptable:

- **Strategy A (preferred):** Use the Phase 4 TTS infrastructure to generate the backchannel text via the same voice/provider that speaker uses. Import the TTS router from `src/agents/phase4/tts_router.py` and make a single utterance synthesis call for the backchannel text (e.g., `"mm-hm"` for the host's voice). Cache the result by `(speaker, text)` pair to avoid repeat API calls for the same sound.
- **Strategy B (fallback):** Ship a small set of royalty-free pre-recorded backchannel clips bundled in `data/audio/assets/backchannels/`. Select the appropriate clip by matching on rough speaker gender from voice metadata.

After obtaining the backchannel clip, normalize its format to match pipeline standard. Reduce its volume to `-8dB` relative to the main timeline: `backchannel_clip = backchannel_clip - 8`.

Compute the insertion point: look up `start_ms` of `during_utterance_id` in `utterance_timestamp_map`, then add `insertion_offset_ms` (default: 30% of the utterance's duration, to place the backchannel during the middle of the speech, not at the very start). Insert using `pydub.overlay()`: `timeline = timeline.overlay(backchannel_clip, position=insertion_point_ms)`. This does not change the total timeline duration — it only mixes the backchannel on top.

**Sub-step 2.6 — Apply LAUGH operations:**

For each `laugh_op`:

Laughter handling has two sub-strategies, configured per provider:

- **ElevenLabs path:** The laugh is already embedded in the TTS clip because the SSML/text included "hehe" or "haha". In this case, the `laugh_op` is informational only — no audio action is needed. Mark it as `handled_by_tts = true` and skip.
- **Generic path:** Fetch a short pre-recorded laughter clip from `data/audio/assets/laughs/`. Select by speaker gender (light/medium laughter type) based on `laugh_type` from the directive (`LAUGH:light` vs `LAUGH:medium`). Pitch-shift the laughter clip to approximately match the speaker's voice fundamental frequency (use ffmpeg's `atempo` or `asetrate` for a subtle shift — no more than ±15%). Apply `-4dB` volume reduction. Insert the laugh clip immediately after the utterance identified by `after_utterance_id` in the timeline — use `append()` (not `overlay()` since this is a sequential laugh after speech, not simultaneous).

**Sub-step 2.7 — Apply cross-fades between speaker turns:**

After all interrupt, backchannel, and laugh operations are applied, iterate through the timeline and identify all speaker-turn boundaries (positions where `utterance_N.speaker != utterance_{N+1}.speaker`). At each such boundary, apply a 75ms cross-fade (default, configurable):

Use pydub's `fade_out()` on the last 75ms of the ending clip and `fade_in()` on the first 75ms of the starting clip, then mix them. This eliminates the hard "click" artefact at audio segment boundaries that comes from abrupt amplitude changes at cut points.

**Sub-step 2.8 — Export chapter overlap-mixed file:**

Export the completed timeline as a WAV file to the configured path: `data/audio/phase5/{episode_id}/overlap/chapter_{N}_overlap_mixed.wav`. Use `pydub`'s `export()` with `format="wav"`. Use the atomic write pattern: export to a `.tmp` path first, then `os.replace()` to the final path.

**Sub-step 2.9 — Validate the output:**

After export, reopen the file using `src/tools/audio_tools.inspect_wav_file()` (already implemented in the codebase). Verify: duration is greater than the sum of individual clip durations minus a small tolerance (overlaps shorten total duration), file is non-empty, and format matches pipeline standard. Log a warning if duration is unexpectedly short (may indicate an overlay error).

#### Output (per chapter — accumulated into list across all chapters)
- `chapter_mixed_audio_paths` — dict: `{chapter_number: path_to_chapter_N_overlap_mixed.wav}`
- `utterance_timestamp_maps` — dict: `{chapter_number: {utterance_id: {start_ms, end_ms}}}`
- `overlap_engine_reports` — list of per-chapter processing reports with stats

#### Implementation guidance
- The entire logic for this node belongs in `src/agents/phase5/overlap_engine.py`. The node function in `phase5_graph.py` should only call the agent's `run()` method and update state.
- Process chapters sequentially (not in parallel) to keep memory usage bounded. A single chapter of mixed audio at 44100 Hz stereo 16-bit is approximately 15–30 MB in memory. Processing 6–8 chapters simultaneously would require 120–240 MB.
- Keep the baseline timeline building, interrupt application, backchannel insertion, and cross-fading as separate private functions. This makes unit-testing each operation possible in isolation.
- All millisecond offsets must use integers, not floats, when passed to pydub (pydub slice notation uses integer milliseconds).

---

### Step 3: Audio Post-Processor Node

**Node name in graph:** `run_post_processor`

**Agent file:** `src/agents/phase5/post_processor.py`

#### Objective
Apply a professional mastering chain to each chapter's overlap-mixed audio to achieve consistent, clean, broadcast-standard sound quality. This eliminates TTS-specific artifacts, normalizes dynamic range, and optionally adds subtle ambience to fill unnatural silences.

#### Why we implement this
TTS providers output audio that is technically correct but not broadcast-ready. Common problems include: silence regions with subtle digital noise floor, volume inconsistency between speakers (TTS-generated speech has no natural variation), unnatural silence between turns, and clinical cleanness that makes the recording sound synthetic. Each mastering step below addresses one of these problems.

#### Input
- `chapter_mixed_audio_paths` — from Step 2: paths to `chapter_N_overlap_mixed.wav` files
- `degraded_chapter_numbers` — from Step 1: to skip or flag failed chapters
- `phase5_output_paths` — configured output directories

#### Detailed process (per chapter)

The entire mastering chain for a chapter is implemented as a sequence of `ffmpeg` CLI invocations chained together. Each step writes to a temporary intermediate file, and the final step writes to the permanent output path. All intermediate temp files are cleaned up regardless of success or failure (use `try/finally`).

**Sub-step 3.1 — Noise Gate:**

Goal: Remove extremely quiet noise artifacts that TTS providers sometimes inject into silence regions.

Use `ffmpeg`'s `silenceremove` filter. Command pattern:

```
ffmpeg -i chapter_N_overlap_mixed.wav \
  -af "silenceremove=stop_periods=-1:stop_duration=0.1:stop_threshold=-40dB" \
  chapter_N_step1_noisegate.wav
```

Parameter explanation:
- `stop_periods=-1`: apply to all silence regions in the file, not just the end.
- `stop_duration=0.1`: only remove silence runs longer than 100ms (this preserves the intentional short gaps between turns that were set in the Overlap Engine).
- `stop_threshold=-40dB`: anything below -40dB is considered silence noise.

After this step, verify the output file exists and has duration within 5% of the input. If duration dropped more than 5%, log a warning — this means the silence threshold was too aggressive and removed intentional pauses.

**Sub-step 3.2 — Equalisation (EQ):**

Goal: Make voices sound clear and present. Reduce low-frequency rumble.

Use `ffmpeg`'s `equalizer` filter applied twice in sequence (one boost, one cut):

```
ffmpeg -i chapter_N_step1_noisegate.wav \
  -af "equalizer=f=3000:t=o:w=2000:g=2,equalizer=f=80:t=o:w=100:g=-6" \
  chapter_N_step2_eq.wav
```

Parameter explanation for the two filter stages:
- `f=3000:t=o:w=2000:g=2`: Boost frequencies centered at 3000 Hz (the vocal presence range) with a bandwidth of 2000 Hz and a gain of +2dB. This adds clarity and "cut-through" to voices.
- `f=80:t=o:w=100:g=-6`: Cut frequencies centered at 80 Hz with a bandwidth of 100 Hz and a gain of -6dB. This removes low-frequency rumble and muddiness common in TTS output at lower pitches.

These values are gentle and safe for speech. They should be exposed as config values rather than hardcoded so they can be tuned without code changes.

**Sub-step 3.3 — Dynamic Compression:**

Goal: Even out volume differences between speakers and between loud and quiet passages.

Use `ffmpeg`'s `acompressor` filter:

```
ffmpeg -i chapter_N_step2_eq.wav \
  -af "acompressor=threshold=-20dB:ratio=2:attack=5:release=50:makeup=2" \
  chapter_N_step3_compressed.wav
```

Parameter explanation:
- `threshold=-20dB`: Start compressing when signal exceeds -20dB.
- `ratio=2`: For every 2dB above threshold, only 1dB passes through (gentle ratio — podcast mastering does not use aggressive compression).
- `attack=5`: Compression kicks in over 5ms (fast enough to catch loud consonants).
- `release=50`: Compression releases over 50ms (slow enough to not create pumping artifacts).
- `makeup=2`: Apply +2dB makeup gain after compression to restore perceived loudness.

This is the "standard podcast mastering" compression profile. It is not meant to be heavy — just to reduce the dynamic range enough that listeners don't constantly adjust their volume.

**Sub-step 3.4 — Loudness Normalisation:**

Goal: Normalise the chapter to -16 LUFS, which is the standard target for Apple Podcasts and Spotify.

Use `ffmpeg`'s `loudnorm` filter in two-pass mode for the most accurate result:

**Pass 1** (analysis only — no output file written):
```
ffmpeg -i chapter_N_step3_compressed.wav \
  -af loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json \
  -f null -
```

Capture stdout JSON output which contains measured loudness values (`input_i`, `input_tp`, `input_lra`, `input_thresh`, `offset`, `target_offset`).

**Pass 2** (apply with measured values for linear normalisation):
```
ffmpeg -i chapter_N_step3_compressed.wav \
  -af loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={input_i}:measured_TP={input_tp}:measured_LRA={input_lra}:measured_thresh={input_thresh}:offset={target_offset}:linear=true \
  chapter_N_step4_normalized.wav
```

Using two-pass `loudnorm` with `linear=true` gives more accurate results than single-pass, which uses a more aggressive dynamic normalization algorithm. The `-16 LUFS` target and `-1.5 TP` (True Peak) values are the Apple/Spotify recommendation.

**Sub-step 3.5 — Ambient Room Tone (Improvement #4 from design doc):**

Goal: Fill the "digital silence" between utterances — the perfectly clean zero-noise gaps that unmask TTS audio as synthetic — with a very subtle room ambience.

This step is **optional** and controlled by a config flag `enable_ambient_room_tone: bool` (default: `true`). If disabled, skip to Sub-step 3.6.

Room tone asset selection:
- Check for `data/audio/assets/room_tone/room_tone_default.wav` in the repository. This file must be a royalty-free, seamlessly loopable room ambience (soft background hum, very subtle). If it does not exist, log a warning and skip this step rather than failing.
- The asset must be long enough to cover the chapter duration. If it is shorter, create a looped version using `pydub`'s `AudioSegment * N` repeat operator before overlaying.

Apply in Python with pydub (not ffmpeg) because overlaying at precise dB levels is more convenient:
1. Load the normalized chapter WAV: `chapter = AudioSegment.from_wav(chapter_N_step4_normalized.wav)`.
2. Load and loop the room tone asset to match chapter duration: `room_tone = (room_tone_clip * repeats)[:len(chapter)]`.
3. Reduce room tone to -32dB: `room_tone = room_tone - 32`.
4. Overlay: `chapter_with_ambience = chapter.overlay(room_tone)`.
5. Export: `chapter_with_ambience.export(chapter_N_step5_ambience.wav, format="wav")`.

The room tone must never exceed -30dB in the mix under any circumstances. It should be inaudible when speech is present but perceptible in silence regions.

**Sub-step 3.6 — Export mastered chapter file:**

Copy (or rename, if no ambience step ran) the final step's output to the permanent mastered path: `data/audio/phase5/{episode_id}/mastered/chapter_{N}_mastered.wav`. Use atomic write (temp file rename pattern).

**Sub-step 3.7 — Validate mastered output:**

Use `src/tools/audio_tools.inspect_wav_file()` to verify the mastered file:
- Duration is within 10% of the overlap-mixed input (mastering should not dramatically change duration).
- File is non-empty and non-corrupt.
- Sample rate and channels match pipeline standard.

Log key mastering stats in the processing report: input duration, output duration, loudness target achieved, steps applied.

#### Output
- `chapter_mastered_audio_paths` — dict: `{chapter_number: path_to_chapter_N_mastered.wav}`
- `post_processor_reports` — list of per-chapter mastering reports

#### Implementation guidance
- All ffmpeg calls are synchronous `subprocess.run()` calls. Do not use asyncio here — the chapter-by-chapter loop is sequential by design.
- All ffmpeg CLI commands must specify `-y` flag to auto-overwrite output files (for idempotent re-runs) and `-loglevel error` to suppress verbose ffmpeg stdout, which would clutter logs.
- Capture ffmpeg stderr and include it in the per-chapter report for debugging.
- Verify `ffmpeg` is available at startup of this node (run `subprocess.run(["ffmpeg", "-version"])` and fail clearly if it is not installed).
- The full mastering chain logic belongs in `src/agents/phase5/post_processor.py`. Expose a `run_mastering_chain(input_path, output_path, config, episode_id, chapter_number)` function.
- Clean up all intermediate temp files in a `finally` block regardless of success or failure.

---

### Step 4: Cold Open Generator Node

**Node name in graph:** `generate_cold_open`

**Agent file:** `src/agents/phase5/cold_open_generator.py`

#### Objective
Create a 15–30 second attention-grabbing cold open teaser — the very first audio the listener hears before the intro music — by identifying the single most compelling moment in the episode and extracting it from the already-rendered audio. This is the only node in Phase 5 that uses an LLM.

#### Why we implement this
Every professional podcast opens with a cold open. It exists to prevent listeners from skipping. The cold open should not introduce the topic — it should play a moment from the episode so compelling that the listener is immediately hooked. Since all chapter audio is already rendered at this point, generating the cold open requires no new TTS calls — just one LLM call to identify the right moment and then a pydub audio slice operation.

#### Input
- `chapter_dialogues` — the final script JSON (all chapters, all utterances) — needed for the LLM script scan
- `chapter_mastered_audio_paths` — from Step 3: paths to mastered chapter WAVs
- `utterance_timestamp_maps` — from Step 2: `{chapter_number: {utterance_id: {start_ms, end_ms}}}`
- `phase5_output_paths` — for the cold open output path
- LLM configuration (model: Claude Haiku, temperature: 0.3)

#### Detailed process

**Sub-step 4.1 — Prepare the script for LLM scanning:**

Build a condensed text representation of the full episode script for the LLM. For each utterance across all chapters (in order), format as:
```
[CH{chapter_number} | {speaker} | utterance_id={utterance_id}]
{text_clean}
```

Concatenate all utterances with newline separators. This format must be compact to stay within a single LLM call's input token budget. For a 30-minute podcast (~4500 words / ~6000 tokens), this fits comfortably within Claude Haiku's context window.

Do **not** include SSML tags in the text sent to the LLM — use `text_clean` (the plain-text version), not `text_ssml`. The LLM needs to read the content, not parse markup.

**Sub-step 4.2 — LLM script scan call:**

Make a single LLM call to Claude Haiku (via the existing `src/llm/llm_factory.py` infrastructure) with the following system prompt and instruction:

System prompt:
> You are a podcast producer assistant. Your job is to identify the single most compelling moment from a podcast episode script that will be used as the cold open (teaser). A compelling moment is one that: creates curiosity or surprise in a listener who hasn't heard the episode, features strong emotional contrast between speakers (excitement vs. skepticism, wonder vs. challenge), or contains a memorable analogy, a surprising fact, or a moment of genuine disagreement. You must NOT select opening or introductory moments — the cold open should feel like it drops the listener into the middle of something interesting.

User instruction:
> Below is the full transcript of the episode. Identify the 3 best candidate moments. For each candidate, return a JSON object with the following fields:
> - `candidate_rank` (1 = best, 2 = second best, 3 = third best)
> - `chapter_number` (integer)
> - `start_utterance_id` (string — the utterance_id where the excerpt should start)
> - `end_utterance_id` (string — the utterance_id where the excerpt should end, inclusive)
> - `reason` (1–2 sentences explaining why this is compelling for a cold open)
>
> Return ONLY a JSON array with exactly 3 objects. No other text.

Append the full script text prepared in Sub-step 4.1.

**Sub-step 4.3 — Parse LLM response and select best candidate:**

Parse the LLM JSON response. Validate:
- Response is a valid JSON array.
- Contains at least 1 item.
- Each item has required fields: `chapter_number`, `start_utterance_id`, `end_utterance_id`.
- `chapter_number` resolves to an actual chapter in `utterance_timestamp_maps`.
- Both `start_utterance_id` and `end_utterance_id` exist in the timestamp map for that chapter.

If the response fails validation (malformed JSON, bad utterance IDs), implement a simple fallback: use the utterance with `beat == 3` (Deep Dive/Tension beat) from the chapter with the most total utterances. This guarantees a cold open even if the LLM fails.

Select `candidate_rank == 1` as the primary choice.

**Sub-step 4.4 — Compute audio slice boundaries:**

Using `utterance_timestamp_maps`, look up:
- `excerpt_start_ms = timestamp_map[chapter_number][start_utterance_id]["start_ms"]`
- `excerpt_end_ms = timestamp_map[chapter_number][end_utterance_id]["end_ms"]`

Compute `excerpt_duration_ms = excerpt_end_ms - excerpt_start_ms`.

Validate that the excerpt duration is between 12,000ms (12 seconds) and 25,000ms (25 seconds). If outside this range:
- If too short (< 12s): extend `end_utterance_id` to the next 1–2 utterances in the same chapter.
- If too long (> 25s): trim `end_utterance_id` back by 1 utterance at a time until within range.

**Sub-step 4.5 — Generate the framing line audio:**

The cold open begins with a host framing line like: `"Later in this episode..."`. This line must be spoken in the host character's voice.

Two strategies:
- **Strategy A (preferred):** Call the Phase 4 TTS infrastructure (import `src/agents/phase4/tts_router.py`) to synthesize `"Later in this episode..."` using the host speaker's voice ID from `voice_metadata`. This ensures voice consistency.
- **Strategy B (fallback):** Use a pre-recorded framing line clip from `data/audio/assets/framing/later_in_this_episode.wav` if it exists.

Load the framing line as an `AudioSegment`. Normalize it to match pipeline format. Apply the same loudness as the chapter audio (use pydub's `match_dB()` approach or simply normalize to the same target dBFS).

**Sub-step 4.6 — Extract the audio excerpt:**

Load the mastered chapter WAV for the selected chapter: `chapter_audio = AudioSegment.from_wav(chapter_mastered_audio_paths[chapter_number])`.

Slice the excerpt: `excerpt = chapter_audio[excerpt_start_ms:excerpt_end_ms]`.

Apply a 200ms fade-in at the start of the excerpt and a 300ms fade-out at the end of the excerpt to prevent abrupt audio entry/exit:
```python
excerpt = excerpt.fade_in(200).fade_out(300)
```

**Sub-step 4.7 — Add transition sound after excerpt:**

After the excerpt, append a short 500ms silence, then optionally a short transition sound effect (a subtle audio sting of 1–2 seconds). Check for a transition asset at `data/audio/assets/transitions/cold_open_end.wav`. If it exists, load and append it with a 100ms fade-in. If it does not exist, append 1000ms of silence.

**Sub-step 4.8 — Assemble the full cold open:**

Concatenate the components in order:
1. Framing line audio (with 100ms silence after it)
2. Excerpt audio
3. Transition sound

Total target: 15–30 seconds. Log the actual assembled duration.

**Sub-step 4.9 — Export cold open:**

Export to `data/audio/phase5/{episode_id}/cold_open.wav` using the atomic write pattern. After export, validate: duration is between 10,000ms and 32,000ms, file is non-empty and readable.

#### Output
- `cold_open_path` — path to `cold_open.wav`
- `cold_open_report` — dict with `selected_candidate`, `chapter_number`, `utterance_range`, `duration_ms`, `framing_line_strategy_used`
- `cold_open_failed` — boolean, `true` if generation failed even after fallback (cold open is optional — stitcher should proceed without it)

#### Implementation guidance
- Use `src/llm/llm_factory.py` to get the LLM client — do not hardcode API calls.
- Use Claude Haiku (`claude-haiku-4-5-20251001`) for the script scan — this is a simple selection task, not reasoning-heavy.
- The LLM call should have a timeout of 30 seconds. If it times out, use the fallback strategy from Sub-step 4.3.
- The entire cold open generation is best-effort: if the node raises any exception, log it, set `cold_open_failed = True`, and continue. The Chapter Stitcher will check this flag and assemble without a cold open if needed.
- All cold open logic belongs in `src/agents/phase5/cold_open_generator.py`.

---

### Step 5: Chapter Stitcher Node

**Node name in graph:** `run_chapter_stitcher`

**Agent file:** `src/agents/phase5/chapter_stitcher.py`

#### Objective
Assemble all audio components — cold open, intro music, all mastered chapter files, inter-chapter transition sounds, host outro, and outro music — into the single final podcast episode file, export as MP3, and embed ID3 metadata tags.

#### Why we implement this
All prior steps produce individual audio components. They are not yet a podcast episode — they are parts. The Chapter Stitcher is the final assembly step that creates the artifact the user actually receives: a complete, ready-to-publish MP3 file.

#### Input
- `cold_open_path` — from Step 4 (may be `None` if cold open failed)
- `cold_open_failed` — boolean flag from Step 4
- `chapter_mastered_audio_paths` — from Step 3: ordered dict of chapter paths
- `degraded_chapter_numbers` — from Step 1: chapters to flag in output
- `phase5_output_paths` — configured output paths
- `topic` — from pipeline state (for ID3 title tag)
- `episode_id` — from pipeline state (for file naming and ID3 tags)
- `character_personas` — from pipeline state (for host name extraction in intro text)
- Asset paths: `data/audio/assets/music/intro_music.wav`, `data/audio/assets/music/outro_music.wav`, `data/audio/assets/transitions/chapter_sting.wav`

#### Detailed process

**Sub-step 5.1 — Load and validate all input components:**

Before assembling anything, verify every required audio component:
- For each chapter in `chapter_mastered_audio_paths`: verify file exists, is non-empty, is readable as WAV.
- If `cold_open_failed == False`: verify `cold_open_path` exists and is readable.
- Check for optional music assets. Do not fail if assets are missing — the stitcher must work without music files (it will use silence instead). Log a warning for each missing optional asset.

Build an `asset_availability` dict: `{"cold_open": bool, "intro_music": bool, "outro_music": bool, "chapter_sting": bool}`.

**Sub-step 5.2 — Load all audio assets:**

Load the following as `AudioSegment` objects, normalizing format (44100 Hz, stereo, 16-bit) for each:

- Cold open: `AudioSegment.from_wav(cold_open_path)` — or `AudioSegment.silent(duration=0)` if unavailable.
- Intro music: load from `data/audio/assets/music/intro_music.wav` or `AudioSegment.silent(duration=0)` if missing.
- Outro music: load from `data/audio/assets/music/outro_music.wav` or `AudioSegment.silent(duration=0)` if missing.
- Chapter transition sting: load from `data/audio/assets/transitions/chapter_sting.wav` or `AudioSegment.silent(duration=500)` (500ms silence) if missing.
- All mastered chapter files: iterate `chapter_mastered_audio_paths` in sorted chapter number order.

**Sub-step 5.3 — Prepare host intro clip:**

The host intro is a short spoken segment: `"Welcome to [Podcast Name]. I'm your host, [Host Name]. Today we're exploring: [Topic]."` (~15 seconds when spoken at 150 wpm).

Check for a pre-rendered host intro at `data/audio/phase5/{episode_id}/host_intro.wav`. If it exists (e.g., from a previous run), use it. If not:

- Extract the host character's name from `character_personas` (look for `role == "host"`).
- Construct the intro text: `"Welcome to [topic] — A Deep Dive Podcast. I'm your host, [host_name]. Today, we're exploring: {topic}. Let's dive in."`
- Synthesize via Phase 4 TTS router using the host's `tts_voice_id`. Use `Strategy A` from Step 4.5 (TTS call). Export to `data/audio/phase5/{episode_id}/host_intro.wav`.

If TTS synthesis fails, fall back to `AudioSegment.silent(duration=2000)` (2 seconds of silence) as the host intro.

Similarly, prepare a host outro: `"That's all for today's episode. Thank you for listening. If you found this valuable, please subscribe and share. Until next time."` Using the same synthesis strategy. Export to `data/audio/phase5/{episode_id}/host_outro.wav`.

**Sub-step 5.4 — Build the intro music cross-fade:**

If intro music is available:
- Trim intro music to exactly 8 seconds: `intro_music = intro_music[:8000]`.
- The last 3 seconds of the cold open (if present) should have the intro music fading in underneath it. Implement this as:
  1. Take the last 3000ms of the cold open: `cold_open_tail = cold_open[-3000:]`.
  2. Reduce intro music by -8dB and trim to 3000ms: `intro_fade_in = intro_music[:3000] - 8`.
  3. Mix: `blended_cold_open_tail = cold_open_tail.overlay(intro_fade_in)`.
  4. Replace the cold open tail: `cold_open_with_intro = cold_open[:-3000] + blended_cold_open_tail`.
  5. Append the remaining 5 seconds of intro music at full volume: `assembled_intro = cold_open_with_intro + intro_music[3000:]`.

If no cold open, the intro music plays standalone for 8 seconds before the host intro.

**Sub-step 5.5 — Assemble the complete episode timeline:**

Build the final episode as a sequence of concatenations. Use an `episode = AudioSegment.empty()` accumulator and append each component. Record the absolute start timestamp (in milliseconds from episode start) of each component in a `chapter_markers` dict — this will be embedded in the MP3 as chapter markers.

Assembly order:
1. `cold_open_with_intro` (or `intro_music` alone if no cold open): add to episode. Record timestamp.
2. 500ms silence gap.
3. `host_intro_clip`: add to episode. Record as "Introduction" chapter marker.
4. 300ms silence gap.
5. For each chapter `N` in sorted order:
   a. Add `chapter_N_mastered.wav` to episode. Record `chapter_N_start_ms` as chapter marker with label = chapter title (from `chapter_dialogues[N]["title"]` if available, else `"Chapter {N}"`).
   b. If this is not the last chapter: append `chapter_sting` (500ms transition sound or silence).
   c. After the sting: append 200ms silence.
6. `host_outro_clip`: add to episode. Record as "Outro" chapter marker.
7. 500ms silence.
8. `outro_music` at -6dB under/after the outro: implement same cross-fade as intro — outro music fades in under the last 3 seconds of the host outro.

**Sub-step 5.6 — Apply final loudness normalisation to complete episode:**

The assembled episode may have slight loudness variation at seam points due to different processing paths. Apply a final loudness normalisation pass to the complete assembled episode.

Export the complete `AudioSegment` to a temporary WAV: `data/audio/phase5/{episode_id}/final/episode_assembled_temp.wav`. Then run the two-pass `ffmpeg loudnorm` chain from Step 3.4 on this assembled WAV, targeting -16 LUFS. Write the final normalized WAV to `data/audio/phase5/{episode_id}/final/episode_final_normalized.wav`.

**Sub-step 5.7 — Encode to MP3:**

Run `ffmpeg` to encode the final normalized WAV to MP3:

```
ffmpeg -i episode_final_normalized.wav \
  -codec:a libmp3lame \
  -b:a 128k \
  -q:a 2 \
  podcast_episode_final.mp3
```

Bitrate is 128 kbps CBR by default. Expose a config key `mp3_bitrate: int` (accepted values: 96, 128, 192) to allow higher quality. Use atomic write to final path.

**Sub-step 5.8 — Embed ID3 metadata tags:**

Use the `mutagen` Python library (add to `requirements.txt` if not present) to write ID3 tags to the MP3 file after encoding:

Required tags:
- `TIT2` (Title): `{topic} — Episode {episode_id}`
- `TPE1` (Artist): `AI Podcast Generator`
- `TALB` (Album): `AI Podcast Generator`
- `TDRC` (Year): current year (from `datetime.date.today().year`)
- `TCON` (Genre): `Podcast`
- `COMM` (Comment): Episode description, sourced from the chapter plan summary if available, otherwise `f"An AI-generated podcast episode on the topic: {topic}."`

Optional but important — Chapter markers using `CHAP` ID3 frames. Use `mutagen`'s `CTOC` and `CHAP` tags. For each entry in `chapter_markers`:
- `CTOC` (Table of Contents) frame listing all chapter IDs.
- `CHAP` frame per chapter with: element ID, start time (ms), end time (ms), title string.

Chapter markers allow podcast players (Apple Podcasts, Overcast, Pocket Casts) to show named chapters with skip navigation — a professional feature that significantly improves listener experience.

**Sub-step 5.9 — Validate final MP3:**

After all tags are written, validate the final MP3:
- File exists and `st_size > 0`.
- Duration is within expected range: use `mutagen.mp3.MP3(path).info.length` — should be between 600 seconds (10 minutes) and 2400 seconds (40 minutes).
- ID3 tags are readable: open with `mutagen.id3.ID3(path)` and verify title tag is present.
- Log a final summary: total duration in minutes:seconds format, file size in MB, chapter count, bitrate.

#### Output
- `final_podcast_mp3_path` — absolute path to `podcast_episode_final.mp3`
- `chapter_markers` — dict of chapter timestamps embedded in the file
- `stitcher_report` — assembly report: total duration, file size, bitrate, chapters assembled, degraded chapters included, cold open present/absent, assets used

#### Implementation guidance
- The assembly is performed entirely in pydub and Python — `ffmpeg` is only called for the final loudness pass and MP3 encoding.
- All pydub concatenation operations are O(n) in memory. A 30-minute stereo 44100 Hz episode is approximately 300 MB in memory as an `AudioSegment`. This is acceptable for development hardware but should be noted in deployment docs.
- Add `mutagen` to `requirements.txt` (check it is not already present first).
- All assembly logic belongs in `src/agents/phase5/chapter_stitcher.py`.
- The Chapter Stitcher must never fail if optional components (cold open, music assets) are missing. It is the only node where "best effort with graceful degradation" is the primary design goal.

---

### Step 6: Phase 5 Output Packaging Node

**Node name in graph:** `package_phase5_output`

#### Objective
Emit a clean, versioned Phase 5 output contract that documents the final episode artifact, its quality, and any degraded or missing components, so the pipeline can be audited and the result can be consumed by any downstream system (dashboard, file store, API response).

#### Why we implement this
The pipeline needs a clean termination point with a definitive record of what was produced. This node is the equivalent of Phase 4's `package_output_node`. It does no processing — it only assembles the output contract from accumulated state.

#### Input
- `final_podcast_mp3_path` — from Step 5
- `cold_open_report` — from Step 4
- `post_processor_reports` — from Step 3
- `overlap_engine_reports` — from Step 2
- `phase5_validation_report` — from Step 1
- `stitcher_report` — from Step 5
- `degraded_chapter_numbers` — from Step 1
- `episode_id` — from pipeline state

#### Detailed process

**Sub-step 6.1 — Build the Phase 5 output dict:**

Construct a `phase5_output` dict containing:
- `phase5_contract_version`: `"1.0"`
- `episode_id`: from state
- `final_podcast_path`: absolute path to the MP3
- `total_duration_seconds`: from `stitcher_report`
- `file_size_bytes`: `os.path.getsize(final_podcast_mp3_path)` (re-read to confirm file is present)
- `chapter_count`: number of chapters assembled
- `degraded_chapters`: list of chapter numbers that were processed with partial clip failures
- `cold_open_included`: boolean
- `music_assets_used`: list of which assets were available and used
- `loudness_target_lufs`: `-16.0`
- `mp3_bitrate_kbps`: configured bitrate
- `chapter_markers`: from stitcher state
- `processing_reports`: nested dict containing all reports from Steps 1–5
- `ready`: boolean — `true` if `final_podcast_path` exists and `len(degraded_chapters) < total_chapters`

**Sub-step 6.2 — Log episode production summary:**

Print a human-readable production summary to the log. Example format:
```
========================================
PHASE 5 COMPLETE
========================================
Episode ID : {episode_id}
MP3 Path   : {final_podcast_path}
Duration   : {mm}:{ss}
File Size  : {X} MB
Chapters   : {N} ({M} degraded)
Cold Open  : Yes / No
Loudness   : -16 LUFS
Bitrate    : 128 kbps
========================================
```

**Sub-step 6.3 — Update pipeline-level state:**

Set the following keys in the LangGraph state for consumption by the main `orchestrator.py`:
- `final_podcast_mp3` — path string
- `phase5_output` — the full output dict
- `phase5_complete` — `True`

#### Output
- `phase5_output` — complete output contract dict
- `final_podcast_mp3` — path string
- `phase5_complete` — boolean

---

## 6. Phase 5 State Schema (LangGraph)

```python
class Phase5State(TypedDict, total=False):
    # Inherited from Phase 4
    phase4_output: Dict[str, Any]
    chapter_audio_manifests: List[Dict[str, Any]]
    timing_metadata: Dict[str, Any]
    audio_files: List[Dict[str, Any]]
    voice_metadata: Dict[str, Any]
    episode_id: str
    ready_for_phase5: bool

    # Inherited from earlier phases (needed by stitcher and cold open)
    topic: str
    character_personas: List[Dict[str, Any]]
    chapter_dialogues: List[Dict[str, Any]]

    # Step 1: Input validation
    validated_manifests: List[Dict[str, Any]]
    degraded_chapter_numbers: List[int]
    validated_timing_directives: List[Dict[str, Any]]
    input_audio_format: Dict[str, Any]
    phase5_output_paths: Dict[str, str]
    phase5_blocked: bool
    phase5_validation_report: Dict[str, Any]

    # Step 2: Overlap Engine
    chapter_mixed_audio_paths: Dict[int, str]
    utterance_timestamp_maps: Dict[int, Dict[str, Dict[str, int]]]
    overlap_engine_reports: List[Dict[str, Any]]

    # Step 3: Post-Processor
    chapter_mastered_audio_paths: Dict[int, str]
    post_processor_reports: List[Dict[str, Any]]

    # Step 4: Cold Open Generator
    cold_open_path: str
    cold_open_failed: bool
    cold_open_report: Dict[str, Any]

    # Step 5: Chapter Stitcher
    final_podcast_mp3_path: str
    chapter_markers: Dict[str, Any]
    stitcher_report: Dict[str, Any]

    # Step 6: Output Packaging
    phase5_output: Dict[str, Any]
    final_podcast_mp3: str
    phase5_complete: bool
```

---

## 7. Proposed Graph Topology

```
validate_phase5_input
        │
        ├── [phase5_blocked == True] ──────────────────────── package_phase5_output ── END
        │
        ▼
run_overlap_engine
        │
        ▼
run_post_processor
        │
        ▼
generate_cold_open      ← best-effort; never blocks graph on failure
        │
        ▼
run_chapter_stitcher
        │
        ▼
package_phase5_output
        │
        ▼
       END
```

**Conditional edge:** After `validate_phase5_input`, route to `package_phase5_output` immediately if `phase5_blocked == True`. This is the same fail-fast pattern used in Phase 4.

There are no retry loops in Phase 5. All operations are deterministic DSP operations or best-effort. The only exception is backchannel TTS generation in the Overlap Engine — if a TTS call fails for a backchannel, fall back to the pre-recorded clip strategy silently.

---

## 8. New Files to Create

The following files must be created from scratch (all currently exist as empty stubs):

| File | Purpose |
|------|---------|
| `src/agents/phase5/overlap_engine.py` | Audio Overlap Engine logic |
| `src/agents/phase5/post_processor.py` | Audio Post-Processor (mastering chain) |
| `src/agents/phase5/cold_open_generator.py` | Cold Open Generator (LLM scan + audio slice) |
| `src/agents/phase5/chapter_stitcher.py` | Chapter Stitcher (final assembly + MP3 export) |
| `src/models/phase5.py` | Pydantic models for Phase 5 contracts |
| `src/pipeline/phases/phase5_graph.py` | LangGraph graph (replace placeholder) |

The following existing files must be extended (not replaced):

| File | Change Required |
|------|----------------|
| `src/tools/audio_tools.py` | Add `convert_audio_format()`, `apply_crossfade()`, `compute_lufs()` helper functions |
| `config/settings.py` | Add Phase 5 config section: mastering parameters, asset paths, MP3 bitrate |
| `requirements.txt` | Add `mutagen` for ID3 tagging (verify `pydub` and `ffmpeg-python` are already present) |

Audio asset files to provision (royalty-free sources required — document source in a `data/audio/assets/README.md`):

| Asset Path | Description |
|------------|-------------|
| `data/audio/assets/music/intro_music.wav` | 8–10 second intro jingle |
| `data/audio/assets/music/outro_music.wav` | 8–10 second outro music |
| `data/audio/assets/transitions/chapter_sting.wav` | 0.5–1.5 second chapter transition sting |
| `data/audio/assets/transitions/cold_open_end.wav` | 1–2 second cold open end transition |
| `data/audio/assets/room_tone/room_tone_default.wav` | 30–60 second loopable subtle room tone |
| `data/audio/assets/backchannels/mmhm_neutral.wav` | 0.8s "mm-hm" backchannel clip |
| `data/audio/assets/backchannels/yeah_neutral.wav` | 0.5s "yeah" backchannel clip |
| `data/audio/assets/laughs/laugh_light.wav` | 1s light chuckle clip |
| `data/audio/assets/laughs/laugh_medium.wav` | 1.5s medium laugh clip |

---

## 9. New Pydantic Models to Define (`src/models/phase5.py`)

The following models should be defined to provide schema validation at the Phase 5 boundary:

**`OverlapOperation`** — Represents a single timing directive resolved to a concrete audio operation:
- `op_type`: `Literal["interrupt", "backchannel", "laugh"]`
- `chapter_number`: int
- `target_utterance_id`: str
- `trigger_utterance_id`: str (the utterance being overlapped or reacted to)
- `speaker`: str
- `duration_ms`: Optional[int] (for interrupts)
- `insertion_offset_ms`: Optional[int] (for backchannels)

**`UtteranceTimestamp`** — An entry in the timestamp map:
- `utterance_id`: str
- `speaker`: str
- `start_ms`: int
- `end_ms`: int
- `chapter_number`: int

**`ChapterMixReport`** — Per-chapter Overlap Engine output:
- `chapter_number`: int
- `input_clip_count`: int
- `interrupts_applied`: int
- `backchannels_applied`: int
- `laughs_applied`: int
- `total_duration_ms`: int
- `output_path`: str

**`MasteringReport`** — Per-chapter Post-Processor output:
- `chapter_number`: int
- `steps_applied`: List[str] (e.g., `["noise_gate", "eq", "compression", "loudnorm", "room_tone"]`)
- `input_lufs`: float
- `output_lufs`: float
- `input_duration_ms`: int
- `output_duration_ms`: int
- `output_path`: str

**`ColdOpenReport`** — Cold Open Generator output:
- `selected_chapter_number`: int
- `start_utterance_id`: str
- `end_utterance_id`: str
- `duration_ms`: int
- `framing_strategy`: `Literal["tts", "pre_recorded"]`
- `llm_used`: bool
- `fallback_used`: bool

**`Phase5Output`** — Final output contract:
- `phase5_contract_version`: str
- `episode_id`: str
- `final_podcast_path`: str
- `total_duration_seconds`: float
- `file_size_bytes`: int
- `chapter_count`: int
- `degraded_chapters`: List[int]
- `cold_open_included`: bool
- `loudness_target_lufs`: float
- `mp3_bitrate_kbps`: int
- `chapter_markers`: Dict[str, Any]
- `ready`: bool

---

## 10. Config Keys to Add (`config/settings.py`)

Add a `Phase5Config` section with the following keys and defaults:

```python
# Audio format pipeline standard
PHASE5_TARGET_SAMPLE_RATE = 44100       # Hz
PHASE5_TARGET_CHANNELS = 2             # stereo
PHASE5_TARGET_SAMPLE_WIDTH = 2         # 16-bit PCM

# Overlap Engine
PHASE5_TURN_GAP_MS = 300               # gap between sequential turns
PHASE5_CROSSFADE_MS = 75               # cross-fade between speaker turns
PHASE5_INTERRUPT_VOLUME_REDUCTION_DB = -3   # attenuate interrupted speaker tail
PHASE5_BACKCHANNEL_VOLUME_DB = -8      # backchannel mixed under main speaker
PHASE5_LAUGH_VOLUME_DB = -4            # laughter mixed level

# Post-Processor (EQ)
PHASE5_EQ_PRESENCE_FREQ = 3000         # Hz center for presence boost
PHASE5_EQ_PRESENCE_GAIN = 2            # dB
PHASE5_EQ_RUMBLE_FREQ = 80             # Hz center for rumble cut
PHASE5_EQ_RUMBLE_GAIN = -6             # dB

# Post-Processor (Compression)
PHASE5_COMP_THRESHOLD_DB = -20
PHASE5_COMP_RATIO = 2
PHASE5_COMP_ATTACK_MS = 5
PHASE5_COMP_RELEASE_MS = 50
PHASE5_COMP_MAKEUP_GAIN_DB = 2

# Post-Processor (Loudness)
PHASE5_LOUDNESS_TARGET_LUFS = -16.0
PHASE5_LOUDNESS_TRUE_PEAK_DB = -1.5
PHASE5_NOISE_GATE_THRESHOLD_DB = -40
PHASE5_NOISE_GATE_SILENCE_DURATION = 0.1   # seconds

# Post-Processor (Room Tone)
PHASE5_ENABLE_ROOM_TONE = True
PHASE5_ROOM_TONE_LEVEL_DB = -32

# Cold Open
PHASE5_COLD_OPEN_MIN_MS = 12000
PHASE5_COLD_OPEN_MAX_MS = 25000
PHASE5_COLD_OPEN_LLM_MODEL = "claude-haiku-4-5-20251001"

# Chapter Stitcher
PHASE5_INTRO_MUSIC_DURATION_MS = 8000
PHASE5_COLD_OPEN_INTRO_CROSSFADE_MS = 3000
PHASE5_MP3_BITRATE_KBPS = 128          # accepted: 96, 128, 192

# Output
PHASE5_OUTPUT_BASE_DIR = "data/audio/phase5"
```

---

## 11. Implementation Checklist (Practical Order)

This checklist defines the exact order in which a developer should implement Phase 5. Each item is independently completable and testable before moving to the next.

- [ ] **0. Environment setup:** Verify `pydub`, `ffmpeg` (CLI), and `mutagen` are installed and importable. Add missing packages to `requirements.txt`. Verify `ffmpeg` CLI is on PATH by running `ffmpeg -version` in a test.

- [ ] **1. Add Phase 5 config keys** to `config/settings.py` as described in Section 10.

- [ ] **2. Create `src/models/phase5.py`** with all Pydantic models described in Section 9.

- [ ] **3. Extend `src/tools/audio_tools.py`** with three new helpers:
  - `convert_to_pipeline_format(segment: AudioSegment) -> AudioSegment` — applies sample rate, channel, and bit depth normalization.
  - `apply_crossfade(seg_a: AudioSegment, seg_b: AudioSegment, fade_ms: int) -> AudioSegment` — returns the crossfaded junction.
  - `get_file_duration_ms(path: str) -> int` — opens a WAV and returns duration in milliseconds.

- [ ] **4. Provision audio assets** (`data/audio/assets/` subtree). Document the source URL and license of each asset in `data/audio/assets/README.md`. The `room_tone_default.wav`, `intro_music.wav`, and `outro_music.wav` are the most impactful and should be provisioned first.

- [ ] **5. Implement `src/agents/phase5/overlap_engine.py`:**
  - Implement `build_sequential_timeline()`.
  - Implement `apply_interrupt_ops()`.
  - Implement `generate_backchannel_clip()`.
  - Implement `apply_backchannel_ops()`.
  - Implement `apply_laugh_ops()`.
  - Implement `apply_all_crossfades()`.
  - Implement the top-level `run_overlap_engine(manifest, timing_directives, config, output_dir) -> (path, timestamp_map, report)`.
  - Write unit tests for each sub-function with small synthetic `AudioSegment` fixtures.

- [ ] **6. Implement `src/agents/phase5/post_processor.py`:**
  - Implement `run_noise_gate()`.
  - Implement `run_equalisation()`.
  - Implement `run_compression()`.
  - Implement `run_loudness_normalisation()` (two-pass ffmpeg).
  - Implement `apply_room_tone()`.
  - Implement the top-level `run_mastering_chain(input_path, output_path, config) -> MasteringReport`.
  - Write integration test using a real WAV file (can be a 5-second synthetic tone) to verify each ffmpeg filter applies without error.

- [ ] **7. Implement `src/agents/phase5/cold_open_generator.py`:**
  - Implement `scan_script_for_candidates()` (LLM call).
  - Implement `validate_candidate()`.
  - Implement `apply_fallback_candidate()`.
  - Implement `generate_framing_line()` (TTS or pre-recorded).
  - Implement `extract_audio_excerpt()`.
  - Implement the top-level `generate_cold_open(script, mastered_paths, timestamp_maps, config) -> (path, report)`.
  - Write unit test with mock LLM response for candidate parsing logic.

- [ ] **8. Implement `src/agents/phase5/chapter_stitcher.py`:**
  - Implement `load_and_validate_assets()`.
  - Implement `prepare_host_intro_clip()`.
  - Implement `prepare_host_outro_clip()`.
  - Implement `build_intro_crossfade()`.
  - Implement `assemble_episode_timeline()`.
  - Implement `run_final_loudness_pass()`.
  - Implement `encode_to_mp3()`.
  - Implement `embed_id3_tags()`.
  - Implement `validate_final_mp3()`.
  - Implement the top-level `run_chapter_stitcher(inputs, config) -> (path, markers, report)`.
  - Write integration test assembling a 3-chapter episode from small synthetic WAV files.

- [ ] **9. Replace placeholder `src/pipeline/phases/phase5_graph.py`** with the full 6-node graph as described in Section 7. Each node function wraps the agent call and updates state.

- [ ] **10. Wire Phase 5 into the main orchestrator** (`src/pipeline/orchestrator.py` or `src/pipeline/graph.py`). Phase 5 receives state from Phase 4 — verify the key names match exactly (`chapter_audio_manifests`, `timing_metadata`, `audio_files`, `voice_metadata`, `episode_id`).

- [ ] **11. Run end-to-end integration test** using `tests/test_graph_with_cached.py` or a new `tests/integration/test_phase5_graph.py` test. Use a small 2-chapter episode with pre-cached Phase 4 outputs to verify the full Phase 5 run produces a valid MP3.

- [ ] **12. Update `ARCHITECTURE.md`** to document Phase 5 components.

---

## 12. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| pydub format mismatch during `overlay()` or `append()` — silent corruption when sample rates differ | High | High | Sub-step 2.1 normalizes all clips to pipeline standard before any mixing begins. Unit tests must verify format of every intermediate output. |
| ffmpeg not on PATH in the runtime environment | Medium | High | Step 6.0 of the checklist: verify ffmpeg at node startup. Emit a clear `RuntimeError` with install instructions if absent. |
| Phase 4 WAV files missing or corrupted | Medium | High | Step 1 validates physical presence of all clips before processing. Degraded-chapter path allows partial episodes. |
| LLM response for Cold Open contains invalid utterance IDs | Medium | Low | Sub-step 4.3 validates all IDs against the timestamp map and falls back to a deterministic beat-3 utterance selection. |
| Memory overflow from 300MB AudioSegment for 30-minute episode | Low | High | Process chapters sequentially. Do not load all mastered chapters into memory simultaneously in the Stitcher — load, append, then delete reference to free memory after each chapter. |
| Backchannel TTS synthesis adds latency during Overlap Engine | Low | Medium | Cache backchannel clips by `(speaker, text)` key. In practice only 3–5 unique backchannel texts are used. Pre-render them all before the per-chapter loop. |
| ffmpeg `loudnorm` two-pass fails on very short chapters | Low | Low | Add a minimum duration guard before loudnorm: if chapter duration < 5 seconds, skip loudnorm and log a warning. |
| Music assets are missing at runtime | Low | Low | All music/sting assets are optional. The Stitcher degrades gracefully to silence where assets are unavailable. |

---

## 13. Definition of Done for Phase 5

Phase 5 is complete when ALL of the following are true:

1. `podcast_episode_final.mp3` exists at `data/audio/phase5/{episode_id}/final/podcast_episode_final.mp3`.
2. The MP3 is decodable and its duration is between 10 and 40 minutes.
3. The MP3's loudness is within ±1 LUFS of the -16 LUFS target (measured with a loudness meter tool such as `ffmpeg`'s `ebur128` filter or the `pyloudnorm` library).
4. ID3 title and episode metadata tags are present and non-empty.
5. The `phase5_output` contract has `ready == True`.
6. The chapter manifest list in the output contains all chapters that had complete Phase 4 manifests.
7. The Overlap Engine has produced timestamp maps for all non-degraded chapters.
8. No unhandled exceptions were raised during the graph run — any partial failures are captured in the output contract's processing reports.
9. At least one unit test covers each of the four agent modules.
10. At least one integration test produces a valid MP3 from a cached Phase 4 fixture.

---

## 14. Notes for Transition to Implementation

Before writing any code, lock the following decisions:

1. **Backchannel strategy (TTS vs pre-recorded):** The TTS strategy produces better voice-consistency but adds API calls and latency. For an initial implementation, use pre-recorded clips (Strategy B) to avoid complexity and cost. Upgrade to Strategy A in a later iteration.

2. **Music assets:** Obtain royalty-free assets before beginning stitcher development. Without them, the stitcher tests run in silence-fallback mode, which is valid but less representative of the final product.

3. **Ambient room tone:** This is Improvement #4 from the design doc. It is safe to defer it to a post-launch iteration if audio quality is acceptable without it. Set `PHASE5_ENABLE_ROOM_TONE = False` initially to skip this step.

4. **Host intro/outro synthesis:** This adds TTS calls inside the Stitcher. For the first implementation, generate the host intro and outro as a one-time offline pre-render and bundle them as static files in `data/audio/assets/`. This removes the TTS dependency from the Stitcher node entirely.

5. **ffmpeg availability:** Ensure that `ffmpeg` is documented as a system dependency in `INSTALL.md` with clear installation instructions for macOS (via Homebrew: `brew install ffmpeg`) and Linux (via apt: `apt-get install ffmpeg`).
