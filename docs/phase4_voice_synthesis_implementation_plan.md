# Phase 4: Voice Synthesis - Detailed Implementation Plan (Gemini 2.5 Pro TTS)

**Version:** 1.0  
**Date:** March 17, 2026  
**Primary TTS Engine:** `google/gemini-2.5-pro-tts`  
**Architecture Target:** LangGraph subgraph (`phase4_graph.py`)  
**Authoring Context:** Built from `design_docs/design_doc.pdf` + current repository state

---

## 1. Executive Summary

Phase 4 converts Phase 3 SSML-annotated chapter scripts into per-utterance WAV files that can be consumed by Phase 5 (overlap/mixing/mastering). This phase must be:

- Deterministic in file naming, ordering, and metadata handoff.
- Reliable under API failures, rate limits, and transient provider errors.
- Optimized for Gemini 2.5 Pro TTS constraints and payload behavior.
- Strict about preserving conversational intent markers (interruptions, backchannels, pacing).

This document defines a complete **8-step plan** that can be implemented directly as LangGraph nodes with clear state transitions and quality gates.

---

## 2. Why This Phase Exists

Phase 3 produces text + SSML + conversational metadata. That is not yet "audio production material." Phase 4 exists to perform the following critical transformations:

- Convert each utterance into an individual audio unit for precise overlap control in Phase 5.
- Preserve speaker identity across the entire episode with stable voice assignment.
- Maintain marker-derived timing intent as machine-readable metadata.
- Reduce downstream failures by validating and repairing audio before Phase 5.
- Create an idempotent contract so retries and resumed runs are safe.

Without this phase discipline, the pipeline would generate inconsistent voices, missing utterances, fragile handoffs, and expensive reruns.

---

## 3. Total Information Flow (Bullet View)

- Phase 3 emits chapter scripts with SSML + speaker metadata + naturalness marker metadata.
- Phase 4 validates that contract before any TTS calls.
- Speaker-to-voice mapping is resolved with explicit fallback policy.
- Scripts are flattened into deterministic utterance-level synthesis jobs.
- Jobs are routed to Gemini-compatible request payloads.
- Synthesis runs in controlled parallel mode with retries and fallback behavior.
- Generated clips pass through technical QC and selective auto-repair.
- Final chapter manifests plus clip registry are packaged for Phase 5.

Flow string:

`SSML Scripts -> Contract Validation -> Voice Resolution -> TTS Job Queue -> Gemini Routing -> Parallel Synthesis -> Audio QC/Repair -> Chapter Manifest + Phase 5 Handoff`

---

## 4. Gemini 2.5 Pro TTS Optimization Principles

These are cross-cutting implementation rules used in every Phase 4 step.

### 4.1 Provider-first assumptions

- Gemini is treated as **primary** engine.
- Payload normalization should target Gemini syntax and behavior first.
- Any fallback provider policy should preserve voice/persona continuity as much as possible.

### 4.2 Utterance sizing and chunk strategy

- Keep utterances short-to-medium (podcast turn-sized) to preserve natural pacing and reduce timeout risk.
- If an utterance exceeds engine-safe length thresholds, split using punctuation-aware boundaries while preserving semantic intent.
- Preserve original `utterance_id` lineage via sub IDs (`ch3_u014_a`, `ch3_u014_b`) for traceability.

### 4.3 Concurrency and quota safety

- Use bounded parallelism; do not flood TTS API.
- Implement provider-specific QPS and token/character guards.
- Use exponential backoff on transient failures (`429`, `503`, timeout).

### 4.4 Voice consistency

- Voice identity must be fixed per speaker for entire episode.
- Any fallback voice must be pre-mapped by similarity profile and logged.

### 4.5 Metadata preservation

- `INTERRUPT` and `BACKCHANNEL` are not "rendered away"; they are forwarded as timing directives for Phase 5.
- All transformations must keep linkage from generated clip back to source utterance.

---

## 5. Phase 4 Node Plan (8 Steps)

## Step 1: Input Contract Validation Node

### Objective
Guarantee that Phase 4 receives complete, parseable, and internally consistent Phase 3 output before making any external API calls.

### Why we implement this
Most expensive failures are preventable by validating schema and SSML early. This node prevents noisy failures and protects cost.

### Input
- `ssml_annotated_scripts` (chapter-level)
- Speaker identifiers and persona voice hints
- Naturalness marker metadata

### How to achieve objective
- Validate required fields on every utterance: `chapter`, `utterance_id`, `speaker`, `text_ssml`.
- Validate strict ordering and uniqueness:
  - no duplicate `utterance_id`
  - no missing chapter sequence positions
- Validate SSML format:
  - parseability
  - disallowed tag handling
  - nesting correctness
- Validate marker consistency:
  - `INTERRUPT` references target valid adjacent turns
  - `BACKCHANNEL` has owning speaker and timing hook

### Output
- `validated_ssml_scripts`
- `validation_report` with errors/warnings/stats
- `phase4_blocked` boolean when hard failures exist

### Implementation guidance
- Use Pydantic models for early schema strictness.
- Implement `normalize_empty_fields()` to convert null/empty edge cases into consistent defaults.
- Fail fast on hard contract errors; do not proceed to synthesis.

---

## Step 2: Voice Assignment and Provider Policy Node

### Objective
Resolve and freeze per-speaker voice configuration and fallback route for the entire episode.

### Why we implement this
Voice drift is one of the fastest ways to break listener immersion. This node makes voice selection deterministic and auditable.

### Input
- `validated_ssml_scripts`
- Character/persona `tts_voice_id` data
- Global TTS policy config

### How to achieve objective
- Build canonical `speaker_voice_map`:
  - `speaker -> primary_provider(gemini) -> primary_voice_id`
  - optional fallback provider/voice chain
- Enforce one-speaker-one-voice lock for episode scope.
- Validate configured voices exist in allowed list.
- Attach voice profile metadata for observability.

### Output
- `speaker_voice_map`
- `provider_fallback_policy`
- `voice_resolution_report`

### Implementation guidance
- Keep policy in config file, not hardcoded.
- Add an explicit `voice_lock_signature` hash to detect accidental mapping changes mid-run.

---

## Step 3: Utterance Normalization and Synthesis Job Planner Node

### Objective
Convert validated scripts into deterministic utterance-level jobs that can be run, retried, and audited independently.

### Why we implement this
Phase 5 depends on fine-grained timing control; that is only possible when every turn is a stable independent audio artifact.

### Input
- `validated_ssml_scripts`
- `speaker_voice_map`

### How to achieve objective
- Flatten chapters into `tts_jobs[]`.
- Assign deterministic IDs:
  - `job_id`
  - `chapter_id`
  - `utterance_id`
- Normalize payload text:
  - trim unsafe whitespace
  - sanitize unsupported SSML patterns
  - optionally split very long utterances
- Attach retry metadata and output path blueprint.

### Output
- `tts_jobs[]`
- `job_lookup_maps`
- `planned_output_paths`

### Implementation guidance
- Use stable file path pattern:
  - `data/audio/raw/{episode_id}/ch_{chapter}/utt_{utterance}.wav`
- Keep job planner pure (no API calls) so it is testable with fixtures.

---

## Step 4: Gemini Routing and Request Construction Node

### Objective
Build Gemini-optimized request payloads and route every job through provider-compatible synthesis settings.

### Why we implement this
Even if a job is valid at logical level, provider payload mismatches will fail at runtime. This node isolates provider-specific compatibility work.

### Input
- `tts_jobs[]`
- `provider_fallback_policy`
- Gemini-specific request templates

### How to achieve objective
- Convert each job into provider-ready payload format.
- Apply Gemini constraints:
  - text length guard
  - supported markup transformation
  - voice parameter injection
- Pre-flag risky jobs (too long, unusual symbols, repeated pauses).
- Route incompatible jobs to fallback policy where needed.

### Output
- `routed_tts_jobs[]`
- `routing_decisions`
- `payload_validation_report`

### Implementation guidance
- Implement a `build_gemini_tts_payload(job)` function in a reusable tool module.
- Store final payload snapshot for debugging failed jobs.

---

## Step 5: Parallel Synthesis Executor Node

### Objective
Execute TTS generation for all jobs with high throughput and controlled reliability.

### Why we implement this
This node performs the largest number of external calls in Phase 4. Without robust execution control, completion rates and latency become unstable.

### Input
- `routed_tts_jobs[]`

### How to achieve objective
- Run jobs in bounded async pool.
- Respect global + provider concurrency caps.
- On success:
  - write WAV bytes to deterministic path
  - store technical metadata (duration, sample rate, channels)
- On transient failure:
  - retry with exponential backoff and jitter
- On hard failure:
  - apply configured fallback provider route (if available)

### Output
- `raw_audio_clips[]`
- `failed_jobs[]`
- `synthesis_log`

### Implementation guidance
- Persist per-job execution timeline for observability.
- Make file write atomic (`tmp -> final rename`) to avoid partial clip corruption.

---

## Step 6: Audio Quality Gate and Auto-Repair Node

### Objective
Detect unusable or degraded clips and recover them without rerunning the full phase.

### Why we implement this
Provider success response does not guarantee production-quality audio. Early QC protects Phase 5 from silent or broken inputs.

### Input
- `raw_audio_clips[]`
- `failed_jobs[]`
- `synthesis_log`

### How to achieve objective
- Validate each clip:
  - decodable WAV
  - duration above minimum
  - non-silent amplitude profile
  - clipping threshold check
  - sample-rate/channel consistency
- Auto-repair strategy:
  - targeted re-synthesize with modified request
  - fallback route for repeat failures
  - hard-stop after max attempts

### Output
- `qc_passed_audio_clips[]`
- `qc_failed_jobs[]`
- `qc_report`

### Implementation guidance
- Implement QC thresholds in config, not code constants.
- Keep repair actions idempotent with incremented attempt metadata.

---

## Step 7: Chapter Manifest and Timing Directive Node

### Objective
Build chapter-level manifests that combine ordered clips with conversational timing directives for overlap/post-processing.

### Why we implement this
Phase 5 needs semantic timing intent, not just raw files. This node materializes that intent into an executable manifest format.

### Input
- `qc_passed_audio_clips[]`
- marker metadata from validated script
- job lookup maps

### How to achieve objective
- Group clips by chapter and sort by utterance order.
- Attach timing directives:
  - baseline turn gaps
  - interrupt overlap offsets
  - backchannel insertion points
- Validate manifest completeness:
  - required utterances present
  - marker references resolve to actual clips

### Output
- `chapter_audio_manifests[]`
- `timing_metadata`
- `manifest_integrity_report`

### Implementation guidance
- Keep manifests JSON-serializable and versioned (`manifest_version`).
- Add checksum of clip list per chapter for reproducibility checks.

---

## Step 8: Final Packaging and Phase 5 Handoff Node

### Objective
Emit a strict, complete Phase 4 output contract that Phase 5 can consume directly.

### Why we implement this
A clean handoff boundary prevents brittle integration logic and simplifies retries, checkpoint recovery, and debugging.

### Input
- `chapter_audio_manifests[]`
- `timing_metadata`
- reports from validation/routing/synthesis/QC

### How to achieve objective
- Build `phase4_output` object with:
  - `audio_files`
  - `chapter_manifests`
  - `voice_metadata`
  - `timing_metadata`
  - `quality_reports`
- Determine readiness:
  - set `ready_for_phase5 = true` only if completeness threshold is met
- Include unresolved failures with enough data for resumption.

### Output
- `phase4_output` (official handoff contract)
- `phase4_summary_metrics`

### Implementation guidance
- Add explicit schema version field (`phase4_contract_version`).
- Write a `validate_phase4_output_contract()` guard before returning graph END.

---

## 6. Proposed Phase 4 State Schema (LangGraph)

```python
class Phase4State(TypedDict):
    # Input from Phase 3
    ssml_annotated_scripts: List[Dict[str, Any]]
    episode_id: str

    # Step outputs
    validated_ssml_scripts: List[Dict[str, Any]]
    validation_report: Dict[str, Any]

    speaker_voice_map: Dict[str, Any]
    provider_fallback_policy: Dict[str, Any]

    tts_jobs: List[Dict[str, Any]]
    routed_tts_jobs: List[Dict[str, Any]]

    raw_audio_clips: List[Dict[str, Any]]
    failed_jobs: List[Dict[str, Any]]
    synthesis_log: List[Dict[str, Any]]

    qc_passed_audio_clips: List[Dict[str, Any]]
    qc_failed_jobs: List[Dict[str, Any]]
    qc_report: Dict[str, Any]

    chapter_audio_manifests: List[Dict[str, Any]]
    timing_metadata: Dict[str, Any]

    phase4_output: Dict[str, Any]
    ready_for_phase5: bool

    # Operational controls
    retry_count: int
    max_retries: int
```

---

## 7. Suggested Graph Topology

- `node_1_validate_input`
- `node_2_resolve_voice_policy`
- `node_3_plan_tts_jobs`
- `node_4_route_gemini_payloads`
- `node_5_execute_parallel_synthesis`
- `node_6_audio_qc_and_repair`
- `node_7_build_chapter_manifests`
- `node_8_package_phase4_output`

Conditional edges:

- After Step 1: if blocking validation errors -> fail fast.
- After Step 5: if high failure ratio -> retry route or abort with detailed report.
- After Step 6: if unrecoverable critical clips -> mark `ready_for_phase5=false`.
- After Step 8: only end when output contract passes schema validation.

---

## 8. Implementation Checklist (Practical)

- Create/complete `src/agents/phase4/tts_router.py` with provider abstraction.
- Replace placeholder graph in `src/pipeline/phases/phase4_graph.py` with 8 real nodes.
- Add Pydantic models for Phase 4 contracts under `src/models/`.
- Implement audio technical checks in `src/tools/audio_tools.py`.
- Integrate Gemini TTS invocation utility in `src/tools/` (reuse existing Gemini helper module naming where applicable).
- Add unit tests:
  - contract validation
  - routing compatibility
  - retry/fallback behavior
  - manifest integrity checks
- Add integration test from mock Phase 3 output -> Phase 4 output contract.

---

## 9. Risks and Mitigations

- Risk: SSML incompatibility with Gemini payload shape.
  - Mitigation: dedicated payload builder + compatibility transform layer.
- Risk: API quota/rate limit spikes during parallel synthesis.
  - Mitigation: adaptive concurrency and exponential backoff with jitter.
- Risk: Voice inconsistency after fallback.
  - Mitigation: pre-approved similarity-matched fallback voice mapping.
- Risk: Silent or clipped audio passing as "success."
  - Mitigation: strict QC gate with selective auto-repair loop.

---

## 10. Definition of Done for Phase 4

Phase 4 is complete when:

- All valid utterances produce recoverable or finalized WAV outcomes.
- Voice identity remains consistent per speaker through episode.
- QC pass rate satisfies configured threshold.
- Chapter manifests are complete and timing directives are preserved.
- `phase4_output` validates against contract schema and `ready_for_phase5=true`.

---

## 11. Notes for Brainstorm-to-Implementation Transition

Before coding starts, lock these decisions:

- Gemini payload format and SSML compatibility policy.
- Voice catalog and fallback mapping strategy.
- Concurrency/retry thresholds based on expected episode size.
- Minimum Phase 5 readiness threshold when partial failures exist.

Once these are fixed, implementation can proceed node-by-node with low rework risk.
