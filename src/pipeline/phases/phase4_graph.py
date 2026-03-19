"""Phase 4: Voice Synthesis subgraph."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.phase4.tts_router import (
    audio_qc_and_repair,
    build_chapter_manifests,
    ensure_episode_id,
    execute_parallel_synthesis,
    package_phase4_output,
    plan_tts_jobs,
    resolve_voice_policy,
    route_tts_jobs,
    translate_tts_jobs,
    validate_phase4_input_contract,
)


class Phase4State(TypedDict, total=False):
    """State for Phase 4: Voice Synthesis."""

    topic: str
    episode_id: str
    character_personas: List[Dict[str, Any]]
    chapter_dialogues: List[Dict[str, Any]]
    ssml_annotated_scripts: List[Dict[str, Any]]

    validated_ssml_scripts: List[Dict[str, Any]]
    validation_report: Dict[str, Any]
    phase4_blocked: bool

    speaker_voice_map: Dict[str, Any]
    provider_fallback_policy: Dict[str, Any]
    voice_resolution_report: Dict[str, Any]

    tts_jobs: List[Dict[str, Any]]
    translated_tts_jobs: List[Dict[str, Any]]
    translation_report: Dict[str, Any]
    job_lookup_maps: Dict[str, Any]
    planned_output_paths: Dict[str, str]

    routed_tts_jobs: List[Dict[str, Any]]
    routing_decisions: List[Dict[str, Any]]
    payload_validation_report: Dict[str, Any]

    raw_audio_clips: List[Dict[str, Any]]
    failed_jobs: List[Dict[str, Any]]
    synthesis_log: List[Dict[str, Any]]

    qc_passed_audio_clips: List[Dict[str, Any]]
    qc_failed_jobs: List[Dict[str, Any]]
    qc_report: Dict[str, Any]

    chapter_audio_manifests: List[Dict[str, Any]]
    timing_metadata: Dict[str, Any]
    manifest_integrity_report: Dict[str, Any]

    phase4_output: Dict[str, Any]
    phase4_summary_metrics: Dict[str, Any]
    ready_for_phase5: bool
    audio_files: List[Dict[str, Any]]


def _record_error(state: Phase4State, message: str) -> None:
    report = dict(state.get("validation_report") or {})
    errors = list(report.get("errors") or [])
    errors.append(message)
    report["errors"] = errors
    state["validation_report"] = report
    state["phase4_blocked"] = True


def validate_input_node(state: Phase4State) -> Phase4State:
    """Validate Phase 3 output before any TTS work begins."""

    print("\n🎙️ PHASE 4 - INPUT CONTRACT VALIDATION")
    validated_scripts, validation_report, blocked = validate_phase4_input_contract(state)
    state["ssml_annotated_scripts"] = state.get("ssml_annotated_scripts") or state.get("chapter_dialogues") or []
    state["validated_ssml_scripts"] = validated_scripts
    state["validation_report"] = validation_report
    state["phase4_blocked"] = blocked
    state["episode_id"] = ensure_episode_id(state, validated_scripts)

    stats = validation_report.get("stats", {})
    print(
        f"   Chapters: {stats.get('chapters', 0)} | "
        f"Utterances: {stats.get('utterances', 0)}"
    )
    if blocked:
        print(f"   ❌ Blocked: {len(validation_report.get('errors', []))} error(s)")
    else:
        print("   ✅ Input contract validated")
    return state


def resolve_voice_policy_node(state: Phase4State) -> Phase4State:
    """Resolve per-speaker provider and voice policy."""

    if state.get("phase4_blocked"):
        return state

    print("\n🗣️ VOICE ASSIGNMENT")
    try:
        speaker_voice_map, provider_policy, report = resolve_voice_policy(
            state.get("validated_ssml_scripts") or [],
            state.get("character_personas") or [],
        )
    except Exception as exc:
        _record_error(state, f"Voice resolution failed: {exc}")
        return state

    state["speaker_voice_map"] = speaker_voice_map
    state["provider_fallback_policy"] = provider_policy
    state["voice_resolution_report"] = report
    print(f"   ✅ Locked voices for {len(speaker_voice_map)} speaker(s)")
    return state


def plan_tts_jobs_node(state: Phase4State) -> Phase4State:
    """Flatten chapter scripts into stable synthesis jobs."""

    if state.get("phase4_blocked"):
        return state

    print("\n🧾 TTS JOB PLANNING")
    jobs, job_lookup_maps, planned_output_paths = plan_tts_jobs(
        state.get("validated_ssml_scripts") or [],
        state.get("speaker_voice_map") or {},
        state["episode_id"],
    )
    state["tts_jobs"] = jobs
    state["job_lookup_maps"] = job_lookup_maps
    state["planned_output_paths"] = planned_output_paths
    print(f"   ✅ Planned {len(jobs)} synthesis job(s)")
    return state


def route_jobs_node(state: Phase4State) -> Phase4State:
    """Build provider-ready payloads for each job."""

    if state.get("phase4_blocked"):
        return state

    print("\n🧭 PROVIDER ROUTING")
    jobs_for_routing = state.get("translated_tts_jobs") or state.get("tts_jobs") or []
    routed_jobs, routing_decisions, payload_report = route_tts_jobs(jobs_for_routing)
    state["routed_tts_jobs"] = routed_jobs
    state["routing_decisions"] = routing_decisions
    state["payload_validation_report"] = payload_report
    print(f"   ✅ Routed {len(routed_jobs)} job(s)")
    return state


def translate_jobs_node(state: Phase4State) -> Phase4State:
    """Translate text jobs when multilingual Sarvam mode is active."""

    if state.get("phase4_blocked"):
        return state

    print("\n🌐 TEXT TRANSLATION")
    translated_jobs, translation_report = translate_tts_jobs(state.get("tts_jobs") or [])
    state["translated_tts_jobs"] = translated_jobs
    state["translation_report"] = translation_report
    print(
        f"   ✅ Enabled: {translation_report.get('enabled', False)} | "
        f"Translated: {translation_report.get('translated_jobs', 0)}"
    )
    return state


def execute_synthesis_node(state: Phase4State) -> Phase4State:
    """Execute TTS calls with retries and optional fallback."""

    if state.get("phase4_blocked"):
        return state

    print("\n⚙️ SYNTHESIS EXECUTION")
    raw_audio_clips, failed_jobs, synthesis_log = execute_parallel_synthesis(
        state.get("routed_tts_jobs") or []
    )
    state["raw_audio_clips"] = raw_audio_clips
    state["failed_jobs"] = failed_jobs
    state["synthesis_log"] = synthesis_log
    print(
        f"   ✅ Clips: {len(raw_audio_clips)} | "
        f"Failures: {len(failed_jobs)}"
    )
    return state


def qc_and_repair_node(state: Phase4State) -> Phase4State:
    """Validate generated WAVs and attempt one targeted repair when needed."""

    if state.get("phase4_blocked"):
        return state

    print("\n🩺 AUDIO QC")
    qc_passed_audio_clips, qc_failed_jobs, qc_report = audio_qc_and_repair(
        state.get("routed_tts_jobs") or [],
        state.get("raw_audio_clips") or [],
        state.get("failed_jobs") or [],
        state.get("synthesis_log") or [],
    )
    state["qc_passed_audio_clips"] = qc_passed_audio_clips
    state["qc_failed_jobs"] = qc_failed_jobs
    state["qc_report"] = qc_report
    print(
        f"   ✅ Passed: {len(qc_passed_audio_clips)} | "
        f"Failed: {len(qc_failed_jobs)}"
    )
    return state


def build_manifests_node(state: Phase4State) -> Phase4State:
    """Package ordered clips and timing directives for Phase 5."""

    if state.get("phase4_blocked"):
        return state

    print("\n📚 CHAPTER MANIFESTS")
    manifests, timing_metadata, integrity_report = build_chapter_manifests(
        state.get("validated_ssml_scripts") or [],
        state.get("qc_passed_audio_clips") or [],
        state.get("job_lookup_maps") or {},
        state.get("character_personas") or [],
    )
    state["chapter_audio_manifests"] = manifests
    state["timing_metadata"] = timing_metadata
    state["manifest_integrity_report"] = integrity_report
    print(
        f"   ✅ Manifests: {len(manifests)} | "
        f"Incomplete: {len(integrity_report.get('errors', []))}"
    )
    return state


def package_output_node(state: Phase4State) -> Phase4State:
    """Emit the final Phase 4 output contract."""

    print("\n📦 PHASE 4 OUTPUT PACKAGING")
    phase4_output, summary_metrics = package_phase4_output(state)
    state["phase4_output"] = phase4_output
    state["phase4_summary_metrics"] = summary_metrics
    state["ready_for_phase5"] = phase4_output["ready_for_phase5"]
    state["audio_files"] = phase4_output["audio_files"]
    print(
        f"   {'✅' if phase4_output['ready_for_phase5'] else '⚠️ '} "
        f"Ready for Phase 5: {phase4_output['ready_for_phase5']}"
    )
    return state


def route_after_validation(state: Phase4State) -> str:
    """Fail fast when the input contract is already broken."""

    if state.get("phase4_blocked"):
        return "package_phase4_output"
    return "resolve_voice_policy"


def create_phase4_graph():
    """Create and compile the Phase 4 voice synthesis graph."""

    workflow = StateGraph(Phase4State)

    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("resolve_voice_policy", resolve_voice_policy_node)
    workflow.add_node("plan_tts_jobs", plan_tts_jobs_node)
    workflow.add_node("translate_tts_jobs", translate_jobs_node)
    workflow.add_node("route_tts_jobs", route_jobs_node)
    workflow.add_node("execute_parallel_synthesis", execute_synthesis_node)
    workflow.add_node("audio_qc_and_repair", qc_and_repair_node)
    workflow.add_node("build_chapter_manifests", build_manifests_node)
    workflow.add_node("package_phase4_output", package_output_node)

    workflow.set_entry_point("validate_input")
    workflow.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {
            "resolve_voice_policy": "resolve_voice_policy",
            "package_phase4_output": "package_phase4_output",
        },
    )
    workflow.add_edge("resolve_voice_policy", "plan_tts_jobs")
    workflow.add_edge("plan_tts_jobs", "translate_tts_jobs")
    workflow.add_edge("translate_tts_jobs", "route_tts_jobs")
    workflow.add_edge("route_tts_jobs", "execute_parallel_synthesis")
    workflow.add_edge("execute_parallel_synthesis", "audio_qc_and_repair")
    workflow.add_edge("audio_qc_and_repair", "build_chapter_manifests")
    workflow.add_edge("build_chapter_manifests", "package_phase4_output")
    workflow.add_edge("package_phase4_output", END)

    return workflow.compile()
