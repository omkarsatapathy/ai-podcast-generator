"""Phase 4 helpers for provider routing, synthesis, QC, and packaging."""

from __future__ import annotations

import hashlib
import html
import json
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests

from config.settings import settings
from src.models.phase4 import (
    AudioClip,
    ChapterAudioManifest,
    Phase4AudioMetadata,
    Phase4ChapterScript,
    Phase4Output,
    SpeakerVoiceAssignment,
    TTSJob,
)
from src.tools.audio_tools import inspect_wav_file, validate_wav_file, write_audio_bytes_atomic
from src.tools.elevenlabs_tts import synthesize_elevenlabs_speech
from src.tools.gemini_tts import synthesize_gemini_speech
from src.tools.openai_tts import synthesize_openai_speech
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global concurrency limiter — caps simultaneous outgoing TTS API calls
# Lazily initialised so it picks up runtime settings value.
_api_semaphore: threading.Semaphore | None = None
_semaphore_lock = threading.Lock()


def _get_api_semaphore() -> threading.Semaphore:
    global _api_semaphore
    if _api_semaphore is None:
        with _semaphore_lock:
            if _api_semaphore is None:
                _api_semaphore = threading.Semaphore(settings.PHASE4_MAX_CONCURRENT_API_CALLS)
    return _api_semaphore


def _throttle_api_call() -> None:
    """Acquire concurrency slot before making a TTS API call."""
    _get_api_semaphore().acquire()


def validate_phase4_input_contract(state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], bool]:
    """Validate and normalize the Phase 3 -> Phase 4 handoff."""

    provider = _provider_name(settings.TTS_PROVIDER)
    raw_scripts = state.get("ssml_annotated_scripts") or state.get("chapter_dialogues") or []
    personas = state.get("character_personas") or []
    known_speakers = {persona.get("name") for persona in personas if persona.get("name")}
    known_roles = {persona.get("role") for persona in personas if persona.get("role")}

    report: Dict[str, Any] = {
        "errors": [],
        "warnings": [],
        "stats": {
            "chapters": 0,
            "utterances": 0,
        },
    }
    if not raw_scripts:
        report["errors"].append("No Phase 3 scripts found for Phase 4 input")
        return [], report, True

    validated_scripts: List[Dict[str, Any]] = []
    seen_utterance_ids: set[str] = set()
    chapter_numbers: List[int] = []

    for chapter in sorted(raw_scripts, key=lambda item: item.get("chapter_number", 0)):
        try:
            validated_chapter = Phase4ChapterScript.model_validate(chapter)
        except Exception as exc:
            report["errors"].append(f"Chapter validation failed: {exc}")
            continue

        chapter_numbers.append(validated_chapter.chapter_number)
        normalized_chapter = dict(chapter)
        normalized_chapter["chapter_number"] = validated_chapter.chapter_number
        normalized_chapter["utterances"] = []

        for index, original in enumerate(chapter.get("utterances", [])):
            utterance = dict(original)
            utterance["audio_metadata"] = Phase4AudioMetadata.model_validate(
                utterance.get("audio_metadata") or {}
            ).model_dump()

            try:
                normalized_utterance = Phase4ChapterScript.model_validate({
                    "chapter_number": validated_chapter.chapter_number,
                    "utterances": [utterance],
                }).utterances[0]
            except Exception as exc:
                report["errors"].append(
                    f"Utterance validation failed for chapter {validated_chapter.chapter_number}: {exc}"
                )
                continue

            utterance.update(normalized_utterance.model_dump())
            report["stats"]["utterances"] += 1

            utterance_id = utterance["utterance_id"]
            if utterance_id in seen_utterance_ids:
                report["errors"].append(f"Duplicate utterance_id detected: {utterance_id}")
            else:
                seen_utterance_ids.add(utterance_id)

            text_ssml = utterance.get("text_ssml", "").strip()
            if not text_ssml:
                report["errors"].append(f"Missing text_ssml for {utterance_id}")
            elif text_ssml.startswith("<"):
                try:
                    ET.fromstring(text_ssml)
                except ET.ParseError:
                    # Auto-repair: wrap clean text in <speak> so one bad
                    # SSML utterance doesn't block the entire pipeline.
                    clean = utterance.get("text_clean", "")
                    clean_escaped = re.sub(r'&(?!amp;|lt;|gt;|apos;|quot;|#)', '&amp;', clean)
                    utterance["text_ssml"] = f"<speak>{clean_escaped}</speak>"
                    report["warnings"].append(
                        f"Auto-repaired invalid SSML for {utterance_id} (using text_clean fallback)"
                    )
            elif provider == "google":
                report["warnings"].append(
                    f"Google provider received plain text for {utterance_id}; routing will normalize it"
                )

            audio_metadata = utterance["audio_metadata"]
            if audio_metadata.get("interrupt_duration") and index == 0:
                report["errors"].append(
                    f"Interrupt marker cannot appear on the first utterance of chapter {validated_chapter.chapter_number}"
                )

            if audio_metadata.get("backchannel_speaker"):
                backchannel_target = audio_metadata["backchannel_speaker"]
                if (
                    known_speakers
                    and backchannel_target not in known_speakers
                    and backchannel_target not in known_roles
                ):
                    report["warnings"].append(
                        f"Backchannel target '{backchannel_target}' does not match a known speaker or role"
                    )

            normalized_chapter["utterances"].append(utterance)

        report["stats"]["chapters"] += 1
        validated_scripts.append(normalized_chapter)

    if chapter_numbers:
        unique_chapters = sorted(set(chapter_numbers))
        expected = list(range(unique_chapters[0], unique_chapters[-1] + 1))
        if unique_chapters != expected:
            report["errors"].append(
                "Chapter numbers are not contiguous in Phase 4 input"
            )

    return validated_scripts, report, bool(report["errors"])


def resolve_voice_policy(
    validated_scripts: List[Dict[str, Any]],
    personas: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Resolve primary and fallback voice assignments for the episode."""

    provider = _provider_name(settings.TTS_PROVIDER)
    fallback_provider = _fallback_provider()
    persona_map = {persona.get("name"): persona for persona in personas if persona.get("name")}
    speakers_in_order = _unique_speakers(validated_scripts)

    speaker_voice_map: Dict[str, Any] = {}
    warnings: List[str] = []
    errors: List[str] = []

    for speaker in speakers_in_order:
        persona = persona_map.get(speaker, {})
        role = persona.get("role") or _role_from_scripts(validated_scripts, speaker)

        voice_id, source, warning = _resolve_provider_voice(provider, role, persona)
        if warning:
            warnings.append(f"{speaker}: {warning}")
        if not voice_id:
            errors.append(f"Unable to resolve {provider} voice for speaker '{speaker}'")
            continue

        fallback_voice_id = None
        fallback_model = None
        if fallback_provider and fallback_provider != provider:
            fallback_voice_id, _, fallback_warning = _resolve_provider_voice(
                fallback_provider,
                role,
                persona,
            )
            if fallback_warning:
                warnings.append(f"{speaker}: {fallback_warning}")
            if fallback_voice_id:
                fallback_model = _model_for_provider(fallback_provider)

        assignment = SpeakerVoiceAssignment(
            speaker=speaker,
            role=role,
            provider=provider,
            model=_model_for_provider(provider),
            voice_id=voice_id,
            source=source,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
            fallback_voice_id=fallback_voice_id,
        )
        speaker_voice_map[speaker] = assignment.model_dump()

    voice_lock_signature = _stable_hash(speaker_voice_map)
    report = {
        "errors": errors,
        "warnings": warnings,
        "voice_lock_signature": voice_lock_signature,
    }
    if errors:
        raise ValueError("; ".join(errors))

    policy = {
        "primary_provider": provider,
        "primary_model": _model_for_provider(provider),
        "fallback_provider": fallback_provider,
        "fallback_model": _model_for_provider(fallback_provider) if fallback_provider else None,
        "voice_lock_signature": voice_lock_signature,
    }
    return speaker_voice_map, policy, report


def plan_tts_jobs(
    validated_scripts: List[Dict[str, Any]],
    speaker_voice_map: Dict[str, Any],
    episode_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, str]]:
    """Flatten chapter scripts into deterministic utterance-level jobs."""

    jobs: List[Dict[str, Any]] = []
    job_lookup_maps: Dict[str, Any] = {
        "jobs_by_chapter": defaultdict(list),
        "jobs_by_lineage_utterance": defaultdict(list),
    }
    planned_output_paths: Dict[str, str] = {}
    order_index = 0

    for chapter in sorted(validated_scripts, key=lambda item: item["chapter_number"]):
        chapter_number = chapter["chapter_number"]
        for utterance in chapter.get("utterances", []):
            assignment = speaker_voice_map[utterance["speaker"]]
            segment_texts = _split_for_synthesis(_plain_text_for_job(utterance))
            segment_count = len(segment_texts)
            estimated_duration = utterance.get("estimated_duration_seconds", 0.0)
            segment_duration = estimated_duration / segment_count if segment_count else estimated_duration

            for segment_index, segment_text in enumerate(segment_texts):
                suffix = "" if segment_count == 1 else f"_{chr(ord('a') + segment_index)}"
                job_id = f"{utterance['utterance_id']}{suffix}"
                output_path = (
                    Path(settings.PHASE4_RAW_AUDIO_DIR)
                    / episode_id
                    / f"ch_{chapter_number:02d}"
                    / f"{job_id}.wav"
                )

                job = TTSJob(
                    job_id=job_id,
                    episode_id=episode_id,
                    chapter_number=chapter_number,
                    utterance_id=job_id,
                    lineage_utterance_id=utterance["utterance_id"],
                    order_index=order_index,
                    segment_index=segment_index,
                    segment_count=segment_count,
                    speaker=utterance["speaker"],
                    role=utterance.get("role", "unknown"),
                    provider=assignment["provider"],
                    model=assignment["model"],
                    voice_id=assignment["voice_id"],
                    text_clean=segment_text,
                    text_with_naturalness=segment_text,
                    text_ssml=utterance.get("text_ssml", "") if segment_count == 1 else segment_text,
                    estimated_duration_seconds=max(segment_duration, 0.0),
                    audio_metadata=(
                        Phase4AudioMetadata.model_validate(utterance.get("audio_metadata") or {})
                        if segment_index == 0
                        else Phase4AudioMetadata()
                    ),
                    output_path=str(output_path),
                    fallback_provider=assignment.get("fallback_provider"),
                    fallback_model=assignment.get("fallback_model"),
                    fallback_voice_id=assignment.get("fallback_voice_id"),
                    original_text_clean=utterance.get("text_clean", ""),
                    original_text_ssml=utterance.get("text_ssml", ""),
                )
                job_dict = job.model_dump()
                jobs.append(job_dict)
                job_lookup_maps["jobs_by_chapter"][str(chapter_number)].append(job_id)
                job_lookup_maps["jobs_by_lineage_utterance"][utterance["utterance_id"]].append(job_id)
                planned_output_paths[job_id] = str(output_path)
                order_index += 1

    return jobs, _materialize_lookup_maps(job_lookup_maps), planned_output_paths


def route_tts_jobs(
    tts_jobs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Build provider-ready payloads for each planned job."""

    routed_jobs: List[Dict[str, Any]] = []
    routing_decisions: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for raw_job in tts_jobs:
        job = TTSJob.model_validate(raw_job)
        payload = build_provider_payload(job)
        risk_flags = []
        if len(payload["text"]) > settings.PHASE4_MAX_TEXT_CHARS_PER_JOB:
            risk_flags.append("long_text")
        if "<" in payload["text"] or ">" in payload["text"]:
            risk_flags.append("markup_residue")
        if not payload["text"]:
            warnings.append(f"{job.job_id}: routed payload text is empty")

        routed_job = job.model_copy(update={"payload": payload}).model_dump()
        routed_jobs.append(routed_job)
        routing_decisions.append({
            "job_id": job.job_id,
            "provider": job.provider,
            "voice_id": job.voice_id,
            "risk_flags": risk_flags,
        })

    report = {
        "total_jobs": len(routed_jobs),
        "warnings": warnings,
        "risky_jobs": [decision for decision in routing_decisions if decision["risk_flags"]],
    }
    return routed_jobs, routing_decisions, report


def build_provider_payload(job: TTSJob) -> Dict[str, Any]:
    """Build a provider-specific request payload for a synthesis job."""

    text = _plain_text_for_job(job.model_dump())
    if job.provider == "google":
        prompt_prefix = "Speak clearly and naturally. Say exactly:"
        if not job.repair_mode:
            prompt_prefix = (
                f"Speak naturally as a {job.role} in a podcast conversation. "
                "Keep the wording exact. Say:"
            )
        return {
            "provider": "google",
            "model": job.model,
            "voice_id": job.voice_id,
            "text": text,
            "prompt": f"{prompt_prefix} {text}",
        }

    if job.provider == "elevenlabs":
        return {
            "provider": "elevenlabs",
            "model": job.model,
            "voice_id": job.voice_id,
            "text": text,
        }

    if job.provider == "openai":
        instructions = "Speak clearly and naturally."
        if not job.repair_mode:
            instructions = (
                f"You are a {job.role} on a podcast. "
                "Speak naturally and conversationally. Keep the wording exact."
            )
        return {
            "provider": "openai",
            "model": job.model,
            "voice_id": job.voice_id,
            "text": text,
            "instructions": instructions,
        }

    raise ValueError(f"Unsupported TTS provider: {job.provider}")


def execute_parallel_synthesis(
    routed_tts_jobs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run synthesis jobs with bounded parallelism and retry support."""

    if not routed_tts_jobs:
        return [], [], []

    raw_audio_clips: List[Dict[str, Any]] = []
    failed_jobs: List[Dict[str, Any]] = []
    synthesis_log: List[Dict[str, Any]] = []

    total = len(routed_tts_jobs)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, settings.PHASE4_MAX_WORKERS)) as executor:
        future_map = {
            executor.submit(_run_job_with_retries, job, settings.PHASE4_MAX_RETRIES): job
            for job in routed_tts_jobs
        }

        for future in as_completed(future_map):
            result = future.result()
            done += 1
            job_meta = future_map[future]
            if result["clip"]:
                raw_audio_clips.append(result["clip"])
                dur = result["clip"].get("duration_seconds", 0.0)
                print(
                    f"   ✅ [{done:>3}/{total}] {job_meta['job_id']}  "
                    f"({job_meta['provider']} / {job_meta['voice_id']})  {dur:.1f}s",
                    flush=True,
                )
            else:
                err = (result["failed_job"] or {}).get("error", "unknown error")
                print(
                    f"   ❌ [{done:>3}/{total}] {job_meta['job_id']}  "
                    f"({job_meta['provider']})  {err}",
                    flush=True,
                )
            synthesis_log.extend(result["log"])
            if result["failed_job"]:
                failed_jobs.append(result["failed_job"])

    raw_audio_clips.sort(key=lambda clip: clip["order_index"])
    failed_jobs.sort(key=lambda job: job.get("order_index", 0))
    synthesis_log.sort(key=lambda entry: (entry.get("order_index", 0), entry.get("attempt", 0)))
    return raw_audio_clips, failed_jobs, synthesis_log


def synthesize_routed_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize a single routed job and write the WAV file."""

    validated_job = TTSJob.model_validate(job)

    # Reuse an already-saved WAV from a previous run instead of re-calling the API
    output_path = Path(validated_job.output_path)
    if output_path.exists() and output_path.stat().st_size > 0:
        metadata = inspect_wav_file(str(output_path))
        clip = AudioClip(
            job_id=validated_job.job_id,
            utterance_id=validated_job.utterance_id,
            lineage_utterance_id=validated_job.lineage_utterance_id,
            chapter_number=validated_job.chapter_number,
            order_index=validated_job.order_index,
            speaker=validated_job.speaker,
            role=validated_job.role,
            provider=validated_job.provider,
            model=validated_job.model,
            voice_id=validated_job.voice_id,
            path=str(output_path),
            duration_seconds=metadata["duration_seconds"],
            sample_rate=metadata["sample_rate"],
            channels=metadata["channels"],
            size_bytes=metadata["size_bytes"],
            audio_metadata=validated_job.audio_metadata,
            qc_passed=True,
        )
        return {
            "clip": clip.model_dump(),
            "provider_result": {
                "sample_rate": metadata["sample_rate"],
                "channels": metadata["channels"],
                "sample_width": 2,
            },
        }

    payload = validated_job.payload or build_provider_payload(validated_job)

    _throttle_api_call()
    try:
        if validated_job.provider == "google":
            result = synthesize_gemini_speech(
                prompt=payload["prompt"],
                voice_name=payload["voice_id"],
                model=payload["model"],
                project_id=settings.GCP_PROJECT_ID,
                location=settings.GCP_LOCATION,
            )
            from src.utils.cost_tracker import cost_tracker
            cost_tracker.track_tts(
                model=payload["model"],
                input_tokens=result.get("input_tokens", 0),
            )
        elif validated_job.provider == "elevenlabs":
            result = synthesize_elevenlabs_speech(
                text=payload["text"],
                voice_id=payload["voice_id"],
                model=payload["model"],
                api_key=settings.ELEVENLABS_API_KEY,
                timeout_seconds=settings.PHASE4_REQUEST_TIMEOUT_SECONDS,
            )
        elif validated_job.provider == "openai":
            result = synthesize_openai_speech(
                text=payload["text"],
                voice=payload["voice_id"],
                model=payload["model"],
                api_key=settings.OPENAI_API_KEY,
                instructions=payload.get("instructions", ""),
                timeout_seconds=settings.PHASE4_REQUEST_TIMEOUT_SECONDS,
            )
            from src.utils.cost_tracker import cost_tracker
            # Approximate token count from character length (1 token ≈ 4 chars)
            estimated_tokens = max(1, len(payload.get("text", "")) // 4)
            cost_tracker.track_tts(
                model=payload["model"],
                input_tokens=estimated_tokens,
            )
        else:
            raise ValueError(f"Unsupported TTS provider: {validated_job.provider}")
    finally:
        _get_api_semaphore().release()

    write_audio_bytes_atomic(
        path=validated_job.output_path,
        audio_bytes=result["audio_bytes"],
        sample_rate=result["sample_rate"],
        channels=result["channels"],
        sample_width=result["sample_width"],
    )

    metadata = inspect_wav_file(validated_job.output_path)
    clip = AudioClip(
        job_id=validated_job.job_id,
        utterance_id=validated_job.utterance_id,
        lineage_utterance_id=validated_job.lineage_utterance_id,
        chapter_number=validated_job.chapter_number,
        order_index=validated_job.order_index,
        speaker=validated_job.speaker,
        role=validated_job.role,
        provider=validated_job.provider,
        model=validated_job.model,
        voice_id=validated_job.voice_id,
        path=validated_job.output_path,
        duration_seconds=metadata["duration_seconds"],
        sample_rate=metadata["sample_rate"],
        channels=metadata["channels"],
        size_bytes=metadata["size_bytes"],
        audio_metadata=validated_job.audio_metadata,
        qc_passed=True,
    )
    return {
        "clip": clip.model_dump(),
        "provider_result": {
            "sample_rate": result["sample_rate"],
            "channels": result["channels"],
            "sample_width": result["sample_width"],
        },
    }


def audio_qc_and_repair(
    routed_tts_jobs: List[Dict[str, Any]],
    raw_audio_clips: List[Dict[str, Any]],
    failed_jobs: List[Dict[str, Any]],
    synthesis_log: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Run technical QC and selectively repair bad clips."""

    job_map = {job["job_id"]: job for job in routed_tts_jobs}
    qc_passed_audio_clips: List[Dict[str, Any]] = []
    qc_failed_jobs: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    repaired_count = 0

    for clip in raw_audio_clips:
        try:
            report = validate_wav_file(
                path=clip["path"],
                min_duration_seconds=settings.PHASE4_MIN_DURATION_SECONDS,
                silence_peak_threshold=settings.PHASE4_SILENCE_PEAK_THRESHOLD,
                clipping_sample_threshold=settings.PHASE4_CLIPPING_SAMPLE_THRESHOLD,
                clipping_ratio_threshold=settings.PHASE4_CLIPPING_RATIO_THRESHOLD,
                expected_sample_rate=settings.PHASE4_TARGET_SAMPLE_RATE,
                expected_channels=settings.PHASE4_TARGET_CHANNELS,
            )
        except Exception as exc:
            report = {
                "path": clip["path"],
                "passed": False,
                "reasons": [f"wave_decode_failed: {exc}"],
            }

        report["job_id"] = clip["job_id"]
        reports.append(report)
        if report["passed"]:
            clip["qc_passed"] = True
            qc_passed_audio_clips.append(clip)
            continue

        repair_job = job_map.get(clip["job_id"])
        if repair_job:
            repair_result = _attempt_repair(repair_job)
            synthesis_log.extend(repair_result["log"])
            if repair_result["clip"]:
                try:
                    repair_report = validate_wav_file(
                        path=repair_result["clip"]["path"],
                        min_duration_seconds=settings.PHASE4_MIN_DURATION_SECONDS,
                        silence_peak_threshold=settings.PHASE4_SILENCE_PEAK_THRESHOLD,
                        clipping_sample_threshold=settings.PHASE4_CLIPPING_SAMPLE_THRESHOLD,
                        clipping_ratio_threshold=settings.PHASE4_CLIPPING_RATIO_THRESHOLD,
                        expected_sample_rate=settings.PHASE4_TARGET_SAMPLE_RATE,
                        expected_channels=settings.PHASE4_TARGET_CHANNELS,
                    )
                except Exception as exc:
                    repair_report = {
                        "path": repair_result["clip"]["path"],
                        "passed": False,
                        "reasons": [f"wave_decode_failed: {exc}"],
                    }

                repair_report["job_id"] = clip["job_id"]
                reports.append(repair_report)
                if repair_report["passed"]:
                    repaired_clip = repair_result["clip"]
                    repaired_clip["qc_passed"] = True
                    qc_passed_audio_clips.append(repaired_clip)
                    repaired_count += 1
                    continue

        qc_failed_jobs.append({
            **clip,
            "qc_reasons": report["reasons"],
        })

    qc_passed_audio_clips.sort(key=lambda clip: clip["order_index"])
    qc_failed_jobs.sort(key=lambda clip: clip.get("order_index", 0))
    qc_report = {
        "total_clips": len(raw_audio_clips),
        "passed": len(qc_passed_audio_clips),
        "failed": len(qc_failed_jobs),
        "repaired": repaired_count,
        "reports": reports,
        "initial_failures": len(failed_jobs),
    }
    return qc_passed_audio_clips, qc_failed_jobs, qc_report


def build_chapter_manifests(
    validated_scripts: List[Dict[str, Any]],
    qc_passed_audio_clips: List[Dict[str, Any]],
    job_lookup_maps: Dict[str, Any],
    personas: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Build chapter-level manifests and timing directives for Phase 5."""

    clips_by_chapter: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for clip in qc_passed_audio_clips:
        clips_by_chapter[clip["chapter_number"]].append(clip)

    role_to_speaker = {
        persona.get("role"): persona.get("name")
        for persona in personas
        if persona.get("role") and persona.get("name")
    }
    manifests: List[Dict[str, Any]] = []
    timing_metadata: Dict[str, Any] = {"chapters": {}}
    errors: List[str] = []

    for chapter in sorted(validated_scripts, key=lambda item: item["chapter_number"]):
        chapter_number = chapter["chapter_number"]
        chapter_clips = sorted(
            clips_by_chapter.get(chapter_number, []),
            key=lambda clip: clip["order_index"],
        )
        expected_job_ids = job_lookup_maps["jobs_by_chapter"].get(str(chapter_number), [])
        actual_job_ids = [clip["job_id"] for clip in chapter_clips]
        complete = actual_job_ids == expected_job_ids
        if not complete:
            errors.append(
                f"Chapter {chapter_number} manifest incomplete: expected {len(expected_job_ids)} clips, got {len(actual_job_ids)}"
            )

        timing_directives = []
        for utterance in chapter.get("utterances", []):
            audio_metadata = Phase4AudioMetadata.model_validate(
                utterance.get("audio_metadata") or {}
            )
            timing_directives.append({
                "utterance_id": utterance["utterance_id"],
                "gap_after_seconds": settings.PHASE4_DEFAULT_TURN_GAP_SECONDS,
                "overlap_previous_seconds": _seconds_from_duration(audio_metadata.interrupt_duration),
                "backchannel_speaker": _resolve_backchannel_speaker(
                    audio_metadata.backchannel_speaker,
                    role_to_speaker,
                ),
            })

        timing_metadata["chapters"][str(chapter_number)] = timing_directives
        manifest = ChapterAudioManifest(
            manifest_version="1.0",
            chapter_number=chapter_number,
            utterance_count=len(chapter.get("utterances", [])),
            complete=complete,
            clip_checksum=_stable_hash({
                "chapter": chapter_number,
                "job_ids": actual_job_ids,
            }),
            clips=chapter_clips,
            timing_directives=timing_directives,
        )
        manifests.append(manifest.model_dump())

    report = {
        "total_chapters": len(validated_scripts),
        "complete_chapters": sum(1 for manifest in manifests if manifest["complete"]),
        "errors": errors,
    }
    return manifests, timing_metadata, report


def package_phase4_output(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Package the final Phase 4 handoff contract and summary metrics."""

    provider_policy = state.get("provider_fallback_policy") or {}
    primary_provider = provider_policy.get("primary_provider") or _provider_name(settings.TTS_PROVIDER)
    primary_model = provider_policy.get("primary_model") or _model_for_provider(primary_provider)

    audio_files = sorted(
        state.get("qc_passed_audio_clips") or [],
        key=lambda clip: clip.get("order_index", 0),
    )
    chapter_manifests = state.get("chapter_audio_manifests") or []
    failed_jobs = [*(state.get("failed_jobs") or []), *(state.get("qc_failed_jobs") or [])]
    quality_reports = {
        "validation_report": state.get("validation_report") or {},
        "voice_resolution_report": state.get("voice_resolution_report") or {},
        "payload_validation_report": state.get("payload_validation_report") or {},
        "synthesis_log": state.get("synthesis_log") or [],
        "qc_report": state.get("qc_report") or {},
        "manifest_integrity_report": state.get("manifest_integrity_report") or {},
    }

    qc_failed = state.get("qc_failed_jobs") or []
    total_clips = len(audio_files) + len(qc_failed)
    failure_ratio = len(qc_failed) / total_clips if total_clips else 0.0
    ready_for_phase5 = (
        not (state.get("phase4_blocked") or False)
        and not (state.get("validation_report") or {}).get("errors")
        and not (state.get("manifest_integrity_report") or {}).get("errors")
        and failure_ratio <= settings.PHASE4_MAX_FAILURE_RATIO
        and bool(audio_files)
    )

    phase4_output = Phase4Output(
        phase4_contract_version="1.0",
        episode_id=state.get("episode_id") or "",
        provider=primary_provider,
        model=primary_model,
        ready_for_phase5=ready_for_phase5,
        audio_files=audio_files,
        chapter_manifests=chapter_manifests,
        voice_metadata={
            "speaker_voice_map": state.get("speaker_voice_map") or {},
            "voice_lock_signature": provider_policy.get("voice_lock_signature"),
        },
        timing_metadata=state.get("timing_metadata") or {},
        quality_reports=quality_reports,
        failed_jobs=failed_jobs,
    ).model_dump()
    validate_phase4_output_contract(phase4_output)

    summary_metrics = {
        "chapters": len(chapter_manifests),
        "audio_files": len(audio_files),
        "failed_jobs": len(failed_jobs),
        "ready_for_phase5": ready_for_phase5,
    }
    return phase4_output, summary_metrics


def validate_phase4_output_contract(phase4_output: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the final Phase 4 handoff contract."""

    return Phase4Output.model_validate(phase4_output).model_dump()


def ensure_episode_id(state: Dict[str, Any], validated_scripts: List[Dict[str, Any]]) -> str:
    """Ensure the run has a stable episode_id for deterministic file naming."""

    if state.get("episode_id"):
        return str(state["episode_id"])

    topic = state.get("topic", "podcast-episode")
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-") or "podcast-episode"
    chapter_signature = [
        {
            "chapter_number": chapter["chapter_number"],
            "utterance_ids": [utterance["utterance_id"] for utterance in chapter.get("utterances", [])],
        }
        for chapter in validated_scripts
    ]
    digest = _stable_hash({"topic": topic, "chapters": chapter_signature})[:10]
    return f"{slug[:40]}-{digest}"


def _run_job_with_retries(job: Dict[str, Any], max_retries: int) -> Dict[str, Any]:
    """Synthesize a job with retries and optional provider fallback."""

    attempts = 0
    execution_log: List[Dict[str, Any]] = []
    job_dict = dict(job)

    while attempts <= max_retries:
        attempts += 1
        attempt_job = dict(job_dict)
        attempt_job["retry_count"] = attempts - 1
        started = time.time()
        try:
            result = synthesize_routed_job(attempt_job)
            execution_log.append({
                "job_id": attempt_job["job_id"],
                "order_index": attempt_job["order_index"],
                "attempt": attempts,
                "provider": attempt_job["provider"],
                "status": "success",
                "elapsed_seconds": round(time.time() - started, 3),
            })
            return {"clip": result["clip"], "failed_job": None, "log": execution_log}
        except Exception as exc:
            execution_log.append({
                "job_id": attempt_job["job_id"],
                "order_index": attempt_job["order_index"],
                "attempt": attempts,
                "provider": attempt_job["provider"],
                "status": "retry" if attempts <= max_retries and _is_retryable(exc) else "failed",
                "error": str(exc),
                "elapsed_seconds": round(time.time() - started, 3),
            })
            if attempts <= max_retries and _is_retryable(exc):
                delay = _retry_delay_seconds(attempts, exc)
                print(
                    f"   ⏳  [{attempt_job['job_id']}] retry {attempts}/{max_retries} in {delay:.1f}s  —  {exc}",
                    flush=True,
                )
                time.sleep(delay)
                continue
            break

    fallback_job = _build_fallback_job(job_dict)
    if fallback_job:
        started = time.time()
        try:
            result = synthesize_routed_job(fallback_job)
            execution_log.append({
                "job_id": fallback_job["job_id"],
                "order_index": fallback_job["order_index"],
                "attempt": attempts + 1,
                "provider": fallback_job["provider"],
                "status": "success_via_fallback",
                "elapsed_seconds": round(time.time() - started, 3),
            })
            return {"clip": result["clip"], "failed_job": None, "log": execution_log}
        except Exception as exc:
            execution_log.append({
                "job_id": fallback_job["job_id"],
                "order_index": fallback_job["order_index"],
                "attempt": attempts + 1,
                "provider": fallback_job["provider"],
                "status": "fallback_failed",
                "error": str(exc),
                "elapsed_seconds": round(time.time() - started, 3),
            })

    failed_job = dict(job_dict)
    failed_job["error"] = execution_log[-1].get("error", "Synthesis failed")
    return {"clip": None, "failed_job": failed_job, "log": execution_log}


def _attempt_repair(job: Dict[str, Any]) -> Dict[str, Any]:
    """Re-run a job once with a safer payload."""

    repaired = dict(job)
    repaired["repair_mode"] = True
    repaired["text_clean"] = repaired.get("original_text_clean") or repaired["text_clean"]
    repaired["text_with_naturalness"] = repaired["text_clean"]
    repaired["text_ssml"] = repaired["text_clean"]
    repaired["payload"] = build_provider_payload(TTSJob.model_validate(repaired))
    return _run_job_with_retries(repaired, 0)


def _build_fallback_job(job: Dict[str, Any]) -> Dict[str, Any] | None:
    """Build a fallback-provider version of a routed job if configured."""

    if not job.get("fallback_provider") or not job.get("fallback_voice_id"):
        return None

    fallback_job = dict(job)
    fallback_job["provider"] = job["fallback_provider"]
    fallback_job["model"] = job["fallback_model"] or _model_for_provider(job["fallback_provider"])
    fallback_job["voice_id"] = job["fallback_voice_id"]
    fallback_job["payload"] = build_provider_payload(TTSJob.model_validate(fallback_job))
    return fallback_job


def _plain_text_for_job(job: Dict[str, Any]) -> str:
    """Extract provider-ready plain text from SSML or marker-rich text."""

    text_ssml = (job.get("text_ssml") or "").strip()
    if text_ssml.startswith("<"):
        text = re.sub(r"<break[^>]*>", " ... ", text_ssml)
        text = re.sub(r"</?[^>]+>", " ", text)
    else:
        text = job.get("text_with_naturalness") or job.get("text_clean") or text_ssml

    text = re.sub(r"\[[A-Z_]+(?::[^\]]+)?\]", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _split_for_synthesis(text: str) -> List[str]:
    """Split only when an utterance is unusually long for a single TTS call."""

    text = text.strip()
    if not text:
        return [text]
    if len(text) <= settings.PHASE4_MAX_TEXT_CHARS_PER_JOB:
        return [text]

    segments: List[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?;:])\s+", text):
        candidate = sentence.strip()
        if not candidate:
            continue
        proposed = f"{current} {candidate}".strip()
        if current and len(proposed) > settings.PHASE4_MAX_TEXT_CHARS_PER_JOB:
            segments.append(current)
            current = candidate
        else:
            current = proposed

    if current:
        segments.append(current)

    if any(len(segment) > settings.PHASE4_MAX_TEXT_CHARS_PER_JOB for segment in segments):
        return [
            text[i:i + settings.PHASE4_MAX_TEXT_CHARS_PER_JOB].strip()
            for i in range(0, len(text), settings.PHASE4_MAX_TEXT_CHARS_PER_JOB)
        ]
    return segments or [text]


def _resolve_provider_voice(provider: str, role: str, persona: Dict[str, Any]) -> Tuple[str | None, str, str | None]:
    """Resolve the best available voice for a provider."""

    role = role or "host"
    if provider == "google":
        voice_id = (
            persona.get("google_tts_voice_id")
            or persona.get("tts_voice_id")
            or _default_voice_for_role(provider, role)
        )
        if voice_id not in settings.GOOGLE_TTS_ALLOWED_VOICES:
            fallback_voice = _default_voice_for_role(provider, role)
            if fallback_voice:
                return fallback_voice, "default", f"Unsupported Google voice '{voice_id}', using role default"
            return None, "missing", f"Unsupported Google voice '{voice_id}' and no default configured"
        source = "persona" if persona.get("tts_voice_id") == voice_id else "default"
        return voice_id, source, None

    if provider == "elevenlabs":
        voice_id = (
            persona.get("elevenlabs_voice_id")
            or persona.get("provider_voice_id")
            or _default_voice_for_role(provider, role)
        )
        if not voice_id:
            return None, "missing", "No ElevenLabs voice_id configured"
        source = "persona" if persona.get("elevenlabs_voice_id") == voice_id else "default"
        return voice_id, source, None

    if provider == "openai":
        voice_id = (
            persona.get("openai_tts_voice")
            or persona.get("tts_voice_id")
            or _default_voice_for_role(provider, role)
        )
        if voice_id and voice_id.lower() not in {v.lower() for v in settings.OPENAI_TTS_ALLOWED_VOICES}:
            fallback_voice = _default_voice_for_role(provider, role)
            if fallback_voice:
                return fallback_voice, "default", f"Unsupported OpenAI voice '{voice_id}', using role default"
            return None, "missing", f"Unsupported OpenAI voice '{voice_id}' and no default configured"
        if not voice_id:
            return None, "missing", "No OpenAI TTS voice configured"
        source = "persona" if persona.get("openai_tts_voice") == voice_id else "default"
        return voice_id.lower(), source, None

    return None, "missing", f"Unsupported provider '{provider}'"


def _default_voice_for_role(provider: str, role: str) -> str | None:
    """Return configured role defaults for the selected provider."""

    role = role or "host"
    if provider == "google":
        return {
            "host": settings.GOOGLE_TTS_HOST_VOICE,
            "expert": settings.GOOGLE_TTS_EXPERT_VOICE,
            "skeptic": settings.GOOGLE_TTS_SKEPTIC_VOICE,
        }.get(role, settings.GOOGLE_TTS_HOST_VOICE)

    if provider == "elevenlabs":
        return {
            "host": settings.ELEVENLABS_HOST_VOICE_ID,
            "expert": settings.ELEVENLABS_EXPERT_VOICE_ID,
            "skeptic": settings.ELEVENLABS_SKEPTIC_VOICE_ID,
        }.get(role, settings.ELEVENLABS_HOST_VOICE_ID)

    if provider == "openai":
        return {
            "host": settings.OPENAI_TTS_HOST_VOICE,
            "expert": settings.OPENAI_TTS_EXPERT_VOICE,
            "skeptic": settings.OPENAI_TTS_SKEPTIC_VOICE,
        }.get(role, settings.OPENAI_TTS_HOST_VOICE)

    return None


def _provider_name(value: str) -> str:
    provider = (value or "google").strip().lower()
    if provider not in {"google", "elevenlabs", "openai"}:
        raise ValueError(f"Unsupported TTS provider: {value}")
    return provider


def _fallback_provider() -> str | None:
    fallback = (settings.TTS_FALLBACK_PROVIDER or "").strip().lower()
    if not fallback:
        return None
    return _provider_name(fallback)


def _model_for_provider(provider: str | None) -> str | None:
    if provider == "google":
        return settings.GOOGLE_TTS_MODEL
    if provider == "elevenlabs":
        return settings.ELEVENLABS_TTS_MODEL
    if provider == "openai":
        return settings.OPENAI_TTS_MODEL
    return None


def _role_from_scripts(validated_scripts: Iterable[Dict[str, Any]], speaker: str) -> str:
    for chapter in validated_scripts:
        for utterance in chapter.get("utterances", []):
            if utterance.get("speaker") == speaker:
                return utterance.get("role", "host")
    return "host"


def _unique_speakers(validated_scripts: Iterable[Dict[str, Any]]) -> List[str]:
    speakers: List[str] = []
    seen: set[str] = set()
    for chapter in validated_scripts:
        for utterance in chapter.get("utterances", []):
            speaker = utterance.get("speaker")
            if speaker and speaker not in seen:
                speakers.append(speaker)
                seen.add(speaker)
    return speakers


def _resolve_backchannel_speaker(target: str | None, role_to_speaker: Dict[str, str]) -> str | None:
    if not target:
        return None
    return role_to_speaker.get(target, target)


def _seconds_from_duration(value: str | None) -> float:
    if not value:
        return 0.0
    match = re.match(r"([\d.]+)s", value)
    return float(match.group(1)) if match else 0.0


def _retry_delay_seconds(attempt_number: int, exc: Exception | None = None) -> float:
    # For 429 rate-limit errors use a much longer exponential backoff
    if exc is not None and isinstance(exc, requests.HTTPError) and exc.response is not None:
        if exc.response.status_code == 429:
            retry_after = exc.response.headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after) + 1.0
                except ValueError:
                    pass
            # 30s, 60s, 120s … for successive 429 attempts
            base = 30.0 * (2 ** max(attempt_number - 1, 0))
            return base + random.uniform(0.0, 5.0)
    base = settings.PHASE4_RETRY_BASE_SECONDS * (2 ** max(attempt_number - 1, 0))
    return base + random.uniform(0.0, 0.25)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}

    message = str(exc).lower()
    transient_markers = ["timeout", "temporarily", "rate limit", "429", "500", "502", "503", "504"]
    return any(marker in message for marker in transient_markers)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _materialize_lookup_maps(job_lookup_maps: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: dict(value) if isinstance(value, defaultdict) else value
        for key, value in job_lookup_maps.items()
    }
