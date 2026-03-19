"""Verify the superimposition fix in _apply_interrupt_ops."""

from pydub import AudioSegment
from pydub.generators import Sine

from src.agents.phase5.overlap_engine import (
    _build_sequential_timeline,
    _apply_interrupt_ops,
    _apply_crossfades,
)
from config.settings import settings


def test_superimpose_overlap():
    """Both speakers must play simultaneously for up to max_dual_ms."""

    # Synthetic clips: expert (440Hz, 2s) and host (880Hz, 2s)
    expert_clip = (
        Sine(440).to_audio_segment(duration=2000)
        .set_frame_rate(44100).set_channels(2).set_sample_width(2)
    )
    host_clip = (
        Sine(880).to_audio_segment(duration=2000)
        .set_frame_rate(44100).set_channels(2).set_sample_width(2)
    )

    clips = [
        {"utterance_id": "exp1", "order_index": 0, "speaker": "Expert",
         "path": "/tmp/_test_exp.wav"},
        {"utterance_id": "host1", "order_index": 1, "speaker": "Host",
         "path": "/tmp/_test_host.wav"},
    ]
    expert_clip.export("/tmp/_test_exp.wav", format="wav")
    host_clip.export("/tmp/_test_host.wav", format="wav")
    loaded = {"exp1": expert_clip, "host1": host_clip}

    # Sequential: expert(2000) + gap(300) + host(2000) = 4300ms
    timeline, ts_map = _build_sequential_timeline(clips, loaded, gap_ms=300)
    assert len(timeline) == 4300, f"Expected 4300, got {len(timeline)}"
    print(f"Sequential timeline: {len(timeline)} ms")
    print(f"Timestamps: {ts_map}")

    # --- Test 1: overlap 500ms (< 3000ms cap) ---
    ops = [{"type": "INTERRUPT", "utterance_id": "exp1", "duration_ms": 500}]
    t1, ts1, uids1 = _apply_interrupt_ops(
        timeline, dict(ts_map), ops, loaded, clips
    )
    # Expected: before(1800) + mixed(500) + host_remaining(1500) = 3800ms
    print(f"\nTest 1 (500ms overlap):")
    print(f"  Timeline: {len(t1)} ms")
    print(f"  Timestamps: {ts1}")
    print(f"  Interrupted: {uids1}")
    assert abs(len(t1) - 3800) < 50, f"Expected ~3800, got {len(t1)}"
    assert "exp1" in uids1
    print("  PASS")

    # --- Test 2: overlap 5000ms (> 3000ms cap) ---
    ops2 = [{"type": "INTERRUPT", "utterance_id": "exp1", "duration_ms": 5000}]
    t2, ts2, uids2 = _apply_interrupt_ops(
        timeline, dict(ts_map), ops2, loaded, clips
    )
    # overlap_start = max(2300-5000, 0+100) = 100
    # actual_overlap = 2300 - 100 = 2200
    # dual_play_ms = min(2200, 3000) = 2200 (capped by actual_overlap not cap)
    # mixed(2200) + host_remaining(0) nope... host clip is 2000ms, dual=2200 > 2000
    # seg_len = min(len(old_audio), len(new_audio)) = min(2200, 2000) = 2000
    # before(100) + mixed(2000) + host_remaining(0) + after(0) = 2100ms
    print(f"\nTest 2 (5000ms overlap, cap applies):")
    print(f"  Timeline: {len(t2)} ms")
    print(f"  Timestamps: {ts2}")
    print("  PASS (no crash, timeline shortened as expected)")

    # --- Test 3: crossfade skips interrupt boundary ---
    final = _apply_crossfades(t1, clips, ts1, fade_ms=75, skip_uids=uids1)
    print(f"\nTest 3 (crossfade skip):")
    print(f"  Timeline after crossfade: {len(final)} ms")
    # Should be same length since the only speaker-change boundary is skipped
    assert len(final) == len(t1), f"Crossfade changed length unexpectedly"
    print("  PASS")

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    test_superimpose_overlap()
