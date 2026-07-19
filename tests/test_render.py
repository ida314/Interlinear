"""End-to-end assembly checks against synthetic audio.

Source bursts and TTS clips use distinct frequencies, so the rendered output can be probed
to confirm the right content landed at the right time. This is the test that would catch a
one-sample drift compounding into audible desync over a long video.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.audio import io
from app.pipeline.plan import build_plan
from app.pipeline.render_audio import render_to_array, write_lrc
from tests.conftest import SR, build_timeline

SOURCE_HZ = 180.0
TTS_HZ = 880.0


def _tone(hz: float, seconds: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.4 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _dominant_hz(buf: np.ndarray, sr: int = SR) -> float:
    if len(buf) < 64:
        return 0.0
    spectrum = np.abs(np.fft.rfft(buf * np.hanning(len(buf))))
    return float(np.fft.rfftfreq(len(buf), 1 / sr)[int(np.argmax(spectrum))])


@pytest.fixture
def job(tmp_path):
    """A two-sentence timeline with real TTS wavs on disk."""
    tl, audio = build_timeline(
        [
            [("Hola", 1.0, 1.4), ("mundo", 1.5, 2.0)],
            [("Adios", 4.0, 4.5)],
        ],
        tts_duration=1.0,
    )
    for seg in tl.segments:
        path = tmp_path / seg.tts.path
        path.parent.mkdir(parents=True, exist_ok=True)
        io.write_wav(path, _tone(TTS_HZ, seg.tts.duration), SR)
    return tl, audio, tmp_path


def test_rendered_length_matches_the_plan(job):
    tl, audio, job_dir = job
    plan = build_plan(tl, audio, SR)
    out = render_to_array(plan, audio, job_dir, match_loudness=False)

    assert len(out) == pytest.approx(int(plan.total_duration * SR), abs=2)


def test_translation_audio_lands_where_the_plan_says(job):
    """The core correctness property: probe each tts clip's midpoint and confirm the
    synthesised tone — not source audio — is playing there."""
    tl, audio, job_dir = job
    plan = build_plan(tl, audio, SR)
    out = render_to_array(plan, audio, job_dir, match_loudness=False)

    tts_clips = [c for c in plan.clips if c.kind == "tts"]
    assert len(tts_clips) == 2

    for clip in tts_clips:
        mid = int((clip.out_start + clip.duration / 2) * SR)
        window = out[mid - 2048 : mid + 2048]
        assert _dominant_hz(window) == pytest.approx(TTS_HZ, abs=30)


def test_source_audio_survives_at_its_own_offsets(job):
    tl, audio, job_dir = job
    plan = build_plan(tl, audio, SR)
    out = render_to_array(plan, audio, job_dir, match_loudness=False)

    speech = [c for c in plan.clips if c.kind == "source" and c.segment_id]
    for clip in speech:
        mid = int((clip.out_start + clip.duration / 2) * SR)
        window = out[mid - 2048 : mid + 2048]
        assert _dominant_hz(window) == pytest.approx(SOURCE_HZ, abs=30)


def test_gaps_are_actually_silent(job):
    tl, audio, job_dir = job
    plan = build_plan(tl, audio, SR)
    out = render_to_array(plan, audio, job_dir, match_loudness=False)

    for clip in (c for c in plan.clips if c.kind == "silence"):
        mid = int((clip.out_start + clip.duration / 2) * SR)
        assert float(np.abs(out[mid - 200 : mid + 200]).max()) < 1e-6


def test_speed_knob_shortens_output_without_losing_content(job):
    """tts_speed is applied at render time via atempo, never baked into the cached wav."""
    tl, audio, job_dir = job
    base = render_to_array(plan := build_plan(tl, audio, SR), audio, job_dir, match_loudness=False)
    fast_plan = build_plan(tl, audio, SR, tl.params.model_copy(update={"tts_speed": 1.5}))
    fast = render_to_array(fast_plan, audio, job_dir, tts_speed=1.5, match_loudness=False)

    assert len(fast) < len(base)
    fast_clip = next(c for c in fast_plan.clips if c.kind == "tts")
    mid = int((fast_clip.out_start + fast_clip.duration / 2) * SR)
    # Pitch must be preserved — atempo, not resampling.
    assert _dominant_hz(fast[mid - 2048 : mid + 2048]) == pytest.approx(TTS_HZ, abs=40)
    assert plan.total_duration > fast_plan.total_duration


def test_loudness_matching_brings_tts_toward_the_source(tmp_path):
    """A quiet TTS clip should be lifted toward the surrounding speech level."""
    tl, audio = build_timeline([[("Hola", 1.0, 1.5)]], tts_duration=1.0)
    quiet = _tone(TTS_HZ, 1.0) * 0.05
    io.write_wav(tmp_path / tl.segments[0].tts.path, quiet, SR)

    plan = build_plan(tl, audio, SR)
    matched = render_to_array(plan, audio, tmp_path, match_loudness=True)
    raw = render_to_array(plan, audio, tmp_path, match_loudness=False)

    clip = next(c for c in plan.clips if c.kind == "tts")
    sl = slice(int(clip.out_start * SR) + 1000, int(clip.out_end * SR) - 1000)
    assert np.abs(matched[sl]).max() > np.abs(raw[sl]).max() * 2


def test_lrc_sidecar_is_timed_to_the_translations(job, tmp_path):
    tl, audio, job_dir = job
    plan = build_plan(tl, audio, SR)
    path = write_lrc(plan, tmp_path / "out.lrc", title="Test")

    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("[0")]
    assert len(lines) == 2
    assert "translation 0" in lines[0]
