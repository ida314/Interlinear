"""Fast excerpt rendering for live knob adjustment.

The whole feature rests on stage 6 being a pure function that is separate from stage 7. A
pacing change re-plans and re-renders two or three sentences in pure numpy — no GPU, no
model, single-digit milliseconds — so a slider can be dragged and the result heard
immediately.

The source waveform is cached because decoding it is the only slow part. One entry is
enough: a user tuning knobs is working on one job at a time.
"""

from __future__ import annotations

import io as _io
from pathlib import Path

import numpy as np
import soundfile as sf

from app.audio import io as audio_io
from app.models.timeline import RenderParams, Timeline
from app.pipeline.plan import build_plan
from app.pipeline.render_audio import render_to_array

# Sentences either side of the requested one, for context. Hearing the transition into and
# out of a translation is the point; a single sentence in isolation tells you nothing about
# whether the pacing works.
CONTEXT_SENTENCES = 1


class SourceCache:
    """One decoded waveform, kept in memory.

    A 60-minute video is ~345MB as float32, which is nothing on this hardware and far
    cheaper than re-reading it on every slider movement.
    """

    def __init__(self) -> None:
        self._job_id: str | None = None
        self._audio: np.ndarray | None = None
        self._sr: int = 0

    def get(self, job_id: str, wav_path: Path) -> tuple[np.ndarray, int]:
        if self._job_id != job_id or self._audio is None:
            self._audio, self._sr = audio_io.read_wav(wav_path)
            self._job_id = job_id
        return self._audio, self._sr

    def invalidate(self, job_id: str | None = None) -> None:
        if job_id is None or job_id == self._job_id:
            self._job_id, self._audio = None, None


_cache = SourceCache()


def excerpt_segment_ids(timeline: Timeline, index: int, *, context: int = CONTEXT_SENTENCES) -> set[str]:
    """Segment ids around `index`, clamped to the timeline."""
    translatable = [s for s in timeline.segments if s.kind == "speech"]
    if not translatable:
        return set()
    index = max(0, min(index, len(translatable) - 1))
    lo = max(0, index - context)
    hi = min(len(translatable), index + context + 1)
    return {s.id for s in translatable[lo:hi]}


def render_excerpt(
    timeline: Timeline,
    job_dir: Path,
    *,
    segment_index: int = 0,
    params: RenderParams | None = None,
    cache: SourceCache | None = None,
) -> bytes:
    """Render a short excerpt with the given pacing knobs. Returns WAV bytes.

    Only render-time knobs are honoured. Anything that would change sentence boundaries,
    translations or voices needs the GPU stages and is not previewable — the caller is
    responsible for not offering those as live sliders.
    """
    params = params or timeline.params
    cache = cache or _cache

    audio, sr = cache.get(timeline.job_id, Path(job_dir) / timeline.source.wav_path)
    ids = excerpt_segment_ids(timeline, segment_index)
    if not ids:
        raise ValueError("no speech segments to preview")

    plan = build_plan(timeline, audio, sr, params, segment_ids=ids)
    buf = render_to_array(
        plan,
        audio,
        Path(job_dir),
        tts_speed=params.tts_speed,
        match_loudness=params.match_loudness,
    )

    out = _io.BytesIO()
    sf.write(out, buf, sr, format="WAV", subtype="PCM_16")
    return out.getvalue()
