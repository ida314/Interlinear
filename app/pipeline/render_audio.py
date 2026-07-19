"""Stage 7: RenderPlan -> mp3.

Dumb by design. Every timing decision was already made in stage 6; this just fills a buffer.
Keeping it that way is what makes the phase-2 video renderer a small addition rather than a
rewrite — it consumes the same plan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.audio import dsp, io
from app.models.plan import RenderPlan
from app.models.timeline import Timeline


def render_to_array(
    plan: RenderPlan,
    source: np.ndarray,
    job_dir: Path,
    *,
    tts_speed: float = 1.0,
    match_loudness: bool = True,
) -> np.ndarray:
    """Execute the edit list into a single waveform."""
    sr = plan.sample_rate
    out = np.zeros(int(round(plan.total_duration * sr)) + sr, dtype=np.float32)

    # Match synthesised speech to the loudness of the actual speech around it. Without this
    # the TTS voice sits noticeably above YouTube audio and the result is fatiguing.
    target_rms = dsp.speech_rms(source, sr) if match_loudness else 0.0

    cursor = 0
    for clip in plan.clips:
        want = int(round(clip.duration * sr))

        if clip.kind == "silence":
            buf = np.zeros(want, dtype=np.float32)
        elif clip.kind == "source":
            a = int(round(clip.src_start * sr))
            buf = source[a : a + want]
        else:
            buf = _load_tts(job_dir / clip.path, sr, tts_speed, target_rms)

        buf = _fit(buf, want)
        buf = dsp.apply_fades(buf, sr, clip.fade_in, clip.fade_out)
        out[cursor : cursor + len(buf)] = buf
        cursor += want

    return out[:cursor]


def _load_tts(path: Path, sr: int, speed: float, target_rms: float) -> np.ndarray:
    buf, clip_sr = io.read_wav(path)
    if clip_sr != sr:
        raise ValueError(f"{path}: {clip_sr}Hz does not match plan rate {sr}Hz")
    if target_rms > 0:
        buf = dsp.match_gain(buf, sr, target_rms)
    # Applied here rather than at synthesis time so the TTS cache stays speed-independent.
    return io.time_stretch(buf, sr, speed)


def _fit(buf: np.ndarray, want: int) -> np.ndarray:
    """Pad or trim to the exact sample count the plan promised.

    Time-stretching and resampling both land within a sample or two of the requested length;
    forcing the issue here is what guarantees the plan's arithmetic stays authoritative.
    """
    if len(buf) == want:
        return buf
    if len(buf) > want:
        return buf[:want]
    return np.concatenate([buf, np.zeros(want - len(buf), dtype=np.float32)])


def render(
    timeline: Timeline,
    plan: RenderPlan,
    job_dir: Path,
    *,
    source: np.ndarray | None = None,
    basename: str = "out",
) -> Path:
    """Full render: plan -> wav -> mp3, plus an .lrc sidecar."""
    job_dir = Path(job_dir)
    if source is None:
        source, _ = io.read_wav(job_dir / timeline.source.wav_path, expect_sr=plan.sample_rate)

    audio = render_to_array(
        plan,
        source,
        job_dir,
        tts_speed=timeline.params.tts_speed,
        match_loudness=timeline.params.match_loudness,
    )

    wav_path = io.write_wav(job_dir / f"{basename}.wav", audio, plan.sample_rate)
    mp3_path = io.encode_mp3(
        wav_path,
        job_dir / f"{basename}.mp3",
        title=timeline.source.title,
        artist=timeline.source.uploader or "",
    )
    write_lrc(plan, job_dir / f"{basename}.lrc", title=timeline.source.title)
    wav_path.unlink(missing_ok=True)
    return mp3_path


def write_lrc(plan: RenderPlan, path: Path, *, title: str = "") -> Path:
    """Timed lyrics of the translations — lets a player show the English while it plays."""
    lines = []
    if title:
        lines.append(f"[ti:{title}]")
    for clip in plan.clips:
        if clip.kind != "tts" or not clip.label:
            continue
        m, s = divmod(clip.out_start, 60)
        text = clip.label.replace("\n", " ").strip()
        lines.append(f"[{int(m):02d}:{s:05.2f}]{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
