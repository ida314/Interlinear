"""Synthetic fixtures. Everything here is CPU-only so the core logic can be tested on a
machine with no GPU and no network."""

from __future__ import annotations

import numpy as np
import pytest

from app.models.timeline import RenderParams, Segment, SourceInfo, Timeline, TTSClip, Word

SR = 24000


def make_audio(words: list[Word], *, sr: int = SR, tail: float = 1.0, noise: float = 0.0) -> np.ndarray:
    """Build a waveform that is loud exactly where words are and near-silent between.

    This lets the quiet-seam search be tested against ground truth: the correct cut is the
    silent trough, and we know precisely where it is.
    """
    duration = (words[-1].end if words else 0.0) + tail
    n = int(duration * sr)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(n).astype(np.float32) * noise) if noise else np.zeros(n, dtype=np.float32)
    t = np.arange(n) / sr
    for w in words:
        mask = (t >= w.start) & (t < w.end)
        # A tone burst is enough; the search only cares about energy.
        x[mask] += 0.4 * np.sin(2 * np.pi * 180.0 * t[mask]).astype(np.float32)
    return x


def add_music(audio: np.ndarray, start: float, end: float, *, sr: int = SR) -> np.ndarray:
    """Fill a span with audible non-speech, so it counts as content rather than dead air."""
    out = audio.copy()
    a, b = int(start * sr), min(len(out), int(end * sr))
    t = np.arange(b - a) / sr
    out[a:b] += (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return out


def build_timeline(
    spec: list[list[tuple[str, float, float]]],
    *,
    lang: str = "es",
    params: RenderParams | None = None,
    translated: bool = True,
    tts_duration: float = 1.0,
) -> tuple[Timeline, np.ndarray]:
    """Build a Timeline from a nested spec: outer list = segments, inner = (text, start, end).

    Returns (timeline, audio).
    """
    words: list[Word] = []
    segments: list[Segment] = []
    for i, seg_words in enumerate(spec):
        start_idx = len(words)
        words.extend(Word(text=t, start=s, end=e) for t, s, e in seg_words)
        seg = Segment(id=f"seg{i:03d}", index=i, word_start=start_idx, word_end=len(words))
        if translated:
            seg.translation = f"translation {i}"
            seg.tts = TTSClip(
                path=f"tts/{i:04d}.wav",
                duration=tts_duration,
                voice="af_heart",
                lang="en",
                text_sha=f"sha{i}",
            )
        segments.append(seg)

    audio = make_audio(words)
    tl = Timeline(
        job_id="test",
        source=SourceInfo(
            url="https://example.test/v",
            detected_lang=lang,
            duration=len(audio) / SR,
            wav_path="source.wav",
        ),
        params=params or RenderParams(),
        words=words,
        segments=segments,
    )
    tl.rebuild_all_derived()
    return tl, audio


@pytest.fixture
def simple_timeline():
    """Three sentences with clear silence between them."""
    return build_timeline(
        [
            [("Hola", 1.0, 1.4), ("mundo", 1.5, 2.0)],
            [("Como", 3.0, 3.4), ("estas", 3.5, 4.0)],
            [("Adios", 5.0, 5.6)],
        ]
    )


@pytest.fixture
def tight_timeline():
    """Two sentences butted directly against each other — no silence to trim."""
    return build_timeline(
        [
            [("Hola", 1.0, 1.4), ("mundo", 1.42, 2.0)],
            [("Como", 2.02, 2.4), ("estas", 2.42, 3.0)],
        ]
    )
