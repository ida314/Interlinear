"""Pure-numpy audio primitives. No I/O, no ffmpeg, no GPU — so this is fast to test.

The important one is `find_quietest_point`. Cutting audio at an ASR segment boundary
sounds bad because the boundary lands wherever the decoder decided a token ended, which is
frequently mid-vowel. Searching a small window for the lowest-energy instant lands the cut
in a stop consonant or glottal closure instead, where a splice is inaudible.
"""

from __future__ import annotations

import numpy as np

# Below this RMS a window counts as genuine silence and the splice is free.
SILENCE_RMS = 0.005


def to_float32(x: np.ndarray) -> np.ndarray:
    """Normalise int16/int32 PCM to float32 in [-1, 1]. Float input passes through."""
    if x.dtype == np.float32:
        return x
    if x.dtype == np.float64:
        return x.astype(np.float32)
    if np.issubdtype(x.dtype, np.integer):
        return (x.astype(np.float32) / float(np.iinfo(x.dtype).max)).astype(np.float32)
    return x.astype(np.float32)


def rms_envelope(
    x: np.ndarray, sr: int, *, frame_ms: float = 10.0, hop_ms: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Short-time RMS. Returns (rms, centre_times_in_seconds).

    Frames are built with a strided view rather than a Python loop, so this stays cheap
    enough to call per segment boundary on a full-length video.
    """
    x = to_float32(x)
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))

    if len(x) < frame:
        val = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
        return np.array([val], dtype=np.float32), np.array([len(x) / (2 * sr)])

    n_frames = 1 + (len(x) - frame) // hop
    strided = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, frame), strides=(x.strides[0] * hop, x.strides[0]), writeable=False
    )
    rms = np.sqrt(np.mean(strided.astype(np.float64) ** 2, axis=1)).astype(np.float32)
    times = (np.arange(n_frames) * hop + frame / 2.0) / sr
    return rms, times


def find_quietest_point(
    x: np.ndarray,
    sr: int,
    lo: float,
    hi: float,
    *,
    frame_ms: float = 10.0,
    hop_ms: float = 5.0,
    prefer_centre: float | None = None,
) -> tuple[float, float]:
    """Find the best cut instant within [lo, hi] seconds. Returns (time, rms_at_time).

    When several frames are near-equally quiet, `prefer_centre` biases the choice toward a
    preferred time (normally the midpoint of the inter-word gap). Without that tiebreak the
    argmin jitters to an arbitrary edge of a flat silent region, which produces
    inconsistent-feeling pacing between sentences.
    """
    lo = max(0.0, lo)
    hi = min(len(x) / sr, hi)
    if hi <= lo:
        t = max(0.0, min(hi, lo))
        return t, float("inf")

    a, b = int(lo * sr), int(hi * sr)
    window = x[a:b]
    if len(window) == 0:
        return lo, float("inf")

    rms, times = rms_envelope(window, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    times = times + lo

    if prefer_centre is None:
        idx = int(np.argmin(rms))
        return float(times[idx]), float(rms[idx])

    # Among frames within 20% of the minimum, take the one nearest the preferred centre.
    floor = float(rms.min())
    tolerance = floor + max(0.2 * floor, 1e-5)
    candidates = np.flatnonzero(rms <= tolerance)
    idx = int(candidates[np.argmin(np.abs(times[candidates] - prefer_centre))])
    return float(times[idx]), float(rms[idx])


def fade_in(buf: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    if seconds <= 0 or len(buf) == 0:
        return buf
    n = min(len(buf), int(sr * seconds))
    if n <= 1:
        return buf
    out = buf.copy()
    out[:n] *= np.linspace(0.0, 1.0, n, dtype=out.dtype)
    return out


def fade_out(buf: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    if seconds <= 0 or len(buf) == 0:
        return buf
    n = min(len(buf), int(sr * seconds))
    if n <= 1:
        return buf
    out = buf.copy()
    out[-n:] *= np.linspace(1.0, 0.0, n, dtype=out.dtype)
    return out


def apply_fades(buf: np.ndarray, sr: int, fin: float, fout: float) -> np.ndarray:
    return fade_out(fade_in(buf, sr, fin), sr, fout)


def speech_rms(x: np.ndarray, sr: int, *, percentile: float = 75.0) -> float:
    """Loudness of the *speech* in a signal, ignoring its silences.

    A plain mean over a whole video is dominated by pauses and would push the synthesised
    voice far too loud. Taking a high percentile of the frame RMS distribution approximates
    the level of active speech. Adequate for matching a TTS clip to its surroundings;
    swap in pyloudnorm's BS.1770 integrated loudness if this ever needs to be exact.
    """
    if len(x) == 0:
        return 0.0
    rms, _ = rms_envelope(x, sr, frame_ms=50.0, hop_ms=25.0)
    voiced = rms[rms > SILENCE_RMS]
    if voiced.size == 0:
        return float(np.percentile(rms, percentile))
    return float(np.percentile(voiced, percentile))


def match_gain(buf: np.ndarray, sr: int, target_rms: float, *, max_gain_db: float = 12.0) -> np.ndarray:
    """Scale `buf` toward `target_rms`, clamped so a near-silent clip cannot explode."""
    if target_rms <= 0 or len(buf) == 0:
        return buf
    current = speech_rms(buf, sr)
    if current <= 1e-9:
        return buf
    gain = target_rms / current
    limit = 10 ** (max_gain_db / 20.0)
    gain = float(np.clip(gain, 1.0 / limit, limit))
    out = buf * gain
    # Guard against clipping introduced by the gain.
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def silence(sr: int, seconds: float) -> np.ndarray:
    return np.zeros(max(0, int(round(sr * seconds))), dtype=np.float32)
