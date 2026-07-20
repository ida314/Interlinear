"""ffmpeg and soundfile interop.

ffmpeg is used only at the edges — decode on ingest, encode on output, and time-stretching
individual TTS clips. All splicing happens in numpy (see `render_audio`), because ffmpeg's
concat demuxer over hundreds of tiny files is slow, fragile at frame boundaries, and cannot
cut at sub-millisecond precision.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

# 24kHz mono matches Kokoro's native output rate, so no resampling is needed when splicing
# synthesised speech into source audio.
TARGET_SR = 24000
ASR_SR = 16000      # what every speech model in this stack expects


class FFmpegError(RuntimeError):
    pass


def ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FFmpegError("ffmpeg not found on PATH")
    return exe


def _run(args: list[str], stdin: bytes | None = None) -> bytes:
    proc = subprocess.run(
        args, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-8:]
        raise FFmpegError("ffmpeg failed:\n" + "\n".join(tail))
    return proc.stdout


def decode_to_wav(src: str | Path, dst: str | Path, *, sr: int = TARGET_SR) -> Path:
    """Normalise any input media to mono PCM at `sr`. This becomes the render source of
    truth, so every timestamp downstream refers to it."""
    dst = Path(dst)
    _run([
        ffmpeg_path(), "-nostdin", "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst),
    ])
    return dst


def decode_to_array(src: str | Path, *, sr: int = ASR_SR) -> np.ndarray:
    """Decode straight to a mono float32 array at `sr`, without writing a file.

    Speech models want 16kHz while the render source of truth is 24kHz. Resampling changes
    no timestamp — seconds are seconds — so this needs no offset correction anywhere; it
    exists so the model path never mutates the audio the renderer splices.
    """
    raw = _run([
        ffmpeg_path(), "-nostdin", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ])
    return np.frombuffer(raw, dtype=np.float32).copy()


def read_wav(path: str | Path, *, expect_sr: int | None = None) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)
    if expect_sr and sr != expect_sr:
        raise ValueError(f"{path}: expected {expect_sr}Hz, got {sr}Hz")
    return data, sr


def write_wav(path: str | Path, x: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr, subtype="PCM_16")
    return path


def _atempo_chain(speed: float) -> str:
    """ffmpeg's atempo is well-conditioned near 1.0; chain filters for extreme values."""
    factors, remaining = [], speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={f:.6f}" for f in factors)


def time_stretch(buf: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Change playback rate without altering pitch, via ffmpeg's atempo.

    Piped rather than written to temp files: a few-second clip round-trips in single-digit
    milliseconds, which is what keeps the live preview responsive.
    """
    if abs(speed - 1.0) < 1e-3 or len(buf) == 0:
        return buf
    raw = _run(
        [
            ffmpeg_path(), "-nostdin", "-loglevel", "error",
            "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
            "-af", _atempo_chain(speed),
            "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1",
        ],
        stdin=buf.astype(np.float32).tobytes(),
    )
    return np.frombuffer(raw, dtype=np.float32)


def encode_mp3(
    wav_path: str | Path,
    mp3_path: str | Path,
    *,
    title: str = "",
    artist: str = "",
    quality: int = 2,
    loudnorm: bool = True,
) -> Path:
    args = [ffmpeg_path(), "-nostdin", "-y", "-i", str(wav_path)]
    if loudnorm:
        # Single-pass EBU R128. Two-pass would be more accurate but doubles render time for
        # a difference nobody listening on headphones will notice.
        args += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
    args += ["-c:a", "libmp3lame", "-q:a", str(quality)]
    if title:
        args += ["-metadata", f"title={title}"]
    if artist:
        args += ["-metadata", f"artist={artist}"]
    args.append(str(mp3_path))
    _run(args)
    return Path(mp3_path)


def probe_duration(path: str | Path) -> float:
    exe = shutil.which("ffprobe")
    if not exe:
        raise FFmpegError("ffprobe not found on PATH")
    out = _run([
        exe, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(out.decode().strip())
