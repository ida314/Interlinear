"""Stage 1: URL -> normalised audio on disk.

yt-dlp is invoked as a subprocess rather than imported, so it can be upgraded independently
of this project. It breaks against YouTube changes regularly and you will need to update it
out of band; pinning it in the lockfile and never touching it is not a viable strategy.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from app.audio import io
from app.config import Settings, settings
from app.models.timeline import SourceInfo


class FetchError(RuntimeError):
    """Carries yt-dlp's own stderr.

    Deliberately verbatim: geo-blocks, age gates, members-only videos and signature
    extraction failures all surface here, and a generic "download failed" would strip the
    one piece of information that tells you which it was.
    """


def ytdlp_command() -> list[str]:
    """How to invoke yt-dlp here.

    Prefer the console script, but fall back to running the module under the *current*
    interpreter. A virtualenv that was never `activate`d has yt-dlp installed and importable
    while its bin directory is absent from PATH, which is the common case for a service run
    by an absolute path to `.venv/bin/python`.
    """
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    try:
        import yt_dlp  # noqa: F401, PLC0415
    except ImportError:
        raise FetchError(
            "yt-dlp not found on PATH and not importable. Install it with "
            "`pip install yt-dlp` into the same environment as this app."
        ) from None
    return [sys.executable, "-m", "yt_dlp"]


# yt-dlp only enables deno by default, but YouTube extraction now needs *some* JS runtime and
# fails with a bare 403 without one. Anything on this list will do, so use whatever the box
# happens to have rather than making the operator install a second runtime.
JS_RUNTIMES = ("deno", "node", "bun")


def js_runtime_args() -> list[str]:
    """Enable a non-default JS runtime when one is present and deno is not."""
    if shutil.which("deno"):
        return []          # the default already covers this
    for runtime in JS_RUNTIMES:
        if shutil.which(runtime):
            return ["--js-runtimes", runtime]
    return []              # let yt-dlp fail with its own explanatory message


def _ytdlp(args: list[str], cfg: Settings) -> subprocess.CompletedProcess[str]:
    cmd = [*ytdlp_command(), "--no-playlist", "--no-progress", *js_runtime_args()]
    if cfg.ytdlp_cookies_from_browser:
        cmd += ["--cookies-from-browser", cfg.ytdlp_cookies_from_browser]
    cmd += args
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise FetchError(proc.stderr.strip() or "yt-dlp failed with no output")
    return proc


def probe(url: str, cfg: Settings | None = None) -> dict:
    """Metadata only — no download. Used to reject over-long videos before spending disk."""
    cfg = cfg or settings
    proc = _ytdlp(["--dump-single-json", "--skip-download", url], cfg)
    return json.loads(proc.stdout)


def fetch(
    url: str,
    job_dir: Path,
    *,
    cfg: Settings | None = None,
    want_video: bool = False,
) -> SourceInfo:
    cfg = cfg or settings
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    meta = probe(url, cfg)
    duration = float(meta.get("duration") or 0.0)
    if duration > cfg.max_video_duration:
        raise FetchError(
            f"Video is {duration / 60:.0f} minutes; the limit is "
            f"{cfg.max_video_duration / 60:.0f}. Raise BAG_MAX_VIDEO_DURATION to override."
        )

    audio_tmpl = str(job_dir / "source.%(ext)s")
    _ytdlp(["-f", cfg.ytdlp_format, "-o", audio_tmpl, "--", url], cfg)
    downloaded = next(
        (p for p in sorted(job_dir.glob("source.*")) if p.suffix not in {".wav", ".json"}), None
    )
    if downloaded is None:
        raise FetchError("yt-dlp reported success but produced no file")

    video_path = None
    if want_video:
        video_tmpl = str(job_dir / "video.%(ext)s")
        _ytdlp(["-f", "bv*+ba/b", "-o", video_tmpl, "--", url], cfg)
        video = next(iter(sorted(job_dir.glob("video.*"))), None)
        video_path = video.name if video else None

    # One canonical decode. Every timestamp downstream refers to this file, so it must be
    # produced exactly once and never re-derived.
    wav = io.decode_to_wav(downloaded, job_dir / "source.wav", sr=cfg.tts_sample_rate)

    return SourceInfo(
        url=url,
        video_id=meta.get("id", ""),
        title=meta.get("title", ""),
        uploader=meta.get("uploader"),
        duration=duration or io.probe_duration(wav),
        audio_path=downloaded.name,
        wav_path=wav.name,
        video_path=video_path,
    )
