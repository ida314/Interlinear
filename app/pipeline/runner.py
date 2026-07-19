"""Stage orchestration, and the CLI entry point.

Every stage persists `timeline.json` on completion, so a job can resume from wherever it
died rather than re-running an expensive ASR pass. That is also what makes knob changes
cheap: adjusting pacing re-runs only `plan` and `render`, while re-segmenting re-runs from
`segment` onward and the TTS cache absorbs most of even that.

Run the whole pipeline without any web layer:

    python -m app.pipeline.runner <url> --target en
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Callable

from app.audio import io
from app.config import Settings, settings
from app.models.timeline import RenderParams, Timeline

# Ordered. Each stage's name is recorded in `timeline.stages_done`.
STAGES = ["fetch", "asr", "segment", "translate", "tts", "render"]

# Changing a knob invalidates its stage and everything after it.
STAGE_DEPS = {
    "segmentation_mode": "segment", "max_words_per_chunk": "segment", "min_words": "segment",
    "min_duration": "segment", "max_duration": "segment", "max_words": "segment",
    "never_merge_across_gap": "segment", "sat_threshold": "segment",
    "pause_weight": "segment", "pause_saturation": "segment", "llm_arbitration": "segment",
    "target_lang": "translate",
    "voice": "tts",
}

ProgressFn = Callable[[str, float], None]


class Cancelled(RuntimeError):
    """Raised when a cancellation was requested between stages."""


def _noop(stage: str, fraction: float) -> None: ...


def _never() -> bool:
    return False


def timeline_path(job_dir: Path) -> Path:
    return Path(job_dir) / "timeline.json"


def save(timeline: Timeline, job_dir: Path) -> Path:
    path = timeline_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
    return path


def load(job_dir: Path) -> Timeline | None:
    path = timeline_path(job_dir)
    if not path.exists():
        return None
    return Timeline.model_validate_json(path.read_text(encoding="utf-8"))


def invalidated_from(old: RenderParams, new: RenderParams) -> str | None:
    """Earliest stage that must re-run given a parameter change.

    Returns None when only render-time knobs moved — the case that keeps the preview
    instant.
    """
    earliest = None
    for field, stage in STAGE_DEPS.items():
        if getattr(old, field) != getattr(new, field):
            if earliest is None or STAGES.index(stage) < STAGES.index(earliest):
                earliest = stage
    return earliest


def run(
    url: str,
    *,
    job_id: str | None = None,
    params: RenderParams | None = None,
    cfg: Settings | None = None,
    progress: ProgressFn = _noop,
    resume: bool = True,
    language: str | None = None,
    should_cancel: Callable[[], bool] = _never,
) -> tuple[Timeline, Path]:
    """Run the pipeline to completion. Returns (timeline, mp3_path).

    Cancellation is cooperative and checked between stages. A stage in flight runs to
    completion — interrupting a CUDA pass mid-kernel is not worth the added complexity, and
    every stage persists its result, so nothing is lost either way.
    """
    cfg = cfg or settings

    def checkpoint() -> None:
        if should_cancel():
            raise Cancelled

    job_id = job_id or uuid.uuid4().hex[:12]
    job_dir = cfg.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    params = params or RenderParams()

    timeline = load(job_dir) if resume else None
    if timeline is not None:
        stale = invalidated_from(timeline.params, params)
        timeline.params = params
        if stale:
            cut = STAGES.index(stale)
            timeline.stages_done = [s for s in timeline.stages_done if STAGES.index(s) < cut]

    done = set(timeline.stages_done) if timeline else set()

    # --- 1. fetch ---
    if timeline is None or "fetch" not in done:
        from app.pipeline import fetch  # noqa: PLC0415

        progress("fetch", 0.0)
        source = fetch.fetch(url, job_dir, cfg=cfg)
        timeline = Timeline(job_id=job_id, source=source, params=params)
        timeline.mark_done("fetch")
        save(timeline, job_dir)
    assert timeline is not None

    # --- 2. asr ---
    checkpoint()
    if "asr" not in timeline.stages_done:
        from app.pipeline import asr  # noqa: PLC0415

        progress("asr", 0.0)
        words_only = asr.transcribe(
            job_dir, timeline.source, job_id=job_id, cfg=cfg, language=language
        )
        timeline.words = words_only.words
        timeline.source = words_only.source
        timeline.mark_done("asr")
        save(timeline, job_dir)

    # --- 3. segment ---
    checkpoint()
    if "segment" not in timeline.stages_done:
        from app.pipeline import segment as segment_stage  # noqa: PLC0415
        from app.pipeline import translate as translate_stage  # noqa: PLC0415

        progress("segment", 0.0)
        detector = _load_detector(cfg, timeline)
        arbiter = translate_stage.make_arbiter(cfg=cfg) if params.llm_arbitration else None
        segment_stage.segment_timeline(timeline, detector=detector, arbiter=arbiter)
        timeline.mark_done("segment")
        save(timeline, job_dir)

    # --- 4. translate ---
    checkpoint()
    if "translate" not in timeline.stages_done:
        from app.pipeline import translate as translate_stage  # noqa: PLC0415

        translate_stage.translate_timeline(
            timeline, cfg=cfg, progress=lambda f: progress("translate", f)
        )
        save(timeline, job_dir)

    # --- 5. tts ---
    checkpoint()
    if "tts" not in timeline.stages_done:
        from app.pipeline import tts as tts_stage  # noqa: PLC0415

        tts_stage.synthesize_timeline(
            timeline, job_dir, cfg=cfg, progress=lambda f: progress("tts", f)
        )
        tts_stage.prune_cache(job_dir, timeline)
        save(timeline, job_dir)

    checkpoint()

    # --- 6 & 7. plan + render (pure, always re-run — they are cheap) ---
    from app.pipeline import plan as plan_stage  # noqa: PLC0415
    from app.pipeline import render_audio  # noqa: PLC0415

    progress("render", 0.0)
    source_audio, sr = io.read_wav(job_dir / timeline.source.wav_path)
    render_plan = plan_stage.build_plan(timeline, source_audio, sr)
    (job_dir / "plan.json").write_text(render_plan.model_dump_json(indent=2), encoding="utf-8")

    mp3 = render_audio.render(timeline, render_plan, job_dir, source=source_audio)
    timeline.mark_done("render")
    save(timeline, job_dir)
    progress("render", 1.0)
    return timeline, mp3


def _load_detector(cfg: Settings, timeline: Timeline):
    """SaT if available, punctuation+pause otherwise.

    Degrading rather than failing keeps the pipeline runnable on a machine without the GPU
    extras. `segment_timeline` records a warning when the fallback is used for a language
    whose punctuation Whisper tends to omit.
    """
    try:
        from app.pipeline.sbd import SaTDetector  # noqa: PLC0415

        detector = SaTDetector(cfg.sat_model, device=cfg.device)
        detector._load()  # surface a missing dependency now, not mid-job
        return detector
    except Exception as exc:  # noqa: BLE001
        timeline.warn("segment", f"Textual boundary model unavailable ({exc}); using fallback.")
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--job-id")
    parser.add_argument("--target", default="en", help="bilingual target language")
    parser.add_argument("--language", help="override source language detection")
    parser.add_argument("--mode", choices=["sentence", "words"], default="sentence")
    parser.add_argument("--chunk-words", type=int, default=12)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pre-gap", type=float, default=0.25)
    parser.add_argument("--post-gap", type=float, default=0.35)
    parser.add_argument("--no-arbitration", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    params = RenderParams(
        target_lang=args.target,
        segmentation_mode=args.mode,
        max_words_per_chunk=args.chunk_words,
        tts_speed=args.speed,
        pre_gap=args.pre_gap,
        post_gap=args.post_gap,
        llm_arbitration=not args.no_arbitration,
    )

    def report(stage: str, fraction: float) -> None:
        print(f"  [{stage:<9}] {fraction * 100:5.1f}%", file=sys.stderr, flush=True)

    timeline, mp3 = run(
        args.url,
        job_id=args.job_id,
        params=params,
        progress=report,
        resume=not args.no_resume,
        language=args.language,
    )

    print(json.dumps({
        "job_id": timeline.job_id,
        "title": timeline.source.title,
        "language": timeline.source.detected_lang,
        "segments": len(timeline.segments),
        "translated": timeline.speech_segment_count,
        "warnings": len(timeline.warnings),
        "mp3": str(mp3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
