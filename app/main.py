"""FastAPI app: submit jobs, watch progress, tune knobs, download results.

Server-rendered Jinja with a few lines of vanilla JS rather than an SPA. Three screens for
one user does not justify a Node toolchain or a client framework, and polling a status row
every two seconds is simpler and more robust than a websocket.

The API process never touches the GPU — it only reads job rows and re-renders audio, both
cheap. All model work happens in the worker process.

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from app import db
from app.config import settings
from app.jobs import store
from app.models.timeline import RenderParams
from app.pipeline import preview as preview_stage
from app.pipeline import runner
from app.pipeline.plan import build_plan
from app.pipeline.render_audio import render
from app.pipeline.tts import KOKORO_VOICES

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Schema is applied once here, not per request. Each request then opens its own
    # connection — SQLite connections belong to the thread that created them, and the ASGI
    # server serves requests from a pool.
    conn = db.init(settings)
    conn.close()
    yield


app = FastAPI(title="Bilingual Audio Generator", lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "web" / "templates"))


def get_conn() -> sqlite3.Connection:
    conn = db.connect(settings)
    try:
        yield conn
    finally:
        conn.close()


def _job_or_404(conn: sqlite3.Connection, job_id: str) -> store.Job:
    job = store.get(conn, job_id)
    if job is None:
        raise HTTPException(404, f"No such job: {job_id}")
    return job


def _timeline_or_404(job_id: str):
    timeline = runner.load(settings.job_dir(job_id))
    if timeline is None:
        raise HTTPException(404, "Job has not produced a timeline yet")
    return timeline


# --- pages -----------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(request: Request, conn: sqlite3.Connection = Depends(get_conn)):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "jobs": store.list_recent(conn, limit=25),
            "defaults": RenderParams(),
            "voices": sorted(KOKORO_VOICES),
        },
    )


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str, request: Request, conn: sqlite3.Connection = Depends(get_conn)):
    job = _job_or_404(conn, job_id)
    timeline = runner.load(settings.job_dir(job_id))
    return templates.TemplateResponse(
        request,
        "job.html",
        {
            "job": job,
            "timeline": timeline,
            "segments": (timeline.segments[:200] if timeline else []),
            "has_output": (settings.job_dir(job_id) / "out.mp3").exists(),
        },
    )


@app.get("/jobs/{job_id}/status", response_class=HTMLResponse)
def job_status_fragment(
    job_id: str, request: Request, conn: sqlite3.Connection = Depends(get_conn)
):
    """Polled every 2s by the job page and swapped in. A fragment, not JSON — the server
    already knows how to render this and the client should not have to."""
    job = _job_or_404(conn, job_id)
    return templates.TemplateResponse(
        request,
        "_status.html",
        {"job": job, "has_output": (settings.job_dir(job_id) / "out.mp3").exists()},
    )


# --- job lifecycle ---------------------------------------------------------------------


@app.post("/jobs")
def submit(
    request: Request,
    url: str = Form(...),
    target_lang: str = Form("en"),
    language: str = Form(""),
    segmentation_mode: str = Form("sentence"),
    max_words_per_chunk: int = Form(12),
    tts_speed: float = Form(1.0),
    pre_gap: float = Form(0.25),
    post_gap: float = Form(0.35),
    voice: str = Form("af_heart"),
    conn: sqlite3.Connection = Depends(get_conn),
):
    params = RenderParams(
        target_lang=target_lang,
        segmentation_mode=segmentation_mode,  # type: ignore[arg-type]
        max_words_per_chunk=max_words_per_chunk,
        tts_speed=tts_speed,
        pre_gap=pre_gap,
        post_gap=post_gap,
        voice=voice,
    )
    job = store.create(conn, url.strip(), params, language=(language.strip() or None))
    return Response(status_code=303, headers={"Location": f"/jobs/{job.id}"})


@app.post("/api/jobs")
def api_submit(payload: dict, conn: sqlite3.Connection = Depends(get_conn)):
    url = (payload.get("url") or "").strip()
    if not url:
        raise HTTPException(400, "url is required")
    params = RenderParams(**(payload.get("params") or {}))
    job = store.create(conn, url, params, language=payload.get("language"))
    return job.to_public()


@app.get("/api/jobs")
def api_list(conn: sqlite3.Connection = Depends(get_conn)):
    return [j.to_public() for j in store.list_recent(conn)]


@app.get("/api/jobs/{job_id}")
def api_get(job_id: str, conn: sqlite3.Connection = Depends(get_conn)):
    return _job_or_404(conn, job_id).to_public()


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel(job_id: str, conn: sqlite3.Connection = Depends(get_conn)):
    if not store.request_cancel(conn, job_id):
        raise HTTPException(409, "Job is already finished")
    return {"ok": True}


@app.post("/api/jobs/{job_id}/retry")
def api_retry(job_id: str, conn: sqlite3.Connection = Depends(get_conn)):
    """Requeue a failed job. It resumes from the last completed stage — a download or ASR
    pass that already succeeded is not repeated."""
    job = _job_or_404(conn, job_id)
    if not job.is_terminal:
        raise HTTPException(409, "Job is still running")
    store.requeue(conn, job_id)
    return {"ok": True}


# --- knobs -----------------------------------------------------------------------------


@app.get("/api/jobs/{job_id}/preview")
def api_preview(
    job_id: str,
    segment: int = 0,
    tts_speed: float | None = None,
    pre_gap: float | None = None,
    post_gap: float | None = None,
    head_pad: float | None = None,
    tail_pad: float | None = None,
    max_source_gap: float | None = None,
):
    """Render a short excerpt with the given pacing knobs. Pure CPU, milliseconds."""
    timeline = _timeline_or_404(job_id)
    overrides = {
        k: v
        for k, v in {
            "tts_speed": tts_speed, "pre_gap": pre_gap, "post_gap": post_gap,
            "head_pad": head_pad, "tail_pad": tail_pad, "max_source_gap": max_source_gap,
        }.items()
        if v is not None
    }
    params = timeline.params.model_copy(update=overrides)

    try:
        wav = preview_stage.render_excerpt(
            timeline, settings.job_dir(job_id), segment_index=segment, params=params
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/jobs/{job_id}/rerender")
def api_rerender(job_id: str, payload: dict, conn: sqlite3.Connection = Depends(get_conn)):
    """Apply new knobs to a finished job.

    Render-time knobs are applied here and now — no GPU, no queue. Anything that changes
    sentence boundaries, translations or voices is requeued for the worker instead.
    """
    job = _job_or_404(conn, job_id)
    timeline = _timeline_or_404(job_id)
    new_params = timeline.params.model_copy(update=payload.get("params") or {})

    stale = runner.invalidated_from(timeline.params, new_params)
    store.update_params(conn, job_id, new_params)

    if stale is not None:
        timeline.params = new_params
        runner.save(timeline, settings.job_dir(job_id))
        store.requeue(conn, job_id)
        return {"ok": True, "requeued": True, "from_stage": stale}

    if not job.is_terminal:
        raise HTTPException(409, "Job is still running")

    job_dir = settings.job_dir(job_id)
    timeline.params = new_params
    audio, sr = preview_stage._cache.get(job_id, job_dir / timeline.source.wav_path)
    plan = build_plan(timeline, audio, sr, new_params)
    render(timeline, plan, job_dir, source=audio)
    runner.save(timeline, job_dir)

    return {
        "ok": True,
        "requeued": False,
        "duration": round(plan.total_duration, 2),
        "expansion": round(plan.stats.expansion_ratio, 3),
    }


# --- output ----------------------------------------------------------------------------


@app.get("/files/{job_id}/{name}")
def download(job_id: str, name: str):
    if name not in {"out.mp3", "out.lrc", "timeline.json", "plan.json"}:
        raise HTTPException(404, "Not available")
    path = settings.job_dir(job_id) / name
    if not path.exists():
        raise HTTPException(404, "Not generated yet")

    filename = name
    if name == "out.mp3":
        timeline = runner.load(settings.job_dir(job_id))
        if timeline and timeline.source.title:
            safe = "".join(c for c in timeline.source.title if c.isalnum() or c in " -_").strip()
            filename = f"{safe or 'bilingual'}.mp3"
    return FileResponse(path, filename=filename)


@app.get("/healthz")
def healthz(conn: sqlite3.Connection = Depends(get_conn)):
    return JSONResponse({"ok": True, "jobs": store.stats(conn), "device": settings.device})
