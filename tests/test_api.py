"""HTTP layer, including the live preview.

The preview is the load-bearing test here: it must produce audio without touching the GPU,
and it must respond to knob changes. That is the whole justification for keeping stage 6
pure and separate from stage 7.
"""

from __future__ import annotations

import io as _io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import db, main
from app.audio import io as audio_io
from app.config import Settings
from app.jobs import store
from app.models.timeline import RenderParams, SourceInfo, Timeline, Word
from app.pipeline import preview as preview_stage
from app.pipeline.runner import save
from app.pipeline.segment import segment_timeline

SR = 24000


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg = Settings(data_dir=tmp_path, db_path=tmp_path / "jobs.sqlite3", device="cpu")
    monkeypatch.setattr(main, "settings", cfg)

    db.init(cfg).close()
    preview_stage._cache.invalidate()

    # Mirror production: a fresh connection per request, in that request's own thread.
    # SQLite connections are not portable across threads and the ASGI server uses a pool.
    def override():
        conn = db.connect(cfg)
        try:
            yield conn
        finally:
            conn.close()

    main.app.dependency_overrides[main.get_conn] = override

    with TestClient(main.app) as c:
        c.cfg = cfg              # type: ignore[attr-defined]
        c.conn = db.connect(cfg)  # separate connection, for assertions in the test thread
        yield c
        c.conn.close()

    main.app.dependency_overrides.clear()


def build_finished_job(client, *, job_id: str = "job1") -> str:
    """A job with a real timeline, real source audio and real TTS clips on disk."""
    job_dir = client.cfg.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    tokens = [
        ("Привет,", 1.00, 1.55), ("меня", 1.58, 1.88), ("зовут", 1.90, 2.30), ("Анна.", 2.32, 2.85),
        ("Сегодня", 4.00, 4.60), ("мы", 4.62, 4.76), ("готовим.", 4.78, 5.40),
        ("Ты", 6.60, 6.85), ("готов?", 6.87, 7.45),
    ]
    words = [Word(text=t, start=s, end=e) for t, s, e in tokens]

    total = words[-1].end + 1.0
    audio = np.zeros(int(total * SR), dtype=np.float32)
    t = np.arange(len(audio)) / SR
    for w in words:
        m = (t >= w.start) & (t < w.end)
        audio[m] += (0.35 * np.sin(2 * np.pi * 200 * t[m])).astype(np.float32)
    audio_io.write_wav(job_dir / "source.wav", audio, SR)

    tl = Timeline(
        job_id=job_id,
        source=SourceInfo(url="https://example.test/ru", title="Тест", detected_lang="ru",
                          duration=total, wav_path="source.wav"),
        params=RenderParams(target_lang="en"),
        words=words,
    )
    segment_timeline(tl)

    (job_dir / "tts").mkdir(exist_ok=True)
    for i, seg in enumerate(s for s in tl.segments if s.kind == "speech"):
        seg.translation = f"translation {i}"
        clip = np.sin(2 * np.pi * 440 * np.arange(int(SR * 1.0)) / SR).astype(np.float32) * 0.3
        audio_io.write_wav(job_dir / "tts" / f"c{i}.wav", clip, SR)
        from app.models.timeline import TTSClip

        seg.tts = TTSClip(path=f"tts/c{i}.wav", duration=1.0, voice="af_heart",
                          lang="en", text_sha=f"c{i}")
    tl.mark_done("tts")
    save(tl, job_dir)

    store.create(client.conn, "https://example.test/ru", tl.params, job_id=job_id)
    store.finish(client.conn, job_id, "done")
    return job_id


# --- pages -----------------------------------------------------------------------------


def test_index_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Bilingual Audio" in res.text


def test_submitting_a_url_creates_a_queued_job(client):
    res = client.post("/jobs", data={"url": "https://example.test/v"}, follow_redirects=False)

    assert res.status_code == 303
    jobs = store.list_recent(client.conn)
    assert len(jobs) == 1 and jobs[0].status == "queued"


def test_submitted_knobs_reach_the_job(client):
    client.post("/jobs", data={
        "url": "https://example.test/v", "target_lang": "en",
        "segmentation_mode": "words", "max_words_per_chunk": "7", "tts_speed": "1.25",
    }, follow_redirects=False)

    params = store.list_recent(client.conn)[0].params
    assert params.segmentation_mode == "words"
    assert params.max_words_per_chunk == 7
    assert params.tts_speed == 1.25


def test_job_page_renders_sentences(client):
    job_id = build_finished_job(client)
    res = client.get(f"/jobs/{job_id}")

    assert res.status_code == 200
    assert "Привет, меня зовут Анна." in res.text
    assert "translation 0" in res.text


def test_unknown_job_is_404(client):
    assert client.get("/jobs/nope").status_code == 404


def test_status_fragment_stops_polling_once_terminal(client):
    job_id = build_finished_job(client)
    assert 'data-poll="0"' in client.get(f"/jobs/{job_id}/status").text

    queued = store.create(client.conn, "u", RenderParams()).id
    assert 'data-poll="1"' in client.get(f"/jobs/{queued}/status").text


# --- api -------------------------------------------------------------------------------


def test_api_submit_and_fetch(client):
    created = client.post("/api/jobs", json={"url": "https://example.test/v",
                                             "params": {"tts_speed": 1.5}}).json()
    fetched = client.get(f"/api/jobs/{created['id']}").json()

    assert fetched["status"] == "queued"
    assert fetched["params"]["tts_speed"] == 1.5


def test_api_submit_requires_a_url(client):
    assert client.post("/api/jobs", json={}).status_code == 400


def test_cancel_then_retry(client):
    job_id = client.post("/api/jobs", json={"url": "u"}).json()["id"]

    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled"
    assert client.post(f"/api/jobs/{job_id}/retry").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "queued"


def test_cancelling_a_finished_job_conflicts(client):
    job_id = build_finished_job(client)
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["ok"] is True


# --- preview ---------------------------------------------------------------------------


def _wav_duration(payload: bytes) -> float:
    data, sr = sf.read(_io.BytesIO(payload), dtype="float32")
    return len(data) / sr


def test_preview_returns_playable_audio(client):
    job_id = build_finished_job(client)
    res = client.get(f"/api/jobs/{job_id}/preview", params={"segment": 0})

    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert _wav_duration(res.content) > 0.5


def test_preview_is_an_excerpt_not_the_whole_file(client):
    """A three-sentence window is what makes this fast enough to drag a slider against."""
    job_id = build_finished_job(client)
    res = client.get(f"/api/jobs/{job_id}/preview", params={"segment": 0})

    assert _wav_duration(res.content) < 12.0


def test_longer_gaps_produce_a_longer_preview(client):
    """The knob must actually change the audio, not just the response."""
    job_id = build_finished_job(client)
    tight = client.get(f"/api/jobs/{job_id}/preview",
                       params={"segment": 1, "pre_gap": 0.0, "post_gap": 0.0})
    roomy = client.get(f"/api/jobs/{job_id}/preview",
                       params={"segment": 1, "pre_gap": 1.0, "post_gap": 1.0})

    assert _wav_duration(roomy.content) > _wav_duration(tight.content) + 1.0


def test_faster_speech_produces_a_shorter_preview(client):
    job_id = build_finished_job(client)
    slow = client.get(f"/api/jobs/{job_id}/preview", params={"segment": 1, "tts_speed": 0.8})
    fast = client.get(f"/api/jobs/{job_id}/preview", params={"segment": 1, "tts_speed": 1.8})

    assert _wav_duration(fast.content) < _wav_duration(slow.content)


def test_preview_before_transcription_is_404(client):
    job_id = store.create(client.conn, "u", RenderParams()).id
    assert client.get(f"/api/jobs/{job_id}/preview").status_code == 404


def test_preview_segment_index_is_clamped(client):
    """Dragging the sentence slider to the end must not 500."""
    job_id = build_finished_job(client)
    assert client.get(f"/api/jobs/{job_id}/preview", params={"segment": 9999}).status_code == 200


# --- rerender --------------------------------------------------------------------------


def test_pacing_change_rerenders_immediately_without_requeueing(client):
    """The point of separating stage 6 from stage 7: no GPU, no queue, no wait."""
    job_id = build_finished_job(client)
    res = client.post(f"/api/jobs/{job_id}/rerender",
                      json={"params": {"tts_speed": 1.4, "pre_gap": 0.5}})

    body = res.json()
    assert body["requeued"] is False
    assert body["duration"] > 0
    assert (client.cfg.job_dir(job_id) / "out.mp3").exists()
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "done"


def test_segmentation_change_requeues_from_the_segment_stage(client):
    job_id = build_finished_job(client)
    res = client.post(f"/api/jobs/{job_id}/rerender",
                      json={"params": {"segmentation_mode": "words"}})

    body = res.json()
    assert body["requeued"] is True
    assert body["from_stage"] == "segment"
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "queued"


def test_target_language_change_requeues_from_translate(client):
    job_id = build_finished_job(client)
    body = client.post(f"/api/jobs/{job_id}/rerender",
                       json={"params": {"target_lang": "de"}}).json()

    assert body["from_stage"] == "translate"


# --- downloads -------------------------------------------------------------------------


def test_mp3_download_uses_the_video_title_as_filename(client):
    job_id = build_finished_job(client)
    client.post(f"/api/jobs/{job_id}/rerender", json={"params": {"tts_speed": 1.1}})

    res = client.get(f"/files/{job_id}/out.mp3")
    assert res.status_code == 200
    assert "filename" in res.headers.get("content-disposition", "")


def test_arbitrary_paths_are_not_served(client):
    """The filename is a fixed allowlist, so traversal has nothing to grab."""
    job_id = build_finished_job(client)
    for name in ("../../etc/passwd", "source.wav", "..%2Ftimeline.json"):
        assert client.get(f"/files/{job_id}/{name}").status_code == 404


def test_downloading_before_render_is_404(client):
    job_id = build_finished_job(client)
    assert client.get(f"/files/{job_id}/out.mp3").status_code == 404
