"""Job queue.

The properties that matter for a single-worker home server: a job is claimed exactly once,
progress survives a restart, cancellation is honoured, and a dead worker's job comes back
rather than being lost.
"""

from __future__ import annotations

import time

import pytest

from app import db
from app.config import Settings
from app.jobs import store
from app.models.timeline import RenderParams


@pytest.fixture
def conn(tmp_path):
    connection = db.init(Settings(db_path=tmp_path / "jobs.sqlite3", data_dir=tmp_path))
    yield connection
    connection.close()


def test_created_job_starts_queued(conn):
    job = store.create(conn, "https://example.test/a", RenderParams())

    assert job.status == "queued"
    assert job.progress == 0.0
    assert not job.is_terminal


def test_params_round_trip_through_the_database(conn):
    params = RenderParams(target_lang="en", tts_speed=1.3, segmentation_mode="words", min_words=5)
    job_id = store.create(conn, "u", params).id

    loaded = store.get(conn, job_id)
    assert loaded.params.tts_speed == 1.3
    assert loaded.params.segmentation_mode == "words"
    assert loaded.params.min_words == 5


def test_claim_takes_the_oldest_job_first(conn):
    first = store.create(conn, "first", RenderParams())
    time.sleep(0.01)
    store.create(conn, "second", RenderParams())

    assert store.claim_next(conn).id == first.id


def test_a_job_is_claimed_exactly_once(conn):
    """Guards against a second worker started by accident picking up the same job."""
    store.create(conn, "only", RenderParams())

    assert store.claim_next(conn) is not None
    assert store.claim_next(conn) is None


def test_claim_on_an_empty_queue_returns_none(conn):
    assert store.claim_next(conn) is None


def test_heartbeat_records_stage_and_progress(conn):
    job_id = store.create(conn, "u", RenderParams()).id
    store.claim_next(conn)
    store.heartbeat(conn, job_id, stage="asr", progress=0.42)

    loaded = store.get(conn, job_id)
    assert loaded.stage == "asr"
    assert loaded.progress == pytest.approx(0.42)
    assert loaded.heartbeat_at is not None


def test_progress_is_clamped(conn):
    job_id = store.create(conn, "u", RenderParams()).id
    store.heartbeat(conn, job_id, stage="asr", progress=3.7)
    assert store.get(conn, job_id).progress == 1.0


def test_finishing_marks_terminal_and_records_warnings(conn):
    job_id = store.create(conn, "u", RenderParams()).id
    store.claim_next(conn)
    store.finish(conn, job_id, "done", warnings=3)

    loaded = store.get(conn, job_id)
    assert loaded.is_terminal
    assert loaded.warnings == 3
    assert loaded.finished_at is not None


def test_failure_preserves_the_error_message(conn):
    """yt-dlp's stderr is the whole diagnosis for a geo-block or age gate."""
    job_id = store.create(conn, "u", RenderParams()).id
    store.claim_next(conn)
    store.finish(conn, job_id, "failed", error="ERROR: Video unavailable in your country")

    assert "your country" in store.get(conn, job_id).error


# --- cancellation ----------------------------------------------------------------------


def test_cancelling_a_queued_job_is_immediate(conn):
    """Nothing has started, so there is no worker to cooperate with."""
    job_id = store.create(conn, "u", RenderParams()).id

    assert store.request_cancel(conn, job_id) is True
    assert store.get(conn, job_id).status == "cancelled"


def test_cancelling_a_running_job_sets_a_flag_for_the_worker(conn):
    job_id = store.create(conn, "u", RenderParams()).id
    store.claim_next(conn)

    assert store.request_cancel(conn, job_id) is True
    assert store.get(conn, job_id).status == "running"   # still running until the worker sees it
    assert store.cancel_requested(conn, job_id) is True


def test_cancelling_a_finished_job_is_rejected(conn):
    job_id = store.create(conn, "u", RenderParams()).id
    store.finish(conn, job_id, "done")

    assert store.request_cancel(conn, job_id) is False


# --- crash recovery --------------------------------------------------------------------


def test_stale_running_job_is_requeued(conn):
    """A worker that died leaves its job 'running' forever. Requeuing is safe because every
    stage persisted its output — the job resumes rather than restarting."""
    job_id = store.create(conn, "u", RenderParams()).id
    store.claim_next(conn)
    store.heartbeat(conn, job_id, stage="asr", progress=0.5)

    assert store.reap_stale(conn, stale_after=0.0) == [job_id]

    loaded = store.get(conn, job_id)
    assert loaded.status == "queued"
    assert loaded.stage is None
    assert loaded.cancel_requested is False


def test_a_live_job_is_not_reaped(conn):
    job_id = store.create(conn, "u", RenderParams()).id
    store.claim_next(conn)
    store.heartbeat(conn, job_id, stage="asr", progress=0.1)

    assert store.reap_stale(conn, stale_after=300.0) == []
    assert store.get(conn, job_id).status == "running"


def test_requeue_clears_stale_state(conn):
    job_id = store.create(conn, "u", RenderParams()).id
    store.claim_next(conn)
    store.finish(conn, job_id, "failed", error="boom")
    store.requeue(conn, job_id)

    loaded = store.get(conn, job_id)
    assert loaded.status == "queued"
    assert loaded.error is None
    assert store.claim_next(conn).id == job_id


# --- housekeeping ----------------------------------------------------------------------


def test_purge_removes_only_old_finished_jobs(conn):
    old = store.create(conn, "old", RenderParams()).id
    store.finish(conn, old, "done")
    conn.execute("UPDATE jobs SET created_at = ? WHERE id = ?", (time.time() - 40 * 86400, old))

    recent = store.create(conn, "recent", RenderParams()).id
    store.finish(conn, recent, "done")
    running = store.create(conn, "running", RenderParams()).id
    store.claim_next(conn)
    conn.execute("UPDATE jobs SET created_at = ? WHERE id = ?", (time.time() - 99 * 86400, running))

    assert store.purge_older_than(conn, days=7) == [old]
    assert store.get(conn, recent) is not None
    assert store.get(conn, running) is not None, "an unfinished job must never be purged"


def test_stats_counts_by_status(conn):
    store.create(conn, "a", RenderParams())
    store.finish(conn, store.create(conn, "b", RenderParams()).id, "done")

    assert store.stats(conn) == {"queued": 1, "done": 1}


def test_list_recent_is_newest_first(conn):
    store.create(conn, "first", RenderParams())
    time.sleep(0.01)
    store.create(conn, "second", RenderParams())

    assert [j.url for j in store.list_recent(conn)] == ["second", "first"]
