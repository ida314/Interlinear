"""Job queue operations. All SQL lives here.

Concurrency of one is a property of running exactly one worker process, not of a lock. The
claim below is still atomic, so a second worker started by accident cannot pick up the same
job.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from app.models.timeline import RenderParams

Status = Literal["queued", "running", "done", "failed", "cancelled"]
TERMINAL: set[str] = {"done", "failed", "cancelled"}

# A worker that has not written a heartbeat in this long is presumed dead and its job is
# requeued. Generous, because ASR on a long video does not check in mid-pass.
STALE_AFTER = 180.0


@dataclass
class Job:
    id: str
    url: str
    status: str
    stage: str | None
    progress: float
    params: RenderParams
    language: str | None
    title: str | None
    error: str | None
    warnings: int
    created_at: float
    started_at: float | None
    finished_at: float | None
    heartbeat_at: float | None
    cancel_requested: bool

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Job:
        return cls(
            id=row["id"],
            url=row["url"],
            status=row["status"],
            stage=row["stage"],
            progress=row["progress"],
            params=RenderParams.model_validate_json(row["params"] or "{}"),
            language=row["language"],
            title=row["title"],
            error=row["error"],
            warnings=row["warnings"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            heartbeat_at=row["heartbeat_at"],
            cancel_requested=bool(row["cancel_requested"]),
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.finished_at or time.time()) - self.started_at

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "title": self.title,
            "error": self.error,
            "warnings": self.warnings,
            "elapsed": round(self.elapsed, 1),
            "params": self.params.model_dump(),
        }


def create(
    conn: sqlite3.Connection,
    url: str,
    params: RenderParams,
    *,
    language: str | None = None,
    job_id: str | None = None,
) -> Job:
    job_id = job_id or uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO jobs (id, url, status, params, language, created_at) "
        "VALUES (?, ?, 'queued', ?, ?, ?)",
        (job_id, url, params.model_dump_json(), language, time.time()),
    )
    job = get(conn, job_id)
    assert job is not None
    return job


def get(conn: sqlite3.Connection, job_id: str) -> Job | None:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return Job.from_row(row) if row else None


def list_recent(conn: sqlite3.Connection, limit: int = 50) -> list[Job]:
    rows = conn.execute(
        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [Job.from_row(r) for r in rows]


def claim_next(conn: sqlite3.Connection) -> Job | None:
    """Atomically take the oldest queued job. Returns None when the queue is empty."""
    now = time.time()
    row = conn.execute(
        """
        UPDATE jobs SET status = 'running', started_at = ?, heartbeat_at = ?, error = NULL
        WHERE id = (
            SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1
        )
        RETURNING *
        """,
        (now, now),
    ).fetchone()
    return Job.from_row(row) if row else None


def heartbeat(conn: sqlite3.Connection, job_id: str, *, stage: str, progress: float) -> None:
    conn.execute(
        "UPDATE jobs SET stage = ?, progress = ?, heartbeat_at = ? WHERE id = ?",
        (stage, max(0.0, min(1.0, progress)), time.time(), job_id),
    )


def set_title(conn: sqlite3.Connection, job_id: str, title: str) -> None:
    conn.execute("UPDATE jobs SET title = ? WHERE id = ?", (title, job_id))


def finish(
    conn: sqlite3.Connection,
    job_id: str,
    status: Status,
    *,
    error: str | None = None,
    warnings: int = 0,
) -> None:
    conn.execute(
        "UPDATE jobs SET status = ?, error = ?, warnings = ?, finished_at = ?, progress = ? "
        "WHERE id = ?",
        (status, error, warnings, time.time(), 1.0 if status == "done" else 0.0, job_id),
    )


def request_cancel(conn: sqlite3.Connection, job_id: str) -> bool:
    """Cooperative — the worker notices between stages. A queued job is cancelled outright."""
    job = get(conn, job_id)
    if job is None or job.is_terminal:
        return False
    if job.status == "queued":
        finish(conn, job_id, "cancelled")
        return True
    conn.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,))
    return True


def cancel_requested(conn: sqlite3.Connection, job_id: str) -> bool:
    row = conn.execute("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return bool(row and row["cancel_requested"])


def update_params(conn: sqlite3.Connection, job_id: str, params: RenderParams) -> None:
    conn.execute(
        "UPDATE jobs SET params = ? WHERE id = ?", (params.model_dump_json(), job_id)
    )


def requeue(conn: sqlite3.Connection, job_id: str) -> None:
    conn.execute(
        "UPDATE jobs SET status = 'queued', stage = NULL, progress = 0.0, "
        "started_at = NULL, heartbeat_at = NULL, cancel_requested = 0, error = NULL "
        "WHERE id = ?",
        (job_id,),
    )


def reap_stale(conn: sqlite3.Connection, *, stale_after: float = STALE_AFTER) -> list[str]:
    """Requeue jobs whose worker died.

    Safe because every stage persisted `timeline.json` before returning, so a requeued job
    resumes from the last completed stage rather than re-running ASR.
    """
    cutoff = time.time() - stale_after
    rows = conn.execute(
        "SELECT id FROM jobs WHERE status = 'running' AND COALESCE(heartbeat_at, 0) < ?",
        (cutoff,),
    ).fetchall()
    ids = [r["id"] for r in rows]
    for job_id in ids:
        requeue(conn, job_id)
    return ids


def purge_older_than(conn: sqlite3.Connection, days: float) -> list[str]:
    cutoff = time.time() - days * 86400
    rows = conn.execute(
        "SELECT id FROM jobs WHERE status IN ('done','failed','cancelled') AND created_at < ?",
        (cutoff,),
    ).fetchall()
    ids = [r["id"] for r in rows]
    if ids:
        conn.executemany("DELETE FROM jobs WHERE id = ?", [(i,) for i in ids])
    return ids


def stats(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
    return {r["status"]: r["n"] for r in rows}
