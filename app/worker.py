"""The job worker. Run as its own process:

    python -m app.worker

Deliberately separate from uvicorn. CUDA inside a forked or threaded ASGI worker is a
reliable source of trouble, and this way restarting the API never kills a running job.
Concurrency of one is a property of running one of these.
"""

from __future__ import annotations

import logging
import signal
import sqlite3
import sys
import time
from types import FrameType

from app import db
from app.config import Settings, settings
from app.jobs import store
from app.pipeline import runner

log = logging.getLogger("worker")

POLL_INTERVAL = 1.0
IDLE_MAINTENANCE_INTERVAL = 3600.0


class Worker:
    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings
        self.conn: sqlite3.Connection = db.init(self.cfg)
        self.running = True
        self._last_maintenance = 0.0

    def stop(self, signum: int, frame: FrameType | None) -> None:
        # Finish the current stage rather than dying mid-write; the job resumes either way.
        log.info("signal %s received, finishing current stage then exiting", signum)
        self.running = False

    def run_forever(self) -> None:
        # Anything left 'running' belongs to a previous process that died.
        for job_id in store.reap_stale(self.conn, stale_after=0.0):
            log.warning("requeued orphaned job %s", job_id)

        log.info("worker ready (device=%s, whisper=%s)", self.cfg.device, self.cfg.whisper_model)
        while self.running:
            job = store.claim_next(self.conn)
            if job is None:
                self._maintenance()
                time.sleep(POLL_INTERVAL)
                continue
            self.run_job(job)

    def run_job(self, job: store.Job) -> None:
        log.info("job %s starting: %s", job.id, job.url)

        def progress(stage: str, fraction: float) -> None:
            store.heartbeat(self.conn, job.id, stage=stage, progress=fraction)

        def should_cancel() -> bool:
            return store.cancel_requested(self.conn, job.id) or not self.running

        try:
            timeline, mp3 = runner.run(
                job.url,
                job_id=job.id,
                params=job.params,
                cfg=self.cfg,
                progress=progress,
                language=job.language,
                should_cancel=should_cancel,
            )
        except runner.Cancelled:
            log.info("job %s cancelled", job.id)
            store.finish(self.conn, job.id, "cancelled")
            return
        except Exception as exc:  # noqa: BLE001 — a failed job must not take down the worker
            log.exception("job %s failed", job.id)
            store.finish(self.conn, job.id, "failed", error=_readable(exc))
            return

        store.set_title(self.conn, job.id, timeline.source.title)
        store.finish(self.conn, job.id, "done", warnings=len(timeline.warnings))
        log.info("job %s done -> %s", job.id, mp3)

    def _maintenance(self) -> None:
        now = time.time()
        if now - self._last_maintenance < IDLE_MAINTENANCE_INTERVAL:
            return
        self._last_maintenance = now
        for job_id in store.purge_older_than(self.conn, self.cfg.job_retention_days):
            log.info("purged expired job %s", job_id)


def _readable(exc: Exception) -> str:
    """Keep the message useful.

    yt-dlp's stderr in particular is the whole diagnosis for geo-blocks, age gates and
    signature failures — a generic 'download failed' throws away the one thing worth knowing.
    """
    text = str(exc).strip() or exc.__class__.__name__
    return text if len(text) <= 2000 else text[:2000] + "…"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    worker = Worker()
    signal.signal(signal.SIGINT, worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)
    worker.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
