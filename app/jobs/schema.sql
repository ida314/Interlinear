-- One table. A single-user home server running one GPU job at a time does not need a broker,
-- a result backend, or a second daemon; it needs durability across restarts, live progress,
-- and cancellation. SQLite in WAL mode provides all three.

CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    url              TEXT    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'queued',  -- queued|running|done|failed|cancelled
    stage            TEXT,                               -- current pipeline stage
    progress         REAL    NOT NULL DEFAULT 0.0,       -- 0..1 within the stage
    params           TEXT    NOT NULL DEFAULT '{}',      -- RenderParams as JSON
    language         TEXT,                               -- source language override, if any
    title            TEXT,
    error            TEXT,
    warnings         INTEGER NOT NULL DEFAULT 0,
    created_at       REAL    NOT NULL,
    started_at       REAL,
    finished_at      REAL,
    heartbeat_at     REAL,                               -- worker liveness; stale => crashed
    cancel_requested INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_jobs_queued  ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
