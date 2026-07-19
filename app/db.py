"""SQLite connection handling.

WAL mode so the API process can read while the worker writes. Both processes open their own
connection; there is no pool because there is no concurrency worth pooling for.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.config import Settings, settings

SCHEMA_PATH = Path(__file__).parent / "jobs" / "schema.sql"


def connect(cfg: Settings | None = None) -> sqlite3.Connection:
    cfg = cfg or settings
    Path(cfg.db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(cfg.db_path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    """Apply the schema. Idempotent — every statement is CREATE ... IF NOT EXISTS."""
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def init(cfg: Settings | None = None) -> sqlite3.Connection:
    conn = connect(cfg)
    migrate(conn)
    return conn
