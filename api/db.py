"""SQLite connections for the API store (docs/api_design.md §7). The SQL lives in api/repository.py.

One short-lived connection per operation: sqlite3 connections must not cross threads, and the
repository runs in worker threads.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(path: Path) -> sqlite3.Connection:
    """Open the database file, creating its folder if needed. Foreign keys are enforced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
