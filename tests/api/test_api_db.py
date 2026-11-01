import sqlite3

import pytest

from api.db import connect, connect_readonly


def test_connect_creates_the_folder_and_enforces_foreign_keys(tmp_path):
    path = tmp_path / "new" / "dir" / "api.sqlite"
    conn = connect(path)
    try:
        assert path.parent.is_dir()
        assert conn.execute("PRAGMA foreign_keys").fetchone() == (1,)
    finally:
        conn.close()


def test_connect_readonly_does_not_create_a_missing_file_or_folder(tmp_path):
    path = tmp_path / "missing" / "api.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        connect_readonly(path, timeout_s=0.1).execute("SELECT 1").fetchall()
    assert not path.parent.exists()


def test_connect_readonly_refuses_writes(tmp_path):
    path = tmp_path / "api.sqlite"
    connect(path).close()
    conn = connect_readonly(path, timeout_s=0.1)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("CREATE TABLE t (x INTEGER)")
    finally:
        conn.close()
