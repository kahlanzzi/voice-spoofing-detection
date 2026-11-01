from api.db import connect


def test_connect_creates_the_folder_and_enforces_foreign_keys(tmp_path):
    path = tmp_path / "new" / "dir" / "api.sqlite"
    conn = connect(path)
    try:
        assert path.parent.is_dir()
        assert conn.execute("PRAGMA foreign_keys").fetchone() == (1,)
    finally:
        conn.close()
