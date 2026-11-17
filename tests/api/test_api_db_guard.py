"""The skipped-tests guard (tests/api/conftest.py): without a PostgreSQL server the database tests are
skipped and the run ends with a WARNING line; with REQUIRE_POSTGRES_TESTS=1 they fail instead (CI).
Each case runs pytest in a subprocess on a port where nothing listens, so it needs no server."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UNREACHABLE = "postgresql://postgres@127.0.0.1:1/postgres"  # port 1: nothing listens
TARGET = "tests/api/test_api_health.py"  # two tests, both need the database
WARNING = ("WARNING: 2 database tests skipped: no PostgreSQL server "
           "(set REQUIRE_POSTGRES_TESTS=1 to fail instead)")


def run_pytest(require: bool) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("REQUIRE_POSTGRES_TESTS", "PYTEST_ADDOPTS")}
    env["VSD_TEST_DATABASE_URL"] = UNREACHABLE
    if require:
        env["REQUIRE_POSTGRES_TESTS"] = "1"
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", TARGET],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=120)


def test_unreachable_server_skips_and_ends_with_the_warning_line():
    result = run_pytest(require=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 skipped" in result.stdout
    assert result.stdout.rstrip().splitlines()[-1] == WARNING  # the last line of the run


def test_require_postgres_tests_turns_the_skip_into_a_failure():
    result = run_pytest(require=True)
    assert result.returncode != 0
    assert "REQUIRE_POSTGRES_TESTS=1 makes this a failure" in result.stdout
    assert WARNING not in result.stdout
