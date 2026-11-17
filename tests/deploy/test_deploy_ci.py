"""CI (.github/workflows/ci.yml) and the PostgreSQL store (P2a, D53 (4)): the checks job runs the API tests
against a PostgreSQL service container with the same image as docker-compose.yml and makes an unreachable
server a failure; the e2e job makes and masks a per-run value for every secret the stack needs."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
COMPOSE = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
CHECKS = WORKFLOW["jobs"]["checks"]
E2E = WORKFLOW["jobs"]["e2e"]


def step(job: dict, name_start: str) -> dict:
    (found,) = [s for s in job["steps"] if s.get("name", "").startswith(name_start)]
    return found


def test_checks_job_has_a_postgresql_service_with_the_compose_image():
    service = CHECKS["services"]["postgres"]
    assert service["image"] == COMPOSE["services"]["db"]["image"] == COMPOSE["x-vsd-postgres-image"]
    assert "--locale=C" in service["env"]["POSTGRES_INITDB_ARGS"]
    assert service["env"]["POSTGRES_HOST_AUTH_METHOD"] == "trust"  # throwaway test server, no secret
    assert service["ports"] == ["55432:5432"]
    assert "--health-cmd" in service["options"]


def test_checks_job_runs_the_database_tests_and_never_skips_them():
    env = step(CHECKS, "Run tests")["env"]
    assert env["REQUIRE_POSTGRES_TESTS"] == "1"
    assert env["VSD_TEST_DATABASE_URL"] == "postgresql://postgres@127.0.0.1:55432/postgres"
    assert step(CHECKS, "Run tests")["run"] == "pytest -q"


def test_e2e_job_makes_and_masks_every_stack_secret_per_run():
    run = step(E2E, "Make a per-run sign-in key")["run"]
    for name in ("VSD_SIGNIN_HMAC_KEY", "VSD_DB_PASSWORD", "VSD_DB_APP_PASSWORD"):
        assert name in run
    assert "openssl rand -hex 32" in run and '::add-mask::$k' in run
    assert run.index("::add-mask::") < run.index("GITHUB_ENV")  # masked before it is exported
    assert step(E2E, "Stop the stack")["run"] == "docker compose down -v"
