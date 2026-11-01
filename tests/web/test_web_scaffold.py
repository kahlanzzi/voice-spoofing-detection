"""web/ toolchain pins (Stage 18 PR 1a), checked by the required `checks` job so it also guards web/.

Exact versions only (no ^, ~, ranges, tags or URLs) in package.json and the lockfile root; Node pinned in
web/.nvmrc, equal to engines.node and read by the CI `web` job; web/.npmrc keeps installs exact and
engine-strict; node_modules/ and web/dist/ never reach git (and so never reach the D20 guard's scan).
Text checks only: CI's `checks` job has no Node.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "web"
EXACT = re.compile(r"\d+\.\d+\.\d+")
NPMRC_LINES = {"save-exact=true", "engine-strict=true"}


def loose_pins(package: dict) -> list[str]:
    """'name: spec' for every dependency or devDependency that is not an exact X.Y.Z version."""
    return [f"{name}: {spec}"
            for group in ("dependencies", "devDependencies")
            for name, spec in package.get(group, {}).items()
            if not EXACT.fullmatch(spec)]


def missing_npmrc_lines(text: str) -> set[str]:
    lines = {line.strip() for line in text.splitlines()}
    return NPMRC_LINES - lines


def web_job_node_file(workflow: dict) -> str | None:
    """The node-version-file of the setup-node step in the `web` job, or None."""
    for step in workflow.get("jobs", {}).get("web", {}).get("steps", []):
        if str(step.get("uses", "")).startswith("actions/setup-node@"):
            return step.get("with", {}).get("node-version-file")
    return None


PACKAGE = json.loads((WEB / "package.json").read_text(encoding="utf-8"))
WORKFLOW = yaml.safe_load((REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))


def test_package_json_pins_are_exact():
    assert PACKAGE["dependencies"] and PACKAGE["devDependencies"]
    assert loose_pins(PACKAGE) == []


@pytest.mark.parametrize("spec", ["^19.3.1", "~19.3.1", ">=19", "19.x", "latest", "*", "19.3", "git+https://x/y"])
def test_loose_pin_is_caught(spec):
    assert loose_pins({"dependencies": {"react": spec}}) == [f"react: {spec}"]


def test_lockfile_root_matches_package_json():
    lock = json.loads((WEB / "package-lock.json").read_text(encoding="utf-8"))
    assert lock["lockfileVersion"] >= 3
    root = lock["packages"][""]
    for group in ("dependencies", "devDependencies"):
        assert root.get(group) == PACKAGE[group]


def test_nvmrc_is_exact_and_equals_engines():
    nvmrc = (WEB / ".nvmrc").read_text(encoding="utf-8").strip()
    assert EXACT.fullmatch(nvmrc)
    assert PACKAGE["engines"]["node"] == nvmrc


def test_npmrc_keeps_installs_exact_and_engine_strict():
    assert missing_npmrc_lines((WEB / ".npmrc").read_text(encoding="utf-8")) == set()


def test_npmrc_missing_line_is_caught():
    assert missing_npmrc_lines("save-exact=true\n") == {"engine-strict=true"}
    assert missing_npmrc_lines("engine-strict=false\nsave-exact=true\n") == {"engine-strict=true"}


def test_ci_web_job_reads_nvmrc_and_checks_job_stays():
    assert "checks" in WORKFLOW["jobs"]
    assert web_job_node_file(WORKFLOW) == "web/.nvmrc"


def test_ci_without_nvmrc_is_caught():
    pinned_inline = {"jobs": {"web": {"steps": [{"uses": "actions/setup-node@v4", "with": {"node-version": "26"}}]}}}
    assert web_job_node_file(pinned_inline) is None
    assert web_job_node_file({"jobs": {}}) is None


def test_gitignore_keeps_node_output_out_of_git():
    lines = {line.strip() for line in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()}
    assert {"node_modules/", "web/dist/"} <= lines
