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


def web_job_node_file(workflow: dict, job: str = "web") -> str | None:
    """The node-version-file of the setup-node step in the job (default `web`), or None."""
    for step in workflow.get("jobs", {}).get(job, {}).get("steps", []):
        if str(step.get("uses", "")).startswith("actions/setup-node@"):
            return step.get("with", {}).get("node-version-file")
    return None


E2E_UPLOAD_IF = "failure() && steps.e2e.outputs.scan == 'clean'"
E2E_REPORT = "web/playwright-report/results.json"


def e2e_job_problems(workflow: dict) -> list[str]:
    """What the `e2e` job (Stage 18 PR 5a) gets wrong: Chromium only, the JSON report uploaded only on failure
    with a clean secret scan, never an HTML report."""
    steps = workflow.get("jobs", {}).get("e2e", {}).get("steps", [])
    runs = [str(step.get("run", "")) for step in steps]
    uploads = [step for step in steps if str(step.get("uses", "")).startswith("actions/upload-artifact@")]
    problems = []
    if web_job_node_file(workflow, "e2e") != "web/.nvmrc":
        problems.append("node not from web/.nvmrc")
    installs = [run for run in runs if "playwright install" in run]
    if installs != ["npx playwright install --with-deps chromium"]:
        problems.append(f"browser install {installs}")
    if not uploads or any(step.get("if") != E2E_UPLOAD_IF or step.get("with", {}).get("path") != E2E_REPORT
                          for step in uploads):
        problems.append("upload not limited to the JSON report on failure with a clean scan")
    if any("html" in json.dumps(step).lower() for step in steps):
        problems.append("an html report")
    return problems


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


def test_ci_e2e_job_chromium_only_and_uploads_only_a_clean_json_report():
    assert {"checks", "web", "e2e"} <= set(WORKFLOW["jobs"])
    assert e2e_job_problems(WORKFLOW) == []


@pytest.mark.parametrize(("change", "problem"), [
    (lambda steps: steps[1]["with"].update({"node-version": "26"}) or steps[1]["with"].pop("node-version-file"),
     "node not from web/.nvmrc"),
    (lambda steps: steps.append({"run": "npx playwright install --with-deps chromium firefox"}), "browser install"),
    (lambda steps: [step.update({"if": "always()"}) for step in steps if "uses" in step and "upload" in step["uses"]],
     "upload not limited"),
    (lambda steps: [step["with"].update({"path": "web/playwright-report"}) for step in steps
                    if "upload" in str(step.get("uses", ""))], "upload not limited"),
    (lambda steps: steps.append({"run": "npx playwright show-report --reporter=html"}), "an html report"),
])
def test_ci_e2e_job_mistakes_are_caught(change, problem):
    workflow = json.loads(json.dumps(WORKFLOW))
    change(workflow["jobs"]["e2e"]["steps"])
    assert any(found.startswith(problem) for found in e2e_job_problems(workflow))
