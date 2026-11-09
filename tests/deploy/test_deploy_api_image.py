"""deploy/api.Dockerfile, requirements-api.txt and .dockerignore (Stage 17 Step 8a).

Text checks only, so CI needs no Docker: a pinned base image, a non-root user, the API's dependencies only
(each pinned as in requirements.txt), no client IP in uvicorn's logs, a health check on GET /api/health, and a
build context that never holds .env, the store, the venv, data or audio.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCKERFILE = (REPO / "deploy" / "api.Dockerfile").read_text(encoding="utf-8")
IGNORE = [line.strip() for line in (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()
          if line.strip() and not line.startswith("#")]
# Import name -> the requirements-api.txt package that provides it (pydantic comes with fastapi).
PROVIDER = {"argon2": "argon2-cffi", "yaml": "pyyaml", "pydantic": "fastapi"}
FORBIDDEN = {"torch", "librosa", "scikit-learn", "pandas", "kaggle", "huggingface-hub", "pytest", "httpx",
             "wandb", "safetensors", "python-dotenv"}


def pins(path: Path) -> dict[str, str]:
    """name (lower case) -> exact version; raise if a line is not name==version."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([0-9][A-Za-z0-9.]*)", line)
        assert match, f"{path.name}: not an exact pin: {line!r}"
        out[match.group(1).lower()] = match.group(2)
    return out


def instructions(name: str) -> list[str]:
    """The Dockerfile instructions of one kind, continuation lines joined."""
    text = DOCKERFILE.replace("\\\n", " ")
    return [line.split(None, 1)[1] for line in text.splitlines() if line.startswith(name + " ")]


def test_base_image_has_exact_tag_and_digest():
    (base,) = instructions("FROM")
    assert re.fullmatch(r"python:3\.13\.\d+-slim-[a-z]+@sha256:[0-9a-f]{64}", base), base


def test_runs_as_non_root_user():
    users = instructions("USER")
    assert users and users[-1] not in ("root", "0")


def test_copies_only_api_code_and_its_configs():
    copied = {part for line in instructions("COPY") for part in line.split()[:-1]}
    assert copied == {"requirements-api.txt", "api/", "configs/api.yaml", "configs/api_logging.yaml"}


def test_uvicorn_logs_no_client_ip_and_uses_the_log_config():
    (cmd,) = instructions("CMD")
    for flag in ("--no-access-log", "--no-proxy-headers", "configs/api_logging.yaml"):
        assert f'"{flag}"' in cmd


def test_healthcheck_calls_api_health():
    (check,) = instructions("HEALTHCHECK")
    assert "http://127.0.0.1:8000/api/health" in check
    assert "curl" not in check and "wget" not in check


def test_api_requirements_pin_exactly_as_requirements_txt():
    api, main = pins(REPO / "requirements-api.txt"), pins(REPO / "requirements.txt")
    assert api, "requirements-api.txt is empty"
    for name, version in api.items():
        assert main.get(name) == version, f"{name}=={version} differs from requirements.txt ({main.get(name)})"


def test_api_requirements_hold_no_model_dataset_or_test_package():
    assert not FORBIDDEN & set(pins(REPO / "requirements-api.txt"))


def test_api_requirements_cover_every_third_party_import_in_api():
    api = set(pins(REPO / "requirements-api.txt"))
    for path in (REPO / "api").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 else [])
            for name in names:
                top = name.split(".")[0]
                if top in sys.stdlib_module_names or top in ("api", "__future__"):
                    continue
                assert PROVIDER.get(top, top).lower() in api, f"{path.name} imports {top}, not in requirements-api.txt"


def test_dockerignore_is_an_allowlist_without_secrets_store_or_data():
    assert IGNORE[0] == "*"
    allowed = {line[1:] for line in IGNORE if line.startswith("!")}
    assert allowed == {"api/", "configs/api.yaml", "configs/api_logging.yaml", "requirements-api.txt"}
    for excluded in (".env", "var/", ".venv/", "data/", "**/*.wav", "**/*.flac", "**/*.mp3"):
        assert excluded in IGNORE
