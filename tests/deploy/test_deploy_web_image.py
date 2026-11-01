"""deploy/web.Dockerfile and its build-context ignore file (Stage 18 PR 2).

Text checks only, so CI needs no Docker: Node as in web/.nvmrc and nginx as in Stage 17 Step 8b, both pinned by
tag and digest; npm ci with strict engines; a final image with the built app and the nginx config only; a build
context that never holds .env, the store, the venv, data, audio, node_modules or a local build.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCKERFILE = (REPO / "deploy" / "web.Dockerfile").read_text(encoding="utf-8")
IGNORE = [line.strip() for line in (REPO / "deploy" / "web.Dockerfile.dockerignore").read_text(encoding="utf-8")
          .splitlines() if line.strip() and not line.startswith("#")]
NGINX = "nginx:1.30.6-alpine@sha256:0e52bd054126cdc040b1276c0af347aeb02a41b9cfe305d6094767ac30e97949"


def stages() -> list[list[str]]:
    """The instructions of each stage (comments dropped), split at FROM."""
    out: list[list[str]] = []
    for line in DOCKERFILE.splitlines():
        if line.startswith("FROM "):
            out.append([])
        if line and not line.startswith("#"):
            out[-1].append(line)
    return out


def test_two_stages_pinned_by_tag_and_digest():
    build, final = stages()
    node = (REPO / "web" / ".nvmrc").read_text(encoding="utf-8").strip()
    assert re.fullmatch(rf"FROM node:{re.escape(node)}-alpine@sha256:[0-9a-f]{{64}} AS build", build[0]), build[0]
    assert final[0] == f"FROM {NGINX}"  # the Stage 17 Step 8b pin, unchanged


def test_build_stage_uses_the_lockfile_and_strict_engines():
    build, _ = stages()
    assert "RUN npm ci" in build and "RUN npm run build" in build
    assert build.index("RUN npm ci") < build.index("RUN npm run build")
    assert "COPY web/package.json web/package-lock.json web/.npmrc ./" in build
    assert "npm install" not in DOCKERFILE and "ignore-engines" not in DOCKERFILE
    assert "engine-strict=false" not in DOCKERFILE
    assert "engine-strict=true" in (REPO / "web" / ".npmrc").read_text(encoding="utf-8").split()


def test_final_image_holds_the_build_and_the_nginx_config_only():
    _, final = stages()
    copies = [line for line in final if line.startswith("COPY ")]
    assert copies == ["COPY --from=build /web/dist /usr/share/vsd-web",
                      "COPY deploy/nginx.conf /etc/nginx/nginx.conf",
                      "COPY deploy/nginx_security_headers.conf /etc/nginx/vsd_security_headers.conf"]
    assert not [line for line in final if line.startswith(("RUN ", "ADD "))]


def test_build_context_is_an_allowlist_without_secrets_dependencies_or_data():
    assert IGNORE[0] == "*"
    allowed = {line[1:] for line in IGNORE if line.startswith("!")}
    assert allowed == {"web/", "deploy/nginx.conf", "deploy/nginx_security_headers.conf"}
    for excluded in ("web/node_modules", "web/dist", "**/.env", "var/", ".venv/", "data/", "**/*.wav", "**/*.flac",
                     "**/*.mp3"):
        assert excluded in IGNORE, excluded
