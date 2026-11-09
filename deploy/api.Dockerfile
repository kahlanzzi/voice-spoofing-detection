# The Stage 17 API image (docs/api_design.md §2, §4.8; docs/setup.md "Running with Docker Compose").
# Build from the repo root: docker compose build api. The context is limited by .dockerignore.
# API dependencies only (requirements-api.txt); no datasets, no model packages, no .env.
# Python 3.13.15 is the official version (D15). The tag and digest are pinned together.
FROM python:3.13.15-slim-trixie@sha256:7c61056e61ac89e852de05f3dc6fa51a6dd2181797bceed46aa725dd7cb2cd3b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements-api.txt ./
RUN pip install -r requirements-api.txt

# Non-root user. /app/var holds the SQLite file (database_path: var/api.sqlite) on a named volume;
# the volume takes this folder's owner when Docker first creates it.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/var && chown app:app /app/var
COPY api/ api/
COPY configs/api.yaml configs/api_logging.yaml configs/
USER app

EXPOSE 8000
# Liveness only (§4.8 GET /health); Python's urllib because the slim image has no curl.
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status != 200)"]
# No access log and no proxy headers: no client IP is logged or read (§1, §7). The api logger runs at INFO.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--no-access-log", "--no-proxy-headers", "--log-config", "configs/api_logging.yaml"]
