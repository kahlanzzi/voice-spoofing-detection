"""App factory for the Stage 17 API. Run with: uvicorn api.main:app

Every route is under /api (docs/api_design.md §2); every error uses the §6 shape; every response
carries an X-Request-ID header. The database tables and the temp folder are created at startup
(lifespan), not at import. The lifespan also reads VSD_SIGNIN_HMAC_KEY: without a valid key the API
refuses to start (§4.8). Locally: uvicorn api.main:app --port 8001 --env-file .env
"""
from __future__ import annotations

import secrets
from contextlib import asynccontextmanager

from argon2 import PasswordHasher
from fastapi import FastAPI
from starlette.middleware.body_limit import RequestBodyLimitMiddleware

from api.analyzer import Analyzer, MockAnalyzer
from api.auth import password_hasher, signin_key_from_env
from api.config import ApiConfig, load_api_config, resolve_path
from api.errors import install_error_handlers
from api.intake import ContentLengthGuard, body_limit, note_body_limit
from api.middleware import RequestIdMiddleware
from api.repository import AccountRepository, AnalysisRepository
from api.routers import analyses, health, limits, notes, session, stats, status
from api.session_guard import SessionGuard

API_PREFIX = "/api"


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = app.state.config
    app.state.signin_key = signin_key_from_env()  # ValueError: refuse to start; the message never holds the key
    # The unknown-username path verifies against this (§4.8): made at start, so it always has the
    # hasher's current parameters and costs the same as a real verify.
    app.state.dummy_hash = app.state.hasher.hash(secrets.token_urlsafe(32))
    app.state.repository.create_schema()
    if config.temp_dir:
        resolve_path(config.temp_dir).mkdir(parents=True, exist_ok=True)
    yield


def create_app(config: ApiConfig | None = None, analyzer: Analyzer | None = None,
               hasher: PasswordHasher | None = None) -> FastAPI:
    """Build the app. Without a config, load it from $VSD_API_CONFIG or configs/api.yaml.
    Without an analyzer, use the MockAnalyzer (§8). Without a hasher, use the §4.8 argon2id one.
    Raise ValueError if stats_time_zone is not UTC."""
    config = config if config is not None else load_api_config()
    if config.stats_time_zone != "UTC":
        # History days are read as UTC days with no conversion until open decision 6 (§4.4).
        raise ValueError(f"stats_time_zone must be UTC until open decision 6 is taken, got {config.stats_time_zone!r}")
    app = FastAPI(title="Voice-spoofing detection API", lifespan=lifespan)
    app.state.config = config
    app.state.analyzer = analyzer if analyzer is not None else MockAnalyzer(config.mock_deferral_on_fit)
    app.state.repository = AnalysisRepository(resolve_path(config.database_path))
    app.state.accounts = AccountRepository(resolve_path(config.database_path))
    app.state.hasher = hasher if hasher is not None else password_hasher()
    # Added inner first: RequestId -> SessionGuard -> ContentLengthGuard -> RequestBodyLimit -> routes
    # (intake.py, session_guard.py).
    app.add_middleware(RequestBodyLimitMiddleware, max_body_size=body_limit(config))
    app.add_middleware(ContentLengthGuard, max_body_size=body_limit(config),
                       max_note_body_size=note_body_limit(config))
    app.add_middleware(SessionGuard, accounts=app.state.accounts, allowed_origins=config.allowed_origins,
                       idle_minutes=config.session_idle_minutes)
    app.add_middleware(RequestIdMiddleware)
    install_error_handlers(app)
    app.include_router(status.router, prefix=API_PREFIX)
    app.include_router(limits.router, prefix=API_PREFIX)
    app.include_router(analyses.router, prefix=API_PREFIX)
    app.include_router(notes.router, prefix=API_PREFIX)
    app.include_router(stats.router, prefix=API_PREFIX)
    app.include_router(session.router, prefix=API_PREFIX)
    app.include_router(health.router, prefix=API_PREFIX)
    return app


app = create_app()
