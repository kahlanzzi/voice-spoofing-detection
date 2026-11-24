"""App factory for the Stage 17 API. Run with: uvicorn api.main:app

Every route is under /api (docs/api_design.md §2); every error uses the §6 shape; every response
carries an X-Request-ID header. The database tables and the temp folder are created at startup
(lifespan), not at import. The lifespan also reads VSD_SIGNIN_HMAC_KEY and VSD_DB_APP_PASSWORD:
without a valid key or a reachable PostgreSQL database the API refuses to start (§4.8, D53 (4)). It then
starts the analysis pool (api/pool.py, D53 (2)): if any worker's start-up hook fails, it refuses to start.
Locally: docs/setup.md ("Running the API outside Compose").
"""
from __future__ import annotations

import functools
import secrets
from contextlib import asynccontextmanager

from argon2 import PasswordHasher
from fastapi import FastAPI
from starlette.middleware.body_limit import RequestBodyLimitMiddleware

from api.analyzer import MockAnalyzer
from api.bundle_analyzer import BundleAnalyzer
from api.auth import password_hasher, signin_key_from_env
from api.config import ApiConfig, load_api_config, resolve_path
from api.db import Database, app_password_from_env
from api.errors import install_error_handlers
from api.intake import ContentLengthGuard, body_limit, note_body_limit
from api.middleware import RequestIdMiddleware
from api.pool import AnalysisPool, AnalyzerFactory
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
    database: Database = app.state.database
    database.open(app_password_from_env())  # ValueError or StoreError: refuse to start
    try:
        app.state.repository.create_schema()
        if config.temp_dir:
            resolve_path(config.temp_dir).mkdir(parents=True, exist_ok=True)
        try:
            await app.state.pool.start()  # PoolStartError: refuse to start (no serving with a missing worker)
            yield
        finally:
            await app.state.pool.close()
    finally:
        database.close()


def analyzer_factory_for(config: ApiConfig) -> AnalyzerFactory:
    """The factory for config.analyzer (the loader accepts mock and bundle only)."""
    if config.analyzer == "bundle":
        return functools.partial(BundleAnalyzer, config.bundle_dir, config.bundle_manifest_sha256)
    if config.analyzer == "mock":
        return functools.partial(MockAnalyzer, config.mock_deferral_on_fit)
    raise ValueError(f"unknown analyzer {config.analyzer!r}")


def create_app(config: ApiConfig | None = None, analyzer_factory: AnalyzerFactory | None = None,
               hasher: PasswordHasher | None = None) -> FastAPI:
    """Build the app. Without a config, load it from $VSD_API_CONFIG or configs/api.yaml.
    Without an analyzer factory, each pool worker builds the one config.analyzer names (§8): the MockAnalyzer,
    or the BundleAnalyzer with the pinned manifest SHA-256. A factory must be
    picklable (api/pool.py). Without a hasher, use the §4.8 argon2id one. No process starts here: the
    lifespan starts the pool. Raise ValueError if stats_time_zone is not UTC."""
    config = config if config is not None else load_api_config()
    if config.stats_time_zone != "UTC":
        # History days are read as UTC days with no conversion until open decision 6 (§4.4).
        raise ValueError(f"stats_time_zone must be UTC until open decision 6 is taken, got {config.stats_time_zone!r}")
    app = FastAPI(title="Voice-spoofing detection API", lifespan=lifespan)
    app.state.config = config
    factory = analyzer_factory if analyzer_factory is not None else analyzer_factory_for(config)
    app.state.pool = AnalysisPool(factory, size=config.analysis_workers,
                                  start_timeout_s=config.analysis_worker_start_timeout_s)
    app.state.database = Database.from_config(config)  # opened by the lifespan; building never connects
    app.state.repository = AnalysisRepository(app.state.database)
    app.state.accounts = AccountRepository(app.state.database)
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
