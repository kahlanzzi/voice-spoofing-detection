"""App factory for the Stage 17 API. Run with: uvicorn api.main:app

Every route is under /api (docs/api_design.md §2); every error uses the §6 shape; every response
carries an X-Request-ID header. The database table and the temp folder are created at startup
(lifespan), not at import.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.body_limit import RequestBodyLimitMiddleware

from api.analyzer import Analyzer, MockAnalyzer
from api.config import ApiConfig, load_api_config, resolve_path
from api.errors import install_error_handlers
from api.intake import ContentLengthGuard, body_limit
from api.middleware import RequestIdMiddleware
from api.repository import AnalysisRepository
from api.routers import analyses, limits, status

API_PREFIX = "/api"


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = app.state.config
    app.state.repository.create_schema()
    if config.temp_dir:
        resolve_path(config.temp_dir).mkdir(parents=True, exist_ok=True)
    yield


def create_app(config: ApiConfig | None = None, analyzer: Analyzer | None = None) -> FastAPI:
    """Build the app. Without a config, load it from $VSD_API_CONFIG or configs/api.yaml.
    Without an analyzer, use the MockAnalyzer (§8). Raise ValueError if stats_time_zone is not UTC."""
    config = config if config is not None else load_api_config()
    if config.stats_time_zone != "UTC":
        # History days are read as UTC days with no conversion until open decision 6 (§4.4).
        raise ValueError(f"stats_time_zone must be UTC until open decision 6 is taken, got {config.stats_time_zone!r}")
    app = FastAPI(title="Voice-spoofing detection API", lifespan=lifespan)
    app.state.config = config
    app.state.analyzer = analyzer if analyzer is not None else MockAnalyzer(config.mock_deferral_on_fit)
    app.state.repository = AnalysisRepository(resolve_path(config.database_path))
    # Added inner first: RequestId -> ContentLengthGuard -> RequestBodyLimit -> routes (intake.py).
    app.add_middleware(RequestBodyLimitMiddleware, max_body_size=body_limit(config))
    app.add_middleware(ContentLengthGuard, max_body_size=body_limit(config))
    app.add_middleware(RequestIdMiddleware)
    install_error_handlers(app)
    app.include_router(status.router, prefix=API_PREFIX)
    app.include_router(limits.router, prefix=API_PREFIX)
    app.include_router(analyses.router, prefix=API_PREFIX)
    return app


app = create_app()
