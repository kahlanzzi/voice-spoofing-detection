"""App factory for the Stage 17 API. Run with: uvicorn api.main:app

Every route is under /api (docs/api_design.md §2); every error uses the §6 shape; every response
carries an X-Request-ID header.
"""
from __future__ import annotations

from fastapi import FastAPI

from api.config import ApiConfig, load_api_config
from api.errors import install_error_handlers
from api.middleware import RequestIdMiddleware
from api.routers import status

API_PREFIX = "/api"


def create_app(config: ApiConfig | None = None) -> FastAPI:
    """Build the app. Without a config, load it from $VSD_API_CONFIG or configs/api.yaml."""
    app = FastAPI(title="Voice-spoofing detection API")
    app.state.config = config if config is not None else load_api_config()
    app.add_middleware(RequestIdMiddleware)
    install_error_handlers(app)
    app.include_router(status.router, prefix=API_PREFIX)
    return app


app = create_app()
