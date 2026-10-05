"""The FastAPI app (plan 7.2): the API under /api, health at /healthz, the built page at /.

In development Vite serves the page on :5173 and proxies /api and /healthz to this server on
:8000. Hosted (plan W14) this server serves `web/dist` itself on :7860.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from deixis.mcp import DeixisClient
from webdemo import __version__
from webdemo.models import NOT_USED, models_in_use
from webdemo.settings import Settings

log = logging.getLogger("webdemo")


def make_client(settings: Settings) -> DeixisClient:
    """One client for the process (plan 7.3): it locks per call, and a call is about 10 ms."""
    fallback = None if settings.deixis_fallback == "none" else "direct"
    return DeixisClient(settings.profile, mode=settings.deixis_mode, fallback=fallback)


def create_app(settings: Settings | None = None, client: DeixisClient | None = None) -> FastAPI:
    """The app. Tests pass their own `client` (over MCP, `fallback=None`) and keep it."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = client is None
        app.state.deixis = client or make_client(settings)
        core: DeixisClient = app.state.deixis
        log.info("deixis core: mode=%s over_mcp=%s", core.mode, core.over_mcp)
        try:
            yield
        finally:
            if owned:
                app.state.deixis.close()

    app = FastAPI(title="deixis web demo", version=__version__, lifespan=lifespan)
    app.state.settings = settings

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        core: DeixisClient = app.state.deixis
        try:
            core.describe()  # one real round trip to `deixis-mcp`
            core_ok = True
        except Exception:  # noqa: BLE001 - health reports, never raises
            log.exception("healthz: the core did not answer")
            core_ok = False
        return {
            "ok": core_ok,
            "core": {"ok": core_ok, "mode": core.mode, "over_mcp": core.over_mcp,
                     "fell_back": core.fell_back},
            "speech": {"loaded": False},  # Parakeet arrives in W6
        }  # fmt: skip

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        core: DeixisClient = app.state.deixis
        described = core.describe()
        return {
            "title": settings.title,
            "version": __version__,
            "core": {"version": described["version"], "schema_version": described["schema_version"],
                     "over_mcp": core.over_mcp},
            "profile": described["profile"],
            "models": models_in_use(settings.nemotron_model),
            "not_used": NOT_USED,
        }  # fmt: skip

    if settings.web_dist.is_dir():
        app.mount("/", StaticFiles(directory=settings.web_dist, html=True), name="page")

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = Settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
