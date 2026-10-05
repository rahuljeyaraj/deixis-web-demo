"""The FastAPI app (plan 7.2): the API under /api, health at /healthz, the built page at /.

In development Vite serves the page on :5173 and proxies /api and /healthz to this server on
:8000. Hosted (plan W14) this server serves `web/dist` itself on :7860.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from deixis.mcp import DeixisClient
from webdemo import __version__
from webdemo.models import NOT_USED, models_in_use
from webdemo.settings import Settings
from webdemo.speech import ClipError, Speech

MAX_META_BYTES = 1_000_000  # plan 7.2
MAX_AUDIO_BYTES = 44 + 12 * 16000 * 2 + 1024  # 12 s of 16 kHz 16-bit mono, plus a header

log = logging.getLogger("webdemo")


def make_client(settings: Settings) -> DeixisClient:
    """One client for the process (plan 7.3): it locks per call, and a call is about 10 ms."""
    fallback = None if settings.deixis_fallback == "none" else "direct"
    return DeixisClient(settings.profile, mode=settings.deixis_mode, fallback=fallback)


def load_speech(settings: Settings) -> Speech | None:
    """Parakeet, loaded and warmed up; None (logged) if it is off or not downloaded."""
    if not settings.asr_enabled:
        return None
    try:
        return Speech(settings.asr_models_dir, settings.asr_workers, settings.asr_threads)
    except FileNotFoundError as e:
        log.warning("speech is off: %s", e)
        return None


def create_app(
    settings: Settings | None = None,
    client: DeixisClient | None = None,
    speech: Speech | None = None,
) -> FastAPI:
    """The app. Tests pass their own `client` (over MCP, `fallback=None`) and `speech` (loaded
    once per test session), and keep them."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned_core = client is None
        owned_speech = speech is None
        app.state.deixis = client or make_client(settings)
        app.state.speech = speech if speech is not None else load_speech(settings)
        core: DeixisClient = app.state.deixis
        log.info("deixis core: mode=%s over_mcp=%s", core.mode, core.over_mcp)
        try:
            yield
        finally:
            if owned_core:
                app.state.deixis.close()
            if owned_speech and app.state.speech is not None:
                app.state.speech.close()

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
            "speech": {"loaded": app.state.speech is not None},
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

    @app.post("/api/command")
    async def command(
        meta: Annotated[str, Form()],
        audio: Annotated[UploadFile | None, File()] = None,
    ) -> dict[str, Any]:
        """One command (plan 4.2, 7.2). So far: the clip's words on the browser clock (W6);
        the agent and the core follow in W8."""
        t_start = time.perf_counter()
        if len(meta) > MAX_META_BYTES:
            raise HTTPException(413, "meta is over 1 MB")
        try:
            m = json.loads(meta)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"meta is not JSON: {e}") from e
        timings: dict[str, float] = {}
        words: list[dict[str, Any]] = []
        if audio is not None:
            speech: Speech | None = app.state.speech
            if speech is None:
                raise HTTPException(503, "speech recognition is not loaded on this server")
            if not isinstance(m.get("audio_t0"), int | float):
                raise HTTPException(400, "meta.audio_t0 is needed (the clip's first sample)")
            data = await audio.read(MAX_AUDIO_BYTES + 1)
            if len(data) > MAX_AUDIO_BYTES:
                raise HTTPException(413, "the clip is over 12 s")
            t = time.perf_counter()
            try:
                heard = await speech.transcribe_async(data, float(m["audio_t0"]))
            except ClipError as e:
                raise HTTPException(400, str(e)) from e
            timings["asr_s"] = round(time.perf_counter() - t, 3)
            words = [{"text": w.text, "t0": round(w.t0, 4), "t1": round(w.t1, 4)} for w in heard]
        elif not m.get("text"):
            raise HTTPException(400, "send a clip (audio) or a typed sentence (meta.text)")
        timings["total_s"] = round(time.perf_counter() - t_start, 3)
        return {"words": words, "timings": timings}

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
