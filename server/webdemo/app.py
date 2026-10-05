"""The FastAPI app (plan 7.2): the API under /api, health at /healthz, the built page at /.

In development Vite serves the page on :5173 and proxies /api and /healthz to this server on
:8000. Hosted (plan W14) this server serves `web/dist` itself on :7860.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Cookie, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from deixis.mcp import DeixisClient
from deixis.profile import load_profile
from deixis.refs import DEMONSTRATIVES, norm_word
from webdemo import __version__
from webdemo.agent import AgentResult, ArmAgent, Context
from webdemo.command import (
    Meta,
    Sessions,
    answer_by_label,
    answer_entry,
    nearest_candidate,
    onto_pointed_block,
    say_for,
    typed_words,
)
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


def log_command(words: list[dict[str, Any]], agent: AgentResult, result: dict[str, Any] | None,
                error: str | None, notes: list[str]) -> None:  # fmt: skip
    """One line per command for the console: the words, the calls, what the core said. No
    audio, no key, no address."""
    said = " ".join(w["text"] for w in words)
    calls = "; ".join(str(c) for c in agent.calls) or "-"
    if error:
        outcome = f"error: {error}"
    elif result is None:
        outcome = "not understood"
    elif result["status"] == "ask":
        outcome = f"ask {result['ask']['reason']}: {result['ask']['question']}"
    else:
        places = []
        for c in result["calls"]:
            for k, v in c["args"].items():
                if isinstance(v, dict) and v.get("frame"):
                    what = v.get("thing") or v.get("mean_mm")
                    z = f" z{v['z_mm']:.0f}" if v.get("z_mm") else ""
                    places.append(f"{c['name']}.{k}={what}{z}")
        outcome = "ok " + ", ".join(places)
    extra = f" ({'; '.join(notes)})" if notes else ""
    log.info('command: "%s" -> %s [%s] -> %s%s', said, calls, agent.source, outcome, extra)


class AnswerBody(BaseModel):
    """`POST /api/answer`: the person chose one of the last question's candidates."""

    candidate: int


def create_app(
    settings: Settings | None = None,
    client: DeixisClient | None = None,
    speech: Speech | None = None,
    llm_factory: Callable[[str], Any] | None = None,
) -> FastAPI:
    """The app. Tests pass their own `client` (over MCP, `fallback=None`) and `speech` (loaded
    once per test session), and keep them; `llm_factory(key)` makes the OpenAI-like client for
    Nemotron (tests pass a fake)."""
    settings = settings or Settings()
    profile = load_profile(settings.profile)
    sessions = Sessions()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned_core = client is None
        owned_speech = speech is None
        app.state.deixis = client or make_client(settings)
        app.state.speech = speech if speech is not None else load_speech(settings)
        core: DeixisClient = app.state.deixis
        log.info("deixis core: mode=%s over_mcp=%s", core.mode, core.over_mcp)
        # The tools come from the core, over MCP, as every robot gets them.
        app.state.agent = ArmAgent(
            profile, core.tools_for_agent(), settings.nebius_base_url,
            model=settings.nemotron_model, client_factory=llm_factory,
        )  # fmt: skip
        try:
            yield
        finally:
            if owned_core:
                app.state.deixis.close()
            if owned_speech and app.state.speech is not None:
                app.state.speech.close()

    app = FastAPI(title="deixis web demo", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.sessions = sessions

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

    def visitor(response: Response, sid: str | None) -> str:
        """The visitor's random id (a cookie), made on the first command."""
        if not sid or len(sid) > 64:
            sid = Sessions.new_id()
            response.set_cookie("sid", sid, httponly=True, samesite="strict", max_age=86400)
        return sid

    async def resolve(request: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        """The core's answer over MCP, as JSON; or why it refused the request."""
        core: DeixisClient = app.state.deixis
        try:
            r = await run_in_threadpool(core.resolve, request)
        except (ValidationError, ValueError) as e:
            log.warning("the core refused the request: %s", e)
            return None, str(e).splitlines()[0]
        return r.model_dump(mode="json"), None

    def remember(sid: str, request: dict[str, Any], result: dict[str, Any] | None) -> None:
        if result and result["status"] == "ask":
            sessions.put(sid, request, result)
        else:
            sessions.drop(sid)

    Reply = tuple[dict[str, Any] | None, str | None]  # the core's answer, or why it refused

    async def answer_open(sid: str, entry: dict[str, Any]) -> Reply:
        """Resolve the visitor's open question again with the person's answer, plus what was
        already resolved (as C18's last test does)."""
        open_ask = sessions.get(sid)
        if open_ask is None:
            raise HTTPException(409, "there is no open question (it may have expired)")
        request = {**open_ask.request, "answers": [entry, *open_ask.response["partial"]]}
        result, error = await resolve(request)
        remember(sid, request, result)
        return result, error

    async def spoken_answer(sid: str, m: Meta, words: list[dict[str, Any]]) -> dict | None:
        """The answer to the open question in words: "the red one"; "this one" while pointing
        at a block; "here" while pointing at a spot (when the robot asked where)."""
        open_ask = sessions.get(sid)
        if open_ask is None or not m.state.question:
            return None
        ask = open_ask.response["ask"]
        texts = [norm_word(w["text"]) for w in words]
        picked = answer_by_label(ask, texts)
        if picked is not None:
            return answer_entry(ask, picked)
        k = next((i for i, w in enumerate(texts) if w in DEMONSTRATIVES and w != "so"), None)
        if k is None or not ask.get("arg"):
            return None
        # Where did they point while saying it? The core's own `locate`, over MCP.
        locate = {"name": "locate", "args": {"place": {"deixis": texts[k], "word_index": k}}}
        r, _ = await resolve({"words": words, "samples": m.samples, "holds": m.holds,
                              "things": m.things, "calls": [locate]})  # fmt: skip
        if not r or r["status"] != "ok":
            return None
        where = r["calls"][0]["args"]["place"]
        if any(c.get("thing") for c in ask["candidates"]):  # which block
            picked = nearest_candidate(ask, where.get("mean_mm") or [0, 0], within_mm=40)
            return None if picked is None else answer_entry(ask, picked)
        return {"call": ask.get("call") or 0, "arg": ask["arg"], "where": where}  # which spot

    @app.post("/api/command")
    async def command(
        response: Response,
        meta: Annotated[str, Form()],
        audio: Annotated[UploadFile | None, File()] = None,
        x_nebius_key: Annotated[str | None, Header()] = None,
        sid: Annotated[str | None, Cookie()] = None,
    ) -> dict[str, Any]:
        """One command (plan 4.2, 7.2): words (from the clip, or typed), the agent's tool
        calls, the core's places over MCP. A visitor's own key (`X-Nebius-Key`) is used for
        this request's Nemotron call only; it is never stored, logged or sent back."""
        t_start = time.perf_counter()
        sid = visitor(response, sid)
        if len(meta) > MAX_META_BYTES:
            raise HTTPException(413, "meta is over 1 MB")
        try:
            m = Meta.model_validate_json(meta)
        except ValidationError as e:
            raise HTTPException(400, f"meta: {e.errors()[0]['msg']}") from None
        timings: dict[str, float] = {}
        if audio is not None:
            speech: Speech | None = app.state.speech
            if speech is None:
                raise HTTPException(503, "speech recognition is not loaded on this server")
            if m.audio_t0 is None:
                raise HTTPException(400, "meta.audio_t0 is needed (the clip's first sample)")
            data = await audio.read(MAX_AUDIO_BYTES + 1)
            if len(data) > MAX_AUDIO_BYTES:
                raise HTTPException(413, "the clip is over 12 s")
            t = time.perf_counter()
            try:
                heard = await speech.transcribe_async(data, m.audio_t0)
            except ClipError as e:
                raise HTTPException(400, str(e)) from e
            timings["asr_s"] = round(time.perf_counter() - t, 3)
            words = [{"text": w.text, "t0": round(w.t0, 4), "t1": round(w.t1, 4)} for w in heard]
        elif m.text and m.text.strip():
            words = typed_words(m.text, m.holds, m.t)
        else:
            raise HTTPException(400, "send a clip (audio) or a typed sentence (meta.text)")

        # The agent: Nemotron with the visitor's key or ours, else the rule parser.
        texts = [w["text"] for w in words]
        agent: ArmAgent = app.state.agent
        ours = settings.nebius_api_key.get_secret_value() if settings.nebius_api_key else None
        key = None if settings.ai == "rules" else (x_nebius_key or ours)
        context = Context(previewing=m.state.previewing, question=m.state.question)
        t = time.perf_counter()
        if texts:
            loop = asyncio.get_running_loop()
            agent_res = await loop.run_in_executor(None, agent.understand, texts, key, context)
        else:
            agent_res = AgentResult([], source="rules", model="rules", errors=["no words"])
        key = ours = None
        timings["agent_s"] = round(time.perf_counter() - t, 3)

        # The core: every reference to a place or a block, over MCP.
        calls = [c.to_dict() for c in agent_res.calls]
        result = error = None
        request: dict[str, Any] = {}
        entry = None if calls else await spoken_answer(sid, m, words)
        notes: list[str] = []
        t = time.perf_counter()
        if entry is not None:
            result, error = await answer_open(sid, entry)
            agent_res.source = "answer"
            say = result["ask"]["question"] if result and result["status"] == "ask" else None
        else:
            if calls:
                request = {"words": words, "samples": m.samples, "holds": m.holds,
                           "things": m.things, "calls": calls, "context": m.context,
                           "answers": m.answers}  # fmt: skip
                result, error = await resolve(request)
                again = onto_pointed_block(request, result) if result else None
                if again is not None:
                    result2, error2 = await resolve(again)
                    if result2 is not None:
                        request, result, error = again, result2, error2
                        notes.append("pointed onto a block: on top of it")
            remember(sid, request, result)
            say = "Sorry, something went wrong." if error else say_for(result, calls)
        if result is not None or entry is not None or calls:
            timings["core_s"] = round(time.perf_counter() - t, 3)
        timings["total_s"] = round(time.perf_counter() - t_start, 3)
        log_command(words, agent_res, result, error, notes)
        return {
            "words": words,
            "agent": agent_res.summary(),
            "result": result,
            "error": error,
            "notes": notes,
            "say": say,
            "timings": timings,
        }

    @app.post("/api/answer")
    async def answer(
        body: AnswerBody, response: Response, sid: Annotated[str | None, Cookie()] = None
    ) -> dict[str, Any]:
        """The person picked a candidate of the last question (clicked it): resolve the same
        request again with that answer and what was already resolved (as C18's last test)."""
        t_start = time.perf_counter()
        sid = visitor(response, sid)
        open_ask = sessions.get(sid)
        if open_ask is None:
            raise HTTPException(409, "there is no open question (it may have expired)")
        ask = open_ask.response["ask"]
        if not 0 <= body.candidate < len(ask["candidates"]):
            raise HTTPException(400, "no such candidate")
        result, error = await answer_open(sid, answer_entry(ask, body.candidate))
        timings = {"total_s": round(time.perf_counter() - t_start, 3)}
        say = result["ask"]["question"] if result and result["status"] == "ask" else None
        return {"result": result, "error": error, "say": say, "timings": timings}

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
