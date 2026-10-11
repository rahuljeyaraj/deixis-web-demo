"""The FastAPI app (plan 7.2): the API under /api, health at /healthz, the built page at /.

In development Vite serves the page on :5173 and proxies /api and /healthz to this server on
:8000. Hosted (plan W14) this server serves `web/dist` itself on :7860.

A command may come back as a plan of up to four steps (advanced-commands-plan.md section 6).
`/api/command` resolves only its first step; the page asks for each later one with
`/api/step` just before it runs it, sending the table as it is at that moment, and for a new
plan with `/api/replan` when a step turns out to be impossible. Every reply carries a `trace`
(`trace.py`): what each part said to the next, for the page's wiring panel.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated, Any

from fastapi import (
    Cookie,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from deixis.mcp import DeixisClient
from deixis.profile import load_profile
from deixis.refs import DEMONSTRATIVES, norm_word
from webdemo import __version__, trace
from webdemo.agent import MAX_STEPS, AgentResult, ArmAgent, Context, rules
from webdemo.command import (
    MAX_SAMPLES,
    Meta,
    OpenPlan,
    Plans,
    Sessions,
    answer_by_label,
    answer_entry,
    call_lines,
    choices_for,
    nearest_candidate,
    onto_pointed_block,
    say_for,
    typed_words,
)
from webdemo.limits import NO_KEY_WHY, Caps, Limits, Mode, Why, view
from webdemo.models import NOT_USED, models_in_use
from webdemo.record import CommandLog, cost_usd, in_words
from webdemo.settings import Settings
from webdemo.speech import ClipError, Speech

MAX_META_BYTES = 1_000_000  # plan 7.2
MAX_AUDIO_BYTES = 44 + 12 * 16000 * 2 + 1024  # 12 s of 16 kHz 16-bit mono, plus a header
MAX_KEY_CHARS = 4096
#: One re-plan per command, so a model that keeps failing cannot run away (section 8).
MAX_REPLANS = 1

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
                error: str | None, notes: list[str], mode: str = "") -> None:  # fmt: skip
    """One line per command for the console: the words, the calls, what the core said, and
    whose key (ours, own, simple). No audio, no key, no address."""
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
    source = f"{agent.source}, {mode}" if mode else agent.source
    log.info('command: "%s" -> %s [%s] -> %s%s', said, calls, source, outcome, extra)


EVENTS = {
    "start", "preview", "refused", "go", "done", "failed", "cancel", "stop", "undo", "reset",
    "camera_error", "mic_error", "silent", "no_hand", "server_down",
    "plan", "step_failed", "replan", "declined",
}  # fmt: skip


class EventBody(BaseModel):
    """`POST /api/event`: what happened on the page after a command (for the log only)."""

    kind: str
    via: str | None = None  # voice, button, key, click
    why: str | None = None  # a refusal's or an error's reason, in a few words
    latency_s: float | None = None  # a preview: from the last word to the guess on screen
    block: str | None = None
    step: int | None = None  # which step of a plan this was about
    steps: int | None = None  # how many steps the plan had


class AnswerBody(BaseModel):
    """`POST /api/answer`: the person chose one of the last question's candidates."""

    candidate: int


class StepBody(BaseModel):
    """`POST /api/step`: measure one step of the running plan, now.

    `things` and `context` are the table as the page sees it at this moment, which is the
    point: a step is resolved against the scene it will act on, not the one it was planned
    in. The words, the pointer samples and the holds come from the stored plan, so "there"
    still means where the finger was when it was said."""

    index: int = Field(ge=0, le=MAX_STEPS)
    things: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    context: dict[str, Any] = Field(default_factory=dict)


class ReplanBody(BaseModel):
    """`POST /api/replan`: a step could not be done; ask Nemotron for the rest another way."""

    index: int = Field(ge=0, le=MAX_STEPS)
    why: str = Field(max_length=200)  # the arm's own reason, as it said it
    things: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    context: dict[str, Any] = Field(default_factory=dict)


def create_app(
    settings: Settings | None = None,
    client: DeixisClient | None = None,
    speech: Speech | None = None,
    llm_factory: Callable[[str], Any] | None = None,
    limits: Limits | None = None,
) -> FastAPI:
    """The app. Tests pass their own `client` (over MCP, `fallback=None`) and `speech` (loaded
    once per test session), and keep them; `llm_factory(key)` makes the OpenAI-like client for
    Nemotron (tests pass a fake); `limits` (tests pass one with their own clock)."""
    settings = settings or Settings()
    profile = load_profile(settings.profile)
    sessions = Sessions()
    plans = Plans()
    record = CommandLog(settings.command_log)
    limits = limits or Limits(
        Caps(visitor=settings.cap_visitor_per_day, address=settings.cap_address_per_day,
             total=settings.cap_total_per_day, burst_s=settings.burst_s,
             address_burst_s=settings.burst_s / 3)
    )  # fmt: skip

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
    app.state.limits = limits

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

    def visitor(response: Response, sid: str | None) -> str:
        """The visitor's random id (a cookie), made on the first visit."""
        if not sid or len(sid) > 64:
            sid = Sessions.new_id()
            response.set_cookie("sid", sid, httponly=True, samesite="strict", max_age=86400)
        return sid

    def address(request: Request) -> str:
        """The visitor's address, hashed (plan 8.1). Behind proxies, the one our outermost
        proxy saw (`proxy_hops` from the right of `X-Forwarded-For`); never the left end,
        which anyone can write."""
        ip = request.client.host if request.client else "unknown"
        if settings.proxy_hops > 0:
            seen = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",")]
            seen = [h for h in seen if h]
            if len(seen) >= settings.proxy_hops:
                ip = seen[-settings.proxy_hops]
        return limits.address_of(ip)

    def our_key(sid: str, addr: str) -> tuple[str | None, Why | None]:
        """Our key if this visitor may use it now, else why not (simple mode)."""
        if settings.ai == "rules":
            return None, "rules"
        if not settings.nebius_api_key:
            return None, "no_key"
        _, why = limits.left(sid, addr)
        return (None, why) if why else (settings.nebius_api_key.get_secret_value(), None)

    def limits_now(sid: str, addr: str, mode: Mode, why: Why | None) -> dict[str, Any]:
        left, _ = limits.left(sid, addr)
        if why in ("rules", "no_key"):
            left = 0
        return view(mode, left, limits.caps.visitor, why)

    @app.get("/api/info")
    def info(request: Request, response: Response,
             sid: Annotated[str | None, Cookie()] = None) -> dict[str, Any]:  # fmt: skip
        core: DeixisClient = app.state.deixis
        described = core.describe()
        sid = visitor(response, sid)
        addr = address(request)
        key, why = our_key(sid, addr)
        lim = limits_now(sid, addr, "ours" if key else "simple", why)
        key = None
        models = models_in_use(settings.nemotron_model, off=lim["off"])
        return {
            "title": settings.title,
            "version": __version__,
            "core": {"version": described["version"], "schema_version": described["schema_version"],
                     "over_mcp": core.over_mcp},
            "profile": described["profile"],
            "models": models,
            "not_used": NOT_USED,
            "limits": lim,
            "show_own_key": settings.show_own_key,
        }  # fmt: skip

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
    Step = tuple[dict[str, Any], dict[str, Any] | None, str | None, list[str]]

    async def resolve_one(moment: dict[str, Any], things: list[dict[str, Any]],
                          context: dict[str, Any], calls: list[dict[str, Any]],
                          answers: list[dict[str, Any]] | None = None) -> Step:  # fmt: skip
        """One core request over MCP, with the "pointed onto a block" retry (robot-side glue,
        not a core change). Gives back the request as it was finally resolved, the core's
        answer, why it refused, and any notes.

        `moment` is the words, samples and holds from when the person spoke; `things` and
        `context` are the table now."""
        request = {**moment, "things": things, "calls": calls, "context": context,
                   "answers": answers or []}  # fmt: skip
        notes: list[str] = []
        result, error = await resolve(request)
        again = onto_pointed_block(request, result) if result else None
        if again is not None:
            result2, error2 = await resolve(again)
            if result2 is not None:
                request, result, error = again, result2, error2
                notes.append("pointed onto a block: on top of it")
        return request, result, error, notes

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

    async def pointed_at(
        m: Meta, words: list[dict[str, Any]], word: str, k: int
    ) -> dict[str, Any] | None:
        """Where the person pointed while saying word `k`: the core's own `locate`, over MCP.
        This is the part of an answer the model cannot know - it never sees the finger."""
        locate = {"name": "locate", "args": {"place": {"deixis": word, "word_index": k}}}
        r, _ = await resolve({"words": words, "samples": m.samples, "holds": m.holds,
                              "things": m.things, "calls": [locate]})  # fmt: skip
        if not r or r["status"] != "ok":
            return None
        return r["calls"][0]["args"]["place"]

    async def pointed_answer(
        ask: dict[str, Any], m: Meta, words: list[dict[str, Any]], k: int
    ) -> dict[str, Any] | None:
        """The open question answered by pointing: the candidate nearest the spot they
        pointed at while saying word `k` ("this one"), or that spot itself when the robot
        asked where. Used for Nemotron's `pick(word_index=k)` and by `spoken_answer`."""
        if not ask.get("arg") or not 0 <= k < len(words):
            return None
        where = await pointed_at(m, words, norm_word(words[k]["text"]), k)
        if where is None:
            return None
        if any(c.get("thing") for c in ask["candidates"]):  # which block
            picked = nearest_candidate(ask, where.get("mean_mm") or [0, 0], within_mm=40)
            return None if picked is None else answer_entry(ask, picked)
        return {"call": ask.get("call") or 0, "arg": ask["arg"], "where": where}  # which spot

    async def spoken_answer(sid: str, m: Meta, words: list[dict[str, Any]]) -> dict | None:
        """The answer to the open question read from the words on the server: "the red one";
        "this one" while pointing at a block; "here" while pointing at a spot. Only used when
        Nemotron gave neither a `pick` nor a command (it is down, or it said nothing), so a
        question can always still be answered."""
        open_ask = sessions.get(sid)
        if open_ask is None or not m.state.question:
            return None
        ask = open_ask.response["ask"]
        texts = [norm_word(w["text"]) for w in words]
        picked = answer_by_label(ask, texts)
        if picked is not None:
            return answer_entry(ask, picked)
        k = next((i for i, w in enumerate(texts) if w in DEMONSTRATIVES and w != "so"), None)
        return None if k is None else await pointed_answer(ask, m, words, k)

    Asked = tuple[AgentResult, Mode, Why | None, bool, float]

    async def ask_model(sid: str, addr: str, own: str | None, texts: list[str],
                        context: Context) -> Asked:  # fmt: skip
        """The agent for one command (or one re-plan): Nemotron with the visitor's own key, or
        ours within the day's limits; else the rule parser (simple mode). A refused own key:
        as if none was sent. Gives back what it decided, whose key, why not ours, whether the
        own key was refused, and how long the whole thing took."""
        agent: ArmAgent = app.state.agent
        loop = asyncio.get_running_loop()

        async def understand(key: str | None, why: Why | None, shared: bool) -> AgentResult:
            """`shared`: our own key, which keeps its connection to Token Factory open
            between commands. A visitor's own key never does (it isn't kept at all)."""
            no_key_why = NO_KEY_WHY[why or "no_key"]
            return await loop.run_in_executor(
                None, partial(agent.understand, texts, key, context, no_key_why, shared)
            )

        if settings.ai == "rules":
            own = None
        res = AgentResult([], source="rules", model="rules", errors=["no words"])
        refused = False
        mode: Mode = "own"
        why: Why | None = None
        t = time.perf_counter()
        if texts and own:
            res = await understand(own, None, shared=False)
            refused = res.key_refused
        if refused or not own:
            key, why = our_key(sid, addr)
            mode = "ours" if key else "simple"
            if texts:
                res = await understand(key, why, shared=True)
                if key and res.asked_model:
                    limits.spend(sid, addr)
            key = None
        return res, mode, why, refused, time.perf_counter() - t

    def shadow(texts: list[str], context: Context, calls: list[dict[str, Any]]) -> dict[str, Any]:
        """What the simple word rules alone would have done with the same words (item 18):
        the rule parser runs in well under a millisecond, beside Nemotron."""
        agent: ArmAgent = app.state.agent
        plain = rules(agent.profile, texts, context) if texts else None
        plain_calls = [c.to_dict() for c in plain.calls] if plain else []
        return {
            "understood": bool(plain_calls),
            "words": in_words(plain_calls),
            "calls": plain_calls,
            "same": plain_calls == calls,
        }

    def model_trace(tr: trace.Trace, texts: list[str], things: list[dict[str, Any]],
                    res: AgentResult, agent_s: float,
                    word_rules: dict[str, Any]) -> None:  # fmt: skip
        """The Nemotron rows and the grey word-matcher row under them."""
        agent: ArmAgent = app.state.agent
        drove = res.source == "rules"
        off = (res.errors[0] if res.errors else "it did not answer") if drove else None
        tr.add(*trace.model_rows(texts, things, agent.tools, res, agent_s, off))
        tr.add(trace.shadow_row(word_rules, drove))
        if res.refusal:
            tr.add(trace.refusal_row(res.refusal))

    @app.post("/api/command", response_model=None)
    async def command(
        request: Request,
        response: Response,
        meta: Annotated[str, Form()],
        audio: Annotated[UploadFile | None, File()] = None,
        x_nebius_key: Annotated[str | None, Header()] = None,
        sid: Annotated[str | None, Cookie()] = None,
    ) -> dict[str, Any] | JSONResponse:
        """One command (plan 4.2, 7.2): words (from the clip, or typed), the agent's tool
        calls, the core's places over MCP. A visitor's own key (`X-Nebius-Key`) is used for
        this request's Nemotron call only; it is never stored, logged or sent back. Our key
        is used within the day's limits (plan 8.1), else the rule parser (simple mode).

        The model may answer with a plan of up to four steps. Only the first is resolved
        here; the rest come back in `plan` and the page asks for each one with `/api/step`
        just before the arm runs it."""
        t_start = time.perf_counter()
        sid = visitor(response, sid)
        addr = address(request)
        own = (x_nebius_key or "").strip() or None
        x_nebius_key = None
        if own is not None and len(own) > MAX_KEY_CHARS:
            raise HTTPException(400, "that key is too long")
        wait = limits.wait_s(sid, addr)
        if wait > 0:  # the burst limit (R9); the page waits and sends it again
            busy = JSONResponse(
                {"detail": "one command at a time, please", "wait_s": wait}, status_code=429,
                headers={"Retry-After": str(math.ceil(wait))},
            )  # fmt: skip
            for cookie in response.headers.getlist("set-cookie"):
                busy.headers.append("set-cookie", cookie)
            return busy
        if len(meta) > MAX_META_BYTES:
            raise HTTPException(413, "meta is over 1 MB")
        try:
            m = Meta.model_validate_json(meta)
        except ValidationError as e:
            raise HTTPException(400, f"meta: {e.errors()[0]['msg']}") from None
        timings: dict[str, float] = {}
        audio_s: float | None = None
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
            audio_s = max(0.0, (len(data) - 44) / (16000 * 2))
            words = [{"text": w.text, "t0": round(w.t0, 4), "t1": round(w.t1, 4)} for w in heard]
        elif m.text and m.text.strip():
            words = typed_words(m.text, m.holds, m.t)
        else:
            raise HTTPException(400, "send a clip (audio) or a typed sentence (meta.text)")

        # The agent: Nemotron with the visitor's own key, or ours within the day's limits;
        # else the rule parser (simple mode). A refused own key: as if none was sent.
        texts = [w["text"] for w in words]
        # While a question is open, the agent is given the `pick` tool with the choices the
        # page glows, so Nemotron answers the question itself (rahul, 11 Oct).
        open_ask = sessions.get(sid) if m.state.question else None
        context = Context(
            previewing=m.state.previewing, question=m.state.question,
            choices=choices_for(open_ask.response["ask"]) if open_ask else [],
        )  # fmt: skip
        agent_res, mode, why, key_refused, agent_s = await ask_model(sid, addr, own, texts, context)
        own = None
        timings["agent_s"] = round(agent_s, 3)

        # The core: every reference to a place or a block, over MCP. Only the plan's first
        # step; the page asks for each later one with `/api/step` when it comes to it.
        calls = [c.to_dict() for c in agent_res.calls]
        first = calls[:1]
        result = error = None
        request: dict[str, Any] = {}
        notes: list[str] = []
        # The answer to an open question: Nemotron's `pick`, else the words read on the
        # server (a plain "the red one" when the model gave no call at all).
        entry = None
        picked = False
        pick = agent_res.pick
        if calls and pick is not None:
            notes.append("answered the question and gave a command: the command wins")
        elif pick is not None and open_ask is not None:
            ask = open_ask.response["ask"]
            if pick.choice is not None:  # they named a choice: the words said which
                entry = answer_entry(ask, pick.choice)
            else:  # they only pointed: the core says which, from the finger
                entry = await pointed_answer(ask, m, words, pick.word_index or 0)
                if entry is None:
                    notes.append("pointed at no choice")
            picked = entry is not None
        elif not calls:
            entry = await spoken_answer(sid, m, words)
        t = time.perf_counter()
        if entry is not None:
            result, error = await answer_open(sid, entry)
            if not picked:  # a `pick` stays Nemotron's: it is the model's own tool call
                agent_res.source = "answer"
            say = result["ask"]["question"] if result and result["status"] == "ask" else None
        else:
            if first:
                moment = {"words": words, "samples": m.samples, "holds": m.holds}
                request, result, error, more = await resolve_one(
                    moment, m.things, m.context, first, m.answers
                )
                notes += more
                remember(sid, request, result)
            # Nothing understood (a mis-heard word, or they pointed at no choice): the
            # robot's open question stays open, so they can simply answer it again.
            say = "Sorry, something went wrong." if error else say_for(result, first)
        if agent_res.refusal:
            say = agent_res.refusal
        if result is not None or entry is not None or first:
            timings["core_s"] = round(time.perf_counter() - t, 3)
        # The plan, kept until it is done: the page asks for step 1 onwards when it gets there.
        if len(calls) > 1:
            plans.put(sid, OpenPlan(words, m.samples, m.holds, calls, m.context))
        else:
            plans.drop(sid)
        timings["total_s"] = round(time.perf_counter() - t_start, 3)
        whose = f"{mode} key" if mode != "simple" else f"simple: {why}"
        if key_refused:
            whose = f"own key refused, {whose}"
        log_command(words, agent_res, result, error, notes, whose)
        word_rules = shadow(texts, context, calls)
        tr = trace.Trace()
        tr.add(trace.pointing_row(len(m.samples), len(m.holds), audio is None))
        tr.add(*trace.heard_rows(words, audio is None, audio_s, timings.get("asr_s")))
        model_trace(tr, texts, m.things, agent_res, timings["agent_s"], word_rules)
        tr.add(trace.step_row(0, len(calls)))
        tr.add(*trace.core_rows(first, result, error, timings.get("core_s")))
        cost = cost_usd(agent_res.usage, settings.price_in_per_m, settings.price_out_per_m)
        record.write(
            "command", sid, said=" ".join(texts), typed=audio is None, mode=mode,
            source=agent_res.source, model=agent_res.model, attempts=agent_res.attempts,
            calls=calls, steps=len(calls), refusal=agent_res.refusal,
            pick=agent_res.pick.to_dict() if agent_res.pick else None,
            status=(result or {}).get("status") if not error else "error",
            ask=((result or {}).get("ask") or {}).get("reason"), error=error,
            timings=timings, usage=agent_res.usage, cost_usd=cost,
            word_rules={"understood": word_rules["understood"], "same": word_rules["same"]},
            last_word_t1=words[-1]["t1"] if words else None,
        )  # fmt: skip
        return {
            "words": words,
            "word_rules": word_rules,
            "cost_usd": cost,
            "agent": agent_res.summary(),
            "plan": {"calls": calls, "steps": len(calls)},
            "result": result,
            "error": error,
            "notes": notes,
            "say": say,
            "trace": tr.to_list(),
            "timings": timings,
            "limits": limits_now(sid, addr, mode, why),
            "key_refused": key_refused,
        }

    @app.post("/api/event")
    def event(
        body: EventBody, response: Response, sid: Annotated[str | None, Cookie()] = None
    ) -> dict[str, bool]:
        """What happened on the page after a command: the guess shown (and how long after the
        last word), refused, yes, done, no, stop... Only logged (M30, M31)."""
        if body.kind not in EVENTS:
            raise HTTPException(400, "unknown event")
        sid = visitor(response, sid)
        data = body.model_dump(exclude_none=True, exclude={"kind"})
        if "why" in data:
            data["why"] = data["why"][:120]
        record.write("event", sid, event=body.kind, **data)
        return {"ok": True}

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
        tr = trace.Trace()
        tr.add(trace.Row("you", "page", f"you picked choice {body.candidate}", kind="note"))
        tr.add(*trace.core_rows(open_ask.request["calls"], result, error, timings["total_s"]))
        return {"result": result, "error": error, "say": say, "timings": timings,
                "trace": tr.to_list()}  # fmt: skip

    @app.post("/api/step")
    async def step(
        body: StepBody, response: Response, sid: Annotated[str | None, Cookie()] = None
    ) -> dict[str, Any]:
        """One step of the running plan, measured now (advanced-commands-plan.md section 6).

        The page sends the table as it is at this moment, so step 2 is resolved against what
        step 1 left behind and not against the scene the plan was written in. The words and
        the pointing come from the stored plan: "there" still means where the finger was when
        the person said it."""
        t_start = time.perf_counter()
        sid = visitor(response, sid)
        plan = plans.get(sid)
        if plan is None:
            raise HTTPException(409, "no plan is running (it may have expired)")
        if body.index >= plan.steps:
            raise HTTPException(400, f"no step {body.index}: the plan has {plan.steps}")
        if len(body.things) > 20 or len(plan.samples) > MAX_SAMPLES:
            raise HTTPException(413, "too much scene")
        call = plan.calls[body.index]
        context = body.context or plan.context
        request, result, error, notes = await resolve_one(
            plan.moment(), body.things, context, [call]
        )
        remember(sid, request, result)
        timings = {"core_s": round(time.perf_counter() - t_start, 3)}
        timings["total_s"] = timings["core_s"]
        say = "Sorry, something went wrong." if error else say_for(result, [call])
        tr = trace.Trace()
        tr.add(trace.step_row(body.index, plan.steps))
        tr.add(*trace.core_rows([call], result, error, timings["core_s"], again=body.index > 0))
        log.info("step %d of %d: %s -> %s", body.index + 1, plan.steps, call["name"],
                 error or (result or {}).get("status"))  # fmt: skip
        return {"index": body.index, "steps": plan.steps, "result": result, "error": error,
                "notes": notes, "say": say, "trace": tr.to_list(), "timings": timings}  # fmt: skip

    @app.post("/api/replan")
    async def replan(
        request: Request,
        response: Response,
        body: ReplanBody,
        x_nebius_key: Annotated[str | None, Header()] = None,
        sid: Annotated[str | None, Cookie()] = None,
    ) -> dict[str, Any]:
        """A step could not be done: give Nemotron the plan, the step that failed and the
        arm's own reason, and take a new plan for the rest (job 2 of section 1).

        One re-plan per command, so a model that keeps failing cannot run away. The words are
        the ones the person actually said: nothing is put in their mouth."""
        t_start = time.perf_counter()
        sid = visitor(response, sid)
        addr = address(request)
        own = (x_nebius_key or "").strip() or None
        x_nebius_key = None
        if own is not None and len(own) > MAX_KEY_CHARS:
            raise HTTPException(400, "that key is too long")
        plan = plans.get(sid)
        if plan is None:
            raise HTTPException(409, "no plan is running (it may have expired)")
        if plan.replans >= MAX_REPLANS:
            raise HTTPException(409, "already re-planned once for this command")
        texts = plan.texts()
        context = Context(plan=call_lines(plan.calls), failed_step=body.index,
                          failed_why=body.why)  # fmt: skip
        res, mode, why, key_refused, agent_s = await ask_model(sid, addr, own, texts, context)
        own = None
        calls = [c.to_dict() for c in res.calls]
        timings = {"agent_s": round(agent_s, 3)}
        result = error = None
        notes: list[str] = []
        say = res.refusal
        if calls:
            t = time.perf_counter()
            moment = plan.moment()
            req, result, error, notes = await resolve_one(
                moment, body.things, body.context or plan.context, calls[:1]
            )
            remember(sid, req, result)
            timings["core_s"] = round(time.perf_counter() - t, 3)
            if not say:
                say = "Sorry, something went wrong." if error else say_for(result, calls[:1])
            plans.put(sid, OpenPlan(plan.words, plan.samples, plan.holds, calls,
                                    body.context or plan.context, plan.replans + 1))  # fmt: skip
        else:
            plans.drop(sid)
        timings["total_s"] = round(time.perf_counter() - t_start, 3)
        log_command(plan.words, res, result, error, [f"re-plan after step {body.index + 1}"],
                    f"{mode} key" if mode != "simple" else f"simple: {why}")  # fmt: skip
        word_rules = shadow(texts, context, calls)
        tr = trace.Trace()
        tr.add(trace.Row("arm", "page", f"step {body.index + 1} failed: {body.why}", kind="note"))
        model_trace(tr, texts, body.things, res, timings["agent_s"], word_rules)
        tr.add(trace.step_row(0, len(calls)))
        tr.add(*trace.core_rows(calls[:1], result, error, timings.get("core_s")))
        record.write("replan", sid, said=" ".join(texts), step=body.index, why=body.why[:120],
                     source=res.source, model=res.model, calls=calls, steps=len(calls),
                     refusal=res.refusal, timings=timings, usage=res.usage)  # fmt: skip
        return {
            "words": plan.words,
            "agent": res.summary(),
            "plan": {"calls": calls, "steps": len(calls)},
            "result": result,
            "error": error,
            "notes": notes,
            "say": say,
            "trace": tr.to_list(),
            "timings": timings,
            "limits": limits_now(sid, addr, mode, why),
            "key_refused": key_refused,
        }

    if settings.web_dist.is_dir():
        app.mount("/", StaticFiles(directory=settings.web_dist, html=True), name="page")

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Only warnings from the HTTP libraries: their request lines are noise, and their debug
    # lines carry request details. Ours never carry a key (plan 8.2).
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    settings = Settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
