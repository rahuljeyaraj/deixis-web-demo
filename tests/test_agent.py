"""The arm's agent and the command routes (plan W8): the rule parser on the arm's phrases,
Nemotron through a fake OpenAI client (tool calls, calls written as text, a retry, failures),
and `/api/command` and `/api/answer` end to end over MCP."""

from __future__ import annotations

import io
import json
import logging
import wave
from types import SimpleNamespace as NS

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from conftest import VOICE, needs_parakeet, needs_voice
from deixis.actions import tools_for_agent
from deixis.profile import load_profile
from test_arm_core import BLOCKS, CENTRES, HERE
from view import Finger
from webdemo.agent import ArmAgent, Context, parse, rules
from webdemo.app import create_app
from webdemo.command import NOT_UNDERSTOOD, typed_words
from webdemo.settings import ROOT

PROFILE = load_profile(ROOT / "profiles" / "so101-table.json")
TOOLS = tools_for_agent(PROFILE)


def that(i: int, word: str = "that") -> dict:
    return {"deixis": word, "word_index": i}


# --- the rule parser ---

PHRASES = [
    ("put that here", "move", {"object": that(1), "place": that(2, "here")}),
    ("put the blue one next to that", "move",
     {"object": {"label": "blue one"}, "place": {"relation": "next_to", "target": that(6)}}),
    ("put that on that", "move",
     {"object": that(1), "place": {"relation": "on", "target": that(3)}}),
    ("put this on top of that", "move",
     {"object": that(1, "this"), "place": {"relation": "on", "target": that(5)}}),
    ("put it there", "move", {"object": {"last": True}, "place": that(2, "there")}),
    ("Put the red block here.", "move", {"object": {"label": "red one"}, "place": that(4, "here")}),
    ("put the green box here", "move",
     {"object": {"label": "green one"}, "place": that(4, "here")}),
    ("put that on the blue cube", "move",
     {"object": that(1), "place": {"relation": "on", "target": {"label": "blue one"}}}),
    ("put that to the left of the blue one", "move",
     {"object": that(1), "place": {"relation": "left_of", "target": {"label": "blue one"}}}),
    ("put that behind the green one", "move",
     {"object": that(1), "place": {"relation": "behind", "target": {"label": "green one"}}}),
    ("pick up that and put it here", "move", {"object": that(2), "place": that(6, "here")}),
    ("move it a bit left", "adjust", {"dx": -1}),
    ("a bit to the right", "adjust", {"dx": 1}),
    ("a lot further away", "adjust", {"dy": 3}),
    ("yes", "confirm", {}),
    ("okay go ahead", "confirm", {}),
    ("no", "cancel", {}),
    ("never mind", "cancel", {}),
    ("stop", "stop", {}),
    ("wait", "stop", {}),
    ("go home", "home", {}),
]  # fmt: skip


@pytest.mark.parametrize(("text", "name", "args"), PHRASES, ids=[p[0] for p in PHRASES])
def test_rule_parser(text, name, args):
    res = rules(PROFILE, text.split())
    assert res.errors == [] and len(res.calls) == 1
    call = res.calls[0].to_dict()
    assert call["name"] == name
    for k, v in args.items():  # adjust also carries its defaults (dy=0, scale=1)
        assert call["args"][k] == v


@pytest.mark.parametrize("text", ["this one", "the red one", "hello there robot", ""])
def test_rule_parser_leaves_answers_and_noise(text):
    assert parse(text.split()) == []


def test_typed_words_take_the_clicks_times():
    clicks = [{"t0": 10.0, "t1": 10.3}, {"t0": 12.0, "t1": 12.3}]
    w = typed_words("put that right here", clicks, None)
    assert [x["text"] for x in w] == ["put", "that", "right", "here"]
    assert w[1]["t0"] == 10.15 and w[3]["t0"] == 12.15  # the k-th pointing word, the k-th click
    assert w[0]["t0"] < w[1]["t0"] < w[2]["t0"] < w[3]["t0"]
    assert [x["t0"] for x in typed_words("yes", [], 50.0)] == [49.75]


# --- Nemotron, through a fake client ---


class FakeLLM:
    """An OpenAI-like client: answers in turn from `replies` (a list of tool calls as
    (name, args) pairs, a text string, or an exception); remembers each key and request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.keys: list[str] = []
        self.requests: list[dict] = []
        self.chat = NS(completions=NS(create=self._create))

    def factory(self, key: str):
        self.keys.append(key)
        return self

    def _create(self, **kw):
        self.requests.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        if isinstance(r, str):
            msg = NS(content=r, tool_calls=None)
        else:
            calls = [NS(function=NS(name=n, arguments=json.dumps(a))) for n, a in r]
            msg = NS(content=None, tool_calls=calls)
        return NS(choices=[NS(message=msg)])


def agent(llm: FakeLLM) -> ArmAgent:
    return ArmAgent(PROFILE, TOOLS, "https://example.invalid/v1/", client_factory=llm.factory)


def test_nemotron_tool_calls():
    llm = FakeLLM([("move", {"object": that(1), "place": that(2, "here")})])
    res = agent(llm).understand(["put", "that", "here"], "sk-ours")
    assert res.source == "nemotron" and res.attempts == 1 and res.errors == []
    assert str(res.calls[0]).startswith("move(")
    req = llm.requests[0]
    assert req["tools"] == TOOLS and req["temperature"] == 0.0
    assert req["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "0:put 1:that 2:here" in req["messages"][1]["content"]


def test_nemotron_calls_written_as_text_and_the_context():
    llm = FakeLLM("confirm({})")
    res = agent(llm).understand(["confirm", "it"], "sk-ours", Context(previewing=True))
    assert res.source == "nemotron" and [c.name for c in res.calls] == ["confirm"]
    assert "waits for the person's yes or no" in llm.requests[0]["messages"][1]["content"]


def test_a_bad_call_is_fixed_on_the_retry():
    llm = FakeLLM(
        [("move", {"object": that(2), "place": that(2, "here")})],  # word 2 isn't "that"
        [("move", {"object": that(1), "place": that(2, "here")})],
    )
    res = agent(llm).understand(["put", "that", "here"], "sk-ours")
    assert res.source == "nemotron" and res.attempts == 2 and res.errors  # seen, then fixed
    # The errors go back as one more user turn (system, the words, the errors), never as a
    # made-up assistant turn: that made the model answer in text instead of tool calls.
    msgs = llm.requests[1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "user"]
    assert "You replied with:\nmove(" in msgs[2]["content"]
    assert "word 2 is 'here'" in msgs[2]["content"]


def test_a_timeout_falls_back_to_the_rules_without_the_key():
    llm = FakeLLM(TimeoutError("timed out talking to sk-visitor-123456789"))
    res = agent(llm).understand(["put", "that", "here"], "sk-visitor-123456789")
    assert res.source == "rules" and [c.name for c in res.calls] == ["move"]
    assert "sk-visitor" not in json.dumps(res.summary())


def test_a_refused_key_says_so():
    err = Exception("Incorrect API key provided: sk-vis****789")
    err.status_code = 401
    res = agent(FakeLLM(err)).understand(["confirm", "it"], "sk-visitor-123456789")
    assert res.source == "rules" and res.errors[0] == "the Nebius key was refused"


def test_even_stop_and_yes_go_to_the_model():
    """Nothing is decided behind Nemotron's back (rahul, 8 Oct): the rule parser only
    answers when the model couldn't. The arm's instant stop is the page's Stop button."""
    for said, want in (("stop stop", "stop"), ("yes", "confirm"), ("no", "cancel")):
        llm = FakeLLM(f"{want}({{}})")
        res = agent(llm).understand(said.split(), "sk-ours")
        assert res.source == "nemotron" and [c.name for c in res.calls] == [want]
        assert len(llm.requests) == 1


def test_our_key_keeps_one_connection_and_a_visitors_key_never_does():
    """A fresh TLS handshake to Token Factory costs 2-11 s on a home line and was timing
    the model out (rahul, 8 Oct), so our own key keeps one client between commands. A
    visitor's key still gets a client for that call only: it is never kept."""
    llm = FakeLLM("confirm({})", "confirm({})", "confirm({})", "confirm({})")
    a = agent(llm)
    a.understand(["yes"], "sk-ours", shared=True)
    a.understand(["yes"], "sk-ours", shared=True)
    assert llm.keys == ["sk-ours"] and len(llm.requests) == 2
    a.understand(["yes"], "sk-visitor-123456789", shared=False)
    a.understand(["yes"], "sk-visitor-123456789", shared=False)
    assert llm.keys == ["sk-ours", "sk-visitor-123456789", "sk-visitor-123456789"]


def test_the_retry_is_dropped_once_the_person_has_waited_long_enough():
    """One slow attempt must not cost the person a second wait: past the budget the agent
    keeps what it has (here: nothing, so the rule parser answers)."""
    llm = FakeLLM([("move", {"object": that(2), "place": that(2, "here")})])  # a bad call
    a = ArmAgent(PROFILE, TOOLS, "https://example.invalid/v1/", budget_s=0.0,
                 client_factory=llm.factory)  # fmt: skip
    res = a.understand(["put", "that", "here"], "sk-ours", shared=True)
    assert res.attempts == 1 and len(llm.requests) == 1  # no second go
    assert res.source == "rules" and [c.name for c in res.calls] == ["move"]


def test_no_key_is_simple_mode():
    res = agent(FakeLLM()).understand(["confirm", "it"], None)
    assert res.source == "rules" and res.errors[0] == "no Nebius key: simple mode"


# --- the routes, end to end over MCP ---


def post(api, meta: dict, audio: bytes | None = None, **kw):
    files = {"audio": ("clip.wav", audio, "audio/wav")} if audio else None
    r = api.post("/api/command", data={"meta": json.dumps(meta)}, files=files, **kw)
    assert r.status_code == 200, r.text
    return r.json()


def click(t: float, x: float, y: float) -> dict:
    return {"t0": t - 0.15, "t1": t + 0.15, "mean": [x, y], "spread": [0, 0],
            "frame": "table", "source": "click"}  # fmt: skip


def test_typed_with_clicks_put_that_here(rules_api):
    meta = {"text": "put that here", "holds": [click(10, *CENTRES["red"]), click(11.5, *HERE)],
            "things": BLOCKS, "t": 12.0}  # fmt: skip
    body = post(rules_api, meta)
    assert body["agent"]["source"] == "rules" and body["error"] is None
    res = body["result"]
    assert res["status"] == "ok"
    args = res["calls"][0]["args"]
    assert args["object"]["thing"] == "red"
    assert args["place"]["mean_mm"] == pytest.approx(list(HERE), abs=0.5)
    assert body["say"] is None  # the page says "This one, here?" after its reach check
    assert {"agent_s", "core_s", "total_s"} <= set(body["timings"])


def test_not_understood(rules_api):
    body = post(rules_api, {"text": "hello robot", "things": BLOCKS})
    assert body["result"] is None and body["say"] == NOT_UNDERSTOOD


def test_ask_then_click_the_answer(rules_api):
    between = ((CENTRES["red"][0] + CENTRES["green"][0]) / 2, CENTRES["red"][1])
    meta = {"text": "put that here", "holds": [click(10, *between), click(11.5, *HERE)],
            "things": BLOCKS}  # fmt: skip
    first = post(rules_api, meta)
    assert first["result"]["status"] == "ask" and first["say"] == "This one or that one?"
    things = [c["thing"] for c in first["result"]["ask"]["candidates"]]
    assert set(things) == {"red", "green"}
    r = rules_api.post("/api/answer", json={"candidate": things.index("green")})
    assert r.status_code == 200, r.text
    second = r.json()["result"]
    assert second["status"] == "ok" and second["calls"][0]["args"]["object"]["thing"] == "green"
    # The question is closed now.
    assert rules_api.post("/api/answer", json={"candidate": 0}).status_code == 409


@pytest.mark.parametrize("how", ["the green one", "this one"])
def test_ask_then_say_the_answer(rules_api, how):
    between = ((CENTRES["red"][0] + CENTRES["green"][0]) / 2, CENTRES["red"][1])
    meta = {"text": "put that here", "holds": [click(10, *between), click(11.5, *HERE)],
            "things": BLOCKS}  # fmt: skip
    first = post(rules_api, meta)
    assert first["result"]["status"] == "ask"
    # Spoken: by colour, or "this one" while clicking (pointing at) the green block.
    again = {"text": how, "things": BLOCKS, "holds": [click(20, *CENTRES["green"])], "t": 21,
             "state": {"question": first["say"]}}  # fmt: skip
    second = post(rules_api, again)
    assert second["agent"]["source"] == "answer"
    res = second["result"]
    assert res["status"] == "ok" and res["calls"][0]["args"]["object"]["thing"] == "green"
    assert res["calls"][0]["args"]["place"]["mean_mm"] == pytest.approx(list(HERE), abs=0.5)


def test_where_then_here_while_pointing(rules_api):
    first = post(rules_api, {"text": "put that here", "things": BLOCKS, "t": 30,
                             "holds": [click(28, *CENTRES["red"])]})  # fmt: skip
    assert first["result"]["status"] == "ask", first
    assert first["say"] == "Where? Point at the spot."
    again = {"text": "here", "things": BLOCKS, "holds": [click(40, *HERE)], "t": 41,
             "state": {"question": first["say"]}}  # fmt: skip
    second = post(rules_api, again)
    assert second["agent"]["source"] == "answer", second
    res = second["result"]
    assert res["status"] == "ok" and res["calls"][0]["args"]["object"]["thing"] == "red"
    assert res["calls"][0]["args"]["place"]["mean_mm"] == pytest.approx(list(HERE), abs=0.5)


def test_answer_needs_an_open_question(rules_api):
    assert rules_api.post("/api/answer", json={"candidate": 0}).status_code == 409


def test_the_visitors_key_is_used_for_their_call_only(settings, core, caplog):
    """Our key by default; a visitor's own key for their request only; neither is ever in a
    response or a log line. (The full own-key panel and caps are W11.)"""
    settings.nebius_api_key = SecretStr("sk-ours-0000000000")
    reply = [("confirm", {})]
    llm = FakeLLM(reply, reply, TimeoutError("boom sk-visitor-9999999999 boom"))
    caplog.set_level(logging.DEBUG)
    with TestClient(create_app(settings, client=core, llm_factory=llm.factory)) as c:
        a = c.post("/api/command", data={"meta": json.dumps({"text": "confirm it"})})
        b = c.post("/api/command", data={"meta": json.dumps({"text": "confirm it"})},
                   headers={"X-Nebius-Key": "sk-visitor-9999999999"})  # fmt: skip
        d = c.post("/api/command", data={"meta": json.dumps({"text": "confirm it"})},
                   headers={"X-Nebius-Key": "sk-visitor-9999999999"})  # fmt: skip
    assert llm.keys == ["sk-ours-0000000000", "sk-visitor-9999999999", "sk-visitor-9999999999"]
    assert a.json()["agent"]["source"] == b.json()["agent"]["source"] == "nemotron"
    assert d.json()["agent"]["source"] == "rules"  # the timeout fell back
    for r in (a, b, d):
        assert "sk-ours" not in r.text and "sk-visitor" not in r.text
    assert "sk-ours" not in caplog.text and "sk-visitor" not in caplog.text


@needs_parakeet
@needs_voice
def test_spoken_put_that_here_with_the_finger(rules_api):
    """The whole server path: a voice clip and fingertip samples on one clock."""
    item = json.loads((ROOT / "tests" / "fixtures" / "voice.json").read_text())["items"][0]
    assert item["text"] == "put that here"
    with wave.open(str(VOICE / item["wav"]), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    x = np.concatenate([np.zeros(8000, np.float32), x, np.zeros(6400, np.float32)])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    audio_t0 = 20.0
    t_that = audio_t0 + 0.5 + item["words"][1]["t0"]
    t_here = audio_t0 + 0.5 + item["words"][2]["t0"]
    # The finger holds on the red block around "that", then on the spot around "here".
    f = Finger(CENTRES, t=t_that - 1.2).at_block("red").hold(1.2 - 0.15)
    f.move(HERE, speed=1500).hold(max(0.6, t_here - f.t + 0.4))
    meta = {"audio_t0": audio_t0, "samples": f.samples, "things": BLOCKS}
    body = post(rules_api, meta, buf.getvalue())
    assert [w["text"].lower() for w in body["words"]] == ["put", "that", "here"]
    res = body["result"]
    assert res["status"] == "ok", res
    args = res["calls"][0]["args"]
    assert args["object"]["thing"] == "red"
    assert args["place"]["mean_mm"] == pytest.approx(list(HERE), abs=3)


def test_the_green_box_is_the_green_block(rules_api):
    """rahul, 5 Oct: "green box" wasn't understood (the blocks are called blocks)."""
    for name in ("box", "cube", "block"):
        meta = {"text": f"put the green {name} here", "holds": [click(11.5, *HERE)],
                "things": BLOCKS, "t": 12.0}  # fmt: skip
        res = post(rules_api, meta)["result"]
        assert res["status"] == "ok", (name, res)
        assert res["calls"][0]["args"]["object"]["thing"] == "green"


def test_here_pointed_onto_a_block_stacks_on_it(rules_api):
    """rahul, 5 Oct: "put that here" pointing at a block said "a block in the way"; it means
    on top of it. Off centre, and onto a stack of two (the top block)."""
    meta = {"text": "put that here", "things": BLOCKS, "t": 12.0,
            "holds": [click(10, *CENTRES["red"]), click(11.5, 306, 84)]}  # fmt: skip
    body = post(rules_api, meta)
    place = body["result"]["calls"][0]["args"]["place"]
    assert place["evidence"]["target"] == "blue" and place["z_mm"] == 30
    assert place["mean_mm"] == pytest.approx([300, 90], abs=0.5)
    assert body["notes"] == ["pointed onto a block: on top of it"]
    # Red already on blue: green onto the stack lands on red, 60 mm up.
    stacked = [dict(b) for b in BLOCKS]
    for b in stacked:
        if b["id"] == "red":
            b.update(box_mm=[285, 75, 30, 30], height_mm=60)
        if b["id"] == "blue":
            b.update(selectable=False)
    meta = {"text": "put that here", "things": stacked, "t": 12.0,
            "holds": [click(10, *CENTRES["green"]), click(11.5, 296, 93)]}  # fmt: skip
    place = post(rules_api, meta)["result"]["calls"][0]["args"]["place"]
    assert place["evidence"]["target"] == "red" and place["z_mm"] == 60
    # A spot on the table next to a block stays a spot.
    meta = {"text": "put that here", "things": BLOCKS, "t": 12.0,
            "holds": [click(10, *CENTRES["red"]), click(11.5, *HERE)]}  # fmt: skip
    body = post(rules_api, meta)
    assert body["notes"] == [] and body["result"]["calls"][0]["args"]["place"]["z_mm"] is None
