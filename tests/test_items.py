"""Web demo build items (final/web-demo.md): a plain yes/no without the model (12), tokens
and cost (11), the "Simple word rules" line (18), the command and event log (M30, M31, M33),
the own-key box hidden (16), and Nemotron's prompt knows "between" (the chips)."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from test_agent import TOOLS, FakeLLM, click, post, that
from test_arm_core import BLOCKS, CENTRES, HERE
from webdemo.agent import SYSTEM, ArmAgent, plain_yes_no
from webdemo.app import create_app
from webdemo.record import cost_usd, in_words, visitor_tag
from webdemo.settings import ROOT


@pytest.mark.parametrize(
    ("words", "plain"),
    [("yes", True), ("Yes please", True), ("no", True), ("okay", True), ("go ahead", True),
     ("no there", False), ("yes put it here", False), ("go home", False), ("", False),
     ("confirm it", False)],
)  # fmt: skip
def test_plain_yes_no(words, plain):
    assert plain_yes_no(words.split()) is plain


class UsageLLM(FakeLLM):
    """FakeLLM whose replies carry Token Factory's token counts."""

    def _create(self, **kw):
        r = super()._create(**kw)
        r.usage = NS(prompt_tokens=7000, completion_tokens=60,
                     prompt_tokens_details=NS(cached_tokens=4000))  # fmt: skip
        return r


def app_with(settings, core, llm) -> TestClient:
    settings.nebius_api_key = SecretStr("sk-ours-0000000000")
    return TestClient(create_app(settings, client=core, llm_factory=llm.factory))


def test_yes_never_asks_the_model_and_a_command_costs(settings, core):
    llm = UsageLLM([("move", {"object": that(1), "place": that(2, "here")})])
    with app_with(settings, core, llm) as c:
        meta = {"text": "put that here", "holds": [click(10, *CENTRES["red"]), click(11.5, *HERE)],
                "things": BLOCKS, "t": 12.0}  # fmt: skip
        body = post(c, meta)
        yes = post(c, {"text": "yes", "things": BLOCKS})
    assert body["agent"]["source"] == "nemotron"
    assert body["agent"]["usage"] == {"prompt": 7000, "cached": 4000, "completion": 60}
    assert body["cost_usd"] == pytest.approx((7000 * 0.06 + 60 * 0.24) / 1e6)
    assert yes["agent"]["source"] == "rules" and len(llm.keys) == 1  # yes: no model call
    assert yes["agent"]["calls"][0]["name"] == "confirm"


def test_word_rules_line_and_the_log(settings, core):
    """'pop that one on top of the red one': Nemotron gets it; the simple word rules don't
    ('pop' is not a verb they know). Both are in the reply and the log."""
    call = (
        "move",
        {"object": that(1), "place": {"relation": "on", "target": {"label": "red one"}}},
    )
    llm = UsageLLM([call])
    with app_with(settings, core, llm) as c:
        meta = {"text": "pop that one on top of the red one", "things": BLOCKS, "t": 12.0,
                "holds": [click(11, *CENTRES["blue"])]}  # fmt: skip
        body = post(c, meta)
        ev = c.post("/api/event", json={"kind": "preview", "latency_s": 2.4, "block": "blue"})
        bad = c.post("/api/event", json={"kind": "dance"})
    wr = body["word_rules"]
    assert body["agent"]["source"] == "nemotron"
    assert wr == {"understood": False, "words": "not understood", "calls": [], "same": False}
    assert ev.status_code == 200 and bad.status_code == 400
    rows = [json.loads(x) for x in settings.command_log.read_text().splitlines()]
    cmd, preview = rows
    assert cmd["kind"] == "command" and cmd["said"] == "pop that one on top of the red one"
    assert cmd["usage"]["prompt"] == 7000 and cmd["cost_usd"] > 0 and cmd["status"] == "ok"
    assert cmd["word_rules"] == {"understood": False, "same": False}
    assert preview == {**preview, "kind": "event", "event": "preview", "latency_s": 2.4}
    assert cmd["visitor"] == preview["visitor"] and len(cmd["visitor"]) == 10
    text = settings.command_log.read_text()
    assert "sk-ours" not in text and "testclient" not in text  # no key, no address


def test_in_words():
    assert in_words([]) == "not understood"
    assert in_words([{"name": "move", "args": {"object": {"last": True},
                     "place": {"deixis": "here", "word_index": 5}}}]) == (
        "move the block moved last (“it”) to where you pointed (“here”)")  # fmt: skip
    between = {"relation": "between", "target": {"label": "green one"},
               "target2": {"label": "blue one"}}  # fmt: skip
    move = {"name": "move", "args": {"object": {"label": "red one"}, "place": between}}
    assert in_words([move]) == "move the red one between the green one and the blue one"
    assert cost_usd({"prompt": 1_000_000, "completion": 1_000_000}, 0.06, 0.24) == 0.30
    assert visitor_tag("a") != visitor_tag("b")


def test_own_key_box_is_hidden_by_default(api):
    assert api.get("/api/info").json()["show_own_key"] is False


def test_the_prompt_knows_between():
    assert '"relation": "between"' in SYSTEM and "target2" in SYSTEM


def test_between_resolves_in_the_core(rules_api, core):
    """The chip 'put the red cube between the green and the blue': the core places it."""
    req = {"words": [{"text": w, "t0": i, "t1": i + 0.2} for i, w in enumerate(
        "put the red cube between the green and the blue".split())],
        "things": BLOCKS, "samples": [], "holds": [],
        "calls": [{"name": "move", "args": {"object": {"label": "red one"}, "place": {
            "relation": "between", "target": {"label": "green one"},
            "target2": {"label": "blue one"}}}}]}  # fmt: skip
    r = core.resolve(req).model_dump(mode="json")
    assert r["status"] == "ok", r
    mean = r["calls"][0]["args"]["place"]["mean_mm"]
    y, b = CENTRES["green"], CENTRES["blue"]
    assert mean == pytest.approx([(y[0] + b[0]) / 2, (y[1] + b[1]) / 2], abs=15)


def test_agent_counts_tokens_over_a_retry():
    llm = UsageLLM([("move", {"object": that(9), "place": that(2, "here")})],
                   [("move", {"object": that(1), "place": that(2, "here")})])  # fmt: skip
    a = ArmAgent(
        __import__("test_agent").PROFILE, TOOLS, "https://x/v1/", client_factory=llm.factory
    )
    res = a.understand(["put", "that", "here"], "sk-x")
    assert res.attempts == 2 and res.usage == {"prompt": 14000, "cached": 8000, "completion": 120}


ROOT  # noqa: B018 - imported for the fixtures' paths
