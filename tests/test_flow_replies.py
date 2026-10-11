"""The server's real replies for the plan's scenes (plan 10), saved for the browser's flow
tests (`web/tests/flow.test.ts` plays them into the real arm in MuJoCo). Typed words with
clicks (the mouse-and-typing mode), the rule parser, the core over MCP.

The plan scenes (advanced-commands-plan.md section 4) need the model, so those are taken from
an app with a fake Nemotron that answers with the plan Nemotron writes for that sentence. The
reply holds the whole plan and the core's answer for its first step, which is what the page
really gets; the later steps are measured when the arm reaches them, against a table only the
browser knows, so `flow.test.ts` measures those itself.

The file is written when missing or with UPDATE_FIXTURES=1, and checked otherwise.
"""

from __future__ import annotations

import json
import os

from fastapi.testclient import TestClient
from pydantic import SecretStr

from test_agent import FakeLLM, click, post
from test_arm_core import BLOCKS, CENTRES, HERE, PAINT, SCENE
from webdemo.app import create_app
from webdemo.settings import ROOT

FIXTURE = ROOT / "web" / "tests" / "fixtures" / "replies.json"
BETWEEN = ((CENTRES["red"][0] + CENTRES["green"][0]) / 2, CENTRES["red"][1])


def command(api, text, clicks=(), state=None, context=None, t=30.0):
    meta = {"text": text, "things": SCENE, "t": t,
            "holds": [click(t - 2 + 1.2 * k, *xy) for k, xy in enumerate(clicks)],
            "state": state or {}, "context": context or {}}  # fmt: skip
    body = post(api, meta)
    reply = {k: body[k] for k in ("result", "say", "error", "plan")}
    return {"kind": "command", "text": text, "reply": reply}


def answer(api, candidate):
    r = api.post("/api/answer", json={"candidate": candidate})
    assert r.status_code == 200, r.text
    body = r.json()
    return {"kind": "answer", "candidate": candidate,
            "reply": {k: body[k] for k in ("result", "say", "error")}}  # fmt: skip


def planned(settings, core, text, calls, clicks=()) -> dict:
    """One command answered by a plan: a fake Nemotron writes `calls`, the core measures the
    first step, and the page gets both."""
    # Its own settings: the `rules_api` fixture turns the model off, and these scenes need it.
    own = settings.model_copy(
        update={"ai": "nemotron", "nebius_api_key": SecretStr("sk-ours-0000000000")}
    )
    llm = FakeLLM(list(calls))
    with TestClient(create_app(own, client=core, llm_factory=llm.factory)) as c:
        return command(c, text, clicks)


def label(name: str) -> dict:
    return {"label": name}


def inside(name: str) -> dict:
    return {"relation": "inside", "target": label(name)}


def on(name: str) -> dict:
    return {"relation": "on", "target": label(name)}


def mv(obj: dict, place: dict) -> tuple[str, dict]:
    return ("move", {"object": obj, "place": place})


#: The plans, as Nemotron writes them for these sentences (`SYSTEM`'s examples).
PLANS: dict[str, list[tuple[str, dict]]] = {
    "swap the red and the blue": [
        mv(label("red one"), {"relation": "next_to", "target": label("green one")}),
        mv(label("blue one"), inside("the spot it left")),
        mv(label("red one"), inside("the spot it left")),
    ],
    "stack them all up": [
        mv(label("green one"), on("red one")),
        mv(label("blue one"), on("green one")),
    ],
    "put each block on its own colour": [
        mv(label("red one"), inside("red square")),
        mv(label("green one"), inside("green square")),
        mv(label("blue one"), inside("blue square")),
    ],
    "put the red one on the blue one then go home": [
        mv(label("red one"), on("blue one")),
        ("home", {}),
    ],
    # A square holds one block in the middle: a second one goes beside the first.
    "everything except the blue one goes in the red square": [
        mv(label("red one"), inside("red square")),
        mv(label("green one"), {"relation": "next_to", "target": label("red one")}),
    ],
}


def scenes(api) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    out["put that here"] = [command(api, "put that here", [CENTRES["red"], HERE]), "yes"]
    out["put the blue one next to that"] = [
        command(api, "put the blue one next to that", [CENTRES["green"]]),
        "yes",
    ]
    out["put that on that"] = [
        command(api, "put that on that", [CENTRES["red"], CENTRES["blue"]]),
        "yes",
    ]
    first = command(api, "put that here", [CENTRES["red"], HERE])
    where = first["reply"]["result"]["calls"][0]["args"]["place"]
    out["a bit left"] = [
        first,
        command(
            api, "a bit left", state={"previewing": True}, context={"previous": {"place": where}}
        ),
        "yes",
    ]
    out["outside the area"] = [command(api, "put that here", [CENTRES["red"], (540.0, 120.0)])]
    ask = command(api, "put that here", [BETWEEN, HERE])
    green = [c["thing"] for c in ask["reply"]["result"]["ask"]["candidates"]].index("green")
    out["between two blocks, then a click"] = [ask, answer(api, green), "yes"]
    out["no pointing"] = [command(api, "put that here")]
    out["stop"] = [command(api, "stop")]
    out["go home"] = [command(api, "go home")]
    out["not understood"] = [command(api, "hello robot")]
    # "yes" between steps is the page's own Go: it needs no server.
    return {k: [s if isinstance(s, dict) else {"kind": "go"} for s in v] for k, v in out.items()}


def test_scenes_for_the_browser(rules_api, settings, core):
    plans = {text: [planned(settings, core, text, calls), {"kind": "go"}]
             for text, calls in PLANS.items()}  # fmt: skip
    data = {"about": __doc__.split("\n\n")[0], "things": BLOCKS, "paint": PAINT,
            "here": list(HERE), "scenes": {**scenes(rules_api), **plans}}  # fmt: skip
    # The things the page sends are its own; the core's answers must not depend on the clock.
    text = json.dumps(data, indent=1) + "\n"
    if os.environ.get("UPDATE_FIXTURES") or not FIXTURE.exists():
        FIXTURE.write_text(text, encoding="utf-8")
    saved = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert saved == json.loads(text), "the replies changed: run with UPDATE_FIXTURES=1"
    s = data["scenes"]
    assert s["put that here"][0]["reply"]["result"]["status"] == "ok"
    assert s["outside the area"][0]["reply"]["result"]["ask"]["reason"] == "outside_area"
    assert s["no pointing"][0]["reply"]["result"]["ask"]["reason"] in ("no_gesture", "no_object")
    assert s["not understood"][0]["reply"]["result"] is None
    swap = data["scenes"]["swap the red and the blue"][0]["reply"]
    assert swap["plan"]["steps"] == 3
    assert len(swap["result"]["calls"]) == 1  # only the first step is measured here
    assert swap["result"]["calls"][0]["args"]["object"]["thing"] == "red"
