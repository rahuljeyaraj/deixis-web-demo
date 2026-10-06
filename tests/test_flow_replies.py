"""The server's real replies for the plan's scenes (plan 10), saved for the browser's flow
tests (`web/tests/flow.test.ts` plays them into the real arm in MuJoCo). Typed words with
clicks (the mouse-and-typing mode), the rule parser, the core over MCP.

The file is written when missing or with UPDATE_FIXTURES=1, and checked otherwise.
"""

from __future__ import annotations

import json
import os

from test_agent import click, post
from test_arm_core import BLOCKS, CENTRES, HERE
from webdemo.settings import ROOT

FIXTURE = ROOT / "web" / "tests" / "fixtures" / "replies.json"
BETWEEN = ((CENTRES["red"][0] + CENTRES["yellow"][0]) / 2, CENTRES["red"][1])


def command(api, text, clicks=(), state=None, context=None, t=30.0):
    meta = {"text": text, "things": BLOCKS, "t": t,
            "holds": [click(t - 2 + 1.2 * k, *xy) for k, xy in enumerate(clicks)],
            "state": state or {}, "context": context or {}}  # fmt: skip
    body = post(api, meta)
    reply = {k: body[k] for k in ("result", "say", "error")}
    return {"kind": "command", "text": text, "reply": reply}


def answer(api, candidate):
    r = api.post("/api/answer", json={"candidate": candidate})
    assert r.status_code == 200, r.text
    body = r.json()
    return {"kind": "answer", "candidate": candidate,
            "reply": {k: body[k] for k in ("result", "say", "error")}}  # fmt: skip


def scenes(api) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    out["put that here"] = [command(api, "put that here", [CENTRES["red"], HERE]), "yes"]
    out["put the blue one next to that"] = [
        command(api, "put the blue one next to that", [CENTRES["yellow"]]),
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
    yellow = [c["thing"] for c in ask["reply"]["result"]["ask"]["candidates"]].index("yellow")
    out["between two blocks, then a click"] = [ask, answer(api, yellow), "yes"]
    out["no pointing"] = [command(api, "put that here")]
    out["stop"] = [command(api, "stop")]
    out["go home"] = [command(api, "go home")]
    out["not understood"] = [command(api, "hello robot")]
    # "yes" between steps is the page's own Go: it needs no server.
    return {k: [s if isinstance(s, dict) else {"kind": "go"} for s in v] for k, v in out.items()}


def test_scenes_for_the_browser(rules_api):
    data = {"about": __doc__.split("\n\n")[0], "things": BLOCKS, "here": list(HERE),
            "scenes": scenes(rules_api)}  # fmt: skip
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
