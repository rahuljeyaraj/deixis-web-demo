"""scripts/measure_web.py: M30 right first time, M31 last word to the guess, M33 tokens."""

from __future__ import annotations

import importlib.util
import json
import sys

from webdemo.settings import ROOT

spec = importlib.util.spec_from_file_location("measure_web", ROOT / "scripts" / "measure_web.py")
mw = importlib.util.module_from_spec(spec)
sys.modules["measure_web"] = mw
spec.loader.exec_module(mw)

MOVE = [{"name": "move", "args": {}}]
YES = [{"name": "confirm", "args": {}}]
ADJ = [{"name": "adjust", "args": {"dx": -1}}]
USAGE = {"prompt": 5500, "cached": 2000, "completion": 60}


def cmd(t, calls, status="ok", vis="a", usage=None, cost=0.0):
    return {"utc": f"2026-10-12T10:00:{t:02d}", "kind": "command", "visitor": vis, "said": "x",
            "calls": calls, "status": status, "usage": usage or {}, "cost_usd": cost}  # fmt: skip


def ev(t, kind, vis="a", **kw):
    return {
        "utc": f"2026-10-12T10:00:{t:02d}",
        "kind": "event",
        "visitor": vis,
        "event": kind,
        **kw,
    }


def test_m30_m31_m33(tmp_path):
    rows = [
        cmd(1, MOVE, usage=USAGE, cost=0.00035), ev(2, "preview", latency_s=2.1), ev(3, "go"),
        ev(4, "done"),  # right: Go
        cmd(5, MOVE, usage=USAGE, cost=0.00035), ev(6, "preview", latency_s=2.5),
        cmd(7, YES), ev(8, "go"),  # right: a spoken yes
        cmd(10, MOVE), ev(11, "preview", latency_s=3.0), cmd(12, ADJ), ev(13, "preview"),
        ev(14, "go"),  # corrected first
        cmd(15, MOVE, status="ask"),  # a question
        cmd(16, []),  # not understood
        cmd(17, MOVE), ev(18, "refused", why="out of reach"),  # refused
        cmd(19, MOVE, vis="b"), ev(20, "preview", vis="b"), ev(21, "cancel", vis="b"),
    ]  # fmt: skip
    p = tmp_path / "commands.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    s = mw.summary(mw.load(p))
    m30 = s["M30_right_first_time"]
    assert (m30["right"], m30["of"]) == (2, 7)
    assert m30["not_right_because"] == {
        "corrected": 1, "a question": 1, "not understood": 1, "refused": 1, "cancel": 1,
    }  # fmt: skip
    assert s["M31_last_word_to_guess_s"] == {"n": 3, "median": 2.5, "p90": 3.0}
    m33 = s["M33_tokens_and_cost"]
    assert m33["commands"] == 2 and m33["mean_prompt"] == 5500 and m33["per_1000_usd"] == 0.35
    text = "\n".join(mw.lines(s))
    assert "M30 right first time: 2 of 7" in text and "M31" in text and "M33" in text
