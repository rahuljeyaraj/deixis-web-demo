"""The web demo's numbers from the server's log (measure list M30, M31, M33).

    uv run python scripts/measure_web.py logs/commands.jsonl [--since 2026-10-12] [--out DIR]

- **M30, right first time** (defined before counting, as the judges' PDF says): a command is
  right first time when its first guess was accepted with "yes" (said, Go or Enter), with no
  correction, cancel or question in between. Counted per visitor, in order. Every command
  that asked for a move counts in N, including one that wasn't understood, one the robot
  answered with a question, and one it refused (out of reach); yes, no, stop, "a bit left",
  "go home", undo and a declined off-topic question are answers, not new commands.
- **M31, last word to the guess** (spoken commands only): from the end of the last word to
  the guess on screen, as the page measured it (`preview` events, `latency_s`).
- **M29, arm move time** (a cross-check of the stopwatch): from "yes" (the `go` event) to
  "Done." (the `done` event), per move.
- **M33, tokens and cost per command**: Token Factory's counts for each command that asked
  Nemotron, and the cost at list price (cached tokens at the full input price).

Reads only the log; no audio, video or address is in it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

# Calls that are a reply to the person, not a new command that asked for a move: they do
# not count in M30's N. `decline` is an off-topic question answered in one line;
# `undo` puts the last block back.
ANSWERS = {"confirm", "cancel", "stop", "adjust", "home", "undo", "ask_user", "decline"}


def load(path: Path, since: str | None = None) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if since:
        rows = [r for r in rows if r["utc"] >= since]
    return sorted(rows, key=lambda r: r["utc"])


def is_new_command(r: dict) -> bool:
    if r.get("kind") != "command":
        return False
    names = {c.get("name") for c in r.get("calls") or []}
    return not names or not names <= ANSWERS or r.get("status") == "ask"


def right_first_time(rows: list[dict]) -> list[dict]:
    """One result per new command: right (True/False) and why not."""
    by_visitor: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_visitor[r["visitor"]].append(r)
    out = []
    for vis, seq in by_visitor.items():
        for i, r in enumerate(seq):
            if not is_new_command(r):
                continue
            res = {"visitor": vis, "utc": r["utc"], "said": r.get("said"), "right": False}
            if not r.get("calls"):
                res["why"] = "not understood"
            elif r.get("status") == "ask":
                res["why"] = "a question"
            elif r.get("status") != "ok":
                res["why"] = str(r.get("status"))
            else:
                res["why"] = _after_guess(seq[i + 1 :])
                res["right"] = res["why"] == "yes"
            out.append(res)
    return out


def _after_guess(rest: list[dict]) -> str:
    """What came first after a resolved command: the guess shown, then yes (right)?"""
    shown = False
    for r in rest:
        if r.get("kind") == "event":
            e = r.get("event")
            if e == "preview" and not shown:
                shown = True
                continue
            if e == "refused":
                return "refused"
            if not shown:
                continue
            if e == "go":
                return "yes"
            if e in ("cancel", "stop", "preview"):
                return "corrected" if e == "preview" else e
            continue
        if r.get("kind") == "command":
            names = {c.get("name") for c in r.get("calls") or []}
            if names == {"confirm"}:
                continue  # a spoken yes: the go event follows
            return "corrected" if names & {"adjust", "move"} else "another command"
    return "no answer" if shown else "no guess shown"


def latency(rows: list[dict]) -> list[float]:
    return [r["latency_s"] for r in rows if r.get("event") == "preview" and "latency_s" in r]


def move_times(rows: list[dict]) -> list[float]:
    """Seconds from each `go` to the next `done` of the same visitor."""
    from datetime import datetime

    out, started = [], {}
    for r in rows:
        if r.get("kind") != "event":
            continue
        t = datetime.fromisoformat(r["utc"]).timestamp()
        if r["event"] == "go":
            started[r["visitor"]] = t
        elif r["event"] in ("done", "failed", "stop") and r["visitor"] in started:
            t0 = started.pop(r["visitor"])
            if r["event"] == "done":
                out.append(round(t - t0, 3))
    return out


def tokens(rows: list[dict]) -> dict:
    used = [r for r in rows if r.get("kind") == "command" and (r.get("usage") or {}).get("prompt")]
    if not used:
        return {"commands": 0}
    costs = [r["cost_usd"] for r in used]
    p = [r["usage"]["prompt"] for r in used]
    c = [r["usage"].get("cached", 0) for r in used]
    o = [r["usage"].get("completion", 0) for r in used]
    return {
        "commands": len(used),
        "mean_prompt": round(statistics.mean(p)),
        "mean_cached": round(statistics.mean(c)),
        "mean_completion": round(statistics.mean(o)),
        "median_cost_usd": round(statistics.median(costs), 6),
        "total_cost_usd": round(sum(costs), 4),
        "per_1000_usd": round(1000 * statistics.mean(costs), 3),
    }


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def summary(rows: list[dict]) -> dict:
    rft = right_first_time(rows)
    lat = latency(rows)
    why: dict[str, int] = defaultdict(int)
    for r in rft:
        if not r["right"]:
            why[r["why"]] += 1
    return {
        "M30_right_first_time": {"right": sum(r["right"] for r in rft), "of": len(rft),
                                 "not_right_because": dict(why), "detail": rft},
        "M31_last_word_to_guess_s": {"n": len(lat),
                                     "median": round(statistics.median(lat), 3) if lat else None,
                                     "p90": round(pct(lat, 0.9), 3) if lat else None},
        "M33_tokens_and_cost": tokens(rows),
        "M29_move_s": {"n": len(moves := move_times(rows)),
                       "median": round(statistics.median(moves), 2) if moves else None,
                       "max": max(moves) if moves else None},
    }  # fmt: skip


def lines(s: dict) -> list[str]:
    out = []
    m30 = s["M30_right_first_time"]
    if m30["of"]:
        why = ", ".join(f"{k} {v}" for k, v in sorted(m30["not_right_because"].items()))
        out.append(f"M30 right first time: {m30['right']} of {m30['of']} commands"
                   + (f" (not: {why})" if why else ""))  # fmt: skip
    m31 = s["M31_last_word_to_guess_s"]
    if m31["n"]:
        out.append(f"M31 last word to the guess: median {m31['median']:.2f} s, "
                   f"p90 {m31['p90']:.2f} s, {m31['n']} spoken commands")  # fmt: skip
    m33 = s["M33_tokens_and_cost"]
    if m33["commands"]:
        out.append(
            f"M33 tokens per command: {m33['mean_prompt']} in ({m33['mean_cached']} cached), "
            f"{m33['mean_completion']} out; ${m33['median_cost_usd']:.5f} a command, "
            f"${m33['per_1000_usd']:.2f} per 1,000, at list price ({m33['commands']} commands)"
        )
    m29 = s["M29_move_s"]
    if m29["n"]:
        out.append(f"M29 arm move, yes to Done: median {m29['median']:.1f} s, "
                   f"longest {m29['max']:.1f} s ({m29['n']} moves)")  # fmt: skip
    return out or ["nothing to measure in this log"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", type=Path)
    ap.add_argument("--since", help="only rows from this UTC time on, e.g. 2026-10-12")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    s = summary(load(a.log, a.since))
    text = lines(s)
    print("\n".join(text))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
        (a.out / "summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
