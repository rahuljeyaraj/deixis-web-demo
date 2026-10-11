"""One command, server side (plan 4.2 steps 3 to 5): words (from Parakeet, or typed), the
agent's tool calls, and the core's answer over MCP. Plus the last open question per visitor,
so `/api/answer` can resolve it again with the person's choice (plan 7.2), and the plan it set
off, so `/api/step` can measure each later step just before the arm runs it.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from deixis.refs import DEMONSTRATIVES, norm_word

MAX_SAMPLES = 1000  # plan 7.2
SESSION_TTL_S = 300.0  # the last question is kept for 5 minutes
TYPED_STEP_S = 0.25  # typed words, spread this far apart

NOT_UNDERSTOOD = "Sorry, I didn't get that. Try “put that here”."


class PageState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    previewing: bool = False
    question: str | None = None


class Meta(BaseModel):
    """`meta` of `POST /api/command` (plan 7.2). The robot-side shapes (samples, holds, things,
    context, answers) go to the core as they are; the core checks them."""

    model_config = ConfigDict(extra="ignore")

    audio_t0: float | None = None
    text: str | None = Field(default=None, max_length=300)
    t: float | None = None  # the page clock when sent (typed words with no clicks)
    samples: list[dict[str, Any]] = Field(default_factory=list, max_length=MAX_SAMPLES)
    holds: list[dict[str, Any]] = Field(default_factory=list, max_length=50)
    things: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    context: dict[str, Any] = Field(default_factory=dict)
    state: PageState = Field(default_factory=PageState)
    answers: list[dict[str, Any]] = Field(default_factory=list, max_length=10)


def typed_words(text: str, holds: list[dict[str, Any]], t_now: float | None) -> list[dict]:
    """Times for a typed sentence (mouse-and-typing mode, plan 5.3): the k-th pointing word
    gets the k-th click's time; the other words are spread evenly around them. Robot-side
    glue, not a core change."""
    ws = text.split()
    clicks = sorted(holds, key=lambda h: h["t0"])
    pointing = [i for i, w in enumerate(map(norm_word, ws)) if w in DEMONSTRATIVES and w != "so"]
    anchors = {i: (c["t0"] + c["t1"]) / 2 for i, c in zip(pointing, clicks, strict=False)}
    if not anchors:
        end = t_now if t_now is not None else 0.0
        anchors = {len(ws) - 1: end - TYPED_STEP_S}
    keys = sorted(anchors)
    times: list[float] = []
    for i in range(len(ws)):
        if i in anchors:
            times.append(anchors[i])
        elif i < keys[0]:
            times.append(anchors[keys[0]] - (keys[0] - i) * TYPED_STEP_S)
        elif i > keys[-1]:
            times.append(anchors[keys[-1]] + (i - keys[-1]) * TYPED_STEP_S)
        else:
            a = max(k for k in keys if k < i)
            b = min(k for k in keys if k > i)
            times.append(anchors[a] + (anchors[b] - anchors[a]) * (i - a) / (b - a))
    return [
        {"text": w, "t0": round(t, 4), "t1": round(t + 0.2, 4)}
        for w, t in zip(ws, times, strict=True)
    ]


def say_for(result: dict[str, Any] | None, calls: list[dict[str, Any]]) -> str | None:
    """What the robot says out loud, decided on the server: a question, or "didn't get that".
    (The preview's "This one, here?" is the page's: it checks reach first.)"""
    if not calls:
        return NOT_UNDERSTOOD
    if result and result.get("status") == "ask":
        return result["ask"]["question"]
    for c in calls:
        if c["name"] == "ask_user":
            return c["args"].get("question")
    return None


def answer_by_label(ask: dict[str, Any], texts: list[str]) -> int | None:
    """ "the red one" while the robot asks "This one or that one?": the candidate whose block
    (its id is its colour) was named, if exactly one was."""
    said = {norm_word(w) for w in texts}
    hits = [i for i, c in enumerate(ask["candidates"]) if c.get("thing") in said]
    return hits[0] if len(hits) == 1 else None


def choices_for(ask: dict[str, Any]) -> list[str]:
    """The open question's candidates named the way the page glows and says them ("the red
    one"), for the agent's `pick` tool. Empty unless every candidate is a thing: a question
    between bare spots gives a person nothing to name, and the page glows none of them
    (web/src/flow/flow.ts, `askLine`)."""
    things = [c.get("thing") for c in ask.get("candidates") or []]
    if len(things) < 2 or not all(isinstance(t, str) and t for t in things):
        return []
    return [f"the {t} one" for t in things]


def nearest_candidate(ask: dict[str, Any], point: list[float], within_mm: float) -> int | None:
    """The candidate nearest a pointed spot ("this one" while pointing), if close enough."""
    best = None
    for i, c in enumerate(ask["candidates"]):
        m = c.get("mean_mm")
        if not m:
            continue
        d = ((m[0] - point[0]) ** 2 + (m[1] - point[1]) ** 2) ** 0.5
        if d <= within_mm and (best is None or d < best[0]):
            best = (d, i)
    return None if best is None else best[1]


def answer_entry(ask: dict[str, Any], candidate: int) -> dict[str, Any]:
    """The `Answer` for picking one candidate of an `Ask` (as C18's last test sends it)."""
    back: dict[str, Any] = {"call": ask.get("call") or 0, "arg": ask["arg"], "candidate": candidate}
    thing = ask["candidates"][candidate].get("thing")
    if thing:
        back["thing"] = thing
    else:
        back["where"] = ask["candidates"][candidate]
    return back


def _inside(p: list[float] | None, box: list[float]) -> bool:
    return bool(p) and box[0] <= p[0] <= box[0] + box[2] and box[1] <= p[1] <= box[1] + box[3]


def _stackable(thing: dict[str, Any]) -> bool:
    """Something a block can be put on top of: another block, not the paint. The painted
    squares and the spot a block left are flat (`height_mm` 0), and pointing inside one means
    that spot on the table, not "on top of the square"."""
    return bool(thing.get("height_mm"))


#: Two blocks whose centres are this close are one on top of the other, not side by side.
#: The page's own `under` uses the same fraction of a 30 mm block; side by side, the nearest
#: a block can be put is 31 mm away (half, half and the profile's clearance).
SAME_COLUMN_MM = 20.0


def _centre(thing: dict[str, Any]) -> tuple[float, float]:
    box = thing["box_mm"]
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def _named(thing: dict[str, Any]) -> str:
    """The block by the name the model writes it as: "red one" (the page reports it in
    `attrs.one`, since "the red one" names both the block and the paint of its colour)."""
    one = (thing.get("attrs") or {}).get("one")
    return one if isinstance(one, str) and one else str(thing.get("label") or thing.get("id") or "")


def stacks_in(things: Sequence[dict[str, Any]]) -> list[list[str]]:
    """The towers on the table right now, each one bottom to top, by the names the model
    writes ("red one"). Robot-side knowledge the core does not model in words: the model is
    told which blocks are stacked, because only the top one can be picked up and a plan that
    starts with a covered block is wasted (rahul, 11 Oct: "stack them all up" and then "put
    each block in its own square" did nothing at all).

    Blocks over the same spot (`SAME_COLUMN_MM`) are one column, ordered by how high they
    stand. The flat things - the painted squares, the spot a block left - have no height and
    are never part of a tower."""
    blocks = [t for t in things if _stackable(t) and t.get("box_mm")]
    columns: list[list[dict[str, Any]]] = []
    for b in sorted(blocks, key=lambda t: t["height_mm"]):
        x, y = _centre(b)
        for col in columns:
            cx, cy = _centre(col[0])
            if ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 < SAME_COLUMN_MM:
                col.append(b)
                break
        else:
            columns.append([b])
    return [[_named(t) for t in col] for col in columns if len(col) > 1]


def onto_pointed_block(request: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
    """ "Put that here" while pointing at another block means on top of it (rahul, 5 Oct:
    stacking that way said "a block in the way"). For each move whose place was a pointing
    word and whose pointed spot lies on another block, the same word becomes the target of
    "on", so the core works out the block and the height. Robot-side glue, not a core change:
    returns the request to resolve again, or None when nothing changes."""
    calls = [dict(c) for c in request["calls"]]
    changed = False
    for rc in result.get("calls", []):
        call = calls[rc["index"]]
        place_ref = call["args"].get("place")
        if call["name"] != "move" or not isinstance(place_ref, dict) or "deixis" not in place_ref:
            continue
        mean = (rc["args"].get("place") or {}).get("mean_mm")
        moved = (rc["args"].get("object") or {}).get("thing")
        onto = [t for t in request["things"] if t["id"] != moved and _stackable(t)]
        if any(_inside(mean, t["box_mm"]) for t in onto):
            call["args"] = {**call["args"], "place": {"relation": "on", "target": dict(place_ref)}}
            calls[rc["index"]] = call
            changed = True
    return {**request, "calls": calls} if changed else None


def call_lines(calls: Sequence[dict[str, Any]]) -> list[str]:
    """The calls as the examples write them, one per line: `move({"object": ...})`. Used to
    show the model the plan it wrote when a step of it failed (a re-plan)."""
    return [f"{c['name']}({json.dumps(c.get('args') or {}, separators=(',', ' '))})" for c in calls]


@dataclass
class OpenAsk:
    request: dict[str, Any]
    response: dict[str, Any]
    t: float


@dataclass
class OpenPlan:
    """A plan that is running: the model's calls, and the moment the person said them (the
    words with their times, the pointer samples, the holds).

    Only step 0 is resolved when the command arrives. The page asks for each later step just
    before it runs it (`/api/step`), sending the table as it is then, so a step is measured
    against the scene it will act on and not the one before it
    (advanced-commands-plan.md section 6)."""

    words: list[dict[str, Any]]
    samples: list[dict[str, Any]]
    holds: list[dict[str, Any]]
    calls: list[dict[str, Any]]
    context: dict[str, Any]
    replans: int = 0
    t: float = 0.0

    @property
    def steps(self) -> int:
        return len(self.calls)

    def texts(self) -> list[str]:
        return [w["text"] for w in self.words]

    def moment(self) -> dict[str, Any]:
        """The parts of a core request that come from when it was said, not from now."""
        return {"words": self.words, "samples": self.samples, "holds": self.holds}


class Plans:
    """Each visitor's running plan (in memory, 5 minutes), beside their open question."""

    def __init__(self, ttl_s: float = SESSION_TTL_S) -> None:
        self.ttl_s = ttl_s
        self._plans: dict[str, OpenPlan] = {}
        self._lock = threading.Lock()

    def put(self, sid: str, plan: OpenPlan) -> None:
        plan.t = time.monotonic()
        with self._lock:
            self._expire()
            self._plans[sid] = plan

    def get(self, sid: str) -> OpenPlan | None:
        with self._lock:
            self._expire()
            return self._plans.get(sid)

    def drop(self, sid: str) -> None:
        with self._lock:
            self._plans.pop(sid, None)

    def _expire(self) -> None:
        now = time.monotonic()
        for k in [k for k, v in self._plans.items() if now - v.t > self.ttl_s]:
            del self._plans[k]


class Sessions:
    """Each visitor's last open question (in memory, 5 minutes). A visitor is a random
    cookie; nothing else is kept."""

    def __init__(self, ttl_s: float = SESSION_TTL_S) -> None:
        self.ttl_s = ttl_s
        self._asks: dict[str, OpenAsk] = {}
        self._lock = threading.Lock()

    @staticmethod
    def new_id() -> str:
        return secrets.token_urlsafe(16)

    def put(self, sid: str, request: dict[str, Any], response: dict[str, Any]) -> None:
        with self._lock:
            self._expire()
            self._asks[sid] = OpenAsk(request, response, time.monotonic())

    def get(self, sid: str) -> OpenAsk | None:
        with self._lock:
            self._expire()
            return self._asks.get(sid)

    def drop(self, sid: str) -> None:
        with self._lock:
            self._asks.pop(sid, None)

    def _expire(self) -> None:
        now = time.monotonic()
        for k in [k for k, v in self._asks.items() if now - v.t > self.ttl_s]:
            del self._asks[k]
