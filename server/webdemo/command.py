"""One command, server side (plan 4.2 steps 3 to 5): words (from Parakeet, or typed), the
agent's tool calls, and the core's answer over MCP. Plus the last open question per visitor,
so `/api/answer` can resolve it again with the person's choice (plan 7.2).
"""

from __future__ import annotations

import secrets
import threading
import time
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


@dataclass
class OpenAsk:
    request: dict[str, Any]
    response: dict[str, Any]
    t: float


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
