"""Test doubles for the brain. **Not fallbacks.**

The rule, from rahul, 11 Oct: the demo never falls back on a word matcher. Nemotron answers
or the robot says it cannot (`say.cannot`). A word matcher that quietly covers for an unmet
request is a demo that looks like it understands and does not, which is worse than one that
admits it.

These exist for two legitimate reasons, and neither is "in case the model is down":

- **`ScriptDecider`** returns tool calls from a list you hand it, so the session loop, the
  world, the confirmation rules and the counters can be tested exhaustively with no network
  and no calls spent. Most of `tests/end_demo` runs on it.
- **`StubDecider`** plays the game by matching words and reading the cone, so a 200-trial
  run can be exercised end to end for nothing before it is run for real. Its score means
  nothing and the run header says so.

`StubDecider.observe` is how the stub sees the world. The real brain gets the same
information through the transcript, which is the whole point of the session loop - the stub
needs a side door precisely because it cannot read.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from end_demo.brain import BrainUnavailable, Call, Usage
from end_demo.point import Cone, within_ratio
from end_demo.scene import Thing

STUB_NAME = "stub (word matcher, test double - its score means nothing)"


@dataclass
class ScriptDecider:
    """Returns the tool calls you give it, in order. The workhorse of the tests."""

    script: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    name: str = "script (test double)"
    usage: Usage = field(default_factory=Usage)
    thinking: str = "(scripted)"
    #: Every message list it was handed, so a test can assert what the model was told.
    seen: list[list[dict[str, Any]]] = field(default_factory=list)
    at: int = 0

    def decide(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_reasoning: Callable[[str], None] | None = None,
    ) -> Call:
        self.seen.append([dict(m) for m in messages])
        if self.at >= len(self.script):
            raise BrainUnavailable("the script ran out")
        tool, args = self.script[self.at]
        self.at += 1
        allowed = [t["function"]["name"] for t in tools]
        if tool not in allowed:
            raise AssertionError(f"the script called {tool!r}, which was not offered: {allowed}")
        if on_reasoning:
            on_reasoning(self.thinking)
        call = Call(
            tool=tool,
            args=dict(args),
            thinking=self.thinking,
            call_id=f"call_{self.at}",
            latency_s=0.0,
            first_token_s=0.0,
            in_tokens=10,
            out_tokens=5,
            streamed=False,  # a double streams nothing, and must not appear in the timings
        )
        self.usage.add(call)
        return call


@dataclass
class StubDecider:
    """A word matcher that reads the cone. Spends no calls; proves nothing."""

    name: str = STUB_NAME
    usage: Usage = field(default_factory=Usage)
    #: Set by `observe`, which only a test double gets to use. See the module docstring.
    things: list[Thing] = field(default_factory=list)
    cone: Cone | None = None
    said: str = ""
    asked: list[tuple[str, bool]] = field(default_factory=list)
    rejected: set[str] = field(default_factory=set)

    def observe(
        self,
        things: list[Thing],
        cone: Cone | None,
        said: str,
        asked: list[tuple[str, bool]],
    ) -> None:
        self.things = things
        self.cone = cone
        self.said = said
        self.asked = list(asked)
        self.rejected = {i for i, ok in asked if not ok}

    @staticmethod
    def _fits(t: Thing, said: str) -> bool:
        words = {w for w in said.lower().split() if len(w) > 2}
        text = (t.seen + " " + " ".join(t.says)).lower()
        return bool(words) and any(w in text for w in words)

    def _live(self) -> list[str]:
        index = {t.id: t for t in self.things}
        ids = self.cone.ids() if self.cone else [t.id for t in self.things]
        inside = [i for i in ids if i not in self.rejected and i in index]
        fits = [i for i in inside if self._fits(index[i], self.said)]
        return fits or inside

    def decide(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_reasoning: Callable[[str], None] | None = None,
    ) -> Call:
        allowed = [t["function"]["name"] for t in tools]
        live = self._live()
        tool, args = self._choose(live, allowed)
        if on_reasoning:
            on_reasoning("(stub: no thinking to show)")
        call = Call(
            tool=tool,
            args=args,
            thinking="",
            call_id=f"stub_{self.usage.calls}",
            in_tokens=0,
            out_tokens=0,
            streamed=False,
        )
        self.usage.add(call)
        return call

    def _choose(self, live: list[str], allowed: list[str]) -> tuple[str, dict[str, Any]]:
        if not live:
            if "ask_show_me" in allowed:
                return "ask_show_me", {"why": "nothing left to try"}
            fallback = (self.cone.best if self.cone else None) or (
                self.things[0].id if self.things else ""
            )
            return "choose", {"id": fallback, "why": "nothing left to try"}
        if len(live) == 1:
            if "choose" in allowed:
                return "choose", {"id": live[0], "why": "only one fits"}
            return "ask_this_one", {"id": live[0], "why": "only one fits"}
        if "ask_choice" in allowed and self.cone and within_ratio(self.cone):
            return "ask_choice", {"first": live[0], "second": live[1], "why": "too close to call"}
        if "ask_this_one" in allowed:
            return "ask_this_one", {"id": live[0], "why": "two or more fit; ask"}
        return "choose", {"id": live[0], "why": "no questions left"}
