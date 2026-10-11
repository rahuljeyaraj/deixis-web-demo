"""The wiring trace (advanced-commands-plan.md section 5.1): what each part of the demo said
to the next, in order, so the page can draw it live.

A judge can otherwise finish the whole demo without ever seeing the word "Nemotron". One row
per message between two lanes - you, MediaPipe, Parakeet, the page, Nemotron, the deixis core,
the arm - with plain words on the row and the real JSON behind it (`detail`). The rows the
server can know are built here; the page adds its own as the arm moves (`web/src/ui/wiring.ts`).

Nothing here decides anything: it is a reading of what already happened.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

#: The lanes, left to right, as the panel draws them.
LANES = ("you", "mediapipe", "parakeet", "page", "nemotron", "core", "arm")

#: Row kinds that are not a message from one lane to another.
#: - "shadow": what the simple word matcher alone would have made of the same words.
#: - "step":   a marker between the steps of one plan.
#: - "note":   something the page or the server decided, with no second lane.
KINDS = ("shadow", "step", "note")


@dataclass
class Row:
    """One message, from one lane to another."""

    frm: str
    to: str
    say: str  # plain words, for the row itself
    detail: Any = None  # the real thing, shown when the row is clicked
    ms: float | None = None  # how long it took, on the rows that cost something
    kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"from": self.frm, "to": self.to, "say": self.say}
        if self.detail is not None:
            out["detail"] = self.detail
        if self.ms is not None:
            out["ms"] = round(self.ms)
        if self.kind is not None:
            out["kind"] = self.kind
        return out


@dataclass
class Trace:
    """Rows in the order they happened."""

    rows: list[Row] = field(default_factory=list)

    def add(self, *rows: Row | None) -> None:
        self.rows += [r for r in rows if r is not None]

    def to_list(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.rows]


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else seconds * 1000


def _quote(text: str) -> str:
    return f"“{text}”"


# ---------------------------------------------------------------- plain words for references


def _object_phrase(ref: dict[str, Any]) -> str:
    if "deixis" in ref:
        return _quote(str(ref["deixis"]))
    if "label" in ref:
        return _quote(f"the {ref['label']}")
    if ref.get("last"):
        return "the block that moved last"
    return "that"


def _place_phrase(ref: dict[str, Any]) -> str:
    if "deixis" in ref:
        return _quote(str(ref["deixis"]))
    if ref.get("pointed"):
        return "the spot they pointed at"
    if "relation" in ref:
        rel = str(ref["relation"]).replace("_", " ")
        where = f"{rel} {_object_phrase(ref.get('target') or {})}"
        if ref.get("target2"):
            where += f" and {_object_phrase(ref['target2'])}"
        return where
    if "named" in ref:
        return _quote(str(ref["named"]))
    if ref.get("empty_space"):
        return "the largest empty space"
    return "there"


def asked_for(calls: Sequence[dict[str, Any]]) -> str:
    """What the page is asking the core, in plain words: "where are "that" and "here"?"."""
    phrases: list[str] = []
    for c in calls:
        for name, value in (c.get("args") or {}).items():
            if not isinstance(value, dict):
                continue
            phrases.append(_object_phrase(value) if name == "object" else _place_phrase(value))
    if not phrases:
        names = ", ".join(c["name"] for c in calls)
        return f"nothing to measure ({names})"
    if len(phrases) == 1:
        return f"where is {phrases[0]}?"
    return "where are " + ", ".join(phrases[:-1]) + f" and {phrases[-1]}?"


def _spread_mm(where: dict[str, Any]) -> float | None:
    """The place's own uncertainty as one number: the larger standard deviation, mm."""
    cov = where.get("cov_mm2")
    if not cov:
        return None
    try:
        worst = max(float(cov[0][0]), float(cov[1][1]))
    except (TypeError, IndexError, ValueError):
        return None
    return math.sqrt(worst) if worst > 0 else None


def _where_phrase(where: dict[str, Any]) -> str:
    """One resolved `Where` in plain words: "the red block, +/- 4 mm", "(330, 50) mm"."""
    mean = where.get("mean_mm") or []
    spot = f"({mean[0]:.0f}, {mean[1]:.0f}) mm" if len(mean) == 2 else "a place"
    if where.get("z_mm"):
        spot += f", {where['z_mm']:.0f} mm up"
    said = f"{where['label']} at {spot}" if where.get("label") else spot
    spread = _spread_mm(where)
    return f"{said} ± {spread:.0f} mm" if spread else said


def measured(result: dict[str, Any]) -> str:
    """What the core answered, in plain words."""
    if result.get("status") == "ask":
        return f"it asks: {result['ask']['question']}"
    out: list[str] = []
    for c in result.get("calls") or []:
        for value in (c.get("args") or {}).values():
            if isinstance(value, dict) and value.get("frame"):
                out.append(_where_phrase(value))
    return "; ".join(out) or "nothing to measure"


# ---------------------------------------------------------------- the rows


def pointing_row(samples: int, holds: int, typed: bool) -> Row | None:
    """You pointed (the camera), or you clicked (the mouse)."""
    if samples:
        held = f", {holds} of them still" if holds else ""
        return Row("you", "mediapipe", f"you pointed: {samples} fingertip positions{held}")
    if holds:
        clicks = "click" if holds == 1 else "clicks"
        return Row("you", "page", f"you pointed: {holds} mouse {clicks}")
    return None if typed else Row("you", "mediapipe", "you did not point")


def heard_rows(words: Sequence[dict[str, Any]], typed: bool, audio_s: float | None,
               asr_s: float | None) -> list[Row]:  # fmt: skip
    """You spoke, and the words Parakeet heard; or you typed."""
    said = " ".join(w["text"] for w in words)
    if typed:
        return [Row("you", "page", f"you typed: {_quote(said)}", detail={"words": list(words)})]
    rows = [Row("you", "parakeet", f"you spoke: {audio_s:.1f} s of sound" if audio_s else
                "you spoke")]  # fmt: skip
    n = len(words)
    heard = f"{n} word{'' if n == 1 else 's'}, each with its time: {_quote(said)}" if n else (
        "no words in that clip")  # fmt: skip
    rows.append(Row("parakeet", "page", heard, detail={"words": list(words)}, ms=_ms(asr_s)))
    return rows


def model_rows(words: Sequence[str], things: Sequence[dict[str, Any]],
               tools: Sequence[dict[str, Any]], agent: Any, agent_s: float | None,
               off: str | None) -> list[Row]:  # fmt: skip
    """What the page sent Nemotron, and the plan it sent back.

    `agent` is an `AgentResult`; `off` is why Nemotron was not asked (simple mode), if so."""
    blocks = len([t for t in things if t.get("selectable", True)])
    sent = f"your words + the table ({blocks} blocks) + {len(tools)} tools it may call"
    rows = [Row("page", "nemotron", sent, detail={
        "words": list(words),
        "tools": [t["function"]["name"] for t in tools],
        "things": [t.get("label") or t.get("id") for t in things],
    })]  # fmt: skip
    if off:
        rows.append(Row("nemotron", "page", f"not asked: {off}", ms=_ms(agent_s)))
        return rows
    detail = {"calls": [c.to_dict() for c in agent.calls], "wrote": agent.raw}
    if agent.errors:
        detail["errors"] = agent.errors
    if agent.refusal:
        back = f"it refused: {agent.refusal}"
    elif agent.pick is not None:
        back = "it answered the robot's question: " + (
            f"choice {agent.pick.choice}" if agent.pick.choice is not None
            else f"the word they pointed with (word {agent.pick.word_index})")  # fmt: skip
        detail["pick"] = agent.pick.to_dict()
    elif len(agent.calls) > 1:
        back = f"a plan: {len(agent.calls)} tool calls"
    elif agent.calls:
        back = f"one tool call: {agent.calls[0].name}"
    else:
        back = "no call it could use"
    tries = f" (after {agent.attempts} tries)" if agent.attempts > 1 else ""
    rows.append(Row("nemotron", "page", back + tries, detail=detail, ms=_ms(agent_s)))
    return rows


def shadow_row(word_rules: dict[str, Any], drove: bool) -> Row:
    """The grey row: what the simple word matcher alone made of the very same words. It never
    drives the arm unless Nemotron could not be asked at all (`drove`)."""
    made = word_rules["words"] if word_rules["understood"] else "not understood"
    same = " (the same as Nemotron)" if word_rules.get("same") and not drove else ""
    who = "the simple word matcher answered instead" if drove else "the simple word matcher"
    return Row("page", "page", f"{who}, same words: {_quote(made)}{same}",
               detail={"calls": word_rules["calls"]}, kind="shadow")  # fmt: skip


def step_row(index: int, steps: int) -> Row | None:
    """The marker between the steps of a plan (none when there is only one step)."""
    if steps < 2:
        return None
    return Row("page", "page", f"step {index + 1} of {steps}", kind="step")


def core_rows(calls: Sequence[dict[str, Any]], result: dict[str, Any] | None, error: str | None,
              core_s: float | None, again: bool = False) -> list[Row]:  # fmt: skip
    """The page asking the core where, and the core's answer. `again`: this step is being
    measured again, just before it runs, against the table as it is now."""
    if not calls:
        return []
    ask = asked_for(calls)
    rows = [Row("page", "core", ("measure again: " if again else "") + ask,
                detail={"calls": list(calls)})]  # fmt: skip
    if error:
        rows.append(Row("core", "page", f"it refused the request: {error}", ms=_ms(core_s)))
    elif result is not None:
        rows.append(Row("core", "page", measured(result), detail=result, ms=_ms(core_s)))
    return rows


def refusal_row(say: str) -> Row:
    """The arm saying no, with nobody to send it to."""
    return Row("page", "you", say, kind="note")
