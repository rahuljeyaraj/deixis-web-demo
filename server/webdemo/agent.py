"""The arm's agent (plan 6.4): what the person said, as the arm's tool calls.

The tools are the profile's actions (move, home) plus the core's standard ones (adjust,
confirm, cancel, stop, ask_user), as the core describes them (`tools_for_agent`, over MCP).
Objects and places are references to the words ("that" is word 1), never coordinates: the
core finds them.

`ArmAgent` asks Nemotron 3.5 Lightning on Nebius Token Factory (one call per command, one
retry with the validation errors). `parse` is a small rule parser for the arm's phrases: the
fallback when Nemotron can't be reached or the day's budget is used up ("simple mode"), and
the agent for the tests. Shaped after deixis-doodle's `vacuum/agent.py` (commit 7a7e9cc).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from deixis.actions import Call, CallError, check_call, parse_call
from deixis.profile import Profile
from deixis.refs import norm_word

log = logging.getLogger("webdemo.agent")

DEFAULT_MODEL = "nvidia/Nemotron-3_5-Lightning"


class AgentUnavailable(RuntimeError):
    """Token Factory can't be used (no key, network error, timeout, a refused key)."""


class KeyRefused(AgentUnavailable):
    """Token Factory said no to the key (401 or 403)."""


@dataclass
class Context:
    """What the agent should know about the moment (from the page's state)."""

    previewing: bool = False  # a move is shown (ghost block) and waits for yes or no
    question: str | None = None  # the robot's open question, if any

    def lines(self) -> list[str]:
        out = []
        if self.previewing:
            out.append("A move is previewed and waits for the person's yes or no.")
        if self.question:
            out.append(f'The robot asked: "{self.question}"')
        return out


@dataclass
class AgentResult:
    """What the agent decided for one command."""

    calls: list[Call]
    source: Literal["nemotron", "rules", "answer"] = "nemotron"
    model: str = ""
    raw: list[dict[str, Any]] = field(default_factory=list)  # as the model (or rules) wrote them
    errors: list[str] = field(default_factory=list)  # validation errors seen (and fixed?)
    attempts: int = 0
    latency_s: float = 0.0
    asked_model: bool = False  # Nemotron was asked (it costs, even when it failed)
    key_refused: bool = False  # Token Factory refused the key
    # Token Factory's counts over all attempts: prompt, cached (part of prompt), completion
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.calls)

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model": self.model,
            "calls": [c.to_dict() for c in self.calls],
            "raw": self.raw,
            "errors": self.errors,
            "attempts": self.attempts,
            "usage": self.usage,
        }


SYSTEM = """You turn what a person says to a small robot arm into the arm's tool calls.
The arm stands at a table with three 30 mm blocks (people also say boxes or cubes): red,
green and blue. The person points at blocks and spots on the table with a finger (or
a mouse) while talking.

Rules:
- Reply only with tool calls. Never reply with plain text.
- Never write coordinates or sizes. Blocks and places are references to the person's words:
  {"deixis": "that", "word_index": N}, where N is that word's number in the Words list.
- The object: a pointed block {"deixis": "that", "word_index": N}; a block by colour
  {"label": "blue one"}; "it" (the block just moved) {"last": true}.
- The place: a pointed spot {"deixis": "here", "word_index": N}; next to, on, left of, right
  of, behind or in front of a block {"relation": "next_to", "target": <object>}; between two
  blocks {"relation": "between", "target": <object>, "target2": <object>}.
- "put that here", "move this there" -> move(object=..., place=...).
- "yes", "ok", "go", "do it" -> confirm(). "no", "cancel", "never mind" -> cancel().
- "stop", "halt", "wait", "freeze" -> stop(). "go home", "go back", "rest" -> home().
- "a bit left" -> adjust(dx=-1); "a bit right" -> adjust(dx=1); "further away" -> adjust(dy=1);
  "closer" -> adjust(dy=-1); "a lot" means 3 steps.
- If you really cannot tell what they want, call ask_user with one short question.

Examples:
Words: 0:put 1:that 2:here
move({"object": {"deixis": "that", "word_index": 1}, "place": {"deixis": "here", "word_index": 2}})

Words: 0:put 1:the 2:blue 3:one 4:next 5:to 6:that
move({"object": {"label": "blue one"}, "place": {"relation": "next_to", "target": {"deixis": "that", "word_index": 6}}})

Words: 0:put 1:that 2:on 3:that
move({"object": {"deixis": "that", "word_index": 1}, "place": {"relation": "on", "target": {"deixis": "that", "word_index": 3}}})

Words: 0:put 1:the 2:red 3:cube 4:between 5:the 6:green 7:and 8:the 9:blue
move({"object": {"label": "red one"}, "place": {"relation": "between", "target": {"label": "green one"}, "target2": {"label": "blue one"}}})

Words: 0:put 1:it 2:there
move({"object": {"last": true}, "place": {"deixis": "there", "word_index": 2}})

Words: 0:move 1:it 2:a 3:bit 4:left
adjust({"dx": -1})

Words: 0:yes
confirm({})

Words: 0:go 1:home
home({})"""  # noqa: E501


def indexed(words: Sequence[str]) -> str:
    return " ".join(f"{i}:{w}" for i, w in enumerate(words))


def user_message(words: Sequence[str], context: Context | None = None) -> str:
    lines = (context or Context()).lines()
    lines.append(f'Utterance: "{" ".join(words)}"')
    lines.append("Words: " + indexed(words))
    return "\n".join(lines)


# ------------------------------------------------------------------ the rule parser

STOP = {"stop", "halt", "freeze", "wait", "pause"}
HOME = {"home", "rest"}
YES = {"yes", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm", "correct", "go", "do"}
NO = {"no", "nope", "cancel", "nevermind"}
MOVE = {"put", "move", "place", "stack", "set", "take", "bring", "drop", "pick"}
COLOURS = {"red", "green", "green", "blue"}
BLOCK_WORDS = ("block", "blocks", "cube", "cubes", "box", "boxes")
POINTING = {"this", "that", "these", "those", "here", "there"}
SPOTS = {"here", "there"}
LEFT, RIGHT = {"left"}, {"right"}
AWAY = {"further", "farther", "away", "forward", "forwards", "back", "backwards"}
CLOSER = {"closer", "nearer", "toward", "towards"}
FILLERS = {"um", "uh", "er", "so", "well", "please", "now", "and", "oh", "can", "you", "could"}
YES_PHRASES = {"thats right", "all right", "alright", "go ahead", "do it", "sounds good", "fine"}
# Relation phrases, longest first: the words, and the core's relation name.
RELATIONS = [
    (("on", "top", "of"), "on"),
    (("in", "front", "of"), "in_front_of"),
    (("next", "to"), "next_to"),
    (("to", "the", "left", "of"), "left_of"),
    (("to", "the", "right", "of"), "right_of"),
    (("left", "of"), "left_of"),
    (("right", "of"), "right_of"),
    (("on",), "on"),
    (("onto",), "on"),
    (("beside",), "next_to"),
    (("by",), "next_to"),
    (("behind",), "behind"),
]


def _norm(words: Sequence[str]) -> list[str]:
    return [norm_word(w).replace("'", "") for w in words]


def _object_at(ws: list[str], i: int) -> tuple[dict[str, Any], int] | None:
    """An object reference starting at word i (skipping "the"), and the index after it."""
    while i < len(ws) and ws[i] in ("the", "a", "up", *BLOCK_WORDS):
        i += 1
    if i >= len(ws):
        return None
    w = ws[i]
    if w in ("this", "that"):
        end = i + 1 + (i + 1 < len(ws) and ws[i + 1] in ("one", *BLOCK_WORDS))
        return {"deixis": w, "word_index": i}, end
    if w == "it":
        return {"last": True}, i + 1
    if w in COLOURS:
        end = i + 1 + (i + 1 < len(ws) and ws[i + 1] in ("one", *BLOCK_WORDS))
        return {"label": f"{w} one"}, end
    return None


def _relation_at(ws: list[str], i: int) -> tuple[str, int] | None:
    for phrase, rel in RELATIONS:
        if tuple(ws[i : i + len(phrase)]) == phrase:
            return rel, i + len(phrase)
    return None


def _place_after(ws: list[str], i: int) -> dict[str, Any] | None:
    """The place reference in the words after the object (from word i)."""
    for k in range(i, len(ws)):
        rel = _relation_at(ws, k)
        if rel:
            target = _object_at(ws, rel[1])
            if target:
                return {"relation": rel[0], "target": target[0]}
        if ws[k] in SPOTS:
            return {"deixis": ws[k], "word_index": k}
        if ws[k] in ("this", "that") and (k + 1 >= len(ws) or ws[k + 1] not in ("one", "block")):
            # "put that there" said as "put that that": the second pointing word is the place.
            return {"deixis": ws[k], "word_index": k}
    return None


def parse(words: Sequence[str], context: Context | None = None) -> list[dict[str, Any]]:
    """The arm's phrases, by rule: raw calls ({"name", "args"}), or [] if not understood."""
    ws = _norm(words)
    s = set(ws)
    if not ws:
        return []
    if s & STOP:
        return [{"name": "stop", "args": {}}]
    if s & HOME or (s & {"go", "come"} and "back" in s and not s & MOVE):
        return [{"name": "home", "args": {}}]
    verb = next((i for i, w in enumerate(ws) if w in MOVE), None)
    steps = 3 if s & {"lot", "much", "far"} else 1
    adj: dict[str, Any] = {}
    if s & LEFT:
        adj["dx"] = -steps
    elif s & RIGHT:
        adj["dx"] = steps
    if s & AWAY:
        adj["dy"] = steps
    elif s & CLOSER:
        adj["dy"] = -steps
    has_place_word = bool(s & SPOTS) or any(_relation_at(ws, k) for k in range(len(ws)))
    if verb is not None or (s & POINTING and not (s & YES or s & NO)):
        start = verb + 1 if verb is not None else 0
        obj = _object_at(ws, start)
        # "a bit left" during a move ("move it a bit left"): an adjustment, not a new move.
        if adj and not has_place_word:
            return [{"name": "adjust", "args": adj}]
        if obj is not None and (verb is not None or has_place_word):
            place = _place_after(ws, obj[1]) or {"pointed": True}
            return [{"name": "move", "args": {"object": obj[0], "place": place}}]
        if obj is not None:
            return []  # "this one", "the red one": an answer to a question, not a command
    first = next((w for w in ws if w not in FILLERS), "")
    if first in NO or ("never" in s and "mind" in s):
        return [{"name": "cancel", "args": {}}]
    said = " ".join(w for w in ws if w not in FILLERS)
    if said in YES_PHRASES:
        return [{"name": "confirm", "args": {}}]
    if adj:
        return [{"name": "adjust", "args": adj}]
    if first in YES:
        return [{"name": "confirm", "args": {}}]
    return []


def validate(
    profile: Profile, raw: Sequence[dict[str, Any]], words: Sequence[str]
) -> tuple[list[Call], list[str]]:
    """The raw calls the profile accepts, and the problems with the others (for the model)."""
    calls: list[Call] = []
    errors: list[str] = []
    for r in raw:
        try:
            call = parse_call(profile, r["name"], r.get("args"))
        except CallError as e:
            errors.append(str(e))
            continue
        problems = check_call(call, list(words))
        if problems:
            errors += problems
        else:
            calls.append(call)
    return calls, errors


def rules(profile: Profile, words: Sequence[str], context: Context | None = None) -> AgentResult:
    """The rule parser, as an agent."""
    t = time.perf_counter()
    raw = parse(words, context)
    calls, errors = validate(profile, raw, words)
    return AgentResult(calls, source="rules", model="rules", raw=raw, errors=errors,
                       attempts=1, latency_s=time.perf_counter() - t)  # fmt: skip


# ------------------------------------------------------------------ Nemotron

_TEXT_CALL = re.compile(r"^\s*`*([a-z_]+)\s*\((.*)\)\s*;?`*\s*$")


def calls_from_text(text: str, names: set[str]) -> list[dict[str, Any]]:
    """Tool calls the model wrote as text in the examples' notation, one per line, e.g.
    'confirm({})'. Unknown names are ignored. (From deixis-doodle's `agent/nemotron.py`.)"""
    out = []
    for line in text.splitlines():
        m = _TEXT_CALL.match(line)
        if m and m.group(1) in names:
            out.append({"name": m.group(1), "args": m.group(2).strip() or "{}"})
    return out


def add_usage(total: dict[str, int], usage: Any) -> None:
    """Add one reply's token counts (OpenAI-style `usage`) to `total`."""
    if usage is None:
        return
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None
    for k, v in (("prompt", getattr(usage, "prompt_tokens", None)), ("cached", cached),
                 ("completion", getattr(usage, "completion_tokens", None))):  # fmt: skip
        if isinstance(v, int):
            total[k] = total.get(k, 0) + v


def plain_yes_no(words: Sequence[str]) -> bool:
    """A plain "yes" or "no" (fillers allowed, nothing else): answered at once by the rule
    parser, without the model, like "stop" (web demo item 12). "No, there" is not plain."""
    ws = [w for w in _norm(words) if w and w not in FILLERS and w not in ("thanks", "thank")]
    if not ws or len(ws) > 3:
        return False
    said = " ".join(ws)
    return said in YES_PHRASES or all(w in YES for w in ws) or all(w in NO for w in ws)


def _scrub(text: str, key: str | None) -> str:
    """An error message without the key (or any part an API echoed back)."""
    if key:
        text = text.replace(key, "[key]")
        if len(key) > 8:
            text = text.replace(key[:8], "[key]").replace(key[-4:], "[key]")
    return text


@dataclass
class ArmAgent:
    """Nemotron with the arm's tools; the rule parser if it fails."""

    profile: Profile
    tools: list[dict[str, Any]]  # tools_for_agent, from the core
    base_url: str
    model: str = DEFAULT_MODEL
    timeout_s: float = 8.0
    max_retries: int = 1
    client_factory: Any = None  # (api_key) -> an OpenAI-like client; tests pass a fake

    def _client(self, key: str) -> Any:
        if self.client_factory is not None:
            return self.client_factory(key)
        from openai import OpenAI

        return OpenAI(base_url=self.base_url, api_key=key, timeout=self.timeout_s, max_retries=0)

    def run(self, words: Sequence[str], key: str, context: Context | None = None) -> AgentResult:
        """Nemotron alone (raises `AgentUnavailable`). `key` is used for this call only."""
        if not key:
            raise AgentUnavailable("no Nebius key")
        client = self._client(key)
        names = {t["function"]["name"] for t in self.tools}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message(words, context)},
        ]
        result = AgentResult(calls=[], model=self.model)
        t0 = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=0.0,
                    max_tokens=512,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except Exception as e:  # openai errors, timeouts
                status = getattr(e, "status_code", None)
                if status in (401, 403):
                    raise KeyRefused("the Nebius key was refused") from None
                raise AgentUnavailable(_scrub(f"{type(e).__name__}: {e}", key)) from None
            result.attempts = attempt + 1
            add_usage(result.usage, getattr(resp, "usage", None))
            msg = resp.choices[0].message
            tool_calls = list(getattr(msg, "tool_calls", None) or [])
            raw = [{"name": c.function.name, "args": c.function.arguments} for c in tool_calls]
            if not raw:
                raw = calls_from_text(getattr(msg, "content", None) or "", names)
            result.raw = raw
            calls, errors = validate(self.profile, raw, words)
            if not raw:
                errors = ["no tool call; reply only with tool calls (use ask_user to talk)"]
            result.errors += errors
            result.calls = calls
            if not errors:
                break
            if attempt < self.max_retries:
                messages.append({"role": "assistant", "content": json.dumps(raw)})
                messages.append({
                    "role": "user",
                    "content": "Your reply had errors, so nothing was done:\n- "
                    + "\n- ".join(errors)
                    + "\nReply again with the complete, corrected list of tool calls.",
                })  # fmt: skip
        result.latency_s = time.perf_counter() - t0
        return result

    def understand(
        self,
        words: Sequence[str],
        key: str | None,
        context: Context | None = None,
        no_key_why: str = "no Nebius key: simple mode",
    ) -> AgentResult:
        """Nemotron if it answers with calls, else the rule parser (with why in `errors`).
        "Stop" never waits for the model (about a second): the rule parser answers at once.
        `no_key_why` is the reason given when there is no key (simple mode, plan 8.1)."""
        if parse(words, context) == [{"name": "stop", "args": {}}] or plain_yes_no(words):
            return rules(self.profile, words, context)
        why = no_key_why
        refused = False
        if key:
            try:
                res = self.run(words, key, context)
                res.asked_model = True
                if res.ok:
                    return res
                why = "; ".join(res.errors)
            except AgentUnavailable as e:
                why = str(e)
                refused = isinstance(e, KeyRefused)
                log.warning("nemotron unavailable, using the rule parser: %s", why)
        res = rules(self.profile, words, context)
        res.errors = [why, *res.errors]
        res.asked_model = bool(key)
        res.key_refused = refused
        return res
