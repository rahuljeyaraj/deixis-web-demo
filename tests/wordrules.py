"""A small word parser, as a stand-in for Nemotron in the tests.

This was `webdemo/agent.py`'s rule parser, and until 11 Oct 2026 the demo fell back to it
whenever Nemotron could not be asked - and showed, on the wiring panel, what it would have
made of the same words. rahul had it taken out: the demo is the model, and nothing else may
answer a person or appear in the diagram.

It lives on here because the server's own plumbing - the limits, the flow, the pointing, the
plan loop - is worth testing without a key, a network or a bill. It is a test double. It is
never imported by anything the demo runs, and `test_items.py` checks that it cannot be.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from deixis.refs import norm_word
from webdemo.agent import AgentResult, Context, Profile, validate

STOP = {"stop", "halt", "freeze", "wait", "pause"}
HOME = {"home", "rest"}
UNDO = {"undo"}
YES = {"yes", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm", "correct", "go", "do"}
NO = {"no", "nope", "cancel", "nevermind"}
MOVE = {"put", "move", "place", "stack", "set", "take", "bring", "drop", "pick"}
COLOURS = {"red", "green", "blue"}
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
    # "undo", "put that back where it was": one word, so the word matcher can have it too.
    if s & UNDO or ("back" in s and "was" in s):
        return [{"name": "undo", "args": {}}]
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


#: What the double calls itself where the real agent would name the model.
MODEL = "word rules (a test double, never the demo)"


def rules(profile: Profile, words: Sequence[str], context: Context | None = None) -> AgentResult:
    """The parser in the shape an answer from Nemotron has, so the server cannot tell."""
    t = time.perf_counter()
    raw = parse(words, context)
    calls, errors = validate(profile, raw, words)
    return AgentResult(calls, source="nemotron", model=MODEL, raw=raw, errors=errors,
                       attempts=1, asked_model=True, latency_s=time.perf_counter() - t)  # fmt: skip
