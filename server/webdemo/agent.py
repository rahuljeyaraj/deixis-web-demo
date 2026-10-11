"""The arm's agent (plan 6.4): what the person said, as the arm's tool calls.

The tools are the profile's actions (move, home, undo, decline) plus the core's standard ones
(adjust, confirm, cancel, stop, ask_user), as the core describes them (`tools_for_agent`, over
MCP). Objects and places are references to the words ("that" is word 1), never coordinates:
the core finds them.

One command may need several steps ("stack them all up", "swap the red and the blue"), so the
model answers with a **plan**: up to `MAX_STEPS` calls in the order the arm should do them,
from one model call (advanced-commands-plan.md section 6). The page runs them one at a time
and has the core measure the table again between steps, so step 2 is planned against the table
as it is then. A longer plan is refused out loud rather than attempted badly: the literature
is clear that reliability falls away and hallucinations pile up over long plans.

`ArmAgent` asks Nemotron 3.5 Lightning on Nebius Token Factory for every command, "yes" and
"stop" included (one call per command, one retry with the validation errors): the model is
what the demo shows, so nothing is decided behind its back (rahul, 8 Oct). While the robot
has a question open it also gets `pick`, so the answer to that question is the model's call
too, not words read on the server (rahul, 11 Oct); where the answer was only pointed at
("this one"), `pick` hands the choosing to the core, which is the one part the model cannot
know. `parse` is a small rule parser for the arm's phrases: the fallback when Nemotron can't
be reached or the day's budget is used up ("simple mode"), the comparison the details panel
prints beside the model's answer, and the agent for the tests. The arm's instant stop is the
page's Stop button (Esc), which never waits for anything. Shaped after deixis-doodle's
`vacuum/agent.py` (7a7e9cc).
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
from deixis.refs import EmptySpace, NamedPlace, norm_word

log = logging.getLogger("webdemo.agent")

DEFAULT_MODEL = "nvidia/Nemotron-3_5-Lightning"

#: The longest plan the arm will attempt (open decision D5: 4).
MAX_STEPS = 4
#: Actions that are an answer to the person, not a step of a plan: they only ever come alone.
ALONE = frozenset({"confirm", "cancel", "stop", "undo", "ask_user", "decline"})

TOO_MANY = "That's more steps than I can plan at once. Ask me for up to four."
NOT_FOLLOWED = "Sorry, I didn't follow that. Try one thing at a time."
#: What the arm says to anything that isn't about moving these blocks (the `decline` action).
DECLINED = "I only move the blocks on this table."


class AgentUnavailable(RuntimeError):
    """Token Factory can't be used (no key, network error, timeout, a refused key)."""


class KeyRefused(AgentUnavailable):
    """Token Factory said no to the key (401 or 403)."""


@dataclass
class Context:
    """What the agent should know about the moment (from the page's state)."""

    previewing: bool = False  # a move is shown (ghost block) and waits for yes or no
    question: str | None = None  # the robot's open question, if any
    # The things the open question is between, as the page glows them ("the red one"). While
    # there are any, the agent is given the `pick` tool so the model answers the question
    # itself instead of the server reading the words (rahul, 11 Oct).
    choices: list[str] = field(default_factory=list)
    # A re-plan (`/api/replan`): the plan that was running, as the model wrote it, which step
    # the arm could not do, and the arm's own reason. The model gets one go at a new plan.
    plan: list[str] = field(default_factory=list)
    failed_step: int | None = None
    failed_why: str = ""

    def lines(self) -> list[str]:
        out = []
        if self.previewing:
            out.append("A move is previewed and waits for the person's yes or no.")
        if self.question:
            out.append(f'The robot asked: "{self.question}"')
        if self.choices:
            out.append("Choices: " + "; ".join(f"{i}: {c}" for i, c in enumerate(self.choices)))
        if self.plan and self.failed_step is not None:
            out.append("You planned these steps for this same command:")
            out += [f"  {i + 1}. {c}" for i, c in enumerate(self.plan)]
            out.append(
                f'Step {self.failed_step + 1} could not be done: "{self.failed_why}" '
                "The steps before it were done; the table has moved on. Plan the rest again "
                "from where the arm is now, another way."
            )
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
    pick: Pick | None = None  # how the model answered an open question (the `pick` tool)
    # The plan was refused as a whole (too long, or an answer mixed into a plan): what the
    # robot says instead of acting. The rule parser is not asked to have a go at these.
    refusal: str | None = None
    # Token Factory's counts over all attempts: prompt, cached (part of prompt), completion
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.calls) or self.pick is not None

    @property
    def steps(self) -> int:
        return len(self.calls)

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model": self.model,
            "pick": self.pick.to_dict() if self.pick else None,
            "calls": [c.to_dict() for c in self.calls],
            "raw": self.raw,
            "errors": self.errors,
            "refusal": self.refusal,
            "attempts": self.attempts,
            "usage": self.usage,
        }


SYSTEM = """You turn what a person says to a small robot arm into the arm's tool calls.
The arm stands at a table with three 30 mm blocks (people also say boxes or cubes): red,
green and blue. Three flat coloured squares are painted on the arm's mat: the red square,
the green square and the blue square. A square cannot be picked up; it is a place to put a
block in. The person points at blocks and spots on the table with a finger (or a mouse)
while talking.

Rules:
- Reply only with tool calls. Never reply with plain text.
- One sentence often needs several steps ("stack them all up", "put each one on its own
  colour", "swap these two"). Work out every step the arm must do, and put them ALL in this
  one reply, as several tool calls at once, in the order the arm should do them: at most 4.
  You are asked once. A reply with only the first step is wrong, and the rest of what the
  person asked for will never happen.
  The arm has one gripper, so it carries one block at a time, and a block with another block
  on top of it cannot be picked up until the top one has been moved.
- Never write coordinates or sizes. Blocks and places are references to the person's words:
  {"deixis": "that", "word_index": N}, where N is that word's number in the Words list.
- word_index must point at a pointing word itself - one of here, there, this, that, these,
  those - exactly as the Words list numbers it. Never at any other word. "the red block" is
  not a pointing word: a block named by colour is always {"label": "red one"}.
- The object: a pointed block {"deixis": "that", "word_index": N}; a block by colour
  {"label": "blue one"}; "it" (the block just moved) {"last": true}.
- The place: a pointed spot {"deixis": "here", "word_index": N}; where they pointed with no
  word for it {"pointed": true}; next to, on, left of, right of, behind, in front of or
  inside a block or a square {"relation": "next_to", "target": <object>}; between two blocks
  {"relation": "between", "target": <object>, "target2": <object>}.
- A square is named like a block: {"label": "red square"}. "in the red square" is
  {"relation": "inside", "target": {"label": "red square"}}.
- A square holds one block in the middle. A second block into the same square goes beside the
  one that is already there: {"relation": "next_to", "target": <that block>}.
- The arm reaches a ring around itself only, so there is no place name it can trust: never
  write {"named": ...} and never {"empty_space": true}. A free spot is always beside
  something: {"relation": "next_to", "target": <another block>}.
- A spot a block has just been moved away from stays in the scene as {"label": "the spot it
  left"} until something is put there. That is how one block goes where another one was, so
  a swap takes three steps: park the first block beside the third one, put the second block
  in the spot it left, then put the first block in the spot the second one left.
- Nudging the block being previewed: "a bit left" -> adjust(dx=-1); "a bit right" ->
  adjust(dx=1); "further away" -> adjust(dy=1); "closer" -> adjust(dy=-1); "a lot" means 3
  steps. But "move them all a bit to the left" is not a nudge: it is one move per block, each
  to {"relation": "left_of", "target": <that same block>}. A step of a plan is always a move.
- If they said which block but no word for the place ("move this box", "put this"), they
  pointed at the spot without naming it: the place is {"pointed": true}. Never invent a
  pointing word that is not in the Words list.
- When the robot has asked a question, the Choices list above names what it is between (the
  page has them glowing). A short reply that just picks one of them is the answer to that
  question, not a new command: call pick. Only reply with another call if they said a whole
  new command instead ("no, put the blue one here").
- pick two ways, and the difference matters: if they named one ("the green one"), the words
  say which, so give pick(choice=N). If they only pointed ("this one", "that one"), the words
  do not say which - where they pointed does, and the robot works that out. Then give
  pick(word_index=N) with that pointing word's number, and never guess a choice yourself.
- "put that here", "move this there" -> move(object=..., place=...).
- "yes", "ok", "go", "do it" -> confirm(). "no", "cancel", "never mind" -> cancel().
- "stop", "halt", "wait", "freeze" -> stop(). "go home", "go back", "rest" -> home().
- "undo", "put that back where it was" -> undo().
- confirm, cancel, stop, undo, ask_user and decline are never part of a longer plan: send one
  of them on its own. home may end a plan.
- Anything that is not about moving these blocks - a question about the world, chit-chat, or
  a job this arm cannot do (throw it, pour it, draw) - is decline(). Never hand the person
  their own question back with ask_user, and never answer it.
- Use ask_user only when a real command about the blocks is too vague to act on, and then ask
  one short question of your own.
- Words you cannot make sense of at all ("banana the sideways wednesday") are not a command.
  Call ask_user, or decline. Never move a block and never go home on a sentence you did not
  understand: doing something is worse than saying you did not follow.

Examples. Several lines under one Words line means several tool calls in one reply, which
is what that command needs:

Words: 0:put 1:that 2:here
move({"object": {"deixis": "that", "word_index": 1}, "place": {"deixis": "here", "word_index": 2}})

Words: 0:move 1:this 2:box
move({"object": {"deixis": "this", "word_index": 1}, "place": {"pointed": true}})

Words: 0:put 1:this 2:box 3:on 4:top 5:of 6:the 7:red 8:block
move({"object": {"deixis": "this", "word_index": 1}, "place": {"relation": "on", "target": {"label": "red one"}}})

Words: 0:put 1:the 2:blue 3:one 4:next 5:to 6:that
move({"object": {"label": "blue one"}, "place": {"relation": "next_to", "target": {"deixis": "that", "word_index": 6}}})

Words: 0:put 1:that 2:on 3:that
move({"object": {"deixis": "that", "word_index": 1}, "place": {"relation": "on", "target": {"deixis": "that", "word_index": 3}}})

Words: 0:place 1:this 2:box 3:on 4:top 5:of 6:this 7:box
move({"object": {"deixis": "this", "word_index": 1}, "place": {"relation": "on", "target": {"deixis": "this", "word_index": 6}}})

Words: 0:put 1:the 2:red 3:cube 4:between 5:the 6:green 7:and 8:the 9:blue
move({"object": {"label": "red one"}, "place": {"relation": "between", "target": {"label": "green one"}, "target2": {"label": "blue one"}}})

Words: 0:put 1:it 2:there
move({"object": {"last": true}, "place": {"deixis": "there", "word_index": 2}})

Words: 0:move 1:it 2:a 3:bit 4:left
adjust({"dx": -1})

Words: 0:yes
confirm({})

Words: 0:go 1:home
home({})

Words: 0:undo 1:that
undo({})

Words: 0:put 1:that 2:one 3:here 4:and 5:that 6:one 7:there
move({"object": {"deixis": "that", "word_index": 1}, "place": {"deixis": "here", "word_index": 3}})
move({"object": {"deixis": "that", "word_index": 5}, "place": {"deixis": "there", "word_index": 7}})

Words: 0:put 1:the 2:red 3:one 4:on 5:the 6:blue 7:one 8:then 9:go 10:home
move({"object": {"label": "red one"}, "place": {"relation": "on", "target": {"label": "blue one"}}})
home({})

Words: 0:stack 1:them 2:all 3:up
move({"object": {"label": "green one"}, "place": {"relation": "on", "target": {"label": "red one"}}})
move({"object": {"label": "blue one"}, "place": {"relation": "on", "target": {"label": "green one"}}})

Words: 0:swap 1:the 2:red 3:and 4:the 5:blue
move({"object": {"label": "red one"}, "place": {"relation": "next_to", "target": {"label": "green one"}}})
move({"object": {"label": "blue one"}, "place": {"relation": "inside", "target": {"label": "the spot it left"}}})
move({"object": {"label": "red one"}, "place": {"relation": "inside", "target": {"label": "the spot it left"}}})

Words: 0:put 1:each 2:block 3:on 4:its 5:own 6:colour
move({"object": {"label": "red one"}, "place": {"relation": "inside", "target": {"label": "red square"}}})
move({"object": {"label": "green one"}, "place": {"relation": "inside", "target": {"label": "green square"}}})
move({"object": {"label": "blue one"}, "place": {"relation": "inside", "target": {"label": "blue square"}}})

Words: 0:everything 1:except 2:the 3:blue 4:one 5:goes 6:in 7:the 8:red 9:square
move({"object": {"label": "red one"}, "place": {"relation": "inside", "target": {"label": "red square"}}})
move({"object": {"label": "green one"}, "place": {"relation": "next_to", "target": {"label": "red one"}}})

Words: 0:move 1:them 2:all 3:a 4:bit 5:to 6:the 7:left
move({"object": {"label": "red one"}, "place": {"relation": "left_of", "target": {"label": "red one"}}})
move({"object": {"label": "green one"}, "place": {"relation": "left_of", "target": {"label": "green one"}}})
move({"object": {"label": "blue one"}, "place": {"relation": "left_of", "target": {"label": "blue one"}}})

Words: 0:what 1:is 2:the 3:capital 4:of 5:france
decline({})

Words: 0:pick 1:it 2:up 3:and 4:throw 5:it
decline({})

The robot asked: "The red one or the green one?"
Choices: 0: the red one; 1: the green one
Words: 0:the 1:green 2:one
pick({"choice": 1})

The robot asked: "The red one or the green one?"
Choices: 0: the red one; 1: the green one
Words: 0:this 1:one
pick({"word_index": 0})"""  # noqa: E501


PICK = "pick"


@dataclass
class Pick:
    """How the model answered the robot's open question: the choice the person named, or the
    pointing word they said instead ("this one"), whose spot the core then finds."""

    choice: int | None = None  # the Choice's number
    word_index: int | None = None  # they only pointed: this word's number

    def to_dict(self) -> dict[str, Any]:
        return {"choice": self.choice, "word_index": self.word_index}


def pick_tool(choices: Sequence[str], words: Sequence[str]) -> dict[str, Any]:
    """The extra tool the agent gets only while the robot has a question open: it answers
    that question. It is the web demo's own, not the core's - the answer goes to the core as
    an `Answer` on the same request (`answer_entry`), the way a click on the block does.

    Two ways to answer, because only one of them is the model's to decide. If the person
    named a choice ("the green one"), the words say which: `choice`. If they only pointed
    ("this one"), the words cannot say which - where they pointed can, and that is the core's
    job, not the model's: `word_index` names the pointing word and the core finds the spot."""
    return {
        "type": "function",
        "function": {
            "name": PICK,
            "description": "Answer the robot's open question. Give exactly one of the two.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "choice": {
                        "type": "integer",
                        "description": "The number of the Choice the person named.",
                        "enum": list(range(len(choices))),
                    },
                    "word_index": {
                        "type": "integer",
                        "description": 'Instead, if they only pointed ("this one", "that '
                        "one\"): that pointing word's number. The robot works out which one "
                        "they meant from where they pointed.",
                        "enum": [i for i, w in enumerate(words) if norm_word(w) in POINTED_ANSWER],
                    },
                },
            },
        },
    }


# The pointing words a bare answer can use ("this one", "that one"). "here" and "there" are
# about a spot, not a thing, so they are not among them.
POINTED_ANSWER = frozenset({"this", "that", "these", "those"})


def take_pick(
    raw: list[dict[str, Any]], choices: Sequence[str], words: Sequence[str]
) -> tuple[list[dict[str, Any]], Pick | None, str]:
    """The raw calls with any `pick` taken out, how it answered, and why it was refused.
    Neither field, both, a choice outside the Choices list, a word that is not a pointing
    word, or a second `pick`: an error for the retry."""
    rest: list[dict[str, Any]] = []
    pick: Pick | None = None
    why = ""
    for r in raw:
        if r["name"] != PICK:
            rest.append(r)
            continue
        args = r.get("args")
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except ValueError:
                why = "pick: the arguments were not valid JSON"
                continue
        one, bad = _one_pick(args or {}, choices, words)
        if bad:
            why = bad
        elif pick is not None:
            why = "pick: answer once, not several times"
        else:
            pick = one
    return rest, (None if why else pick), why


def _whole(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _one_pick(
    args: dict[str, Any], choices: Sequence[str], words: Sequence[str]
) -> tuple[Pick | None, str]:
    """One `pick` call's arguments as a `Pick`, or why they were refused."""
    choice, word = args.get("choice"), args.get("word_index")
    if (choice is None) == (word is None):
        return None, "pick: give exactly one of choice (they named it) or word_index (they pointed)"  # noqa: E501
    if choice is not None:
        if not _whole(choice) or not 0 <= choice < len(choices):
            return None, f"pick.choice: must be a Choice's number, 0 to {len(choices) - 1}"
        return Pick(choice=choice), ""
    if not _whole(word) or not 0 <= word < len(words):
        return None, f"pick.word_index: must be a word's number, 0 to {len(words) - 1}"
    if norm_word(words[word]) not in POINTED_ANSWER:
        said = " ".join(sorted(POINTED_ANSWER))
        return None, f"pick.word_index: word {word} is '{words[word]}', not one of: {said}"
    return Pick(word_index=word), ""


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


def out_of_reach(call: Call) -> str | None:
    """A place this arm can never use, however well the core resolves it: a named region of
    the work area ("the top left corner") or the largest empty space. Both are worked out in
    the core's 500 x 250 mm rectangle, while the arm reaches a half ring inside it (L2), so
    they land off the mat and the page then says "I can't reach there". Robot-side knowledge,
    not a core limit: the message goes back for the retry."""
    place = call.ref("place") if call.name == "move" else None
    if isinstance(place, NamedPlace):
        what = f'"{place.named}" is a place name'
    elif isinstance(place, EmptySpace):
        what = "empty space is worked out over the whole table"
    else:
        return None
    return (
        f"place: {what}, and the arm reaches a ring around itself only, so it would be out "
        'of reach. Use a pointed spot, or {"relation": "next_to", "target": <a block>}'
    )


def plan_problems(calls: Sequence[Call]) -> list[str]:
    """What is wrong with the plan as a whole, rather than with any one call: too many steps,
    or an answer to the person (yes, stop, undo) mixed into a plan of moves."""
    out: list[str] = []
    if len(calls) > MAX_STEPS:
        out.append(f"{len(calls)} steps is too many: plan at most {MAX_STEPS}")
    alone = [c.name for c in calls if c.name in ALONE]
    if len(calls) > 1 and alone:
        out.append(f"{alone[0]} is never a step of a plan: send it on its own")
    return out


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
        unreachable = out_of_reach(call)
        if unreachable:
            problems = [*problems, unreachable]
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


def arg_text(raw: dict[str, Any]) -> str:
    """One raw call's arguments as the examples write them (they arrive as JSON text)."""
    args = raw.get("args")
    return args if isinstance(args, str) else json.dumps(args or {})


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
    timeout_s: float = 15.0  # one attempt
    budget_s: float = 20.0  # all the attempts for one command together
    max_retries: int = 1
    client_factory: Any = None  # (api_key) -> an OpenAI-like client; tests pass a fake
    _shared: Any = field(default=None, init=False, repr=False)

    def _make(self, key: str, keepalive_s: float) -> Any:
        if self.client_factory is not None:
            return self.client_factory(key)
        import httpx
        from openai import OpenAI

        # One pool that holds the connection open: a fresh TLS handshake to Token Factory
        # costs 2-11 s on a home line, which is most of a command's wait (rahul, 8 Oct).
        http = httpx.Client(
            limits=httpx.Limits(
                max_connections=4, max_keepalive_connections=4, keepalive_expiry=keepalive_s
            )
        )
        return OpenAI(
            base_url=self.base_url, api_key=key, timeout=self.timeout_s, max_retries=0,
            http_client=http,
        )  # fmt: skip

    def _client(self, key: str, shared: bool) -> Any:
        """The client for this call. Our own key keeps one client, so the connection stays
        open between commands; a visitor's key gets a client for that call only, so their
        key is never held on to (plan 8.2)."""
        if not shared:
            return self._make(key, 0.0)
        if self._shared is None:
            self._shared = self._make(key, 600.0)
        return self._shared

    def run(
        self,
        words: Sequence[str],
        key: str,
        context: Context | None = None,
        shared: bool = False,
    ) -> AgentResult:
        """Nemotron alone (raises `AgentUnavailable`). Unless `shared`, `key` is used for
        this call only: a visitor's own key is never kept."""
        if not key:
            raise AgentUnavailable("no Nebius key")
        client = self._client(key, shared)
        choices = (context or Context()).choices
        tools = [*self.tools, pick_tool(choices, words)] if choices else self.tools
        names = {t["function"]["name"] for t in tools}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message(words, context)},
        ]
        result = AgentResult(calls=[], model=self.model)
        t0 = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            # The person is waiting: one slow attempt must not cost them the next one too.
            left = self.budget_s - (time.perf_counter() - t0)
            if attempt and left < 2.0:
                break
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.0,
                    max_tokens=512,
                    timeout=min(self.timeout_s, max(2.0, left)),
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
            rest, result.pick, bad_pick = (
                take_pick(raw, choices, words) if choices else (raw, None, "")
            )
            calls, errors = validate(self.profile, rest, words)
            whole = plan_problems(calls)
            errors += whole
            if bad_pick:
                errors = [bad_pick, *errors]
            if not raw:
                errors = ["no tool call; reply only with tool calls (use ask_user to talk)"]
            result.errors += errors
            result.calls = calls
            # A plan that is wrong as a whole is refused out loud, not trimmed to fit and not
            # handed to the word matcher: the person is told, and can ask for less.
            result.refusal = None
            if whole:
                result.calls = []
                result.refusal = TOO_MANY if len(calls) > MAX_STEPS else NOT_FOLLOWED
            if not errors:
                break
            if attempt < self.max_retries:
                # The retry is one more user turn. Told what it wrote as an assistant turn
                # whose content was JSON text, the model copied that shape and answered in
                # text instead of tool calls: 0 of 6, against 6 of 6 this way (8 Oct).
                wrote = "\n".join(f"{r['name']}({arg_text(r)})" for r in raw) or "(nothing)"
                messages = [*messages[:2], {
                    "role": "user",
                    "content": f"You replied with:\n{wrote}\n"
                    "Your reply had errors, so nothing was done:\n- "
                    + "\n- ".join(errors)
                    + "\nReply again with the complete, corrected list of tool calls.",
                }]  # fmt: skip
        result.latency_s = time.perf_counter() - t0
        return result

    def understand(
        self,
        words: Sequence[str],
        key: str | None,
        context: Context | None = None,
        no_key_why: str = "no Nebius key: simple mode",
        shared: bool = False,
    ) -> AgentResult:
        """Nemotron if it answers with calls, else the rule parser (with why in `errors`).
        Every command goes to the model, "yes" and "stop" included: the rule parser only ever
        answers when the model couldn't (no key, an error, the day's budget).
        `no_key_why` is the reason given when there is no key (simple mode, plan 8.1)."""
        why = no_key_why
        refused = False
        if key:
            try:
                res = self.run(words, key, context, shared)
                res.asked_model = True
                # A refusal is the model's answer too: the rule parser never gets a second
                # go at a plan that was too long.
                if res.ok or res.refusal:
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
