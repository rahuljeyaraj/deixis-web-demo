"""What the robot says, writes and does - all three, every time.

The robot here is a **communication partner trained in Supported Conversation for Adults
with Aphasia** (SCA, Aphasia Institute). That is an evidence-based method and it reads
almost like a specification for this module:

- short, simple sentences; **one question at a time**;
- **write the keywords down in large bold print** as well as saying them;
- **gesture**; show, don't describe;
- ask **yes/no** and **fixed-choice** questions, never open ones;
- ask the person to **point, gesture or show you**;
- **give them time.**

So an `Utterance` is not a string. It is three channels at once - spoken, written and
gestured - and it is a dataclass with no defaults for any of them, because the rule that
every utterance appears all three ways (plan R8, E12) should be impossible to break by
forgetting rather than caught by review.

**Why not the core's `ASK_TEXTS` as they stand.** The core already has a question for an
ambiguous object: `"The {options}?"` - "the blue mug or the grey one?". That demands words
back, which is the one thing the person we built this for cannot give (plan 8.4). So this
module supplies its own texts and feeds them to the core through the seam the core already
has for exactly this: `ResolverParams.asks`, looked up by `deixis.where.ask_text`. The core
is used, not reimplemented, and a profile can still override any of it.

**The written line is not the spoken line.** Spoken is a sentence; written is two or three
keywords in capitals, because that is what carries across a room at 28 pt and what the SCA
guidance asks for. Keeping them in separate fields is deliberate: shrinking a spoken
sentence to fit a screen is how you end up with neither.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from deixis.where import ask_text

#: What the arm does while the robot speaks. "hover" holds the gripper over one thing and
#: the thing glows; "each" hovers over the first, then the second, which is what makes a
#: fixed choice answerable with no words at all; "open" is an open, waiting hand.
GestureKind = Literal["none", "hover", "each", "open", "give", "take"]

UtteranceKind = Literal[
    "this_one",
    "choice",
    "show_me",
    "contradiction",
    "heard",
    "cannot",
    "giving",
    "done",
    "waiting",
]

#: The robot's questions, as an override for the core's `ASK_TEXTS`. The keys that exist in
#: the core (`ambiguous_object`, `no_object`, ...) are replaced with versions that can be
#: answered by a point or a nod; the rest are new and come from SCA.
#:
#: Pass this as `ResolverParams(asks=tuple(ASKS.items()))` and the core asks the person
#: these instead of its own.
ASKS: dict[str, str] = {
    # Replacing core texts that would demand words back.
    "ambiguous_object": "This one?",
    "ambiguous_object.two": "This one... or this one?",
    "no_object": "Show me.",
    "no_object.empty": "Show me.",
    "ambiguous": "This one... or this one?",
    "no_gesture": "Show me.",
    "unknown_label": "Show me.",
    # New, from SCA (plan 3.7, 8.1).
    "this_one": "This one?",
    "choice": "This one... or this one?",
    "show_me": "Show me.",
    "contradiction": "Sorry. This one, or this one?",
    "giving": "Here you are.",
    "done": "There you go.",
    "cannot": "I can't do that.",
    "waiting": "Take your time.",
}

#: The written line: keywords, capitals, no punctuation to read. 28 pt on the page.
WRITTEN: dict[str, str] = {
    "this_one": "THIS ONE?",
    "choice": "THIS ONE  OR  THIS ONE?",
    "show_me": "SHOW ME",
    "contradiction": "WHICH ONE?",
    "heard": "I HEARD",
    "cannot": "I CAN'T",
    "giving": "HERE",
    "done": "DONE",
    "waiting": "TAKE YOUR TIME",
}


@dataclass(frozen=True)
class Gesture:
    """What the arm is doing while the robot speaks. The third channel (R8)."""

    kind: GestureKind
    #: The things it hovers over, in the order it hovers over them.
    ids: tuple[str, ...]
    #: Which arm. Two arms means a gesture always belongs to one of them.
    arm: str

    def __post_init__(self) -> None:
        if self.kind in ("hover", "each") and not self.ids:
            raise ValueError(f"a {self.kind} gesture has to be over something")
        if self.kind == "each" and len(self.ids) != 2:
            raise ValueError("a fixed choice is between exactly two things")


@dataclass(frozen=True)
class Utterance:
    """One thing the robot says. Spoken, written and gestured - all three, always."""

    kind: UtteranceKind
    spoken: str
    written: str
    gesture: Gesture

    def __post_init__(self) -> None:
        # R8 is enforced here rather than reviewed for. An utterance with a missing channel
        # is not a lesser utterance; it is a bug in a demo whose whole claim is that the
        # person never has to read, or hear, or guess, only one of the three.
        if not self.spoken.strip():
            raise ValueError(f"{self.kind}: nothing to say")
        if not self.written.strip():
            raise ValueError(f"{self.kind}: nothing written (R8)")

    @property
    def is_question(self) -> bool:
        return self.kind in ("this_one", "choice", "show_me", "contradiction")

    @property
    def needs_words(self) -> bool:
        """Can this be answered without words? It always can - that is R7. The property
        exists so a test can assert it over every utterance the robot can make."""
        return False


def _text(key: str, variant: str | None = None, **fill: object) -> str:
    return ask_text(key, variant, ASKS, **fill)


def this_one(thing_id: str, arm: str) -> Utterance:
    """The gripper hovers over one thing and the robot asks. Answerable with a nod, or -
    better, because yes/no is not always reliable (plan 3.6) - with a second point."""
    return Utterance(
        "this_one", _text("this_one"), WRITTEN["this_one"], Gesture("hover", (thing_id,), arm)
    )


def choice(first: str, second: str, arm: str) -> Utterance:
    """Point at one, then the other. The best interaction in the demo: it needs no words at
    all, from either side. The person answers by pointing at the one they mean."""
    return Utterance(
        "choice", _text("choice"), WRITTEN["choice"], Gesture("each", (first, second), arm)
    )


def show_me(arm: str) -> Utterance:
    """Nothing in the cone. An open hand and two words, rather than a guess or a question
    that needs a sentence back."""
    return Utterance("show_me", _text("show_me"), WRITTEN["show_me"], Gesture("open", (), arm))


def contradiction(first: str, second: str, arm: str) -> Utterance:
    """The word and the hand disagreed. The robot says so and asks again; it never acts on
    the contradiction (plan R5)."""
    return Utterance(
        "contradiction",
        _text("contradiction"),
        WRITTEN["contradiction"],
        Gesture("each", (first, second), arm),
    )


def heard(said: str, arm: str) -> Utterance:
    """What the recogniser thought it heard, shown plainly and never blocking (plan R6).

    Dysarthric word recognition runs about 64-77%, so a misheard word is the normal case,
    not the edge case. It is shown so the person can see why the robot is confused, and the
    point can override it.
    """
    text = said.strip() or "(nothing)"
    return Utterance("heard", f'I heard: "{text}"', f"I HEARD\n{text}", Gesture("none", (), arm))


def cannot(why: str, arm: str) -> Utterance:
    """The robot cannot do it, and says so.

    This exists so there is never a reason to fall back on a word matcher. If Nemotron
    cannot be reached, the robot says it cannot. It does not guess (plan 8.4).
    """
    line = _text("cannot")
    return Utterance(
        "cannot",
        f"{line} {why}".strip() if why else line,
        WRITTEN["cannot"],
        Gesture("none", (), arm),
    )


def giving(thing_id: str, arm: str) -> Utterance:
    """Handing it over. The top-ranked thing users of assistive arms ask for (plan 3.4)."""
    return Utterance(
        "giving", _text("giving"), WRITTEN["giving"], Gesture("give", (thing_id,), arm)
    )


def taking(thing_id: str, arm: str) -> Utterance:
    """Passing a thing to the other arm, because this one cannot reach the person."""
    return Utterance(
        "giving", _text("giving"), WRITTEN["giving"], Gesture("take", (thing_id,), arm)
    )


def done(arm: str) -> Utterance:
    return Utterance("done", _text("done"), WRITTEN["done"], Gesture("none", (), arm))


def waiting(arm: str) -> Utterance:
    """The visible waiting state. There is no timeout that cuts the person off: the person
    or the Stop button ends it, nothing else (plan 8.1)."""
    return Utterance("waiting", _text("waiting"), WRITTEN["waiting"], Gesture("open", (), arm))
