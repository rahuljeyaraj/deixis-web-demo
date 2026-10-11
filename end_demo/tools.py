"""The tools the model is given, and which of them are on the table when.

Three things worth knowing before changing anything here.

**No coordinates, ever.** The core's standing invariant is that the language model never
writes a number for a place (`deixis/refs.py`): every place is a reference - a thing's id, a
named place, or "where they pointed". `place` takes `tray`, `shelf` or `pointed`, and
nothing else. That is not a simplification of the core's reference grammar; it is the same
rule, with this scene's vocabulary.

**The question tools are the contribution.** `ask_this_one`, `ask_choice` and `ask_show_me`
all come straight out of Supported Conversation for Adults with Aphasia (plan 3.7): a yes/no
question, a fixed choice, and "show me". Every one of them is answerable with a point, a nod,
or nothing - never with a word the person has to find. `ask_choice` is the best of the three
because it needs no words from either side, and the descriptions say so, because the model
chooses by reading them.

**The toolset is deliberately swappable.** The three-rounds harness measured 98% with
exactly two tools, `choose` and `ask_about`. Adding tools changes behaviour, so the numbers
have to be re-measured rather than assumed - `Toolset.selection()` reproduces the old two
on purpose, and `Toolset.selection(choice=True, show_me=True)` is the thing E2 measures
against it. Nothing here guesses which is better; the run does.

`cannot` exists so the robot can say it cannot. There is no word-matcher fallback anywhere
in the end demo: Nemotron answers, or the robot says it cannot (plan 8.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ARM = {
    "type": "string",
    "enum": ["left_arm", "right_arm"],
    "description": (
        "Which arm. Leave it out and I use an arm that can do it. Name it when the person "
        "said which one they meant."
    ),
}
THING = {"type": "string", "description": "the thing's id, exactly as the shelf lists it"}
WHY = {"type": "string", "description": "one short line, in plain words"}


def _fn(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


# ---------------------------------------------------------------- deciding which thing

CHOOSE = _fn(
    "choose",
    "Commit: this is the thing the person means. Getting it wrong is costly - they have to "
    "start again, and they have very few words to start again with.",
    {"id": THING, "why": WHY},
    ["id"],
)

ASK_THIS_ONE = _fn(
    "ask_this_one",
    "Hold the gripper over one thing and ask the person: this one? They answer by pointing "
    "at it, or by a nod, or by pointing at a different thing instead. Cheap - it costs them "
    "no words at all. Use it whenever two things are close to equally likely.",
    {"id": THING, "arm": ARM, "why": WHY},
    ["id"],
)

ASK_CHOICE = _fn(
    "ask_choice",
    "Point at one thing, then at the other, and ask: this one, or this one? The person "
    "answers by pointing at the one they mean. This is the best question available: it "
    "needs no words from them and none from you. Use it when the top two are close.",
    {"first": THING, "second": THING, "arm": ARM, "why": WHY},
    ["first", "second"],
)

ASK_SHOW_ME = _fn(
    "ask_show_me",
    "Open a hand and ask them to show you. Use this when you have nothing - nothing inside "
    "their pointing cone, or nothing that fits what they said. Better than a guess, and "
    "better than a question they would need words to answer.",
    {"why": WHY},
    [],
)

CANNOT = _fn(
    "cannot",
    "Say that you cannot do it. Use this rather than guessing when the request is not "
    "something you can carry out.",
    {"why": WHY},
    ["why"],
)

# ---------------------------------------------------------------- doing the job

LOOK = _fn(
    "look",
    "Look at the shelf again, and at what each arm is holding. Use it after something has "
    "moved and you are not sure what is where.",
    {},
    [],
)

POINT_AT = _fn(
    "point_at",
    "Hold the gripper over a thing without asking anything and without moving it. Use it to "
    "show the person what you are about to do.",
    {"id": THING, "arm": ARM},
    ["id"],
)

PICK = _fn(
    "pick",
    "Take a thing off the shelf. Each arm only reaches part of the shelf - the shelf "
    "listing says which part.",
    {"id": THING, "arm": ARM},
    ["id"],
)

PASS_TO = _fn(
    "pass_to",
    "Pass the thing being held to the other arm. The arm at the far end of the table cannot "
    "reach the person, so anything it picks up has to come across this way.",
    {"arm": {**ARM, "description": "the arm that should take it"}},
    ["arm"],
)

PLACE = _fn(
    "place",
    "Put the held thing down: in the tray, or back where it came from. Use it when they "
    "have changed their mind, or to free a gripper for something else.",
    {
        "where": {
            "type": "string",
            "enum": ["tray", "shelf"],
            "description": "the tray, or back where it came from",
        },
        "arm": ARM,
    },
    ["where"],
)

HAND_OVER = _fn(
    "hand_over",
    "Put the held thing into the person's hand. Only the arm beside them can do this.",
    {"arm": ARM},
    [],
)

DONE = _fn(
    "done",
    "The job is finished - the thing they wanted is in their hand. Say so and stop.",
    {},
    [],
)


@dataclass(frozen=True)
class Toolset:
    """Which tools are on the table. See the module docstring on why this is a dial."""

    tools: tuple[dict[str, Any], ...]
    #: A name the run's header can print, so no number is ever quoted without its toolset.
    label: str

    def as_list(self) -> list[dict[str, Any]]:
        return [dict(t) for t in self.tools]

    def names(self) -> list[str]:
        return [t["function"]["name"] for t in self.tools]

    def has(self, name: str) -> bool:
        return name in self.names()

    @staticmethod
    def selection(choice: bool = False, show_me: bool = False, cannot: bool = False) -> Toolset:
        """Deciding which thing, and nothing else. The default pair is exactly what the
        three-rounds harness measured, so the port can be checked against it."""
        tools = [CHOOSE, ASK_THIS_ONE]
        bits = ["choose", "ask_this_one"]
        if choice:
            tools.append(ASK_CHOICE)
            bits.append("ask_choice")
        if show_me:
            tools.append(ASK_SHOW_ME)
            bits.append("ask_show_me")
        if cannot:
            tools.append(CANNOT)
            bits.append("cannot")
        return Toolset(tuple(tools), "+".join(bits))

    @staticmethod
    def commit_only() -> Toolset:
        """Out of questions. The model still makes the final call - if the harness picked a
        fallback here, part of every score would be the harness's judgement, which is the
        bug that cost the three-rounds numbers two points once it was fixed."""
        return Toolset((CHOOSE,), "choose")

    @staticmethod
    def job(can_ask: bool = True) -> Toolset:
        """The whole job: work out which thing, then get it into their hand.

        `can_ask=False` is what the question cap actually does. Taking the tools away is
        better than telling the model off for using them: a model that has been refused
        goes on trying, which burns the step budget and reads, in a transcript, as a robot
        arguing with its own harness. Seen on the first dry run.
        """
        asking = (ASK_THIS_ONE, ASK_CHOICE, ASK_SHOW_ME) if can_ask else ()
        return Toolset(
            (LOOK, *asking, POINT_AT, PICK, PASS_TO, PLACE, HAND_OVER, CANNOT, DONE),
            "the whole job" if can_ask else "the whole job, out of questions",
        )


#: The tools that ask the person something. Everything else either moves or commits.
QUESTIONS = ("ask_this_one", "ask_choice", "ask_show_me")
