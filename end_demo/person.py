"""The person on the other side: knows exactly what they want, has almost no words for it.

This stands in for someone with anomia - the loss of the names of things, which is present
in everyone who has aphasia and in about a third of stroke survivors. The textbook
description is that they can pick up a pencil and write with it but cannot say the word
"pencil". Their thinking is untouched.

So the person is modelled as three abilities and two limits:

- they can **point**, badly, like everyone (`point.py`);
- they can **answer**, by pointing again or by a word;
- they can get **a word or two** out about a thing, but not its name - and two things of the
  same kind have the same words, because that is the real situation and not a trick;
- the first limit is a **word budget**, a counter. A counter rather than a model policing
  which phrases are allowed, because a counter is mechanical and visible, it never misfires
  in front of a judge, and a filter could never be complete anyway ("second from the left"
  is a position, not a name);
- the second limit is the hard one: **their yes and no are not always reliable** (plan 3.6).

## The unreliable responder, and why the hand stays honest

In severe aphasia a yes or no can be wrong or inconsistent, while pointing and understanding
stay intact. `unreliable` is the fraction of spoken yes/no answers that come out wrong - and
it flips **only the word**, never the point. That is not a convenience: it is the clinical
picture, and it is what makes the robot's contradiction rule (R5) do real work instead of
sitting in the code untested. Turn it up and the robot should ask again more often and
should still never act on a disagreement; if accuracy collapses instead, R5 is broken.

## Two people

`Person` is scripted: its phrases come out of `scene.py`, so it is deterministic and good
for regression. `ModelPerson` is a **second Nemotron session** given a goal and a word
budget, which is how tau-bench evaluates agents, and it is what makes 200 unattended trials
worth running - a scripted person is tidier than a real one, and tidiness is the thing most
likely to be flattering us.

In both, the **point comes from geometry, never from a model**. A language model has no hand
and should not be asked to pretend it has aim; `point.py` draws the wobble. The model plays
the language, the geometry plays the hand.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Literal

from end_demo.brain import BrainUnavailable, Decider
from end_demo.confirm import Reply
from end_demo.point import SIGMA_DEG, Cone, point_at
from end_demo.scene import Thing

#: How the person answers a question. "word" is yes/no only, which is what the three-rounds
#: harness measured; "both" is the end demo, where they point again as well (R4); "point" is
#: someone with no usable speech at all.
AnswerStyle = Literal["word", "point", "both"]


@dataclass
class Person:
    """Someone who wants one thing off the shelf and has almost no words for it."""

    target: Thing
    things: list[Thing]
    rng: random.Random
    sigma_deg: float = SIGMA_DEG
    style: AnswerStyle = "both"
    #: Fraction of spoken yes/no answers that come out wrong. Only the word is affected.
    unreliable: float = 0.0
    #: Fraction of questions they do not answer at all. There is no timeout, so a silence
    #: has to be survivable (plan 8.1, E13).
    silence: float = 0.0
    #: Say as little as possible rather than as much as the budget allows. Off by default,
    #: because a person uses the words they have and a person who says *less* than they
    #: could makes the words-only baseline look worse than it really is - and that baseline
    #: is the thing we are claiming to beat. On for the port check against the harness,
    #: whose person always took its shortest phrase.
    terse: bool = False

    words_used: int = 0
    points: int = 0
    answers: int = 0
    #: Every yes/no they gave, and whether it was the truth. The count of the wrong ones is
    #: reported, not hidden: it is what the robot had to cope with.
    misled: int = 0
    said: list[str] = field(default_factory=list)

    # ------------------------------------------------------------ what they say

    def say(self, budget: int) -> str:
        """What they manage to get out, inside the budget. Empty when the budget is 0.

        With room to spare they use it - the fullest phrase that fits, not the shortest
        they could get away with. See `terse` for why that direction matters.
        """
        if budget <= 0:
            return ""
        fits = [p for p in self.target.says if len(p.split()) <= budget]
        if fits:
            phrase = fits[0] if self.terse else max(fits, key=lambda p: len(p.split()))
        else:
            phrase = " ".join(self.target.says[0].split()[:budget])
        self.words_used += len(phrase.split())
        self.said.append(phrase)
        return phrase

    # ------------------------------------------------------------ what their hand does

    def point(self) -> Cone:
        """They point at it. They miss a bit - everyone does."""
        self.points += 1
        return point_at(self.target, self.things, self.rng, self.sigma_deg)

    # ------------------------------------------------------------ answering

    def _word(self, truth: bool) -> bool:
        """Their spoken yes or no, which is `unreliable` of the time not the truth."""
        self.answers += 1
        if self.unreliable and self.rng.random() < self.unreliable:
            self.misled += 1
            return not truth
        return truth

    def _quiet(self) -> bool:
        return bool(self.silence) and self.rng.random() < self.silence

    def to_this_one(self, thing_id: str) -> Reply:
        """The robot is hovering over `thing_id` and asked "this one?"."""
        if self._quiet():
            return Reply()
        truth = thing_id == self.target.id
        cone = self.point() if self.style in ("point", "both") else None
        word = "" if self.style == "point" else ("yes" if self._word(truth) else "no")
        return Reply(point=cone, said=word)

    def to_choice(self, first: str, second: str) -> Reply:
        """The robot pointed at two things in turn. They answer by pointing - this question
        needs no words from either side, which is why it is the best one available."""
        if self._quiet():
            return Reply()
        if self.style == "word":
            # A fixed choice cannot be answered with a word, so someone restricted to
            # yes/no genuinely cannot answer it. Reporting that honestly matters more than
            # letting the harness invent an answer.
            return Reply(said="")
        return Reply(point=self.point())

    def to_show_me(self) -> Reply:
        """They show the robot. A fresh point, with fresh wobble."""
        if self._quiet():
            return Reply()
        return Reply(point=self.point())


# ---------------------------------------------------------------- a second model as the person

PERSON_SYSTEM = """You are playing a person who wants one particular thing off a shelf.

You have had a stroke. Your thinking is completely normal, but the names of things will not
come - you cannot say what anything is called. You can point, and you can get one or two
words out, and you can nod or shake your head.

So: say as little as possible, and never say the thing's name. Say what it is for, or what
you do with it, or just "that one". If several things on the shelf are the same as each
other, there is no word that would help anyway, so do not try to invent one.

You are not acting distressed and you are not explaining your condition. You just want the
thing.

Answer only with a tool call."""

PERSON_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "say",
            "description": (
                "Say a few words about the thing you want. Not its name - you cannot "
                "produce that. Keep it as short as you can."
            ),
            "parameters": {
                "type": "object",
                "properties": {"words": {"type": "string"}},
                "required": ["words"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Answer the robot's yes/no question.",
            "parameters": {
                "type": "object",
                "properties": {"yes": {"type": "boolean"}},
                "required": ["yes"],
            },
        },
    },
]


@dataclass
class ModelPerson(Person):
    """A second Nemotron session as the person: a goal, a word budget, and a hand.

    Everything `Person` does mechanically, this does too - the budget is still a counter
    applied to whatever the model says, and the point still comes from geometry. What
    changes is the language: a real person does not read their phrases out of `scene.py`.

    When the model cannot be reached, this falls back on the scripted phrases and says so
    through `fell_back`. That is a *test harness* falling back, not the robot: the robot
    itself never falls back on anything (plan 8.4).
    """

    model: Decider | None = None
    fell_back: int = 0
    _session: Any = None

    def _ask_model(self, prompt: str, want: str) -> dict[str, Any] | None:
        if self.model is None:
            return None
        from end_demo.brain import Session

        if self._session is None:
            self._session = Session(self.model, system=PERSON_SYSTEM)
            self._session.tell(self._brief())
        self._session.tell(prompt)
        try:
            call = self._session.decide(PERSON_TOOLS)
        except BrainUnavailable:
            self.fell_back += 1
            return None
        self._session.result(call, "ok")
        if call.tool != want:
            self.fell_back += 1
            return None
        return call.args

    def _brief(self) -> str:
        """What this person knows: which thing they want, and what is on the shelf."""
        lines = ["On the shelf, left to right, top shelf then lower:"]
        for t in self.things:
            lines.append(f"  {t.seen}")
        lines += [
            "",
            f"The one you want is: {self.target.seen}, "
            + ("top shelf" if self.target.row == 0 else "lower shelf")
            + f", number {self.target.col + 1} from the left.",
            "",
            "You cannot say which number it is and you cannot say what it is called. You can "
            "point at it, and the robot can see roughly where you pointed.",
        ]
        return "\n".join(lines)

    def say(self, budget: int) -> str:
        """Whatever the model gets out, cut to the budget by a counter."""
        if budget <= 0:
            return ""
        args = self._ask_model(
            f"Say what you want. You can manage about {budget} words, no more.", "say"
        )
        if args is None:
            return super().say(budget)
        words = str(args.get("words", "")).split()[:budget]
        phrase = " ".join(words)
        if not phrase:
            return super().say(budget)
        self.words_used += len(words)
        self.said.append(phrase)
        return phrase

    def to_this_one(self, thing_id: str) -> Reply:
        if self._quiet():
            return Reply()
        truth = thing_id == self.target.id
        cone = self.point() if self.style in ("point", "both") else None
        if self.style == "point":
            return Reply(point=cone)
        seen = next((t.seen for t in self.things if t.id == thing_id), thing_id)
        args = self._ask_model(
            f"The robot is holding its gripper over: {seen}. It asks: this one?", "answer"
        )
        if args is None:
            word = "yes" if self._word(truth) else "no"
        else:
            # The model's own answer, then the unreliability on top of it: a responder whose
            # yes/no cannot be trusted is not the same as one who is lying on purpose.
            said_yes = bool(args.get("yes"))
            word = "yes" if self._word(said_yes) else "no"
        return Reply(point=cone, said=word)


def pick_target(things: list[Thing], rng: random.Random) -> Thing:
    """Something a person would actually ask for."""
    from end_demo.scene import wanted

    return rng.choice(wanted(things))


def brief_json(person: Person) -> str:
    """What the person cost, for the attempt record."""
    return json.dumps(
        {
            "words": person.words_used,
            "points": person.points,
            "answers": person.answers,
            "misled": person.misled,
            "said": person.said,
        }
    )
