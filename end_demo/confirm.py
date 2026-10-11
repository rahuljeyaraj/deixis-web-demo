"""How an answer to "this one?" is read: the hand first, and never act on a contradiction.

This module exists because of the most awkward finding in the research, and it cuts against
the mechanism the whole demo is built on.

**Yes/no is not always reliable.** In severe aphasia, yes/no answers can be wrong or
inconsistent. It matters enough clinically that there is a dedicated instrument - the
Yes/No Questionnaire, ten closed questions - whose only job is to sort people into reliable
and unreliable responders, and studies of people with severe communication difficulty found
some who misunderstood the task entirely, so their answers could not be interpreted at all.

When someone *is* a reliable responder, a yes is dependable. We cannot assume it. So:

- **R4: the hand outranks the word.** A point at the thing the gripper is over is a yes. A
  point at something else is not a rejection to re-ask about - it is the new answer.
- **R5: a contradiction is never acted on.** If the word says yes and the hand says a
  different thing, the robot says so and asks again. Every one is counted, because the
  count is a real result and hiding it would be the dishonest kind of polish.

**On the yes/no reading, and the no-word-matcher rule.** `yes_no` below looks for literal
yes and no words. That is not the forbidden word matcher: the forbidden thing is a word
matcher deciding **which thing the person means**, and nothing here decides that - the
brain does, from the whole transcript, including the raw text of what was heard. What
`yes_no` decides is only whether a *contradiction* exists, and that has to be mechanical
for the same reason the word budget is mechanical rather than a model's judgement: a guard
that refuses to act must be impossible to talk out of, and must never misfire in front of a
judge. The brain is told the answer and the raw words both, and draws its own conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from end_demo.point import RATIO, Cone, sole

#: Literal yes and no, for the contradiction guard only - see the module docstring.
YES = frozenset({"yes", "yeah", "yep", "yup", "aye", "mm", "mhm", "uhhuh", "ok", "okay", "right"})
NO = frozenset({"no", "nope", "nah", "not", "wrong"})

AnswerKind = Literal[
    "yes",  # they confirmed the thing the gripper is over
    "no",  # they rejected it, and showed nothing else
    "new_target",  # they pointed at something else: that is the answer now
    "contradiction",  # the word and the hand disagree - never act (R5)
    "unclear",  # they pointed, but the point settles nothing
    "said",  # words that are not yes or no: for the brain to read, never a blocker (R6)
    "nothing",  # silence. Wait, then ask differently. Do not act.
]


def yes_no(text: str) -> bool | None:
    """Yes, no, or neither. The contradiction guard's only input - see the module docstring.

    A misheard word comes back as neither, which is the safe direction: the point rules and
    nothing is blocked (R6).
    """
    words = [w.strip(".,!?;:'\"").lower() for w in text.split()]
    words = [w.replace("-", "").replace("_", "") for w in words if w]
    if not words:
        return None
    hit_no = any(w in NO for w in words)
    hit_yes = any(w in YES for w in words)
    if hit_no and not hit_yes:
        return False
    if hit_yes and not hit_no:
        return True
    return None  # both, or neither: not a yes/no answer


@dataclass(frozen=True)
class Reply:
    """Everything the person did in response to one question.

    All of it optional, because the answer may be a point with no words, a nod with no
    point, words with no point, or nothing at all - and every one of those has to mean
    something (plan R7). `said` is kept raw and never cleaned up.
    """

    point: Cone | None = None
    said: str = ""
    #: A nod or a shake, if the tracker saw one. Same standing as a spoken yes: outranked
    #: by the hand.
    nod: bool | None = None

    @property
    def silent(self) -> bool:
        return self.point is None and not self.said.strip() and self.nod is None


@dataclass(frozen=True)
class Answer:
    """What the reply amounted to, and what the robot is allowed to do about it."""

    kind: AnswerKind
    #: The thing the hand indicated, when it indicated one.
    id: str = ""
    #: The word's own verdict, kept separate from the hand's so the disagreement is visible.
    word: bool | None = None
    said: str = ""
    why: str = ""

    @property
    def acts(self) -> bool:
        """May the robot act on this? Only a plain yes, or a clear new target."""
        return self.kind in ("yes", "new_target")


@dataclass
class Counts:
    """What the session is counting. These are the result, not diagnostics.

    `contradictions` in particular: plan R5 says log every one and report the count. A demo
    that quietly swallowed them would be claiming a reliability it had not earned.
    """

    questions: int = 0
    answers_by_point: int = 0
    answers_by_word: int = 0
    contradictions: int = 0
    unclear: int = 0
    silences: int = 0
    words_spoken: int = 0
    kinds: dict[str, int] = field(default_factory=dict)

    def add(self, a: Answer) -> None:
        self.kinds[a.kind] = self.kinds.get(a.kind, 0) + 1
        if a.kind == "contradiction":
            self.contradictions += 1
        elif a.kind == "unclear":
            self.unclear += 1
        elif a.kind == "nothing":
            self.silences += 1


def read(reply: Reply, hovering: str, ratio: float = RATIO) -> Answer:
    """Read a reply to "this one?", hand first (R4), and refuse a contradiction (R5).

    `hovering` is the thing the gripper is over - the thing the question was about.

    The order is the whole point:

    1. **The hand.** If they pointed and the point settles on something, that is the
       answer: the hovered thing means yes, anything else means *that* one instead.
    2. **The word, cross-checked.** A yes that disagrees with the hand is a contradiction
       and the robot acts on neither half of it.
    3. **The word alone**, when there was no point.
    4. **Nothing**, which is an answer too: wait, then ask differently, and do not act.
    """
    word = reply.nod if reply.nod is not None else yes_no(reply.said)
    said = reply.said.strip()

    if reply.point is not None:
        pointed = sole(reply.point, ratio)
        if pointed is None:
            # They pointed, but the point settles nothing - the top two are inside the
            # core's 2:1 rule, or the cone is empty. Not an answer; ask differently.
            empty = not reply.point.candidates
            why = "nothing in the cone" if empty else "the point was too close to call"
            return Answer("unclear", word=word, said=said, why=why)
        agrees = pointed == hovering
        if word is None:
            return Answer(
                "yes" if agrees else "new_target",
                id=pointed,
                word=None,
                said=said,
                why="they pointed at it" if agrees else "they pointed at something else",
            )
        if word and not agrees:
            # "Yes" while pointing somewhere else. Never act on this (R5).
            return Answer(
                "contradiction",
                id=pointed,
                word=True,
                said=said,
                why=f"said yes but pointed at {pointed}, not {hovering}",
            )
        if not word and agrees:
            # "No" while pointing straight at it. Also a contradiction.
            return Answer(
                "contradiction",
                id=pointed,
                word=False,
                said=said,
                why=f"said no but pointed at {hovering}",
            )
        if word and agrees:
            return Answer("yes", id=pointed, word=True, said=said, why="word and hand agree")
        # Said no, and pointed elsewhere: consistent. They told us what they meant instead.
        return Answer(
            "new_target", id=pointed, word=False, said=said, why="said no and showed which one"
        )

    if word is not None:
        return Answer("yes" if word else "no", word=word, said=said, why="no point, took the word")
    if said:
        # Words that are neither yes nor no. Shown as "I heard ...", passed to the brain,
        # and never allowed to block (R6).
        return Answer("said", said=said, why="not a yes or a no")
    return Answer("nothing", why="no point, no word, no nod")


def read_choice(reply: Reply, first: str, second: str, ratio: float = RATIO) -> Answer:
    """Read a reply to "this one... or this one?".

    A fixed choice is answered by pointing at one of the two, which needs no words from
    anybody. A point at a third thing is still taken as the answer - the person has just
    told us both options were wrong and which one is right, and refusing that would be
    making them repeat themselves.
    """
    if reply.point is not None:
        pointed = sole(reply.point, ratio)
        if pointed is None:
            return Answer("unclear", said=reply.said.strip(), why="the point was too close to call")
        if pointed in (first, second):
            return Answer("yes", id=pointed, said=reply.said.strip(), why="they pointed at one")
        return Answer(
            "new_target", id=pointed, said=reply.said.strip(), why="they pointed at a third thing"
        )
    # No point. A fixed choice cannot be answered with a bare yes or no - "yes" to "this one
    # or this one?" says nothing - so a word here is words for the brain to read, not an
    # answer to this question.
    said = reply.said.strip()
    if said:
        return Answer("said", said=said, why="a choice cannot be answered with a word")
    return Answer("nothing", why="no point, no word")
