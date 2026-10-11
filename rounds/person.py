"""The person on the other side: knows what they want, has very few words for it.

This stands in for someone with anomia - the loss of the names of things, which is present
in everyone who has aphasia, and in about a third of stroke survivors. The textbook example
is that they can pick up a pencil and write with it but cannot say the word "pencil".

So this person is modelled as three abilities and one limit:

- they can **point** (badly, like everyone: `pointer.py`);
- they can **answer yes or no**, truthfully and reliably;
- they can **say a few words** about the thing, but not its name - and the words they have
  are the same for two things of the same kind, because that is the real situation;
- the limit is a **word budget**. A budget is used instead of filtering out forbidden words
  because a budget is mechanical and visible: it counts, it never misfires, and it mirrors
  the real thing (someone with severe aphasia produces one or two words at a time).

`words_used` and `answers` are what the scoreboard reports. They matter as much as whether
the robot got it right: the claim this project makes is about *how little* a person has to
produce, not only about accuracy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from rounds.pointer import SIGMA_DEG, Cone, point_at
from rounds.scene import Thing


@dataclass
class Person:
    """Someone who wants one thing off the shelf and has almost no words for it."""

    target: Thing
    things: list[Thing]
    rng: random.Random
    sigma_deg: float = SIGMA_DEG
    words_used: int = 0
    answers: int = 0
    points: int = 0
    said: list[str] = field(default_factory=list)

    def say(self, budget: int) -> str:
        """What they manage to get out, inside the budget. Empty when the budget is 0."""
        if budget <= 0:
            return ""
        fits = [p for p in self.target.says if len(p.split()) <= budget]
        phrase = fits[0] if fits else " ".join(self.target.says[0].split()[:budget])
        self.words_used += len(phrase.split())
        self.said.append(phrase)
        return phrase

    def point(self) -> Cone:
        """They point at it. They miss a bit."""
        self.points += 1
        return point_at(self.target, self.things, self.rng, self.sigma_deg)

    def answer(self, thing_id: str) -> bool:
        """The robot pointed at something and asked. Yes or no - one word, and it is free:
        a nod costs nothing, which is exactly why asking is the cheap move here."""
        self.answers += 1
        return thing_id == self.target.id
