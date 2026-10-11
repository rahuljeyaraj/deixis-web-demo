"""The three rounds, run on the same shelf, the same target and the same missed point.

What a judge will do at the booth, and what this harness does a hundred times over:

1. **Point only.** They point; the robot takes its best guess and goes. No words, no
   questions. This is how published pointing systems work, and they report 51-69%.
2. **Words only.** No pointing, four words. This is how every voice assistant works, and on
   four identical bottles there is no answer.
3. **Point, a word or two, and the robot may ask.** Ours.

The design is **paired**: inside one trial all three rounds want the same thing, and rounds
1 and 3 are handed *the same point*, with the same wobble, drawn once. So round 3 cannot win
by getting a luckier aim - the only thing it has that round 1 does not is the words and the
right to ask. That makes the difference between the two numbers attributable, not suggestive.

Scoring is deliberately strict:

- The robot must end up on the thing the person wanted. A yes to "this one?" counts as
  getting there (the person confirmed it); nothing else does.
- A model that names a thing that is not on the shelf scores zero and is recorded as
  `invalid`, never quietly retried.
- Words are counted as the *person* spoke them, because that is the cost this project
  claims to lower. A yes or no is not counted: it is a nod.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from rounds.brain import Brain, Decision
from rounds.person import Person
from rounds.pointer import SIGMA_DEG, Cone
from rounds.scene import Thing, by_id, kind_counts

#: Round 2's budget: enough for "small bottle", not enough to describe a shelf position.
WORDS_ONLY_BUDGET = 4
#: Round 3's budget: what someone with severe aphasia gets out at once.
POINT_AND_ASK_BUDGET = 2
#: How many times the robot may point at something and ask before it has to commit.
MAX_ASKS = 3


@dataclass
class Attempt:
    """One go at one round."""

    round: str
    target: str
    picked: str
    words: int
    asks: int
    said: list[str] = field(default_factory=list)
    cone: list[tuple[str, float]] = field(default_factory=list)
    ambiguous: bool = False
    #: Was the thing they wanted one of a kind, or one of several identical ones? This is
    #: the split that matters: words can only ever work on the one-of-a-kind things.
    unique: bool = True
    invalid: bool = False  #: the brain named something that is not on the shelf
    failed: str = ""  #: the brain could not be reached at all
    log: list[str] = field(default_factory=list)
    #: Every model call this attempt made: what it decided, why, what it was thinking, how
    #: long it took and what it spent. Round 1 has none - it never asks a model.
    thoughts: list[dict[str, object]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.picked == self.target and not self.invalid and not self.failed

    @property
    def model_calls(self) -> int:
        return len(self.thoughts)

    @property
    def model_s(self) -> float:
        return sum(float(t.get("latency_s", 0.0)) for t in self.thoughts)


def _remember(a: Attempt, what: str, d: Decision) -> None:
    """Keep one model call on the attempt, so the page can show the thinking."""
    a.thoughts.append(
        {
            "at": what,
            "action": d.action,
            "id": d.id,
            "why": d.why,
            "thinking": d.thinking,
            "latency_s": d.latency_s,
            "out_tokens": d.out_tokens,
            "ran_out_thinking": d.ran_out_thinking,
        }
    )


def _cone_pairs(cone: Cone) -> list[tuple[str, float]]:
    return [(c.id, round(c.weight, 3)) for c in cone.candidates]


def point_only(target: Thing, cone: Cone) -> Attempt:
    """Round 1: the point, and nothing else. No model is asked - there is nothing to think
    about. The robot takes the likeliest thing in the cone and commits."""
    picked = cone.best or ""
    a = Attempt(
        round="point only",
        target=target.id,
        picked=picked,
        words=0,
        asks=0,
        cone=_cone_pairs(cone),
        ambiguous=cone.ambiguous,
    )
    inside = ", ".join(f"{i} {w:.2f}" for i, w in a.cone) or "nothing in the cone"
    a.log = [
        f"person  points at {target.id} (and misses a little)",
        f"cone    {inside}",
        f"robot   takes the best one: {picked or 'nothing'}",
    ]
    if cone.ambiguous and len(cone.candidates) > 1:
        a.log.append("        (the core would have said: too close to call, ask)")
    return a


def words_only(
    target: Thing,
    things: list[Thing],
    person: Person,
    brain: Brain,
    budget: int = WORDS_ONLY_BUDGET,
) -> Attempt:
    """Round 2: no pointing. The person's few words against the whole shelf."""
    said = person.say(budget)
    a = Attempt(
        round="words only",
        target=target.id,
        picked="",
        words=len(said.split()),
        asks=0,
        said=[said],
    )
    a.log = [f'person  says "{said}" ({a.words} of {budget} words), and cannot point']
    try:
        d = brain.from_words(things, said)
    except Exception as e:
        a.failed = str(e)
        a.log.append(f"robot   could not reach the model: {e}")
        return a
    _remember(a, "from words alone", d)
    a.picked = d.id
    a.invalid = d.id not in by_id(things)
    a.log.append(
        f"robot   picks {d.id}"
        + (f" - {d.why}" if d.why else "")
        + (" (not on the shelf!)" if a.invalid else "")
    )
    return a


def point_and_ask(
    target: Thing,
    things: list[Thing],
    person: Person,
    brain: Brain,
    cone: Cone,
    budget: int = POINT_AND_ASK_BUDGET,
    max_asks: int = MAX_ASKS,
) -> Attempt:
    """Round 3: the same point as round 1, two words, and the robot may ask by pointing."""
    said = person.say(budget)
    a = Attempt(
        round="point + ask",
        target=target.id,
        picked="",
        words=len(said.split()),
        asks=0,
        said=[said],
        cone=_cone_pairs(cone),
        ambiguous=cone.ambiguous,
    )
    index = by_id(things)
    inside = ", ".join(f"{i} {w:.2f}" for i, w in a.cone) or "nothing in the cone"
    a.log = [
        f"person  points at {target.id} (the same point as round 1)",
        f'        and says "{said}" ({a.words} of {budget} words)',
        f"cone    {inside}",
    ]
    asked: list[tuple[str, bool]] = []
    # At most `max_asks` questions, then a commit. Not `max_asks + 1`: the header says "up
    # to N questions" and it has to be true, or the mean-asks column is measuring a cap we
    # do not have.
    for _ in range(max_asks):
        try:
            d: Decision = brain.step(things, cone, said, asked)
        except Exception as e:
            a.failed = str(e)
            a.log.append(f"robot   could not reach the model: {e}")
            return a
        _remember(a, "commit" if d.action == "choose" else f"question {a.asks + 1}", d)
        if d.id not in index:
            a.invalid = True
            a.picked = d.id
            a.log.append(f"robot   named {d.id}, which is not on the shelf")
            return a
        if d.action == "choose":
            a.picked = d.id
            a.log.append(f"robot   commits to {d.id}" + (f" - {d.why}" if d.why else ""))
            return a
        yes = person.answer(d.id)
        asked.append((d.id, yes))
        a.asks += 1
        a.log.append(
            f"robot   points at {d.id} and asks: this one?" + (f" ({d.why})" if d.why else "")
        )
        a.log.append(f"person  {'yes' if yes else 'no'}")
        if yes:
            a.picked = d.id
            return a
    # Out of questions. The *model* makes the final choice, with only `choose` on the table:
    # if the harness picked a fallback here, part of the score would be the harness's.
    try:
        d = brain.commit(things, cone, said, asked)
    except Exception as e:
        a.failed = str(e)
        a.log.append(f"robot   could not reach the model: {e}")
        return a
    _remember(a, "out of questions", d)
    a.picked = d.id
    a.invalid = d.id not in index
    a.log.append(
        f"robot   out of questions; commits to {d.id or 'nothing'}"
        + (f" - {d.why}" if d.why else "")
        + (" (not on the shelf!)" if a.invalid else "")
    )
    return a


def trial(
    target: Thing,
    things: list[Thing],
    brain: Brain,
    seed: int,
    sigma_deg: float = SIGMA_DEG,
    budget2: int = WORDS_ONLY_BUDGET,
    budget3: int = POINT_AND_ASK_BUDGET,
    max_asks: int = MAX_ASKS,
) -> list[Attempt]:
    """One target, three rounds. Rounds 1 and 3 share one drawn point (see the module doc)."""
    cone = Person(target, things, random.Random(seed), sigma_deg).point()
    unique = kind_counts(things)[target.kind] == 1
    out = [
        point_only(target, cone),
        words_only(
            target, things, Person(target, things, random.Random(seed), sigma_deg), brain, budget2
        ),
        point_and_ask(
            target,
            things,
            Person(target, things, random.Random(seed), sigma_deg),
            brain,
            cone,
            budget3,
            max_asks,
        ),
    ]
    for a in out:
        a.unique = unique
    return out


@dataclass
class Tally:
    """The scoreboard for one round."""

    round: str
    n: int = 0
    correct: int = 0
    words: int = 0
    asks: int = 0
    invalid: int = 0
    failed: int = 0
    ambiguous: int = 0

    def add(self, a: Attempt) -> None:
        self.n += 1
        self.correct += int(a.ok)
        self.words += a.words
        self.asks += a.asks
        self.invalid += int(a.invalid)
        self.failed += int(bool(a.failed))
        self.ambiguous += int(a.ambiguous)

    @property
    def scored(self) -> int:
        """Attempts that actually got an answer out of the brain."""
        return self.n - self.failed

    @property
    def pct(self) -> float:
        return 100.0 * self.correct / self.scored if self.scored else 0.0

    @property
    def mean_words(self) -> float:
        return self.words / self.n if self.n else 0.0

    @property
    def mean_asks(self) -> float:
        return self.asks / self.n if self.n else 0.0


def tally(attempts: list[Attempt]) -> list[Tally]:
    """Group attempts by round, keeping the order they were run in."""
    out: dict[str, Tally] = {}
    for a in attempts:
        out.setdefault(a.round, Tally(a.round)).add(a)
    return list(out.values())
