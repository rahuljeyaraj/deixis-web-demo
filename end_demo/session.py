"""One command, start to finish: the loop that runs between the person and the model.

Two tasks live here, and they are scored differently on purpose.

**Selection** - *which thing do they mean?* One attempt, right or wrong, directly comparable
to the published 51-69% for pointing-only selection. This is what the three rounds measure
and what the booth measures. Scoring is strict in the way the harness's was: an id that is
not on the shelf scores zero and is recorded, never quietly retried, because a model that
invents a thing has not selected one.

**The job** - *hand me that one.* Work out which thing, pick it up, pass it across if the
arm that can reach it cannot reach the person, put it in their hand. Here a failed action is
told to the model and it recovers, because recovering is the thing being shown. This is the
showpiece, and it is deliberately **not** part of the measurement.

## The two rules this loop exists to enforce

**R4, the hand outranks the word.** A point at the thing the gripper is over is a yes. A
point at something else is not a rejection to re-ask - it is the new answer, and it is
handed to the model as one.

**R5, never act on a contradiction.** When the word and the hand disagree, the loop does not
pick a winner and it does not pick either. It tells the model that it has no answer, counts
the contradiction, and the model asks again. This is the single most important thing in the
file: the research says yes/no can be unreliable, so a loop that broke the tie itself would
be building the demo's accuracy out of the exact answer that cannot be trusted.

## What the loop does not decide

When the questions run out, the **model** makes the final choice, with only `choose` on the
table. That is not a style preference: in the three-rounds harness the loop used to break
that tie itself, which meant part of every score was the harness's judgement rather than the
model's. Fixing it cost two points off the headline number, and the number was wrong before.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from end_demo import say
from end_demo.brain import BrainUnavailable, Call, Decider, Session, system_for
from end_demo.confirm import Answer, Counts, read, read_choice
from end_demo.point import SIGMA_DEG, Cone
from end_demo.scene import Thing, by_id, is_unique, layout
from end_demo.tools import QUESTIONS, Toolset
from end_demo.world import World

#: Round 2's budget: enough for "small bottle", not enough to describe a shelf position.
WORDS_ONLY_BUDGET = 4
#: Round 3's budget: what someone with severe aphasia gets out at once.
POINT_AND_ASK_BUDGET = 2
#: Questions the robot may ask before it has to commit.
MAX_QUESTIONS = 3
#: A hard stop on loop passes, so a model that keeps naming things that are not there
#: cannot spin forever. Hitting it is recorded on the attempt, never swallowed.
MAX_STEPS = 12

Reasoner = Callable[[str], None] | None


@dataclass
class Attempt:
    """One go at one round, and everything that is counted about it."""

    round: str
    target: str
    picked: str = ""
    words: int = 0
    questions: int = 0
    said: list[str] = field(default_factory=list)
    cone: list[tuple[str, float]] = field(default_factory=list)
    ambiguous: bool = False
    #: Was the thing they wanted one of a kind, or one of several identical ones? The split
    #: that carries the argument: words can only ever work on the one-of-a-kind things.
    unique: bool = True
    invalid: bool = False  #: the model named something that is not on the shelf
    failed: str = ""  #: the model could not be reached at all
    #: R5. Reported, never hidden.
    contradictions: int = 0
    unclear: int = 0
    silences: int = 0
    #: Yes/no answers the person gave wrongly (the unreliable-responder mode).
    misled: int = 0
    gave_up: bool = False  #: the model said it could not
    out_of_steps: bool = False
    log: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    shelf: list[tuple[str, int, int]] = field(default_factory=list)
    tools: str = ""
    #: For the job: did the thing end up in the person's hand?
    delivered: bool = False
    actions: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.picked == self.target and not self.invalid and not self.failed

    @property
    def model_calls(self) -> int:
        return len(self.calls)

    @property
    def model_s(self) -> float:
        return sum(float(c.get("latency_s", 0.0)) for c in self.calls)

    @property
    def worst_call_s(self) -> float:
        return max((float(c.get("latency_s", 0.0)) for c in self.calls), default=0.0)


def cone_pairs(cone: Cone) -> list[tuple[str, float]]:
    return [(c.id, round(c.weight, 3)) for c in cone.candidates]


def observe(
    model: Decider, things: list[Thing], cone: Cone | None, said: str, asked: list[tuple[str, bool]]
) -> None:
    """The side door the stub needs because it cannot read (see `stub.py`)."""
    hook = getattr(model, "observe", None)
    if callable(hook):
        hook(things, cone, said, asked)


# ---------------------------------------------------------------- round 1: pointing alone


def point_only(target: Thing, things: list[Thing], cone: Cone) -> Attempt:
    """The point, and nothing else. No model is asked - there is nothing to think about.

    This is how published pointing systems work, and they report 51-69%. Our landing inside
    that band is the check that the rig is fair rather than tuned to fail, so this function
    must stay exactly this dumb.
    """
    a = Attempt(
        round="point only",
        target=target.id,
        picked=cone.best or "",
        cone=cone_pairs(cone),
        ambiguous=cone.ambiguous,
        unique=is_unique(things, target.id),
        shelf=layout(things),
        tools="none",
    )
    inside = ", ".join(f"{i} {w:.2f}" for i, w in a.cone) or "nothing in the cone"
    a.log = [
        f"person  points at {target.id} (and misses a little)",
        f"cone    {inside}",
        f"robot   takes the best one: {a.picked or 'nothing'}",
    ]
    if cone.ambiguous and len(cone.candidates) > 1:
        a.log.append("        (the core would have said: too close to call, ask)")
    return a


# ---------------------------------------------------------------- round 2: words alone


def words_only(
    target: Thing,
    things: list[Thing],
    person: Any,
    model: Decider,
    budget: int = WORDS_ONLY_BUDGET,
    system: str | None = None,
    on_reasoning: Reasoner = None,
) -> Attempt:
    """No pointing at all. The person's few words against the whole shelf.

    This is how every voice assistant works. On four identical bottles there is nothing to
    work with, and that is the result, not a failure of the model.
    """
    said = person.say(budget)
    a = Attempt(
        round="words only",
        target=target.id,
        words=len(said.split()),
        said=[said],
        unique=is_unique(things, target.id),
        shelf=layout(things),
        tools="choose",
    )
    a.log = [f'person  says "{said}" ({a.words} of {budget} words), and cannot point']
    world = World(things=list(things))
    kit = Toolset.commit_only()
    session = Session(model, system=system or system_for(kit.names()))
    session.tell(
        world.describe(said=said)
        + "\n\nThe person wants one thing off the shelf. They did not point - you have only "
        f'their words: "{said}". Choose one. You cannot ask, and there is nothing else to go on.'
    )
    observe(model, list(things), None, said, [])
    try:
        call = session.decide(kit.as_list(), on_reasoning)
    except BrainUnavailable as e:
        a.failed = str(e)
        a.log.append(f"robot   could not reach the model: {e}")
        return a
    a.calls = session.records()
    a.picked = str(call.args.get("id", ""))
    a.invalid = a.picked not in by_id(things)
    a.log.append(
        f"robot   picks {a.picked or 'nothing'}"
        + (f" - {call.why}" if call.why else "")
        + (" (not on the shelf!)" if a.invalid else "")
    )
    return a


# ---------------------------------------------------------------- round 3: point, words, ask


def point_and_ask(
    target: Thing,
    things: list[Thing],
    person: Any,
    model: Decider,
    cone: Cone,
    budget: int = POINT_AND_ASK_BUDGET,
    max_questions: int = MAX_QUESTIONS,
    toolset: Toolset | None = None,
    system: str | None = None,
    on_reasoning: Reasoner = None,
) -> Attempt:
    """The same point as round 1, two words, and the robot may ask - by pointing.

    The loop keeps one session, so the model's own earlier questions and the answers to
    them are in front of it rather than summarised for it.
    """
    kit = toolset or Toolset.selection()
    said = person.say(budget)
    a = Attempt(
        round="point + ask",
        target=target.id,
        words=len(said.split()),
        said=[said],
        cone=cone_pairs(cone),
        ambiguous=cone.ambiguous,
        unique=is_unique(things, target.id),
        shelf=layout(things),
        tools=kit.label,
    )
    index = by_id(things)
    inside = ", ".join(f"{i} {w:.2f}" for i, w in a.cone) or "nothing in the cone"
    a.log = [
        f"person  points at {target.id} (the same point as round 1)",
        f'        and says "{said}" ({a.words} of {budget} words)',
        f"cone    {inside}",
    ]
    world = World(things=list(things))
    session = Session(model, system=system or system_for(kit.names()))
    session.tell(
        world.describe(cone, said)
        + "\n\nWhich thing do they mean? Commit when you are sure. Ask when you are not - "
        "asking costs them no words."
    )
    counts = Counts()
    asked: list[tuple[str, bool]] = []
    steps = 0

    while a.questions < max_questions and steps < MAX_STEPS:
        steps += 1
        observe(model, list(things), cone, said, asked)
        try:
            call = session.decide(kit.as_list(), on_reasoning)
        except BrainUnavailable as e:
            a.failed = str(e)
            a.log.append(f"robot   could not reach the model: {e}")
            a.calls = session.records()
            return a
        a.calls = session.records()

        if call.tool == "choose":
            a.picked = str(call.args.get("id", ""))
            a.invalid = a.picked not in index
            a.log.append(
                f"robot   commits to {a.picked or 'nothing'}"
                + (f" - {call.why}" if call.why else "")
                + (" (not on the shelf!)" if a.invalid else "")
            )
            return finish_attempt(a, counts, person)

        if call.tool == "cannot":
            a.gave_up = True
            a.log.append(f"robot   says it cannot: {call.args.get('why', '')}")
            return finish_attempt(a, counts, person)

        if call.tool not in QUESTIONS:
            # Nothing else is on the table in a selection round. Tell the model rather than
            # guessing what it meant.
            session.result(call, not_offered(call, kit))
            a.log.append(f"robot   asked for {call.tool}, which it does not have here")
            continue

        bad = bad_ids(call, index)
        if bad:
            # Strict, as the harness was: naming a thing that is not on the shelf is not a
            # selection, and retrying it quietly would be scoring our patience, not the model.
            a.invalid = True
            a.picked = bad[0]
            a.log.append(f"robot   named {bad[0]}, which is not on the shelf")
            return finish_attempt(a, counts, person)

        wrong = malformed(call)
        if wrong:
            # A call with no thing in it, or a choice between a thing and itself. This is a
            # malformed call rather than a wrong answer, so the model is told and gets to
            # try again - but it costs a step, so it cannot do this forever.
            session.result(call, wrong)
            a.log.append(f"robot   tried to ask about nothing: {wrong}")
            continue

        a.questions += 1
        counts.questions += 1
        answer, note = ask_person(call, person, a, counts)
        a.log += note
        if answer.kind == "yes":
            # The person confirmed it, by hand or by word. That is arrival.
            a.picked = answer.id or asked_about(call)[0]
            return finish_attempt(a, counts, person)
        if answer.kind == "no":
            asked.append((asked_about(call)[0], False))
        session.result(call, result_text(call, answer, world))

    if steps >= MAX_STEPS:
        a.out_of_steps = True
        a.log.append(f"robot   gave up after {steps} steps")

    # Out of questions. The model makes the final choice - see the module docstring.
    observe(model, list(things), cone, said, asked)
    session.tell("You have no questions left. Choose the one you think they mean.")
    try:
        call = session.decide(Toolset.commit_only().as_list(), on_reasoning)
    except BrainUnavailable as e:
        a.failed = str(e)
        a.log.append(f"robot   could not reach the model: {e}")
        a.calls = session.records()
        return a
    a.calls = session.records()
    a.picked = str(call.args.get("id", ""))
    a.invalid = a.picked not in index
    a.log.append(
        f"robot   out of questions; commits to {a.picked or 'nothing'}"
        + (f" - {call.why}" if call.why else "")
        + (" (not on the shelf!)" if a.invalid else "")
    )
    return finish_attempt(a, counts, person)


def asked_about(call: Call) -> list[str]:
    """The ids a question was about, in the order the arm hovers over them."""
    if call.tool == "ask_this_one":
        return [str(call.args.get("id", ""))]
    if call.tool == "ask_choice":
        return [str(call.args.get("first", "")), str(call.args.get("second", ""))]
    return []


def bad_ids(call: Call, index: dict[str, Thing]) -> list[str]:
    return [i for i in asked_about(call) if i and i not in index]


def malformed(call: Call) -> str:
    """Why this question cannot be asked at all, or "" if it can.

    A question with no thing under the gripper is not a question, it is the arm waving. The
    first live run produced one, because the prompt had advertised `ask_choice` while the
    toolset did not offer it, so the model called it with no arguments at all - which is
    also why `system_for` now builds the prompt from the toolset.
    """
    if call.tool == "ask_this_one":
        if not str(call.args.get("id", "")).strip():
            return "You did not say which thing to point at. Name one from the shelf."
        return ""
    if call.tool == "ask_choice":
        first, second = (str(call.args.get(k, "")).strip() for k in ("first", "second"))
        if not first or not second:
            return "A choice needs two things. Name both, from the shelf."
        if first == second:
            return f"{first} twice is not a choice. Name two different things."
    return ""


def not_offered(call: Call, kit: Toolset) -> str:
    """What the model is told when it reaches for a tool it does not have."""
    return (
        f"I do not have a tool called {call.tool!r} right now. I have: "
        + ", ".join(kit.names())
        + "."
    )


def ask_person(call: Call, person: Any, a: Attempt, counts: Counts) -> tuple[Answer, list[str]]:
    """Put one question to the person and read the answer, hand first (R4, R5)."""
    ids = asked_about(call)
    arm = str(call.args.get("arm", "") or "right_arm")
    log: list[str] = []

    if call.tool == "ask_this_one":
        utterance = say.this_one(ids[0], arm)
        log.append(
            f'robot   hovers over {ids[0]}: "{utterance.spoken}"'
            + (f" ({call.why})" if call.why else "")
        )
        reply = person.to_this_one(ids[0])
        answer = read(reply, ids[0])
    elif call.tool == "ask_choice":
        utterance = say.choice(ids[0], ids[1], arm)
        log.append(
            f'robot   points at {ids[0]}, then {ids[1]}: "{utterance.spoken}"'
            + (f" ({call.why})" if call.why else "")
        )
        reply = person.to_choice(ids[0], ids[1])
        answer = read_choice(reply, ids[0], ids[1])
    else:  # ask_show_me
        utterance = say.show_me(arm)
        log.append(
            f'robot   opens a hand: "{utterance.spoken}"' + (f" ({call.why})" if call.why else "")
        )
        reply = person.to_show_me()
        # "Show me" has no thing in mind, so there is nothing for the hand to agree with:
        # whatever they point at is the answer.
        answer = read_choice(reply, "", "")

    counts.add(answer)
    if answer.kind == "contradiction":
        a.contradictions += 1
    elif answer.kind == "unclear":
        a.unclear += 1
    elif answer.kind == "nothing":
        a.silences += 1
    log.append("person  " + answer_line(answer))
    return answer, log


def answer_line(answer: Answer) -> str:
    said = f'"{answer.said}"' if answer.said else "(no words)"
    if answer.kind == "yes":
        return f"yes - {answer.why}"
    if answer.kind == "no":
        return f"no - {answer.why}"
    if answer.kind == "new_target":
        return f"points at {answer.id} instead ({answer.why})"
    if answer.kind == "contradiction":
        return f"{said} but pointed at {answer.id} - CONTRADICTION, not acted on"
    if answer.kind == "unclear":
        return f"points, but {answer.why}"
    if answer.kind == "nothing":
        return "says and does nothing"
    return f"says {said}"


def result_text(call: Call, answer: Answer, world: World) -> str:
    """What the model is told came of its question. The wording carries R4 and R5."""
    ids = asked_about(call)
    if answer.kind == "no":
        return f"They said no. {ids[0]} is not the one."
    if answer.kind == "new_target":
        seen = world.index[answer.id].seen if answer.id in world.index else answer.id
        if call.tool == "ask_show_me":
            return f"They showed you: {answer.id} ({seen})."
        return (
            f"They did not answer - they pointed at {answer.id} ({seen}) instead. That is "
            "their answer, not a refusal: it is the one they mean."
        )
    if answer.kind == "contradiction":
        return (
            f"No answer: what they said and where they pointed disagree ({answer.why}). "
            "Do not act on this. Ask again, differently - a choice between two things is "
            "answerable by pointing alone."
        )
    if answer.kind == "unclear":
        return f"They pointed, but it settles nothing: {answer.why}. Ask again, differently."
    if answer.kind == "nothing":
        return (
            "They did not answer. Give them time, then ask something easier to answer - a "
            "choice between two things, or ask them to show you. Do not guess."
        )
    if answer.kind == "said":
        return f'They said "{answer.said}", which is not a yes or a no. It may be misheard.'
    # A yes has to say yes *to what*. "They said yes" on its own sent the loop round again
    # on the first dry run, because nothing in it named the thing that had just been settled.
    settled = answer.id or (ids[0] if ids else "")
    seen = world.index[settled].seen if settled in world.index else settled
    how = "they pointed at it" if answer.word is None else "yes"
    return f"They confirmed it ({how}): {settled} ({seen}) is the one they mean."


def finish_attempt(a: Attempt, counts: Counts, person: Any) -> Attempt:
    a.misled = getattr(person, "misled", 0)
    a.words = getattr(person, "words_used", a.words)
    return a


# ---------------------------------------------------------------- one trial, three rounds


def trial(
    target: Thing,
    things: list[Thing],
    model: Decider,
    seed: int,
    make_person: Callable[..., Any],
    sigma_deg: float = SIGMA_DEG,
    budget2: int = WORDS_ONLY_BUDGET,
    budget3: int = POINT_AND_ASK_BUDGET,
    max_questions: int = MAX_QUESTIONS,
    toolset: Toolset | None = None,
    system: str | None = None,
) -> list[Attempt]:
    """One target, three rounds.

    Rounds 1 and 3 are handed **the same drawn point, with the same wobble**. So round 3
    cannot win on a luckier aim: the only things it has that round 1 does not are two words
    and the right to ask, which makes the gap between the two numbers attributable rather
    than suggestive.
    """
    cone = make_person(target, things, random.Random(seed), sigma_deg).point()
    return [
        point_only(target, things, cone),
        words_only(
            target,
            things,
            make_person(target, things, random.Random(seed), sigma_deg),
            model,
            budget2,
            system,
        ),
        point_and_ask(
            target,
            things,
            make_person(target, things, random.Random(seed), sigma_deg),
            model,
            cone,
            budget3,
            max_questions,
            toolset,
            system,
        ),
    ]
