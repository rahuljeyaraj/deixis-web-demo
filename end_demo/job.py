"""The whole job: *hand me that one.*

Surveys of wheelchair users put picking something up off a shelf first, and in one study of
89 users and 52 clinicians "being able to reach adequately for objects" was the single
biggest concern. So the task is not a tidy-up and not a tower: it is getting one thing off a
shelf and into somebody's hand, which is the thing people actually ask an assistive arm for
and which needs no noun anywhere in it.

This is the showpiece, and it is **not** part of the measurement. The numbers come from
selection (`session.py`), which is quick, repeatable and comparable to published work. What
this adds is the loop a judge can watch: the model works out which thing, finds that the arm
which can reach it cannot reach the person, passes it across, and hands it over.

**Failures come back as sentences.** A pick the arm cannot reach, a hand-over from the wrong
arm, an id that does not exist - each one is told to the model as the tool's result and the
model works out what to do. Nothing is corrected behind its back. That is the difference
between a demo that recovers and a demo that is being steered, and the steering version is
the bug that once made part of the selection score the harness's own judgement.
"""

from __future__ import annotations

from typing import Any

from end_demo.brain import BrainUnavailable, Call, Decider, Session, system_for
from end_demo.confirm import Counts
from end_demo.point import Cone
from end_demo.scene import is_unique, layout
from end_demo.session import (
    POINT_AND_ASK_BUDGET,
    Attempt,
    Reasoner,
    ask_person,
    cone_pairs,
    finish_attempt,
    malformed,
    not_offered,
    result_text,
)
from end_demo.tools import QUESTIONS, Toolset
from end_demo.world import Outcome, World

#: A hard stop. The job needs more passes than a selection does - ask, pick, pass, hand over
#: is four on its own - and a model that keeps trying something impossible must still stop.
MAX_JOB_STEPS = 20
#: Questions the robot may ask during the job. Higher than the selection cap because here a
#: question costs a nod and a wrong pick costs the person their thing back on the shelf.
MAX_JOB_QUESTIONS = 4


def job(
    target: Any,
    world: World,
    model: Decider,
    person: Any,
    cone: Cone | None = None,
    budget: int = POINT_AND_ASK_BUDGET,
    max_questions: int = MAX_JOB_QUESTIONS,
    max_steps: int = MAX_JOB_STEPS,
    system: str | None = None,
    on_reasoning: Reasoner = None,
) -> Attempt:
    """Point, two words, and the thing ends up in their hand."""
    said = person.say(budget)
    if cone is None:
        cone = person.point()
    a = Attempt(
        round="the job",
        target=target.id,
        words=len(said.split()),
        said=[said],
        cone=cone_pairs(cone),
        ambiguous=cone.ambiguous,
        unique=is_unique(world.things, target.id),
        shelf=layout(world.things),
        tools=Toolset.job().label,
    )
    inside = ", ".join(f"{i} {w:.2f}" for i, w in a.cone) or "nothing in the cone"
    a.log = [
        f"person  points at {target.id}",
        f'        and says "{said}" ({a.words} of {budget} words)',
        f"cone    {inside}",
    ]
    kit = Toolset.job()
    session = Session(model, system=system or system_for(kit.names()))
    session.tell(
        world.describe(cone, said)
        + "\n\nThey want one thing off the shelf, put into their hand. Work out which one, "
        "then get it there. Ask when you are not sure - asking costs them no words."
    )
    counts = Counts()
    steps = 0

    while steps < max_steps:
        steps += 1
        _observe(model, world, cone, said, counts)
        try:
            call = session.decide(kit.as_list(), on_reasoning)
        except BrainUnavailable as e:
            a.failed = str(e)
            a.log.append(f"robot   could not reach the model: {e}")
            a.calls = session.records()
            return a
        a.calls = session.records()

        if call.tool == "done":
            a.log.append("robot   says the job is done")
            break
        if call.tool == "cannot":
            a.gave_up = True
            a.log.append(f"robot   says it cannot: {call.args.get('why', '')}")
            break

        if call.tool in QUESTIONS:
            if call.tool not in kit.names():
                session.result(call, not_offered(call, kit))
                a.log.append(f"robot   asked for {call.tool}, which it does not have here")
                continue
            wrong = malformed(call)
            if wrong:
                session.result(call, wrong)
                a.log.append(f"robot   tried to ask about nothing: {wrong}")
                continue
            a.questions += 1
            counts.questions += 1
            answer, note = ask_person(call, person, a, counts)
            a.log += note
            session.result(call, result_text(call, answer, world))
            if a.questions >= max_questions:
                # Out of questions: take them away rather than refuse them one at a time.
                kit = Toolset.job(can_ask=False)
                session.tell(
                    "You have no questions left. Decide from what you have, or say you "
                    "cannot - do not guess."
                )
            continue

        outcome = _do(call, world)
        a.actions.append(f"{call.tool} {call.args} -> {'ok' if outcome.ok else 'no'}")
        a.log.append(("robot   " if outcome.ok else "robot   could not: ") + outcome.text)
        session.result(call, outcome.text)
        if outcome.finished:
            a.picked = world.with_person[-1] if world.with_person else ""
            a.delivered = a.picked == target.id
            a.log.append(
                f"        the person is holding {a.picked}"
                + ("" if a.delivered else f" - but they wanted {target.id}")
            )
            break
    else:
        a.out_of_steps = True
        a.log.append(f"robot   ran out of steps after {steps}")

    if not a.picked and world.with_person:
        a.picked = world.with_person[-1]
        a.delivered = a.picked == target.id
    return finish_attempt(a, counts, person)


def _do(call: Call, world: World) -> Outcome:
    """Carry out one action. Every refusal comes back as a sentence for the model to read."""
    args = call.args
    arm = str(args.get("arm", "") or "") or None
    if call.tool == "look":
        return Outcome(True, world.describe())
    if call.tool == "point_at":
        return world.point_at(str(args.get("id", "")), arm)
    if call.tool == "pick":
        return world.pick(str(args.get("id", "")), arm)
    if call.tool == "pass_to":
        return world.pass_to(arm)
    if call.tool == "place":
        return world.place(str(args.get("where", "")), arm)
    if call.tool == "hand_over":
        return world.hand_over(arm)
    return Outcome(
        False,
        f"I do not have a tool called {call.tool!r}. I have: " + ", ".join(Toolset.job().names()),
    )


def _observe(model: Decider, world: World, cone: Cone | None, said: str, counts: Counts) -> None:
    hook = getattr(model, "observe", None)
    if callable(hook):
        hook(world.on_shelf(), cone, said, [])


__all__ = ["MAX_JOB_QUESTIONS", "MAX_JOB_STEPS", "job"]
