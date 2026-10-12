"""The whole job, and the two-arm story it exists to show.

The test that earns the second arm is `test_the_far_arm_recovers_by_passing_it_across`: the
model asks for a hand-over the far arm cannot do, is told so in words, and works out the
pass by itself. Nothing is corrected behind its back, which is the difference between a
demo that recovers and a demo that is being steered.
"""

from __future__ import annotations

import random

from end_demo.confirm import Reply
from end_demo.job import job
from end_demo.person import Person
from end_demo.point import angles, weigh
from end_demo.scene import arms, by_id, shelf, wanted
from end_demo.stub import ScriptDecider
from end_demo.tools import Toolset
from end_demo.world import World

THINGS = shelf()
INDEX = by_id(THINGS)

# Read out of the scene, never written down: the arms' reach is computed from a measured
# ring, so naming a side here would mean testing the name instead of the geometry.
GIVER = next(a for a in arms() if a.reaches_person)
FAR = next(a for a in arms() if not a.reaches_person)
#: Something only the far arm can pick up, so it has to be passed across to be handed over.
PASSED = next(t for t in wanted(THINGS) if FAR.can_reach(t) and not GIVER.can_reach(t))
#: Something the near arm can pick up and hand straight over.
NEAR = next(t for t in wanted(THINGS) if GIVER.can_reach(t))
#: Another thing the near arm can reach, for "it handed over the wrong one".
OTHER_NEAR = next(t for t in wanted(THINGS) if GIVER.can_reach(t) and t.id != NEAR.id)


def straight_at(thing_id: str):
    return weigh(angles(INDEX[thing_id].pos), THINGS)


def world() -> World:
    return World(things=shelf())


def person(target: str, **kw) -> Person:
    return Person(INDEX[target], THINGS, random.Random(2), sigma_deg=0.0, **kw)


def script(*steps) -> ScriptDecider:
    return ScriptDecider(script=list(steps))


# ---------------------------------------------------------------- the easy path


def test_the_near_arm_just_hands_it_over():
    model = script(("pick", {"id": NEAR.id, "arm": GIVER.id}), ("hand_over", {}))
    a = job(NEAR, world(), model, person(NEAR.id), straight_at(NEAR.id))
    assert a.delivered
    assert a.picked == NEAR.id


def test_the_job_is_not_a_measurement():
    """It is recorded as its own round so nothing can quietly fold it into a score."""
    model = script(("pick", {"id": NEAR.id}), ("hand_over", {}))
    a = job(NEAR, world(), model, person(NEAR.id), straight_at(NEAR.id))
    assert a.round == "the job"


# ---------------------------------------------------------------- the two-arm story


def test_the_far_arm_recovers_by_passing_it_across():
    """The model tries the hand-over, is told the far arm cannot reach the person, and
    works out the pass itself. This is the whole argument for a second arm."""
    model = script(
        ("pick", {"id": PASSED.id, "arm": FAR.id}),
        ("hand_over", {"arm": FAR.id}),  # refused, in words
        ("pass_to", {"arm": GIVER.id}),
        ("hand_over", {"arm": GIVER.id}),
    )
    w = world()
    a = job(PASSED, w, model, person(PASSED.id), straight_at(PASSED.id))
    assert a.delivered
    assert w.with_person == [PASSED.id]
    refused = [line for line in a.log if "could not" in line]
    assert any("cannot reach the person" in line for line in refused)
    assert any(f"Pass {PASSED.id} to {GIVER.id} first" in line for line in refused)


def test_the_refusal_reaches_the_model_as_the_tool_result():
    model = script(
        ("pick", {"id": PASSED.id, "arm": FAR.id}),
        ("hand_over", {"arm": FAR.id}),
        ("pass_to", {"arm": GIVER.id}),
        ("hand_over", {"arm": GIVER.id}),
    )
    job(PASSED, world(), model, person(PASSED.id), straight_at(PASSED.id))
    told = " ".join(str(m.get("content", "")) for m in model.seen[2])
    assert "cannot reach the person" in told


def test_the_wrong_arm_for_the_shelf_is_told_which_arm_can():
    model = script(
        ("pick", {"id": PASSED.id, "arm": GIVER.id}),  # out of reach
        ("pick", {"id": PASSED.id, "arm": FAR.id}),
        ("pass_to", {"arm": GIVER.id}),
        ("hand_over", {}),
    )
    a = job(PASSED, world(), model, person(PASSED.id), straight_at(PASSED.id))
    assert a.delivered
    assert any(f"{FAR.id} can reach it" in line for line in a.log)


# ---------------------------------------------------------------- asking, during the job


def test_it_can_ask_before_it_picks():
    model = script(
        ("ask_this_one", {"id": PASSED.id}),
        ("pick", {"id": PASSED.id, "arm": FAR.id}),
        ("pass_to", {"arm": GIVER.id}),
        ("hand_over", {}),
    )
    a = job(PASSED, world(), model, person(PASSED.id, style="word"), straight_at(PASSED.id))
    assert a.questions == 1
    assert a.delivered


def test_a_contradiction_during_the_job_is_counted_and_not_acted_on():
    class Contrary(Person):
        def to_this_one(self, thing_id: str) -> Reply:
            return Reply(point=straight_at("bottle_4"), said="yes")

    model = script(
        ("ask_this_one", {"id": "bottle_2"}),
        ("ask_choice", {"first": "bottle_2", "second": "bottle_4"}),
        ("cannot", {"why": "I cannot tell which one"}),
    )
    p = Contrary(INDEX["bottle_4"], THINGS, random.Random(1), sigma_deg=0.0)
    a = job(INDEX["bottle_4"], world(), model, p, straight_at("bottle_2"))
    assert a.contradictions == 1
    assert not a.delivered  # nothing was handed over on the strength of a contradiction


def test_the_question_cap_takes_the_questions_away_rather_than_refusing_them():
    """A model that is refused goes on trying, which burns the step budget and reads like a
    robot arguing with its harness. So at the cap the question tools come off the table."""
    model = script(
        ("ask_this_one", {"id": "bottle_1"}),
        ("ask_this_one", {"id": "bottle_2"}),
        ("cannot", {"why": "I cannot tell which one"}),
    )
    a = job(
        INDEX["bottle_3"],
        world(),
        model,
        person("bottle_3", style="word"),
        straight_at("bottle_3"),
        max_questions=2,
    )
    assert a.questions == 2
    assert a.gave_up
    offered = [t["function"]["name"] for t in Toolset.job(can_ask=False).as_list()]
    assert "ask_this_one" not in offered
    assert "cannot" in offered
    told = " ".join(str(m.get("content", "")) for m in model.seen[-1])
    assert "no questions left" in told


# ---------------------------------------------------------------- stopping


def test_a_wrong_hand_over_is_recorded_as_not_delivered():
    """Honest scoring: the thing got into their hand, but it was the wrong thing."""
    model = script(("pick", {"id": OTHER_NEAR.id}), ("hand_over", {}))
    a = job(NEAR, world(), model, person(NEAR.id), straight_at(NEAR.id))
    assert a.picked == OTHER_NEAR.id
    assert not a.delivered
    assert any(f"but they wanted {NEAR.id}" in line for line in a.log)


def test_the_robot_can_say_it_cannot():
    model = script(("cannot", {"why": "nothing is in their cone"}))
    a = job(NEAR, world(), model, person(NEAR.id), straight_at(NEAR.id))
    assert a.gave_up and not a.delivered


def test_a_model_that_never_stops_is_stopped():
    model = ScriptDecider(script=[("look", {})] * 50)
    a = job(NEAR, world(), model, person(NEAR.id), straight_at(NEAR.id), max_steps=5)
    assert a.out_of_steps
    assert a.model_calls == 5


def test_an_unknown_tool_is_reported_to_the_model_not_guessed_at():
    model = ScriptDecider(script=[("done", {})])
    # `done` is in the job toolset, so this checks the normal stop rather than the unknown
    # path; the unknown path is driven through `_do` directly.
    from end_demo.brain import Call
    from end_demo.job import _do

    out = _do(Call("teleport", {}), world())
    assert not out.ok
    assert "do not have a tool called 'teleport'" in out.text
    assert "hand_over" in out.text
    a = job(NEAR, world(), model, person(NEAR.id), straight_at(NEAR.id))
    assert not a.delivered and not a.out_of_steps


def test_looking_again_shows_what_has_changed():
    model = script(("pick", {"id": NEAR.id}), ("look", {}), ("hand_over", {}))
    a = job(NEAR, world(), model, person(NEAR.id), straight_at(NEAR.id))
    assert a.delivered
    told = " ".join(str(m.get("content", "")) for m in model.seen[2])
    assert f"holding {NEAR.id}" in told


def test_the_job_toolset_is_the_whole_body():
    names = Toolset.job().names()
    for needed in ("look", "pick", "pass_to", "place", "hand_over", "cannot", "done"):
        assert needed in names
    for asking in ("ask_this_one", "ask_choice", "ask_show_me"):
        assert asking in names
    # `choose` is not here: in the job, committing means picking the thing up.
    assert "choose" not in names
