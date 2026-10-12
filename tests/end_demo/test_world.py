"""The body: what the arms can and cannot do, and what the model is told when they can't.

The point of these tests is the *wording* as much as the state. A refusal the model cannot
act on is a dead end, so each failure has to say what went wrong in terms the model can do
something about - which arm, what it can reach instead, what to try next.

**Which arm is which is read out of the scene, never written down here.** The arms' reach is
computed from a measured ring, so moving a base by a centimetre changes who can do what. A
test that named a side would, after that, be testing the name.
"""

from __future__ import annotations

from end_demo.point import angles, weigh
from end_demo.scene import arms, by_id, shelf, wanted
from end_demo.world import World

#: The arm beside the person: the only one that can put a thing in their hand.
GIVER = next(a for a in arms() if a.reaches_person)
#: The arm across the table: it reaches more of the shelf and none of the person.
FAR = next(a for a in arms() if not a.reaches_person)
#: Something only the far arm can pick up, so it has to be passed across.
NEEDS_A_PASS = next(t for t in wanted(shelf()) if FAR.can_reach(t) and not GIVER.can_reach(t))
#: Something the near arm can pick up and hand straight over.
STRAIGHT_OVER = next(t for t in wanted(shelf()) if GIVER.can_reach(t))
#: Something either arm can reach, so "which arm" is a real choice.
EITHER = next(t for t in wanted(shelf()) if GIVER.can_reach(t) and FAR.can_reach(t))


def fresh() -> World:
    return World(things=shelf())


def straight_at(thing_id: str):
    things = shelf()
    return weigh(angles(by_id(things)[thing_id].pos), things)


# ---------------------------------------------------------------- looking


def test_the_shelf_description_says_what_is_identical():
    text = fresh().describe()
    assert "4 bottles and 2 mugs are identical" in text
    assert "No word tells them apart" in text


def test_the_description_never_hints_at_which_one_is_meant():
    """Two bottles must be indistinguishable in the robot's own report, or the model could
    separate them without the person."""
    lines = [ln for ln in fresh().describe().splitlines() if "bottle_" in ln]
    rest = [ln.split(maxsplit=1)[1] for ln in lines]
    assert all("a small white bottle" in r for r in rest)


def test_the_cone_numbers_are_in_the_description():
    """The resolver's actual weights, in the robot's own report. Note that a dead-straight
    point still shares its cone with the neighbours - the thing itself keeps well under all
    of it, which is the whole reason the robot has to ask."""
    world = fresh()
    cone = straight_at("bottle_2")
    text = world.describe(cone)
    best = max(cone.candidates, key=lambda c: c.weight)
    assert best.id == "bottle_2"
    assert f"{best.weight:.2f}" in text
    assert "degrees per axis" in text


def test_the_description_says_which_arm_cannot_reach_the_person():
    text = fresh().describe()
    assert "CANNOT reach the person" in text
    assert "can put things in the person's hand" in text
    # And what each arm can pick up from, read out of the measured ring rather than written
    # down - so the sentence can never promise something the arm cannot do.
    assert "picks up from" in text
    assert "from the left on the front row" in text


def test_what_was_heard_is_passed_on_with_a_warning_that_it_mishears():
    text = fresh().describe(said="that one")
    assert 'heard: "that one"' in text
    assert "mishears" in text


# ---------------------------------------------------------------- picking


def test_picking_something_in_reach():
    world = fresh()
    out = world.pick(STRAIGHT_OVER.id, GIVER.id)
    assert out.ok
    assert world.held[GIVER.id] == STRAIGHT_OVER.id
    assert STRAIGHT_OVER.id not in [t.id for t in world.on_shelf()]


def test_the_wrong_arm_is_told_what_it_can_reach_and_which_arm_can():
    world = fresh()
    out = world.pick(NEEDS_A_PASS.id, GIVER.id)
    assert not out.ok
    assert "cannot reach" in out.text
    assert f"{FAR.id} can reach it" in out.text
    assert "picks up from" in out.text


def test_leaving_the_arm_out_picks_one_that_can():
    """Choosing between two arms that can both do it is kinematics, not judgement, so the
    model is not made to care unless the person said which."""
    world = fresh()
    assert world.pick(NEEDS_A_PASS.id).ok
    assert world.held[FAR.id] == NEEDS_A_PASS.id


def test_a_thing_that_is_not_there():
    out = fresh().pick("teapot")
    assert not out.ok
    assert "nothing called 'teapot'" in out.text


def test_an_arm_that_is_not_there_is_told_the_names_of_the_arms():
    out = fresh().pick(STRAIGHT_OVER.id, "third_arm")
    assert not out.ok
    assert GIVER.id in out.text and FAR.id in out.text


def test_an_arm_already_holding_something():
    world = fresh()
    assert world.pick(NEEDS_A_PASS.id, FAR.id).ok
    other = next(t for t in wanted(shelf()) if FAR.can_reach(t) and t.id != NEEDS_A_PASS.id)
    out = world.pick(other.id, FAR.id)
    assert not out.ok
    assert f"already holding {NEEDS_A_PASS.id}" in out.text


def test_picking_the_same_thing_twice():
    world = fresh()
    world.pick(NEEDS_A_PASS.id)
    out = world.pick(NEEDS_A_PASS.id, GIVER.id)
    assert not out.ok
    assert "already off the shelf" in out.text


# ---------------------------------------------------------------- handing over


def test_the_far_arm_cannot_hand_over_and_is_told_to_pass_it_across():
    """The two-arm story, in one test. This is why two arms were worth the cost."""
    world = fresh()
    world.pick(NEEDS_A_PASS.id, FAR.id)
    out = world.hand_over(FAR.id)
    assert not out.ok
    assert "cannot reach the person" in out.text
    assert f"Pass {NEEDS_A_PASS.id} to {GIVER.id} first" in out.text


def test_passing_across_then_handing_over():
    world = fresh()
    assert world.pick(NEEDS_A_PASS.id, FAR.id).ok
    assert world.pass_to(GIVER.id).ok
    assert world.held == {GIVER.id: NEEDS_A_PASS.id}
    out = world.hand_over(GIVER.id)
    assert out.ok and out.finished
    assert world.with_person == [NEEDS_A_PASS.id]
    assert world.gave(NEEDS_A_PASS.id)


def test_handing_over_with_nothing_held():
    out = fresh().hand_over(GIVER.id)
    assert not out.ok
    assert "Neither arm is holding anything" in out.text


def test_handing_over_without_naming_an_arm_uses_the_one_beside_the_person():
    world = fresh()
    world.pick(STRAIGHT_OVER.id, GIVER.id)
    assert world.hand_over().ok
    assert world.with_person == [STRAIGHT_OVER.id]


def test_passing_to_an_arm_that_is_full():
    world = fresh()
    world.pick(NEEDS_A_PASS.id, FAR.id)
    world.pick(STRAIGHT_OVER.id, GIVER.id)
    out = world.pass_to(GIVER.id)
    assert not out.ok
    assert f"already holding {STRAIGHT_OVER.id}" in out.text


# ---------------------------------------------------------------- putting down


def test_the_tray_and_the_shelf_are_the_only_places():
    world = fresh()
    world.pick(STRAIGHT_OVER.id, GIVER.id)
    out = world.place("the floor")
    assert not out.ok
    assert "tray" in out.text and "shelf" in out.text


def test_putting_it_back_on_the_shelf_puts_it_back_on_the_shelf():
    world = fresh()
    world.pick(STRAIGHT_OVER.id, GIVER.id)
    assert world.place("shelf").ok
    assert STRAIGHT_OVER.id in [t.id for t in world.on_shelf()]
    assert not world.held


def test_putting_it_in_the_tray():
    world = fresh()
    world.pick(STRAIGHT_OVER.id, GIVER.id)
    assert world.place("tray").ok
    assert world.in_tray == [STRAIGHT_OVER.id]
    assert f"in the tray: {STRAIGHT_OVER.id}" in world.describe()


# ---------------------------------------------------------------- pointing


def test_an_arm_may_point_at_something_it_cannot_pick_up():
    """Asking has to be cheaper than reaching, so pointing is a lean and never fails for
    reach. A robot that could only ask about things it could already grasp would ask the
    wrong questions."""
    world = fresh()
    out = world.point_at(NEEDS_A_PASS.id, GIVER.id)
    assert out.ok
    assert not world.held  # pointing moves nothing
    assert NEEDS_A_PASS.id in [t.id for t in world.on_shelf()]


def test_pointing_at_nothing():
    assert not fresh().point_at("teapot", GIVER.id).ok


def test_either_arm_can_take_the_shared_middle():
    """The overlap is what makes "you, that one" a real instruction rather than a formality:
    for some things there genuinely is a choice of arm."""
    for arm in (GIVER, FAR):
        world = fresh()
        assert world.pick(EITHER.id, arm.id).ok


def test_every_failure_is_written_down():
    world = fresh()
    world.pick("teapot")
    world.hand_over()
    assert len(world.did) == 2
    assert all(line.startswith("could not") for line in world.did)
