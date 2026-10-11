"""The body: what the arms can and cannot do, and what the model is told when they can't.

The point of these tests is the *wording* as much as the state. A refusal the model cannot
act on is a dead end, so each failure has to say what went wrong in terms the model can do
something about - which arm, which columns, what to try instead.
"""

from __future__ import annotations

from end_demo.point import angles, weigh
from end_demo.scene import by_id, shelf
from end_demo.world import World


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
    text = fresh().describe()
    lines = [ln for ln in text.splitlines() if "bottle_" in ln]
    rest = [ln.split(maxsplit=1)[1] for ln in lines]
    # They differ only in where they are, never in what they look like.
    assert all("a small white bottle" in r for r in rest)


def test_the_cone_numbers_are_in_the_description():
    """The resolver's actual weights, in the robot's own report. Note that a dead-straight
    point still shares its cone with the neighbours - at 2.5 m a perfect aim leaves 0.72 on
    the thing and the rest next door, which is the whole reason the robot has to ask."""
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


def test_what_was_heard_is_passed_on_with_a_warning_that_it_mishears():
    text = fresh().describe(said="that one")
    assert 'heard: "that one"' in text
    assert "mishears" in text


# ---------------------------------------------------------------- picking


def test_picking_something_in_reach():
    world = fresh()
    out = world.pick("bottle_1", "left_arm")
    assert out.ok
    assert world.held["left_arm"] == "bottle_1"
    assert "bottle_1" not in [t.id for t in world.on_shelf()]


def test_the_wrong_arm_is_told_which_columns_it_has_and_which_arm_can():
    world = fresh()
    out = world.pick("bottle_1", "right_arm")
    assert not out.ok
    assert "cannot reach" in out.text
    assert "left_arm can reach it" in out.text


def test_leaving_the_arm_out_picks_one_that_can():
    """Choosing between two arms that can both do it is kinematics, not judgement, so the
    model is not made to care unless the person said which."""
    world = fresh()
    assert world.pick("bottle_1").ok
    assert world.held["left_arm"] == "bottle_1"


def test_a_thing_that_is_not_there():
    out = fresh().pick("teapot")
    assert not out.ok
    assert "nothing called 'teapot'" in out.text


def test_an_arm_that_is_not_there_is_told_the_names_of_the_arms():
    out = fresh().pick("bottle_1", "third_arm")
    assert not out.ok
    assert "left_arm" in out.text and "right_arm" in out.text


def test_an_arm_already_holding_something():
    world = fresh()
    world.pick("bottle_1", "left_arm")
    out = world.pick("bottle_2", "left_arm")
    assert not out.ok
    assert "already holding bottle_1" in out.text


def test_picking_the_same_thing_twice():
    world = fresh()
    world.pick("remote", "right_arm")
    out = world.pick("remote", "left_arm")
    assert not out.ok
    assert "already off the shelf" in out.text


# ---------------------------------------------------------------- handing over


def test_the_far_arm_cannot_hand_over_and_is_told_to_pass_it_across():
    """The two-arm story, in one test. This is why two arms were worth the cost."""
    world = fresh()
    world.pick("bottle_1", "left_arm")
    out = world.hand_over("left_arm")
    assert not out.ok
    assert "cannot reach the person" in out.text
    assert "Pass bottle_1 to right_arm first" in out.text


def test_passing_across_then_handing_over():
    world = fresh()
    assert world.pick("bottle_1", "left_arm").ok
    assert world.pass_to("right_arm").ok
    assert world.held == {"right_arm": "bottle_1"}
    out = world.hand_over("right_arm")
    assert out.ok and out.finished
    assert world.with_person == ["bottle_1"]
    assert world.gave("bottle_1")


def test_handing_over_with_nothing_held():
    out = fresh().hand_over("right_arm")
    assert not out.ok
    assert "Neither arm is holding anything" in out.text


def test_handing_over_without_naming_an_arm_uses_the_one_beside_the_person():
    world = fresh()
    world.pick("remote", "right_arm")
    assert world.hand_over().ok
    assert world.with_person == ["remote"]


def test_passing_to_an_arm_that_is_full():
    world = fresh()
    world.pick("bottle_1", "left_arm")
    world.pick("remote", "right_arm")
    out = world.pass_to("right_arm")
    assert not out.ok
    assert "already holding remote" in out.text


# ---------------------------------------------------------------- putting down


def test_the_tray_and_the_shelf_are_the_only_places():
    world = fresh()
    world.pick("remote", "right_arm")
    out = world.place("the floor")
    assert not out.ok
    assert "tray" in out.text and "shelf" in out.text


def test_putting_it_back_on_the_shelf_puts_it_back_on_the_shelf():
    world = fresh()
    world.pick("remote", "right_arm")
    assert world.place("shelf").ok
    assert "remote" in [t.id for t in world.on_shelf()]
    assert not world.held


def test_putting_it_in_the_tray():
    world = fresh()
    world.pick("remote", "right_arm")
    assert world.place("tray").ok
    assert world.in_tray == ["remote"]
    assert "in the tray: remote" in world.describe()


# ---------------------------------------------------------------- pointing


def test_an_arm_may_point_at_something_it_cannot_pick_up():
    """Asking has to be cheaper than reaching, so pointing is a lean and never fails for
    reach. A robot that could only ask about things it could already grasp would ask the
    wrong questions."""
    world = fresh()
    out = world.point_at("bottle_1", "right_arm")
    assert out.ok
    assert not world.held  # pointing moves nothing
    assert "bottle_1" in [t.id for t in world.on_shelf()]


def test_pointing_at_nothing():
    assert not fresh().point_at("teapot", "right_arm").ok


def test_every_failure_is_written_down():
    world = fresh()
    world.pick("teapot")
    world.hand_over()
    assert len(world.did) == 2
    assert all(line.startswith("could not") for line in world.did)
