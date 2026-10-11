"""The cone: that a point is vague, and vague in the way we say it is."""

from __future__ import annotations

import math
import random

from rounds.pointer import RATIO, Cone, aim, angles, point_at, ray, separation_deg, weigh
from rounds.scene import DIST, STEP_X, shelf


def test_angles_round_trip() -> None:
    for pos in [(0.0, 2.5, 0.0), (0.5, 2.5, -0.35), (-0.25, 2.5, 0.0)]:
        back = ray(*angles(pos))
        assert separation_deg(back, pos) < 1e-6


def test_neighbours_are_a_few_degrees_apart() -> None:
    """25 cm at 2.5 m is about 5.7 degrees - less than twice the pointing error, which is
    the whole reason one point cannot pick one thing."""
    gap = math.degrees(math.atan2(STEP_X, DIST))
    assert 5.0 < gap < 6.5


def test_a_perfect_point_still_names_the_neighbours() -> None:
    things = shelf()
    target = next(t for t in things if t.id == "bottle_2")
    cone = weigh(angles(target.pos), things)
    assert cone.best == "bottle_2"
    # Its neighbours are inside the cone too: a point is never a single thing.
    assert len(cone.candidates) > 1
    assert "bottle_1" in cone.ids() and "bottle_3" in cone.ids()


def test_a_perfect_point_is_not_ambiguous() -> None:
    things = shelf()
    target = next(t for t in things if t.id == "bottle_2")
    assert not weigh(angles(target.pos), things).ambiguous


def test_a_point_between_two_things_is_ambiguous() -> None:
    """Aimed exactly between two neighbours, the core's 2:1 rule must say: ask."""
    things = shelf()
    a = next(t for t in things if t.id == "bottle_1")
    b = next(t for t in things if t.id == "bottle_2")
    mid = (
        (a.pos[0] + b.pos[0]) / 2,
        (a.pos[1] + b.pos[1]) / 2,
        (a.pos[2] + b.pos[2]) / 2,
    )
    cone = weigh(angles(mid), things)
    assert cone.ambiguous
    top, second = cone.candidates[0], cone.candidates[1]
    assert top.weight < RATIO * second.weight


def test_weights_sum_to_one() -> None:
    things = shelf()
    cone = point_at(things[0], things, random.Random(7))
    assert math.isclose(sum(c.weight for c in cone.candidates), 1.0, rel_tol=1e-9)


def test_wobble_is_the_size_we_claim() -> None:
    """The error we add is the error we print in the header: about 3 degrees per axis."""
    things = shelf()
    target = things[4]
    rng = random.Random(3)
    truth = angles(target.pos)
    off = [math.degrees(aim(target, rng, 3.0)[0] - truth[0]) for _ in range(4000)]
    mean = sum(off) / len(off)
    sd = math.sqrt(sum((x - mean) ** 2 for x in off) / len(off))
    assert abs(mean) < 0.3
    assert 2.7 < sd < 3.3


def test_point_only_lands_in_the_published_range() -> None:
    """Pointing alone, over the whole shelf, must come out near the published 51-69%. If
    this test starts failing the scene or the error model has drifted, and every number in
    the report moves with it."""
    things = shelf()
    rng = random.Random(11)
    hits = 0
    runs = 600
    for i in range(runs):
        target = things[i % len(things)]
        cone: Cone = point_at(target, things, rng)
        hits += int(cone.best == target.id)
    assert 0.45 <= hits / runs <= 0.75
