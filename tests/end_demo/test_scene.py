"""The shelf has to be hard for an honest reason, and the arms have to need each other.

The geometry tests are the load-bearing ones. The plan put the shelf 2.5 m away so that
neighbours would be about 5.7 degrees apart, and the arm reaches 280 mm - so the scene had
to shrink. What must survive the shrinking is the **angle**, because the angle is what makes
one point cover several things and it is what every measured number is a number about.
"""

from __future__ import annotations

import math
import random

import pytest

from end_demo.point import SIGMA_DEG
from end_demo.scene import (
    COLS,
    GIVE_MM,
    PAN_LIMIT_DEG,
    REACH_RING_MM,
    ROWS,
    arms,
    by_id,
    column_x,
    from_eye,
    is_unique,
    kind_counts,
    layout,
    reshuffle,
    separation_deg,
    shelf,
    wanted,
)


def test_twelve_things_in_two_rows():
    things = shelf()
    assert len(things) == COLS * ROWS == 12
    assert sorted({t.row for t in things}) == [0, 1]
    for row in (0, 1):
        assert sorted(t.col for t in things if t.row == row) == list(range(COLS))


# ---------------------------------------------------------------- the angles


def test_neighbours_are_closer_together_than_a_person_can_point():
    """5.2 degrees across, against a 3 degree aim. This is the whole difficulty of the
    scene, and it is the number the plan's 2.5 m shelf was built to get (5.7 degrees)."""
    index = by_id(shelf())
    across = separation_deg(index["bottle_1"], index["bottle_2"])
    assert 4.5 < across < 6.5
    assert across < 2 * SIGMA_DEG  # inside twice the aim error: one point covers both


def test_the_rows_are_further_apart_than_the_columns():
    """Up and down should be easier than left and right, as on a real shelf - otherwise the
    ambiguity is coming from the layout being odd rather than from things being alike."""
    index = by_id(shelf())
    across = separation_deg(index["bottle_1"], index["bottle_2"])
    up = separation_deg(index["bottle_1"], index["mug_1"])
    assert up > across
    assert 7.0 < up < 11.0


def test_the_shelf_is_within_arms_length_of_the_person():
    """It has to be: the arm reaches 280 mm. The plan's 2.5 m shelf could not have been
    picked from at all, which is why the scene moved onto the table."""
    eye_to_shelf = math.dist((0.0, 0.0, 0.0), shelf()[0].pos)
    assert 0.6 < eye_to_shelf < 0.9


def test_the_pointing_geometry_comes_from_the_physics_geometry():
    """One layout, two coordinate systems. `pos` is derived from `place_mm`, so the thing
    the arm reaches for and the thing the cone weighs cannot be in different places."""
    for t in shelf():
        assert t.pos == from_eye(t.place_mm)


def test_columns_are_evenly_spaced_and_centred():
    xs = [column_x(c) for c in range(COLS)]
    gaps = {round(b - a, 6) for a, b in zip(xs, xs[1:], strict=False)}
    assert len(gaps) == 1
    assert round(sum(xs) / len(xs), 6) == 250.0


# ---------------------------------------------------------------- what no word can separate


def test_six_things_no_word_can_separate():
    counts = kind_counts(shelf())
    assert counts["bottle"] == 4
    assert counts["mug"] == 2
    assert sum(n for n in counts.values() if n > 1) == 6


def test_identical_things_are_word_for_word_identical():
    """If two bottles differed in `seen` or `says`, the robot's camera or the person's words
    could separate them and the demo would be rigged in our favour instead of against."""
    things = shelf()
    for kind, n in kind_counts(things).items():
        if n > 1:
            same = [t for t in things if t.kind == kind]
            assert len({t.seen for t in same}) == 1, kind
            assert len({t.says for t in same}) == 1, kind


def test_the_baseline_can_still_win():
    """Words-only must be able to score 100% on something, or the comparison is a stitch-up.
    Six things are one of a kind, and two of them are describable without their name."""
    things = shelf()
    one_offs = [t for t in things if is_unique(things, t.id)]
    assert len(one_offs) == 6
    assert is_unique(things, "scissors") and is_unique(things, "phone")
    assert any("cutting" in p for p in by_id(things)["scissors"].says)
    assert any("writing" in p for p in by_id(things)["pen"].says)


def test_the_nameless_one():
    """A bracket is there because nobody has a word for it - the case words cannot reach
    even when the thing is one of a kind."""
    bracket = by_id(shelf())["bracket"]
    assert is_unique(shelf(), "bracket")
    assert all("bracket" not in p for p in bracket.says)


def test_the_tray_is_not_something_to_want():
    things = shelf()
    assert "tray" not in [t.id for t in wanted(things)]
    assert len(wanted(things)) == 11


# ---------------------------------------------------------------- the arms


def test_every_thing_is_reachable_by_someone():
    """E7's gate. Computed from the measured ring, so it cannot drift."""
    all_arms = arms()
    for t in shelf():
        assert any(a.can_reach(t) for a in all_arms), f"{t.id} at {t.where}"


def test_only_one_arm_can_reach_the_person():
    """The two-arm story depends on this. If both could hand over, the second arm would be
    scenery."""
    reach_them = [a.id for a in arms() if a.reaches_person]
    assert reach_them == ["left_arm"]


def test_the_hand_that_works_is_the_left_one():
    """A left-hemisphere stroke takes language and the right side of the body together, so
    the person who cannot say "mug" is frequently the same person whose right arm does not
    work. The thing is handed to their left."""
    assert GIVE_MM[0] < 250.0


def test_some_things_need_the_arm_that_cannot_hand_over():
    """So they have to be passed across. Without this the pass is never exercised, and
    "you, that one" has nothing to be about."""
    things, all_arms = shelf(), arms()
    givers = [a for a in all_arms if a.reaches_person]
    needs_a_pass = [t.id for t in wanted(things) if not any(a.can_reach(t) for a in givers)]
    assert needs_a_pass  # there is at least one
    assert len(needs_a_pass) >= 3  # and enough that it is not a fluke of one object


def test_each_arm_owns_part_of_the_shelf_and_shares_part():
    left, right = arms()
    things = shelf()
    only_left = [t.id for t in things if left.can_reach(t) and not right.can_reach(t)]
    only_right = [t.id for t in things if right.can_reach(t) and not left.can_reach(t)]
    both = [t.id for t in things if left.can_reach(t) and right.can_reach(t)]
    assert only_left and only_right and both


def test_reach_is_the_measured_ring_and_nothing_else():
    """Not a hand-written list of columns. Move a base and the answers change, rather than
    silently contradicting what the arm can really do."""
    right = next(a for a in arms() if a.facing_deg == 0)
    x, y = right.base_mm
    near, far = REACH_RING_MM[1]
    # Straight ahead is +y, away from the person, which is where this arm faces.
    assert not right.reaches(x, y)  # too close in: the gripper would hit its own shoulder
    assert right.reaches(x, y + near + 1)
    assert right.reaches(x, y + far - 1)
    assert not right.reaches(x, y + far + 1)


def test_reach_runs_out_as_the_arm_turns_away_from_forward():
    """Reach is not a circle: the further round the shoulder goes, the less of it is left.
    A spot the arm can just manage straight ahead is out of reach off to one side."""
    right = next(a for a in arms() if a.facing_deg == 0)
    x, y = right.base_mm
    _, far = REACH_RING_MM[1]
    assert right.reaches(x, y + far - 2)  # straight ahead, almost at the edge
    # The same distance away, a quarter turn round: nearer than the edge, still out of reach.
    assert not right.reaches(x + far - 2, y)


def test_nothing_behind_an_arm_is_reachable_however_close():
    """The shoulder stops at PAN_LIMIT_DEG. This is what the plain ring got wrong: it put
    the person's hand well inside the reach and the arm could not turn far enough to touch
    it."""
    right = next(a for a in arms() if a.facing_deg == 0)
    x, y = right.base_mm
    near, far = REACH_RING_MM[1]
    middle = (near + far) / 3
    assert not right.reaches(x, y - middle)  # directly behind
    # Straight behind wraps to -180, not +180; it is the size of the turn that matters.
    assert abs(right.bearing_to(x, y - middle)) == pytest.approx(180.0, abs=1e-6)
    assert PAN_LIMIT_DEG < 180.0


def test_an_arms_columns_agree_with_its_reach():
    for arm in arms():
        for row in (0, 1):
            for col in arm.columns(row):
                assert arm.reaches(column_x(col), [400.0, 300.0][row], row)


# ---------------------------------------------------------------- shuffling


def test_reshuffle_keeps_the_same_twelve_things_in_new_places():
    before = shelf()
    after = reshuffle(before, random.Random(7))
    assert {t.id for t in after} == {t.id for t in before}
    assert len({t.place for t in after}) == 12
    assert layout(after) != layout(before)
    # Nothing but the position changes - a shuffled bottle is still that bottle.
    assert {(t.id, t.kind, t.seen) for t in after} == {(t.id, t.kind, t.seen) for t in before}


def test_a_shuffled_shelf_is_still_all_reachable():
    all_arms = arms()
    for seed in range(20):
        for t in reshuffle(shelf(), random.Random(seed)):
            assert any(a.can_reach(t) for a in all_arms), (seed, t.id)
