"""The shelf has to be hard for an honest reason, and the arms have to need each other."""

from __future__ import annotations

import math
import random

from end_demo.point import separation_deg
from end_demo.scene import (
    COLS,
    DIST,
    ROWS,
    arms,
    by_id,
    is_unique,
    kind_counts,
    layout,
    reshuffle,
    shelf,
    wanted,
)


def test_twelve_things_in_two_rows():
    things = shelf()
    assert len(things) == COLS * ROWS == 12
    assert sorted({t.row for t in things}) == [0, 1]
    for row in (0, 1):
        assert sorted(t.col for t in things if t.row == row) == list(range(COLS))


def test_six_things_no_word_can_separate():
    """Four identical bottles and two identical mugs. This is the whole demo: no word
    separates bottle 2 from bottle 3, for a judge as completely as for anyone."""
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
    # R3: describable by what it is for, not by its name.
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


def test_neighbours_are_closer_than_a_person_can_point():
    """At 2.5 m the thing next door is about 5.7 degrees away, and a person's aim is worse
    than that. Perspective makes the task hard; no noise is added anywhere."""
    index = by_id(shelf())
    across = separation_deg(index["bottle_1"].pos, index["bottle_2"].pos)
    assert 5.0 < across < 6.5
    up = separation_deg(index["bottle_1"].pos, index["mug_1"].pos)
    assert 7.0 < up < 9.0
    assert math.isclose(index["bottle_1"].pos[1], DIST)


def test_each_arm_alone_cannot_do_the_job():
    """Two arms are only worth their cost if they need each other. The arm that can reach
    the far left cannot reach the person, so a thing there has to be passed across."""
    left, right = arms()
    things = by_id(shelf())
    assert not left.reaches_person
    assert right.reaches_person
    assert left.can_reach(things["bottle_1"]) and not right.can_reach(things["bottle_1"])
    assert right.can_reach(things["remote"]) and not left.can_reach(things["remote"])
    # And an overlap in the middle, so "which arm" is sometimes a real choice.
    assert left.can_reach(things["bracket"]) and right.can_reach(things["bracket"])


def test_every_thing_is_reachable_by_someone():
    all_arms = arms()
    for t in shelf():
        assert any(a.can_reach(t) for a in all_arms), t.id


def test_reshuffle_keeps_the_same_twelve_things_in_new_places():
    before = shelf()
    after = reshuffle(before, random.Random(7))
    assert {t.id for t in after} == {t.id for t in before}
    assert len({t.place for t in after}) == 12
    assert layout(after) != layout(before)
    # Nothing but the position changes - a shuffled bottle is still that bottle.
    assert {(t.id, t.kind, t.seen) for t in after} == {(t.id, t.kind, t.seen) for t in before}
