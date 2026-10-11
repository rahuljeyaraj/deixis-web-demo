"""R7 and R8, over every utterance the robot can make.

R8 says every utterance appears three ways at once - spoken, written large, and gestured.
R7 says every question is answerable without words. Both are checked over the whole set
rather than one at a time, so a new utterance added later cannot quietly skip them.
"""

from __future__ import annotations

import pytest

from deixis.where import ASK_TEXTS
from end_demo import say
from end_demo.say import ASKS, WRITTEN, Gesture, Utterance

ALL = [
    say.this_one("bottle_2", "right_arm"),
    say.choice("bottle_2", "bottle_3", "right_arm"),
    say.show_me("right_arm"),
    say.contradiction("bottle_2", "bottle_3", "right_arm"),
    say.heard("that one", "right_arm"),
    say.cannot("I can't reach it", "right_arm"),
    say.giving("bottle_2", "right_arm"),
    say.taking("bottle_2", "left_arm"),
    say.done("right_arm"),
    say.waiting("right_arm"),
]


@pytest.mark.parametrize("u", ALL, ids=lambda u: u.kind)
def test_every_utterance_is_spoken_written_and_gestured(u: Utterance):
    assert u.spoken.strip()
    assert u.written.strip()
    assert isinstance(u.gesture, Gesture)


@pytest.mark.parametrize("u", ALL, ids=lambda u: u.kind)
def test_no_utterance_needs_words_back(u: Utterance):
    assert not u.needs_words


@pytest.mark.parametrize("u", ALL, ids=lambda u: u.kind)
def test_the_written_line_is_short_enough_to_read_from_three_metres(u: Utterance):
    """Written keywords, not a transcript of the sentence. At 28 pt a long line stops being
    readable across a room, which is the only reason the written channel exists."""
    longest = max(len(line) for line in u.written.splitlines())
    assert longest <= 24, u.written


def test_a_question_cannot_be_built_without_something_to_point_at():
    """A "this one?" with nothing under the gripper is not a question, it is noise."""
    with pytest.raises(ValueError):
        Gesture("hover", (), "right_arm")
    with pytest.raises(ValueError):
        Gesture("each", ("bottle_1",), "right_arm")


def test_an_utterance_with_nothing_written_is_refused():
    with pytest.raises(ValueError, match="R8"):
        Utterance("done", "There you go.", "  ", Gesture("none", (), "right_arm"))


def test_the_questions_are_the_ones_that_can_be_answered_by_pointing():
    questions = [u for u in ALL if u.is_question]
    assert {u.kind for u in questions} == {"this_one", "choice", "show_me", "contradiction"}
    for u in questions:
        assert u.gesture.kind in ("hover", "each", "open")


def test_the_fixed_choice_hovers_over_both_in_turn():
    g = say.choice("mug_1", "mug_2", "left_arm").gesture
    assert g.kind == "each"
    assert g.ids == ("mug_1", "mug_2")


def test_we_replace_the_core_question_that_would_demand_a_word():
    """The core's own text for an ambiguous object is "The {options}?" - "the blue mug or
    the grey one?" - which is the one question this person cannot answer. We override it
    through the seam the core already has, rather than forking the core."""
    assert "{options}" in ASK_TEXTS["ambiguous_object"]
    assert "{options}" not in ASKS["ambiguous_object"]
    assert say.this_one("mug_1", "right_arm").spoken == ASKS["this_one"]


def test_the_written_and_the_spoken_are_not_the_same_string():
    """Keeping them in one field is how you end up with a spoken sentence on screen in
    capitals, or a two-word fragment read aloud."""
    for u in ALL:
        if u.kind in WRITTEN:
            assert u.spoken != u.written


def test_what_was_heard_is_shown_even_when_it_is_nothing():
    """R6: the transcript is shown, never a blocker. Empty is a legitimate thing to show."""
    u = say.heard("", "right_arm")
    assert "(nothing)" in u.spoken
    assert u.gesture.kind == "none"
