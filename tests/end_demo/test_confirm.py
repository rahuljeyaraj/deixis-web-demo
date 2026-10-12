"""R4 and R5, the whole truth table.

These are the tests that matter most in the repo. The demo's accuracy rests on the robot
being allowed to ask - and the research says the answer to a yes/no question cannot always
be trusted. So the rules are: the hand outranks the word, and a disagreement is not an
answer at all.

If `read` ever starts breaking a tie between a word and a hand, the headline number stops
being a measurement of the system and becomes a measurement of the tie-break.
"""

from __future__ import annotations

import pytest

from end_demo.confirm import Answer, Counts, Reply, read, read_choice, yes_no
from end_demo.point import weigh
from end_demo.scene import by_id, shelf

THINGS = shelf()
INDEX = by_id(THINGS)


def point_at_thing(thing_id: str):
    """A dead-straight point at one thing: no wobble, so the test is about the rules."""
    from end_demo.point import angles

    return weigh(angles(INDEX[thing_id].pos), THINGS)


def between(a: str, b: str):
    """A point exactly between two things, which the core's 2:1 rule calls too close."""
    from end_demo.point import angles

    (x1, y1), (x2, y2) = angles(INDEX[a].pos), angles(INDEX[b].pos)
    return weigh(((x1 + x2) / 2, (y1 + y2) / 2), THINGS)


# ---------------------------------------------------------------- the yes/no guard


@pytest.mark.parametrize(
    "text,expected",
    [
        ("yes", True),
        ("Yes!", True),
        ("yeah", True),
        ("mhm", True),
        ("no", False),
        ("nope", False),
        ("not that", False),
        ("", None),
        ("bottle", None),  # a misheard word is not an answer, and must not block (R6)
        ("yes no", None),  # both: no answer
    ],
)
def test_yes_no_only_reads_yes_and_no(text, expected):
    assert yes_no(text) is expected


# ---------------------------------------------------------------- the hand outranks the word


def test_a_point_at_the_thing_is_a_yes():
    answer = read(Reply(point=point_at_thing("bottle_2")), "bottle_2")
    assert answer.kind == "yes"
    assert answer.acts


def test_a_point_at_something_else_is_the_new_answer_not_a_rejection():
    """R4. They have told us what they mean; making them say no first and then point again
    would be charging them twice for one gesture."""
    answer = read(Reply(point=point_at_thing("bottle_4")), "bottle_2")
    assert answer.kind == "new_target"
    assert answer.id == "bottle_4"
    assert answer.acts


def test_a_word_alone_still_works():
    """Someone who is a reliable responder should not be made to point for everything."""
    assert read(Reply(said="yes"), "mug_1").kind == "yes"
    assert read(Reply(said="no"), "mug_1").kind == "no"


def test_a_nod_counts_as_a_word():
    assert read(Reply(nod=True), "mug_1").kind == "yes"
    assert read(Reply(nod=False), "mug_1").kind == "no"


def test_the_nod_outranks_the_transcript():
    """If the tracker saw a shake and the recogniser heard "yes", the body wins: the
    recogniser is the thing with a 64-77% hit rate on this speech."""
    assert read(Reply(said="yes", nod=False), "mug_1").kind == "no"


# ---------------------------------------------------------------- contradictions (R5)


def test_yes_while_pointing_elsewhere_is_a_contradiction_and_is_never_acted_on():
    answer = read(Reply(point=point_at_thing("bottle_4"), said="yes"), "bottle_2")
    assert answer.kind == "contradiction"
    assert not answer.acts
    assert "bottle_4" in answer.why and "bottle_2" in answer.why


def test_no_while_pointing_straight_at_it_is_also_a_contradiction():
    answer = read(Reply(point=point_at_thing("bottle_2"), said="no"), "bottle_2")
    assert answer.kind == "contradiction"
    assert not answer.acts


def test_no_and_pointing_elsewhere_is_consistent_and_usable():
    """Saying no and showing which one is not a contradiction - it is a complete answer."""
    answer = read(Reply(point=point_at_thing("bottle_4"), said="no"), "bottle_2")
    assert answer.kind == "new_target"
    assert answer.id == "bottle_4"


def test_yes_and_pointing_at_it_agree():
    answer = read(Reply(point=point_at_thing("bottle_2"), said="yes"), "bottle_2")
    assert answer.kind == "yes"
    assert answer.word is True


# ---------------------------------------------------------------- nothing to go on


def test_a_point_too_close_to_call_is_not_an_answer():
    """The core's own 2:1 rule. A point that lands between two bottles has not answered."""
    answer = read(Reply(point=between("bottle_2", "bottle_3")), "bottle_2")
    assert answer.kind == "unclear"
    assert not answer.acts


def test_silence_is_an_answer_that_must_not_be_acted_on():
    answer = read(Reply(), "bottle_2")
    assert answer.kind == "nothing"
    assert not answer.acts


def test_a_misheard_word_does_not_block(caplog):
    """R6. One word misheard must not cost the whole command: it is passed on as words, and
    the robot is told it may be a mishearing."""
    answer = read(Reply(said="bottle"), "bottle_2")
    assert answer.kind == "said"
    assert answer.said == "bottle"
    assert not answer.acts


# ---------------------------------------------------------------- a fixed choice


def test_a_choice_is_answered_by_pointing_at_one_of_the_two():
    answer = read_choice(Reply(point=point_at_thing("mug_2")), "mug_1", "mug_2")
    assert answer.kind == "yes"
    assert answer.id == "mug_2"


def test_pointing_at_a_third_thing_answers_the_choice_too():
    answer = read_choice(Reply(point=point_at_thing("phone")), "mug_1", "mug_2")
    assert answer.kind == "new_target"
    assert answer.id == "phone"


def test_a_choice_cannot_be_answered_with_a_word():
    """ "Yes" to "this one or this one?" says nothing, so it is not taken as an answer."""
    answer = read_choice(Reply(said="yes"), "mug_1", "mug_2")
    assert answer.kind == "said"
    assert not answer.acts


# ---------------------------------------------------------------- counting


def test_contradictions_are_counted():
    counts = Counts()
    for kind in ("contradiction", "contradiction", "unclear", "nothing", "yes"):
        counts.add(Answer(kind))
    assert counts.contradictions == 2
    assert counts.unclear == 1
    assert counts.silences == 1
    assert counts.kinds["yes"] == 1


# ---------------------------------------------------------------- an unreadable point


def test_an_unreadable_point_leaves_the_word_standing():
    """The expensive bug. A point whose top two are inside the core's 2:1 rule is the hand
    saying nothing - not the hand disagreeing - so a clean spoken yes still counts. Dropping
    both cost 28 points of accuracy in the first run on the twelve-thing shelf."""
    reply = Reply(point=between("bottle_2", "bottle_3"), said="yes")
    answer = read(reply, "bottle_2")
    assert answer.kind == "yes"
    assert answer.acts
    assert "settled nothing" in answer.why


def test_an_unreadable_point_with_a_no_is_still_a_no():
    answer = read(Reply(point=between("bottle_2", "bottle_3"), said="no"), "bottle_2")
    assert answer.kind == "no"
    assert not answer.acts


def test_an_unreadable_point_and_no_word_is_still_unclear():
    answer = read(Reply(point=between("bottle_2", "bottle_3")), "bottle_2")
    assert answer.kind == "unclear"
    assert not answer.acts


def test_an_empty_cone_with_a_word_takes_the_word():
    """They pointed at the ceiling and said yes. The point is useless; the yes is not."""
    from end_demo.point import weigh

    nowhere = weigh((1.2, 1.2), THINGS)
    assert not nowhere.candidates
    assert read(Reply(point=nowhere, said="yes"), "bottle_2").kind == "yes"


def test_a_contradiction_still_needs_a_readable_point():
    """R5 must fire on a real disagreement and not on a wobble. The point has to settle on
    something before it can disagree with anything."""
    assert read(Reply(point=point_at_thing("bottle_4"), said="yes"), "bottle_2").kind == (
        "contradiction"
    )
    assert read(Reply(point=between("bottle_2", "bottle_3"), said="yes"), "bottle_2").kind == "yes"
