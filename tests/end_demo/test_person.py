"""The simulated person: a counter for the budget, and an unreliable word with an honest hand.

The unreliable-responder tests are the ones that matter. The clinical picture is that yes/no
can be wrong while pointing and understanding stay intact, so `unreliable` must flip the
**word only**. If it ever started wobbling the point as well, the robot's contradiction rule
would stop being exercised - both channels would be wrong together and agree.
"""

from __future__ import annotations

import random

from end_demo.confirm import read
from end_demo.person import ModelPerson, Person, pick_target
from end_demo.point import SIGMA_DEG, sole
from end_demo.scene import by_id, shelf, wanted
from end_demo.stub import ScriptDecider

THINGS = shelf()
INDEX = by_id(THINGS)


def person(target="bottle_3", **kw) -> Person:
    return Person(INDEX[target], THINGS, random.Random(4), **kw)


# ---------------------------------------------------------------- the word budget


def test_the_budget_is_a_counter_and_it_counts():
    p = person("bottle_3")
    said = p.say(2)
    assert len(said.split()) <= 2
    assert p.words_used == len(said.split())


def test_a_budget_of_zero_gets_nothing_out():
    p = person()
    assert p.say(0) == ""
    assert p.words_used == 0


def test_they_use_the_words_they_have():
    """The fullest phrase that fits, not the shortest they could get away with. Saying less
    than they could would make the words-only baseline look worse than it is, and that
    baseline is the thing we claim to beat."""
    assert person("scissors").say(1) == "cutting"
    assert person("scissors").say(3) == "to cut with"
    assert person("scissors").say(2) == "for cutting"


def test_terse_reproduces_the_harness_person_for_the_port_check():
    assert person("scissors", terse=True).say(4) == "cutting"


def test_two_things_of_the_same_kind_have_the_same_words():
    """Not a trick: there is no word that separates one white bottle from the next."""
    assert person("bottle_1").say(2) == person("bottle_4").say(2)
    assert person("mug_1").say(2) == person("mug_2").say(2)


def test_a_name_is_never_needed_to_describe_the_describable_ones():
    """R1 and R3: the scissors can be asked for by what they are for."""
    assert "scissors" not in person("scissors").say(2)
    assert "pen" not in person("pen").say(2)


# ---------------------------------------------------------------- the hand


def test_they_point_and_they_miss():
    p = person("bottle_3")
    cones = [p.point() for _ in range(20)]
    assert p.points == 20
    # Their aim is not perfect, so the best candidate is not always the one they want.
    assert any(c.best != "bottle_3" for c in cones)
    assert any(c.best == "bottle_3" for c in cones)


def test_a_perfect_pointer_lands_on_the_thing_every_time():
    p = person("bottle_3", sigma_deg=0.0)
    assert all(sole(p.point()) == "bottle_3" for _ in range(5))


def test_the_default_aim_is_the_published_scale():
    assert person().sigma_deg == SIGMA_DEG == 3.0


# ---------------------------------------------------------------- answering


def test_a_reliable_person_tells_the_truth():
    p = person("bottle_3", style="word")
    assert p.to_this_one("bottle_3").said == "yes"
    assert p.to_this_one("bottle_1").said == "no"
    assert p.misled == 0


def test_answering_by_both_gives_a_word_and_a_point():
    reply = person("bottle_3", style="both", sigma_deg=0.0).to_this_one("bottle_3")
    assert reply.said == "yes"
    assert reply.point is not None
    assert read(reply, "bottle_3").kind == "yes"


def test_someone_with_no_speech_answers_with_their_hand_alone():
    p = person("bottle_3", style="point", sigma_deg=0.0)
    reply = p.to_this_one("bottle_1")
    assert reply.said == ""
    assert p.answers == 0  # not one yes or no was spent
    assert read(reply, "bottle_1").kind == "new_target"


def test_yes_no_answers_are_free_and_never_counted_as_words():
    p = person("bottle_3", style="both")
    p.say(2)
    before = p.words_used
    for _ in range(5):
        p.to_this_one("bottle_1")
    assert p.words_used == before  # a nod costs nothing


# ---------------------------------------------------------------- the unreliable responder


def test_unreliable_flips_the_word_and_only_the_word():
    """The clinical picture: the yes/no is unreliable, the pointing is not. If this test
    ever passes while the point is also being corrupted, R5 has stopped being tested."""
    p = person("bottle_3", style="both", unreliable=1.0, sigma_deg=0.0)
    reply = p.to_this_one("bottle_3")
    assert reply.said == "no"  # the word is wrong
    assert sole(reply.point) == "bottle_3"  # the hand is right
    assert p.misled == 1
    # And the robot refuses to act on the disagreement rather than believing either half.
    assert read(reply, "bottle_3").kind == "contradiction"


def test_unreliable_is_a_rate_not_a_switch():
    p = person("bottle_3", style="word", unreliable=0.5)
    for _ in range(200):
        p.to_this_one("bottle_3")
    assert 60 < p.misled < 140  # about half, and recorded


def test_the_wrong_answers_are_counted_so_they_can_be_reported():
    p = person("bottle_3", style="word", unreliable=1.0)
    p.to_this_one("bottle_1")
    p.to_this_one("bottle_3")
    assert p.misled == 2
    assert p.answers == 2


# ---------------------------------------------------------------- silence


def test_silence_costs_nothing_and_says_nothing():
    p = person("bottle_3", silence=1.0)
    reply = p.to_this_one("bottle_3")
    assert reply.silent
    assert p.answers == 0 and p.points == 0


def test_a_fixed_choice_cannot_be_answered_by_someone_limited_to_yes_and_no():
    """Reported honestly rather than papered over: "this one or this one?" has no yes/no
    answer, so a person with only yes/no genuinely cannot answer it."""
    reply = person("mug_2", style="word").to_choice("mug_1", "mug_2")
    assert reply.silent


def test_show_me_gets_a_fresh_point():
    p = person("bottle_3", sigma_deg=0.0)
    assert sole(p.to_show_me().point) == "bottle_3"
    assert p.points == 1


# ---------------------------------------------------------------- the model as the person


def test_the_model_person_is_still_held_to_the_counter():
    """The budget is mechanical whoever is speaking: whatever the model produces is cut to
    the budget, because a model policing its own word count would be a judgement call in
    the one place this project insists on a counter."""
    model = ScriptDecider(script=[("say", {"words": "the one for cutting things with"})])
    p = ModelPerson(INDEX["scissors"], THINGS, random.Random(1), model=model)
    said = p.say(2)
    assert said == "the one"
    assert p.words_used == 2


def test_the_model_person_is_told_which_thing_it_wants_and_that_it_cannot_name_it():
    model = ScriptDecider(script=[("say", {"words": "that one"})])
    p = ModelPerson(INDEX["bottle_3"], THINGS, random.Random(1), model=model)
    p.say(2)
    brief = model.seen[0][1]["content"]
    assert "number 3 from the left" in brief
    assert "cannot say which number it is" in brief
    assert "cannot say what it is called" in brief


def test_the_model_persons_hand_is_geometry_not_a_model():
    """A language model has no aim and must not be asked to pretend it has one."""
    model = ScriptDecider(script=[])
    p = ModelPerson(INDEX["bottle_3"], THINGS, random.Random(1), model=model, sigma_deg=0.0)
    assert sole(p.point()) == "bottle_3"
    assert model.at == 0  # the model was not consulted about where their finger went


def test_the_harness_falls_back_to_the_script_but_the_robot_never_does():
    """When the person-model cannot be reached the *test harness* uses the scripted phrases
    and records it. That is a harness fallback; the robot has none anywhere."""
    model = ScriptDecider(script=[])  # runs out immediately
    p = ModelPerson(INDEX["scissors"], THINGS, random.Random(1), model=model)
    assert p.say(2) == "for cutting"
    assert p.fell_back == 1


def test_the_model_persons_answer_can_still_be_unreliable():
    model = ScriptDecider(script=[("answer", {"yes": True})])
    p = ModelPerson(
        INDEX["bottle_3"], THINGS, random.Random(1), model=model, unreliable=1.0, sigma_deg=0.0
    )
    reply = p.to_this_one("bottle_3")
    assert reply.said == "no"
    assert p.misled == 1


# ---------------------------------------------------------------- picking a target


def test_nobody_ever_wants_the_tray():
    rng = random.Random(0)
    picks = {pick_target(THINGS, rng).id for _ in range(200)}
    assert "tray" not in picks
    assert picks <= {t.id for t in wanted(THINGS)}


# ---------------------------------------------------------------- where a confirming point goes


def test_confirming_points_at_the_thing_the_robot_is_showing_them():
    """Not re-aimed from scratch. The robot has put its gripper over the thing; agreeing
    means pointing at that. Re-aiming at their own target turned correct yeses into
    contradictions whenever the wobble landed on a neighbour."""
    p = person("bottle_3", style="both", sigma_deg=0.0)
    reply = p.to_this_one("bottle_3")
    assert reply.said == "yes"
    assert sole(reply.point) == "bottle_3"


def test_rejecting_points_at_what_they_actually_want():
    p = person("bottle_3", style="both", sigma_deg=0.0)
    reply = p.to_this_one("mug_1")
    assert reply.said == "no"
    assert sole(reply.point) == "bottle_3"  # not mug_1: they are showing the right one


def test_the_hand_follows_the_truth_even_when_the_word_does_not():
    """An unreliable responder says the wrong word and still points honestly. If the hand
    followed the word instead, both channels would be wrong together and agree - and R5
    would never catch anything."""
    p = person("bottle_3", style="both", unreliable=1.0, sigma_deg=0.0)
    reply = p.to_this_one("bottle_3")
    assert reply.said == "no"
    assert sole(reply.point) == "bottle_3"
    assert read(reply, "bottle_3").kind == "contradiction"
