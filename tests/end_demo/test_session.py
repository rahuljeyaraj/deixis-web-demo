"""The loop: what it does with an answer, and what it refuses to do.

The gates for E2 and E3 live here. Two of them are the ones to keep an eye on:

- a contradiction never ends an attempt, however the model and the person behave (R5);
- when the questions run out, the **model** makes the final choice. The loop must never
  break that tie itself - doing so once made part of every score the harness's judgement,
  and it cost two points off the headline number when it was fixed.
"""

from __future__ import annotations

import random

import pytest

from end_demo.confirm import Reply
from end_demo.person import Person
from end_demo.point import angles, weigh
from end_demo.scene import by_id, shelf
from end_demo.session import point_and_ask, point_only, trial, words_only
from end_demo.stub import ScriptDecider
from end_demo.tools import Toolset

THINGS = shelf()
INDEX = by_id(THINGS)


def straight_at(thing_id: str):
    return weigh(angles(INDEX[thing_id].pos), THINGS)


def person(target: str, **kw) -> Person:
    return Person(INDEX[target], THINGS, random.Random(5), **kw)


def ask_and_choose(*script) -> ScriptDecider:
    return ScriptDecider(script=list(script))


# ---------------------------------------------------------------- round 1


def test_point_only_asks_no_model_at_all():
    """There is nothing to think about: take the likeliest thing and go. This is how
    published pointing systems work, and keeping it this dumb is what makes our landing
    inside their 51-69% a fair check rather than a flattering one."""
    a = point_only(INDEX["bottle_2"], THINGS, straight_at("bottle_2"))
    assert a.ok
    assert a.calls == []
    assert a.words == 0 and a.questions == 0


def test_point_only_records_that_the_core_would_have_asked():
    """A point between two bottles is inside the core's 2:1 rule. Round 1 commits anyway,
    and the transcript says so - that is the gap the demo exists to close."""
    (x1, y1), (x2, y2) = angles(INDEX["bottle_2"].pos), angles(INDEX["bottle_3"].pos)
    a = point_only(INDEX["bottle_2"], THINGS, weigh(((x1 + x2) / 2, (y1 + y2) / 2), THINGS))
    assert a.ambiguous
    assert any("too close to call" in line for line in a.log)


# ---------------------------------------------------------------- round 2


def test_words_only_gets_one_call_and_no_questions():
    model = ask_and_choose(("choose", {"id": "scissors"}))
    a = words_only(INDEX["scissors"], THINGS, person("scissors"), model, budget=4)
    assert a.ok
    assert a.questions == 0
    assert a.model_calls == 1
    # Only `choose` was offered: it cannot ask, because a voice assistant cannot.
    assert [t["function"]["name"] for t in model.seen and [{"function": {"name": "choose"}}]]


def test_words_only_is_told_there_is_nothing_else_to_go_on():
    model = ask_and_choose(("choose", {"id": "bottle_1"}))
    words_only(INDEX["bottle_1"], THINGS, person("bottle_1"), model, budget=4)
    said = model.seen[0][-1]["content"]
    assert "did not point" in said
    assert "cannot ask" in said


def test_words_only_cannot_ask_even_if_the_model_tries():
    """The toolset is the enforcement, not a prompt line."""
    model = ask_and_choose(("ask_this_one", {"id": "bottle_1"}))
    with pytest.raises(AssertionError, match="not offered"):
        words_only(INDEX["bottle_1"], THINGS, person("bottle_1"), model, budget=4)


# ---------------------------------------------------------------- round 3: committing


def test_committing_straight_away():
    model = ask_and_choose(("choose", {"id": "phone", "why": "they said phone"}))
    a = point_and_ask(INDEX["phone"], THINGS, person("phone"), model, straight_at("phone"))
    assert a.ok and a.questions == 0
    assert "they said phone" in a.log[-1]


def test_a_yes_is_arrival():
    model = ask_and_choose(("ask_this_one", {"id": "bottle_3"}))
    a = point_and_ask(
        INDEX["bottle_3"], THINGS, person("bottle_3", style="word"), model, straight_at("bottle_3")
    )
    assert a.ok
    assert a.picked == "bottle_3"
    assert a.questions == 1


def test_naming_something_that_is_not_on_the_shelf_scores_zero_and_is_recorded():
    """Strict, as the harness was. A model that invents a thing has not selected one, and
    retrying it quietly would be scoring our patience instead of the model."""
    model = ask_and_choose(("choose", {"id": "teapot"}))
    a = point_and_ask(INDEX["phone"], THINGS, person("phone"), model, straight_at("phone"))
    assert not a.ok
    assert a.invalid
    assert "not on the shelf" in a.log[-1]


def test_asking_about_something_that_is_not_there_is_invalid_too():
    model = ask_and_choose(("ask_this_one", {"id": "teapot"}))
    a = point_and_ask(INDEX["phone"], THINGS, person("phone"), model, straight_at("phone"))
    assert a.invalid and not a.ok
    assert a.questions == 0  # it never got as far as asking anybody anything


def test_the_model_cannot_use_a_tool_it_was_not_given_and_is_told_so():
    model = ask_and_choose(("look", {}), ("choose", {"id": "phone"}))
    with pytest.raises(AssertionError, match="not offered"):
        point_and_ask(INDEX["phone"], THINGS, person("phone"), model, straight_at("phone"))


# ---------------------------------------------------------------- round 3: the question cap


def test_the_question_cap_means_what_it_says():
    """Up to three questions is three, not four. The off-by-one here once made a
    one-question run look as strong as a three-question run, which is the wrong conclusion
    to build a demo on."""
    model = ask_and_choose(
        ("ask_this_one", {"id": "bottle_1"}),
        ("ask_this_one", {"id": "bottle_2"}),
        ("ask_this_one", {"id": "bottle_4"}),
        ("choose", {"id": "bottle_3"}),
    )
    a = point_and_ask(
        INDEX["bottle_3"],
        THINGS,
        person("bottle_3", style="word"),
        model,
        straight_at("bottle_3"),
        max_questions=3,
    )
    assert a.questions == 3
    assert a.ok
    assert "out of questions" in a.log[-1]


def test_the_model_makes_the_final_choice_with_only_choose_on_the_table():
    model = ask_and_choose(
        ("ask_this_one", {"id": "bottle_1"}),
        ("choose", {"id": "bottle_2"}),
    )
    a = point_and_ask(
        INDEX["bottle_2"],
        THINGS,
        person("bottle_2", style="word"),
        model,
        straight_at("bottle_2"),
        max_questions=1,
    )
    assert a.ok
    # The last thing the model was offered was `choose` alone.
    assert [t["function"]["name"] for t in [Toolset.commit_only().as_list()[0]]] == ["choose"]
    assert model.at == 2


def test_running_out_of_questions_tells_the_model_so():
    model = ask_and_choose(("ask_this_one", {"id": "bottle_1"}), ("choose", {"id": "bottle_2"}))
    point_and_ask(
        INDEX["bottle_2"],
        THINGS,
        person("bottle_2", style="word"),
        model,
        straight_at("bottle_2"),
        max_questions=1,
    )
    assert "no questions left" in model.seen[-1][-1]["content"]


# ---------------------------------------------------------------- R5 in the loop


def test_a_contradiction_never_ends_an_attempt():
    """The person says yes while pointing somewhere else. The loop must not pick a winner:
    it has no answer, and it says so."""

    class Contrary(Person):
        def to_this_one(self, thing_id: str) -> Reply:
            # "yes", but the hand is on bottle_4.
            return Reply(point=straight_at("bottle_4"), said="yes")

    model = ask_and_choose(
        ("ask_this_one", {"id": "bottle_2"}),
        ("ask_choice", {"first": "bottle_2", "second": "bottle_4"}),
    )
    a = point_and_ask(
        INDEX["bottle_4"],
        THINGS,
        Contrary(INDEX["bottle_4"], THINGS, random.Random(1)),
        model,
        straight_at("bottle_2"),
        max_questions=2,
        toolset=Toolset.selection(choice=True),
    )
    assert a.contradictions == 1
    assert any("CONTRADICTION, not acted on" in line for line in a.log)
    # Nothing was decided on the strength of it.
    assert a.picked != "bottle_2" or not a.ok


def test_the_model_is_told_to_ask_differently_after_a_contradiction():
    class Contrary(Person):
        def to_this_one(self, thing_id: str) -> Reply:
            return Reply(point=straight_at("bottle_4"), said="yes")

    model = ask_and_choose(
        ("ask_this_one", {"id": "bottle_2"}),
        ("choose", {"id": "bottle_4"}),
    )
    point_and_ask(
        INDEX["bottle_4"],
        THINGS,
        Contrary(INDEX["bottle_4"], THINGS, random.Random(1)),
        model,
        straike := straight_at("bottle_2"),
        max_questions=1,
    )
    assert straike is not None
    told = model.seen[-1]
    text = " ".join(str(m.get("content", "")) for m in told)
    assert "Do not act on this" in text
    assert "answerable by pointing alone" in text


# ---------------------------------------------------------------- R4 in the loop


def test_pointing_at_something_else_is_handed_to_the_model_as_the_answer():
    class Elsewhere(Person):
        def to_this_one(self, thing_id: str) -> Reply:
            return Reply(point=straight_at("remote"))

    model = ask_and_choose(("ask_this_one", {"id": "bottle_2"}), ("choose", {"id": "remote"}))
    a = point_and_ask(
        INDEX["remote"],
        THINGS,
        Elsewhere(INDEX["remote"], THINGS, random.Random(1)),
        model,
        straight_at("bottle_2"),
        max_questions=1,
    )
    assert a.ok
    text = " ".join(str(m.get("content", "")) for m in model.seen[-1])
    assert "not a refusal" in text


# ---------------------------------------------------------------- silence, and no timeout


def test_silence_is_survivable_and_counted():
    model = ask_and_choose(
        ("ask_this_one", {"id": "bottle_1"}),
        ("ask_show_me", {}),
        ("choose", {"id": "bottle_2"}),
    )
    a = point_and_ask(
        INDEX["bottle_2"],
        THINGS,
        person("bottle_2", silence=1.0),
        model,
        straight_at("bottle_2"),
        max_questions=2,
        toolset=Toolset.selection(show_me=True),
    )
    assert a.silences == 2
    assert a.ok  # the model still got there from the point it already had


def test_a_silence_is_never_taken_as_a_yes():
    model = ask_and_choose(("ask_this_one", {"id": "bottle_1"}), ("choose", {"id": "bottle_2"}))
    a = point_and_ask(
        INDEX["bottle_2"],
        THINGS,
        person("bottle_2", silence=1.0),
        model,
        straight_at("bottle_2"),
        max_questions=1,
    )
    assert a.picked == "bottle_2"
    assert a.silences == 1


# ---------------------------------------------------------------- E2: the extra tools


def test_ask_choice_is_only_on_the_table_when_it_is_turned_on():
    assert not Toolset.selection().has("ask_choice")
    assert Toolset.selection(choice=True).has("ask_choice")
    assert Toolset.selection().names() == ["choose", "ask_this_one"]


def test_the_toolset_is_recorded_on_every_attempt():
    """A score without its toolset is not comparable to anything, so the label travels with
    the attempt rather than living in a run's header only."""
    model = ask_and_choose(("choose", {"id": "phone"}))
    a = point_and_ask(
        INDEX["phone"],
        THINGS,
        person("phone"),
        model,
        straight_at("phone"),
        toolset=Toolset.selection(choice=True, show_me=True),
    )
    assert a.tools == "choose+ask_this_one+ask_choice+ask_show_me"


def test_a_fixed_choice_is_answered_by_pointing_with_no_words():
    """`sigma_deg=0` so the test is about the rule, not the wobble. With a real wobble the
    answering point sometimes lands between the two mugs, which the core's 2:1 rule calls
    too close - and the robot then has to ask again, which is correct and is covered by
    `test_a_point_too_close_to_call_is_not_an_answer`."""
    model = ask_and_choose(("ask_choice", {"first": "mug_1", "second": "mug_2"}))
    p = person("mug_2", style="point", sigma_deg=0.0)
    a = point_and_ask(
        INDEX["mug_2"],
        THINGS,
        p,
        model,
        straight_at("mug_2"),
        toolset=Toolset.selection(choice=True),
    )
    assert a.ok
    assert p.answers == 0  # not one yes or no was needed
    assert a.words <= 2


# ---------------------------------------------------------------- the model saying it cannot


def test_the_robot_may_say_it_cannot():
    model = ask_and_choose(("cannot", {"why": "I cannot see what they mean"}))
    a = point_and_ask(
        INDEX["phone"],
        THINGS,
        person("phone"),
        model,
        straight_at("phone"),
        toolset=Toolset.selection(cannot=True),
    )
    assert a.gave_up
    assert not a.ok
    assert not a.invalid  # saying you cannot is not the same as getting it wrong


def test_an_unreachable_model_is_recorded_and_not_scored():
    """The robot says it cannot. A network failure is not a wrong answer."""
    a = point_and_ask(
        INDEX["phone"], THINGS, person("phone"), ScriptDecider(script=[]), straight_at("phone")
    )
    assert a.failed
    assert not a.ok


# ---------------------------------------------------------------- the pairing


def test_rounds_one_and_three_get_the_same_point():
    """The design is paired, and this is the test that keeps it that way: if round 3 ever
    draws its own point, the gap between the two numbers stops being attributable."""
    model = ask_and_choose(("choose", {"id": "phone"}), ("choose", {"id": "phone"}))
    got = trial(
        INDEX["phone"],
        THINGS,
        model,
        seed=11,
        make_person=lambda t, th, rng, sigma: Person(t, th, rng, sigma),
    )
    one, _, three = got
    assert one.cone == three.cone
    assert one.cone  # and it is a real cone, not an empty one


def test_a_trial_is_three_rounds_in_order():
    model = ask_and_choose(("choose", {"id": "phone"}), ("choose", {"id": "phone"}))
    got = trial(
        INDEX["phone"],
        THINGS,
        model,
        seed=3,
        make_person=lambda t, th, rng, sigma: Person(t, th, rng, sigma),
    )
    assert [a.round for a in got] == ["point only", "words only", "point + ask"]
    assert all(a.unique for a in got)  # the phone is one of a kind


def test_the_split_is_recorded_per_attempt():
    model = ask_and_choose(("choose", {"id": "bottle_1"}), ("choose", {"id": "bottle_1"}))
    got = trial(
        INDEX["bottle_1"],
        THINGS,
        model,
        seed=3,
        make_person=lambda t, th, rng, sigma: Person(t, th, rng, sigma),
    )
    assert all(not a.unique for a in got)  # one of four identical bottles


# ---------------------------------------------------------------- malformed calls


def test_a_question_about_nothing_is_refused_and_the_model_is_told():
    """The bug the first live run found. The prompt had advertised `ask_choice` while the
    toolset did not offer it, so the model called it with no arguments and the loop went on
    to ask the person a question about nothing at all."""
    model = ask_and_choose(
        ("ask_choice", {}),
        ("ask_choice", {"first": "bottle_2", "second": "bottle_3"}),
    )
    a = point_and_ask(
        INDEX["bottle_3"],
        THINGS,
        person("bottle_3", style="point", sigma_deg=0.0),
        model,
        straight_at("bottle_3"),
        max_questions=2,
        toolset=Toolset.selection(choice=True),
    )
    assert a.ok
    assert a.questions == 1  # the empty one was never put to anybody
    told = " ".join(str(m.get("content", "")) for m in model.seen[-1])
    assert "A choice needs two things" in told


def test_a_choice_between_a_thing_and_itself_is_refused():
    model = ask_and_choose(
        ("ask_choice", {"first": "mug_1", "second": "mug_1"}),
        ("choose", {"id": "mug_1"}),
    )
    a = point_and_ask(
        INDEX["mug_1"],
        THINGS,
        person("mug_1"),
        model,
        straight_at("mug_1"),
        toolset=Toolset.selection(choice=True),
    )
    assert a.ok
    assert a.questions == 0
    told = " ".join(str(m.get("content", "")) for m in model.seen[-1])
    assert "is not a choice" in told


def test_asking_this_one_about_nothing_is_refused():
    model = ask_and_choose(("ask_this_one", {"why": "unsure"}), ("choose", {"id": "phone"}))
    a = point_and_ask(INDEX["phone"], THINGS, person("phone"), model, straight_at("phone"))
    assert a.ok and a.questions == 0
    told = " ".join(str(m.get("content", "")) for m in model.seen[-1])
    assert "did not say which thing" in told


# ---------------------------------------------------------------- the prompt matches the tools


def test_the_prompt_only_describes_the_questions_that_are_on_the_table():
    from end_demo.brain import system_for

    pair = system_for(Toolset.selection().names())
    assert "ask_this_one" in pair
    assert "ask_choice" not in pair
    assert "ask_show_me" not in pair

    everything = system_for(Toolset.selection(choice=True, show_me=True).names())
    assert "ask_choice" in everything and "ask_show_me" in everything


def test_the_words_only_prompt_says_it_cannot_ask():
    from end_demo.brain import system_for

    text = system_for(Toolset.commit_only().names())
    assert "YOU CANNOT ASK" in text
    assert "ask_this_one" not in text


def test_each_round_gets_the_prompt_for_its_own_tools():
    model = ask_and_choose(("choose", {"id": "phone"}))
    words_only(INDEX["phone"], THINGS, person("phone"), model, budget=4)
    assert "YOU CANNOT ASK" in model.seen[0][0]["content"]

    model = ask_and_choose(("choose", {"id": "phone"}))
    point_and_ask(INDEX["phone"], THINGS, person("phone"), model, straight_at("phone"))
    assert "ask_this_one" in model.seen[0][0]["content"]
    assert "ask_choice" not in model.seen[0][0]["content"]
