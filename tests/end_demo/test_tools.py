"""The tools, and the two rules about them that are not style.

**No coordinates, ever.** The core's standing invariant is that the language model never
writes a number for a place (`deixis/refs.py`). Every place in every tool here is a
reference: a thing's id, a named place, or where the person pointed. A single `x` parameter
would quietly break the invariant the whole project rests on.

**Every question must be answerable without words.** Checked against the tool descriptions
as well as the behaviour, because the descriptions are what the model reads when it decides
which question to ask.
"""

from __future__ import annotations

import json

import pytest

from end_demo.point import RATIO, core_ratio
from end_demo.tools import QUESTIONS, Toolset

ALL = {t["function"]["name"]: t["function"] for t in Toolset.job().as_list()} | {
    t["function"]["name"]: t["function"]
    for t in Toolset.selection(choice=True, show_me=True, cannot=True).as_list()
}

#: Anything that would be a coordinate, a measurement or a raw number for a place.
FORBIDDEN = ("x", "y", "z", "mm", "cm", "metres", "meters", "angle", "degrees", "position")


@pytest.mark.parametrize("name", sorted(ALL))
def test_no_tool_takes_a_coordinate(name):
    params = ALL[name]["parameters"]["properties"]
    for key, spec in params.items():
        assert key.lower() not in FORBIDDEN, f"{name}.{key}"
        assert spec["type"] != "number", f"{name}.{key} is a number"
        assert spec["type"] != "integer", f"{name}.{key} is a number"


def test_where_a_thing_goes_is_a_reference_not_a_place():
    where = ALL["place"]["parameters"]["properties"]["where"]
    assert where["enum"] == ["tray", "shelf"]


@pytest.mark.parametrize("name", sorted(ALL))
def test_every_tool_has_a_description_a_model_can_act_on(name):
    assert len(ALL[name]["description"]) > 40, name


def test_the_arm_is_always_optional():
    """Choosing between two arms that can both do it is kinematics, not judgement. The
    model only has to care when the person said *you*."""
    for name, fn in ALL.items():
        props = fn["parameters"]["properties"]
        if "arm" in props and name != "pass_to":
            assert "arm" not in fn["parameters"]["required"], name


def test_pass_to_needs_to_know_which_arm_takes_it():
    assert ALL["pass_to"]["parameters"]["required"] == ["arm"]


def test_the_questions_say_they_need_no_words():
    """The description is what the model reads when it picks a question, so the fact that
    asking is cheap has to be in the description and not only in our heads."""
    for name in QUESTIONS:
        text = ALL[name]["description"].lower()
        assert "point" in text or "show" in text, name
    assert "no words" in ALL["ask_choice"]["description"].lower()
    assert "costs them no words" in ALL["ask_this_one"]["description"].lower()


def test_the_choice_is_between_exactly_two_things():
    required = ALL["ask_choice"]["parameters"]["required"]
    assert required == ["first", "second"]


def test_show_me_needs_nothing_at_all():
    assert ALL["ask_show_me"]["parameters"]["required"] == []


def test_choose_says_that_being_wrong_is_costly():
    """The whole deal the model is being offered: a question is cheap, a wrong commit is
    not. If that is not in the descriptions, the measurement is of a model that was never
    told the stakes."""
    assert "costly" in ALL["choose"]["description"].lower()


# ---------------------------------------------------------------- the toolsets


def test_the_default_selection_pair_is_what_the_harness_measured():
    assert Toolset.selection().names() == ["choose", "ask_this_one"]
    assert Toolset.selection().label == "choose+ask_this_one"


def test_the_label_travels_with_the_toolset():
    kit = Toolset.selection(choice=True, show_me=True, cannot=True)
    assert kit.label == "choose+ask_this_one+ask_choice+ask_show_me+cannot"


def test_the_final_call_offers_only_choose():
    assert Toolset.commit_only().names() == ["choose"]


def test_a_toolset_hands_out_copies():
    """A tool list handed to the model must not be the one the next call will mutate."""
    kit = Toolset.selection()
    first, second = kit.as_list(), kit.as_list()
    first[0]["function"] = {"name": "tampered"}
    assert second[0]["function"]["name"] == "choose"


def test_every_schema_is_json():
    json.dumps(Toolset.job().as_list())


# ---------------------------------------------------------------- the core's own rule


def test_the_ask_threshold_is_the_cores_and_has_not_drifted():
    """2:1 is not invented here. If the core ever changes its mind about when to stop
    guessing and ask, this is the test that notices."""
    assert RATIO == core_ratio() == 2.0
