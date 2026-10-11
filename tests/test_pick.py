"""The `pick` tool (rahul, 11 Oct): while the robot has a question open, Nemotron answers it
itself with a tool call, instead of the server reading the person's words.

Nemotron here is a fake client (as in `test_agent.py`); the real model answering a real
question is `test_live_web.py::test_nemotron_picks_the_answer`.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from test_agent import PROFILE, TOOLS, FakeLLM, click, post, that
from test_arm_core import BLOCKS, CENTRES, HERE
from webdemo.agent import PICK, ArmAgent, Context, Pick, pick_tool, take_pick
from webdemo.app import create_app
from webdemo.command import NOT_UNDERSTOOD, choices_for

BETWEEN = ((CENTRES["red"][0] + CENTRES["green"][0]) / 2, CENTRES["red"][1])
# A move the core cannot settle: the finger was between two blocks. One reply, as FakeLLM
# takes them (a list of calls), so the first command below really is Nemotron's.
AMBIGUOUS = [("move", {"object": that(1), "place": that(2, "here")})]


def app_with(settings, core, llm) -> TestClient:
    settings.nebius_api_key = SecretStr("sk-ours-0000000000")
    return TestClient(create_app(settings, client=core, llm_factory=llm.factory))


def ask_first(c) -> dict:
    """One command that the core can't settle: it asks which block was meant."""
    body = post(c, {"text": "put that here", "things": BLOCKS,
                    "holds": [click(10, *BETWEEN), click(11.5, *HERE)]})  # fmt: skip
    assert body["agent"]["source"] == "nemotron", body["agent"]["errors"]
    assert body["result"]["status"] == "ask" and body["say"] == "This one or that one?"
    return body


# --- the pieces ---


def test_choices_are_the_blocks_the_page_glows():
    ask = {"candidates": [{"thing": "red"}, {"thing": "green"}], "arg": "object"}
    assert choices_for(ask) == ["the red one", "the green one"]


@pytest.mark.parametrize(
    "ask",
    [
        {"candidates": [{"thing": "red"}]},  # one candidate: nothing to choose between
        {"candidates": [{"mean_mm": [1, 2]}, {"mean_mm": [3, 4]}]},  # bare spots
        {"candidates": [{"thing": "red"}, {"mean_mm": [3, 4]}]},  # mixed
        {},
    ],
)
def test_no_choices_when_there_is_nothing_to_name(ask):
    assert choices_for(ask) == []


def test_the_tool_lists_the_choices_and_the_pointing_words():
    tool = pick_tool(["the red one", "the green one"], ["put", "this", "one", "here"])
    props = tool["function"]["parameters"]["properties"]
    assert tool["function"]["name"] == PICK
    assert props["choice"]["enum"] == [0, 1]
    # Only "this": "here" is about a spot, not a thing, and the others are not pointing words.
    assert props["word_index"]["enum"] == [1]


def test_the_context_shows_the_choices_by_number():
    lines = Context(question="The red one or the green one?",
                    choices=["the red one", "the green one"]).lines()  # fmt: skip
    assert lines[-1] == "Choices: 0: the red one; 1: the green one"


CHOICES = ["the red one", "the green one"]
WORDS = ["this", "one"]
ONE_OF = "pick: give exactly one of choice (they named it) or word_index (they pointed)"


@pytest.mark.parametrize(
    ("args", "pick", "why"),
    [
        ({"choice": 1}, Pick(choice=1), ""),
        ({"choice": 0}, Pick(choice=0), ""),
        ({"word_index": 0}, Pick(word_index=0), ""),
        ({"choice": 2}, None, "pick.choice: must be a Choice's number, 0 to 1"),
        ({"choice": -1}, None, "pick.choice: must be a Choice's number, 0 to 1"),
        ({"choice": "green"}, None, "pick.choice: must be a Choice's number, 0 to 1"),
        ({"choice": True}, None, "pick.choice: must be a Choice's number, 0 to 1"),
        ({"word_index": 9}, None, "pick.word_index: must be a word's number, 0 to 1"),
        # "one" is not a pointing word: the model may not call it one.
        (
            {"word_index": 1},
            None,
            "pick.word_index: word 1 is 'one', not one of: that these this those",
        ),
        ({}, None, ONE_OF),
        ({"choice": 0, "word_index": 0}, None, ONE_OF),  # both: which did they do?
    ],
)
def test_take_pick_checks_the_arguments(args, pick, why):
    raw = [{"name": PICK, "args": json.dumps(args)}]
    rest, got, got_why = take_pick(raw, CHOICES, WORDS)
    assert rest == [] and got == pick and got_why == why


def test_bad_json_is_an_error():
    rest, got, why = take_pick([{"name": PICK, "args": "{nope"}], CHOICES, WORDS)
    assert rest == [] and got is None and why == "pick: the arguments were not valid JSON"


def test_two_picks_are_an_error():
    raw = [{"name": PICK, "args": '{"choice": 0}'}, {"name": PICK, "args": '{"choice": 1}'}]
    rest, got, why = take_pick(raw, CHOICES, WORDS)
    assert rest == [] and got is None and why == "pick: answer once, not several times"


def test_a_pick_beside_a_real_call_is_kept_apart():
    raw = [{"name": PICK, "args": '{"choice": 1}'}, {"name": "stop", "args": "{}"}]
    rest, got, why = take_pick(raw, CHOICES, WORDS)
    assert rest == [{"name": "stop", "args": "{}"}] and got == Pick(choice=1) and why == ""


def test_the_tool_is_offered_only_while_a_question_is_open():
    llm = FakeLLM([("confirm", {})], [("confirm", {})])
    agent = ArmAgent(PROFILE, TOOLS, "https://example.invalid/v1/", client_factory=llm.factory)
    agent.run(["yes"], "sk-ours", Context())
    agent.run(["the", "green", "one"], "sk-ours", Context(choices=["the red one", "the green one"]))
    plain, with_pick = (r["tools"] for r in llm.requests)
    assert PICK not in {t["function"]["name"] for t in plain}
    assert PICK in {t["function"]["name"] for t in with_pick}


# --- end to end over MCP ---


def test_nemotron_answers_the_open_question_itself(settings, core):
    """The model picks a choice; the core resolves the first request again with it. The
    answer stays Nemotron's (`source`), because the model made the call."""
    llm = FakeLLM(AMBIGUOUS, [(PICK, {"choice": 0})])
    with app_with(settings, core, llm) as c:
        first = ask_first(c)
        want = first["result"]["ask"]["candidates"][0]["thing"]
        second = post(c, {"text": "that one", "things": BLOCKS, "t": 21.0,
                          "holds": [click(20, *CENTRES[want])],
                          "state": {"question": first["say"]}})  # fmt: skip
    assert second["agent"]["source"] == "nemotron"  # not "answer": the model decided
    assert second["agent"]["pick"] == {"choice": 0, "word_index": None}
    assert second["agent"]["calls"] == []
    res = second["result"]
    assert res["status"] == "ok" and res["calls"][0]["args"]["object"]["thing"] == want
    # The place from the first command is kept: the answer only filled in the block.
    assert res["calls"][0]["args"]["place"]["mean_mm"] == pytest.approx(list(HERE), abs=0.5)


def test_the_model_sees_the_question_and_the_choices(settings, core):
    llm = FakeLLM(AMBIGUOUS, [(PICK, {"choice": 1})])
    with app_with(settings, core, llm) as c:
        first = ask_first(c)
        post(c, {"text": "the green one", "things": BLOCKS, "t": 21.0,
                 "state": {"question": first["say"]}})  # fmt: skip
    said = llm.requests[1]["messages"][1]["content"]
    assert 'The robot asked: "This one or that one?"' in said
    assert "Choices: 0: the " in said and "1: the " in said
    assert PICK in {t["function"]["name"] for t in llm.requests[1]["tools"]}


def test_a_bad_choice_is_retried(settings, core):
    """A choice outside the list is an error the model is told about, as any bad call is."""
    llm = FakeLLM(AMBIGUOUS, [(PICK, {"choice": 7})], [(PICK, {"choice": 1})])
    with app_with(settings, core, llm) as c:
        first = ask_first(c)
        second = post(c, {"text": "that one", "things": BLOCKS, "t": 21.0,
                          "state": {"question": first["say"]}})  # fmt: skip
    assert second["agent"]["attempts"] == 2
    assert "pick.choice" in second["agent"]["errors"][0]
    assert second["agent"]["pick"]["choice"] == 1 and second["result"]["status"] == "ok"
    retry = llm.requests[2]["messages"][-1]["content"]
    assert "pick.choice" in retry and "pick({" in retry


def test_a_new_command_beats_a_stray_pick(settings, core):
    """If the model both picks and gives a command, the command wins: a stale pick must
    never quietly act on the old question."""
    llm = FakeLLM(AMBIGUOUS, [(PICK, {"choice": 0}), ("stop", {})])
    with app_with(settings, core, llm) as c:
        first = ask_first(c)
        second = post(c, {"text": "no stop", "things": BLOCKS, "t": 21.0,
                          "state": {"question": first["say"]}})  # fmt: skip
    assert [x["name"] for x in second["agent"]["calls"]] == ["stop"]
    assert "the command wins" in " ".join(second["notes"])


def test_the_words_are_still_read_when_the_model_gives_no_call(settings, core):
    """Nemotron down, or it answered with nothing: the server reads the words as before
    (`source: "answer"`), so a question can always still be answered."""
    llm = FakeLLM(AMBIGUOUS, "")
    with app_with(settings, core, llm) as c:
        first = ask_first(c)
        second = post(c, {"text": "the green one", "things": BLOCKS, "t": 21.0,
                          "holds": [click(20, *CENTRES["green"])],
                          "state": {"question": first["say"]}})  # fmt: skip
    assert second["agent"]["source"] == "answer" and second["agent"]["pick"] is None
    assert second["result"]["calls"][0]["args"]["object"]["thing"] == "green"


def test_a_pointed_answer_is_the_cores_work_not_the_models(settings, core):
    """pick(word_index=N): the model says only "they pointed while saying this word". Which
    block that is comes from the finger, through the core - so pointing at the green block
    picks green, and pointing at neither picks nothing."""
    llm = FakeLLM(AMBIGUOUS, [(PICK, {"word_index": 0})])
    with app_with(settings, core, llm) as c:
        first = ask_first(c)
        second = post(c, {"text": "this one", "things": BLOCKS, "t": 21.0,
                          "holds": [click(20, *CENTRES["green"])],
                          "state": {"question": first["say"]}})  # fmt: skip
    assert second["agent"]["source"] == "nemotron"
    assert second["agent"]["pick"] == {"choice": None, "word_index": 0}
    assert second["result"]["calls"][0]["args"]["object"]["thing"] == "green"


def test_pointing_at_neither_choice_answers_nothing(settings, core):
    """And the question stays open: a bad answer must not throw away what the robot asked,
    so the person can simply answer it again."""
    llm = FakeLLM(AMBIGUOUS, [(PICK, {"word_index": 0})], [(PICK, {"choice": 1})])
    with app_with(settings, core, llm) as c:
        first = ask_first(c)
        want = first["result"]["ask"]["candidates"][1]["thing"]
        second = post(c, {"text": "this one", "things": BLOCKS, "t": 21.0,
                          "holds": [click(20, 430.0, 250.0)],  # a far corner, no block
                          "state": {"question": first["say"]}})  # fmt: skip
        assert "pointed at no choice" in " ".join(second["notes"])
        assert second["result"] is None and second["say"] == NOT_UNDERSTOOD
        again = post(c, {"text": "the green one", "things": BLOCKS, "t": 31.0,
                         "state": {"question": first["say"]}})  # fmt: skip
    assert again["result"]["calls"][0]["args"]["object"]["thing"] == want


def test_the_pick_is_logged(settings, core):
    llm = FakeLLM(AMBIGUOUS, [(PICK, {"choice": 0})])
    with app_with(settings, core, llm) as c:
        first = ask_first(c)
        post(c, {"text": "that one", "things": BLOCKS, "t": 21.0,
                 "state": {"question": first["say"]}})  # fmt: skip
    rows = [json.loads(x) for x in settings.command_log.read_text().splitlines()]
    assert rows[0]["pick"] is None and rows[1]["pick"] == {"choice": 0, "word_index": None}
