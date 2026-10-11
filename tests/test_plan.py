"""Plans of several steps, the painted squares, and the wiring trace
(deixis-planning/deixis-web-demo/advanced-commands-plan.md).

One command, one model call, up to four steps. `/api/command` resolves the first step only;
the page asks for each later one with `/api/step` just before the arm runs it, sending the
table as it is at that moment - which is what makes a swap possible and what fixes the stale
scene found live on 11 Oct. A longer plan is refused out loud, an answer (yes, stop) is never
a step of a plan, and anything that is not about the blocks is declined in one line instead
of being handed back as a question.

All over MCP, no fallback. No change to the core: the squares and the spot a block left are
things the robot side reports, as the core's L6 expects.
"""

from __future__ import annotations

import json
import re

import pytest
from pydantic import SecretStr
from tests.wordrules import parse, rules

from test_agent import FakeLLM, agent, click, post, that
from test_arm_core import BLOCKS, CENTRES, HERE, PAINT, SCENE, SQUARE_MM, SQUARES, block, spot
from webdemo.agent import (
    DECLINED,
    MAX_STEPS,
    NOT_FOLLOWED,
    SYSTEM,
    TOO_MANY,
    plan_problems,
    validate,
)
from webdemo.app import create_app
from webdemo.settings import ROOT
from webdemo.trace import asked_for, measured

PROFILE = agent(FakeLLM()).profile


def app_with(settings, core, llm):
    from fastapi.testclient import TestClient

    settings.nebius_api_key = SecretStr("sk-ours-0000000000")
    return TestClient(create_app(settings, client=core, llm_factory=llm.factory))


def move(obj: dict, place: dict) -> tuple[str, dict]:
    return ("move", {"object": obj, "place": place})


def label(name: str) -> dict:
    return {"label": name}


def inside(name: str) -> dict:
    return {"relation": "inside", "target": label(name)}


SPOT = "the spot it left"
#: The swap of command 10: park the red one, then two moves into the spot the last one left.
SWAP = [
    move(label("red one"), {"relation": "next_to", "target": label("green one")}),
    move(label("blue one"), inside(SPOT)),
    move(label("red one"), inside(SPOT)),
]


def step(api, index: int, things: list[dict], context: dict | None = None,
         showing: bool = False) -> dict:  # fmt: skip
    body = {"index": index, "things": things, "context": context or {}, "showing": showing}
    r = api.post("/api/step", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def place_of(body: dict) -> list[float]:
    return body["result"]["calls"][0]["args"]["place"]["mean_mm"]


def object_of(body: dict) -> str:
    return body["result"]["calls"][0]["args"]["object"]["thing"]


# --- the page and the server agree on where the paint is ---


def test_the_squares_and_the_spot_match_the_page():
    """The squares' size and places live in the page (`web/src/sim/table.ts`); the tests and
    the server's own fixtures copy them. If one side moves, this fails rather than the demo."""
    table = (ROOT / "web" / "src" / "sim" / "table.ts").read_text(encoding="utf-8")
    found = re.findall(r"id: '(\w+)-square', color: '\w+', xMm: ([\d.]+), yMm: ([\d.]+)", table)
    assert {c: (float(x), float(y)) for c, x, y in found} == SQUARES
    assert f"export const SQUARE_MM = {SQUARE_MM:.0f};" in table
    things = (ROOT / "web" / "src" / "flow" / "things.ts").read_text(encoding="utf-8")
    assert "one: `${b.id} one`" in things  # keeps "the red one" meaning the block
    assert f"export const SPOT_LABEL = '{SPOT}'" in things


def test_a_block_and_its_square_both_answer_to_the_red_one(core):
    """With paint on the mat, "the red one" names two things. The core takes an exact match
    over a word-by-word one, so the block wins and nothing has to be asked."""
    calls = [{"name": "move", "args": {"object": label("red one"), "place": inside("red square")}}]
    r = core.resolve({"words": [], "things": SCENE, "calls": calls}).model_dump(mode="json")
    assert r["status"] == "ok", r
    assert r["calls"][0]["args"]["object"]["thing"] == "red"
    assert r["calls"][0]["args"]["place"]["mean_mm"] == pytest.approx(list(SQUARES["red"]))
    # And "the red square" is only ever the paint.
    target = r["calls"][0]["args"]["place"]["evidence"]["target"]
    assert target == "red-square"


def test_the_spot_a_block_left_is_a_place(core):
    """The spot a step emptied, by name: this is how one block goes where another one was,
    and the core needs nothing new for it - it is a thing the robot side reports (L6)."""
    things = [*SCENE, spot(*CENTRES["red"])]
    calls = [{"name": "move", "args": {"object": label("blue one"), "place": inside(SPOT)}}]
    r = core.resolve({"words": [], "things": things, "calls": calls}).model_dump(mode="json")
    assert r["status"] == "ok", r
    assert r["calls"][0]["args"]["place"]["mean_mm"] == pytest.approx(list(CENTRES["red"]))


def test_pointing_at_the_paint_finds_no_block(rules_api):
    """Neither a square nor an empty spot can be picked up (`selectable: false`), so "that"
    pointed at bare paint is "I don't see a block there" - never the square itself."""
    body = post(rules_api, {"text": "put that here", "things": SCENE, "t": 12.0,
                            "holds": [click(10, *SQUARES["red"]), click(11, *HERE)]})  # fmt: skip
    assert body["result"]["status"] == "ask"
    assert body["result"]["ask"]["reason"] == "no_object"


# --- a plan arrives whole, and is measured one step at a time ---


def test_only_the_first_step_is_resolved_and_the_rest_come_back_as_a_plan(settings, core):
    llm = FakeLLM(SWAP)
    with app_with(settings, core, llm) as c:
        body = post(c, {"text": "swap the red and the blue", "things": SCENE, "t": 12.0})
    assert body["agent"]["source"] == "nemotron"
    assert body["plan"]["steps"] == 3
    assert [x["name"] for x in body["plan"]["calls"]] == ["move", "move", "move"]
    # Only step 1 went to the core: steps 2 and 3 name a spot that does not exist yet.
    assert len(body["result"]["calls"]) == 1
    assert object_of(body) == "red"


def test_each_step_is_measured_against_the_table_as_it_is_then(settings, core):
    """The swap, step by step, as the page plays it. Step 2 asks for "the spot it left" and
    gets where the red one was; step 3 asks again and gets where the blue one was. Neither
    could be said at all before the step in front of it had happened."""
    llm = FakeLLM(SWAP)
    with app_with(settings, core, llm) as c:
        first = post(c, {"text": "swap the red and the blue", "things": SCENE, "t": 12.0})
        assert first["plan"]["steps"] == 3

        # Step 1 ran: the red one is parked at HERE and left a spot where it was.
        parked = [block("red", 10, HERE), *[b for b in BLOCKS if b["id"] != "red"], *PAINT]
        second = step(c, 1, [*parked, spot(*CENTRES["red"])])
        assert second["index"] == 1 and second["steps"] == 3
        assert object_of(second) == "blue"
        assert place_of(second) == pytest.approx(list(CENTRES["red"]))

        # Step 2 ran: the blue one is where the red one was, and left its own spot behind.
        swapped = [block("red", 10, HERE), block("blue", 11, CENTRES["red"]),
                   *[b for b in BLOCKS if b["id"] == "green"], *PAINT]  # fmt: skip
        third = step(c, 2, [*swapped, spot(*CENTRES["blue"])])
        assert object_of(third) == "red"
        assert place_of(third) == pytest.approx(list(CENTRES["blue"]))
    assert len(llm.requests) == 1  # one model call for the whole swap


def test_the_same_step_asked_twice_can_answer_differently(settings, core):
    """Measuring again is the point: the same call, the same words, a table that has moved
    on, and a different answer. This is the stale-scene bug found live on 11 Oct."""
    llm = FakeLLM([move(label("blue one"), inside(SPOT)), move(label("red one"), inside(SPOT))])
    with app_with(settings, core, llm) as c:
        post(c, {"text": "swap them", "things": SCENE, "t": 12.0})
        a = step(c, 1, [*SCENE, spot(*CENTRES["red"])])
        b = step(c, 1, [*SCENE, spot(*CENTRES["blue"])])
    assert place_of(a) == pytest.approx(list(CENTRES["red"]))
    assert place_of(b) == pytest.approx(list(CENTRES["blue"]))


def test_a_step_needs_a_plan_and_must_be_one_of_its_steps(settings, core):
    llm = FakeLLM([move(that(1), that(2, "here"))])
    with app_with(settings, core, llm) as c:
        assert c.post("/api/step", json={"index": 0, "things": SCENE}).status_code == 409
        post(c, {"text": "put that here", "things": SCENE, "t": 12.0,
                 "holds": [click(10, *CENTRES["red"]), click(11, *HERE)]})  # fmt: skip
        # One call is not a plan, so nothing is kept for the page to ask about.
        assert c.post("/api/step", json={"index": 0, "things": SCENE}).status_code == 409
        post(c, {"text": "swap them", "things": SCENE, "t": 20.0})
    llm2 = FakeLLM(SWAP)
    with app_with(settings, core, llm2) as c:
        post(c, {"text": "swap the red and the blue", "things": SCENE, "t": 12.0})
        assert c.post("/api/step", json={"index": 3, "things": SCENE}).status_code == 400


# --- the caps, and what is never a step ---


@pytest.mark.parametrize("steps", [5, 6])
def test_more_than_four_steps_is_refused_out_loud(settings, core, steps):
    long_plan = [move(label("red one"), inside("red square"))] * steps
    llm = FakeLLM(long_plan, long_plan)  # it says the same thing again on the retry
    with app_with(settings, core, llm) as c:
        body = post(c, {"text": "stack all six into one tower", "things": SCENE, "t": 12.0})
    assert body["say"] == TOO_MANY
    assert body["plan"]["steps"] == 0 and body["result"] is None
    assert body["agent"]["refusal"] == TOO_MANY
    # Refusing is the model's answer: the word matcher is not given a second go at it.
    assert body["agent"]["source"] == "nemotron"
    assert any(f"plan at most {MAX_STEPS}" in e for e in body["agent"]["errors"])


def test_yes_and_stop_are_never_a_step_of_a_plan(settings, core):
    mixed = [move(label("red one"), inside("red square")), ("confirm", {})]
    llm = FakeLLM(mixed, mixed)  # it says the same thing again on the retry
    with app_with(settings, core, llm) as c:
        body = post(c, {"text": "put the red one in the red square and do it", "things": SCENE})
    assert body["say"] == NOT_FOLLOWED and body["plan"]["steps"] == 0


def test_four_steps_are_fine():
    calls, errors = validate(PROFILE, [
        {"name": "move", "args": {"object": label("red one"), "place": inside("red square")}},
    ] * MAX_STEPS, [])  # fmt: skip
    assert errors == [] and plan_problems(calls) == []


# --- places the arm could never reach ---


@pytest.mark.parametrize(("place", "says"), [
    ({"named": "top left corner"}, "top left corner"),
    ({"empty_space": True}, "empty space"),
])  # fmt: skip
def test_a_place_the_arm_cannot_reach_goes_back_for_the_retry(place, says):
    """A named region and the largest empty space are both worked out over the core's whole
    rectangle, while the arm reaches a half ring inside it, so they land off the mat. The
    model is told, and told what to say instead."""
    llm = FakeLLM(
        [move(label("red one"), place)],
        [move(label("red one"), {"relation": "next_to", "target": label("green one")})],
    )
    res = agent(llm).understand(["put", "the", "red", "one", "away"], "sk-ours")
    assert res.attempts == 2 and [c.name for c in res.calls] == ["move"]
    assert any(says in e and "next_to" in e for e in res.errors)


# --- declining, and undo ---


def test_a_question_about_the_world_is_declined_not_handed_back(settings, core):
    """Found live on 11 Oct: "what is the capital of France?" came back as the robot asking
    the person their own question, with multiple-choice options."""
    llm = FakeLLM([("decline", {})])
    with app_with(settings, core, llm) as c:
        body = post(c, {"text": "what is the capital of france", "things": SCENE})
    assert [x["name"] for x in body["agent"]["calls"]] == ["decline"]
    assert body["result"]["status"] == "ok"
    assert body["say"] is None  # the page says the one line, DECLINED
    assert DECLINED == "I only move the blocks on this table."


def test_the_prompt_tells_the_model_to_decline_and_never_to_parrot():
    assert "decline()" in SYSTEM
    assert "Never hand the person" in SYSTEM
    assert "the spot it left" in SYSTEM and "red square" in SYSTEM


@pytest.mark.parametrize("said", ["undo", "undo that", "put that back where it was"])
def test_undo_is_a_tool_call_and_the_word_matcher_knows_it_too(said):
    assert [c["name"] for c in parse(said.split())] == ["undo"]
    res = rules(PROFILE, said.split())
    assert res.errors == [] and [c.name for c in res.calls] == ["undo"]


def test_go_back_still_means_home():
    assert [c["name"] for c in parse(["go", "back"])] == ["home"]


# --- re-planning ---


def test_a_step_that_cannot_be_done_gets_one_more_plan(settings, core):
    """Job 2 of section 1: the arm's own reason goes back to the model, which plans another
    way. Once per command, so a model that keeps failing cannot run away."""
    second = [move(label("green one"), inside("green square"))]
    llm = FakeLLM(SWAP, second)
    with app_with(settings, core, llm) as c:
        post(c, {"text": "swap the red and the blue", "things": SCENE, "t": 12.0})
        r = c.post("/api/replan", json={"index": 1, "why": "I can't reach there.",
                                        "things": SCENE})  # fmt: skip
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["plan"]["steps"] == 1
        assert object_of(body) == "green"
        # The new plan replaced the old one, and that was the one re-plan.
        again = c.post("/api/replan", json={"index": 0, "why": "nope", "things": SCENE})
        assert again.status_code == 409
    # The model was shown its own plan, which step failed, and the arm's words.
    asked = llm.requests[1]["messages"][1]["content"]
    assert "You planned these steps" in asked and "Step 2 could not be done" in asked
    assert "I can't reach there." in asked
    assert '"swap the red and the blue"' in asked  # the words the person really said


def test_a_replan_needs_a_plan(settings, core):
    llm = FakeLLM()
    with app_with(settings, core, llm) as c:
        r = c.post("/api/replan", json={"index": 0, "why": "nope", "things": SCENE})
    assert r.status_code == 409


# --- the wiring trace ---


def test_the_trace_says_what_each_part_said_to_the_next(settings, core):
    llm = FakeLLM(SWAP)
    with app_with(settings, core, llm) as c:
        body = post(c, {"text": "swap the red and the blue", "things": SCENE, "t": 12.0,
                        "holds": [click(10, *CENTRES["red"])]})  # fmt: skip
        rows = body["trace"]
        pairs = [(r["from"], r["to"]) for r in rows]
        assert ("you", "page") in pairs  # you pointed, you typed
        assert ("page", "nemotron") in pairs and ("nemotron", "page") in pairs
        assert ("page", "core") in pairs and ("core", "page") in pairs
        model = next(r for r in rows if r["from"] == "nemotron")
        assert model["say"] == "a plan: 3 tool calls"
        assert model["detail"]["calls"][0]["name"] == "move"
        assert model["ms"] is not None
        # No word matcher anywhere on the panel: the model is the only thing that decides.
        assert not [r for r in rows if r.get("kind") == "shadow"]
        assert not [r for r in rows if "matcher" in r["say"]]
        # No step marker from the server: the page draws one as each step really starts, and
        # this reply only measures the first step to show the plan.
        assert not [r for r in rows if r.get("kind") == "step"]
        # And the core's row says, in words, what it measured.
        measured_row = next(r for r in rows if r["from"] == "core")
        assert "red block" in measured_row["say"]

        # Each later step brings its own rows, marked as a fresh measurement.
        later = step(c, 1, [*SCENE, spot(*CENTRES["red"])])
        says = [r["say"] for r in later["trace"]]
        assert any(s.startswith("again, for real:") for s in says)
        # The same step, measured to draw the plan, says that is what it is for: the panel
        # must not call a prediction a measurement.
        shown = step(c, 1, [*SCENE, spot(*CENTRES["red"])], showing=True)
        assert any(s.startswith("to draw the plan:") for s in [r["say"] for r in shown["trace"]])


def test_one_command_one_step_has_no_step_markers(settings, core):
    llm = FakeLLM([move(that(1), that(2, "here"))])
    with app_with(settings, core, llm) as c:
        body = post(c, {"text": "put that here", "things": SCENE, "t": 12.0,
                        "holds": [click(10, *CENTRES["red"]), click(11, *HERE)]})  # fmt: skip
    assert not [r for r in body["trace"] if r.get("kind") == "step"]
    assert next(r for r in body["trace"] if r["from"] == "nemotron")["say"] == "one tool call: move"


def test_the_trace_says_when_nemotron_was_not_asked(settings, core):
    """Past the day's share, the panel says the model was not asked and why. Nothing
    answers in its place and no second row appears (rahul, 11 Oct)."""
    with app_with(settings, core, FakeLLM()) as c:
        settings.nebius_api_key = None
        body = post(c, {"text": "put that here", "things": SCENE, "t": 12.0,
                        "holds": [click(10, *CENTRES["red"]), click(11, *HERE)]})  # fmt: skip
    back = next(r for r in body["trace"] if r["from"] == "nemotron")
    assert back["say"].startswith("not asked:")
    assert body["agent"]["calls"] == []


def test_the_trace_words_for_references_and_places():
    calls = [{"name": "move", "args": {"object": that(1), "place": inside("red square")}}]
    assert asked_for(calls) == "where are “that” and inside “the red square”?"
    assert asked_for([{"name": "home", "args": {}}]) == "nothing to measure (home)"
    result = {"status": "ok", "calls": [{"name": "move", "args": {
        "place": {"frame": "table", "mean_mm": [330.0, 50.4], "cov_mm2": [[16, 0], [0, 16]]},
    }}]}  # fmt: skip
    assert measured(result) == "(330, 50) mm ± 4 mm"
    # A move's two places, the thing and the spot, with an arrow between: which is which.
    both = {"status": "ok", "calls": [{"name": "move", "args": {
        "object": {"frame": "table", "label": "green block", "mean_mm": [2.0, 186.0]},
        "place": {"frame": "table", "mean_mm": [337.0, 95.0], "z_mm": 30.0},
    }}]}  # fmt: skip
    assert measured(both) == "green block at (2, 186) mm → (337, 95) mm, 30 mm up"
    ask = {"status": "ask", "ask": {"question": "The red one or the green one?"}}
    assert measured(ask) == "it asks: The red one or the green one?"


# --- pointing at the paint ---


def test_pointing_inside_a_square_is_a_spot_on_the_table(rules_api):
    """A painted square is flat: pointing inside one means that spot, not "on top of the
    paint". (A block really is stacked on: `test_arm_core` and `test_agent` cover that.)"""
    on_paint = (SQUARES["blue"][0] + 10, SQUARES["blue"][1] - 8)
    clicks = [click(10, *CENTRES["red"]), click(11, *on_paint)]
    body = post(rules_api, {"text": "put that here", "things": SCENE, "t": 12.0,
                            "holds": clicks})  # fmt: skip
    assert body["notes"] == []
    place = body["result"]["calls"][0]["args"]["place"]
    assert place["z_mm"] is None
    assert place["mean_mm"] == pytest.approx(list(on_paint), abs=0.5)


def test_the_log_counts_the_steps(settings, core):
    llm = FakeLLM(SWAP)
    with app_with(settings, core, llm) as c:
        post(c, {"text": "swap the red and the blue", "things": SCENE, "t": 12.0})
    lines = [json.loads(x) for x in settings.command_log.read_text().splitlines()]
    assert lines[-1]["steps"] == 3 and lines[-1]["refusal"] is None
