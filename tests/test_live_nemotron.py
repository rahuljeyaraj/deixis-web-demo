"""Live Nemotron on the arm's phrases (opt-in: costs about $0.01 a run).

    uv run pytest -m network tests/test_live_nemotron.py -s

Needs NEBIUS_API_KEY in .env. Each phrase must come back from Nemotron itself (not the rule
parser) with the same calls the rule parser gives. The whole server with the real model is
`test_live_web.py`; this file is the agent alone.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr
from tests.wordrules import rules

from test_agent import PHRASES, PROFILE, TOOLS
from webdemo.agent import MAX_STEPS, ArmAgent, Context
from webdemo.settings import Settings

pytestmark = pytest.mark.network

COLOURS = ("red", "green", "blue")


def same_block(call: dict) -> dict:
    """One call with every label cut to its colour: Nemotron writes the words the person said
    ("the blue cube"), the rule parser always writes "blue one", and the core finds the same
    block from either (it matches every word against the label and `attrs`). The difference is
    wording, not meaning, so the comparison below ignores it."""
    if isinstance(call, dict):
        label = call.get("label")
        if isinstance(label, str):
            colour = next((c for c in COLOURS if c in label.lower()), None)
            call = {**call, "label": colour or label}
        return {k: same_block(v) for k, v in call.items()}
    return call


@pytest.fixture(scope="module")
def live():
    """The agent and our key. The key is a `SecretStr`, so a failing test's fixture repr
    can't print it into the output (pytest shows every argument of a failed test)."""
    s = Settings()
    if not s.nebius_api_key:
        pytest.skip("NEBIUS_API_KEY is not set")
    agent = ArmAgent(PROFILE, TOOLS, s.nebius_base_url, model=s.nemotron_model)
    return agent, SecretStr(s.nebius_api_key.get_secret_value())


@pytest.mark.parametrize("text", [p[0] for p in PHRASES])
def test_nemotron_agrees_with_the_plain_reading(live, text):
    """`tests/wordrules.py` says what the plain reading of each phrase is, and the real model
    must come to the same answer. It is a test double for stating the expectation: the demo
    has no word matcher in it (rahul, 11 Oct)."""
    agent, key = live
    words = text.split()
    res = agent.run(words, key.get_secret_value())
    print(f"{text!r}: {[str(c) for c in res.calls]} in {res.latency_s:.2f} s, "
          f"{res.attempts} attempt(s)")  # fmt: skip
    expected = [same_block(c.to_dict()) for c in rules(PROFILE, words).calls]
    assert [same_block(c.to_dict()) for c in res.calls] == expected, res.errors


# The commands that need a plan (advanced-commands-plan.md section 4). The rule parser gives
# nothing for these - that is the point - so what is checked is the shape of the plan: how
# many steps, which actions, and which blocks they move. Not the exact places: those are the
# core's, and the model may word a reference several right ways.
PLANS = [
    ("stack them all up", ["move", "move"], {"green", "blue"}),
    ("swap the red and the blue", ["move", "move", "move"], {"red", "blue"}),
    ("put each block on its own colour", ["move"] * 3, {"red", "green", "blue"}),
    ("everything except the blue one goes in the red square", ["move", "move"], {"red", "green"}),
    ("put the red one on the blue one, then go home", ["move", "home"], {"red"}),
    ("move them all a bit to the left", ["move"] * 3, {"red", "green", "blue"}),
]


def moved(calls) -> set[str]:
    """The colours the plan picks up, from each move's object reference."""
    out = set()
    for c in calls:
        args = c.to_dict()["args"]
        label = (args.get("object") or {}).get("label", "")
        out |= {w for w in ("red", "green", "blue") if w in label.lower()}
    return out


@pytest.mark.parametrize(("text", "actions", "blocks"), PLANS, ids=[p[0] for p in PLANS])
def test_nemotron_plans_the_steps(live, text, actions, blocks):
    agent, key = live
    res = agent.run(text.split(), key.get_secret_value())
    print(f"{text!r}: {[str(c) for c in res.calls]} in {res.latency_s:.2f} s, "
          f"{res.attempts} attempt(s)")  # fmt: skip
    assert res.refusal is None, res.refusal
    assert [c.name for c in res.calls] == actions, res.errors
    assert moved(res.calls) >= blocks, res.errors


# A tower in the order the person asks for (command 4, and rahul on 11 Oct: "stack them with
# green on the bottom and blue on the top" came back as the green one onto the blue one and
# then the blue one onto the green one, each onto the other). The person names the bottom, the
# top, or both; the block they do not name is the middle one, the bottom one never moves, and
# the moves go upwards. Each case below is (what was said, the colours picked up in order, the
# colours they go onto).
TOWER_ORDER = [
    ("stack them with green on the bottom and blue on the top", "red blue", "green red"),
    ("stack them with green at the bottom and red on top", "blue red", "green blue"),
    ("stack them with red on top and blue at the bottom", "green red", "blue green"),
    ("make a tower with red at the bottom and green on top", "blue green", "red blue"),
    ("stack them with the blue one at the bottom", "green red", "blue green"),
    ("build a tower with the red one on top", "green red", "blue green"),
    ("build a tower with the green one on top", "red green", "blue red"),
]


@pytest.mark.parametrize(("text", "picked", "onto"), TOWER_ORDER, ids=[t[0] for t in TOWER_ORDER])
def test_nemotron_builds_a_tower_in_the_order_it_was_asked_for(live, text, picked, onto):
    agent, key = live
    res = agent.run(text.split(), key.get_secret_value())
    print(f"{text!r}: {[str(c) for c in res.calls]} in {res.latency_s:.2f} s")
    assert res.refusal is None, res.refusal
    assert [c.name for c in res.calls] == ["move", "move"], res.errors
    assert [one(c) for c in res.calls] == picked.split(), res.errors
    assert [onto_of(c) for c in res.calls] == onto.split(), res.errors


def onto_of(call) -> str:
    """The colour of the block a move puts its block on top of."""
    place = call.to_dict()["args"].get("place") or {}
    assert place.get("relation") == "on", place
    label = (place.get("target") or {}).get("label", "")
    found = [w for w in COLOURS if w in label.lower()]
    return found[0] if len(found) == 1 else label


#: What the model is told when the three blocks are one tower ("stack them all up").
TOWER = Context(stacks=[["red one", "green one", "blue one"]])

#: Said to a table that is already a tower. Every one of these must be planned from the top
#: of the tower down, or the first step picks up a covered block and the plan is thrown away.
ON_A_TOWER = [
    "put each block in its own square",
    "put each block on its own colour",
]


@pytest.mark.parametrize("text", ON_A_TOWER)
def test_nemotron_takes_a_tower_apart_from_the_top(live, text):
    """Found by rahul on 11 Oct: "stack them all up", then "put each block in its own square",
    and nothing happened at all. The model planned the red one first, and the red one was at
    the bottom of the tower - a step the arm cannot do, which throws the whole plan away. It
    is told what is stacked now (`stacks_in`), so the order is the model's to get right."""
    agent, key = live
    res = agent.run(text.split(), key.get_secret_value(), TOWER)
    print(f"{text!r}: {[str(c) for c in res.calls]} in {res.latency_s:.2f} s, "
          f"{res.attempts} attempt(s)")  # fmt: skip
    assert res.refusal is None, res.refusal
    assert [c.name for c in res.calls] == ["move"] * 3, res.errors
    # Top of the tower first, then the one under it, then the block at the bottom.
    assert [one(c) for c in res.calls] == ["blue", "green", "red"], res.errors


def test_nemotron_replans_a_tower_from_the_top(live):
    """The recovery path, as rahul's log had it on 11 Oct: step 1 picked up a covered block,
    and the one re-plan came back with another covered block (the green one, still under the
    blue one). The re-plan is told what is stacked too, so it has what it needs."""
    agent, key = live
    planned = [
        'move({"object": {"label": "red one"}, "place": {"relation": "inside", "target": {"label": "red square"}}})',  # noqa: E501
        'move({"object": {"label": "green one"}, "place": {"relation": "inside", "target": {"label": "green square"}}})',  # noqa: E501
        'move({"object": {"label": "blue one"}, "place": {"relation": "inside", "target": {"label": "blue square"}}})',  # noqa: E501
    ]
    context = Context(
        stacks=[["red one", "green one", "blue one"]], plan=planned, failed_step=0,
        failed_why="There's a block on top of that one. Move that one first.",
    )  # fmt: skip
    res = agent.run("put each block on its own colour".split(), key.get_secret_value(), context)
    print(f"re-plan: {[str(c) for c in res.calls]} in {res.latency_s:.2f} s")
    assert res.refusal is None, res.refusal
    assert [c.name for c in res.calls] == ["move"] * 3, res.errors
    assert [one(c) for c in res.calls] == ["blue", "green", "red"], res.errors


def one(call) -> str:
    """The colour of the block a move picks up."""
    label = (call.to_dict()["args"].get("object") or {}).get("label", "")
    found = [w for w in COLOURS if w in label.lower()]
    return found[0] if len(found) == 1 else label


@pytest.mark.parametrize("text", [
    "what is the capital of france",
    "pick it up and throw it",
    "tell me a joke",
])  # fmt: skip
def test_nemotron_declines_anything_that_is_not_about_the_blocks(live, text):
    """Found live on 11 Oct: an off-topic question came back as the robot asking the person
    their own question, with multiple-choice options."""
    agent, key = live
    res = agent.run(text.split(), key.get_secret_value())
    print(f"{text!r}: {[str(c) for c in res.calls]}")
    assert [c.name for c in res.calls] == ["decline"], res.errors


def test_a_plan_that_is_too_long_is_refused(live):
    agent, key = live
    res = agent.run("stack all six blocks into one tall tower".split(), key.get_secret_value())
    print(f"steps: {[str(c) for c in res.calls]}, refusal: {res.refusal}")
    # Either it plans within the cap, or it is refused out loud - never attempted badly.
    assert res.refusal is not None or len(res.calls) <= MAX_STEPS
