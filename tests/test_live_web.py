"""The whole server with the real Nemotron (opt-in: costs about $0.01 a run).

    uv run pytest -m network tests/test_live_web.py -s

Needs NEBIUS_API_KEY in .env. Every other app test answers with a fake model, so this is the
one place that checks what the judges will see: the real key, the real core over MCP, and
Nemotron deciding every command. The demo's promise is that the model decides every time
(rahul, 8 Oct), so each command here must come back `source: "nemotron"`; a command answered
by the word matcher fails the test and prints why the model didn't answer.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from test_agent import click, post
from test_arm_core import BLOCKS, CENTRES, HERE
from webdemo.app import create_app
from webdemo.settings import ROOT, Settings

pytestmark = pytest.mark.network

AWAY = (330.0, 20.0)  # a second empty spot on the mat


@pytest.fixture(scope="module")
def live_api(core, tmp_path_factory):
    """The app as it is hosted, with our real key: Nemotron, the core over MCP, no speech."""
    s = Settings(web_dist=tmp_path_factory.mktemp("no-dist") / "none", deixis_fallback="none",
                 asr_enabled=False, burst_s=0,
                 command_log=tmp_path_factory.mktemp("log") / "commands.jsonl")  # fmt: skip
    if not s.nebius_api_key:
        pytest.skip("NEBIUS_API_KEY is not set")
    assert s.ai == "nemotron", "AI is turned off in .env (AI=rules)"
    with TestClient(create_app(s, client=core)) as c:
        yield c


def ask(api, text: str, *, holds=(), question=None, previewing=False, t=None):
    """One typed command, as the page sends it, and the reply - from Nemotron or nothing."""
    meta = {"text": text, "things": BLOCKS, "holds": list(holds),
            "state": {"previewing": previewing, "question": question}}  # fmt: skip
    if t is not None:
        meta["t"] = t
    body = post(api, meta)
    agent = body["agent"]
    assert agent["source"] == "nemotron", (
        f"{text!r} was answered by {agent['source']}, not Nemotron: {agent['errors']}"
    )
    assert agent["model"] == "nvidia/Nemotron-3_5-Lightning"
    assert body["limits"]["mode"] == "ours" and body["limits"]["banner"] is None
    assert body["error"] is None, body["error"]
    return body


def names(body) -> list[str]:
    return [c["name"] for c in body["agent"]["calls"]]


def ok(body) -> dict:
    assert body["result"] and body["result"]["status"] == "ok", body["result"]
    return body["result"]["calls"][0]["args"]


# --- a whole conversation, every turn through Nemotron ---


def test_point_at_a_block_and_a_spot(live_api):
    body = ask(live_api, "put that here", t=12.0,
               holds=[click(10, *CENTRES["red"]), click(11.5, *HERE)])  # fmt: skip
    assert names(body) == ["move"]
    args = ok(body)
    assert args["object"]["thing"] == "red"
    assert args["place"]["mean_mm"] == pytest.approx(list(HERE), abs=0.5)


def test_yes_goes_to_the_model_too(live_api):
    body = ask(live_api, "yes", previewing=True, t=20.0)
    assert names(body) == ["confirm"]


def test_no_goes_to_the_model_too(live_api):
    assert names(ask(live_api, "no", previewing=True, t=21.0)) == ["cancel"]


def test_stop_goes_to_the_model_too(live_api):
    assert names(ask(live_api, "stop", t=22.0)) == ["stop"]


def test_go_home(live_api):
    assert names(ask(live_api, "go home", t=23.0)) == ["home"]


def test_a_block_by_its_colour(live_api):
    body = ask(live_api, "put the blue block here", t=31.0, holds=[click(30, *HERE)])
    assert names(body) == ["move"]
    args = ok(body)
    assert args["object"]["thing"] == "blue"
    assert args["place"]["mean_mm"] == pytest.approx(list(HERE), abs=0.5)


@pytest.mark.parametrize(
    ("text", "about"),
    [
        ("put that on the blue cube", "on"),
        ("put that next to the blue one", "next_to"),
        ("put that to the left of the blue one", "left_of"),
        ("put that behind the green one", "behind"),
    ],
)
def test_a_place_beside_another_block(live_api, text, about):
    """A relation to a named block: the core works out the spot, so only `ok` can judge it."""
    body = ask(live_api, text, t=41.0, holds=[click(40, *CENTRES["red"])])
    assert names(body) == ["move"], body["agent"]
    args = ok(body)
    assert args["object"]["thing"] == "red"
    assert args["place"]["mean_mm"], about


def test_between_two_blocks(live_api):
    body = ask(live_api, "put the blue one between the red one and the green one", t=50.0)
    assert names(body) == ["move"]
    args = ok(body)
    assert args["object"]["thing"] == "blue"
    mid = (CENTRES["red"][0] + CENTRES["green"][0]) / 2
    assert args["place"]["mean_mm"][0] == pytest.approx(mid, abs=25)


def test_it_means_the_block_just_moved(live_api):
    body = ask(live_api, "put it there", t=61.0, holds=[click(60, *AWAY)],
               previewing=False)  # fmt: skip
    assert names(body) == ["move"]
    assert body["agent"]["calls"][0]["args"]["object"] == {"last": True}


@pytest.mark.parametrize(
    ("text", "want"),
    [("move it a bit left", {"dx": -1}), ("a lot further away", {"dy": 3})],
)
def test_nudges(live_api, text, want):
    body = ask(live_api, text, previewing=True, t=70.0)
    assert names(body) == ["adjust"]
    args = body["agent"]["calls"][0]["args"]
    for k, v in want.items():
        assert args[k] == v


def test_a_phrase_the_word_matcher_misses(live_api):
    """ "pop that one onto the red one": the word matcher has no "pop", so this only works
    because Nemotron answered. The reply carries both, for the details panel (item 18)."""
    body = ask(live_api, "pop that one onto the red one", t=81.0,
               holds=[click(80, *CENTRES["blue"])])  # fmt: skip
    assert names(body) == ["move"]
    assert ok(body)["object"]["thing"] == "blue"
    assert body["word_rules"]["understood"] is False
    assert body["word_rules"]["same"] is False


def test_nonsense_is_the_models_question_not_a_guess(live_api):
    """Nemotron must still answer (ask_user), never hand the words to the word matcher."""
    body = ask(live_api, "banana the sideways wednesday", t=110.0)
    assert names(body) in (["ask_user"], []), body["agent"]
    assert body["agent"]["source"] == "nemotron"


def test_the_key_is_never_in_a_reply_or_the_log(live_api):
    key = Settings().nebius_api_key.get_secret_value()
    body = ask(live_api, "stop", t=120.0)
    assert key not in str(body)
    log = ROOT / "logs" / "commands.jsonl"
    if log.exists():
        assert key not in log.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("how", "by"),
    [
        # Named: the words say which block, so the model may choose it itself.
        ("the green one", "choice"),
        ("the green block", "choice"),
        # Only pointed: the words cannot say which, so the model must hand it to the core.
        ("this one", "word_index"),
        ("that one", "word_index"),
    ],
)
def test_nemotron_picks_the_answer(live_api, how, by):
    """The `pick` tool end to end with the real model: the robot asks which block, and
    Nemotron answers the question itself (`source` stays "nemotron" - no words are read on
    the server). It must also pick the right *kind* of answer: a named block is the model's
    to choose, a pointed one is the core's, from the finger."""
    between = ((CENTRES["red"][0] + CENTRES["green"][0]) / 2, CENTRES["red"][1])
    first = ask(live_api, "put that here", t=131.0,
                holds=[click(130, *between), click(130.8, *HERE)])  # fmt: skip
    assert first["result"]["status"] == "ask", first["result"]
    things = [c["thing"] for c in first["result"]["ask"]["candidates"]]
    assert set(things) == {"red", "green"}
    second = ask(live_api, how, t=141.0, question=first["say"],
                 holds=[click(140, *CENTRES["green"])])  # fmt: skip
    pick = second["agent"]["pick"]
    assert pick and pick[by] is not None, (
        f"{how!r}: answered with {pick}, expected {by}; "
        f"calls {[c['name'] for c in second['agent']['calls']]}"
    )
    if by == "choice":
        assert pick["choice"] == things.index("green")
    res = second["result"]
    assert res["status"] == "ok" and res["calls"][0]["args"]["object"]["thing"] == "green"
    # The place from the first command is kept: the answer only filled in the block.
    assert res["calls"][0]["args"]["place"]["mean_mm"] == pytest.approx(list(HERE), abs=0.5)


def test_a_new_command_while_a_question_is_open(live_api):
    """Not every reply to a question is an answer: a whole new command must still be one."""
    between = ((CENTRES["red"][0] + CENTRES["green"][0]) / 2, CENTRES["red"][1])
    first = ask(live_api, "put that here", t=151.0,
                holds=[click(150, *between), click(150.8, *HERE)])  # fmt: skip
    assert first["result"]["status"] == "ask", first["result"]
    second = ask(live_api, "no, stop", t=161.0, question=first["say"])
    assert names(second) == ["stop"], second["agent"]
