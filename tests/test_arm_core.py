"""The six arm sentences of the core's C18 (deixis tests/test_arm_mcp.py), on this demo's
profile `so101-table` (plan W7): a fingertip seen through the page's camera, raycast onto the
blocks and the table as the page does, words, 30 mm blocks on the SO-101's reach mat. Out
come places on the table (and a height for "on that") or questions. All over MCP, no fallback;
each answer equals the core's in-process answer. No change to the core.
"""

from __future__ import annotations

import math

import pytest

from deixis.api import resolve
from deixis.profile import load_profile
from view import VIEW, Finger, View
from webdemo.settings import ROOT

PROFILE = load_profile(ROOT / "profiles" / "so101-table.json")
# On the mat (140 to 280 mm from the base at (250, 250)); red and green side by side.
CENTRES = {"red": (180.0, 100.0), "green": (225.0, 100.0), "blue": (300.0, 90.0)}
HERE = (330.0, 50.0)  # an empty spot on the mat, 213 mm from the arm's base


def block(name: str, t: float, centre: tuple[float, float] | None = None, z: float = 0) -> dict:
    """A block as the page sends it (plan 6.2): footprint, height, colour.

    `one` is in the attributes because "the red one" now names two things, the block and the
    red square: the core takes an exact match over a word-by-word one, so an exact "red one"
    on the block keeps the demo's oldest phrase meaning the block (web/src/flow/things.ts)."""
    cx, cy = centre or CENTRES[name]
    return {"id": name, "label": f"{name} block", "box_mm": [cx - 15, cy - 15, 30, 30],
            "height_mm": 30 + z, "t": t,
            "attrs": {"color": name, "one": f"{name} one", "also": "box cube"}}  # fmt: skip


BLOCKS = [block(n, t) for t, n in enumerate(CENTRES, start=1)]

# The three painted squares (advanced-commands-plan.md 3.1), as the page reports them
# (web/src/sim/table.ts, web/src/flow/things.ts). Flat, and never selectable: a square is a
# place to put a block in, not a thing to pick up. `test_plan.py` checks these numbers still
# match the page's.
SQUARE_MM = 70.0
SQUARES = {"red": (79.3, 127.7), "green": (250.0, 40.0), "blue": (420.7, 127.7)}


def square(colour: str) -> dict:
    cx, cy = SQUARES[colour]
    h = SQUARE_MM / 2
    return {"id": f"{colour}-square", "label": f"{colour} square", "height_mm": 0, "t": 0,
            "box_mm": [cx - h, cy - h, SQUARE_MM, SQUARE_MM], "selectable": False,
            "attrs": {"also": "patch area zone"}}  # fmt: skip


PAINT = [square(c) for c in SQUARES]
#: The whole table as the page now sends it: the blocks and the paint.
SCENE = [*BLOCKS, *PAINT]


def spot(x: float, y: float, t: float = 9.0) -> dict:
    """The spot a block was moved away from, as the page reports it while a plan runs."""
    return {"id": "left-spot", "label": "the spot it left", "height_mm": 0, "t": t,
            "box_mm": [x - 15, y - 15, 30, 30], "selectable": False,
            "attrs": {"also": "gap"}}  # fmt: skip


def words(*timed: tuple[str, float]) -> list[dict]:
    return [{"text": w, "t0": round(t, 3), "t1": round(t + 0.25, 3)} for w, t in timed]


def call(name: str, **args) -> dict:
    return {"name": name, "args": args}


def ask(core, **req):
    req = {"things": BLOCKS, **req}
    r = core.resolve(req)
    assert core.over_mcp
    assert r.model_dump_json() == resolve(req, PROFILE).model_dump_json()
    return r


def test_the_python_view_is_the_page_view():
    """This file's raycasts equal the page's real ones (web/tests/fixtures → tests/fixtures)."""
    view = View()
    blocks = {b["id"]: tuple(b["center_mm"]) for b in VIEW["blocks"]}
    for c in VIEW["cases"]:
        h = view.hit(*c["canvas"], blocks)
        if c["hit"] is None:
            assert h is None, c
            continue
        assert h is not None and h.block == c["hit"]["block"], c
        assert math.dist((h.x, h.y), (c["hit"]["x"], c["hit"]["y"])) < 0.3, c
    for name, (cx, cy) in blocks.items():
        u, v = view.project((cx, cy, 30))
        assert view.hit(u, v, blocks).block == name  # its top face, seen from the seat


def test_put_that_here(core):
    f = Finger(CENTRES).at_block("red").hold(0.7)
    t_that = f.t - 0.4
    f.move(HERE).hold(0.7)
    t_here = f.t - 0.4
    r = ask(core, samples=f.samples,
            words=words(("put", t_that - 0.25), ("that", t_that), ("here", t_here)),
            calls=[call("move", object={"deixis": "that", "word_index": 1},
                        place={"deixis": "here", "word_index": 2})])  # fmt: skip
    c = r.calls[0]
    assert c.where("object").thing == "red"
    place = c.where("place")
    assert place.frame == "table" and place.box_mm[2:] == (30, 30)  # the red block's size
    assert math.dist(place.mean_mm, HERE) < 3


def test_put_the_blue_one_next_to_that(core):
    f = Finger(CENTRES).at_block("green").hold(0.7)
    t_that = f.t - 0.4
    w = words(("put", t_that - 1.5), ("the", t_that - 1.25), ("blue", t_that - 1.0),
              ("one", t_that - 0.75), ("next", t_that - 0.5), ("to", t_that - 0.25),
              ("that", t_that))  # fmt: skip
    r = ask(core, samples=f.samples, words=w,
            calls=[call("move", object={"label": "blue one"},
                        place={"relation": "next_to",
                               "target": {"deixis": "that", "word_index": 6}})])  # fmt: skip
    c = r.calls[0]
    assert c.where("object").thing == "blue"
    place = c.where("place")
    assert place.evidence["target"] == "green"
    x, y, bw, bh = place.box_mm
    assert x >= 240 + 10 and (bw, bh) == (30, 30)  # to the viewer's right of green, 10 mm gap
    assert abs((y + bh / 2) - 100) < 1


def test_put_that_on_that_stacks_it(core):
    f = Finger(CENTRES).at_block("red").hold(0.7)
    t1 = f.t - 0.4
    f.at_block("blue").hold(0.7)
    t2 = f.t - 0.4
    r = ask(core, samples=f.samples,
            words=words(("put", t1 - 0.25), ("that", t1), ("on", t2 - 0.25), ("that", t2)),
            calls=[call("move", object={"deixis": "that", "word_index": 1},
                        place={"relation": "on",
                               "target": {"deixis": "that", "word_index": 3}})])  # fmt: skip
    assert r.calls[0].where("object").thing == "red"
    place = r.calls[0].where("place")
    assert place.z_mm == 30 and math.dist(place.mean_mm, (300, 90)) < 1  # on the blue block


def test_a_bit_left_is_one_step(core):
    prev = {"version": "1.0", "frame": "table", "kind": "box", "mean_mm": list(HERE),
            "box_mm": [HERE[0] - 15, HERE[1] - 15, 30.0, 30.0], "source": "gesture"}  # fmt: skip
    r = ask(core, words=words(("a", 8.0), ("bit", 8.25), ("left", 8.5)),
            calls=[call("adjust", dx=-1)], context={"previous": {"place": prev}})  # fmt: skip
    assert r.calls[0].where("place").mean_mm == (HERE[0] - 20, HERE[1])


def test_outside_the_area_asks_and_never_moves_it(core):
    f = Finger(CENTRES).at_block("red").hold(0.7)
    t_that = f.t - 0.4
    f.move((540.0, 120.0)).hold(0.7)  # on the table, past the work area's right edge
    t_here = f.t - 0.4
    r = ask(core, samples=f.samples,
            words=words(("put", t_that - 0.25), ("that", t_that), ("here", t_here)),
            calls=[call("move", object={"deixis": "that", "word_index": 1},
                        place={"deixis": "here", "word_index": 2})])  # fmt: skip
    assert r.status == "ask" and r.ask.reason == "outside_area"
    assert r.ask.question == "I can't reach there. Please move the dot closer."
    assert math.dist(r.ask.candidates[0].mean_mm, (540, 120)) < 5  # as pointed, not moved


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_between_two_blocks_asks_which_then_the_answer_resolves(core, seed):
    f = Finger(CENTRES, seed=seed).move((202.5, 120.0)).hold(0.7)  # between red and green
    t_that = f.t - 0.4
    f.move(HERE).hold(0.7)
    t_here = f.t - 0.4
    req = dict(samples=f.samples,
               words=words(("put", t_that - 0.25), ("that", t_that), ("here", t_here)),
               calls=[call("move", object={"deixis": "that", "word_index": 1},
                           place={"deixis": "here", "word_index": 2})])  # fmt: skip
    first = ask(core, **req)
    assert first.status == "ask" and first.ask.reason == "ambiguous_object"
    assert {c.thing for c in first.ask.candidates} == {"red", "green"}
    assert first.ask.question == "This one or that one?"
    pick = [c.thing for c in first.ask.candidates].index("green")  # the person clicks it
    back = {"call": 0, "arg": "object", "candidate": pick, "thing": "green"}
    second = ask(core, **req, answers=[back, *[a.model_dump(mode="json") for a in first.partial]])
    assert second.status == "ok" and second.calls[0].where("object").thing == "green"
