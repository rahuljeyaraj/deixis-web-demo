"""The workbench's geometry, and the committed scene that is generated from it.

The world exists twice over: once as numbers in `workbench/scene.py` that the core, the robot
code and the page work from, and once as bodies in `web/model/workbench_scene.xml` that MuJoCo
simulates. If those two drift, the page draws a laser spot on one shelf while the arm reaches
into another, and nothing visible says so. The XML is generated; these keep the committed copy
honest, and keep the world inside what stage 0 measured.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

from workbench.scene import (
    BACK_ROW_Y_MM,
    BAY_CLEAR_X_MM,
    BAY_CLEAR_Y_MM,
    BAY_GRIP_Z_MM,
    BAY_TALLEST_FLAT_MM,
    BAYS,
    BENCH_THICK_MM,
    BOARD_Z_MM,
    EYE_MM,
    FRONT_ROW_Y_MM,
    GRIP_WIDEST_MM,
    RAIL_FROM_MM,
    RAIL_GUIDE_MM,
    RAIL_GUIDE_Y_MM,
    RAIL_SUNK_MM,
    RAIL_TO_MM,
    RAIL_WIDE_MM,
    RAIL_Y_MM,
    RING_MM,
    SHELF_FROM_MM,
    SHELF_SIDE_MM,
    SHELF_TO_MM,
    arm,
    behind_from_the_eye,
    by_id,
    clashes,
    corners,
    kind_counts,
    things,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "make_workbench_scene.py"
SCENE = ROOT / "web" / "model" / "workbench_scene.xml"
REACH = ROOT / "web" / "model" / "workbench_reach.json"


def test_the_committed_scene_is_not_stale():
    got = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--quiet"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert got.returncode == 0, got.stderr or got.stdout


def test_every_thing_is_in_the_scene_file():
    text = SCENE.read_text(encoding="utf-8")
    for t in things():
        assert f'<body name="{t.id}"' in text, t.id
        assert f'<freejoint name="{t.id}"/>' in text, t.id


def test_the_keyframe_has_a_number_for_every_joint():
    """Six per arm and seven per free joint, or the whole bench is laid out somewhere else.

    This is not a style point. A keyframe two numbers short does not fail to load: it shifts
    every free joint that follows it, so each thing wears the next thing's position and the
    scene looks plausible and is wrong. It cost an afternoon the first time - every thing in
    the world rendered at the height of the thing before it.
    """
    text = SCENE.read_text(encoding="utf-8")
    qpos = text.split('<key name="home" qpos="')[1].split('"')[0].split()
    joints = 1 + 6  # the carriage's slide, then the arm's six
    assert len(qpos) == joints + 7 * (1 + len(things()))  # one parking body, then the things


def test_the_arm_is_attached_rather_than_copied():
    """`<attach prefix>`, because `<replicate>` is not in this MuJoCo build and a prefixed copy
    of so101.xml would be a file to maintain."""
    text = SCENE.read_text(encoding="utf-8")
    assert text.count('<attach model="so101"') == 1
    assert 'prefix="arm_"' in text


def test_there_is_one_weld():
    text = SCENE.read_text(encoding="utf-8")
    assert text.count("<weld ") == 1
    assert 'name="grip"' in text


# ------------------------------------------------- the world against what stage 0 measured


def test_twenty_one_things():
    """Five fewer than the first pass (rahul, 24 Oct: the world was too crowded). A third tin, a
    second mug, the G clamp, the steel rule and the roll of tape went; nothing an act needs went
    with them, and the bench could not have held them in any case - see `BENCH_PLACES`."""
    assert len(things()) == 21


def test_five_sets_of_things_no_word_separates():
    """The heart of the demo. One set would be a trick; five is what a bench looks like, and it
    means a judge meets the same difficulty at several sizes and on several boards."""
    sets = {k: n for k, n in kind_counts(things()).items() if n > 1}
    assert sets == {
        "jar": 4,
        "tin": 2,
        "glue": 2,
        "screwdriver": 2,
        "offcut": 2,
    }


def test_there_is_one_mug_and_its_handle_is_joined_up():
    """rahul, 24 Oct: "the mug's handle is now 3 floating balls". A torus is not one of MuJoCo's
    shapes, so the handle is an arc of beads - and the beads have to overlap, or it is a row of
    balls floating beside a mug. Each has to be nearer its neighbour than it is wide."""
    mug = by_id(things())["mug"]
    beads = [p for p in mug.parts if p.shape == "sphere"]
    assert len(beads) >= 7
    for a, b in zip(beads, beads[1:], strict=False):
        apart = math.dist(a.at_mm, b.at_mm)
        assert apart < a.size_mm[0], f"{apart:.1f} mm apart, {a.size_mm[0]:.0f} mm wide"
    # Both ends of the arc end inside the mug's wall, which is where a handle joins a mug.
    wall = mug.size_mm[0] / 2
    assert beads[0].at_mm[0] < wall and beads[-1].at_mm[0] < wall


def test_the_pliers_have_two_legs():
    """rahul, 24 Oct: they had three. A thing's first shape stands on its own middle and cannot
    be moved off it, so a first shape that was a leg put a third leg down the middle line
    between the two real ones. The first shape is the pivot now."""
    pliers = by_id(things())["pliers"]
    legs = [p for p in pliers.parts if p.colour == "red"]
    assert len(legs) == 2
    assert pliers.colour != "red", "the pivot is steel; a red first shape would be a third leg"
    assert legs[0].at_mm[1] == -legs[1].at_mm[1], "the two legs sit either side of the pivot"


def test_nothing_stands_on_the_rail():
    """It is a moving part, the carriage sweeps the strip in front of it, and the trough is a
    hole in the bench top - a thing standing on it would stand in mid-air. Measured on the
    turned footprint's real corners, not on a width about its middle."""
    for t in things():
        if t.in_bay is not None:
            continue
        near = min(c[1] for c in corners(t))
        assert near > RAIL_Y_MM + RAIL_WIDE_MM / 2, f"{t.id} is over the trough"


def test_nothing_stands_where_something_else_stands():
    """A bench is crowded and this is easy to get wrong by hand: the first full layout had
    nineteen pairs of things inside one another. The places were solved against this test."""
    assert clashes() == []


def test_a_things_footprint_covers_every_shape_it_is_built_from():
    """The reason things stood inside each other (rahul, 24 Oct).

    Each thing used to carry a footprint typed out by hand, and every one of those was too
    small - the screwdriver's said 236 mm for a tool 300 mm long. So the layout was told there
    was room where there was none. `Thing.box_mm` counts it from the shapes instead, and this is
    the test that it really covers them: every corner of every part, turned as the part is
    turned, has to fall inside the footprint the thing claims.
    """
    for t in things():
        mid_x, mid_y, across, deep = t.box_mm
        for n, p in enumerate(t.parts):
            turn = math.radians(p.turn_deg)
            tilt = math.radians(p.tilt_deg)
            half_across = abs(p.size_mm[0] / 2 * math.cos(tilt)) + abs(
                p.size_mm[2] / 2 * math.sin(tilt)
            )
            for dx in (-half_across, half_across):
                for dy in (-p.size_mm[1] / 2, p.size_mm[1] / 2):
                    x = p.at_mm[0] + dx * math.cos(turn) - dy * math.sin(turn)
                    y = p.at_mm[1] + dx * math.sin(turn) + dy * math.cos(turn)
                    assert abs(x - mid_x) <= across / 2 + 1e-6, f"{t.id} part {n + 1} across"
                    assert abs(y - mid_y) <= deep / 2 + 1e-6, f"{t.id} part {n + 1} deep"


def test_nothing_in_a_bay_pokes_into_the_unit_it_stands_in():
    """A thing in a bay has solid matter on three sides and overhead - two uprights, the back
    panel, the board above - and a part through any of them is what "things are going inside
    other things" looks like. The front is the one side that is open, and it is tested
    separately below, because a rim a few millimetres over the edge is what a shelf looks like.
    """
    for t in things():
        if t.in_bay is None:
            continue
        xs = [c[0] for c in corners(t)]
        ys = [c[1] for c in corners(t)]
        assert min(xs) > BAY_CLEAR_X_MM[0], f"{t.id} is through the left upright"
        assert max(xs) < BAY_CLEAR_X_MM[1], f"{t.id} is through the right upright"
        assert max(ys) < BAY_CLEAR_Y_MM[1], f"{t.id} is through the back panel"
        assert t.rises_mm[1] < BOARD_Z_MM[t.in_bay][0], f"{t.id} is through the board above it"
        assert t.rises_mm[0] >= BAYS[t.in_bay][0] - 1e-6, f"{t.id} is through its own floor"


#: How far a thing in a bay may stand over the open front edge of its board, millimetres.
#:
#: Not nothing, and the measured numbers are the reason. The front row stands 35 mm back from the
#: board's front edge; a tin of varnish is 38 mm to its rim, a mug 40 mm to its lip, and the lid
#: of the box of odds and ends - turned 7 degrees, as everything here is turned - reaches 44 mm.
#: So those three sit 3, 5 and 9 mm over the edge, which is what a shelf of things looks like and
#: cannot be seen at the size this page is looked at. What this rules out is a thing perched on
#: the edge of a board.
OVERHANG_MM = 12.0


def _reaches_forward(t) -> float:
    """How far forward the shapes that rest on the board reach, from the thing's own middle.

    Not the box round the whole thing: a mug's handle throws the corner of that box 18 mm
    forward over thin air, and a bay's front is open, so air is allowed. A round shape standing
    up keeps its own depth however the thing is turned; anything else gets the box round its
    turned self, the same way `corners` does.
    """
    a = math.radians(t.yaw_deg)

    def half_of(size, upright_round):
        if upright_round:
            return size[1] / 2
        return abs(size[0] / 2 * math.sin(a)) + abs(size[1] / 2 * math.cos(a))

    half = half_of(t.size_mm, t.shape in ("cylinder", "sphere") and t.axis == "up")
    for p in t.parts:
        if p.at_mm[0] or p.at_mm[1]:
            continue  # off to one side: a handle, a claw, not what stands on the board
        half = max(half, half_of(p.size_mm, p.shape in ("cylinder", "sphere") and p.axis == "up"))
    return half


def test_nothing_in_a_bay_is_perched_on_the_front_edge():
    for t in things():
        if t.in_bay is None:
            continue
        near = t.place_mm[1] - _reaches_forward(t)
        over = BAY_CLEAR_Y_MM[0] - near
        assert over < OVERHANG_MM, f"{t.id} stands {over:.0f} mm over the front edge"


def test_nothing_on_the_bench_is_sunk_into_it():
    """The bench top is the floor of this world and a part that dips below it is a part you
    cannot see. The roll of tape had a hub a millimetre too tall and sat 1 mm into the bench."""
    for t in things():
        if t.in_bay is None:
            assert t.rises_mm[0] >= -1e-6, f"{t.id} is sunk into the bench"


def test_everything_has_somewhere_on_the_rail_to_be_worked_on_from():
    """The gate on the layout. Every ring here was swept at the height the thing is gripped at,
    so this is arithmetic over measurements rather than a guess."""
    a = arm()
    for t in things():
        x, y, _ = t.place_mm
        assert a.slide_for(x, y, t.grip_how, t.grip_z_mm) is not None, t.id


def test_nothing_is_wider_than_the_gripper_can_hold():
    """Measured on 24 Oct: 76 mm on the jaws and friction alone, 96 with the grip assist
    (`out/stage0-grip-width.txt`). The narrower figure is the one used, because it does not
    depend on the weld."""
    for t in things():
        if t.graspable:
            assert t.grip_mm <= GRIP_WIDEST_MM, f"{t.id} is {t.grip_mm} mm across"


def test_nothing_in_a_bay_is_too_flat_to_grip():
    """A thing in a bay has a board over it, so the hand has to come in level - and the jaws
    need the side of it to close on, clear of the board it stands on. A flat thing in a bay
    cannot be picked up at all, which is why the tin of blades and the chisel are on the
    bench."""
    for t in things():
        if t.in_bay is not None:
            assert t.tall_mm >= BAY_TALLEST_FLAT_MM, f"{t.id} is {t.tall_mm} mm tall"


def test_nothing_in_a_bay_is_too_tall_for_it():
    for t in things():
        bay = t.in_bay
        if bay is None:
            continue
        top = t.at_mm[2] + t.tall_mm
        for part in t.parts:
            top = max(top, t.place_mm[2] + part.at_mm[2] + part.size_mm[2] / 2)
        assert top <= BAYS[bay][1], f"{t.id} reaches {top} mm in a bay {BAYS[bay][1]} mm high"


def test_the_pot_can_be_dropped_into_even_though_it_is_never_picked_up():
    pot = by_id(things())["pot"]
    assert not pot.graspable
    x, y, _ = pot.place_mm
    assert arm().reaches(x, y, pot.grip_how, pot.grip_z_mm)


def test_every_ring_a_thing_uses_was_actually_swept():
    """Asking for a ring that was never measured is the quiet way a world stops being honest.
    Reaching down above 100 mm, in particular, has no ring at all: the sweep found nothing."""
    for t in things():
        assert (t.grip_how, t.grip_z_mm) in RING_MM, t.id


def test_nothing_stands_on_the_top_board():
    """It is a lid, not a place: reaching down onto 366 mm is not a pose this arm has, with or
    without tilt (`out/stage0-bay-ring.txt`)."""
    lid = BAYS[-1][1]
    for t in things():
        assert t.at_mm[2] < lid - 1.0, f"{t.id} stands on the top board"


# ------------------------------------------------------------------- the one asymmetry


def test_the_fourth_jar_stands_behind_the_second_as_seen_from_the_default_view():
    """Act 4 is that the robot has not been told about it and the judge finds it by moving the
    camera. Straight behind across is not the same as straight behind from where a person
    stands: put at the same x it leaned out from behind the jar in front of it and 39 per cent
    of it could be seen."""
    index = by_id(things())
    jar2, jar4 = index["jar2"], index["jar4"]
    want = behind_from_the_eye(jar2.at_mm[0], jar2.at_mm[1], jar4.at_mm[1])
    assert jar4.at_mm[0] == pytest.approx(want)
    assert jar4.at_mm[1] > jar2.at_mm[1]
    # And on the far side of the eye's line from the jar in front, not the near side. Both
    # terms are nothing here, because jar2 stands on the middle line now and the eye is on it
    # too: that is the one case where straight behind across and straight behind as seen from
    # somewhere are the same answer, and the line is still what puts the jar there.
    assert (jar4.at_mm[0] - jar2.at_mm[0]) * (jar2.at_mm[0] - EYE_MM[0]) >= 0


def test_the_four_jars_are_the_same_jar_four_times_over():
    counts = kind_counts(things())
    assert counts["jar"] == 4
    index = by_id(things())
    first = index["jar1"]
    for n in (2, 3, 4):
        other = index[f"jar{n}"]
        assert (other.size_mm, other.colour, other.what) == (
            first.size_mm,
            first.colour,
            first.what,
        )


def test_the_keyframe_puts_every_thing_where_its_body_is_declared():
    """The other half of the keyframe trap, and the half that does not fail loudly.

    The qpos list is read in **declaration order**. Get the order wrong and nothing errors: each
    thing simply wears the next thing's position and the whole bench comes out somewhere else.
    That has happened twice here - once from a short vector, once from a different order - so
    both halves are tested.
    """
    text = SCENE.read_text(encoding="utf-8")
    qpos = text.split('<key name="home" qpos="')[1].split('"')[0].split()
    declared = re.findall(r'<body name="([a-z0-9_]+)" pos="([^"]+)"', text)
    # Past the carriage's slide and the arm's six joints.
    at = 7
    for name, pos in declared:
        if name == "carriage":
            continue
        want = [float(v) for v in pos.split()]
        got = [float(v) for v in qpos[at : at + 3]]
        assert got == pytest.approx(want, abs=1e-4), f"{name} is not where the keyframe puts it"
        at += 7
    assert at == len(qpos)


def test_the_two_tubes_of_glue_differ_by_a_state_and_not_by_a_word():
    index = by_id(things())
    one, two = index["glue1"], index["glue2"]
    assert one.kind == two.kind
    assert one.size_mm == two.size_mm
    assert one.what != two.what  # capped, open


# -------------------------------------------------------------------- where things stand


def test_there_are_no_hooks():
    """Scrapped (rahul, 24 Oct): a thing is put on the bench or on a shelf, and nothing else."""
    text = SCENE.read_text(encoding="utf-8")
    assert "hook" not in text.lower()


def test_the_shelf_is_the_whole_back_edge():
    """rahul, 24 Oct. It was 440 mm in the middle of a 700 mm bench, which left a strip at each
    end that nothing could use: further from the person than the shelf, and reached over
    nothing. The rail puts every part of every bay in reach, so there is no reason to stop
    short."""
    from workbench.scene import BENCH_FROM_MM, BENCH_TO_MM

    assert SHELF_FROM_MM[0] == BENCH_FROM_MM[0]
    assert SHELF_TO_MM[0] == BENCH_TO_MM[0]
    assert SHELF_TO_MM[1] == BENCH_TO_MM[1]
    assert BAY_CLEAR_X_MM == (SHELF_FROM_MM[0] + SHELF_SIDE_MM, SHELF_TO_MM[0] - SHELF_SIDE_MM)


def test_the_rail_is_a_trough_cut_out_of_the_bench_top_and_not_a_slab_inside_it():
    """rahul, 24 Oct: "the rail is not visible". It was a box buried in the bench top with only
    its top face showing, and that face lay in exactly the same plane as the bench's own - two
    coplanar faces, so which one is drawn is up to the graphics card, and what came out was no
    rail. The bench top is two slabs now with the trough's floor 16 mm down between them, which
    gives the rail real sides and somewhere for the guide rails to lie.
    """
    text = SCENE.read_text(encoding="utf-8")
    assert '<geom name="bench_near"' in text
    assert '<geom name="bench_far"' in text
    assert '<geom name="bench"' not in text
    assert '<geom name="rail_guide_near"' in text
    assert '<geom name="rail_guide_far"' in text
    # The trough's floor is below the bench top, so no two faces are in the same plane.
    assert RAIL_SUNK_MM < BENCH_THICK_MM
    # And the carriage runs between the guide rails rather than over them.
    assert RAIL_GUIDE_Y_MM + RAIL_GUIDE_MM / 2 < RAIL_WIDE_MM / 2


def test_three_bays():
    assert len(BAYS) == 3
    assert [round(f) for f, _ in BAYS] == [0, 122, 244]


def test_one_arm_on_a_rail_across_the_front():
    """Two arms standing still left 40 per cent of the bottom bay's back row out of reach of
    either: an arm cannot work closer in than 135 mm from itself nor more than 100 degrees off
    the way it faces, and no placing makes those dead zones go away. Sliding one arm does."""
    a = arm()
    assert a.facing_deg == 0.0  # square to the shelf: it slides rather than turns
    assert a.rail_y_mm == RAIL_Y_MM
    assert a.rail_from_mm < 100 and a.rail_to_mm > 600


def test_the_rail_can_stand_the_arm_in_front_of_every_bay():
    """The point of a rail: the bearing can always be nothing, so only how far back it runs
    matters. Every bay's whole width is reachable, which two fixed arms could not manage."""
    a = arm()
    for z in BAY_GRIP_Z_MM:
        for x in range(int(BAY_CLEAR_X_MM[0]), int(BAY_CLEAR_X_MM[1]) + 1, 20):
            assert a.reaches(x, FRONT_ROW_Y_MM, "level", z), f"({x}, front row) at z {z}"


def test_the_back_row_is_reachable_in_the_two_lower_bays_but_not_the_top_one():
    """A finding, kept as a test so it cannot be forgotten.

    Reaching level is a different ring at every height, and up at the top bay it is 90 to **430**
    mm - the outer edge having shrunk as the inner one collapsed. The back row is 435 mm from
    the rail, which is five millimetres too far. So the top bay is a front-row-only bay, and
    nothing is put behind in it. The two below reach the back row comfortably.
    """
    a = arm()
    for z in BAY_GRIP_Z_MM[:2]:
        for x in range(int(BAY_CLEAR_X_MM[0]), int(BAY_CLEAR_X_MM[1]) + 1, 20):
            assert a.reaches(x, BACK_ROW_Y_MM, "level", z), f"({x}, back row) at z {z}"
    assert not a.reaches(350, BACK_ROW_Y_MM, "level", BAY_GRIP_Z_MM[2])
    for t in things():
        if t.in_bay == 2:
            assert t.at_mm[1] == FRONT_ROW_Y_MM, f"{t.id} is in the top bay's back row"


def test_the_rail_is_sunk_so_the_measured_rings_still_hold():
    """Every ring was measured with the arm's base on the bench at z = 0. A rail standing proud
    would raise the arm and put all of them out by its own height."""
    text = SCENE.read_text(encoding="utf-8")
    rail = re.search(r'<geom name="rail"[^>]*pos="[-\d.]+ [-\d.]+ ([-\d.]+)"', text)
    assert rail and float(rail.group(1)) < 0
    assert '<body name="carriage" pos="' in text
    assert 'type="slide"' in text


def test_the_slide_has_a_motor_and_comes_first():
    """so101.xml brings six position actuators; the slide is the seventh and must be first in
    the list, because the page sets every control straight from the keyframe by index."""
    text = SCENE.read_text(encoding="utf-8")
    assert '<position name="slide" joint="slide"' in text
    assert text.index("<actuator>") < text.index("<equality>")


def test_everything_stands_on_the_bench_or_in_the_unit():
    from workbench.scene import BENCH_FROM_MM, BENCH_TO_MM

    for t in things():
        x, y, _ = t.place_mm
        assert BENCH_FROM_MM[0] <= x <= BENCH_TO_MM[0], f"{t.id} is off the bench across"
        assert BENCH_FROM_MM[1] <= y <= BENCH_TO_MM[1], f"{t.id} is off the bench in depth"


def test_nothing_on_the_bench_stands_inside_the_shelf_unit():
    """The unit runs the whole width of the bench now, so there is no end to slip past: every
    loose thing has to keep its whole turned footprint in front of the unit's front face."""
    for t in things():
        if t.in_bay is not None:
            continue
        far = max(c[1] for c in corners(t))
        assert far < SHELF_FROM_MM[1], f"{t.id} is inside the unit"


def test_the_reach_json_is_written_for_the_page_and_the_probes():
    import json

    out = json.loads(REACH.read_text(encoding="utf-8"))
    assert len(out["things"]) == 21
    assert out["eye"] == list(EYE_MM)
    assert out["rail"]["y"] == RAIL_Y_MM
    assert out["rail"]["from"] == RAIL_FROM_MM and out["rail"]["to"] == RAIL_TO_MM
