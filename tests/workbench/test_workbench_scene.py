"""The workbench's geometry, and the committed scene that is generated from it.

The world exists twice over: once as numbers in `workbench/scene.py` that the core, the robot
code and the page work from, and once as bodies in `web/model/workbench_scene.xml` that MuJoCo
simulates. If those two drift, the page draws a laser spot on one shelf while the arm reaches
into another, and nothing visible says so. The XML is generated; these keep the committed copy
honest, and keep the world inside what stage 0 measured.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from workbench.scene import (
    BACK_ROW_Y_MM,
    BAY_CLEAR_X_MM,
    BAY_GRIP_Z_MM,
    BAY_TALLEST_FLAT_MM,
    BAYS,
    EYE_MM,
    FRONT_ROW_Y_MM,
    GRIP_WIDEST_MM,
    RAIL_FROM_MM,
    RAIL_TO_MM,
    RAIL_Y_MM,
    RING_MM,
    SHELF_FROM_MM,
    arm,
    behind_from_the_eye,
    by_id,
    clashes,
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


def test_twenty_six_things():
    assert len(things()) == 26


def test_six_sets_of_things_no_word_separates():
    """The heart of the demo. One set would be a trick; six is what a bench looks like, and it
    means a judge meets the same difficulty at several sizes and on several boards."""
    sets = {k: n for k, n in kind_counts(things()).items() if n > 1}
    assert sets == {
        "jar": 4,
        "tin": 3,
        "mug": 2,
        "glue": 2,
        "screwdriver": 2,
        "offcut": 2,
    }


def test_nothing_stands_on_the_rail():
    """It is a moving part, and the carriage sweeps the strip in front of it."""
    for t in things():
        if t.in_bay is not None:
            continue
        assert t.place_mm[1] - t.room_mm[1] / 2 > RAIL_Y_MM + 20, t.id


def test_nothing_stands_where_something_else_stands():
    """A bench is crowded and this is easy to get wrong by hand: the first full layout had
    nineteen pairs of things inside one another. The places were solved against this test."""
    assert clashes() == []


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
    # And on the far side of the eye's line from the jar in front, not the near side.
    assert (jar4.at_mm[0] - jar2.at_mm[0]) * (jar2.at_mm[0] - EYE_MM[0]) > 0


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
    for t in things():
        if t.in_bay is not None:
            continue
        x, y, _ = t.place_mm
        inside_x = SHELF_FROM_MM[0] < x < BAY_CLEAR_X_MM[1] + 12.0
        assert not (inside_x and y + t.room_mm[1] / 2 > SHELF_FROM_MM[1]), f"{t.id} is in the unit"


def test_the_reach_json_is_written_for_the_page_and_the_probes():
    import json

    out = json.loads(REACH.read_text(encoding="utf-8"))
    assert len(out["things"]) == 26
    assert out["eye"] == list(EYE_MM)
    assert out["rail"]["y"] == RAIL_Y_MM
    assert out["rail"]["from"] == RAIL_FROM_MM and out["rail"]["to"] == RAIL_TO_MM
