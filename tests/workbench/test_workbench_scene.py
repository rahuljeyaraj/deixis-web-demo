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
    BAY_CLEAR_X_MM,
    BAY_TALLEST_FLAT_MM,
    BAYS,
    BOTTOM_BAY_USABLE_X_MM,
    EYE_MM,
    GRIP_WIDEST_MM,
    RING_MM,
    SHELF_FROM_MM,
    arms,
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
    parks = 2  # one parking body per arm
    assert len(qpos) == 6 * len(arms()) + 7 * (parks + len(things()))


def test_both_arms_are_attached_rather_than_copied():
    """`<attach prefix>`, because `<replicate>` is not in this MuJoCo build and a prefixed copy
    of so101.xml would be a file to maintain."""
    text = SCENE.read_text(encoding="utf-8")
    assert text.count('<attach model="so101"') == 2
    for a in arms():
        assert f'prefix="{a.id.split("_")[0]}_"' in text


def test_there_is_one_weld_per_gripper():
    """Two grips means the page cannot use MuJoCo's global equality-disable flag: it would let
    go of both at once."""
    text = SCENE.read_text(encoding="utf-8")
    assert 'name="left_grip"' in text and 'name="right_grip"' in text


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


def test_nothing_stands_where_something_else_stands():
    """A bench is crowded and this is easy to get wrong by hand: the first full layout had
    nineteen pairs of things inside one another. The places were solved against this test."""
    assert clashes() == []


def test_nothing_in_the_bottom_bay_is_outside_what_an_arm_can_work_in():
    """Its ring is 280 to 470 mm, the narrowest in the world, so two arms at the sides cover
    only its middle. The two bays above reach in much closer and are usable right across."""
    for t in things():
        if t.in_bay == 0:
            assert BOTTOM_BAY_USABLE_X_MM[0] <= t.at_mm[0] <= BOTTOM_BAY_USABLE_X_MM[1], t.id


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


def test_everything_is_reachable_by_at_least_one_arm():
    """The gate on the layout. Every ring here was swept at the height the thing is gripped at,
    so this is arithmetic over measurements rather than a guess - and
    `web/tests/probes/visible.probe.test.ts` and stage 2 drive the real solver at them."""
    both = arms()
    for t in things():
        if not t.graspable:
            continue
        who = [a.id for a in both if a.can_reach(t)]
        assert who, f"no arm reaches {t.id}"


def test_the_pot_can_be_dropped_into_even_though_it_is_never_picked_up():
    pot = by_id(things())["pot"]
    assert not pot.graspable
    x, y, _ = pot.place_mm
    assert any(a.reaches(x, y, pot.grip_how, pot.grip_z_mm) for a in arms())


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
    # Past the two arms' joints and the two parking bodies.
    at = 6 * len(arms())
    for name, pos in declared:
        if name.endswith("_stand"):
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


def test_the_arms_stand_at_the_sides_facing_each_other():
    """Facing the shelf from the front put both arms' dead zones on the bench, in front of the
    person. At the edges, facing inward, half of each dead disc and the whole of each dead
    sector fall off it."""
    left, right = arms()
    assert left.facing_deg == -90.0 and right.facing_deg == 90.0
    assert left.base_mm[1] == right.base_mm[1]
    assert left.base_mm[0] < 150 and right.base_mm[0] > 550


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
    assert {a["id"] for a in out["arms"]} == {a.id for a in arms()}
