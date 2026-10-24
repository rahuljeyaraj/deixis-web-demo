"""Write `web/model/workbench_scene.xml` from the geometry in `workbench/scene.py`.

    python scripts/make_workbench_scene.py            # write it
    python scripts/make_workbench_scene.py --check    # fail if the committed file is stale

**Why generate it** (FR5). The workbench exists twice over: once as numbers the core, the robot
code and the page work from, and once as bodies MuJoCo simulates. If those two are typed out
separately they will disagree, and the disagreement will be invisible - the page would put a
laser spot on one shelf while the arm reached into another. So `workbench/scene.py` is the
single source and this writes the XML from it. The file is committed, because a scene is easier
to read than a generator, and `--check` in the tests keeps the committed copy honest.

Modelled on `scripts/make_shelf_scene.py`, which has done this job for the end demo since
5 Oct; the hard-won parts are carried over with their reasons, not rediscovered.

**The frame.** MuJoCo world: metres, z up, origin on the bench top in the middle of the
person's edge. +X is away from the person, towards the shelf; +Y is to the person's left. The
bench frame the rest of the demo uses is millimetres with x to the person's right and y away
from them, so

    X = y_mm / 1000,    Y = -(x_mm - 350) / 1000

and `bench_to_world` below is the only place that conversion is written. The 350 is this
world's centre line; the end demo's was 250.

**Two arms** come in by `<attach model="so101" prefix="left_"/>`, which this MuJoCo build
supports and `<replicate>` does not (`web/tests/probes/twoarms.probe.test.ts`). The prefix is
applied to the attached model's own root body, which is called `base`, so the wrapper bodies are
named `left_stand` and `right_stand` - `left_base` would collide.

**Two welds and two parking bodies**, one per gripper. `data.eq_active` cannot be touched in
this build (a `memory_view<bool>`; emscripten throws on any access), and MuJoCo's global
equality-disable flag would let go of both grips at once, so both welds stay active and an arm
holding nothing has its weld on an invisible parking body. See
`web/tests/probes/eqactive.probe.test.ts`. Nothing drives them in stage 1; they are here so the
keyframe's qpos order is settled once rather than twice.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workbench.scene import (  # noqa: E402
    BACK_ROW_Y_MM,
    BAYS,
    BENCH_FROM_MM,
    BENCH_THICK_MM,
    BENCH_TO_MM,
    BOARD_Z_MM,
    BOTTOM_BAY_USABLE_X_MM,
    CENTRE_X_MM,
    EYE_MM,
    FRONT_ROW_Y_MM,
    GIVE_MM,
    LOOK_AT_MM,
    RING_MM,
    ROBOT_EYE_MM,
    SHELF_BACK_MM,
    SHELF_FROM_MM,
    SHELF_SIDE_MM,
    SHELF_TO_MM,
    Thing,
    arms,
    clashes,
    reach_report,
    things,
)

OUT = ROOT / "web" / "model" / "workbench_scene.xml"
#: What the arithmetic says each arm can reach, written out beside the scene so that a probe in
#: the browser can drive the real inverse kinematics at every thing and say where the two
#: disagree. The same pattern as `shelf_reach.json`, and for the same reason: the ring is a
#: measurement of an arm on a bare bench, and this world has a shelf unit in the way of it.
REACH_OUT = ROOT / "web" / "model" / "workbench_reach.json"

#: The arms' resting pose, after the shoulder's turn: lift, elbow, wrist flex, wrist roll, and
#: the gripper. Five numbers, and the pan from `REST_PAN_RAD` makes six - which is what the
#: keyframe below needs per arm.
#:
#: Written into the scene as a keyframe, not left to the page, because MuJoCo's default pose has
#: the arm stretched out flat along +X - straight through the shelf. The end demo learned this
#: the hard way: its first physics run had the right arm lying on the remote, which then tipped
#: over and slid 49 mm while the scene settled.
REST_ARM = (-1.0, 1.69, 1.0, 0.0, 0.2)

#: Each arm's shoulder at rest, radians (the joint's limit is 1.92).
#:
#: Both arms face the shelf, and at rest they must be out of the way of it: the demo's whole
#: question is which thing the person means, so the bench and both bays have to be in plain view
#: before anything is asked.
#:
#: **Measured.** `web/tests/probes/visible.probe.test.ts` casts rays at every thing from the
#: default view, for a grid of nine turns and seven ways of folding, and asks how much of the
#: world each pose leaves visible. Turned nearly all the way out - away from the middle of the
#: bench and along its edge - is the best of them, and it is also what a parked arm looks like.
#:
#: With the arms at the sides the pose hardly matters: the sweep runs from 69 to 86 per cent,
#: and most of it is at 86. That is the measurement agreeing with the reason they were moved
#: there. Standing at the front facing the shelf, the same sweep ran from 48 to 71.
REST_PAN_RAD = {"left_arm": -1.8, "right_arm": 1.8}

#: Rough masses, kilograms. Nothing here is heavy; the gripper's weld does the carrying.
MASSES: dict[str, float] = {
    "offcut": 0.180,
    "screwdriver": 0.070,
    "knife": 0.030,
    "blades": 0.040,
    "chisel": 0.120,
    "rule": 0.045,
    "tape": 0.060,
    "hammer": 0.450,
    "pot": 0.200,
    "oil": 0.250,
    "pliers": 0.180,
    "jar": 0.340,
    "glue": 0.050,
    "odds": 0.150,
    "tin": 0.420,
    "mug": 0.320,
    "clamp": 0.260,
}

#: Every colour in the world, and nothing else. NFR1 asks that every difference that matters be
#: visible at a glance at 1366 x 768 - colour, size and silhouette only - so these are chosen to
#: separate at that size and to stay apart in a photograph of a screen.
COLOURS: dict[str, str] = {
    "bench": "0.68 0.52 0.36 1",
    "shelf": "0.86 0.81 0.72 1",
    # Darker than the boards on purpose: with the back the same shade as everything else, a bay
    # read as a flat panel rather than as a space with things standing in it.
    "shelf back": "0.54 0.48 0.40 1",
    "pale wood": "0.86 0.73 0.52 1",
    "wood": "0.60 0.42 0.24 1",
    "pencil": "0.38 0.35 0.32 1",
    "handle blue": "0.13 0.35 0.72 1",
    "red": "0.80 0.15 0.13 1",
    "red dark": "0.58 0.10 0.09 1",
    "steel": "0.74 0.76 0.80 1",
    "grey": "0.50 0.52 0.55 1",
    "beige": "0.90 0.82 0.63 1",
    "beige dark": "0.78 0.69 0.50 1",
    "shadow": "0.17 0.16 0.15 1",
    "dark green": "0.08 0.29 0.17 1",
    "oil label": "0.88 0.84 0.70 1",
    "brown card": "0.66 0.50 0.33 1",
    "white": "0.95 0.95 0.92 1",
    "brass": "0.74 0.57 0.25 1",
    "tin": "0.62 0.66 0.70 1",
    "tin label": "0.20 0.45 0.38 1",
    "mug": "0.92 0.93 0.95 1",
    "jar lid": "0.46 0.48 0.52 1",
    # The jars are the point of the whole demo: four of them that no word separates. Glass, so
    # the screws inside show through and the lid reads as a lid.
    "glass": "0.80 0.88 0.90 0.40",
}

#: The shapes a geom may be, and what MuJoCo calls them.
SHAPES = {
    "box": "box",
    "cylinder": "cylinder",
    "capsule": "capsule",
    "sphere": "sphere",
    "ellipsoid": "ellipsoid",
}


def quat_mul(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    """Two turns, one after the other (w, x, y, z), the way MuJoCo wants them written."""
    return (
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


def quat_about(axis: str, deg: float) -> tuple[float, ...]:
    """A turn of `deg` about one of the **world** axes."""
    half = math.radians(deg) / 2.0
    c, s = math.cos(half), math.sin(half)
    return {"x": (c, s, 0.0, 0.0), "y": (c, 0.0, s, 0.0), "z": (c, 0.0, 0.0, s)}[axis]


def quat_str(q: tuple[float, ...]) -> str:
    return " ".join(f"{v:.6f}" for v in q)


#: The bench frame turned into the world's: world X is bench y, world Y is minus bench x, which
#: is a quarter turn about z. Written once, here, and used for every direction in the scene.
BENCH_TO_WORLD_QUAT = quat_about("z", -90.0)


def bench_to_world(x_mm: float, y_mm: float) -> tuple[float, float]:
    """Bench millimetres to world metres. The only place this conversion is written."""
    return (y_mm / 1000.0, -(x_mm - CENTRE_X_MM) / 1000.0)


def mm(v: float) -> str:
    return f"{v / 1000.0:.4f}"


def shape_geom(
    name: str,
    shape: str,
    size_mm: tuple[float, float, float],
    colour: str,
    axis: str,
    tilt_deg: float,
    turn_deg: float,
    at_mm: tuple[float, float, float] | None,
    mass: str,
) -> str:
    """One geom: its size, its turn and its place, converted from the bench frame to the world.

    The world is a quarter turn from the bench - world X is depth, world Y is across - so a
    **size** has to be converted as well as a position, and not doing it is how the end demo's
    first run laid every long thing across its shelf instead of front to back.

    A cylinder or a capsule runs along its own z, so lying one down takes a quarter turn:
    `across` puts it on the bench's x, `deep` on the bench's y. `tilt_deg` then turns it in the
    across-up plane and `turn_deg` about up, which is what gives a hammer its claw and a pair of
    pliers its splayed legs. Everything is worked out in the bench frame and turned into the
    world's at the end, so there is one conversion and not one per shape.
    """
    across, deep, up = size_mm
    if shape in ("box", "ellipsoid"):
        half = (across / 2, deep / 2, up / 2)
        spin: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    elif shape == "sphere":
        half = (across / 2, 0.0, 0.0)
        spin = (1.0, 0.0, 0.0, 0.0)
    elif axis == "up":
        half = (across / 2, up / 2, 0.0)
        spin = (1.0, 0.0, 0.0, 0.0)
    elif axis == "across":
        half = (up / 2, across / 2, 0.0)
        spin = quat_about("y", 90.0)  # the shape's z onto the bench's x
    else:
        half = (up / 2, deep / 2, 0.0)
        spin = quat_about("x", -90.0)  # the shape's z onto the bench's y
    # In the bench frame: turn about up, then tilt in the across-up plane, then the shape's own.
    bench = quat_mul(quat_about("z", turn_deg), quat_mul(quat_about("y", tilt_deg), spin))
    quat = quat_mul(BENCH_TO_WORLD_QUAT, bench)
    if shape in ("box", "ellipsoid"):
        # A box's half-sizes are in its own axes, and the quaternion above already carries the
        # bench-to-world quarter turn, so these stay in bench order.
        size = f"{mm(half[0])} {mm(half[1])} {mm(half[2])}"
    elif shape == "sphere":
        size = mm(half[0])
    else:
        size = f"{mm(half[0])} {mm(half[1])}"
    where = ""
    if at_mm is not None:
        dx, dy, dz = at_mm
        where = f' pos="{dy / 1000.0:.4f} {-dx / 1000.0:.4f} {mm(dz)}"'
    turned = "" if quat == (1.0, 0.0, 0.0, 0.0) else f' quat="{quat_str(quat)}"'
    return (
        f'<geom name="{name}" type="{SHAPES[shape]}" size="{size}"{where}{turned}'
        f' rgba="{COLOURS[colour]}"{mass} class="thing"/>'
    )


def body(thing: Thing) -> list[str]:
    """One thing, at the centre `scene.py` gives it, turned as it stands, with its parts on.

    A part carries no mass worth the name: the thing's first geom carries all of it, so adding
    a hammer's head cannot quietly change what the hammer weighs.
    """
    x_mm, y_mm, z_mm = thing.place_mm
    X, Y = bench_to_world(x_mm, y_mm)
    yaw = quat_mul(BENCH_TO_WORLD_QUAT, quat_about("z", thing.yaw_deg))
    out = [
        f'    <body name="{thing.id}" pos="{X:.4f} {Y:.4f} {mm(z_mm)}" quat="{quat_str(yaw)}">',
        f'      <freejoint name="{thing.id}"/>',
        "      "
        + shape_geom(
            thing.id,
            thing.shape,
            thing.size_mm,
            thing.colour,
            thing.axis,
            0.0,
            0.0,
            None,
            f' mass="{MASSES[thing.kind]}"',
        ),
    ]
    for n, p in enumerate(thing.parts, start=1):
        out.append(
            "      "
            + shape_geom(
                f"{thing.id}_{n}",
                p.shape,
                p.size_mm,
                p.colour,
                p.axis,
                p.tilt_deg,
                p.turn_deg,
                p.at_mm,
                ' mass="0.0001"',
            )
        )
    out.append("    </body>")
    return out


def slab(name, x0, x1, y0, y1, z0, z1, colour) -> str:
    """A fixed box given by its edges in the bench frame: a board, an upright, the bench top."""
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    X, Y = bench_to_world(cx, cy)
    size = f"{mm((y1 - y0) / 2)} {mm((x1 - x0) / 2)} {mm((z1 - z0) / 2)}"
    return (
        f'    <geom name="{name}" type="box" pos="{X:.4f} {Y:.4f} {mm(cz)}" size="{size}"\n'
        f'      rgba="{COLOURS[colour]}" friction="0.8 0.005 0.0001" group="0"'
        ' contype="3" conaffinity="3"/>'
    )


def shelf_unit() -> list[str]:
    """The three-bay unit: two uprights, a back panel, and three boards.

    The bottom bay's floor is the bench top itself, so there is no board at zero. The top board
    is the unit's lid and nothing stands on it, because nothing can: reaching down onto anything
    above about 100 mm is not a pose this arm has (`out/stage0-bay-ring.txt`).
    """
    x0, y0 = SHELF_FROM_MM
    x1, y1 = SHELF_TO_MM
    top = BOARD_Z_MM[-1][0] + BOARD_Z_MM[-1][1]
    out = [
        "    <!-- The shelf unit: three bays, 440 mm across. The bottom bay is the hard one -",
        "         reaching level into it stops at 280 mm from an arm, so only its middle 320 mm",
        "         is usable; the two above it reach in much closer. -->",
        slab("shelf_left", x0, x0 + SHELF_SIDE_MM, y0, y1, 0.0, top, "shelf"),
        slab("shelf_right", x1 - SHELF_SIDE_MM, x1, y0, y1, 0.0, top, "shelf"),
        slab("shelf_back", x0, x1, y1 - SHELF_BACK_MM, y1, 0.0, top, "shelf back"),
    ]
    for n, (under, thick) in enumerate(BOARD_Z_MM):
        what = (
            "the lid - not a place, nothing can reach it"
            if n == len(BOARD_Z_MM) - 1
            else f"board {n + 1}"
        )
        out.append(f"    <!-- {what} -->")
        out.append(slab(f"shelf_board{n}", x0, x1, y0, y1, under, under + thick, "shelf"))
    return out


def declared(items: list[Thing]) -> list[Thing]:
    """The things in the order their bodies are written into the scene.

    There is exactly one of these, and both the bodies and the keyframe below take it from
    here, because the keyframe lists qpos in **declaration order**. Getting that wrong does not
    fail to load: it quietly gives every thing the next thing's position, and the whole bench
    comes out somewhere else. It has now done that twice.
    """
    return [t for bay in (None, 0, 1, 2) for t in items if t.in_bay == bay]


def _home_qpos(items: list[Thing]) -> str:
    """The whole qpos vector for the resting pose.

    Both arms' joints first, in the order `<attach>` adds them, then every free joint in the
    order its body is declared: the two parking bodies, then the things. Getting that order
    wrong does not fail, it scrambles the scene - in the end demo it moved the remote 38 mm and
    looked like a physics problem.
    """
    out: list[str] = []
    for arm in arms():
        # Six per arm - the five joints and the gripper - and not five. Five put every free
        # joint after them two numbers out, which does not fail: it quietly lays the whole
        # bench out somewhere else, with each thing wearing the next thing's position.
        out += [f"{q:g}" for q in (REST_PAN_RAD[arm.id], *REST_ARM)]
    for arm in arms():
        X, Y = bench_to_world(*arm.base_mm)
        out += [f"{X:.4f}", f"{Y:.4f}", "0.3000", "1", "0", "0", "0"]
    for t in declared(items):
        x_mm, y_mm, z_mm = t.place_mm
        X, Y = bench_to_world(x_mm, y_mm)
        yaw = quat_mul(BENCH_TO_WORLD_QUAT, quat_about("z", t.yaw_deg))
        out += [f"{X:.4f}", f"{Y:.4f}", mm(z_mm), *(f"{v:.6f}" for v in yaw)]
    return " ".join(out)


def xml() -> str:
    items = things()
    left, right = arms()
    lx, ly = bench_to_world(*left.base_mm)
    rx, ry = bench_to_world(*right.base_mm)
    give_x, give_y = bench_to_world(GIVE_MM[0], GIVE_MM[1])
    eye_x, eye_y = bench_to_world(ROBOT_EYE_MM[0], ROBOT_EYE_MM[1])
    bx0, by0 = BENCH_FROM_MM
    bx1, by1 = BENCH_TO_MM

    lines = [
        "<!--",
        "  GENERATED by scripts/make_workbench_scene.py - do not edit by hand.",
        "",
        "  The workbench demo's scene: a bench, a three-bay shelf unit, two SO-101 arms facing",
        "  each other across it, and twenty-six things. The geometry comes from",
        "  workbench/scene.py, which is also what the core, the robot code and the page work",
        "  from, so the shelf the arm reaches into and the shelf the page draws a spot on are",
        "  the same shelf.",
        "",
        "  World: metres, z up, origin on the bench top in the middle of the person's edge.",
        "  +X away from the person towards the shelf, +Y to the person's left.",
        "",
        "  Two arms by <attach prefix>: <replicate> is not in this MuJoCo build",
        "  (web/tests/probes/twoarms.probe.test.ts). so101.xml is unchanged.",
        "",
        "  Two welds and two parking bodies, both welds always active: data.eq_active cannot be",
        "  written in this build, and the global equality-disable flag would let go of both",
        "  grips at once. See web/tests/probes/eqactive.probe.test.ts.",
        "-->",
        '<mujoco model="the workbench: a bench, a shelf and two arms">',
        '  <asset><model name="so101" file="so101.xml"/></asset>',
        "",
        f'  <statistic center="{0.30:.2f} 0 0.12" extent="0.9"/>',
        "",
        "  <visual>",
        '    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35" specular="0 0 0"/>',
        "  </visual>",
        "",
        "  <default>",
        '    <default class="thing">',
        "      <!-- contype/conaffinity 3: things touch the bench, each other and a gripper (1).",
        "           A held thing is switched to 2 so it no longer touches the gripper, and no",
        "           squeeze can push it while it is carried. -->",
        '      <geom friction="1.2 0.01 0.001" condim="4" solref="0.004 1" group="0"',
        '        contype="3" conaffinity="3"/>',
        "    </default>",
        "  </default>",
        "",
        "  <worldbody>",
        '    <light name="key" pos="-0.2 -0.3 1.1" dir="0.3 0.3 -1" directional="true"/>',
        '    <light name="fill" pos="0.6 0.4 0.9" dir="-0.2 -0.3 -1" directional="true"/>',
        "",
        "    <!-- The bench top is z = 0. -->",
        slab("bench", bx0, bx1, by0, by1, -BENCH_THICK_MM, 0.0, "bench"),
        "",
    ]
    lines += shelf_unit()
    lines += [
        "",
        "    <!-- Where a thing is held out to the person, and where the robot's own camera",
        "         looks from. Sites, not bodies: nothing to collide with, somewhere to aim. -->",
        f'    <site name="give" pos="{give_x:.4f} {give_y:.4f} {mm(GIVE_MM[2])}" size="0.012"'
        ' rgba="0.2 0.6 0.3 0.35"/>',
        f'    <site name="robot_eye" pos="{eye_x:.4f} {eye_y:.4f} {mm(ROBOT_EYE_MM[2])}"'
        ' size="0.010" rgba="0.12 0.52 0.90 0.30"/>',
        "",
        "    <!-- The arms, one at each side of the bench, turned to face each other. Unturned,",
        "         forward (shoulder_pan = 0) is +X, which is the shelf; each stand is turned a",
        "         quarter so that the arm's dead disc and its dead sector fall off the bench",
        "         rather than across it. The turn is a quaternion because this file has no",
        "         <compiler>, so `euler` would be degrees here and radians inside so101.xml. -->",
        f'    <body name="left_stand" pos="{lx:.4f} {ly:.4f} 0"'
        f' quat="{quat_str(quat_about("z", left.facing_deg))}">',
        '      <attach model="so101" prefix="left_"/>',
        "    </body>",
        f'    <body name="right_stand" pos="{rx:.4f} {ry:.4f} 0"'
        f' quat="{quat_str(quat_about("z", right.facing_deg))}">',
        '      <attach model="so101" prefix="right_"/>',
        "    </body>",
        "",
        "    <!-- One parking body per arm: where its weld points while it holds nothing.",
        "         Declared after the arms and before the things, because the keyframe lists qpos",
        "         in declaration order. -->",
        f'    <body name="left_park" pos="{lx:.4f} {ly:.4f} 0.30"><freejoint name="left_park"/>',
        '      <geom name="left_park" type="sphere" size="0.002" mass="0.0001"',
        '        contype="0" conaffinity="0" rgba="0 0 0 0"/></body>',
        f'    <body name="right_park" pos="{rx:.4f} {ry:.4f} 0.30"><freejoint name="right_park"/>',
        '      <geom name="right_park" type="sphere" size="0.002" mass="0.0001"',
        '        contype="0" conaffinity="0" rgba="0 0 0 0"/></body>',
        "",
    ]
    where = {
        None: "on the bench top",
        0: "in the bottom bay",
        1: "in the middle bay",
        2: "in the top bay",
    }
    for bay in (None, 0, 1, 2):
        lines.append(f"    <!-- {where[bay]} -->")
        for t in (t for t in declared(items) if t.in_bay == bay):
            lines += body(t)
        lines.append("")
    lines += [
        "  </worldbody>",
        "",
        "  <!-- The arms' resting pose, folded down and turned outwards. MuJoCo's default pose",
        "       lies flat along +X, straight through the shelf. Order: the two arms' joints as",
        "       attached, then every freejoint (7 numbers each: a place and a turn), exactly",
        "       where `declared()` put the bodies, and in that same order. -->",
        "  <keyframe>",
        f'    <key name="home" qpos="{_home_qpos(items)}"/>',
        "  </keyframe>",
        "",
        "  <!-- One weld per gripper, both always active, each starting on its own parking body.",
        "       Nothing drives them until stage 2; see the module docstring. -->",
        "  <equality>",
        '    <weld name="left_grip" body1="left_gripper" body2="left_park"/>',
        '    <weld name="right_grip" body1="right_gripper" body2="right_park"/>',
        "  </equality>",
        "</mujoco>",
    ]
    return "\n".join(lines) + "\n"


def reach_json() -> str:
    """The reach claim, every thing's place, and what the page needs to name a thing.

    One file, because the page and the probe want the same facts: the probe to drive the real
    inverse kinematics at every thing and find where the ring is wrong, the page to write the
    name of whatever is under the laser spot.
    """
    all_arms = arms()
    out = {
        "_": "GENERATED by scripts/make_workbench_scene.py - do not edit by hand.",
        "ringMm": {f"{how} {z}": list(v) for (how, z), v in sorted(RING_MM.items())},
        "arms": [
            {
                "id": a.id,
                "prefix": f"{a.id.split('_')[0]}_",
                "base": [round(v, 4) for v in bench_to_world(*a.base_mm)],
                "holdsOut": a.reaches_person,
            }
            for a in all_arms
        ],
        "give": [round(v, 4) for v in bench_to_world(GIVE_MM[0], GIVE_MM[1])],
        "eye": list(EYE_MM),
        "lookAt": list(LOOK_AT_MM),
        "bench": {
            "from": list(BENCH_FROM_MM),
            "to": list(BENCH_TO_MM),
            "centreX": CENTRE_X_MM,
        },
        "shelf": {
            "from": list(SHELF_FROM_MM),
            "to": list(SHELF_TO_MM),
            "bays": [list(b) for b in BAYS],
            "rows": [FRONT_ROW_Y_MM, BACK_ROW_Y_MM],
            "bottomBayUsableX": list(BOTTOM_BAY_USABLE_X_MM),
        },
        "things": [
            {
                "id": t.id,
                "what": t.what,
                "kind": t.kind,
                "bay": t.in_bay,
                "at": [round(v, 1) for v in t.place_mm],
                "gripMm": t.grip_mm,
                "gripHow": t.grip_how,
                "gripZ": t.grip_z_mm,
                "wanted": t.wanted,
                "graspable": t.graspable,
                "reachedBy": [a.id for a in all_arms if t.graspable and a.can_reach(t)],
            }
            for t in things()
        ],
    }
    return json.dumps(out, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="fail if the committed file is stale")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    wanted = {OUT: xml(), REACH_OUT: reach_json()}
    if a.check:
        for path, want in wanted.items():
            if not path.exists():
                print(f"{path} is missing; run scripts/make_workbench_scene.py", file=sys.stderr)
                return 1
            if path.read_text(encoding="utf-8") != want:
                print(
                    f"{path} is out of date with workbench/scene.py;"
                    " run scripts/make_workbench_scene.py",
                    file=sys.stderr,
                )
                return 1
        if not a.quiet:
            print(f"{OUT.name} and {REACH_OUT.name} match workbench/scene.py")
        return 0
    for path, want in wanted.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(want, encoding="utf-8")
    if clashes():
        print(f"two things stand in the same place: {clashes()}", file=sys.stderr)
        return 1
    if not a.quiet:
        print(f"wrote {OUT}")
        print(f"wrote {REACH_OUT}")
        print("\n".join(reach_report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
