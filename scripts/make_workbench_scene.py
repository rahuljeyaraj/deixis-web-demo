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

**One arm, on a rail.** It comes in by `<attach model="so101" prefix="arm_"/>`, which this
MuJoCo build supports and `<replicate>` does not (`web/tests/probes/twoarms.probe.test.ts`). The
prefix is applied to the attached model's own root body, which is called `base`, so the wrapper
body is `carriage` - `arm_base` would collide.

The carriage carries a **slide joint** along the bench's x. The rail is sunk into the bench so
the carriage's top is flush with it: every ring in `RING_MM` was measured with the arm's base at
z = 0, and a rail standing proud of the bench would put all of them out by its own height.

**The bench top is three slabs, not one** - the near half, the far half, and the floor of the
trough between them. Sunk in one solid slab, the rail was a box buried inside the bench with
only its top face showing, and that face lay in exactly the same plane as the bench's own: two
coplanar faces, so which one a graphics card draws is anyone's guess, and what rahul saw was no
rail at all. Cutting the trough out of the bench top instead gives the rail real sides, a floor
16 mm down, and two bright guide rails lying in it with the carriage riding between them.

**One weld and one parking body.** `data.eq_active` cannot be touched in this build (a
`memory_view<bool>`; emscripten throws on any access), so the weld stays active and an arm
holding nothing has it pointed at an invisible parking body. See
`web/tests/probes/eqactive.probe.test.ts`. Nothing drives it in stage 1; it is here so the
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
    BAY_GRIP_Z_MM,
    BAYS,
    BENCH_FROM_MM,
    BENCH_THICK_MM,
    BENCH_TO_MM,
    BOARD_Z_MM,
    CENTRE_X_MM,
    EYE_MM,
    FRONT_ROW_Y_MM,
    GIVE_MM,
    LOOK_AT_MM,
    RAIL_GUIDE_MM,
    RAIL_GUIDE_Y_MM,
    RAIL_SUNK_MM,
    RAIL_TO_MM,
    RAIL_WIDE_MM,
    RAIL_Y_MM,
    RING_MM,
    ROBOT_EYE_MM,
    SHELF_BACK_MM,
    SHELF_FROM_MM,
    SHELF_SIDE_MM,
    SHELF_TO_MM,
    Thing,
    arm,
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
REST_ARM = (-0.6, 1.3, 0.9, 0.0, 0.2)

#: Each arm's shoulder at rest, radians (the joint's limit is 1.92).
#:
#: The arm faces the shelf, and at rest it must be out of the way of it: the demo's whole
#: question is which thing the person means, so the bench and all three bays have to be in plain
#: view before anything is asked.
#:
#: **Measured.** `web/tests/probes/visible.probe.test.ts` casts rays at every thing from the
#: default view, for a grid of nine turns and seven ways of folding, and asks how much of the
#: world each pose leaves visible. Parked at the right-hand end, the turn that pays is the one
#: that swings the arm **off** the bench, which is a positive pan, not the negative one that had
#: been here: 88 per cent of the world visible against 85, and the only thing left mostly hidden
#: is jar4, which is meant to be.
#:
#: That sweep was worth nothing until 24 Oct. It looked up joints called `left_shoulder_pan` and
#: `right_shoulder_pan`, which stopped existing when one arm on a rail replaced two standing
#: still and the prefix became `arm_`; `mj_name2id` answered -1, `qpos[undefined] = x` threw
#: nothing and did nothing, and all sixty-three poses were reported as equally good because not
#: one of them had been tried.
REST_PAN_RAD = 1.4
#: Where the carriage parks, in the bench frame.
#:
#: At the right-hand end, which is measured rather than chosen, and measured again after the
#: layout changed rather than quoted from the world before this one. Wherever the arm parks it
#: hides a band of the bench behind it, and the only question is what is in that band
#: (`out/stage1-visible.txt`):
#:
#: | parked at | the world visible | the least seen thing |
#: |---|---|---|
#: | x = 70, the left end | 84% | the chisel, at 20% |
#: | x = 350, the middle | 76% | **jar2, at 5%** - one of the three the demo turns on |
#: | x = 630, the right end | **88%** | the offcut, at 67% |
#:
#: At the right-hand end the band falls on the bench's far corner, where nothing needs to stand.
REST_SLIDE_X_MM = RAIL_TO_MM

#: Rough masses, kilograms. Nothing here is heavy; the gripper's weld does the carrying.
MASSES: dict[str, float] = {
    "offcut": 0.180,
    "screwdriver": 0.070,
    "knife": 0.030,
    "blades": 0.040,
    "chisel": 0.120,
    "rule": 0.045,
    "hammer": 0.450,
    "pot": 0.200,
    "oil": 0.250,
    "pliers": 0.180,
    "jar": 0.340,
    "glue": 0.050,
    "odds": 0.150,
    "tin": 0.420,
    "mug": 0.320,
}

#: Every colour in the world, and nothing else. NFR1 asks that every difference that matters be
#: visible at a glance at 1366 x 768 - colour, size and silhouette only - so these are chosen to
#: separate at that size and to stay apart in a photograph of a screen.
COLOURS: dict[str, str] = {
    "bench": "0.68 0.52 0.36 1",
    # The trough the carriage runs in: dark, so that the two bright guide rails in it and the
    # carriage between them read as machinery against a pale wooden bench.
    "rail": "0.17 0.18 0.21 1",
    "rail guide": "0.80 0.82 0.86 1",
    "carriage": "0.42 0.45 0.50 1",
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
    #
    # **And that is all of it.** A geom's quaternion is read in its body's frame, and the body
    # already carries `BENCH_TO_WORLD_QUAT` times its own yaw, so turning it here as well puts
    # the bench's quarter turn on twice - every shape a further 90 degrees round inside its own
    # thing. On a box or an upright cylinder that cannot be seen, which is why it survived a
    # stage that never stepped the physics; on an offcut of wood it swung a 132 mm plank across
    # the bench into a craft knife, and MuJoCo answered by firing the knife off the bench.
    quat = quat_mul(quat_about("z", turn_deg), quat_mul(quat_about("y", tilt_deg), spin))
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
        # **In the thing's own frame, not the world's.** A geom's pos is read in its body's
        # frame, and the body already carries `BENCH_TO_WORLD_QUAT` times its own yaw - so a
        # part written in world components is turned by that quarter turn a second time.
        #
        # This was wrong from the first build and it is not a small wrong: every part that
        # stands off the middle of its thing sat 90 degrees round from where the geometry puts
        # it. A hammer's head lay beside its handle instead of on the end of it, and a craft
        # knife's blade reached across the bench into an offcut of wood 126 mm away - which
        # MuJoCo answered by firing the knife off the bench, over and over (rahul, 24 Oct).
        # Centred parts - a lid, a label, the screws in a jar - were unharmed, which is why it
        # took a running physics to show it.
        #
        # The sizes a few lines up were already written in the thing's own frame, with a
        # comment saying so. The two disagreed.
        dx, dy, dz = at_mm
        where = f' pos="{dx / 1000.0:.4f} {dy / 1000.0:.4f} {mm(dz)}"'
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


def rod(name, x0, x1, y_mm, colour) -> str:
    """One guide rail: a cylinder lying along the bench's x, in the trough, below the top.

    It sits low enough that the arm's own base, which is bolted to the carriage at z = 0, cannot
    pass through it, and far enough off the middle that the carriage runs between the two rather
    than over them.
    """
    X, Y = bench_to_world((x0 + x1) / 2, y_mm)
    z = -(RAIL_SUNK_MM + BENCH_THICK_MM) / 2
    return (
        f'    <geom name="{name}" type="cylinder" size="{mm(RAIL_GUIDE_MM / 2)}'
        f' {mm((x1 - x0) / 2)}" pos="{X:.4f} {Y:.4f} {z / 1000:.4f}"\n'
        f'      quat="{quat_str(quat_about("x", 90.0))}" rgba="{COLOURS[colour]}"'
        ' contype="0" conaffinity="0"/>'
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
        "    <!-- The shelf unit: three bays across the whole back edge of the bench, 676 mm of",
        "         clear width in each. It used to be 440 mm in the middle, which left a strip of",
        "         bare bench at each end that nothing could use - further from the person than",
        "         the shelf, and reached over nothing. The rail puts every part of every bay in",
        "         reach, so there is no reason to stop short of the ends. -->",
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

    The carriage's slide first, then the arm's six joints, then every free joint in declaration
    order - the parking body, then the things. Getting that order wrong does not fail: it gives
    every thing the next thing's position and the whole bench comes out somewhere else. It has
    done that twice here, which is why `declared()` exists and why two tests watch it.
    """
    out = [f"{slide_of(REST_SLIDE_X_MM):.4f}"]
    out += [f"{q:g}" for q in (REST_PAN_RAD, *REST_ARM)]
    X, Y = bench_to_world(REST_SLIDE_X_MM, RAIL_Y_MM)
    out += [f"{X:.4f}", f"{Y:.4f}", "0.3000", "1", "0", "0", "0"]
    for t in declared(items):
        x_mm, y_mm, z_mm = t.place_mm
        X, Y = bench_to_world(x_mm, y_mm)
        yaw = quat_mul(BENCH_TO_WORLD_QUAT, quat_about("z", t.yaw_deg))
        out += [f"{X:.4f}", f"{Y:.4f}", mm(z_mm), *(f"{v:.6f}" for v in yaw)]
    return " ".join(out)


def slide_of(x_mm: float) -> float:
    """Where the carriage sits on its rail, metres, for a place across the bench.

    The slide joint runs along world **-Y**, because world Y is minus the bench's x. Written
    that way round so that sliding the joint forward moves the arm towards the person's right,
    which is the way everything else in this world is measured.
    """
    return (x_mm - CENTRE_X_MM) / 1000.0


def xml() -> str:
    items = things()
    a = arm()
    cx, cy = bench_to_world(CENTRE_X_MM, RAIL_Y_MM)
    slide_lo, slide_hi = slide_of(a.rail_from_mm), slide_of(a.rail_to_mm)
    # The parking body starts where the carriage parks, so that the keyframe and the body agree.
    px, py = bench_to_world(REST_SLIDE_X_MM, RAIL_Y_MM)
    give_x, give_y = bench_to_world(GIVE_MM[0], GIVE_MM[1])
    eye_x, eye_y = bench_to_world(ROBOT_EYE_MM[0], ROBOT_EYE_MM[1])
    bx0, by0 = BENCH_FROM_MM
    bx1, by1 = BENCH_TO_MM
    trough0 = RAIL_Y_MM - RAIL_WIDE_MM / 2
    trough1 = RAIL_Y_MM + RAIL_WIDE_MM / 2

    lines = [
        "<!--",
        "  GENERATED by scripts/make_workbench_scene.py - do not edit by hand.",
        "",
        "  The workbench demo's scene: a bench with a rail sunk across the front of it, one",
        "  SO-101 on a carriage that slides along it, a three-bay shelf unit on the whole of",
        "  the back edge, and twenty-one things. The geometry comes from",
        "  workbench/scene.py, which is also what the core, the robot code and the page work",
        "  from, so the shelf the arm reaches into and the shelf the page draws a spot on are",
        "  the same shelf.",
        "",
        "  World: metres, z up, origin on the bench top in the middle of the person's edge.",
        "  +X away from the person towards the shelf, +Y to the person's left.",
        "",
        "  The arm by <attach prefix>: <replicate> is not in this MuJoCo build",
        "  (web/tests/probes/twoarms.probe.test.ts). so101.xml is unchanged.",
        "",
        "  One weld and one parking body, the weld always active: data.eq_active cannot be",
        "  written in this build, and the global equality-disable flag would let go of every",
        "  grip at once. See web/tests/probes/eqactive.probe.test.ts.",
        "-->",
        '<mujoco model="the workbench: a bench, a shelf and an arm on a rail">',
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
        "    <!-- The bench top is z = 0, in two halves with the rail's trough between them.",
        "         One slab with the rail buried in it left the rail's top face in the same plane",
        "         as the bench's, which draws as whichever the card happens to pick - and what",
        "         you saw was no rail. -->",
        slab("bench_near", bx0, bx1, by0, trough0, -BENCH_THICK_MM, 0.0, "bench"),
        slab("bench_far", bx0, bx1, trough1, by1, -BENCH_THICK_MM, 0.0, "bench"),
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
        "    <!-- The rail: the floor of the trough, and the two guide rails lying in it. The",
        "         carriage's top comes flush with the bench, which is not decoration - every",
        "         ring in RING_MM was measured with the arm's base at z = 0, and a rail standing",
        "         proud would put every one of them out by its own height. The trough runs the",
        "         whole width of the bench, because a rail that stops where the travel stops",
        "         looks like a slot cut for one arm rather than a bench with a rail on it. -->",
        slab(
            "rail",
            bx0,
            bx1,
            trough0,
            trough1,
            -BENCH_THICK_MM,
            -RAIL_SUNK_MM,
            "rail",
        ),
        rod("rail_guide_near", bx0, bx1, RAIL_Y_MM - RAIL_GUIDE_Y_MM, "rail guide"),
        rod("rail_guide_far", bx0, bx1, RAIL_Y_MM + RAIL_GUIDE_Y_MM, "rail guide"),
        "",
        "    <!-- The arm, on a carriage that slides along the rail. Forward (shoulder_pan = 0)",
        "         is +X, which is the shelf, so the stand is not turned: the arm can always",
        "         slide to stand square in front of whatever it is reaching for, and that is the",
        "         whole reason for the rail. -->",
        f'    <body name="carriage" pos="{cx:.4f} {cy:.4f} 0">',
        '      <joint name="slide" type="slide" axis="0 -1 0"',
        f'        range="{slide_lo:.4f} {slide_hi:.4f}" damping="60" armature="0.4"/>',
        f'      <geom name="carriage" type="box" pos="0 0 {-RAIL_SUNK_MM / 2000:.4f}"',
        f'        size="{mm(RAIL_GUIDE_Y_MM - RAIL_GUIDE_MM / 2 - 2)} 0.045'
        f' {mm(RAIL_SUNK_MM / 2 - 1)}"',
        f'        rgba="{COLOURS["carriage"]}" contype="0" conaffinity="0"/>',
        '      <attach model="so101" prefix="arm_"/>',
        "    </body>",
        "",
        "    <!-- One parking body: where the weld points while the arm holds nothing.",
        "         Declared after the arm and before the things, because the keyframe lists qpos",
        "         in declaration order. -->",
        "    <!-- gravcomp: the weld is switched off whenever nothing is held, and a free body",
        "         with nothing to hold it falls for as long as the page is open. It is invisible",
        "         and touches nothing, so nobody would see it - but it is a speed that grows",
        "         without end, and MuJoCo resets a whole simulation when one of those gets out",
        "         of hand. Weightless, it simply waits where it is. -->",
        f'    <body name="park" pos="{px:.4f} {py:.4f} 0.30" gravcomp="1">'
        '<freejoint name="park"/>',
        '      <geom name="park" type="sphere" size="0.002" mass="0.0001"',
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
        "  <!-- The arm's resting pose, folded down and turned outwards. MuJoCo's default",
        "       pose lies flat along +X, straight through the shelf. Order: the carriage's",
        "       slide, then the arm's joints as attached, then every freejoint (7 numbers",
        "       each: a place and a turn), exactly where `declared()` put the bodies, and in",
        "       that same order. -->",
        "  <keyframe>",
        f'    <key name="home" qpos="{_home_qpos(items)}"/>',
        "  </keyframe>",
        "",
        "  <!-- The carriage's own motor. so101.xml brings six position actuators with it; this",
        "       is the seventh, and it comes first because the slide is first in qpos, which is",
        "       what lets the page set every control straight from the keyframe. -->",
        "  <actuator>",
        '    <position name="slide" joint="slide" kp="4000" kv="120" forcerange="-150 150"/>',
        "  </actuator>",
        "",
        "  <!-- One weld, always active, starting on the parking body. Nothing drives it until",
        "       stage 2; see the module docstring. -->",
        "  <equality>",
        '    <weld name="grip" body1="arm_gripper" body2="park"/>',
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
    one = all_arms[0]
    out = {
        "_": "GENERATED by scripts/make_workbench_scene.py - do not edit by hand.",
        "ringMm": {f"{how} {z}": list(v) for (how, z), v in sorted(RING_MM.items())},
        "rail": {
            "y": one.rail_y_mm,
            "from": one.rail_from_mm,
            "to": one.rail_to_mm,
            "holdsOut": one.reaches_person,
        },
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
            # The height the hand works at inside each bay, as a plain number per bay rather
            # than per thing: a thing that has been moved has to be gripped at the height of
            # the bay it is in **now**, and until this was here the only copy of these three
            # numbers the page had was the one baked into each thing's own `gripZ`.
            "gripZ": list(BAY_GRIP_Z_MM),
        },
        "things": [
            {
                "id": t.id,
                "what": t.what,
                "kind": t.kind,
                "bay": t.in_bay,
                "at": [round(v, 1) for v in t.place_mm],
                # What the robot code needs to work out a grip and a place for itself, rather
                # than being told one: how big the thing is, which way it lies, how much room
                # it takes with every part counted, and where its top and bottom are.
                "size": list(t.size_mm),
                "axis": t.axis,
                "yaw": t.yaw_deg,
                "room": [round(v, 1) for v in t.room_mm],
                # Where that room sits, from the thing's own middle: nothing here is centred on
                # its own footprint - a hammer's head is all at one end - so a footprint drawn
                # about the middle of a hammer covers 50 mm of bare bench and misses 50 mm of
                # hammer. Needed live, to answer whether something is already standing where a
                # held thing would go.
                "box": [round(v, 1) for v in t.box_mm[:2]],
                "rises": [round(v, 1) for v in t.rises_mm],
                "round": t.is_round,
                "gripMm": t.grip_mm,
                "gripHow": t.grip_how,
                "gripZ": t.grip_z_mm,
                "gripAbove": t.grip_above_mm,
                "wanted": t.wanted,
                "graspable": t.graspable,
                "slideTo": (
                    one.slide_for(t.place_mm[0], t.place_mm[1], t.grip_how, t.grip_z_mm)
                    if t.graspable
                    else None
                ),
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
