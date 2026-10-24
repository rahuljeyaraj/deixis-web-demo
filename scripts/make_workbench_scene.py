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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workbench.scene import (  # noqa: E402
    BAYS,
    BENCH_FROM_MM,
    BENCH_THICK_MM,
    BENCH_TO_MM,
    BOARD_Z_MM,
    CENTRE_X_MM,
    EYE_MM,
    GIVE_MM,
    HOOK_Z_MM,
    HOOKS,
    LOOK_AT_MM,
    RING_MM,
    ROBOT_EYE_MM,
    SHELF_BACK_MM,
    SHELF_FROM_MM,
    SHELF_SIDE_MM,
    SHELF_TO_MM,
    Part,
    Thing,
    arms,
    hooks,
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
#: **Zero, and measured.** `web/tests/probes/visible.probe.test.ts` casts rays at all nineteen
#: things from the default view, for a grid of nine turns and seven ways of folding, and asks
#: how much of the world each pose leaves visible. Turning either way makes it worse: an arm
#: square to the shelf is the narrowest thing it can be from in front, and every turn swings its
#: forearm across the bench. The sweep sits in a band of 48 to 71 per cent, this pose is at the
#: top of it with one thing mostly hidden, and the end demo's conclusion does not carry over
#: because there the person sat to one side of both arms rather than square in front of them.
REST_PAN_RAD = {"left_arm": 0.0, "right_arm": 0.0}

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
    "spanner": 0.140,
    "jar": 0.090,
    "glue": 0.050,
    "odds": 0.150,
}

#: Every colour in the world, and nothing else. NFR1 asks that every difference that matters be
#: visible at a glance at 1366 x 768 - colour, size and silhouette only - so these are chosen to
#: separate at that size and to stay apart in a photograph of a screen.
COLOURS: dict[str, str] = {
    "bench": "0.70 0.54 0.38 1",
    "shelf": "0.84 0.79 0.70 1",
    # Darker than the boards on purpose: with the back the same shade as everything else, a bay
    # read as a flat panel rather than as a space with things standing in it.
    "shelf back": "0.56 0.50 0.42 1",
    "hook": "0.42 0.44 0.47 1",
    "pale wood": "0.85 0.72 0.52 1",
    "pencil": "0.40 0.37 0.34 1",
    "wood": "0.62 0.44 0.26 1",
    "blue": "0.16 0.38 0.74 1",
    "red": "0.80 0.15 0.13 1",
    "steel": "0.72 0.74 0.78 1",
    "grey": "0.52 0.54 0.57 1",
    "beige": "0.88 0.80 0.62 1",
    "shadow": "0.20 0.19 0.17 1",
    "dark green": "0.09 0.30 0.18 1",
    "brown card": "0.64 0.49 0.33 1",
    "white": "0.95 0.95 0.92 1",
    # The jars are the point of the whole demo: three of them that no word separates. Glass, so
    # the screws inside show, and so that the one behind is visible through nothing at all.
    "glass": "0.80 0.88 0.90 0.45",
    "brass": "0.72 0.56 0.26 1",
}


def bench_to_world(x_mm: float, y_mm: float) -> tuple[float, float]:
    """Bench millimetres to world metres. The only place this conversion is written."""
    return (y_mm / 1000.0, -(x_mm - CENTRE_X_MM) / 1000.0)


def mm(v: float) -> str:
    return f"{v / 1000.0:.4f}"


def box_geom(name: str, size_mm: tuple[float, float, float], colour: str, mass: str = "") -> str:
    """A box, with its half-sizes converted from bench axes to world ones.

    The world is turned a quarter against the bench: world X is depth, world Y is across. So a
    size has to be converted as well as a position, and not doing it is how the end demo's first
    run laid every long thing across its shelf instead of front to back.
    """
    across, deep, up = size_mm
    size = f"{mm(deep / 2)} {mm(across / 2)} {mm(up / 2)}"
    return (
        f'<geom name="{name}" type="box" size="{size}" rgba="{COLOURS[colour]}"'
        f'{mass} class="thing"/>'
    )


def cylinder_geom(
    name: str, size_mm: tuple[float, float, float], colour: str, axis: str, mass: str = ""
) -> str:
    """A cylinder lying up, across the bench, or front to back.

    MuJoCo's cylinder runs along its own z, so lying one down takes a quarter turn: `across`
    turns it onto world Y, `deep` onto world X. The quaternion is written out rather than
    `euler` because this file has no `<compiler>`, so `euler` would be degrees here and radians
    inside so101.xml.
    """
    across, deep, up = size_mm
    if axis == "up":
        radius, half = across / 2, up / 2
        quat = ""
    elif axis == "across":
        radius, half = up / 2, across / 2
        quat = ' quat="0.707107 0.707107 0 0"'  # z onto world Y
    else:
        radius, half = up / 2, deep / 2
        quat = ' quat="0.707107 0 0.707107 0"'  # z onto world X
    return (
        f'<geom name="{name}" type="cylinder" size="{mm(radius)} {mm(half)}"{quat}'
        f' rgba="{COLOURS[colour]}"{mass} class="thing"/>'
    )


def part_geom(thing: Thing, part: Part, n: int) -> str:
    """One of a thing's extra shapes, offset from its centre in bench axes.

    A part carries no mass: the thing's first geom carries all of it, so that adding a hammer's
    head cannot quietly change what the hammer weighs.
    """
    name = f"{thing.id}_{n}"
    dx, dy, dz = part.at_mm
    X, Y = (dy / 1000.0, -dx / 1000.0)  # an offset, so no centre line to subtract
    pos = f' pos="{X:.4f} {Y:.4f} {mm(dz)}"'
    geom = (
        box_geom(name, part.size_mm, part.colour, ' mass="0.0001"')
        if part.shape == "box"
        else cylinder_geom(name, part.size_mm, part.colour, part.axis, ' mass="0.0001"')
    )
    return geom.replace("/>", f"{pos}/>")


def body(thing: Thing) -> list[str]:
    """One thing, at the centre `scene.py` gives it, with its parts bolted on."""
    x_mm, y_mm, z_mm = thing.place_mm
    X, Y = bench_to_world(x_mm, y_mm)
    mass = f' mass="{MASSES[thing.kind]}"'
    first = (
        box_geom(thing.id, thing.size_mm, thing.colour, mass)
        if thing.shape == "box"
        else cylinder_geom(thing.id, thing.size_mm, thing.colour, thing.axis, mass)
    )
    out = [
        f'    <body name="{thing.id}" pos="{X:.4f} {Y:.4f} {mm(z_mm)}">',
        f'      <freejoint name="{thing.id}"/>',
        f"      {first}",
    ]
    out += [f"      {part_geom(thing, p, n)}" for n, p in enumerate(thing.parts, start=1)]
    out.append("    </body>")
    return out


def slab(
    name: str, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, colour: str
) -> str:
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
    """The two-bay unit: two uprights, a back panel, and the two boards.

    The bottom bay's floor is the bench top itself, so there is no board at zero. The top board
    is the unit's lid and nothing stands on it: nothing can, because reaching down onto anything
    above about 100 mm is not a pose this arm has (`out/stage0-bay-ring.txt`).
    """
    x0, y0 = SHELF_FROM_MM
    x1, y1 = SHELF_TO_MM
    top = BOARD_Z_MM[-1][0] + BOARD_Z_MM[-1][1]
    out = [
        "    <!-- The shelf unit. 50 mm further back than the plan's 4.1: reaching level into",
        "         the bottom bay the arm cannot work closer in than 280 mm, and at the plan's",
        "         place the middle of the front row was 255 mm from both arms. -->",
        slab("shelf_left", x0, x0 + SHELF_SIDE_MM, y0, y1, 0.0, top, "shelf"),
        slab("shelf_right", x1 - SHELF_SIDE_MM, x1, y0, y1, 0.0, top, "shelf"),
        slab("shelf_back", x0, x1, y1 - SHELF_BACK_MM, y1, 0.0, top, "shelf back"),
    ]
    for n, (under, thick) in enumerate(BOARD_Z_MM):
        what = "the middle board" if n == 0 else "the top board - a lid, not a place"
        out.append(f"    <!-- {what} -->")
        out.append(slab(f"shelf_board{n}", x0, x1, y0, y1, under, under + thick, "shelf"))
    return out


def hook_rail() -> list[str]:
    """A rail down each end of the unit, and the three hooks sticking out of them.

    On the ends and not across the front: a hook on the front face hangs a pair of pliers
    straight down over the bottom bay's opening, in front of the three jars. See `HOOK_Z_MM`.

    190 mm is a height the ring was swept at, and it is as high as a hook can usefully be: the
    unit's lid is at 244. Lower, and a 170 mm spanner hangs into the bench.
    """
    out = ["    <!-- A rail down each end of the unit, and three hooks: two tools, one empty. -->"]
    for side in ("left", "right"):
        ys = [y for s, y in HOOKS if s == side]
        edge = SHELF_FROM_MM[0] if side == "left" else SHELF_TO_MM[0]
        out.append(
            slab(
                f"hook_rail_{side}",
                edge - 8.0 if side == "left" else edge,
                edge if side == "left" else edge + 8.0,
                min(ys) - 20.0,
                max(ys) + 20.0,
                HOOK_Z_MM - 14.0,
                HOOK_Z_MM + 14.0,
                "hook",
            )
        )
    for n, (side, x, y, z) in enumerate(hooks()):
        edge = SHELF_FROM_MM[0] if side == "left" else SHELF_TO_MM[0]
        out.append(
            slab(
                f"hook{n}",
                min(x, edge - 8.0) if side == "left" else edge + 8.0,
                edge - 8.0 if side == "left" else max(x, edge + 8.0),
                y - 5.0,
                y + 5.0,
                z - 5.0,
                z + 5.0,
                "hook",
            )
        )
    return out


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
    for t in items:
        x_mm, y_mm, z_mm = t.place_mm
        X, Y = bench_to_world(x_mm, y_mm)
        out += [f"{X:.4f}", f"{Y:.4f}", mm(z_mm), "1", "0", "0", "0"]
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
        "  The workbench demo's scene: a bench, a two-bay shelf unit with a hook rail, two",
        "  SO-101 arms, and nineteen things. The geometry comes from workbench/scene.py, which",
        "  is also what the core, the robot code and the page work from, so the shelf the arm",
        "  reaches into and the shelf the page draws a spot on are the same shelf.",
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
    lines.append("")
    lines += hook_rail()
    lines += [
        "",
        "    <!-- Where a thing is held out to the person, and where the robot's own camera",
        "         looks from. Sites, not bodies: nothing to collide with, somewhere to aim. -->",
        f'    <site name="give" pos="{give_x:.4f} {give_y:.4f} {mm(GIVE_MM[2])}" size="0.012"'
        ' rgba="0.2 0.6 0.3 0.35"/>',
        f'    <site name="robot_eye" pos="{eye_x:.4f} {eye_y:.4f} {mm(ROBOT_EYE_MM[2])}"'
        ' size="0.010" rgba="0.12 0.52 0.90 0.30"/>',
        "",
        "    <!-- The arms. Unturned, forward (shoulder_pan = 0) is +X, which is the shelf, so",
        "         neither needs turning on its stand: each one's fan already covers its own half",
        "         of the bench and most of the unit. What they do at rest is fold outwards, out",
        "         of the middle of the frame - see REST_PAN_RAD. -->",
        f'    <body name="left_stand" pos="{lx:.4f} {ly:.4f} 0">',
        '      <attach model="so101" prefix="left_"/>',
        "    </body>",
        f'    <body name="right_stand" pos="{rx:.4f} {ry:.4f} 0">',
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
    where = {None: "on the bench and the hooks", 0: "in the bottom bay", 1: "in the top bay"}
    for bay in (None, 0, 1):
        lines.append(f"    <!-- {where[bay]} -->")
        for t in [t for t in items if t.in_bay == bay]:
            lines += body(t)
        lines.append("")
    lines += [
        "  </worldbody>",
        "",
        "  <!-- The arms' resting pose, folded down and turned outwards. MuJoCo's default pose",
        "       lies flat along +X, straight through the shelf. Order: the two arms' joints as",
        "       attached, then every freejoint (7 numbers each: position then quaternion), left",
        "       exactly where the bodies were placed. -->",
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
            "hookZ": HOOK_Z_MM,
            "hooks": [{"side": side, "at": [x, y, z]} for side, x, y, z in hooks()],
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
    if not a.quiet:
        print(f"wrote {OUT}")
        print(f"wrote {REACH_OUT}")
        print("\n".join(reach_report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
