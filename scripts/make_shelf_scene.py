"""Write `web/model/shelf_scene.xml` from the geometry in `end_demo/scene.py`.

    python scripts/make_shelf_scene.py            # write it
    python scripts/make_shelf_scene.py --check    # fail if the committed file is stale

**Why generate it.** The shelf exists twice over: once as numbers the cone maths and the
brain work from, and once as bodies MuJoCo simulates. If those two are typed out separately
they will disagree, and the disagreement will be invisible - the page would draw a cone over
one shelf while the arm reached for another. So `end_demo/scene.py` is the single source and
this writes the XML from it. The file is committed, because it is easier to read a scene than
a generator, and `--check` in the tests keeps the committed copy honest.

**The frame.** MuJoCo world: metres, z up, origin on the table top in the middle of the
person's edge. +X is away from the person, towards the shelf; +Y is to the person's left.
The table frame the rest of the demo uses is millimetres with x to the person's right and y
away from them, so

    X = y_mm / 1000,    Y = -(x_mm - 250) / 1000

and `table_to_world` below is the only place that conversion is written down.

**Two arms** come in by `<attach model="so101" prefix="left_"/>`, which this MuJoCo build
supports and `<replicate>` does not (`web/tests/probes/twoarms.probe.test.ts`). The prefix is
applied to the attached model's own root body, which is called `base`, so the wrapper bodies
are named `left_stand` and `right_stand` - `left_base` would collide.

**Two welds**, one per gripper. The old scene had one equality and the page switched it with
MuJoCo's global equality-disable flag, which only works while there is exactly one. With two
arms that flag would release both grips at once, so the page needs per-equality control here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from end_demo.scene import (  # noqa: E402
    GIVE_MM,
    REACH_MAX_MM,
    ROW_SURFACE_Z_MM,
    Thing,
    arms,
    shelf,
)

OUT = ROOT / "web" / "model" / "shelf_scene.xml"

#: The arm's resting pose, from `web/src/sim/arm.ts` (HOME). Tucked up and out of the way.
#:
#: It is written into the scene as a keyframe, not left to the page, because MuJoCo's
#: default pose has the arm stretched out flat along +X - straight through the front row.
#: The first physics run had the right arm lying on the remote, which then tipped over and
#: slid 49 mm while the scene settled. A scene whose own resting state knocks the shelf
#: about is not a scene, however carefully the page poses it afterwards.
HOME = (0.0, -1.0, 1.69, 1.0, 0.0, 0.2)

#: How far each arm turns outwards at rest, radians (shoulder_pan, limit 1.92).
#:
#: With both arms facing the shelf they stand between the person and the things. The demo's
#: whole question is which thing on the shelf they mean, so the shelf has to be in plain view
#: before anything is asked. At rest each arm folds down and turns away from the middle; it
#: turns back when it reaches for something.
#:
#: Chosen by measuring, not by eye: for a grid of poses, where the gripper ends up after the
#: physics settles, and how close the line from the person's eye to the nearest thing passes
#: it. This pose puts each gripper 230 mm out - clear of the things, which span 175 mm either
#: side - and leaves 85 mm between the arm and the nearest sight line. Facing forward leaves
#: 5 mm, which is to say the arm is in the way.
REST_PAN_RAD = 1.6

#: The table top, metres. Wide enough for both arms and the shelf, with room to spare.
TABLE = {"half_x": 0.34, "half_y": 0.42, "thick": 0.015}
#: The step the back row stands on, metres. It is the back row's surface, read from
#: `scene.py` rather than repeated, so the row cannot end up floating above its own step.
RISER_Z = ROW_SURFACE_Z_MM[0] / 1000.0
#: How deep the riser is, front to back. It starts behind the deepest thing in the front
#: row, so nothing in that row is leaning against the step.
RISER_FROM_MM, RISER_TO_MM = 372.0, 460.0

#: Rough masses, kilograms. A pill bottle is light; a mug with nothing in it is not heavy.
MASSES: dict[str, float] = {
    "bottle": 0.02,
    "mug": 0.030,
    "scissors": 0.025,
    "pen": 0.008,
    "bracket": 0.030,
    "phone": 0.040,
    "remote": 0.035,
    "tray": 0.120,
}

COLOURS: dict[str, str] = {
    "white": "0.93 0.93 0.90 1",
    "blue": "0.20 0.38 0.70 1",
    "grey": "0.55 0.57 0.60 1",
    "metal": "0.70 0.72 0.75 1",
    "black": "0.14 0.14 0.16 1",
    "wood": "0.66 0.50 0.33 1",
}

MUJOCO_SHAPE = {"cylinder": "cylinder", "box": "box", "capsule": "capsule"}


def table_to_world(x_mm: float, y_mm: float) -> tuple[float, float]:
    """Table millimetres to world metres. The only place this conversion is written."""
    return (y_mm / 1000.0, -(x_mm - 250.0) / 1000.0)


def _geom(thing: Thing) -> str:
    """The geom, with its half-sizes converted from table millimetres to metres.

    `scene.py` writes the sizes in **table axes** - across, deep, up - which is the same
    order MuJoCo wants for a box in this frame, so there is no rotation to get wrong and no
    orientation to keep in step between the two files.
    """
    across, deep, up = (list(thing.size_mm) + [0.0])[:3]
    if thing.shape == "cylinder":
        # radius, half height - no axes to swap.
        size = f"{across / 1000:.4f} {deep / 1000:.4f}"
    else:
        # A box's size is in world axes, and the world is turned a quarter against the
        # table: world X is depth, world Y is across. `table_to_world` converts positions;
        # sizes need converting too, and not doing it put every long thing across the shelf
        # instead of front to back - a 120 mm remote lying over its neighbours at 70 mm
        # spacing, which the physics then shoved 49 mm sideways.
        size = f"{deep / 1000:.4f} {across / 1000:.4f} {up / 1000:.4f}"
    return (
        f'<geom name="{thing.id}" type="{MUJOCO_SHAPE[thing.shape]}" size="{size}" '
        f'mass="{MASSES[thing.kind]}" rgba="{COLOURS[thing.colour]}" class="thing"/>'
    )


def body(thing: Thing) -> list[str]:
    """One thing, at the centre `scene.py` gives it. Nothing is added to its height here:
    `place_mm` already puts the centre above whatever the row stands on."""
    x_mm, y_mm, z_mm = thing.place_mm
    X, Y = table_to_world(x_mm, y_mm)
    Z = z_mm / 1000.0
    return [
        f'    <body name="{thing.id}" pos="{X:.4f} {Y:.4f} {Z:.4f}">',
        f'      <freejoint name="{thing.id}"/>',
        f"      {_geom(thing)}",
        "    </body>",
    ]


def _home_qpos(things: list[Thing]) -> str:
    """The whole qpos vector for the resting pose.

    Both arms' joints first, in the order `<attach>` adds them (left then right), then each
    thing's free joint - position and unit quaternion - at exactly where the body was
    placed, so loading the keyframe does not move the shelf.
    """
    out: list[str] = []
    for arm in arms():
        # Which way is outwards: with this pose a *negative* pan swings the arm standing on
        # the +Y side further out to +Y, so each arm takes the opposite sign to the side it
        # stands on. Measured, because the gripper's final place depends on the whole pose
        # and not on the pan alone.
        _, base_y = table_to_world(*arm.base_mm)
        pan = -REST_PAN_RAD if base_y > 0 else REST_PAN_RAD
        out += [f"{q:g}" for q in (pan, *HOME[1:])]
    # The free joints, in the order the bodies are declared: the two parking bodies come
    # before the shelf, because that is where they are in the worldbody.
    for arm in arms():
        X, Y = table_to_world(*arm.base_mm)
        out += [f"{X:.4f}", f"{Y:.4f}", "0.3000", "1", "0", "0", "0"]
    for t in things:
        x_mm, y_mm, z_mm = t.place_mm
        X, Y = table_to_world(x_mm, y_mm)
        out += [f"{X:.4f}", f"{Y:.4f}", f"{z_mm / 1000.0:.4f}", "1", "0", "0", "0"]
    return " ".join(out)


def xml() -> str:
    things = shelf()
    left, right = arms()
    lx, ly = table_to_world(*left.base_mm)
    rx, ry = table_to_world(*right.base_mm)
    riser_x0, _ = table_to_world(250.0, RISER_FROM_MM)
    riser_x1, _ = table_to_world(250.0, RISER_TO_MM)
    riser_cx = (riser_x0 + riser_x1) / 2
    riser_hx = abs(riser_x1 - riser_x0) / 2
    give_x, give_y = table_to_world(*GIVE_MM)

    lines = [
        "<!--",
        "  GENERATED by scripts/make_shelf_scene.py - do not edit by hand.",
        "",
        "  The end demo's scene: a tiered shelf on a table, and two SO-101 arms either side",
        "  of the person's seat. The geometry comes from end_demo/scene.py, which is also",
        "  what the cone maths and the brain work from, so the shelf the arm reaches for and",
        "  the shelf the page draws a cone over are the same shelf.",
        "",
        "  World: metres, z up, origin on the table top in the middle of the person's edge.",
        "  +X away from the person towards the shelf, +Y to the person's left.",
        "",
        "  Two arms by <attach prefix>: <replicate> is not in this MuJoCo build",
        "  (web/tests/probes/twoarms.probe.test.ts). so101.xml is unchanged.",
        "",
        "  Two welds and two parking bodies. The old one-arm scene switched its single weld",
        "  with MuJoCo's global equality-disable flag, which lets go of every grip at once.",
        "  data.eq_active cannot be touched in this build (a memory_view<bool>; emscripten",
        "  throws on any access), so both welds stay active and an arm holding nothing has",
        "  its weld on an invisible parking body. See tests/probes/eqactive.probe.test.ts.",
        "-->",
        '<mujoco model="the end demo: a shelf and two arms">',
        '  <asset><model name="so101" file="so101.xml"/></asset>',
        "",
        f'  <statistic center="{0.30:.2f} 0 0.10" extent="0.7"/>',
        "",
        "  <visual>",
        '    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35" specular="0 0 0"/>',
        "  </visual>",
        "",
        "  <default>",
        '    <default class="thing">',
        "      <!-- contype/conaffinity 3: things touch the table, each other and a gripper",
        "           (1). A held thing is switched to 2 so it no longer touches the gripper,",
        "           and no squeeze can push it while it is carried. -->",
        '      <geom friction="1.2 0.01 0.001" condim="4" solref="0.004 1" group="0"',
        '        contype="3" conaffinity="3"/>',
        "    </default>",
        "  </default>",
        "",
        "  <worldbody>",
        '    <light name="key" pos="0.2 -0.3 1.0" dir="0.1 0.3 -1" directional="true"/>',
        '    <light name="fill" pos="0.2 0.4 0.8" dir="0.1 -0.4 -1" directional="true"/>',
        "",
        "    <!-- The table top is z = 0. -->",
        f'    <geom name="table" type="box" pos="{TABLE["half_x"] - 0.06:.3f} 0 '
        f'{-TABLE["thick"]:.3f}"',
        f'      size="{TABLE["half_x"]:.3f} {TABLE["half_y"]:.3f} {TABLE["thick"]:.3f}"',
        '      rgba="0.78 0.63 0.47 1" friction="0.8 0.005 0.0001" group="0"'
        ' contype="3" conaffinity="3"/>',
        "",
        "    <!-- The step the back row stands on. Both rows are open from above: reaching",
        "         under a shelf board is where sim demos die and it proves nothing. -->",
        f'    <geom name="riser" type="box" pos="{riser_cx:.4f} 0 {RISER_Z / 2:.4f}"',
        f'      size="{riser_hx:.4f} {TABLE["half_y"] - 0.06:.4f} {RISER_Z / 2:.4f}"',
        '      rgba="0.72 0.58 0.42 1" friction="0.8 0.005 0.0001" group="0"'
        ' contype="3" conaffinity="3"/>',
        "",
        "    <!-- Where a thing goes into the person's hand. A site, not a body: nothing to",
        "         collide with, somewhere for the arm to aim. -->",
        f'    <site name="give" pos="{give_x:.4f} {give_y:.4f} 0.060" size="0.012"'
        ' rgba="0.2 0.6 0.3 0.35"/>',
        "",
        "    <!-- The arms. Their forward (shoulder_pan = 0) is +X, which is the shelf. -->",
        f'    <body name="left_stand" pos="{lx:.4f} {ly:.4f} 0">',
        '      <attach model="so101" prefix="left_"/>',
        "    </body>",
        f'    <body name="right_stand" pos="{rx:.4f} {ry:.4f} 0">',
        '      <attach model="so101" prefix="right_"/>',
        "    </body>",
        "",
        "    <!-- One parking body per arm: where its weld points while it holds nothing.",
        "         Declared after the arms and before the shelf, because the keyframe below",
        "         lists qpos in declaration order and getting that wrong scrambles the",
        "         scene - it moved the remote 38 mm the first time. See the module",
        "         docstring: eq_active cannot be switched in this build, so both welds are",
        "         always on and an idle one holds something harmless. -->",
        f'    <body name="left_park" pos="{lx:.4f} {ly:.4f} 0.30"><freejoint name="left_park"/>',
        '      <geom name="left_park" type="sphere" size="0.002" mass="0.0001"',
        '        contype="0" conaffinity="0" rgba="0 0 0 0"/></body>',
        f'    <body name="right_park" pos="{rx:.4f} {ry:.4f} 0.30"><freejoint name="right_park"/>',
        '      <geom name="right_park" type="sphere" size="0.002" mass="0.0001"',
        '        contype="0" conaffinity="0" rgba="0 0 0 0"/></body>',
        "",
    ]
    for row, name in ((0, "the raised row at the back"), (1, "the front row")):
        lines.append(f"    <!-- {name} -->")
        for t in [t for t in things if t.row == row]:
            lines += body(t)
        lines.append("")
    lines += [
        "  </worldbody>",
        "",
        "  <!-- The arms' resting pose (HOME in web/src/sim/arm.ts), tucked up clear of the",
        "       shelf. MuJoCo's default pose lies flat along +X, straight through the front",
        "       row. Order: the two arms' joints as attached, then every thing's freejoint",
        "       (7 numbers each: position then quaternion), left as the bodies were placed. -->",
        "  <keyframe>",
        f'    <key name="home" qpos="{_home_qpos(things)}"/>',
        "  </keyframe>",
        "",
        "  <!-- One weld per gripper, both always active. Each starts on its own parking",
        "       body, which is what an arm holding nothing is welded to. The page retargets a",
        "       weld to the real thing (eq_obj2id) and sets the relative pose (eq_data) to",
        "       grip, and points it back at the park to let go. eq_active cannot be switched",
        "       in this build at all - see the module docstring. -->",
        "  <equality>",
        '    <weld name="left_grip" body1="left_gripper" body2="left_park"/>',
        '    <weld name="right_grip" body1="right_gripper" body2="right_park"/>',
        "  </equality>",
        "</mujoco>",
    ]
    return "\n".join(lines) + "\n"


def reach_report() -> list[str]:
    """Which arm can reach what, for the person running this by hand."""
    out = ["", "reach, from the measured ring (140-280 mm):"]
    all_arms = arms()
    for t in shelf():
        who = [a.id for a in all_arms if a.can_reach(t)] or ["NOBODY"]
        out.append(f"  {t.id:<10} {t.where:<46} {', '.join(who)}")
    out.append(f"  (the ring's outer edge is {REACH_MAX_MM:.0f} mm)")
    for a in all_arms:
        out.append(f"  {a.id}: can hand over = {a.reaches_person}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="fail if the committed file is stale")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    want = xml()
    if a.check:
        if not OUT.exists():
            print(f"{OUT} is missing; run scripts/make_shelf_scene.py", file=sys.stderr)
            return 1
        if OUT.read_text(encoding="utf-8") != want:
            print(
                f"{OUT} is out of date with end_demo/scene.py; run scripts/make_shelf_scene.py",
                file=sys.stderr,
            )
            return 1
        if not a.quiet:
            print(f"{OUT.name} matches end_demo/scene.py")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(want, encoding="utf-8")
    if not a.quiet:
        print(f"wrote {OUT}")
        print("\n".join(reach_report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
