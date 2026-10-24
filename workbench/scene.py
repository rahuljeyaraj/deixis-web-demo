"""The workbench: a bench, a two-bay shelf unit, a hook rail, two arms, and nineteen things.

One source of truth for the geometry (`workbench-requirements.md` FR5). The physics scene
(`web/model/workbench_scene.xml`) is generated from this by `scripts/make_workbench_scene.py`,
and the page draws what the physics holds, so there is one set of numbers and no second copy
to drift.

## The frame

Millimetres. **x across, to the person's right; y away from the person; z up from the bench
top.** The centre line is x = 350. The world the physics runs in is metres with +X away from
the person, and `scripts/make_workbench_scene.py` holds the only copy of that conversion.

## What stage 0 measured, and what it changed

Everything below rests on two measurements, both run on 24 Oct and both reported in full:

- `out/stage0-grip-width.txt` - the widest thing the gripper holds through a lift and a move.
- `out/stage0-bay-ring.txt` - the reach ring at bay height, reaching in horizontally, and at
  the other heights this world turns out to need.

They changed four things in the plan, and each one is marked where it bites:

1. **The arm row has to stand at least about 215 mm in front of the bay.** Reaching level into
   the bottom bay, the arm cannot work closer in than **280 mm**; the plan had the shelf 180 mm
   behind the arms, which left the middle of the bottom bay's front row - where two of the three
   identical jars stand - out of reach of both arms. The shelf unit is therefore 50 mm further
   back than the plan's 4.1, at y 350 to 500 rather than 300 to 450, and the arms stay where
   the plan put them. See `SHELF_FROM_MM`.
2. **Nothing much above 100 mm can be gripped from above**, and nothing at all above about 130.
   So the unit's top board, a quarter of a metre up, is not a place this robot can put anything
   or take anything from: it holds nothing. The oil bottle the plan stood there is on the bench
   instead, where it can be taken round the middle, level - and it is still too tall for any
   bay, which is the job it came to do. See `oil`.
3. **A thing inside a bay has to be tall enough to grip.** The hand comes in level, and the
   jaws stand about 20 mm above and below the fingers, so a flat thing lying on a board cannot
   be taken hold of at all: there is a board over it, so there is no way in from above, and no
   room beside it for the jaws. Everything in a bay is therefore at least `BAY_TALLEST_FLAT_MM`
   tall. That moved the tin of blades and the chisel out of the top bay and onto the bench,
   where they are gripped from above - which costs Act 2 nothing, because what makes them
   borderline is being sharp-but-a-tin and sharp-but-not-pointed-at, not where they live.
4. **The gripper is not the limit the plan feared.** It holds 96 mm with the grip assist and
   76 mm on the jaws and friction alone, against the plan's assumed 32. No size here is
   constrained by it; `GRIP_WIDEST_MM` is the ceiling and every grip is checked against it.

## The one asymmetry

The shelf is two rows deep and the back row is real. `jar4` stands behind `jar2`, both the same
jar, and from the default view it cannot be seen - the jar in front of it hides it, and the
board above cuts off any look over the top. Orbit the camera up and it appears. That is the
missing information Act 4 is built on, and it is honest: a real robot has exactly this problem.

What the robot's own camera can see is computed, not stored, from `ROBOT_EYE_MM`; it is not
used to hide anything yet (FR4, stage 2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------- measured

#: The reach ring, by how the hand is pointed and the height it works at: how close in the arm
#: can work and how far out it reaches straight ahead, millimetres from its base axis.
#:
#: Measured on 24 Oct by `web/tests/probes/bay-ring.probe.test.ts`, which drives the real
#: inverse kinematics round the ring at each height and rejects a pose that folds the arm into
#: itself or pushes it through the bench. The full sweep is in `out/stage0-bay-ring.txt`.
#:
#: **level** is the hand horizontal, which is the only way into a bay: there is a board over a
#: bay, so the hand cannot come down on what is inside it. **down** is the hand pointing
#: straight at the bench, which is how a thing with nothing above it is picked up.
#:
#: Each height here is a height something in this world is actually gripped at, and was swept
#: at that height - nothing is interpolated:
#:
#: - level 55: the middle of a thing standing in the bottom bay (clear opening 0 to 110).
#: - level 130: a tall bottle standing on the bench, taken round the middle.
#: - level 177: the middle of the top bay (clear opening 122 to 232).
#: - level 190: the hook rail.
#: - down 35: a thing standing on the bench.
#: - down 70: letting go over the rim of the pot.
#:
#: Two heights are **absent on purpose**, because the sweep found nothing at them: down at 130
#: and anything higher. Reaching down dies between 100 mm (where the ring has narrowed to
#: 210-240) and 130 mm (where there is no pose at all), even with 25 degrees of tilt allowed.
#: Asking this table for a downward grip above 100 mm raises a `KeyError`, which is the answer.
RING_MM: dict[tuple[str, int], tuple[float, float]] = {
    ("level", 55): (280.0, 470.0),
    ("level", 130): (260.0, 480.0),
    ("level", 177): (230.0, 475.0),
    ("level", 190): (220.0, 470.0),
    ("down", 35): (135.0, 325.0),
    ("down", 70): (130.0, 300.0),
}

#: How much of that outward reach is lost per degree the shoulder turns off forward.
#:
#: Reach is not a circle. The fall is close enough to straight, and close enough to the same at
#: every height swept (0.48 to 0.51 mm per degree), that one number covers all of them - and it
#: is the same 0.48 that `end_demo` measured for the fingers-down ring by a different route,
#: which is some comfort that it is a property of the arm and not of a sweep.
REACH_FALL_MM_PER_DEG = 0.48

#: How far either side of its facing an arm can work, degrees. The SO-101's `shoulder_pan`
#: stops at +/-110 (`web/model/so101/so101.xml`); 100 is `end_demo`'s measured working limit,
#: where the solver could still get down onto a thing, with a little room.
PAN_LIMIT_DEG = 100.0

#: The widest thing the gripper holds through a lift and a move, millimetres.
#:
#: Measured on 24 Oct by `web/tests/probes/grip-width.probe.test.ts`: boxes of increasing width
#: picked up off the bench, carried 220 mm, put down and let go, each at nine aims so no width
#: fails for being badly aimed. The jaws open to 107 mm tip to tip at the gripper's own limit,
#: which puts the ceiling near 99; **96 mm** is the widest that came through the whole move with
#: the grip assist the demo uses, and **76 mm** the widest held by the closed jaws and friction
#: with the weld switched off.
#:
#: 76 is the number used here, because it is the one that does not depend on the weld. The plan
#: assumed 32 mm, from the 30 mm blocks of the cubes demo - the only width that demo was ever
#: asked for. Nothing in this world is near either figure, so the measurement constrains
#: nothing; it is here so that a size can be checked rather than argued about.
GRIP_WIDEST_MM = 76.0
GRIP_WIDEST_WELDED_MM = 96.0

#: How tall a thing must be before it can be gripped inside a bay, millimetres.
#:
#: Not swept - reasoned from the jaw geometry and then checked by eye against the sweep. The
#: hand comes in level, the pads stand about 20 mm above and below the fingers, so the jaws need
#: that much of the thing's side to close on clear of the board it stands on. A flat thing in a
#: bay has no way in at all: a board above it rules out coming down, and the board below it
#: rules out coming in level. Stage 2 drives the real jaws at every thing and is the gate on
#: this number; until then it is why nothing short stands in a bay.
BAY_TALLEST_FLAT_MM = 50.0

# ------------------------------------------------------------------- the bench

#: The bench top: the person stands at the near edge, the shelf unit stands at the far one.
BENCH_FROM_MM = (0.0, -50.0)
BENCH_TO_MM = (700.0, 500.0)
BENCH_THICK_MM = 18.0
CENTRE_X_MM = 350.0

#: The shelf unit's footprint. **50 mm further back than the plan's 4.1** - see the module
#: docstring, point 1: at the plan's 300 the middle of the bottom bay's front row was 255 mm
#: from either arm and the arm cannot work closer in than 280.
SHELF_FROM_MM = (200.0, 350.0)
SHELF_TO_MM = (500.0, 500.0)
#: The uprights and the back panel, thickness.
SHELF_SIDE_MM = 12.0
SHELF_BACK_MM = 10.0
#: Each board: the height of its underside and its thickness.
BOARD_Z_MM = ((110.0, 12.0), (232.0, 12.0))
#: A bay's clear opening across and in depth: inside the uprights, in front of the back panel.
BAY_CLEAR_X_MM = (SHELF_FROM_MM[0] + SHELF_SIDE_MM, SHELF_TO_MM[0] - SHELF_SIDE_MM)
BAY_CLEAR_Y_MM = (SHELF_FROM_MM[1], SHELF_TO_MM[1] - SHELF_BACK_MM)

#: Where a row of things stands in a bay, away from the person. The rows are 85 mm apart, and
#: the front row is 30 mm inside the opening.
FRONT_ROW_Y_MM = 380.0
BACK_ROW_Y_MM = 465.0

#: The hooks: how high they are, how far they stick out of the unit's side, and where along it.
#:
#: **On the unit's two end faces, not across its front.** The plan put a rail on the front face
#: (4.1), and built that way the first time it hung a pair of pliers and a spanner straight down
#: across the bottom bay's opening - over the three identical jars, which are the one thing in
#: this world that has to be seen. A hook high enough to clear them has nothing to hang from:
#: the unit's lid is at 244. The ends are solid, clear of both openings, and in reach: a thing
#: on a hook there is 250 mm from the arm on its own side, inside the level ring at 190.
HOOK_Z_MM = 190.0
HOOK_OUT_MM = 25.0
#: Which side, and how far along the unit: pliers, an empty one, and the spanner.
HOOKS = (("left", 400.0), ("left", 465.0), ("right", 400.0))

#: Where a thing is held out to the person: the middle of the bench, as near them as an arm can
#: work, 60 mm up. Not on the bench - nothing on the person's own side of the bench can be
#: reached at all (the sweep's near edge is 135 mm and the arms stand at y = 150), so passing a
#: thing over means holding it out rather than putting it down.
GIVE_MM = (350.0, 120.0, 60.0)

#: The robot's own camera: a point above and between the arms. What it can see is computed from
#: here rather than stored, so the list the model is given cannot quietly disagree with the
#: scene (FR4, stage 2).
ROBOT_EYE_MM = (350.0, 60.0, 420.0)

#: Where the person's eye is for the page's default view, and what it looks at.
#:
#: Not a camera preference: the whole of Act 4 is that the robot has not been told about the jar
#: in the back row, and the judge finds it by moving the camera. So the default view has to hide
#: it and a small orbit has to reveal it, and `behind_from_the_eye` puts the jar on this eye's
#: own line. `web/tests/probes/visible.probe.test.ts` casts rays at every thing from here and
#: says what is really visible; it is the gate on both.
EYE_MM = (350.0, -450.0, 400.0)
LOOK_AT_MM = (350.0, 320.0, 110.0)


def behind_from_the_eye(x_mm: float, from_y: float, to_y: float) -> float:
    """Where a thing has to stand at `to_y` to be exactly behind one at (`x_mm`, `from_y`).

    Straight behind is not the same across as straight behind as seen from a particular place.
    The jars stand left of centre and the eye is on the centre line, so a jar put at the same x
    in the back row leans out from behind the one in front of it and can be seen past its edge -
    44 per cent of it, when this was measured. One line of similar triangles fixes that, and
    ties the fourth jar's place to the eye it is hidden from, which is the honest way round.
    """
    return x_mm + (x_mm - EYE_MM[0]) * (to_y - from_y) / (from_y - EYE_MM[1])


def bay_z_mm(bay: int) -> tuple[float, float]:
    """A bay's clear opening: the height of its floor and of the board over it."""
    floor = 0.0 if bay == 0 else BOARD_Z_MM[bay - 1][0] + BOARD_Z_MM[bay - 1][1]
    return (floor, BOARD_Z_MM[bay][0])


#: The two bays, bottom first, as (floor, ceiling). Two and not three: the sweep was run at the
#: two heights the plan proposed and found both reachable, and a third was never measured, so
#: the plan's own rule - "two bays, not three, until stage 0 says a third is reachable" - leaves
#: it at two. A third bay would also have to be gripped above 244 mm, and the fingers-down
#: sweep says nothing is reachable up there from over the top.
BAYS = (bay_z_mm(0), bay_z_mm(1))

#: The height the hand works at inside each bay: the middle of the clear opening, rounded to the
#: height the ring was swept at.
BAY_GRIP_Z_MM = (55, 177)


# -------------------------------------------------------------------- the arms


@dataclass(frozen=True)
class Arm:
    """One SO-101, standing on the bench. Pointable: the person may mean *you*."""

    id: str
    #: Where its base stands, in the bench frame, millimetres.
    base_mm: tuple[float, float]
    #: Which way it faces, degrees, 0 being straight at the shelf and + towards the person's
    #: left. The shoulder turns `PAN_LIMIT_DEG` either side of this.
    facing_deg: float = 0.0

    def radius_to(self, x: float, y: float) -> float:
        return math.hypot(x - self.base_mm[0], y - self.base_mm[1])

    def bearing_to(self, x: float, y: float) -> float:
        """How far a spot lies off this arm's forward, degrees, -180 to 180.

        Forward is +y, away from the person towards the shelf, before the arm is turned. A
        positive turn swings forward towards the person's left, which is a step of -x, which is
        why the x term is negated. The same convention as `end_demo.scene`.
        """
        world = math.degrees(math.atan2(-(x - self.base_mm[0]), y - self.base_mm[1]))
        return (world - self.facing_deg + 180.0) % 360.0 - 180.0

    def reaches(self, x: float, y: float, how: str, z: int) -> bool:
        """Can this arm work at this spot, pointing `how`, at the height the ring was swept at?

        Three things have to hold: the spot is far enough out that the arm is not folded into
        itself, it is in front of the arm rather than behind it, and it is inside the reach left
        at that bearing. `how` and `z` must name a ring that was actually measured - asking for
        a downward grip above 100 mm raises, because the sweep found nothing there.
        """
        bearing = abs(self.bearing_to(x, y))
        if bearing > PAN_LIMIT_DEG:
            return False
        near, far = RING_MM[(how, z)]
        return near <= self.radius_to(x, y) <= far - REACH_FALL_MM_PER_DEG * bearing

    def can_reach(self, thing: Thing) -> bool:
        x, y, _ = thing.place_mm
        return self.reaches(x, y, thing.grip_how, thing.grip_z_mm)

    @property
    def reaches_person(self) -> bool:
        """Can it hold a thing out to them? Their hand is 60 mm up, nearest the down-70 ring."""
        return self.reaches(GIVE_MM[0], GIVE_MM[1], "down", 70)


def arms() -> list[Arm]:
    """The two arms, one at each side, both facing the shelf.

    Two is not decoration, it is forced by the ring. Reaching level into the bottom bay the arm
    cannot work closer in than 280 mm, so each arm has a dead circle in front of it, and the
    far corners of the unit are past the outer edge. Between them:

    - the **right** arm covers the left of the bottom bay's front row, and the **left** arm
      covers the right of it - each arm reaches *across*, because the near half is inside its
      own dead circle. "It is on my side, so you take it" comes out the other way round here,
      which is worth knowing before anyone writes the words.
    - the middle of that row, where the third jar stands, is in **both** arms' reach, and so is
      the whole of the back row. That overlap is where "either of us could do this one" lives.
    - in the top bay, where the ring comes 50 mm further in, both arms reach almost everything.

    Left and right are **the person's**, as they look at the bench. They stay where the plan
    put them (4.2); it is the shelf that moved, see the module docstring.
    """
    return [
        Arm("left_arm", base_mm=(170.0, 150.0)),
        Arm("right_arm", base_mm=(530.0, 150.0)),
    ]


# ------------------------------------------------------------------ the things


@dataclass(frozen=True)
class Part:
    """A shape bolted to a thing, so its silhouette reads without a label.

    `at_mm` is from the thing's own centre, in bench axes. A part has no joint of its own: it
    moves with the thing, which is what makes a hammer a hammer rather than a stick beside a
    block.
    """

    shape: str
    size_mm: tuple[float, float, float]  #: across, deep, up
    at_mm: tuple[float, float, float]
    colour: str
    axis: str = "up"


@dataclass(frozen=True)
class Thing:
    """One thing in the world.

    `size_mm` is always the thing's **bounding box in bench axes** - across, deep, up - whatever
    shape it is, and `axis` says which way a cylinder lies. Writing it that way rather than in
    the thing's own frame means there is no orientation to keep in step between this file and
    the scene file, and that the page can lay a name under the cursor without knowing shapes.
    """

    id: str
    what: str  #: what it is, in the fewest words that name it on the page
    #: Things of the same kind cannot be told apart by any word. This is the heart of the demo:
    #: `jar` has four, and no sentence separates them.
    kind: str
    shape: str  #: box or cylinder
    size_mm: tuple[float, float, float]  #: across, deep, up
    colour: str
    #: Where it stands: across, away, and the height its bottom rests at.
    at_mm: tuple[float, float, float]
    #: How wide the part the gripper closes on is. Checked against `GRIP_WIDEST_MM`.
    grip_mm: float
    #: How the hand comes at it, and the height the ring for that was swept at.
    grip_how: str = "down"
    grip_z_mm: int = 35
    #: False for a thing nobody would ask for but anybody might point at - the pot is somewhere
    #: to put things, not something to want.
    wanted: bool = True
    #: False for a thing that is never picked up at all, only put into.
    graspable: bool = True
    #: Which way a cylinder lies: up, across the bench, or front to back. Ignored for a box.
    axis: str = "up"
    #: The shapes a silhouette needs beyond the first, so a thing reads without a label: a
    #: hammer's head, a jar's lid, a screwdriver's shaft. Each is its own little `Part`.
    parts: tuple[Part, ...] = ()

    @property
    def across_mm(self) -> float:
        return self.size_mm[0]

    @property
    def deep_mm(self) -> float:
        return self.size_mm[1]

    @property
    def tall_mm(self) -> float:
        return self.size_mm[2]

    @property
    def place_mm(self) -> tuple[float, float, float]:
        """Its **centre**: what the physics wants for a body, and what the eye looks at."""
        return (self.at_mm[0], self.at_mm[1], self.at_mm[2] + self.tall_mm / 2)

    @property
    def in_bay(self) -> int | None:
        """Which bay it stands in, or None for the bench and the hooks.

        All three of x, y and z have to say so. The oil bottle stands on the bench at the back
        right, past the end of the unit: without the x test it looked like a thing in the bottom
        bay that was 40 mm too tall for it, which is the one thing the oil is not.
        """
        x, y, z = self.at_mm
        if not (BAY_CLEAR_X_MM[0] <= x <= BAY_CLEAR_X_MM[1]):
            return None
        if not (SHELF_FROM_MM[1] <= y <= SHELF_TO_MM[1]):
            return None
        for bay, (floor, _) in enumerate(BAYS):
            if abs(z - floor) < 1.0:
                return bay
        return None


def things() -> list[Thing]:
    """The nineteen things: ten on the bench, two on hooks, seven on the shelf.

    Every one of them is on a real bench, and the properties the demo needs are not added to
    them - they are what a bench is. Several jars you cannot tell apart without reading a label
    you cannot see. A tube with its cap off beside one with its cap on. Sharp things a person
    would really make a rule about. A bottle too tall for the shelf it belongs on. Tape that can
    be hung up or laid down. A back row nobody has looked at for months.

    The crossing that makes Act 2 work is natural too. Point at the craft knife and say "the
    ones like this": sharp things? small things? red things? The knife is all three. The blade
    tin is sharp and red, but closed, and a tin. The chisel is sharp but big and wooden.
    """
    bench = 0.0
    bay0, bay1 = (bay_z_mm(0)[0], bay_z_mm(1)[0])
    return [
        # ----------------------------------------------------------- on the bench
        Thing(
            "job",
            "the offcut being worked on",
            "offcut",
            "box",
            (120.0, 70.0, 35.0),
            "pale wood",
            (350.0, 200.0, bench),
            grip_mm=70.0,
            parts=(Part("box", (112.0, 3.0, 1.0), (0.0, 8.0, 17.5), "pencil"),),
        ),
        Thing(
            "screwdriver",
            "a screwdriver",
            "screwdriver",
            "cylinder",
            (90.0, 28.0, 28.0),  # the handle; the shaft is a part, so it reads as one tool
            "blue",
            (300.0, 290.0, bench + 14.0),
            grip_mm=28.0,
            axis="across",
            parts=(Part("cylinder", (100.0, 8.0, 8.0), (95.0, 0.0, 0.0), "steel", "across"),),
        ),
        Thing(
            "knife",
            "a craft knife",
            "knife",
            "box",
            (140.0, 14.0, 14.0),
            "grey",
            (420.0, 290.0, bench),
            grip_mm=14.0,
            parts=(
                Part("box", (30.0, 15.0, 15.0), (-40.0, 0.0, 0.0), "red"),
                Part("box", (34.0, 3.0, 8.0), (82.0, 0.0, 1.0), "steel"),
            ),
        ),
        Thing(
            "blades",
            "a tin of spare blades",
            "blades",
            "box",
            (55.0, 30.0, 12.0),
            "red",
            (240.0, 300.0, bench),
            grip_mm=30.0,
        ),
        Thing(
            "chisel",
            "a chisel",
            "chisel",
            "box",
            (90.0, 22.0, 22.0),  # the handle
            "wood",
            (470.0, 300.0, bench),
            grip_mm=22.0,
            parts=(Part("box", (70.0, 16.0, 6.0), (78.0, 0.0, -5.0), "steel"),),
        ),
        Thing(
            "rule",
            "a steel rule",
            "rule",
            "box",
            # Lying front to back, not across: across, at the right of the bench, it lay in the
            # right arm's own shadow from the default view and 19 per cent of it could be seen.
            (25.0, 150.0, 2.0),
            "steel",
            (660.0, 300.0, bench),
            grip_mm=25.0,
        ),
        Thing(
            "tape",
            "a roll of masking tape",
            "tape",
            "cylinder",
            (60.0, 60.0, 25.0),
            "beige",
            (60.0, 240.0, bench),
            grip_mm=60.0,
            parts=(Part("cylinder", (26.0, 26.0, 27.0), (0.0, 0.0, 0.0), "shadow"),),
        ),
        Thing(
            "hammer",
            "a hammer",
            "hammer",
            "box",
            (230.0, 24.0, 24.0),  # the handle, lying across the bench
            "wood",
            (400.0, 240.0, bench),
            grip_mm=24.0,
            parts=(Part("box", (28.0, 90.0, 28.0), (-129.0, 0.0, 2.0), "steel"),),
        ),
        Thing(
            "pot",
            "a pot of loose screws",
            "pot",
            "cylinder",
            (55.0, 55.0, 60.0),
            "grey",
            (620.0, 250.0, bench),
            grip_mm=55.0,
            grip_z_mm=70,  # letting go over its rim, not picking it up
            wanted=False,
            graspable=False,
            parts=(
                Part("cylinder", (46.0, 46.0, 8.0), (0.0, 0.0, 22.0), "brass"),
                Part("cylinder", (50.0, 50.0, 6.0), (0.0, 0.0, 28.0), "shadow"),
            ),
        ),
        Thing(
            "oil",
            "a bottle of oil",
            "oil",
            "cylinder",
            (50.0, 50.0, 120.0),
            "dark green",
            (650.0, 420.0, bench),
            grip_mm=22.0,
            grip_how="level",
            grip_z_mm=130,  # too tall to grip from above: the module docstring, point 2
            parts=(Part("cylinder", (22.0, 22.0, 30.0), (0.0, 0.0, 75.0), "dark green"),),
        ),
        # ----------------------------------------------------------- on the hooks
        Thing(
            "pliers",
            "a pair of pliers",
            "pliers",
            "box",
            (26.0, 28.0, 140.0),  # hanging, so its length is up
            "red",
            (hook_at("left"), HOOKS[0][1], HOOK_Z_MM - 140.0),
            grip_mm=26.0,
            grip_how="level",
            grip_z_mm=190,
            parts=(Part("box", (16.0, 20.0, 46.0), (0.0, 0.0, -56.0), "steel"),),
        ),
        Thing(
            "spanner",
            "a spanner",
            "spanner",
            "box",
            (22.0, 11.0, 170.0),
            "steel",
            (hook_at("right"), HOOKS[2][1], HOOK_Z_MM - 170.0),
            grip_mm=22.0,
            grip_how="level",
            grip_z_mm=190,
            parts=(Part("box", (36.0, 13.0, 34.0), (0.0, 0.0, -72.0), "steel"),),
        ),
        # -------------------------------------------------------- in the bottom bay
        *[
            Thing(
                f"jar{n}",
                "a jar of screws",
                "jar",
                "cylinder",
                # 95 mm tall, in a bay 110 mm high. Not a free choice: at 70 mm there was a
                # 40 mm slot between the top of a jar and the underside of the board, and from
                # the default view you could see straight through it onto the top of the jar in
                # the back row - 39 per cent of it, measured. A jar tall enough to close that
                # slot hides the one behind it, which is what the back row is for. A jam jar is
                # this tall anyway; 70 was the guess.
                (32.0, 32.0, 85.0),
                "glass",
                (x, y, bay0),
                grip_mm=32.0,
                grip_how="level",
                grip_z_mm=BAY_GRIP_Z_MM[0],
                parts=(
                    # The screws inside. Glass on its own is a pale cylinder against a pale
                    # shelf; what makes a jar of screws read as one at a glance is the dark
                    # mass filling the bottom two thirds of it (NFR1).
                    Part("cylinder", (26.0, 26.0, 55.0), (0.0, 0.0, -13.0), "brass"),
                    Part("cylinder", (34.0, 34.0, 10.0), (0.0, 0.0, 47.5), "grey"),
                ),
            )
            for n, x, y in (
                (1, 250.0, FRONT_ROW_Y_MM),
                (2, 300.0, FRONT_ROW_Y_MM),
                (3, 350.0, FRONT_ROW_Y_MM),
                # The back row: behind jar2, the same jar, and the one thing the robot has not
                # been told about. Act 4. Its place across is worked out from the eye it has to
                # hide from, not copied from the jar in front of it - see `behind_from_the_eye`.
                (4, behind_from_the_eye(300.0, FRONT_ROW_Y_MM, BACK_ROW_Y_MM), BACK_ROW_Y_MM),
            )
        ],
        Thing(
            "glue1",
            "a tube of glue, capped",
            "glue",
            "cylinder",
            (26.0, 26.0, 84.0),
            "white",
            (420.0, FRONT_ROW_Y_MM, bay0),
            grip_mm=26.0,
            grip_how="level",
            grip_z_mm=BAY_GRIP_Z_MM[0],
            parts=(Part("cylinder", (16.0, 16.0, 12.0), (0.0, 0.0, 48.0), "red"),),
        ),
        Thing(
            "glue2",
            "a tube of glue, open",
            "glue",
            "cylinder",
            (26.0, 26.0, 84.0),
            "white",
            (460.0, FRONT_ROW_Y_MM, bay0),
            grip_mm=26.0,
            grip_how="level",
            grip_z_mm=BAY_GRIP_Z_MM[0],
            # No cap, and a dark mouth where it should be: the pair differ by a state, not by
            # any word either of them has.
            parts=(Part("cylinder", (12.0, 12.0, 5.0), (0.0, 0.0, 43.0), "shadow"),),
        ),
        # ------------------------------------------------------- in the top bay
        Thing(
            "odds",
            "a box of odds and ends",
            "odds",
            "box",
            (60.0, 50.0, 70.0),
            "brown card",
            (350.0, FRONT_ROW_Y_MM, bay1),
            grip_mm=60.0,
            grip_how="level",
            grip_z_mm=BAY_GRIP_Z_MM[1],
        ),
    ]


def by_id(items: list[Thing]) -> dict[str, Thing]:
    return {t.id: t for t in items}


def kind_counts(items: list[Thing]) -> dict[str, int]:
    """How many of each kind: four jars, two tubes of glue, one of everything else."""
    out: dict[str, int] = {}
    for t in items:
        out[t.kind] = out.get(t.kind, 0) + 1
    return out


def is_unique(items: list[Thing], thing_id: str) -> bool:
    """Is this thing one of a kind - does a word for it exist at all?"""
    return kind_counts(items)[by_id(items)[thing_id].kind] == 1


def hook_at(side: str) -> float:
    """Where a thing hanging on a hook on this side of the unit sits, across the bench."""
    return SHELF_FROM_MM[0] - HOOK_OUT_MM if side == "left" else SHELF_TO_MM[0] + HOOK_OUT_MM


def hooks() -> list[tuple[str, float, float, float]]:
    """The three hooks: which side, and where each one is. One is left empty, to hang on."""
    return [(side, hook_at(side), y, HOOK_Z_MM) for side, y in HOOKS]


def reach_report() -> list[str]:
    """Who can reach what, from the measured rings. The gate on the layout."""
    all_arms = arms()
    out = ["reach, from the measured rings and the fan:"]
    for t in things():
        if not t.graspable:
            who = ["(never picked up)"]
        else:
            who = [a.id for a in all_arms if a.can_reach(t)] or ["NOBODY"]
        r = " ".join(
            f"{a.id[0]}:r{a.radius_to(*t.place_mm[:2]):.0f}/b{abs(a.bearing_to(*t.place_mm[:2])):.0f}"
            for a in all_arms
        )
        out.append(f"  {t.id:<12} {t.grip_how:<5} z{t.grip_z_mm:<4} {', '.join(who):<24} {r}")
    for (how, z), (near, far) in sorted(RING_MM.items()):
        out.append(f"  ring {how:<5} z{z:<4} {near:.0f} to {far:.0f} mm ahead, less off to a side")
    for a in all_arms:
        out.append(f"  {a.id}: can hold a thing out to the person = {a.reaches_person}")
    return out
