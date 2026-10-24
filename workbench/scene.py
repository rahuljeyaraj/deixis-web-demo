"""The workbench: a bench, one arm on a rail, a three-bay shelf unit, and the things.

One source of truth for the geometry (`workbench-requirements.md` FR5). The physics scene
(`web/model/workbench_scene.xml`) is generated from this by `scripts/make_workbench_scene.py`,
and the page draws what the physics holds, so there is one set of numbers and no second copy
to drift.

## The frame

Millimetres. **x across, to the person's right; y away from the person; z up from the bench
top.** The centre line is x = 350. A thing's own frame is the same three axes before it is
turned: **across, deep, up**. The world the physics runs in is metres with +X away from the
person, and `scripts/make_workbench_scene.py` holds the only copy of that conversion.

## What stage 0 measured

Everything here rests on two measurements made on 24 Oct, both reported in full:

- `out/stage0-grip-width.txt` - the widest thing the gripper holds through a lift and a move.
- `out/stage0-bay-ring.txt` - the reach ring at every height something here is gripped at.

The short version: the gripper holds far more than the plan feared, reaching **level** into the
bottom bay stops at 280 mm from the arm, and reaching **down** onto a thing dies above 100 mm.

## Where the arm stands, and why it matters so much

**One arm, on a rail across the front of the bench** (rahul, 24 Oct). An arm cannot work closer
in than 135 mm from itself, nor more than 100 degrees off the way it faces, so it drags a dead
disc and a dead sector about with it wherever it stands - and no amount of placing one well, or
two well, makes those go away. Sliding one does: it can always come and stand square in front
of whatever it is reaching for. 100 per cent of all three bays against 60 to 81 for two arms
standing still, measured over the same points.

The rail is sunk into the bench so the carriage's top is flush, because every ring below was
measured with the arm's base at z = 0.

It costs a kind of pointing, and the cost is written down rather than hidden: with two arms a
person can say *you*, or *the other one*, and mean something by it. With one there is nothing to
choose between. Weighed and spent.

## The shelf

**Three bays** (rahul, 24 Oct). The sweep found a third reachable after all: level reach at
299 mm is 90 to 430 mm, because an arm reaching up can fold in far closer than one reaching
down low. 55, 177 and 299 are the three heights, and each was swept.

**The whole back edge** (rahul, 24 Oct): 700 mm across, wall to wall, rather than 440 in the
middle. The strips it used to leave at each end were bench nothing could use - further from the
person than the shelf itself, and reached over nothing. Each bay now has 676 mm of clear width,
and because the arm slides, every millimetre of that is in reach.

**No hook rail** (rahul, 24 Oct): a thing is put on the bench or on a shelf, and nothing else.

## The one asymmetry

The shelf is two rows deep and the back row is real. `jar4` stands behind `jar2`, both the same
jar, and from the default view it cannot be seen - the jar in front hides it, and the board
above cuts off any look over the top. Orbit the camera round and it appears. That is the
missing information Act 4 is built on, and it is honest: a real robot has exactly this problem.
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
#: Every height here is one something in this world is really gripped at, and was swept at that
#: height - nothing is interpolated. Two are **absent on purpose**, because the sweep found
#: nothing at them: reaching down at 130 mm, and anything higher. Asking this table for a
#: downward grip up there raises a `KeyError`, which is the answer.
#:
#: Note how the inner edge collapses with height - 280, 230, 90 - while the outer edge shrinks.
#: Reaching up, an arm folds in close; reaching down and low, it has to stretch out flat.
RING_MM: dict[tuple[str, int], tuple[float, float]] = {
    ("level", 55): (280.0, 470.0),
    ("level", 130): (260.0, 480.0),
    ("level", 177): (230.0, 475.0),
    ("level", 299): (90.0, 430.0),
    ("down", 35): (135.0, 325.0),
    ("down", 70): (130.0, 300.0),
}

#: How much of that outward reach is lost per degree the shoulder turns off forward.
#:
#: Reach is not a circle. The fall is close enough to straight, and close enough to the same at
#: every height swept (0.48 to 0.51 mm per degree), that one number covers all of them - and it
#: is the same 0.48 that `end_demo` measured for the fingers-down ring by a different route.
REACH_FALL_MM_PER_DEG = 0.48

#: How far either side of its facing an arm can work, degrees. The SO-101's `shoulder_pan`
#: stops at +/-110 (`web/model/so101/so101.xml`); 100 is `end_demo`'s measured working limit.
PAN_LIMIT_DEG = 100.0

#: The widest thing the gripper holds through a lift and a move, millimetres.
#:
#: Measured on 24 Oct by `web/tests/probes/grip-width.probe.test.ts`: boxes of increasing width
#: picked up off the bench, carried 220 mm, put down and let go, each at nine aims so no width
#: fails for a badly aimed grip. The jaws open to 107 mm tip to tip at the gripper's own limit,
#: which puts the ceiling near 99; **96 mm** came through the whole move with the grip assist
#: the demo uses, and **76 mm** with the weld switched off, on the jaws and friction alone.
#:
#: 76 is the number used here, because it is the one that does not depend on the weld. The plan
#: assumed 32 mm, from the 30 mm blocks of the cubes demo - the only width that demo was ever
#: asked for. Knowing the real figure is what lets a jar be a jar and a mug be a mug.
GRIP_WIDEST_MM = 76.0
GRIP_WIDEST_WELDED_MM = 96.0

#: How tall a thing must be before it can be gripped inside a bay, millimetres.
#:
#: Not swept - reasoned from the jaw geometry and then checked against the sweep. The hand comes
#: in level and the pads stand about 20 mm above and below the fingers, so the jaws need that
#: much of a thing's side to close on, clear of the board it stands on. A flat thing in a bay
#: has no way in at all: a board above rules out coming down, the board below rules out coming
#: in level. Stage 2 drives the real jaws at every thing and is the gate on this number.
BAY_TALLEST_FLAT_MM = 50.0

# ------------------------------------------------------------------- the bench

#: The bench top: the person stands at the near edge, the shelf unit stands at the far one.
BENCH_FROM_MM = (0.0, -50.0)
BENCH_TO_MM = (700.0, 500.0)
BENCH_THICK_MM = 18.0
CENTRE_X_MM = 350.0

#: The shelf unit's footprint: **the whole of the back edge** (rahul, 24 Oct), 700 mm across.
#:
#: It was 440 mm in the middle, which left 130 mm of bare bench at each end - bench that is no
#: use for anything, because it is further from the person than the shelf and the arm has to
#: reach over nothing to get there. Running the unit wall to wall turns both of those strips
#: into shelf, and the bays go from 416 mm of clear width to 676.
SHELF_FROM_MM = (0.0, 350.0)
SHELF_TO_MM = (700.0, 500.0)
SHELF_SIDE_MM = 12.0
SHELF_BACK_MM = 10.0
#: Each board over a bay: the height of its underside, and its thickness. Three bays, so three
#: boards - the last is the unit's lid, and nothing stands on it, because reaching down onto
#: anything above about 100 mm is not a pose this arm has.
BOARD_Z_MM = ((110.0, 12.0), (232.0, 12.0), (354.0, 12.0))
#: A bay's clear opening across and in depth: inside the uprights, in front of the back panel.
BAY_CLEAR_X_MM = (SHELF_FROM_MM[0] + SHELF_SIDE_MM, SHELF_TO_MM[0] - SHELF_SIDE_MM)
BAY_CLEAR_Y_MM = (SHELF_FROM_MM[1], SHELF_TO_MM[1] - SHELF_BACK_MM)

#: Where a row of things stands in a bay, away from the person. 50 mm apart: close enough that
#: the row in front hides the row behind, far enough that nothing touches.
FRONT_ROW_Y_MM = 385.0
BACK_ROW_Y_MM = 455.0

#: Where a thing is put for the person to take: in the middle of the bench, just in front of
#: the rail, as near them as the arm can work.
#:
#: Not held out in mid-air. With the rail across the front, everything on the person's side of
#: it is **behind** the arm - past the shoulder's 100 degrees - so the arm cannot hold anything
#: out over that edge at all. It puts the thing down where they can reach it instead, which is
#: what happens at a real bench anyway.
GIVE_MM = (350.0, 150.0, 40.0)

#: The robot's own camera: a point above and between the arms. What it can see is computed from
#: here rather than stored, so the list the model is given cannot quietly disagree with the
#: scene (FR4, stage 2).
ROBOT_EYE_MM = (350.0, 60.0, 460.0)

#: Where the person's eye is for the page's default view, and what it looks at.
#:
#: Not a camera preference: the whole of Act 4 is that the robot has not been told about the jar
#: in the back row and the judge finds it by moving the camera. So the default view has to hide
#: it and a small orbit has to reveal it, and `behind_from_the_eye` puts the jar on this eye's
#: own line. `web/tests/probes/visible.probe.test.ts` casts rays at every thing from here and
#: is the gate on both.
#:
#: It stands well back. At 450 mm the bench's near edge was only 400 mm from the eye and
#: everything on it loomed - a mug in the front corner filled a quarter of the frame while the
#: shelf sat small in the middle. Standing off 700 mm costs nothing and lets the whole bench be
#: one scene rather than a foreground and a background.
EYE_MM = (350.0, -820.0, 620.0)
LOOK_AT_MM = (350.0, 240.0, 150.0)


def behind_from_the_eye(x_mm: float, from_y: float, to_y: float) -> float:
    """Where a thing has to stand at `to_y` to be exactly behind one at (`x_mm`, `from_y`).

    Straight behind across is not the same as straight behind as seen from a particular place.
    The jars stand left of centre and the eye is on the centre line, so a jar put at the same x
    in the back row leans out from behind the one in front of it and can be seen past its edge -
    39 per cent of it, when that was measured. One line of similar triangles fixes it, and ties
    the fourth jar's place to the eye it is hidden from, which is the honest way round.
    """
    return x_mm + (x_mm - EYE_MM[0]) * (to_y - from_y) / (from_y - EYE_MM[1])


def bay_z_mm(bay: int) -> tuple[float, float]:
    """A bay's clear opening: the height of its floor, and of the board over it."""
    floor = 0.0 if bay == 0 else BOARD_Z_MM[bay - 1][0] + BOARD_Z_MM[bay - 1][1]
    return (floor, BOARD_Z_MM[bay][0])


#: The three bays, bottom first, as (floor, ceiling).
BAYS = tuple(bay_z_mm(n) for n in range(len(BOARD_Z_MM)))
#: The height the hand works at inside each bay: the middle of its clear opening, rounded to a
#: height the ring was really swept at.
BAY_GRIP_Z_MM = (55, 177, 299)


# --------------------------------------------------------------------- the arm

#: The rail the arm rides on: how far back from the person it runs, and how far along the bench
#: the carriage can travel.
#:
#: **Across the front, not down a side** (rahul, 24 Oct). Down a side it would be hopeless: the
#: far end of the shelf would be 498 mm away and the arm reaches 470, so half the unit could
#: never be touched from any position on the rail. Across the front, the arm can always slide to
#: stand **square in front of** whatever it is reaching for, which makes every bay reachable
#: along its whole width - the bearing is zero and the only thing that matters is how far back
#: the rail is.
#:
#: How far back it sits is then the only real choice, and it is a squeeze from both ends. The
#: bottom bay's ring - the tightest in the world, 280 to 470 mm - has to swallow the distance to
#: **both** rows of the shelf, which wants the rail near the person; and everything the arm
#: works on has to be at least 135 mm from it, which wants it far from the bench. 20 mm back
#: puts the shelf's rows at 365 and 435 mm, both comfortably inside, and leaves the bench
#: usable from 75 mm onwards - 265 mm of depth for the loose tools to lie in.
RAIL_Y_MM = 20.0
RAIL_FROM_MM = 70.0
RAIL_TO_MM = 630.0
#: The carriage is recessed, so the arm's base still sits at z = 0 - which is where every ring
#: in `RING_MM` was measured from. A rail standing proud of the bench would raise the arm, and
#: every one of those measurements would be 25 mm out.
RAIL_SUNK_MM = 16.0
#: How wide the trough is. 54 mm was exactly the carriage plus a hair, and a rail you cannot see
#: is a rail a judge does not believe in (rahul, 24 Oct): at 54 the only part of it left showing
#: was its top face, which lies in the same plane as the bench top. 80 mm leaves 30 mm each side
#: of the carriage for the two guide rails to show in, and the bench top is written as two slabs
#: with the trough between them so that the trough is really a trough.
RAIL_WIDE_MM = 80.0
#: The guide rails the carriage rides on: how far either side of the middle, and how thick.
RAIL_GUIDE_Y_MM = 30.0
RAIL_GUIDE_MM = 13.0


@dataclass(frozen=True)
class Arm:
    """The SO-101, on its rail. Pointable: the person may mean *you*."""

    id: str
    rail_y_mm: float = RAIL_Y_MM
    rail_from_mm: float = RAIL_FROM_MM
    rail_to_mm: float = RAIL_TO_MM
    #: Which way it faces from the carriage, degrees, 0 being straight at the shelf. The
    #: shoulder turns `PAN_LIMIT_DEG` either side of this.
    facing_deg: float = 0.0

    def base_mm_at(self, slide_x: float) -> tuple[float, float]:
        return (slide_x, self.rail_y_mm)

    def bearing_from(self, slide_x: float, x: float, y: float) -> float:
        """How far a spot lies off the arm's forward from a carriage position, degrees.

        Forward is +y, away from the person towards the shelf. The same convention as
        `end_demo.scene`: a positive turn swings forward towards the person's left, which is a
        step of -x, which is why the x term is negated.
        """
        world = math.degrees(math.atan2(-(x - slide_x), y - self.rail_y_mm))
        return (world - self.facing_deg + 180.0) % 360.0 - 180.0

    def margin_from(self, slide_x: float, x: float, y: float, how: str, z: int) -> float:
        """How much room to spare this carriage position has, millimetres.

        Negative when it cannot be done at all: how far outside the ring, or how far past the
        shoulder's limit, measured in millimetres so the two can be compared.
        """
        bearing = abs(self.bearing_from(slide_x, x, y))
        if bearing > PAN_LIMIT_DEG:
            return -(bearing - PAN_LIMIT_DEG) * 10.0
        near, far = RING_MM[(how, z)]
        r = math.hypot(x - slide_x, y - self.rail_y_mm)
        return min(r - near, far - REACH_FALL_MM_PER_DEG * bearing - r)

    def slide_for(self, x: float, y: float, how: str, z: int) -> float | None:
        """Where to slide to, to work at this spot with the most room to spare - or None.

        Searched rather than solved: the ring's outer edge falls with the bearing, so the best
        place on the rail is not simply the one straight in front, and 5 mm steps over half a
        metre of rail are cheap. Stage 2 drives the real solver from here.
        """
        best, best_margin = None, 0.0
        here = self.rail_from_mm
        while here <= self.rail_to_mm:
            margin = self.margin_from(here, x, y, how, z)
            if margin > best_margin:
                best, best_margin = here, margin
            here += 5.0
        return best

    def reaches(self, x: float, y: float, how: str, z: int) -> bool:
        """Can the arm work at this spot from anywhere on its rail?"""
        return self.slide_for(x, y, how, z) is not None

    def can_reach(self, thing: Thing) -> bool:
        x, y, _ = thing.place_mm
        return self.reaches(x, y, thing.grip_how, thing.grip_z_mm)

    @property
    def reaches_person(self) -> bool:
        """Can it put a thing down where the person can take it?"""
        return self.reaches(GIVE_MM[0], GIVE_MM[1], "down", 70)


def arm() -> Arm:
    """The one arm, on a rail across the front of the bench.

    **One arm, not two** (rahul, 24 Oct). Two SO-101s standing at the sides of the bench, facing
    each other, left 40 per cent of the bottom bay's back row out of reach of either, and 49 per
    cent of the bench top out of reach of both: each arm drags a dead disc about with it and
    cannot turn more than 100 degrees off the way it faces, and no amount of placing them well
    makes those discs go away. Sliding one arm does. Measured over the same points:

    | | two arms | one on a rail |
    |---|---|---|
    | the bottom bay, front row | 79% | 100% |
    | the bottom bay, back row | 60% | 100% |
    | the middle bay | 81% | 100% |
    | the bench top | 51% | 80% |

    What it costs is a kind of pointing: with two arms a person can say *you*, or *the other
    one*, and mean something by it. With one there is nothing to choose between. That was
    weighed and spent.

    The arm faces the shelf and does not turn on its carriage, because it never needs to: it can
    always slide to stand square in front of whatever it is reaching for.
    """
    return Arm("arm")


def arms() -> list[Arm]:
    """The arm, as a list, for the places that walk over them."""
    return [arm()]


# ------------------------------------------------------------------ the things


@dataclass(frozen=True)
class Part:
    """A shape bolted to a thing, so that its silhouette reads without a label.

    `at_mm` is measured from the thing's own centre, in the thing's own frame - across, deep,
    up - and turns with it. A part has no joint of its own: it moves with the thing, which is
    what makes a hammer a hammer rather than a stick lying beside a block.

    `tilt_deg` turns it in the across-up plane (a hammer's claw, a knife's blade); `turn_deg`
    about its own up (a pair of pliers' splayed legs). For a cylinder or a capsule, `axis` says
    which way its length runs before either angle is applied.
    """

    shape: str  #: box, cylinder, capsule, sphere or ellipsoid
    size_mm: tuple[float, float, float]  #: across, deep, up
    at_mm: tuple[float, float, float]
    colour: str
    axis: str = "up"
    tilt_deg: float = 0.0
    turn_deg: float = 0.0


def _span(
    size_mm: tuple[float, float, float],
    at_mm: tuple[float, float, float],
    tilt_deg: float = 0.0,
    turn_deg: float = 0.0,
) -> tuple[float, float, float, float]:
    """One shape's footprint in the thing's own frame: across from, across to, deep from, to.

    Turned as the shape is turned. A tilt in the across-up plane makes a shape wider across by
    the part of its height that has swung over; a turn about up then mixes across and deep. The
    answer is the box round the turned box, which is a little bigger than the shape itself - and
    that is the right way to be wrong when the question is whether two things touch.
    """
    across, deep, up = size_mm
    tilt = math.radians(tilt_deg)
    half_across = abs(across / 2 * math.cos(tilt)) + abs(up / 2 * math.sin(tilt))
    turn = math.radians(turn_deg)
    dx = abs(half_across * math.cos(turn)) + abs(deep / 2 * math.sin(turn))
    dy = abs(half_across * math.sin(turn)) + abs(deep / 2 * math.cos(turn))
    return (at_mm[0] - dx, at_mm[0] + dx, at_mm[1] - dy, at_mm[1] + dy)


def _rise(
    size_mm: tuple[float, float, float], at_mm: tuple[float, float, float], tilt_deg: float
) -> tuple[float, float]:
    """One shape's bottom and top in the thing's own frame. A turn about up cannot change it."""
    tilt = math.radians(tilt_deg)
    half = abs(size_mm[2] / 2 * math.cos(tilt)) + abs(size_mm[0] / 2 * math.sin(tilt))
    return (at_mm[2] - half, at_mm[2] + half)


@dataclass(frozen=True)
class Thing:
    """One thing in the world.

    `size_mm` is the bounding box of its **first** shape in its own frame - across, deep, up -
    whatever shape that is, and `axis` says which way a cylinder lies. Writing it that way
    rather than in the shape's own frame means there is no orientation to get wrong, and that
    the page can put a name under the cursor without knowing anything about shapes.

    `yaw_deg` turns the whole thing on the bench. Nothing on a real bench is square to its edge,
    and the first build of this world had every tool lying parallel to one, which read as a shop
    display rather than as a bench somebody works at.
    """

    id: str
    what: str  #: what it is, in the fewest words that name it on the page
    #: Things of the same kind cannot be told apart by any word. This is the heart of the demo:
    #: four jars, two tins, two screwdrivers, two offcuts, two tubes of glue.
    kind: str
    shape: str
    size_mm: tuple[float, float, float]
    colour: str
    #: Where it stands: across, away, and the height its bottom rests at.
    at_mm: tuple[float, float, float]
    #: How wide the part the gripper closes on is. Checked against `GRIP_WIDEST_MM`.
    grip_mm: float
    #: How the hand comes at it, and the height the ring for that was swept at.
    grip_how: str = "down"
    grip_z_mm: int = 35
    yaw_deg: float = 0.0
    axis: str = "up"
    #: False for a thing nobody would ask for but anybody might point at.
    wanted: bool = True
    #: False for a thing never picked up at all, only put into.
    graspable: bool = True
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
    def is_round(self) -> bool:
        """Does it look the same from every side? A jar, a tin, a pot - not a mug, nor a hammer.

        Asked of the shapes rather than declared, because a declared answer is a second copy of
        the geometry and can disagree with it. A thing is round when every shape it is built
        from is round, stands up and stands on the middle: a lid, a label, a band of screws in
        the bottom. One handle, one tail of tape, one claw, and the answer is no.
        """
        round_ = ("cylinder", "sphere")
        if self.shape not in round_ or self.axis != "up":
            return False
        return all(
            p.shape in round_
            and p.axis == "up"
            and p.at_mm[0] == 0.0
            and p.at_mm[1] == 0.0
            and p.tilt_deg == 0.0
            for p in self.parts
        )

    @property
    def box_mm(self) -> tuple[float, float, float, float]:
        """Its whole footprint in its own frame: middle across, middle deep, across, deep.

        **Counted, not typed.** Every thing used to carry a footprint written out by hand where
        its parts stuck out past its first shape, and every one of those was too small: the
        screwdriver's said 236 mm for a tool 300 mm long, the chisel's 252 for 296, the hammer's
        102 deep for a claw that reaches 122. So the layout was told there was room where there
        was none, and things stood inside each other - which is what rahul saw.

        The middle is given as well as the size, because nothing here is centred on its own
        footprint: a hammer's head is all at one end. Centring it - which is what doubling each
        offset did - makes a hammer out to be twice as long as it is.
        """
        x0, x1, y0, y1 = _span(self.size_mm, (0.0, 0.0, 0.0))
        for p in self.parts:
            a, b, c, d = _span(p.size_mm, p.at_mm, p.tilt_deg, p.turn_deg)
            x0, x1, y0, y1 = min(x0, a), max(x1, b), min(y0, c), max(y1, d)
        return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)

    @property
    def room_mm(self) -> tuple[float, float]:
        """How much room it takes, across and deep, in its own frame."""
        return self.box_mm[2:]

    @property
    def rises_mm(self) -> tuple[float, float]:
        """How high its lowest and its highest point stand above the bench."""
        middle = self.at_mm[2] + self.tall_mm / 2
        lo, hi = -self.tall_mm / 2, self.tall_mm / 2
        for p in self.parts:
            a, b = _rise(p.size_mm, p.at_mm, p.tilt_deg)
            lo, hi = min(lo, a), max(hi, b)
        return (middle + lo, middle + hi)

    @property
    def top_mm(self) -> float:
        """How high its highest point stands above the bench."""
        return self.rises_mm[1]

    @property
    def grip_above_mm(self) -> float | None:
        """How far above its **own bottom** the jaws close on it, millimetres.

        None for a thing taken from above, which is gripped a few millimetres above whatever it
        rests on whatever that is, and so needs no number of its own.

        For a thing taken level this is the number that lets the arm work a grip out from where
        the thing **now** stands rather than from where it started. `grip_z_mm` is a height
        above the bench, and a jar carried from the bottom bay to the top one is then described
        by a height 244 mm below where it is - which is what sent the arm to the wrong bay
        (rahul, 24 Oct). This is the same measurement with the shelf taken out of it: a jar is
        gripped 55 mm up its own side, in whatever bay it stands, and on the bench.
        """
        return None if self.grip_how != "level" else self.grip_z_mm - self.rises_mm[0]

    @property
    def place_mm(self) -> tuple[float, float, float]:
        """Its **centre**: what the physics wants for a body, and what the eye looks at."""
        return (self.at_mm[0], self.at_mm[1], self.at_mm[2] + self.tall_mm / 2)

    @property
    def in_bay(self) -> int | None:
        """Which bay it stands in, or None for the bench.

        All three of x, y and z have to say so. The oil bottle stands on the bench past the end
        of the unit: without the x test it looked like a thing in the bottom bay that was far
        too tall for it, which is the one thing the oil is not.
        """
        x, y, z = self.at_mm
        if not (BAY_CLEAR_X_MM[0] <= x <= BAY_CLEAR_X_MM[1]):
            return None
        if not (BAY_CLEAR_Y_MM[0] <= y <= BAY_CLEAR_Y_MM[1]):
            return None
        for bay, (floor, _) in enumerate(BAYS):
            if abs(z - floor) < 1.0:
                return bay
        return None


#: Where each loose thing lies on the bench: across, away from the person, and how it is turned.
#:
#: **Solved, not typed.** Every one of these had to come out of a search rather than a judgement,
#: because the bench is 700 by 272 mm of usable top and the ten things on it take up more than
#: half of what is left once the rail, the hand-over spot and the arm's parking place are out.
#:
#: The search keeps a placing only when the arm can reach the thing from somewhere on its rail
#: with 20 mm to spare, no two footprints come within 14 mm of each other, nothing stands where
#: the arm parks or where a thing is handed over, and nothing is within 9 degrees of square to an
#: edge. Of 150 searches the best leaves 15 mm between the nearest two things.
#:
#: The footprints it works with are the **counted** ones (`Thing.box_mm`), which is the fix for
#: what rahul saw: the hand-written footprints were all too small, so the old layout was told
#: there was room where there was none and things stood inside each other.
BENCH_PLACES: dict[str, tuple[float, float, float]] = {
    "hammer": (216.0, 173.0, 169.0),
    "offcut1": (597.0, 163.0, 112.0),
    "screwdriver1": (157.0, 290.0, 13.0),
    "screwdriver2": (439.0, 248.0, -108.0),
    "chisel": (57.0, 126.0, 101.0),
    "pliers": (607.0, 284.0, -13.0),
    "knife": (471.0, 173.0, 75.0),
    "pot": (371.0, 294.0, -19.0),
    "oil": (518.0, 106.0, 136.0),
    "blades": (281.0, 259.0, -170.0),
}


def _jar(n: int, x: float, y: float) -> Thing:
    """One of the four jars of screws.

    The same jar four times over: no word separates them, which is the whole demo. 93 mm to the
    top of the lid, which is what fills the bay - at 70 there was a 40 mm slot between a jar's
    lid and the board above it, and the back row could be seen straight over the front one.
    """
    return Thing(
        f"jar{n}",
        "a jar of screws",
        "jar",
        "cylinder",
        (64.0, 64.0, 78.0),
        "glass",
        (x, y, bay_z_mm(0)[0]),
        grip_mm=64.0,
        grip_how="level",
        grip_z_mm=BAY_GRIP_Z_MM[0],
        parts=(
            # The screws inside. Glass on its own is a pale cylinder against a pale shelf, and
            # what makes a jar of screws read as one at a glance is the dark mass in the bottom
            # of it (NFR1).
            Part("cylinder", (54.0, 54.0, 46.0), (0.0, 0.0, -14.0), "brass"),
            Part("cylinder", (56.0, 56.0, 8.0), (0.0, 0.0, 36.0), "glass"),
            Part("cylinder", (66.0, 66.0, 15.0), (0.0, 0.0, 45.0), "jar lid"),
        ),
    )


#: The mug's handle: a ring of beads, and how many of them.
#:
#: **Thirteen, not five** (rahul, 24 Oct: "the mug's handle is now 3 floating balls"). The handle
#: is an arc of beads because a torus is not one of MuJoCo's shapes, and at five they did not
#: touch: 18 mm beads 57 degrees apart on a 23 mm arc stand 22 mm from the middle of one to the
#: next, so there were gaps between them and from the default view only three of the five were in
#: sight. Thirteen 14 mm beads, 19.7 degrees apart on a 25 mm arc, stand 8.6 mm from one middle to
#: the next - well under a bead's width - so they run together into one smooth handle from any
#: side. Both ends of the arc finish inside the mug's own wall, which is where a handle joins a
#: mug.
MUG_HANDLE_BEADS = 13
MUG_HANDLE_ARC_DEG = 118.0


def _mug(x: float, y: float, yaw: float, z: float, how: str, grip_z: int) -> Thing:
    """The mug: a body, a dark inside so it reads as open rather than solid, a lip, a handle.

    **One mug** (rahul, 24 Oct). There were two, as a second pair no word separates; the jars,
    the tins, the glue and the screwdrivers already make that point four times over, and a
    second mug was one more thing on a bench that was too full.
    """
    step = 2 * MUG_HANDLE_ARC_DEG / (MUG_HANDLE_BEADS - 1)
    handle = tuple(
        Part(
            "sphere",
            (14.0, 14.0, 14.0),
            (
                36.0 + 25.0 * math.cos(math.radians(MUG_HANDLE_ARC_DEG - k * step)),
                0.0,
                2.0 + 25.0 * math.sin(math.radians(MUG_HANDLE_ARC_DEG - k * step)),
            ),
            "mug",
        )
        for k in range(MUG_HANDLE_BEADS)
    )
    return Thing(
        "mug",
        "a mug",
        "mug",
        "cylinder",
        (76.0, 76.0, 90.0),
        "mug",
        (x, y, z),
        grip_mm=76.0,
        grip_how=how,
        grip_z_mm=grip_z,
        yaw_deg=yaw,
        parts=(
            Part("cylinder", (62.0, 62.0, 74.0), (0.0, 0.0, 10.0), "shadow"),
            Part("cylinder", (80.0, 80.0, 6.0), (0.0, 0.0, 42.0), "mug"),
            *handle,
        ),
    )


def _tin(n: int, x: float, y: float, yaw: float) -> Thing:
    """One of the two tins of varnish, in the middle bay.

    A second set of things no word separates, at a different size from the jars and on a
    different board, so that a judge meets the same difficulty twice without it being the same
    trick twice. **Two, not three** (rahul, 24 Oct): with four jars already standing in the bay
    below, a third tin added crowding and not one new thing to say.
    """
    return Thing(
        f"tin{n}",
        "a tin of varnish",
        "tin",
        "cylinder",
        (72.0, 72.0, 76.0),
        "tin",
        (x, y, bay_z_mm(1)[0]),
        grip_mm=72.0,
        grip_how="level",
        grip_z_mm=BAY_GRIP_Z_MM[1],
        yaw_deg=yaw,
        parts=(
            Part("cylinder", (74.0, 74.0, 34.0), (0.0, 0.0, -4.0), "tin label"),
            Part("cylinder", (76.0, 76.0, 8.0), (0.0, 0.0, 34.0), "tin"),
            Part("cylinder", (58.0, 58.0, 4.0), (0.0, 0.0, 38.0), "shadow"),
        ),
    )


def _screwdriver(n: int, x: float, y: float, yaw: float) -> Thing:
    """One of the two screwdrivers: handle, collar, shaft and a flat tip.

    Two of them, so that "the screwdriver" settles nothing on its own. 193 mm end to end.
    """
    return Thing(
        f"screwdriver{n}",
        "a screwdriver",
        "screwdriver",
        "cylinder",
        (86.0, 32.0, 32.0),
        "handle blue",
        # 2.5, not 16: a thing's `at_mm` height is where its **first** shape's bottom goes,
        # and a screwdriver's fattest part is its handle, 37 mm across against the 32 of the
        # shaft's collar. Standing it on the collar hung the handle 13 mm in the air, and once
        # the physics was running every screwdriver dropped and rolled on landing.
        (x, y, 2.5),
        grip_mm=32.0,
        yaw_deg=yaw,
        axis="across",
        parts=(
            Part("cylinder", (16.0, 37.0, 37.0), (28.0, 0.0, 0.0), "handle blue", "across"),
            Part("cylinder", (96.0, 9.0, 9.0), (90.0, 0.0, 0.0), "steel", "across"),
            Part("box", (16.0, 10.0, 3.0), (142.0, 0.0, 0.0), "steel"),
        ),
    )


def _offcut(n: int, x: float, y: float, yaw: float, z: float = 0.0, on_edge: bool = False) -> Thing:
    """One of the two offcuts: what the bench is about, and a second of the same piece.

    The second stands **on its edge** in the top bay, which is how a spare bit of board is
    really kept and is also the only way it can live on a shelf at all: lying flat it is 36 mm
    tall, and the jaws need 50 mm of a thing's side to close on inside a bay.
    """
    size = (132.0, 36.0, 76.0) if on_edge else (132.0, 76.0, 36.0)
    line = (0.0, -17.0, 20.0) if on_edge else (0.0, 12.0, 18.0)
    kerf = (48.0, 0.0, 35.0) if on_edge else (48.0, -14.0, 15.0)
    return Thing(
        f"offcut{n}",
        "an offcut of wood",
        "offcut",
        "box",
        size,
        "pale wood",
        (x, y, z),
        grip_mm=36.0 if on_edge else 76.0,
        grip_how="level" if on_edge else "down",
        grip_z_mm=BAY_GRIP_Z_MM[2] if on_edge else 35,
        yaw_deg=yaw,
        parts=(
            Part("box", (120.0, 2.0, 1.0), line, "pencil"),
            Part("box", (3.0, 30.0, 7.0), kerf, "shadow"),
        ),
    )


def _pliers(x: float, y: float, yaw: float) -> Thing:
    """The pliers: a pivot, **two** legs and two jaws.

    They had three legs (rahul, 24 Oct). A thing's first shape stands at its own middle and
    cannot be moved off it, so the first shape was a leg on the middle line and the two real
    legs were parts bolted either side of it - three identical red bars where there should be
    two. Now the first shape is the pivot, which does belong on the middle, and everything else
    hangs off it in pairs: the thing's origin is the pivot pin, legs behind, jaws in front.
    """
    return Thing(
        "pliers",
        "a pair of pliers",
        "pliers",
        "cylinder",
        (24.0, 24.0, 20.0),
        "steel",
        (x, y, 0.0),
        grip_mm=44.0,
        yaw_deg=yaw,
        parts=(
            Part("box", (100.0, 14.0, 16.0), (-56.0, 15.0, -1.0), "red", turn_deg=-9.0),
            Part("box", (100.0, 14.0, 16.0), (-56.0, -15.0, -1.0), "red", turn_deg=9.0),
            Part("box", (58.0, 11.0, 12.0), (32.0, 6.0, -1.0), "steel", turn_deg=4.0),
            Part("box", (58.0, 11.0, 12.0), (32.0, -6.0, -1.0), "steel", turn_deg=-4.0),
        ),
    )


def things() -> list[Thing]:
    """Everything on the bench and in the unit: twenty-one things.

    **Twenty-one of them, not twenty-six** (rahul, 24 Oct: the world was too crowded). What went
    is written down where it went from: a third tin, a second mug, the G clamp, the steel rule
    and the roll of masking tape - the tape because the one job the plan gave it was to be hung
    on a hook or laid down, and the hooks went in the pass before this one. Nothing an act needs
    went, and no set of lookalikes was broken up.

    The bench could not have held them anyway. Ten loose things take up 0.078 of the 0.125 square
    metres of bench that is reachable, clear of the rail and clear of where things are handed
    over - and that is before any of them is allowed a gap from its neighbour.

    Five sets that **no word separates** - four jars, two tins, two screwdrivers, two offcuts and
    two tubes of glue that differ only by a cap - and nine things a word does separate, so the
    words-only baseline has something it can win and the demo is not rigged.

    Every one of them is on a real bench, and the properties the demo needs are not added to
    them: they are what a bench is. Several jars you cannot tell apart without reading a label
    you cannot see. A tube with its cap off beside one with its cap on. Sharp things a person
    would really make a rule about. A bottle too tall for the shelf it belongs on. A back row
    nobody has looked at for months.
    """
    bay1, bay2 = (bay_z_mm(1)[0], bay_z_mm(2)[0])
    bay0 = bay_z_mm(0)[0]
    bench = BENCH_PLACES
    return [
        # --------------------------------------- the bottom bay: the jars and the two tubes
        # Together, 100 mm apart, rather than spread across 676 mm of bay: three jars standing
        # in a row is what makes "that one" need a real point, and three jars at arm's length
        # from each other would not.
        _jar(1, 200.0, FRONT_ROW_Y_MM),
        _jar(2, 300.0, FRONT_ROW_Y_MM),
        _jar(3, 400.0, FRONT_ROW_Y_MM),
        # The back row: behind jar2, the same jar, and the one thing the robot has not been
        # told about. Act 4. Its place across is worked out from the eye it has to hide from,
        # not copied from the jar in front of it.
        _jar(4, behind_from_the_eye(300.0, FRONT_ROW_Y_MM, BACK_ROW_Y_MM), BACK_ROW_Y_MM),
        Thing(
            "glue1",
            "a tube of glue, capped",
            "glue",
            "cylinder",
            (30.0, 30.0, 76.0),
            "white",
            (590.0, FRONT_ROW_Y_MM, bay0),
            grip_mm=30.0,
            grip_how="level",
            grip_z_mm=BAY_GRIP_Z_MM[0],
            parts=(
                Part("cylinder", (21.0, 21.0, 10.0), (0.0, 0.0, 41.0), "white"),
                Part("cylinder", (12.0, 12.0, 10.0), (0.0, 0.0, 49.0), "white"),
                Part("cylinder", (18.0, 18.0, 16.0), (0.0, 0.0, 58.0), "red"),
            ),
        ),
        Thing(
            "glue2",
            "a tube of glue, open",
            "glue",
            "cylinder",
            (30.0, 30.0, 76.0),
            "white",
            (640.0, FRONT_ROW_Y_MM, bay0),
            grip_mm=30.0,
            grip_how="level",
            grip_z_mm=BAY_GRIP_Z_MM[0],
            # No cap, and a dark mouth where one should be: the pair differ by a state, and by
            # nothing that either of them has a word for.
            parts=(
                Part("cylinder", (21.0, 21.0, 10.0), (0.0, 0.0, 41.0), "white"),
                Part("cylinder", (12.0, 12.0, 14.0), (0.0, 0.0, 51.0), "white"),
                Part("cylinder", (8.0, 8.0, 4.0), (0.0, 0.0, 58.0), "shadow"),
            ),
        ),
        # -------------------------------------- the middle bay: the two tins and the mug
        _tin(1, 180.0, FRONT_ROW_Y_MM, 12.0),
        _tin(2, 300.0, FRONT_ROW_Y_MM, -31.0),
        _mug(500.0, FRONT_ROW_Y_MM, 24.0, bay1, "level", BAY_GRIP_Z_MM[1]),
        # ------------------------------------------------------------------- the top bay
        Thing(
            "odds",
            "a box of odds and ends",
            "odds",
            "box",
            # 70 deep, not 86: the gripper holds 76 mm and the box is taken by its sides.
            (116.0, 70.0, 74.0),
            "brown card",
            (210.0, FRONT_ROW_Y_MM, bay2),
            grip_mm=70.0,
            grip_how="level",
            grip_z_mm=BAY_GRIP_Z_MM[2],
            yaw_deg=-7.0,
            parts=(
                Part("box", (120.0, 74.0, 8.0), (0.0, 0.0, 33.0), "brown card"),
                Part("box", (62.0, 4.0, 30.0), (0.0, -37.0, 0.0), "shadow"),
            ),
        ),
        _offcut(2, 520.0, FRONT_ROW_Y_MM, 4.0, bay2, on_edge=True),
        # ---------------------------------------------------------------- on the bench top
        _offcut(1, *bench["offcut1"][:2], bench["offcut1"][2]),
        _screwdriver(1, *bench["screwdriver1"]),
        _screwdriver(2, *bench["screwdriver2"]),
        Thing(
            "hammer",
            "a hammer",
            "hammer",
            "cylinder",
            (180.0, 22.0, 22.0),
            "wood",
            (bench["hammer"][0], bench["hammer"][1], 3.0),
            grip_mm=22.0,
            yaw_deg=bench["hammer"][2],
            axis="across",
            parts=(
                Part("cylinder", (62.0, 28.0, 28.0), (-66.0, 0.0, 0.0), "wood", "across"),
                Part("cylinder", (26.0, 80.0, 26.0), (103.0, 0.0, 3.0), "steel", "deep"),
                Part("cylinder", (33.0, 20.0, 33.0), (103.0, -42.0, 3.0), "steel", "deep"),
                Part("box", (26.0, 34.0, 16.0), (103.0, 42.0, 9.0), "steel", tilt_deg=18.0),
            ),
        ),
        _pliers(*bench["pliers"]),
        Thing(
            "chisel",
            "a chisel",
            "chisel",
            "cylinder",
            (84.0, 28.0, 28.0),
            "wood",
            (bench["chisel"][0], bench["chisel"][1], 2.0),
            grip_mm=28.0,
            yaw_deg=bench["chisel"][2],
            axis="across",
            parts=(
                Part("cylinder", (14.0, 32.0, 32.0), (46.0, 0.0, 0.0), "steel", "across"),
                Part("box", (86.0, 22.0, 8.0), (96.0, 0.0, -2.0), "steel"),
                Part("box", (10.0, 24.0, 4.0), (143.0, 0.0, -4.0), "steel"),
            ),
        ),
        Thing(
            "knife",
            "a craft knife",
            "knife",
            "box",
            (112.0, 16.0, 18.0),
            "grey",
            (bench["knife"][0], bench["knife"][1], 1.5),
            grip_mm=18.0,
            yaw_deg=bench["knife"][2],
            parts=(
                Part("box", (34.0, 18.0, 21.0), (-62.0, 0.0, 0.0), "red"),
                Part("box", (54.0, 18.0, 4.0), (0.0, 0.0, 10.0), "red"),
                Part("box", (38.0, 3.0, 15.0), (72.0, 0.0, 1.0), "steel", tilt_deg=-9.0),
            ),
        ),
        Thing(
            "blades",
            "a tin of spare blades",
            "blades",
            "box",
            (62.0, 36.0, 16.0),
            "red",
            (bench["blades"][0], bench["blades"][1], 0.0),
            grip_mm=36.0,
            yaw_deg=bench["blades"][2],
            parts=(
                Part("box", (64.0, 38.0, 4.0), (0.0, 0.0, 7.0), "red dark"),
                Part("box", (40.0, 22.0, 1.0), (0.0, 0.0, 9.0), "white"),
            ),
        ),
        Thing(
            "pot",
            "a pot of loose screws",
            "pot",
            "cylinder",
            (66.0, 66.0, 58.0),
            "grey",
            (bench["pot"][0], bench["pot"][1], 0.0),
            grip_mm=66.0,
            grip_z_mm=70,  # letting go over its rim, not picking it up
            wanted=False,
            graspable=False,
            parts=(
                Part("cylinder", (58.0, 58.0, 10.0), (0.0, 0.0, 18.0), "brass"),
                Part("cylinder", (70.0, 70.0, 7.0), (0.0, 0.0, 26.0), "grey"),
                Part("cylinder", (54.0, 54.0, 4.0), (0.0, 0.0, 28.0), "shadow"),
            ),
        ),
        Thing(
            "oil",
            "a bottle of oil",
            "oil",
            "cylinder",
            (56.0, 56.0, 96.0),
            "dark green",
            (bench["oil"][0], bench["oil"][1], 0.0),
            grip_mm=24.0,
            grip_how="level",
            grip_z_mm=130,  # 150 mm tall: far too tall to be gripped from over the top
            yaw_deg=bench["oil"][2],
            parts=(
                Part("cylinder", (44.0, 44.0, 14.0), (0.0, 0.0, 54.0), "dark green"),
                Part("cylinder", (24.0, 24.0, 28.0), (0.0, 0.0, 74.0), "dark green"),
                Part("cylinder", (28.0, 28.0, 12.0), (0.0, 0.0, 94.0), "red"),
                Part("cylinder", (58.0, 58.0, 26.0), (0.0, 0.0, -10.0), "oil label"),
            ),
        ),
    ]


def by_id(items: list[Thing]) -> dict[str, Thing]:
    return {t.id: t for t in items}


def kind_counts(items: list[Thing]) -> dict[str, int]:
    """How many of each kind: four jars, and two each of four other things."""
    out: dict[str, int] = {}
    for t in items:
        out[t.kind] = out.get(t.kind, 0) + 1
    return out


def is_unique(items: list[Thing], thing_id: str) -> bool:
    """Is this thing one of a kind - does a word for it exist at all?"""
    return kind_counts(items)[by_id(items)[thing_id].kind] == 1


def corners(thing: Thing) -> list[tuple[float, float]]:
    """Its footprint on the bench, turned the way it is turned: four corners.

    A round thing's footprint does not turn, whatever its yaw says - a jar looks the same from
    every side. Turning its bounding square with it made three tins 85 mm apart look as though
    they were standing in each other, because the corners of a turned square reach further out
    than the circle inside it ever does.

    The footprint's own middle is carried round with it, because a thing does not stand in the
    middle of it: turn a hammer and the head goes round the handle, not the other way about.
    """
    mid_x, mid_y, across, deep = thing.box_mm
    a = 0.0 if thing.is_round else math.radians(thing.yaw_deg)
    cos, sin = math.cos(a), math.sin(a)
    x, y, _ = thing.place_mm
    x, y = (x + cos * mid_x - sin * mid_y, y + sin * mid_x + cos * mid_y)
    return [
        (x + cos * dx - sin * dy, y + sin * dx + cos * dy)
        for dx, dy in (
            (-across / 2, -deep / 2),
            (across / 2, -deep / 2),
            (across / 2, deep / 2),
            (-across / 2, deep / 2),
        )
    ]


def overlaps(a: Thing, b: Thing) -> bool:
    """Do two things stand in the same place? A separating-axis test on their footprints.

    Heights are compared first: two things on different boards of the unit are one above the
    other and do not care how their footprints lie. Both ends of both things are wanted for
    that, not a top against a bottom: a pair of pliers' legs lie a millimetre below the pivot
    they turn on, and a part that dips under its own thing's floor has to count.
    """
    a_lo, a_hi = a.rises_mm
    b_lo, b_hi = b.rises_mm
    if a_hi <= b_lo or b_hi <= a_lo:
        return False
    boxes = (corners(a), corners(b))
    for box in boxes:
        for k in range(4):
            edge = (box[(k + 1) % 4][0] - box[k][0], box[(k + 1) % 4][1] - box[k][1])
            axis = (-edge[1], edge[0])
            spans = [[p[0] * axis[0] + p[1] * axis[1] for p in other] for other in boxes]
            if max(spans[0]) <= min(spans[1]) or max(spans[1]) <= min(spans[0]):
                return False
    return True


def clashes(items: list[Thing] | None = None) -> list[tuple[str, str]]:
    """Every pair of things standing in the same place. Should be empty."""
    items = things() if items is None else items
    return [(a.id, b.id) for i, a in enumerate(items) for b in items[i + 1 :] if overlaps(a, b)]


def reach_report() -> list[str]:
    """What the arm can reach, and where it has to slide to do it. The gate on the layout."""
    one = arm()
    out = ["reach, from the measured rings, the fan, and the rail:"]
    for t in things():
        x, y, _ = t.place_mm
        at = one.slide_for(x, y, t.grip_how, t.grip_z_mm)
        if at is None:
            out.append(f"  {t.id:<13} {t.grip_how:<5} z{t.grip_z_mm:<4} NOWHERE ON THE RAIL")
            continue
        out.append(
            f"  {t.id:<13} {t.grip_how:<5} z{t.grip_z_mm:<4} slide to x={at:<5.0f}"
            f" r={math.hypot(x - at, y - one.rail_y_mm):.0f}"
            f" bearing={abs(one.bearing_from(at, x, y)):.0f}"
            f" room={one.margin_from(at, x, y, t.grip_how, t.grip_z_mm):.0f} mm"
            + ("" if t.graspable else "   (never picked up, only dropped into)")
        )
    for (how, z), (near, far) in sorted(RING_MM.items()):
        out.append(f"  ring {how:<5} z{z:<4} {near:.0f} to {far:.0f} mm ahead, less off to a side")
    for kind, n in sorted(kind_counts(things()).items()):
        if n > 1:
            out.append(f"  {n} x {kind}: no word separates them")
    out.append(
        f"  the rail runs x {one.rail_from_mm:.0f} to {one.rail_to_mm:.0f} at y {one.rail_y_mm:.0f}"
    )
    out.append(f"  it can put a thing where the person can take it: {one.reaches_person}")
    out.append(f"  things standing in the same place: {clashes() or 'none'}")
    return out
