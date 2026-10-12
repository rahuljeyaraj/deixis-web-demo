"""The shelf, the two arms, and where the person sits.

A small tiered shelf on the table in front of someone sitting at it. Twelve things in two
rows of six: four identical pill bottles, two identical mugs, a bracket nobody has a word
for, and five things that do have names. Six of them cannot be told apart by any word.

## Why it is on the table and not across the room

The plan put the shelf 2.5 m away, so that neighbours would be about 5.7 degrees apart -
closer together than a person's aim is good for, which is what makes one point cover four
or five things. The reasoning is right and the distance is not: **the arm reaches 140 to
280 mm from its base** (measured, `web/src/sim/table.ts`), so nothing could pick a thing off
a shelf 2.5 m away, let alone put it into somebody's hand. The plan's own showpiece - *hand
me that one* - was impossible in the plan's own scene, by a factor of nine.

What makes pointing hard is the **angle** between neighbours, not the distance. So the
angles are kept and the scene is shrunk to a size the arm can work at:

| | the plan, at 2.5 m | here, on the table |
|---|---|---|
| neighbours across | 5.7 deg | 5.6 deg |
| row to row | about 8 deg | 9.2 deg |
| spacing | 250 mm | 70 mm |
| eye to the shelf | 2.5 m | about 0.7 m |

Seventy millimetres between things on a shelf is not a contrivance - it is what a row of
pill bottles looks like. The honest sentence to a judge changes from "the shelf is across
the room" to "the bottles are next to each other", which is a better sentence anyway: the
difficulty is a real shelf rather than a chosen distance.

## Two rows, both open from above

The back row stands on a step, 80 mm up and 100 mm further away - a tiered riser, the thing
people keep spices or medicines on. Both rows are reachable from above, which matters:
reaching *under* a shelf board, like grasping a drawer handle, is where sim demos die and it
would prove nothing about thinking.

## Two arms, and why they need each other

The person sits in the middle of the near edge. The arms stand on the table either side.
Reach is not assigned - it is **computed** from the measured ring, so the geometry cannot
quietly drift away from what the arm can really do:

- the **left** arm reaches the left half of the shelf, **and** it is the only one that can
  reach the person's hand - because the hand that works after a left-hemisphere stroke is
  the left one (see `GIVE_MM`);
- the **right** arm reaches the right half, and **cannot** reach the person at all;
- two columns in the front row are shared, so *which arm* is sometimes a real choice.

A thing on the right of the shelf therefore has to be picked up by the right arm and
**passed across** before it can be handed over. That is why the person needs to be able to
say *you*, and it is the one thing a single arm cannot show.

Left and right are **the person's**, as they see the table, which is why the robot being
able to ask "your left, or mine?" is worth having at all.

## Arms are pointable too

They are in the scene as things a person can point at, because "you, that one" is the heart
of telling *any* robot where. They are kept out of the cone over the shelf, though: a point
at the shelf and a point at an arm are different questions, and mixing them would let a low
point at the table turn into a thing on a shelf.

## The coordinates

One table frame, in millimetres, shared with the rest of the demo (`web/src/sim/table.ts`):
x to the person's right, y away from them, z up from the table top. `Thing.pos` converts to
what the cone maths wants - metres from the person's eye - so there is one set of numbers
and one conversion rather than two sets that can disagree.

`says` is what a person might say about a thing when the name will not come, shortest first,
because the person is on a word budget. Things of the same `kind` share their words: that is
the real situation, not a rigged one. `seen` is what the robot's own camera could report, and
it never contains anything that separates two things of the same kind, because nothing does.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

#: The arm's working ring, millimetres from its base axis. Measured on the real model, not
#: chosen: `web/tests/probes/reach2.probe.test.ts`, 5-6 Oct. Everything else here is laid
#: out to fit inside it, which is why the shelf is on the table.
REACH_MIN_MM = 140.0
REACH_MAX_MM = 280.0

#: The person's eye, in the table frame: sitting at the near edge, 300 mm back and 350 mm up.
EYE_MM = (250.0, -300.0, 350.0)

#: Where a thing goes into the person's hand: just over the near edge of the table, on
#: **their left**.
#:
#: Not the middle, and not arbitrary. The left hemisphere holds language and drives the
#: right side of the body, so a left-hemisphere stroke commonly takes both - the person who
#: cannot say "mug" is frequently the same person whose right arm does not work. The hand
#: they point with, and reach out with, is the left one.
#:
#: It also puts the two arms' jobs where they have to be: only the arm on that side can
#: reach their hand, so a thing on the far side of the shelf has to be passed across.
GIVE_MM = (120.0, -40.0)

#: Between neighbours, across. 70 mm is what a row of pill bottles looks like, and from the
#: person's seat it is 5.6 degrees - less than their aim is good for.
STEP_X_MM = 70.0
#: Columns per row, and rows. Twelve things.
COLS = 6
ROWS = 2
#: The middle of the shelf, across.
CENTRE_X_MM = 250.0

#: Row 0 is the raised row at the back; row 1 is the front row, on the table.
ROW_Y_MM = (400.0, 300.0)
#: What each row stands on: the top of the step at the back, the bare table at the front.
ROW_SURFACE_Z_MM = (80.0, 0.0)
ROW_NAMES = ("the raised row at the back", "the front row")

ORDINALS = ("1st", "2nd", "3rd", "4th", "5th", "6th")

#: How big each kind of thing is, in millimetres, in **table axes**: half across, half deep,
#: half tall for a box; radius and half height for a cylinder.
#:
#: Written in table axes rather than in the object's own frame so there is no orientation to
#: get wrong. That matters more than it sounds: at 70 mm between neighbours, a remote lying
#: across the shelf would overlap the thing beside it, and the first run of the physics scene
#: had the remote, the scissors and the tray shoving each other apart as soon as it started.
#: Long things lie front to back, which is also how you put them on a narrow shelf.
SIZE_MM: dict[str, tuple[float, ...]] = {
    "bottle": (16.0, 35.0),  # cylinder: a pill bottle, 32 mm across
    "mug": (32.0, 40.0),  # cylinder: 64 mm across, 6 mm clear of its neighbour
    "scissors": (12.0, 55.0, 5.0),  # lying front to back
    "pen": (8.0, 55.0, 8.0),  # a box, not a capsule: a capsule rolls off the shelf
    "bracket": (20.0, 12.0, 18.0),
    "phone": (18.0, 35.0, 6.0),
    "remote": (20.0, 60.0, 10.0),
    "tray": (25.0, 40.0, 8.0),
}


#: A hair of clearance under every thing. Starting a body exactly flush on a surface lets
#: the contact solver find a small penetration and turn it into a torque: in the first
#: physics run the remote tipped onto its side and slid 22 mm before the scene had settled,
#: which would have left the robot being told a layout that was no longer true.
SETTLE_GAP_MM = 1.0


def rest_z_mm(kind: str, shape: str) -> float:
    """How high a thing's centre sits above whatever it is standing on."""
    size = SIZE_MM[kind]
    half = size[1] if shape == "cylinder" else size[2]
    return half + SETTLE_GAP_MM


def column_x(col: int) -> float:
    """Where a column sits, across, in millimetres."""
    return CENTRE_X_MM + (col - (COLS - 1) / 2) * STEP_X_MM


@dataclass(frozen=True)
class Thing:
    """One thing on the shelf."""

    id: str
    kind: str  #: things of the same kind cannot be told apart by any word
    seen: str  #: what the robot's camera could say about it
    says: tuple[str, ...]  #: what a person might say, shortest first
    col: int  #: 0..5, left to right as the person sees it
    row: int  #: 0 the raised row at the back, 1 the front row
    shape: str = "box"  #: for the physics scene: box, cylinder, capsule
    colour: str = "white"
    #: False for a thing nobody would ask for - the tray is somewhere to put things, not
    #: something to want. It is still pointable, because "put it there" points at it.
    wanted: bool = True
    #: False for a thing a gripper cannot be promised on. Nothing is False: the spike on
    #: non-cube shapes passed for boxes, cylinders and capsules
    #: (`web/tests/probes/twoarms.probe.test.ts`).
    graspable: bool = True

    @property
    def place_mm(self) -> tuple[float, float, float]:
        """Its **centre**, in the table frame, millimetres.

        The centre, not the surface it stands on, because that is both what the physics
        scene wants for a body and what the person's eye is looking at. Computed from the
        row's surface plus the thing's own height, so a taller thing sits higher rather than
        sinking into the shelf - the first physics run had everything drop 15 mm because
        this returned the surface and the generator added the height again.
        """
        return (
            column_x(self.col),
            ROW_Y_MM[self.row],
            ROW_SURFACE_Z_MM[self.row] + self.rest_z_mm,
        )

    @property
    def rest_z_mm(self) -> float:
        return rest_z_mm(self.kind, self.shape)

    @property
    def size_mm(self) -> tuple[float, ...]:
        return SIZE_MM[self.kind]

    @property
    def half_across_mm(self) -> float:
        """Half its width across the shelf. Must stay under half the column spacing, or it
        overlaps the thing beside it."""
        return self.size_mm[0]

    @property
    def pos(self) -> tuple[float, float, float]:
        """Where it is from the person's eye, in metres: right, away, up.

        This is what the cone maths wants (`point.py`), and it is derived from `place_mm`
        rather than stored, so the physics and the pointing can never be laid out
        differently from one another.
        """
        return from_eye(self.place_mm)

    @property
    def place(self) -> tuple[int, int]:
        return (self.row, self.col)

    @property
    def where(self) -> str:
        """Whereabouts it is, in words the robot's camera could honestly report."""
        return f"{ROW_NAMES[self.row]}, {ORDINALS[self.col]} from the left"


def from_eye(place_mm: tuple[float, float, float]) -> tuple[float, float, float]:
    """A table-frame point in millimetres, as metres from the person's eye."""
    return tuple((p - e) / 1000.0 for p, e in zip(place_mm, EYE_MM, strict=True))  # type: ignore[return-value]


@dataclass(frozen=True)
class Arm:
    """One robot arm, standing on the table. Pointable, because the person may mean *you*."""

    id: str
    seen: str
    says: tuple[str, ...]
    #: Where its base stands, in the table frame, millimetres.
    base_mm: tuple[float, float]

    @property
    def pos(self) -> tuple[float, float, float]:
        """Its gripper's resting height, from the person's eye."""
        return from_eye((self.base_mm[0], self.base_mm[1], 120.0))

    def radius_to(self, x: float, y: float) -> float:
        return math.hypot(x - self.base_mm[0], y - self.base_mm[1])

    def reaches(self, x: float, y: float) -> bool:
        """Is this spot inside the measured working ring?"""
        return REACH_MIN_MM <= self.radius_to(x, y) <= REACH_MAX_MM

    def can_reach(self, thing: Thing) -> bool:
        x, y, _ = thing.place_mm
        return self.reaches(x, y)

    @property
    def reaches_person(self) -> bool:
        """Can it put a thing into their hand? Only the arm standing beside them can."""
        return self.reaches(*GIVE_MM)

    def columns(self, row: int) -> list[int]:
        """Which columns of a row it can pick up from."""
        return [c for c in range(COLS) if self.reaches(column_x(c), ROW_Y_MM[row])]


def shelf() -> list[Thing]:
    """The twelve things, left to right, the raised row then the front row.

    The raised row holds the four bottles that are the same bottle, and two things that can
    be described without their name ("for cutting", "for writing"). Those two are there so
    that words-only has something it can win, which is what keeps the baseline fair.

    The front row holds the two mugs that are the same mug, the bracket nobody has a word
    for, a phone (words settle it, so the words-only round is not rigged), a remote, and the
    tray to put things in.
    """
    bottle = ("bottle", "small bottle", "one of those bottles")
    mug = ("mug", "the mug", "something to drink from")
    return [
        Thing("bottle_1", "bottle", "a small white bottle", bottle, 0, 0, "cylinder", "white"),
        Thing("bottle_2", "bottle", "a small white bottle", bottle, 1, 0, "cylinder", "white"),
        Thing("bottle_3", "bottle", "a small white bottle", bottle, 2, 0, "cylinder", "white"),
        Thing("bottle_4", "bottle", "a small white bottle", bottle, 3, 0, "cylinder", "white"),
        Thing(
            "scissors",
            "scissors",
            "a pair of scissors",
            ("cutting", "for cutting", "to cut with"),
            4,
            0,
            "box",
            "grey",
        ),
        Thing(
            "pen",
            "pen",
            "a pen",
            ("writing", "for writing", "to write with"),
            5,
            0,
            # A box, not a capsule. The spike proved a weld holds a capsule, but a capsule
            # lying on a shelf rolls: in the first physics run the pen travelled 164 mm
            # before the judge would have arrived.
            "box",
            "blue",
        ),
        Thing("mug_1", "mug", "a blue mug", mug, 0, 1, "cylinder", "blue"),
        Thing("mug_2", "mug", "a blue mug", mug, 1, 1, "cylinder", "blue"),
        Thing(
            "bracket",
            "bracket",
            "a small metal bracket",
            ("metal", "metal thing", "that metal thing"),
            2,
            1,
            "box",
            "metal",
        ),
        Thing("phone", "phone", "a phone", ("phone", "my phone"), 3, 1, "box", "black"),
        Thing("remote", "remote", "a TV remote", ("remote", "the remote"), 4, 1, "box", "black"),
        Thing(
            "tray",
            "tray",
            "a shallow tray",
            ("there", "in there"),
            5,
            1,
            "box",
            "wood",
            wanted=False,
        ),
    ]


def arms() -> list[Arm]:
    """The two arms, standing on the table either side of the person's seat.

    Bases at x = 110 and x = 390, 150 mm forward. That placement is what makes the two arms
    need each other: the left arm owns the left half and the person's hand, the right arm
    owns the right half and cannot reach them at all, and two columns in the front row are
    shared. Every one of those facts is **computed** from the measured ring by
    `Arm.reaches`, so moving a base by a centimetre changes the answers rather than quietly
    contradicting them.

    Left and right are the person's, as they see the table.
    """
    return [
        Arm(
            "left_arm",
            "the arm on the left, beside the person's working hand",
            ("you", "that arm", "this one"),
            base_mm=(110.0, 150.0),
        ),
        Arm(
            "right_arm",
            "the arm on the right, across the table from the person",
            ("you", "that arm", "the far one"),
            base_mm=(390.0, 150.0),
        ),
    ]


def by_id(items: list[Thing]) -> dict[str, Thing]:
    return {t.id: t for t in items}


def kind_counts(things: list[Thing]) -> dict[str, int]:
    """How many of each kind: 4 bottles, 2 mugs, 1 of everything else."""
    out: dict[str, int] = {}
    for t in things:
        out[t.kind] = out.get(t.kind, 0) + 1
    return out


def is_unique(things: list[Thing], thing_id: str) -> bool:
    """Is this thing one of a kind - does a word for it exist at all?"""
    index = by_id(things)
    return kind_counts(things)[index[thing_id].kind] == 1


def wanted(things: list[Thing]) -> list[Thing]:
    """The things a person might actually ask for (everything but the tray)."""
    return [t for t in things if t.wanted]


def separation_deg(a: Thing, b: Thing) -> float:
    """The angle between two things, from the person's eye. The one number that sets how
    hard the pointing is."""
    from end_demo.point import separation_deg as sep

    return sep(a.pos, b.pos)


def reshuffle(things: list[Thing], rng: random.Random) -> list[Thing]:
    """The same twelve things in a new arrangement.

    This is for the booth: a judge who has just done round 1 would otherwise remember the
    layout, and round 3 would look better for a reason that is not ours.

    It permutes all twelve positions, so identical things can end up apart - which makes the
    point less ambiguous and the task *easier*. That is why offline measurement never
    shuffles, and why the layout is recorded on every attempt: with it, any number can be
    re-derived. Do not compare a shuffled booth run with an unshuffled offline one as if the
    difficulty were the same.
    """
    places = [(r, c) for r in range(ROWS) for c in range(COLS)]
    rng.shuffle(places)
    return [replace(t, row=r, col=c) for t, (r, c) in zip(things, places, strict=True)]


def layout(things: list[Thing]) -> list[tuple[str, int, int]]:
    """The arrangement, for the record on an attempt."""
    return [(t.id, t.row, t.col) for t in things]
