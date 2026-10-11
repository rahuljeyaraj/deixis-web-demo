"""The shelf, the two arms, and where the person sits.

Twelve things in two rows of six, 2.5 m away. Six of them cannot be told apart by any
word - four identical pill bottles and two identical mugs - and one of them has no name at
all. That is the whole difficulty of the scene, and none of it is a trick: it is an
ordinary shelf.

Geometry: the person's eye is the origin, the shelf is `DIST` metres away (+y), x is to
their right and z is up. Columns are numbered 0..5 left to right as the person sees them,
and centred on x = 0, so the shelf is 1.25 m wide - an ordinary shelf unit across a small
room. At that distance neighbours are about 5.7 degrees apart and a person's aim is worse
than that, so one point covers four or five things. Perspective does the work; no noise is
added anywhere.

**Two arms, and why.** The person sits at the right-hand end of the table. The right arm
stands near them: it can reach the right of the shelf *and* hand a thing to the person. The
left arm stands at the far end: it reaches further left, but it cannot reach the person. So
a thing on the left of the shelf has to be picked by the left arm and **passed to the right
arm**, which then hands it over. That is not scenery - it is why the person has to be able
to say *you*, and it is the one thing a single arm cannot show.

**Arms are pointable too.** They are in the scene as things a person can point at, because
"you, that one" is the heart of telling *any* robot where. They are kept out of the cone
over the shelf, though: a point at the shelf and a point at an arm are different questions.

`says` is what a person might say about a thing when the name will not come, shortest
first, because the person is on a word budget. Things of the same `kind` share their words:
that is the real situation, not a rigged one. `seen` is what the robot's own camera could
report, and it never contains anything that separates two things of the same kind, because
nothing does.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

#: How far away the shelf is (metres). Across a small room.
DIST = 2.5
#: Between neighbours: across, and up (metres).
STEP_X = 0.25
STEP_Z = 0.35
#: Columns per row, and rows. Twelve things.
COLS = 6
ROWS = 2


@dataclass(frozen=True)
class Thing:
    """One thing on the shelf."""

    id: str
    kind: str  #: things of the same kind cannot be told apart by any word
    seen: str  #: what the robot's camera could say about it
    says: tuple[str, ...]  #: what a person might say, shortest first
    col: int  #: 0..5, left to right as the person sees it
    row: int  #: 0 top shelf, 1 lower shelf
    shape: str = "box"  #: for the physics scene later: box, cylinder, capsule
    colour: str = "white"
    #: False for a thing nobody would ask for - the tray is somewhere to put things, not
    #: something to want. It is still pointable, because "put it there" points at it.
    wanted: bool = True
    #: False for a thing a gripper cannot be promised on. Nothing is False yet; the spike
    #: on non-cube meshes (E6) decides it.
    graspable: bool = True

    @property
    def pos(self) -> tuple[float, float, float]:
        """Where it is, in metres, from the person's eye."""
        return ((self.col - (COLS - 1) / 2) * STEP_X, DIST, -self.row * STEP_Z)

    @property
    def place(self) -> tuple[int, int]:
        return (self.row, self.col)


@dataclass(frozen=True)
class Arm:
    """One robot arm. Pointable, because the person may mean *you*."""

    id: str
    seen: str
    says: tuple[str, ...]
    x: float  #: where it stands, metres, left-right from the person's eye
    y: float  #: how far in front of the person
    reach: tuple[int, ...]  #: the shelf columns it can reach
    #: Can it put a thing into the person's hand? Only the arm standing beside them can.
    reaches_person: bool = False

    @property
    def pos(self) -> tuple[float, float, float]:
        return (self.x, self.y, -0.15)

    def can_reach(self, thing: Thing) -> bool:
        return thing.col in self.reach


def shelf() -> list[Thing]:
    """The twelve things, left to right, top shelf then lower.

    Top shelf holds the four bottles that are the same bottle, and two things that can be
    described without their name ("for cutting", "for writing") - those are there so that
    words-only has something it can win, which is what keeps the baseline fair.

    Lower shelf holds the two mugs that are the same mug, the bracket nobody has a word
    for, a phone (words settle it, so round 2 is not rigged), a remote, and the tray.
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
            "capsule",
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
    """The two arms. The right one stands beside the person; the left one does not reach
    them, so anything on the left of the shelf has to be passed across."""
    return [
        Arm(
            "left_arm",
            "the arm at the far end of the table",
            ("you", "that arm", "the far one"),
            x=-0.45,
            y=1.1,
            reach=(0, 1, 2, 3),
        ),
        Arm(
            "right_arm",
            "the arm beside the person",
            ("you", "that arm", "this one"),
            x=0.45,
            y=1.1,
            reach=(2, 3, 4, 5),
            reaches_person=True,
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


def reshuffle(things: list[Thing], rng: random.Random) -> list[Thing]:
    """The same twelve things in a new arrangement.

    This is for the booth: a judge who has just done round 1 would otherwise remember the
    layout, and round 3 would look better for a reason that is not ours.

    It permutes all twelve positions, so identical things can end up apart - which makes
    the point less ambiguous and the task *easier*. That is why offline measurement never
    shuffles, and why the layout is recorded on every attempt: with it, any number can be
    re-derived. Do not compare a shuffled booth run with an unshuffled offline one as if
    the difficulty were the same.
    """
    places = [(r, c) for r in range(ROWS) for c in range(COLS)]
    rng.shuffle(places)
    return [replace(t, row=r, col=c) for t, (r, c) in zip(things, places, strict=True)]


def layout(things: list[Thing]) -> list[tuple[str, int, int]]:
    """The arrangement, for the record on an attempt."""
    return [(t.id, t.row, t.col) for t in things]
