"""The shelf: ten things across the room, some of them impossible to tell apart in words.

The point of the scene is that no single channel can pick one thing out of it:

- **Words alone fail** on the four bottles and the two mugs, because there is no word that
  separates one bottle from the next, and on the bracket, because nobody has a word for it.
- **Pointing alone fails** because a point is a cone, not a line: at 2.5 m the neighbours
  are a few degrees away and a human's aim is worse than that (`pointer.py`).

Geometry: the person's eye is the origin, the shelf is `DIST` metres away (+y), x is to
their right and z is up. Two shelves of five, 0.25 m apart across and 0.35 m apart up.
That is an ordinary shelf unit, about 1.2 m wide, seen from across a room.

`says` is what a person might say about a thing when the name will not come - the short
phrases first, because the simulated person is on a word budget (`person.py`). Things that
share a `kind` share their words: that is the whole difficulty, not a trick.

`seen` is what the robot's own eyes could report about a thing. It never contains anything
that separates two things of the same kind, because nothing does.
"""

from __future__ import annotations

from dataclasses import dataclass

#: How far away the shelf is (metres). Across a small room.
DIST = 2.5
#: Across, and up, between neighbours (metres).
STEP_X = 0.25
STEP_Z = 0.35


@dataclass(frozen=True)
class Thing:
    """One thing on the shelf."""

    id: str
    kind: str  #: things of the same kind cannot be told apart by any word
    seen: str  #: what the robot's camera could say about it
    says: tuple[str, ...]  #: what a person might say, shortest first
    col: int  #: -2 .. 2, left to right from the person's seat
    row: int  #: 0 top shelf, 1 lower shelf

    @property
    def pos(self) -> tuple[float, float, float]:
        """Where it is, in metres, from the person's eye."""
        return (self.col * STEP_X, DIST, -self.row * STEP_Z)


def shelf() -> list[Thing]:
    """The ten things, left to right, top shelf then lower."""
    return [
        # Top shelf: four bottles that are the same bottle, and the scissors.
        Thing("bottle_1", "bottle", "a small white bottle", ("bottle", "small bottle"), -2, 0),
        Thing("bottle_2", "bottle", "a small white bottle", ("bottle", "small bottle"), -1, 0),
        Thing("bottle_3", "bottle", "a small white bottle", ("bottle", "small bottle"), 0, 0),
        Thing("bottle_4", "bottle", "a small white bottle", ("bottle", "small bottle"), 1, 0),
        Thing("scissors", "scissors", "a pair of scissors", ("cutting", "for cutting"), 2, 0),
        # Lower shelf: two mugs that are the same mug, and three one-off things.
        Thing("mug_1", "mug", "a blue mug", ("mug", "the mug"), -2, 1),
        Thing("mug_2", "mug", "a blue mug", ("mug", "the mug"), -1, 1),
        Thing("bracket", "bracket", "a small metal bracket", ("metal", "metal thing"), 0, 1),
        Thing("phone", "phone", "a phone", ("phone", "my phone"), 1, 1),
        Thing("remote", "remote", "a TV remote", ("remote", "the remote"), 2, 1),
    ]


def by_id(things: list[Thing]) -> dict[str, Thing]:
    return {t.id: t for t in things}


def kind_counts(things: list[Thing]) -> dict[str, int]:
    """How many of each kind there are: 4 bottles, 2 mugs, 1 of everything else."""
    out: dict[str, int] = {}
    for t in things:
        out[t.kind] = out.get(t.kind, 0) + 1
    return out
