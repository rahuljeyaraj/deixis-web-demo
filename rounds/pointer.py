"""A point is a cone, not a line.

People do not point accurately. Measured with a camera, the head-to-hand line misses by
**25 degrees on average** over ranges up to 4 m, and systems that select an object from
pointing alone get it right **51-69%** of the time. Those two numbers are the baseline this
whole proof of concept exists to beat, and they are why the robot has to be allowed to ask.

So a point here is an aim plus a wobble. `aim` turns a thing's direction into two angles -
across (azimuth) and up (elevation) - and adds independent Gaussian error to each. `weigh`
then asks, of every thing on the shelf, how far off the ray it is, and turns that into a
likelihood.

`SIGMA_DEG` is the one dial that sets the difficulty, and it is printed in every run so no
number is ever quoted without it. The default, 3 degrees per axis, puts round 1 inside the
published 51-69%; the published 25 degree *mean* error would be far worse, because that
study's targets were spread over a much bigger space than one shelf.

`RATIO` (2:1) is not invented here: it is the core's own rule for when to stop guessing and
ask - see `deixis/resolver.py`, "Ask, don't guess".
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Thing(Protocol):
    """Anything a point can land on: it needs an id and a place, nothing else.

    A protocol rather than a class so there is exactly one cone in the codebase. The end
    demo's shelf (`end_demo/scene.py`) has twelve things and two arms and this module knows
    nothing about either; it still weighs them, and the page shows the numbers it produced
    rather than its own copy of this arithmetic.
    """

    @property
    def id(self) -> str: ...

    @property
    def pos(self) -> tuple[float, float, float]: ...


#: Pointing error, standard deviation in degrees, on each of the two angles.
SIGMA_DEG = 3.0
#: The narrowest cone the maths will use. See `weigh`: it keeps a zero pointing error from
#: dividing by zero, and it is far below any aim a person or a sensor actually has.
MIN_SIGMA_DEG = 0.01
#: Things further off the ray than this are not candidates at all.
CONE_DEG = 9.0
#: The core's rule: if the best two are closer than this in likelihood, ask (resolver.py).
RATIO = 2.0


@dataclass(frozen=True)
class Candidate:
    """One thing the point might have meant."""

    id: str
    off_deg: float  #: how far it is from the ray
    weight: float  #: 0..1, shares the cone with the others


@dataclass(frozen=True)
class Cone:
    """What one point tells us: the things inside it, most likely first."""

    candidates: list[Candidate]
    sigma_deg: float

    @property
    def best(self) -> str | None:
        return self.candidates[0].id if self.candidates else None

    @property
    def ambiguous(self) -> bool:
        """True when the point alone must not be trusted (the core's 2:1 rule)."""
        if len(self.candidates) < 2:
            return not self.candidates
        return self.candidates[0].weight < RATIO * self.candidates[1].weight

    def ids(self) -> list[str]:
        return [c.id for c in self.candidates]


def angles(pos: tuple[float, float, float]) -> tuple[float, float]:
    """A direction as (across, up) in radians, from the person's eye."""
    x, y, z = pos
    return math.atan2(x, y), math.atan2(z, math.hypot(x, y))


def separation_deg(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """The angle between two directions, in degrees."""

    def unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
        n = math.sqrt(sum(c * c for c in v)) or 1.0
        return (v[0] / n, v[1] / n, v[2] / n)

    ua, ub = unit(a), unit(b)
    dot = max(-1.0, min(1.0, sum(p * q for p, q in zip(ua, ub, strict=True))))
    return math.degrees(math.acos(dot))


def aim(target: Thing, rng: random.Random, sigma_deg: float = SIGMA_DEG) -> tuple[float, float]:
    """Where the finger actually ended up pointing: the thing's direction, plus a wobble."""
    across, up = angles(target.pos)
    s = math.radians(sigma_deg)
    return across + rng.gauss(0.0, s), up + rng.gauss(0.0, s)


def ray(across: float, up: float) -> tuple[float, float, float]:
    """The two angles back as a direction."""
    return (math.sin(across) * math.cos(up), math.cos(across) * math.cos(up), math.sin(up))


def weigh(
    pointed: tuple[float, float],
    things: Sequence[Thing],
    sigma_deg: float = SIGMA_DEG,
    cone_deg: float = CONE_DEG,
) -> Cone:
    """Every thing inside the cone, with how much of the cone it holds.

    A pointing error of zero is a legitimate thing to ask for - a mouse click, a laser, or
    a test that wants to be about a rule rather than about a wobble - so the width is
    floored rather than special-cased. The floor is far below any real aim, and it gives
    exactly the limiting behaviour: the thing on the ray takes the whole cone and a
    neighbour 5.7 degrees away takes none of it.
    """
    direction = ray(*pointed)
    width = max(sigma_deg, MIN_SIGMA_DEG)
    raw: list[tuple[str, float, float]] = []
    for t in things:
        off = separation_deg(direction, t.pos)
        if off <= cone_deg:
            raw.append((t.id, off, math.exp(-(off**2) / (2 * width**2))))
    total = sum(w for _, _, w in raw) or 1.0
    cands = [Candidate(i, round(off, 2), w / total) for i, off, w in raw]
    cands.sort(key=lambda c: -c.weight)
    return Cone(cands, sigma_deg)


def point_at(
    target: Thing,
    things: Sequence[Thing],
    rng: random.Random,
    sigma_deg: float = SIGMA_DEG,
    cone_deg: float = CONE_DEG,
) -> Cone:
    """One whole point: aim at the target, miss a bit, see what is in the cone."""
    return weigh(aim(target, rng, sigma_deg), things, sigma_deg, cone_deg)
