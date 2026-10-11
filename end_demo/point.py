"""The cone, in one place only.

There is exactly one implementation of "what did that point land on" in this codebase, and
it is `rounds/pointer.py`. This module re-exports it so end-demo code can say
`from end_demo.point import weigh` and never grow a second copy.

Why that matters: the page has to be able to draw the cone from the resolver's **actual
numbers**. If the browser carried its own copy of this arithmetic, the shading on screen
would be a decoration that happens to look right, and a judge asking "is that made up?"
would deserve a worse answer than no.

`RATIO` (2:1) is the core's own rule for when to stop guessing and ask - see
`deixis/resolver.py`, `ResolverParams.ambiguity_ratio`, "Ask, don't guess".
"""

from __future__ import annotations

from collections.abc import Sequence

from deixis.resolver import ResolverParams
from rounds.pointer import (
    CONE_DEG,
    RATIO,
    SIGMA_DEG,
    Candidate,
    Cone,
    aim,
    angles,
    point_at,
    ray,
    separation_deg,
    weigh,
)
from rounds.pointer import Thing as Pointable

__all__ = [
    "CONE_DEG",
    "RATIO",
    "SIGMA_DEG",
    "Candidate",
    "Cone",
    "Pointable",
    "aim",
    "angles",
    "core_ratio",
    "point_at",
    "ray",
    "separation_deg",
    "weigh",
]


def core_ratio() -> float:
    """The core's ask threshold, read from the core rather than copied out of it.

    `RATIO` in `rounds/pointer.py` is a literal 2.0 with a comment pointing here. This
    function is the check that the two have not drifted: if the core ever changes its mind
    about when to ask, the test that compares them fails.
    """
    return ResolverParams().ambiguity_ratio


def names(cone: Cone, limit: int | None = None) -> list[str]:
    """The candidate ids, likeliest first."""
    ids = cone.ids()
    return ids if limit is None else ids[:limit]


def top_two(cone: Cone) -> tuple[Candidate, Candidate] | None:
    """The best two candidates, when there are two."""
    if len(cone.candidates) < 2:
        return None
    return cone.candidates[0], cone.candidates[1]


def within_ratio(cone: Cone, ratio: float = RATIO) -> bool:
    """True when the top two are close enough that the core would say: ask, don't guess.

    This is the trigger for a fixed choice (plan 8.1): two things near enough to equally
    likely that pointing at one and then the other is the right question.
    """
    pair = top_two(cone)
    return bool(pair) and pair[0].weight < ratio * pair[1].weight


def sole(cone: Cone, ratio: float = RATIO) -> str | None:
    """The one thing the point settles on its own, or None if it settles nothing."""
    if not cone.candidates:
        return None
    if len(cone.candidates) == 1:
        return cone.candidates[0].id
    return None if within_ratio(cone, ratio) else cone.candidates[0].id


def shelf_cone(
    things: Sequence[Pointable],
    pointed: tuple[float, float],
    sigma_deg: float = SIGMA_DEG,
    cone_deg: float = CONE_DEG,
) -> Cone:
    """The cone over the shelf. Arms are not in it: a point at the shelf and a point at an
    arm are different questions, and mixing them would let a low point at the table turn
    into a thing on a shelf (plan 6.2, `end_demo/scene.py`)."""
    return weigh(pointed, list(things), sigma_deg, cone_deg)
