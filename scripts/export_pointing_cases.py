"""Export the pointing-pose cases from deixis-doodle for the TypeScript port (plan W4, 6.3).

The browser's `classifyPointing` (web/src/hand/pose.ts) is a port of deixis-doodle's
`classify_pointing` (src/pointdraw/vision/hands.py, commit 7a7e9cc). This script runs that
Python code on the same made-up hands its tests use, plus the 12 frames of rahul's hand
recorded on 1 Oct 2026, and writes every case with Python's answer and numbers to
`web/tests/fixtures/pointing-cases.json`. The vitest test checks the port against it.

Run it with deixis-doodle's environment (this repo doesn't depend on deixis-doodle):

    uv run --project ..\\deixis-doodle python scripts\\export_pointing_cases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOODLE = ROOT.parent / "deixis-doodle"
sys.path.insert(0, str(DOODLE / "tests" / "vision"))

from pointdraw.vision.hands import PoseParams, classify_pointing  # noqa: E402
from test_hands import (  # noqa: E402
    _ORDER,
    CURLED,
    POINT,
    STRAIGHT,
    _random_rotation,
    make_hand,
)

OUT = ROOT / "web" / "tests" / "fixtures" / "pointing-cases.json"


def case(name, lm, params=None):
    p = params or PoseParams()
    res = classify_pointing(lm, p)
    return {
        "name": name,
        "landmarks": [[float(v) for v in row] for row in lm],
        "params": None if params is None else params.__dict__,
        "pointing": bool(res.pointing),
        "fingers": {
            f: {"reach": s.reach, "bend_deg": s.bend_deg, "extended": s.extended,
                "curled": s.curled}
            for f, s in res.fingers.items()
        },
    }  # fmt: skip


def main() -> None:
    cases = []
    for seed in range(5):
        lm = make_hand(POINT, rotation=_random_rotation(seed), noise=0.001, seed=seed)
        cases.append(case(f"pointing, turned at random (seed {seed})", lm))
    lazy = dict(POINT, pinky=(30, 30, 10))
    cases.append(case("pointing, thumb out, lazy pinky", make_hand(lazy, thumb_out=True)))
    pressed = make_hand(dict(POINT, index=(0, 40, 30)))
    cases.append(case("pointing down, index bent a little", pressed))
    cases.append(case("index bent a little, half and hook rules off", pressed,
                      PoseParams(half_index_ok=False, hooked_index_margin=0)))  # fmt: skip
    hooked = make_hand(dict(POINT, index=(20, 70, 40)))
    cases.append(case("pointing, index hooked", hooked))
    cases.append(case("index hooked, hook rule off", hooked, PoseParams(hooked_index_margin=0)))
    for name, flex in [
        ("open hand", {f: STRAIGHT for f in _ORDER}),
        ("fist", {f: CURLED for f in _ORDER}),
        ("fist, index a little looser", dict(POINT, index=(60, 90, 50))),
        ("peace sign", dict(POINT, middle=STRAIGHT)),
        ("middle curled only", dict(POINT, ring=STRAIGHT, pinky=STRAIGHT)),
    ]:
        cases.append(case(name, make_hand(flex)))

    recorded = json.loads((DOODLE / "tests" / "runtime" / "data" / "hands_1oct.json").read_text())
    for kind in ("point", "size"):
        for i, h in enumerate(recorded[kind]):
            cases.append(case(f"rahul's hand, 1 Oct, {kind} frame {i}", h["world"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"source": "deixis-doodle 7a7e9cc", "cases": cases}, indent=1))
    print(f"{len(cases)} cases -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
