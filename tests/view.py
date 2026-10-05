"""The page's camera and raycast, in Python, for the server tests (plan 4.3, W7).

The numbers come from `web/tests/fixtures/view.json`, which the browser tests write from the
real three.js camera and check on every run, so this can't drift from the page. A test below
(`test_arm_core.py`) also checks these raycasts against the fixture's real ones.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from webdemo.settings import ROOT

VIEW = json.loads((ROOT / "tests" / "fixtures" / "view.json").read_text())
# The table top in the scene (web/model/table_scene.xml), table mm: wider than the work area.
TABLE_X = (-60.0, 560.0)
TABLE_Y = (-70.0, 350.0)
CANVAS_PX = (1280, 720)


@dataclass(frozen=True)
class Hit:
    x: float
    y: float
    z: float
    block: str | None


class View:
    """The camera where a seated person's eyes are; table mm, z up."""

    def __init__(self, view: dict = VIEW) -> None:
        self.eye = np.array(view["eye_mm"], float)
        fwd = np.array(view["forward"], float)
        self.z = -fwd / np.linalg.norm(fwd)  # the camera looks down its -z
        self.x = np.cross([0.0, 0.0, 1.0], self.z)
        self.x /= np.linalg.norm(self.x)
        self.y = np.cross(self.z, self.x)
        self.tan = math.tan(math.radians(view["fov_deg"]) / 2)
        self.aspect = view["aspect"]

    def project(self, p: tuple[float, float, float]) -> tuple[float, float]:
        """Canvas fractions (0..1 from the left and the top) of a point in table mm."""
        v = np.asarray(p, float) - self.eye
        depth = -v @ self.z
        nx = (v @ self.x) / depth / (self.tan * self.aspect)
        ny = (v @ self.y) / depth / self.tan
        return (nx + 1) / 2, (1 - ny) / 2

    def hit(
        self, u: float, v: float, blocks: dict[str, tuple[float, float]], size: float = 30.0
    ) -> Hit | None:
        """What the ray through a canvas point hits first: a block (centre, `size` cube) or
        the table. None when it misses both (as `pointAt` in web/src/sim/pick.ts)."""
        nx, ny = 2 * u - 1, 1 - 2 * v
        d = -self.z + nx * self.tan * self.aspect * self.x + ny * self.tan * self.y
        best: tuple[float, str | None] | None = None
        h = size / 2
        for name, (cx, cy) in blocks.items():
            t = _slab(self.eye, d, (cx - h, cy - h, 0.0), (cx + h, cy + h, size))
            if t is not None and (best is None or t < best[0]):
                best = (t, name)
        if d[2] < 0:
            t = -self.eye[2] / d[2]
            px, py = self.eye[:2] + t * d[:2]
            on_table = TABLE_X[0] <= px <= TABLE_X[1] and TABLE_Y[0] <= py <= TABLE_Y[1]
            if on_table and (best is None or t < best[0]):
                best = (t, None)
        if best is None:
            return None
        p = self.eye + best[0] * d
        return Hit(float(p[0]), float(p[1]), float(p[2]), best[1])


def _slab(o: np.ndarray, d: np.ndarray, lo: tuple, hi: tuple) -> float | None:
    """Where a ray first enters an axis-aligned box (its parameter), or None."""
    t0, t1 = 0.0, math.inf
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if not lo[k] <= o[k] <= hi[k]:
                return None
            continue
        a, b = (lo[k] - o[k]) / d[k], (hi[k] - o[k]) / d[k]
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t0 > t1:
            return None
    return t0


class Finger:
    """A fingertip pointing over the page, 30 frames a second: it aims at points in table mm
    (a block's top face, or a spot on the table), is seen with 2 px of jitter on a 1280 x 720
    canvas, and each frame becomes a sample from the raycast, as the page sends them."""

    FPS = 30

    def __init__(self, blocks: dict[str, tuple[float, float]], at=(250.0, 0.0, 0.0), t=1.0,
                 seed=0, jitter_px=2.0) -> None:  # fmt: skip
        self.view = View()
        self.blocks = blocks
        self.at = tuple(at)
        self.t = t
        self.rng = np.random.default_rng(seed)
        self.jitter = jitter_px
        self.samples: list[dict] = []

    def _frame(self, p) -> None:
        u, v = self.view.project(p)
        u += self.rng.normal(0, self.jitter) / CANVAS_PX[0]
        v += self.rng.normal(0, self.jitter) / CANVAS_PX[1]
        h = self.view.hit(u, v, self.blocks)
        point = None if h is None else [round(h.x, 3), round(h.y, 3)]
        self.samples.append({"t": round(self.t, 4), "point": point, "pointing": True})
        self.t += 1 / self.FPS

    def move(self, to, speed=400.0) -> Finger:
        """To a point (x, y) on the table, or (x, y, z) such as a block's top."""
        to = (*to, 0.0) if len(to) == 2 else tuple(to)
        n = max(1, round(math.dist(self.at, to) / speed * self.FPS))
        a = self.at
        for k in range(n):
            self._frame(tuple(a[i] + (to[i] - a[i]) * k / n for i in range(3)))
        self.at = to
        return self

    def at_block(self, name: str, size: float = 30.0) -> Finger:
        cx, cy = self.blocks[name]
        return self.move((cx, cy, size))

    def hold(self, s: float) -> Finger:
        for _ in range(round(s * self.FPS)):
            self._frame(self.at)
        return self
