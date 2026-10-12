"""The committed physics scene must match the geometry the brain works from.

The shelf exists twice over: once as numbers in `end_demo/scene.py` that the cone maths and
the model work from, and once as bodies in `web/model/shelf_scene.xml` that MuJoCo
simulates. If those two drift, the page draws a cone over one shelf while the arm reaches
for another, and nothing visible says so. The XML is generated; this is the test that keeps
the committed copy honest.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from end_demo.scene import arms, shelf

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "make_shelf_scene.py"
SCENE = ROOT / "web" / "model" / "shelf_scene.xml"


def test_the_committed_scene_is_not_stale():
    got = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--quiet"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert got.returncode == 0, got.stderr or got.stdout


def test_every_thing_is_in_the_scene_file():
    text = SCENE.read_text(encoding="utf-8")
    for t in shelf():
        assert f'<body name="{t.id}"' in text, t.id
        assert f'<freejoint name="{t.id}"/>' in text, t.id


def test_both_arms_are_attached_rather_than_copied():
    """`<attach prefix>`, because `<replicate>` is not in this MuJoCo build and a prefixed
    copy of so101.xml would be a file to maintain. See the E6 spike."""
    text = SCENE.read_text(encoding="utf-8")
    assert text.count('<attach model="so101"') == 2
    for a in arms():
        assert f'prefix="{a.id.split("_")[0]}_"' in text


def test_there_is_one_weld_per_gripper():
    """Two grips means the page cannot use MuJoCo's global equality-disable flag any more:
    that flag would let go of both at once."""
    text = SCENE.read_text(encoding="utf-8")
    assert 'name="left_grip"' in text and 'name="right_grip"' in text


def test_the_resting_pose_is_in_the_scene():
    """MuJoCo's default pose lies flat along +X, straight through the front row, and the
    first physics run had both arms resting on it. The pose belongs to the scene, not to
    whatever the page remembers to set."""
    text = SCENE.read_text(encoding="utf-8")
    assert '<key name="home"' in text


def test_nothing_is_wide_enough_to_touch_its_neighbour():
    """At 70 mm between columns, a thing wider than 70 mm overlaps the thing beside it and
    the physics shoves them apart. That is how the remote came to slide 49 mm."""
    from end_demo.scene import STEP_X_MM

    for t in shelf():
        assert t.half_across_mm < STEP_X_MM / 2, t.id
