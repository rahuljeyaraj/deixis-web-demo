"""The app's routes (plan 7.2, W1): health, info with the models list, the built page."""

from __future__ import annotations

import json
import re

from fastapi.testclient import TestClient
from pydantic import SecretStr

from deixis.profile import load_profile
from webdemo.app import create_app
from webdemo.settings import ROOT


def test_healthz_reaches_the_core_over_mcp(api):
    r = api.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["core"] == {"ok": True, "mode": "mcp", "over_mcp": True, "fell_back": None}


def test_info_has_the_profile_and_every_model(api):
    body = api.get("/api/info").json()
    assert body["title"] == "deixis"
    assert body["core"]["over_mcp"] is True
    assert body["profile"]["name"] == "so101-table"
    assert body["profile"]["area_mm"] == [0, -40, 500, 290]
    assert set(body["profile"]["actions"]) >= {"move", "home", "adjust", "confirm", "cancel"}
    names = " ".join(m["name"] for m in body["models"])
    for model in ["Parakeet", "Nemotron 3.5 Lightning", "deixis", "MuJoCo"]:
        assert model in names
    assert all(m["url"].startswith("https://") for m in body["models"])
    assert "Cosmos" not in names  # not used here, and the page says so
    assert body["not_used"][0]["name"] == "NVIDIA Cosmos"


def test_info_never_shows_our_key(settings, core):
    settings.nebius_api_key = SecretStr("sk-our-secret-key-123")
    with TestClient(create_app(settings, client=core)) as c:
        assert "sk-our-secret-key-123" not in c.get("/api/info").text
        assert "sk-our-secret-key-123" not in c.get("/healthz").text


def test_serves_the_built_page(settings, core, tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>Put that here</title>")
    settings.web_dist = dist
    with TestClient(create_app(settings, client=core)) as c:
        assert "Put that here" in c.get("/").text
        assert c.get("/api/info").status_code == 200  # the API still wins over the page


def test_profile_matches_the_arm_in_the_browser():
    """The core's work area is the area the browser's arm draws (web/src/sim/table.ts)."""
    profile = load_profile(ROOT / "profiles" / "so101-table.json")
    ts = (ROOT / "web" / "src" / "sim" / "table.ts").read_text()
    area = re.search(r"AREA_MM = \{ x: (-?\d+), y: (-?\d+), w: (-?\d+), h: (-?\d+) \}", ts)
    assert area, "AREA_MM not found in table.ts"
    assert list(profile.area_mm) == [int(v) for v in area.groups()]
    block = re.search(r"BLOCK_MM = (\d+)", ts)
    assert block and profile.sizes.default_mm == int(block.group(1))
    assert json.loads((ROOT / "profiles" / "so101-table.json").read_text())["frame"] == "table"
