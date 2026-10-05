"""Shared fixtures. The core is always reached over MCP with no fallback: an MCP failure fails
the test instead of silently running the core in-process (plan 7.3)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from deixis.mcp import DeixisClient
from webdemo.app import create_app
from webdemo.settings import ROOT, Settings
from webdemo.speech import PARAKEET, Speech

PROFILE = ROOT / "profiles" / "so101-table.json"
MODELS = ROOT / ".cache" / "models"
VOICE = ROOT / ".cache" / "voice"  # scripts/make_voice_fixtures.py

needs_parakeet = pytest.mark.skipif(
    not PARAKEET.available(MODELS), reason="Parakeet not in .cache/models (scripts/fetch_models.py)"
)
needs_voice = pytest.mark.skipif(
    not (VOICE / "david_00.wav").exists(), reason="run scripts/make_voice_fixtures.py first"
)


@pytest.fixture(scope="session")
def core() -> Iterator[DeixisClient]:
    c = DeixisClient(PROFILE, mode="mcp", fallback=None)
    yield c
    c.close()


@pytest.fixture(scope="session")
def speech() -> Iterator[Speech | None]:
    """Parakeet, loaded once for the whole run (None if it isn't downloaded)."""
    if not PARAKEET.available(MODELS):
        yield None
        return
    s = Speech(MODELS, workers=2, threads=2)
    yield s
    s.close()


@pytest.fixture
def settings(tmp_path) -> Settings:
    # No .env, no key, no built page unless a test makes one; speech comes from the fixture.
    return Settings(_env_file=None, nebius_api_key=None, web_dist=tmp_path / "no-dist",
                    deixis_fallback="none", asr_enabled=False)  # fmt: skip


@pytest.fixture
def api(settings, core, speech) -> Iterator[TestClient]:
    with TestClient(create_app(settings, client=core, speech=speech)) as c:
        yield c


@pytest.fixture
def rules_api(settings, core, speech) -> Iterator[TestClient]:
    """The app with the rule parser instead of Nemotron (no key, no network)."""
    settings.ai = "rules"
    with TestClient(create_app(settings, client=core, speech=speech)) as c:
        yield c
