"""Shared fixtures. The core is always reached over MCP with no fallback: an MCP failure fails
the test instead of silently running the core in-process (plan 7.3)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from deixis.mcp import DeixisClient
from webdemo.app import create_app
from webdemo.settings import ROOT, Settings

PROFILE = ROOT / "profiles" / "so101-table.json"


@pytest.fixture(scope="session")
def core() -> Iterator[DeixisClient]:
    c = DeixisClient(PROFILE, mode="mcp", fallback=None)
    yield c
    c.close()


@pytest.fixture
def settings(tmp_path) -> Settings:
    # No .env, no key, no built page unless a test makes one.
    return Settings(_env_file=None, nebius_api_key=None, web_dist=tmp_path / "no-dist",
                    deixis_fallback="none")  # fmt: skip


@pytest.fixture
def api(settings, core) -> Iterator[TestClient]:
    with TestClient(create_app(settings, client=core)) as c:
        yield c
