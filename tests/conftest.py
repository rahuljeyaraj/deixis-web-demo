"""Shared fixtures. The core is always reached over MCP with no fallback: an MCP failure fails
the test instead of silently running the core in-process (plan 7.3)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from tests import wordrules

from deixis.mcp import DeixisClient
from webdemo.agent import AgentResult, ArmAgent, Context
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
    # No .env, no key, no built page unless a test makes one; speech comes from the fixture;
    # no burst limit (tests send commands fast; `test_limits.py` turns it on).
    return Settings(_env_file=None, nebius_api_key=None, web_dist=tmp_path / "no-dist",
                    deixis_fallback="none", asr_enabled=False, burst_s=0,
                    command_log=tmp_path / "commands.jsonl")  # fmt: skip


@pytest.fixture
def api(settings, core, speech) -> Iterator[TestClient]:
    with TestClient(create_app(settings, client=core, speech=speech)) as c:
        yield c


def _by_rules(agent: ArmAgent):
    """Nemotron's seat, filled by `tests/wordrules.py`. The key is ignored and nothing leaves
    this process. It replaces the method on this one agent, never on the class: other tests
    in the same run build their own app with a fake model and must still get it."""

    def understand(words, key, context: Context | None = None, no_key_why: str = "",
                   shared: bool = False) -> AgentResult:  # fmt: skip
        return wordrules.rules(agent.profile, words, context)

    return understand


@pytest.fixture
def rules_api(settings, core, speech, monkeypatch) -> Iterator[TestClient]:
    """The app with a stand-in for Nemotron: a small word parser that gets the simple
    commands right, so the server's own plumbing - the limits, the flow, the pointing, the
    plan loop - can be tested without a key, a network or a bill.

    The demo has no such fallback. The word matcher was taken out of the server on 11 Oct
    2026 (rahul): nothing but the model may answer a person, or appear in the diagram. What
    is left is this double, which the server cannot reach by itself."""
    settings.nebius_api_key = SecretStr("not-a-real-key")
    app = create_app(settings, client=core, speech=speech)
    with TestClient(app) as c:  # the agent is built when the app starts
        monkeypatch.setattr(app.state.agent, "understand", _by_rules(app.state.agent))
        yield c
