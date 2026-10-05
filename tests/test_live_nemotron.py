"""Live Nemotron on the arm's phrases (opt-in: costs about $0.01 a run).

    uv run pytest -m network tests/test_live_nemotron.py -s

Needs NEBIUS_API_KEY in .env. Each phrase must come back from Nemotron itself (not the rule
parser) with the same calls the rule parser gives.
"""

from __future__ import annotations

import pytest

from test_agent import PHRASES, PROFILE, TOOLS
from webdemo.agent import ArmAgent, rules
from webdemo.settings import Settings

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def live():
    s = Settings()
    if not s.nebius_api_key:
        pytest.skip("NEBIUS_API_KEY is not set")
    agent = ArmAgent(PROFILE, TOOLS, s.nebius_base_url, model=s.nemotron_model)
    return agent, s.nebius_api_key.get_secret_value()


@pytest.mark.parametrize("text", [p[0] for p in PHRASES])
def test_nemotron_agrees_with_the_rules(live, text):
    agent, key = live
    words = text.split()
    res = agent.run(words, key)
    print(f"{text!r}: {[str(c) for c in res.calls]} in {res.latency_s:.2f} s, "
          f"{res.attempts} attempt(s)")  # fmt: skip
    expected = [c.to_dict() for c in rules(PROFILE, words).calls]
    assert [c.to_dict() for c in res.calls] == expected, res.errors
