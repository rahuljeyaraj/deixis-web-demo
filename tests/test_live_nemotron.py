"""Live Nemotron on the arm's phrases (opt-in: costs about $0.01 a run).

    uv run pytest -m network tests/test_live_nemotron.py -s

Needs NEBIUS_API_KEY in .env. Each phrase must come back from Nemotron itself (not the rule
parser) with the same calls the rule parser gives. The whole server with the real model is
`test_live_web.py`; this file is the agent alone.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from test_agent import PHRASES, PROFILE, TOOLS
from webdemo.agent import ArmAgent, rules
from webdemo.settings import Settings

pytestmark = pytest.mark.network

COLOURS = ("red", "green", "blue")


def same_block(call: dict) -> dict:
    """One call with every label cut to its colour: Nemotron writes the words the person said
    ("the blue cube"), the rule parser always writes "blue one", and the core finds the same
    block from either (it matches every word against the label and `attrs`). The difference is
    wording, not meaning, so the comparison below ignores it."""
    if isinstance(call, dict):
        label = call.get("label")
        if isinstance(label, str):
            colour = next((c for c in COLOURS if c in label.lower()), None)
            call = {**call, "label": colour or label}
        return {k: same_block(v) for k, v in call.items()}
    return call


@pytest.fixture(scope="module")
def live():
    """The agent and our key. The key is a `SecretStr`, so a failing test's fixture repr
    can't print it into the output (pytest shows every argument of a failed test)."""
    s = Settings()
    if not s.nebius_api_key:
        pytest.skip("NEBIUS_API_KEY is not set")
    agent = ArmAgent(PROFILE, TOOLS, s.nebius_base_url, model=s.nemotron_model)
    return agent, SecretStr(s.nebius_api_key.get_secret_value())


@pytest.mark.parametrize("text", [p[0] for p in PHRASES])
def test_nemotron_agrees_with_the_rules(live, text):
    agent, key = live
    words = text.split()
    res = agent.run(words, key.get_secret_value())
    print(f"{text!r}: {[str(c) for c in res.calls]} in {res.latency_s:.2f} s, "
          f"{res.attempts} attempt(s)")  # fmt: skip
    expected = [same_block(c.to_dict()) for c in rules(PROFILE, words).calls]
    assert [same_block(c.to_dict()) for c in res.calls] == expected, res.errors
