"""How much of our key each visitor may use, the burst limit, and the visitor's own key (plan
8.1, 8.2, W11). The key is never in a reply, a log line or the server's memory after the call."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from test_agent import FakeLLM
from webdemo.app import create_app
from webdemo.limits import BANNER, Caps, Limits

OURS = "sk-ours-0000000000"
THEIRS = "sk-visitor-9999999999"
YES = [("confirm", {})]


class Clock:
    def __init__(self, t: float = 1_760_000_000.0) -> None:  # 9 Oct 2025, 08:53 UTC
        self.t = t

    def __call__(self) -> float:
        return self.t


class Refused(Exception):
    """What the openai client raises for a bad key."""

    status_code = 401


# --- the counts ---


def test_each_visitor_has_a_share_of_our_key():
    lim = Limits(Caps(visitor=2, address=90, total=100), Clock())
    assert lim.left("a", "net") == (2, None)
    lim.spend("a", "net")
    lim.spend("a", "net")
    assert lim.left("a", "net") == (0, "visitor")
    assert lim.left("b", "net") == (2, None)


def test_a_network_and_everyone_have_a_share_too():
    lim = Limits(Caps(visitor=2, address=3, total=5), Clock())
    for v in ["a", "a", "b"]:
        lim.spend(v, "net")
    assert lim.left("c", "net") == (0, "address")  # one network, many cookies
    lim.spend("d", "home")
    lim.spend("d", "home")
    assert lim.left("e", "cafe") == (0, "total")


def test_the_counts_start_again_at_midnight_utc():
    clock = Clock(1_760_054_399.0)  # 23:59:59 UTC
    lim = Limits(Caps(visitor=1), clock)
    lim.spend("a", "net")
    assert lim.left("a", "net") == (0, "visitor")
    clock.t += 2
    assert lim.left("a", "net") == (1, None)


def test_the_burst_limit():
    clock = Clock()
    lim = Limits(Caps(burst_s=1.5, address_burst_s=0.5), clock)
    assert lim.wait_s("a", "net") == 0
    clock.t += 1.0
    assert lim.wait_s("a", "net") == pytest.approx(0.5)  # the same visitor waits
    assert lim.wait_s("b", "net") == 0  # another visitor on the network doesn't
    clock.t += 0.2
    assert lim.wait_s("c", "net") == pytest.approx(0.3)  # but not all at once
    clock.t += 0.4
    assert lim.wait_s("a", "net") == 0


def test_addresses_are_hashed_and_an_ipv6_home_is_one_address():
    lim = Limits(Caps())
    h = lim.address_of("203.0.113.7")
    assert "203.0.113" not in h and len(h) == 16
    assert lim.address_of("::ffff:203.0.113.7") == h
    assert lim.address_of("2001:db8:1:2::1") == lim.address_of("2001:db8:1:2:ffff::9")
    assert lim.address_of("2001:db8:1:3::1") != lim.address_of("2001:db8:1:2::1")
    assert Limits(Caps()).address_of("203.0.113.7") != h  # a new secret per process


# --- the routes ---


def caps(**kw) -> Caps:
    """Caps with no burst limit unless a test sets one."""
    return Caps(**{"burst_s": 0, "address_burst_s": 0, **kw})


def app(settings, core, llm: FakeLLM, caps: Caps, clock: Clock | None = None, key: str = OURS):
    settings.nebius_api_key = SecretStr(key) if key else None
    limits = Limits(caps, clock or Clock())
    return TestClient(create_app(settings, client=core, llm_factory=llm.factory, limits=limits))


def say(c: TestClient, text: str, key: str | None = None, **headers):
    if key:
        headers["X-Nebius-Key"] = key
    return c.post("/api/command", data={"meta": json.dumps({"text": text})}, headers=headers)


def test_past_the_share_the_word_matcher_answers(settings, core):
    llm = FakeLLM(YES, YES)
    with app(settings, core, llm, caps(visitor=2)) as c:
        assert c.get("/api/info").json()["limits"]["left_today"] == 2
        first, second, third = (say(c, "yes").json() for _ in range(3))
        info = c.get("/api/info").json()
    assert llm.keys == [OURS, OURS]
    sources = [r["agent"]["source"] for r in (first, second, third)]
    assert sources == ["nemotron", "nemotron", "rules"]
    assert first["limits"]["mode"] == "ours" and first["limits"]["left_today"] == 1
    assert first["limits"]["banner"] is None
    lim = third["limits"]
    assert lim["mode"] == "simple" and lim["why"] == "visitor" and lim["left_today"] == 0
    assert lim["banner"] == BANNER["visitor"] and lim["own_key_helps"] is True
    assert third["agent"]["errors"][0] == "today’s free AI commands are used: simple mode"
    assert third["result"]["calls"][0]["name"] == "confirm"  # the word matcher still works
    assert info["limits"]["why"] == "visitor"
    nemotron = next(m for m in info["models"] if m["id"] == "nemotron")
    assert nemotron["off"] == "off for you right now: today’s free commands are used"


def test_clearing_cookies_doesnt_reset_the_share(settings, core):
    llm = FakeLLM(YES, YES, YES)
    with app(settings, core, llm, caps(visitor=2, address=3)) as c:
        say(c, "yes")
        say(c, "yes")
        c.cookies.clear()  # a new visitor id, the same address
        assert say(c, "yes").json()["limits"]["mode"] == "ours"
        c.cookies.clear()
        r = say(c, "yes").json()
    assert r["limits"]["why"] == "address" and len(llm.keys) == 3


def test_everyones_budget(settings, core):
    settings.proxy_hops = 1
    llm = FakeLLM(YES, YES)
    with app(settings, core, llm, caps(total=2)) as c:
        for n in range(3):
            c.cookies.clear()
            r = say(c, "yes", **{"X-Forwarded-For": f"198.51.100.{n}"}).json()
    assert r["limits"]["why"] == "total"
    assert r["limits"]["banner"].startswith("Today’s AI budget is used up")


def test_behind_a_proxy_the_left_of_forwarded_for_is_ignored(settings, core):
    """Anyone can write the left end of X-Forwarded-For; the proxy adds the real address."""
    settings.proxy_hops = 1
    llm = FakeLLM(YES)
    with app(settings, core, llm, caps(visitor=5, address=1)) as c:
        say(c, "yes", **{"X-Forwarded-For": "1.1.1.1, 198.51.100.7"})
        c.cookies.clear()
        r = say(c, "yes", **{"X-Forwarded-For": "2.2.2.2, 198.51.100.7"}).json()
    assert r["limits"]["why"] == "address"


def test_stop_costs_nothing(settings, core):
    llm = FakeLLM()  # no reply queued: a call would fail the test
    with app(settings, core, llm, caps(visitor=2)) as c:
        r = say(c, "stop").json()
    assert r["agent"]["source"] == "rules" and r["limits"]["left_today"] == 2 and not llm.keys


def test_an_own_key_skips_the_share(settings, core):
    llm = FakeLLM(YES, YES)
    with app(settings, core, llm, caps(visitor=0)) as c:
        a = say(c, "yes", key=THEIRS).json()
        b = say(c, "yes", key=THEIRS).json()
        mine = say(c, "yes").json()  # without it: past the share
    assert llm.keys == [THEIRS, THEIRS]
    assert a["agent"]["source"] == b["agent"]["source"] == "nemotron"
    assert a["limits"]["mode"] == "own" and a["limits"]["banner"] is None
    assert mine["limits"]["mode"] == "simple"


def test_a_refused_key_falls_back_to_ours_and_says_so(settings, core, caplog):
    caplog.set_level(logging.DEBUG)
    llm = FakeLLM(Refused(f"Incorrect API key provided: {THEIRS}"), YES)
    with app(settings, core, llm, caps(visitor=2)) as c:
        r = say(c, "yes", key=THEIRS)
    body = r.json()
    assert llm.keys == [THEIRS, OURS]
    assert body["key_refused"] is True
    assert body["agent"]["source"] == "nemotron" and body["limits"]["mode"] == "ours"
    assert body["limits"]["left_today"] == 1
    assert THEIRS not in r.text and THEIRS not in caplog.text


def test_a_refused_key_past_the_share_is_simple_mode(settings, core):
    llm = FakeLLM(Refused("401"))
    with app(settings, core, llm, caps(visitor=0)) as c:
        body = say(c, "yes", key=THEIRS).json()
    assert body["key_refused"] is True and body["limits"]["mode"] == "simple"
    assert llm.keys == [THEIRS]


def test_no_key_on_the_server(settings, core):
    llm = FakeLLM(YES)
    with app(settings, core, llm, caps(), key="") as c:
        info = c.get("/api/info").json()["limits"]
        own = say(c, "yes", key=THEIRS).json()
    assert info["why"] == "no_key" and info["own_key_helps"] is True and info["left_today"] == 0
    assert own["agent"]["source"] == "nemotron" and own["limits"]["mode"] == "own"


def test_rules_mode_ignores_every_key(settings, core):
    settings.ai = "rules"
    llm = FakeLLM()
    with app(settings, core, llm, caps()) as c:
        r = say(c, "yes", key=THEIRS).json()
    assert not llm.keys and r["limits"]["why"] == "rules" and r["limits"]["own_key_helps"] is False


def test_the_burst_limit_on_the_route(settings, core):
    settings.burst_s = 1.5
    clock = Clock()
    llm = FakeLLM(YES, YES)
    with app(settings, core, llm, Caps(burst_s=1.5, address_burst_s=0.5), clock) as c:
        assert say(c, "yes", key=THEIRS).status_code == 200
        clock.t += 1.0
        busy = say(c, "yes", key=THEIRS)  # an own key too
        clock.t += 0.6
        again = say(c, "yes", key=THEIRS)
    assert busy.status_code == 429 and busy.headers["retry-after"] == "1"
    assert busy.json()["wait_s"] == pytest.approx(0.5)
    assert again.status_code == 200 and len(llm.keys) == 2


def test_a_first_visit_that_waits_still_gets_its_cookie(settings, core):
    clock = Clock()
    with app(settings, core, FakeLLM(YES), Caps(burst_s=1.5, address_burst_s=0.5), clock) as c:
        say(c, "yes")
        c.cookies.clear()
        busy = say(c, "yes")  # a new visitor on the same address, at once
    assert busy.status_code == 429 and "sid=" in busy.headers["set-cookie"]


def test_the_key_is_never_sent_back_logged_or_kept(settings, core, caplog):
    """Plan 8.2: send a key every way it can go wrong and search every reply, every log line
    and the server's memory for it."""
    caplog.set_level(logging.DEBUG)
    long_key = "sk-" + "x" * 5000
    llm = FakeLLM(
        YES,
        TimeoutError(f"timed out talking to {THEIRS}"),
        Refused(f"Incorrect API key provided: {THEIRS[:8]}****{THEIRS[-4:]}"),
        YES,
    )
    with app(settings, core, llm, caps(visitor=5)) as c:
        replies = [
            say(c, "yes", key=THEIRS),
            say(c, "put that here", key=THEIRS),
            say(c, "yes", key=THEIRS),
            say(c, "yes", key=long_key),
            c.get("/api/info"),
            c.get("/healthz"),
        ]
        state = c.app.state
        kept = repr(vars(state.limits)) + repr(vars(state.sessions)) + repr(vars(state.agent))
    assert replies[3].status_code == 400
    for secret in (THEIRS, THEIRS[:8], OURS, long_key[:40]):
        for r in replies:
            assert secret not in r.text and secret not in str(r.headers)
        assert secret not in caplog.text
        assert secret not in kept
