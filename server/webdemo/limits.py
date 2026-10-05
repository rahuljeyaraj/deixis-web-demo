"""How much of our Nebius key each visitor may use (plan 8.1), and the burst limit (R9).

With our key a visitor gets `visitor` Nemotron commands a day, an address (several visitors
behind one network, or one visitor clearing cookies) `address`, and everyone together `total`
(about $1 a day). Past any of them the visitor gets simple mode: the rule parser, with a
banner. A visitor's own key skips the daily caps, not the burst limit.

A visitor is the random `sid` cookie; an address is a keyed hash of the IP address (IPv6 by
its /64), made with a secret that lives only in this process, so it can't be turned back into
the address. Everything is in memory and starts again at midnight UTC (and on a restart).
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import secrets
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

# Why a visitor is in simple mode: their own day's share, their network's, everyone's, the
# server has no key, or the server runs the rule parser only (`ai = "rules"`).
Why = Literal["visitor", "address", "total", "no_key", "rules"]


@dataclass(frozen=True)
class Caps:
    visitor: int = 30  # Nemotron commands a day with our key, per visitor [D4]
    address: int = 90  # per address: three visitors' worth, for a shared network
    total: int = 2000  # for everyone, about $1 a day [D4]
    burst_s: float = 1.5  # one command per visitor this often (own key too)
    address_burst_s: float = 0.5  # and per address, looser for a shared network


class Limits:
    """The day's counts, in memory. Thread-safe; `clock` is for tests."""

    def __init__(self, caps: Caps, clock: Callable[[], float] = time.time) -> None:
        self.caps = caps
        self._clock = clock
        self._secret = secrets.token_bytes(32)
        self._lock = threading.Lock()
        self._day = ""
        self._visitors: Counter[str] = Counter()
        self._addresses: Counter[str] = Counter()
        self._total = 0
        self._last: dict[str, float] = {}  # the last command's time, per visitor and address

    def address_of(self, ip: str) -> str:
        """A keyed hash of the address (an IPv6 address by its /64, which one home gets)."""
        try:
            a = ipaddress.ip_address(ip)
            if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped is None:
                ip = str(ipaddress.ip_network(f"{a}/64", strict=False).network_address)
            elif isinstance(a, ipaddress.IPv6Address):
                ip = str(a.ipv4_mapped)
        except ValueError:
            pass  # not an address (the test client's "testclient"): hash it as it is
        return hmac.new(self._secret, ip.encode(), hashlib.sha256).hexdigest()[:16]

    def _today(self) -> None:
        day = datetime.fromtimestamp(self._clock(), UTC).date().isoformat()
        if day != self._day:
            self._day = day
            self._visitors.clear()
            self._addresses.clear()
            self._total = 0
            self._last.clear()

    def wait_s(self, visitor: str, address: str) -> float:
        """Seconds to wait before this command may run (0: go, and it is noted)."""
        with self._lock:
            self._today()
            now = self._clock()
            wait = max(
                self._last.get("v:" + visitor, -1e9) + self.caps.burst_s - now,
                self._last.get("a:" + address, -1e9) + self.caps.address_burst_s - now,
            )
            if wait > 0:
                return round(wait, 3)
            self._last["v:" + visitor] = self._last["a:" + address] = now
            if len(self._last) > 20_000:  # forget visitors who stopped a while ago
                old = now - 60
                self._last = {k: t for k, t in self._last.items() if t > old}
            return 0.0

    def left(self, visitor: str, address: str) -> tuple[int, Why | None]:
        """Commands left today with our key, and why there are none (else None)."""
        with self._lock:
            self._today()
            left = {  # in this order, so "everyone's is used up" wins a tie
                "total": self.caps.total - self._total,
                "address": self.caps.address - self._addresses[address],
                "visitor": self.caps.visitor - self._visitors[visitor],
            }
        why = min(left, key=lambda k: left[k])
        n = max(0, left[why])
        return n, (why if n == 0 else None)  # type: ignore[return-value]

    def spend(self, visitor: str, address: str) -> None:
        """One Nemotron command with our key."""
        with self._lock:
            self._today()
            self._visitors[visitor] += 1
            self._addresses[address] += 1
            self._total += 1


# What the page says (plan 8.1): the banner in simple mode, the note on the models list's
# Nemotron line, and why in the details panel.
BANNER: dict[Why, str] = {
    "visitor": "You’ve used today’s free AI commands, so a simple word matcher is answering.",
    "address": "Today’s free AI commands for your network are used, so a simple word matcher is "
    "answering.",
    "total": "Today’s AI budget is used up, so a simple word matcher is answering.",
    "no_key": "This server has no Nebius key, so a simple word matcher is answering.",
    "rules": "The AI is turned off on this server, so a simple word matcher is answering.",
}
OFF: dict[Why, str] = {
    "visitor": "off for you right now: today’s free commands are used",
    "address": "off for your network right now: today’s free commands are used",
    "total": "off right now: today’s budget is used up",
    "no_key": "off on this server: it has no key",
    "rules": "off on this server",
}
NO_KEY_WHY: dict[Why, str] = {
    "visitor": "today’s free AI commands are used: simple mode",
    "address": "today’s free AI commands for this network are used: simple mode",
    "total": "today’s AI budget is used up: simple mode",
    "no_key": "no Nebius key on this server: simple mode",
    "rules": "the server runs the word matcher only",
}

Mode = Literal["ours", "own", "simple"]


def view(mode: Mode, left: int, per_day: int, why: Why | None) -> dict[str, object]:
    """`limits` in `/api/info` and in each command's reply."""
    simple = mode == "simple" and why is not None
    return {
        "mode": mode,
        "left_today": left,  # commands left today with our key
        "per_day": per_day,
        "why": why if simple else None,
        "banner": BANNER[why] if simple else None,  # type: ignore[index]
        "off": OFF[why] if simple else None,  # type: ignore[index]
        "own_key_helps": simple and why != "rules",
    }
