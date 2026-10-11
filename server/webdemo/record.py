"""What each command cost, and the command log.

- `cost_usd`: a command's Token Factory tokens at list price (Nemotron 3.5 Lightning, $0.06 in
  and $0.24 out per million tokens, 6 Oct 2026; no cached price is published, so cached
  tokens are counted at the full input price). Settings hold the prices, to re-check.
- `CommandLog`: one JSON line per command and per page event (`logs/commands.jsonl`), for
  the measure list: M30 (right first time), M31 (last word to the guess, hosted), M33 (tokens
  and cost). It holds the words and what the robot did: no audio, no video, no address, no
  key. A visitor is a short hash of their random cookie.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PER_M = 1_000_000


def cost_usd(usage: dict[str, int], price_in: float, price_out: float) -> float:
    return round(
        (usage.get("prompt", 0) * price_in + usage.get("completion", 0) * price_out) / PER_M, 8
    )


def visitor_tag(sid: str) -> str:
    return hashlib.sha256(("deixis-web-demo:" + sid).encode()).hexdigest()[:10]


class CommandLog:
    """Appends JSON lines; one lock, so threads don't interleave. `path=None`: off."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, sid: str, **data: Any) -> None:
        if self.path is None:
            return
        row = {"utc": datetime.now(UTC).isoformat(timespec="milliseconds"), "kind": kind,
               "visitor": visitor_tag(sid), **data}  # fmt: skip
        line = json.dumps(row, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
