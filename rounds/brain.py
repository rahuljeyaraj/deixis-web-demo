"""The brain: Nemotron 3.5 Lightning on Nebius Token Factory, and a test double.

Two jobs, one per round that needs thinking:

- **`from_words`** (round 2): no pointing at all. The whole shelf, the person's few words,
  pick one. This is what a voice assistant has to work with, and on four identical bottles
  there is nothing to work with.
- **`step`** (round 3): the cone *and* the words, and - the part nobody builds - the model
  may **ask by pointing** instead of guessing. `ask_about(id)` hovers the gripper over one
  thing; the person answers yes or no, which costs them a nod. `choose(id)` commits. The
  model decides which to do.

Asking is the whole contribution, so the prompt says plainly that a yes/no answer is cheap
and a wrong commit is not. What we are measuring is whether the model takes that deal.

`StubBrain` is a **test double**, not a fallback: it is a word matcher, and the demo itself
must never fall back on one (rahul, 11 Oct) - either Nemotron answers or the robot says it
cannot. It exists so the harness can be tested, and run, without spending calls.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from rounds.pointer import Cone
from rounds.scene import Thing

MODEL = "nvidia/Nemotron-3_5-Lightning"
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
#: Token Factory list prices, US$ per million tokens (as in the demo's settings.py).
PRICE_IN_PER_M = 0.06
PRICE_OUT_PER_M = 0.24


class BrainUnavailable(RuntimeError):
    """Token Factory could not be used. The robot says so; it never falls back on words."""


@dataclass
class Decision:
    """What the brain decided to do next."""

    action: Literal["choose", "ask"]
    id: str
    why: str = ""


@dataclass
class Usage:
    """What the run cost, for the report's cost line."""

    calls: int = 0
    prompt: int = 0
    completion: int = 0

    def add(self, u: Any) -> None:
        self.calls += 1
        self.prompt += int(getattr(u, "prompt_tokens", 0) or 0)
        self.completion += int(getattr(u, "completion_tokens", 0) or 0)

    @property
    def cost_usd(self) -> float:
        return self.prompt / 1e6 * PRICE_IN_PER_M + self.completion / 1e6 * PRICE_OUT_PER_M


class Brain(Protocol):
    name: str
    usage: Usage

    def from_words(self, things: list[Thing], said: str) -> str: ...

    def step(
        self, things: list[Thing], cone: Cone, said: str, asked: list[tuple[str, bool]]
    ) -> Decision: ...


SYSTEM = """You help a robot arm work out which thing on a shelf a person means.

The person has lost the names of things. They can point, they can say one or two words
about a thing, and they can answer yes or no. They cannot produce the name.

Several things on the shelf are identical to each other. No word separates them. When the
person's words fit more than one thing, their words are not the answer: where they pointed
is, and if that is not enough either, ask.

Answer only with a tool call. Never with plain text."""

ASK_DESCRIPTION = (
    "Point the gripper at one thing and ask the person: this one? They answer yes or no. "
    "Cheap: it costs them a nod. Use it whenever two things are close to equally likely."
)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "choose",
            "description": "Commit: this is the thing the person means. Wrong is costly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "the thing's id"},
                    "why": {"type": "string", "description": "one short line"},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_about",
            "description": ASK_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "the thing to point at"},
                    "why": {"type": "string", "description": "one short line"},
                },
                "required": ["id"],
            },
        },
    },
]

CHOOSE_ONLY = [TOOLS[0]]


def _shelf_lines(things: list[Thing], cone: Cone | None = None) -> str:
    weights = {c.id: c.weight for c in (cone.candidates if cone else [])}
    out = []
    for t in things:
        w = weights.get(t.id)
        share = f"{w:.2f}" if w is not None else "  -  "
        out.append(f"  {t.id:<10} {t.seen:<24} {share}")
    return "\n".join(out)


def _words_message(things: list[Thing], said: str) -> str:
    return (
        "The person wants one thing off the shelf. They did not point - you have only "
        "their words:\n"
        f'  "{said}"\n\n'
        "On the shelf:\n" + _shelf_lines(things) + "\n\n"
        "Choose one. You have no other information and you cannot ask."
    )


def _step_message(things: list[Thing], cone: Cone, said: str, asked: list[tuple[str, bool]]) -> str:
    said_line = f'They also said: "{said}"' if said else "They said nothing."
    lines = [
        "The person pointed at the shelf. " + said_line,
        "",
        "The last column is how much of their pointing cone each thing holds (pointing "
        f"error is about {cone.sigma_deg:.0f} degrees per axis, so neighbours overlap). A "
        "dash means it was outside the cone.",
        "",
        _shelf_lines(things, cone),
        "",
    ]
    if asked:
        lines.append("You have already pointed at these and asked:")
        lines += [f"  {i} -> {'yes' if ok else 'no'}" for i, ok in asked]
    else:
        lines.append("You have not asked anything yet.")
    return "\n".join(lines)


def _key() -> str:
    """The Nebius key, from the environment or the repo's .env. Never logged."""
    key = os.environ.get("NEBIUS_API_KEY", "").strip()
    if key:
        return key
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "NEBIUS_API_KEY":
                return value.strip().strip('"').strip("'")
    return ""


@dataclass
class Nemotron:
    """Nemotron 3.5 Lightning, over Token Factory's OpenAI-compatible endpoint."""

    name: str = MODEL
    usage: Usage = field(default_factory=Usage)
    temperature: float = 0.0
    timeout_s: float = 30.0
    _client: Any = None

    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            key = _key()
            if not key:
                raise BrainUnavailable("no NEBIUS_API_KEY")
            self._client = OpenAI(api_key=key, base_url=BASE_URL, timeout=self.timeout_s)
        return self._client

    def _call(self, message: str, tools: list[dict[str, Any]]) -> Decision:
        try:
            resp = self.client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": message},
                ],
                tools=tools,
                tool_choice="required",
                temperature=self.temperature,
                max_tokens=256,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as e:  # network, timeout, a refused key
            raise BrainUnavailable(f"{type(e).__name__}: {e}") from None
        self.usage.add(getattr(resp, "usage", None))
        calls = list(getattr(resp.choices[0].message, "tool_calls", None) or [])
        if not calls:
            raise BrainUnavailable("the model answered with no tool call")
        call = calls[0]
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            raise BrainUnavailable("the model's arguments were not JSON") from None
        action = "ask" if call.function.name == "ask_about" else "choose"
        return Decision(action, str(args.get("id", "")), str(args.get("why", "")))

    def from_words(self, things: list[Thing], said: str) -> str:
        return self._call(_words_message(things, said), CHOOSE_ONLY).id

    def step(
        self, things: list[Thing], cone: Cone, said: str, asked: list[tuple[str, bool]]
    ) -> Decision:
        return self._call(_step_message(things, cone, said, asked), TOOLS)


@dataclass
class StubBrain:
    """A word matcher, as a **test double** only - see the module docstring."""

    name: str = "stub (word matcher, test double)"
    usage: Usage = field(default_factory=Usage)

    @staticmethod
    def _fits(t: Thing, said: str) -> bool:
        words = {w for w in said.lower().split() if len(w) > 2}
        text = (t.seen + " " + " ".join(t.says)).lower()
        return bool(words) and any(w in text for w in words)

    def from_words(self, things: list[Thing], said: str) -> str:
        self.usage.calls += 1
        hits = [t.id for t in things if self._fits(t, said)]
        return hits[0] if hits else things[0].id

    def step(
        self, things: list[Thing], cone: Cone, said: str, asked: list[tuple[str, bool]]
    ) -> Decision:
        self.usage.calls += 1
        done = {i for i, _ in asked}
        index = {t.id: t for t in things}
        fits = [c.id for c in cone.candidates if self._fits(index[c.id], said)]
        live = [i for i in (fits or cone.ids()) if i not in done]
        if not live:
            return Decision("choose", (cone.best or things[0].id), "nothing left to try")
        if len(live) == 1:
            return Decision("choose", live[0], "only one fits the words")
        return Decision("ask", live[0], "two or more fit; ask")
