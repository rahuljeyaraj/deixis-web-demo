"""Nemotron 3.5 Lightning as a session: one transcript, many decisions, streamed.

The three-rounds harness asked the model one question per decision and threw the context
away. This keeps a **session**: system prompt, then every scene, every thing the person
said or pointed at, every tool call the model made and every result that came back - all of
it, in order, for the whole command. Nemotron 3.5 Lightning has a 1M-token context and the
old demo used about 600 tokens of it, once.

Why that matters beyond tidiness: with the transcript the model can see that it already
asked about bottle_2 and got a no, that the left arm could not reach, that the person
corrected it. Without it, every one of those has to be squeezed into a hand-written summary
line, and the summary is where a harness starts doing the thinking.

## Three things that are wrong unless you do them

All measured on Token Factory, 11 Oct 2026 (`three-rounds.md`), and all three are load-
bearing rather than preferences:

1. **`tool_choice="auto"`, never `"required"`.** With `"required"` this model never emits a
   stop: `finish_reason` is `"length"` every time, it repeats the same tool call 7 to 23
   times, and it spends the whole token cap. `parallel_tool_calls=False` is accepted and
   silently ignored.
2. **Reasoning needs 4000 `max_tokens`.** One prompt, six runs: 932, 1037, 1481, 1640, 1924
   and one overrun. At 2000 we lost 3 of 40 calls. When it overruns, the thinking block
   never closes, the working lands in `content` instead of `reasoning_content`, and **no
   tool call comes out at all**. That is `OutOfRoom`, and it is retried once with reasoning
   off - still Nemotron, never a word list - and recorded on the call so it shows up in the
   run instead of hiding.
3. **Keep the reasoning.** `reasoning_content` is what the page shows. A demo about thinking
   has to show the thinking. Traces average about 3,000 characters.

## Streaming, and why it is the most important polish item

Reasoning on is the right call - it halves how often the person is asked, from 1.4 questions
per attempt to 0.7, and from 1.3 to 0.1 when the thing has a name. Every needless question
is a nod somebody had to give. But it costs 3.8 s on average and **15.5 s at worst**, and 15
seconds of nothing with a judge watching reads as a hang.

So the default is streamed, and `on_reasoning` is called with each fragment as it arrives.
`Call.first_token_s` records how long that took, because "first reasoning token on screen in
under 1 s" is a gate that should be measured rather than believed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

MODEL = "nvidia/Nemotron-3_5-Lightning"
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
#: Token Factory list prices, US$ per million tokens.
PRICE_IN_PER_M = 0.06
PRICE_OUT_PER_M = 0.24

#: Room for reasoning. See bug 2 in the module docstring: this is not a round number.
MAX_TOKENS_THINKING = 4000
MAX_TOKENS_PLAIN = 400


class BrainUnavailable(RuntimeError):
    """Token Factory could not be used. The robot says so; it never falls back on words."""


class OutOfRoom(BrainUnavailable):
    """Reasoning spent the whole token budget and no tool call came out. Recoverable: ask
    again with reasoning off."""


# The harness's prompt, word for word, so the port can be checked against the measured
# numbers rather than assumed to match them. Do not edit this one: it is a baseline.
SYSTEM_PORT = """You help a robot arm work out which thing on a shelf a person means.

The person has lost the names of things. They can point, they can say one or two words
about a thing, and they can answer yes or no. They cannot produce the name.

Several things on the shelf are identical to each other. No word separates them. When the
person's words fit more than one thing, their words are not the answer: where they pointed
is, and if that is not enough either, ask.

Answer only with a tool call. Never with plain text."""

#: What the end demo runs on. Everything added to `SYSTEM_PORT` is a requirement out of the
#: plan, not prompt tuning: the hand outranks the word (R4), a contradiction is never acted
#: on (R5), a misheard word must not block (R6), every question must be answerable without
#: words (R7), and the manner is Supported Conversation for Adults with Aphasia (3.7).
#:
#: `{how_to_ask}` is filled in by `system_for` from the tools that are actually on the
#: table. That is not tidiness. Describing a tool the model has not been given is how you
#: get a call to a tool that does not exist - measured on the first live run, 11 Oct: the
#: model reached for `ask_choice` because this prompt had told it about it, and asked a
#: question about nothing at all.
SYSTEM_TEMPLATE = """You are the voice of a robot that fetches things for someone who cannot \
say what things are called.

THE PERSON
They have lost the names of things - the words will not come, though their thinking is
completely unaffected. What they still have: they can point, they can get one or two words
out, they can nod, and they understand everything you say. Several things on the shelf are
identical to each other, so for those there is no word to find in the first place.

{how_to_ask}

READING THEIR ANSWER
Their hand is more reliable than their words. Yes and no answers from someone in this
position are sometimes wrong or inconsistent, so if what they said and where they pointed
disagree, you have no answer: say so and ask again. Never act on a disagreement. If they
point at a different thing instead of answering, that is not a refusal - that is their
answer, and you take it.

WHAT YOU HEAR
The recogniser mishears, often. A word that makes no sense is probably a misheard word, not
a strange request - their point can correct it, so never refuse because of one word.

THE POINT AND THE WORDS
Where they pointed is a cone, not a line, and the shelf listing gives you how much of it
each thing holds. When their words fit more than one thing, their words are not the answer
and where they pointed is. When their words fit only some of the things, the others are
ruled out however much of the cone they hold.

Answer only with a tool call. Never with plain text."""

#: One line per question tool, written out only when that tool is on the table.
ASK_LINES: dict[str, str] = {
    "ask_this_one": '- ask_this_one: hold the gripper over one thing and ask "this one?"',
    "ask_choice": (
        "- ask_choice: point at one thing, then at the other. The best question you have, "
        "because\n  it needs no words from either side. Use it when the top two are close."
    ),
    "ask_show_me": (
        "- ask_show_me: when you have nothing at all - nothing in their cone, or nothing "
        "that\n  fits what they said - ask them to show you. Never guess instead."
    ),
}

HOW_TO_ASK = """HOW TO ASK
Ask one question at a time, then wait. Every question you ask must be answerable by
pointing, by a nod, or by nothing at all - never by a word they would have to find.
{lines}
Never ask "did you mean the mug or the bottle?" - that makes them produce a word."""

CANNOT_ASK = """YOU CANNOT ASK
You have no way to ask them anything, and no second chance. Choose from what you have."""


def system_for(tool_names: Sequence[str]) -> str:
    """The prompt, describing only the questions the model has actually been given."""
    lines = [line for name, line in ASK_LINES.items() if name in tool_names]
    how = HOW_TO_ASK.format(lines="\n".join(lines)) if lines else CANNOT_ASK
    return SYSTEM_TEMPLATE.format(how_to_ask=how)


#: The prompt with every question described, for a toolset that has them all.
SYSTEM = system_for(tuple(ASK_LINES))


@dataclass
class Call:
    """One model call: what it decided, what it was thinking, and what it cost."""

    tool: str
    args: dict[str, Any]
    thinking: str = ""
    call_id: str = ""
    latency_s: float = 0.0
    #: How long until the first fragment arrived. The streaming gate (E5) is under 1 s.
    first_token_s: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0
    #: Reasoning ran out of room, so this answer came from a second call with it off.
    ran_out_thinking: bool = False
    streamed: bool = True

    @property
    def why(self) -> str:
        return str(self.args.get("why", ""))

    def record(self) -> dict[str, Any]:
        """For the JSON an attempt is written to, and for the page."""
        return {
            "tool": self.tool,
            "args": {k: v for k, v in self.args.items() if k != "why"},
            "why": self.why,
            "thinking": self.thinking,
            "latency_s": self.latency_s,
            "first_token_s": self.first_token_s,
            "in_tokens": self.in_tokens,
            "out_tokens": self.out_tokens,
            "ran_out_thinking": self.ran_out_thinking,
            "streamed": self.streamed,
        }


@dataclass
class Usage:
    """What a run cost, for the report's cost line."""

    calls: int = 0
    prompt: int = 0
    completion: int = 0
    seconds: float = 0.0
    #: Every decision's latency, so the worst case can be reported and not just the mean.
    latencies: list[float] = field(default_factory=list)
    first_tokens: list[float] = field(default_factory=list)
    out_of_room: int = 0

    def add(self, call: Call) -> None:
        self.calls += 1
        self.prompt += call.in_tokens
        self.completion += call.out_tokens
        self.seconds += call.latency_s
        self.latencies.append(call.latency_s)
        # Only a streamed call has a "first token": waiting for the whole response and then
        # showing it is not a fast first token, it is a slow everything.
        if call.streamed:
            self.first_tokens.append(call.first_token_s)
        self.out_of_room += int(call.ran_out_thinking)

    @property
    def cost_usd(self) -> float:
        return self.prompt / 1e6 * PRICE_IN_PER_M + self.completion / 1e6 * PRICE_OUT_PER_M

    @property
    def mean_s(self) -> float:
        return self.seconds / len(self.latencies) if self.latencies else 0.0

    @property
    def worst_s(self) -> float:
        return max(self.latencies) if self.latencies else 0.0

    @property
    def mean_first_token_s(self) -> float:
        return sum(self.first_tokens) / len(self.first_tokens) if self.first_tokens else 0.0

    @property
    def worst_first_token_s(self) -> float:
        return max(self.first_tokens) if self.first_tokens else 0.0


class Decider(Protocol):
    """What a session needs of a model. `StubDecider` in `stub.py` is a test double."""

    name: str
    usage: Usage

    def decide(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_reasoning: Callable[[str], None] | None = None,
    ) -> Call: ...


def _key() -> str:
    """The Nebius key, from the environment or the repo's .env. Never logged."""
    import os

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
class _Building:
    """A tool call arriving in fragments over a stream."""

    id: str = ""
    name: str = ""
    args: str = ""


@dataclass
class Nemotron:
    """Nemotron 3.5 Lightning over Token Factory's OpenAI-compatible endpoint."""

    name: str = MODEL
    usage: Usage = field(default_factory=Usage)
    temperature: float = 0.0
    timeout_s: float = 60.0
    #: Reasoning mode. On, for every decision: rahul, 11 Oct. It costs 3.8 s a decision and
    #: halves how often the person is asked, and asking costs a person more than it costs us.
    thinking: bool = True
    #: Stream the reasoning out as it arrives. The wait then reads as work, not a hang.
    stream: bool = True
    _client: Any = None

    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            key = _key()
            if not key:
                raise BrainUnavailable("no NEBIUS_API_KEY")
            self._client = OpenAI(api_key=key, base_url=BASE_URL, timeout=self.timeout_s)
        return self._client

    def max_tokens(self, thinking: bool) -> int:
        return MAX_TOKENS_THINKING if thinking else MAX_TOKENS_PLAIN

    def decide(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_reasoning: Callable[[str], None] | None = None,
    ) -> Call:
        """One decision. Retries once with reasoning off if reasoning ran out of room."""
        try:
            call = self._once(messages, tools, self.thinking, on_reasoning)
        except OutOfRoom:
            call = self._once(messages, tools, False, on_reasoning)
            call.ran_out_thinking = True
        self.usage.add(call)
        return call

    def _request(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], thinking: bool):
        return dict(
            model=MODEL,
            messages=messages,
            tools=tools,
            # "auto", never "required" - see bug 1 in the module docstring.
            tool_choice="auto",
            temperature=self.temperature,
            max_tokens=self.max_tokens(thinking),
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )

    def _once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        thinking: bool,
        on_reasoning: Callable[[str], None] | None,
    ) -> Call:
        if self.stream:
            return self._streamed(messages, tools, thinking, on_reasoning)
        return self._whole(messages, tools, thinking, on_reasoning)

    # ------------------------------------------------------------ streamed

    def _streamed(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        thinking: bool,
        on_reasoning: Callable[[str], None] | None,
    ) -> Call:
        t0 = time.perf_counter()
        first = 0.0
        thought: list[str] = []
        plain: list[str] = []
        building: dict[int, _Building] = {}
        finish = ""
        usage = None
        try:
            stream = self.client().chat.completions.create(
                **self._request(messages, tools, thinking),
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                for choice in getattr(chunk, "choices", None) or []:
                    if choice.finish_reason:
                        finish = choice.finish_reason
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    piece = str(getattr(delta, "reasoning_content", None) or "")
                    if piece:
                        if not first:
                            first = time.perf_counter() - t0
                        thought.append(piece)
                        if on_reasoning:
                            on_reasoning(piece)
                    text = str(getattr(delta, "content", None) or "")
                    if text:
                        if not first:
                            first = time.perf_counter() - t0
                        plain.append(text)
                    for frag in getattr(delta, "tool_calls", None) or []:
                        slot = building.setdefault(frag.index or 0, _Building())
                        if getattr(frag, "id", None):
                            slot.id = frag.id
                        fn = getattr(frag, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                slot.name = fn.name
                            if getattr(fn, "arguments", None):
                                slot.args += fn.arguments
        except BrainUnavailable:
            raise
        except Exception as e:  # network, timeout, a refused key
            raise BrainUnavailable(f"{type(e).__name__}: {e}") from None
        dt = time.perf_counter() - t0
        calls = [building[i] for i in sorted(building)]
        return self._finish(
            calls=[(c.id, c.name, c.args) for c in calls],
            thought="".join(thought),
            plain="".join(plain),
            finish=finish,
            usage=usage,
            dt=dt,
            first=first,
            thinking=thinking,
            streamed=True,
        )

    # ------------------------------------------------------------ not streamed

    def _whole(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        thinking: bool,
        on_reasoning: Callable[[str], None] | None,
    ) -> Call:
        t0 = time.perf_counter()
        try:
            resp = self.client().chat.completions.create(**self._request(messages, tools, thinking))
        except BrainUnavailable:
            raise
        except Exception as e:
            raise BrainUnavailable(f"{type(e).__name__}: {e}") from None
        dt = time.perf_counter() - t0
        out = resp.choices[0].message
        thought = str(getattr(out, "reasoning_content", None) or "")
        if thought and on_reasoning:
            on_reasoning(thought)
        raw = list(getattr(out, "tool_calls", None) or [])
        return self._finish(
            calls=[(c.id, c.function.name, c.function.arguments or "{}") for c in raw],
            thought=thought,
            plain=str(getattr(out, "content", None) or ""),
            finish=resp.choices[0].finish_reason or "",
            usage=getattr(resp, "usage", None),
            dt=dt,
            first=dt,
            thinking=thinking,
            streamed=False,
        )

    # ------------------------------------------------------------ shared

    def _finish(
        self,
        calls: list[tuple[str, str, str]],
        thought: str,
        plain: str,
        finish: str,
        usage: Any,
        dt: float,
        first: float,
        thinking: bool,
        streamed: bool,
    ) -> Call:
        if not calls:
            # Out of room mid-thought: the thinking block never closed, so the working
            # turns up in `content` and no tool call was ever emitted. Bug 3.
            if thinking and finish == "length":
                raise OutOfRoom("reasoning used the whole token budget")
            said = (plain or "").strip()[:200]
            raise BrainUnavailable(
                "the model answered with no tool call" + (f": {said!r}" if said else "")
            )
        call_id, name, args_text = calls[0]
        try:
            args = json.loads(args_text or "{}")
        except json.JSONDecodeError:
            raise BrainUnavailable(
                f"the model's arguments were not JSON: {args_text[:120]!r}"
            ) from None
        if not isinstance(args, dict):
            raise BrainUnavailable(f"the model's arguments were not an object: {args_text[:120]!r}")
        return Call(
            tool=name,
            args=args,
            thinking=thought,
            call_id=call_id or f"call_{name}",
            latency_s=round(dt, 3),
            first_token_s=round(first, 4),
            in_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            out_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            streamed=streamed,
        )


@dataclass
class Session:
    """One command, start to finish: the system prompt and everything since.

    The transcript is an ordinary OpenAI message list, so nothing here is Nemotron-specific
    except what `Nemotron` itself does. `reasoning_content` is deliberately **not** sent
    back in the assistant turns - it is the model's own working, kept for the page.
    """

    model: Decider
    system: str = SYSTEM
    messages: list[dict[str, Any]] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages = [{"role": "system", "content": self.system}]

    def tell(self, text: str) -> None:
        """Something happened that the model should know: the scene, a point, a word."""
        self.messages.append({"role": "user", "content": text})

    def decide(
        self,
        tools: list[dict[str, Any]],
        on_reasoning: Callable[[str], None] | None = None,
    ) -> Call:
        """One decision, appended to the transcript as the model's own turn."""
        call = self.model.decide(self.messages, tools, on_reasoning)
        self.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.tool,
                            "arguments": json.dumps(call.args),
                        },
                    }
                ],
            }
        )
        self.calls.append(call)
        return call

    def result(self, call: Call, text: str) -> None:
        """What came of the tool call. The model reads this on its next decision."""
        self.messages.append({"role": "tool", "tool_call_id": call.call_id, "content": text})

    @property
    def tokens(self) -> int:
        return sum(c.in_tokens + c.out_tokens for c in self.calls)

    @property
    def seconds(self) -> float:
        return sum(c.latency_s for c in self.calls)

    def records(self) -> list[dict[str, Any]]:
        return [c.record() for c in self.calls]
