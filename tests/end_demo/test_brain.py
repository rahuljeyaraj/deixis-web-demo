"""How Nemotron must be called, pinned so the three measured bugs cannot come back.

All three were measured on Token Factory on 11 Oct 2026 and all three fail *silently*,
which is why they are tests and not comments:

1. `tool_choice="required"` never stops on this model - every response comes back
   `finish_reason="length"` with the same tool call repeated 7 to 23 times.
2. Reasoning needs 4000 tokens of room. At 2000 it overran on 3 of 40 calls, and when it
   overruns there is no tool call at all - the working lands in `content`.
3. `reasoning_content` is what the page shows, so it has to survive the round trip.

The fake client below is the whole OpenAI surface these tests need. It is a fake, not a
fallback: nothing in `end_demo` can reach it, and the robot has no word matcher to fall back
on anywhere (plan 8.4).
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from end_demo.brain import (
    MAX_TOKENS_PLAIN,
    MAX_TOKENS_THINKING,
    BrainUnavailable,
    Call,
    Nemotron,
    OutOfRoom,
    Session,
    Usage,
)
from end_demo.tools import Toolset


def tool_call(name: str, args: dict, call_id: str = "c1"):
    return NS(id=call_id, function=NS(name=name, arguments=json.dumps(args)))


class FakeClient:
    """Records every request, replies with whatever the test queued."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        # A copy, because the real client serialises straight away: without this the fake
        # would see messages appended after the call and the test would be reading the
        # future.
        self.requests.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        if not self.replies:
            raise AssertionError("the fake client ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def whole(tool: str | None = "choose", args=None, thinking="because", finish="stop", content=""):
    """A non-streamed response."""
    calls = [tool_call(tool, args or {"id": "bottle_2"})] if tool else None
    return NS(
        choices=[
            NS(
                message=NS(tool_calls=calls, reasoning_content=thinking, content=content),
                finish_reason=finish,
            )
        ],
        usage=NS(prompt_tokens=120, completion_tokens=900),
    )


def chunks(pieces, tool="choose", args=None, finish="stop", with_usage=True):
    """A streamed response: reasoning in fragments, then the tool call in fragments."""
    out = []
    for piece in pieces:
        out.append(NS(choices=[NS(delta=NS(reasoning_content=piece, content=None,
                                          tool_calls=None), finish_reason=None)],
                      usage=None))  # fmt: skip
    if tool:
        text = json.dumps(args or {"id": "bottle_2"})
        out.append(
            NS(
                choices=[
                    NS(
                        delta=NS(
                            reasoning_content=None,
                            content=None,
                            tool_calls=[NS(index=0, id="c1", function=NS(name=tool, arguments=""))],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
        )
        half = len(text) // 2
        for part in (text[:half], text[half:]):
            out.append(
                NS(
                    choices=[
                        NS(
                            delta=NS(
                                reasoning_content=None,
                                content=None,
                                tool_calls=[
                                    NS(index=0, id=None, function=NS(name=None, arguments=part))
                                ],
                            ),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                )
            )
    out.append(NS(choices=[NS(delta=NS(reasoning_content=None, content=None, tool_calls=None),
                              finish_reason=finish)], usage=None))  # fmt: skip
    if with_usage:
        out.append(NS(choices=[], usage=NS(prompt_tokens=120, completion_tokens=900)))
    return out


def brain(replies, **kw) -> Nemotron:
    b = Nemotron(**kw)
    b._client = FakeClient(replies)
    return b


TOOLS = Toolset.selection().as_list()
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


# ---------------------------------------------------------------- bug 1: tool_choice


def test_tool_choice_is_always_auto():
    """Never "required": with it this model never emits a stop and spends the whole cap."""
    b = brain([whole()], stream=False)
    b.decide(MESSAGES, TOOLS)
    assert b._client.requests[0]["tool_choice"] == "auto"


def test_parallel_tool_calls_is_not_relied_on():
    """It is accepted and silently ignored on this endpoint, so asking for it would be a
    comment pretending to be a setting."""
    b = brain([whole()], stream=False)
    b.decide(MESSAGES, TOOLS)
    assert "parallel_tool_calls" not in b._client.requests[0]


# ---------------------------------------------------------------- bug 2: room to think


def test_reasoning_gets_four_thousand_tokens():
    b = brain([whole()], stream=False, thinking=True)
    b.decide(MESSAGES, TOOLS)
    assert b._client.requests[0]["max_tokens"] == MAX_TOKENS_THINKING == 4000


def test_without_reasoning_it_needs_almost_none():
    b = brain([whole(thinking="")], stream=False, thinking=False)
    b.decide(MESSAGES, TOOLS)
    assert b._client.requests[0]["max_tokens"] == MAX_TOKENS_PLAIN


def test_reasoning_mode_is_asked_for_explicitly():
    b = brain([whole()], stream=False, thinking=True)
    b.decide(MESSAGES, TOOLS)
    kwargs = b._client.requests[0]["extra_body"]["chat_template_kwargs"]
    assert kwargs == {"enable_thinking": True}


def test_running_out_of_room_is_retried_once_with_reasoning_off_and_recorded():
    """The failure mode: thinking never closes, no tool call comes out, finish_reason is
    "length". The retry is still Nemotron - never a word list - and it is marked so an
    overrun shows up in the run instead of hiding in it."""
    b = brain([whole(tool=None, finish="length", content="still thinking..."), whole()],
              stream=False)  # fmt: skip
    call = b.decide(MESSAGES, TOOLS)
    assert call.tool == "choose"
    assert call.ran_out_thinking
    assert b.usage.out_of_room == 1
    # The retry asked the same model with reasoning off.
    second = b._client.requests[1]
    assert second["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert second["model"] == b._client.requests[0]["model"]


def test_an_overrun_on_the_retry_is_not_retried_again():
    b = brain([whole(tool=None, finish="length"), whole(tool=None, finish="length")], stream=False)
    with pytest.raises(BrainUnavailable):
        b.decide(MESSAGES, TOOLS)


def test_no_tool_call_without_length_is_a_plain_failure():
    """The model answered in words. That is unusable, and it is not an out-of-room."""
    b = brain([whole(tool=None, finish="stop", content="I think you mean the bottle")],
              stream=False)  # fmt: skip
    with pytest.raises(BrainUnavailable, match="no tool call"):
        b.decide(MESSAGES, TOOLS)


def test_out_of_room_is_a_kind_of_unavailable():
    assert issubclass(OutOfRoom, BrainUnavailable)


# ---------------------------------------------------------------- bug 3: keep the thinking


def test_the_reasoning_survives_the_round_trip():
    b = brain([whole(thinking="the word fits four things, so the point decides")], stream=False)
    call = b.decide(MESSAGES, TOOLS)
    assert "the point decides" in call.thinking


def test_the_reasoning_is_not_sent_back_in_the_transcript():
    """It is the model's own working, kept for the page. Sending it back as an assistant
    turn would grow the context with something the model did not say to anyone."""
    session = Session(brain([whole(thinking="a long trace")], stream=False))
    session.tell("the shelf")
    session.decide(TOOLS)
    assert all("a long trace" not in json.dumps(m) for m in session.messages)
    assert session.calls[0].thinking == "a long trace"


# ---------------------------------------------------------------- streaming (E5)


def test_streaming_puts_the_reasoning_out_as_it_arrives():
    seen: list[str] = []
    b = brain([chunks(["the word ", "fits four ", "things"])], stream=True)
    call = b.decide(MESSAGES, TOOLS, seen.append)
    assert "".join(seen) == "the word fits four things"
    assert call.thinking == "the word fits four things"
    assert call.streamed


def test_streaming_reassembles_a_tool_call_from_fragments():
    b = brain([chunks(["mm"], tool="ask_this_one", args={"id": "mug_1", "why": "two mugs"})])
    call = b.decide(MESSAGES, TOOLS)
    assert call.tool == "ask_this_one"
    assert call.args == {"id": "mug_1", "why": "two mugs"}


def test_streaming_asks_for_usage_and_keeps_it():
    b = brain([chunks(["mm"])])
    call = b.decide(MESSAGES, TOOLS)
    assert b._client.requests[0]["stream_options"] == {"include_usage": True}
    assert call.in_tokens == 120 and call.out_tokens == 900


def test_the_first_token_is_timed_so_the_gate_can_be_measured():
    """ "First reasoning token on screen in under 1 s" is a gate, so it is recorded rather
    than believed. A streamed call always has a first-token time."""
    b = brain([chunks(["mm"])])
    call = b.decide(MESSAGES, TOOLS)
    assert call.first_token_s <= call.latency_s
    assert b.usage.first_tokens == [call.first_token_s]
    # A call that was not streamed has no first-token time to report.
    plain = brain([whole()], stream=False)
    plain.decide(MESSAGES, TOOLS)
    assert plain.usage.first_tokens == []


def test_a_stream_that_runs_out_of_room_is_retried_too():
    b = brain([chunks([], tool=None, finish="length"), chunks(["ok"])])
    call = b.decide(MESSAGES, TOOLS)
    assert call.ran_out_thinking and call.tool == "choose"


def test_a_network_failure_is_unavailable_not_a_crash():
    b = brain([RuntimeError("connection reset")], stream=True)
    with pytest.raises(BrainUnavailable, match="connection reset"):
        b.decide(MESSAGES, TOOLS)


def test_arguments_that_are_not_json_are_a_failure_not_a_guess():
    bad = NS(
        choices=[
            NS(
                message=NS(
                    tool_calls=[NS(id="c1", function=NS(name="choose", arguments="{id: bottle"))],
                    reasoning_content="",
                    content="",
                ),
                finish_reason="stop",
            )
        ],
        usage=NS(prompt_tokens=1, completion_tokens=1),
    )
    b = brain([bad], stream=False)
    with pytest.raises(BrainUnavailable, match="not JSON"):
        b.decide(MESSAGES, TOOLS)


# ---------------------------------------------------------------- the session


def test_the_transcript_keeps_everything():
    """The whole point of a session: the model's earlier question and the answer to it are
    in front of it, not summarised for it."""
    b = brain([whole(tool="ask_this_one", args={"id": "bottle_2"}), whole()], stream=False)
    session = Session(b)
    session.tell("the shelf, and they pointed")
    first = session.decide(TOOLS)
    session.result(first, "They said no. bottle_2 is not the one.")
    session.decide(TOOLS)

    roles = [m["role"] for m in session.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    # The second call saw all of it.
    sent = b._client.requests[1]["messages"]
    assert len(sent) == 4
    assert sent[3]["content"].startswith("They said no")
    assert sent[2]["tool_calls"][0]["function"]["name"] == "ask_this_one"


def test_the_tool_result_is_tied_to_the_call_it_answers():
    b = brain([whole(tool="ask_this_one", args={"id": "mug_1"})], stream=False)
    session = Session(b)
    session.tell("x")
    call = session.decide(TOOLS)
    session.result(call, "They said yes.")
    assert session.messages[-1]["tool_call_id"] == call.call_id


def test_the_system_prompt_is_the_first_thing_and_is_sent_every_time():
    b = brain([whole()], stream=False)
    session = Session(b, system="be brief")
    session.tell("x")
    session.decide(TOOLS)
    assert b._client.requests[0]["messages"][0] == {"role": "system", "content": "be brief"}


def test_usage_reports_the_worst_case_not_only_the_mean():
    """3.8 s on average is tolerable while an arm is moving; 15.5 s at worst is not. A mean
    on its own hides exactly the number that decides whether this can be shown live."""
    u = Usage()
    for seconds in (1.0, 2.0, 15.5):
        u.add(Call("choose", {}, latency_s=seconds))
    assert u.worst_s == 15.5
    assert round(u.mean_s, 2) == 6.17


def test_the_cost_line_uses_token_factory_prices():
    u = Usage()
    u.add(Call("choose", {}, in_tokens=1_000_000, out_tokens=1_000_000))
    assert round(u.cost_usd, 4) == 0.30
