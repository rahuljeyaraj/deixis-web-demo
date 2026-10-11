"""The three rounds: that each one is handicapped the way we say it is, and scored strictly.

These run against the stub brain (a word matcher, a test double - see `rounds/brain.py`),
so they spend no calls and never touch the network.
"""

from __future__ import annotations

import random

from rounds.brain import Decision, StubBrain
from rounds.person import Person
from rounds.pointer import point_at
from rounds.rounds import (
    Attempt,
    point_and_ask,
    point_only,
    tally,
    trial,
    words_only,
)
from rounds.run import main, scoreboard
from rounds.scene import by_id, kind_counts, shelf


def _target(name: str):
    return by_id(shelf())[name]


def test_the_shelf_has_things_no_word_can_separate() -> None:
    counts = kind_counts(shelf())
    assert counts["bottle"] == 4
    assert counts["mug"] == 2
    # And the identical ones really are identical to the robot's eyes.
    bottles = [t for t in shelf() if t.kind == "bottle"]
    assert len({t.seen for t in bottles}) == 1
    assert len({t.says for t in bottles}) == 1


def test_round_one_spends_no_words_and_no_questions() -> None:
    things = shelf()
    target = _target("phone")
    cone = point_at(target, things, random.Random(1))
    a = point_only(target, cone)
    assert a.words == 0 and a.asks == 0


def test_round_two_cannot_point() -> None:
    """Round 2's attempt records no cone at all: the pointing channel is simply absent."""
    things = shelf()
    target = _target("bottle_1")
    person = Person(target, things, random.Random(1))
    a = words_only(target, things, person, StubBrain())
    assert a.cone == []
    assert person.points == 0


def test_round_two_is_a_guess_among_identical_things() -> None:
    """Four bottles, one word, no pointing: whatever it picks, it is a bottle, and it is
    right one time in four. This is the finding, not a rigged scene."""
    things = shelf()
    picked = set()
    for name in ("bottle_1", "bottle_2", "bottle_3", "bottle_4"):
        target = _target(name)
        a = words_only(target, things, Person(target, things, random.Random(1)), StubBrain())
        picked.add(a.picked)
    assert picked <= {"bottle_1", "bottle_2", "bottle_3", "bottle_4"}
    assert len(picked) == 1  # nothing in the words moved it


def test_round_two_works_when_the_thing_has_a_name() -> None:
    for name in ("scissors", "phone", "remote", "bracket"):
        things = shelf()
        target = _target(name)
        a = words_only(target, things, Person(target, things, random.Random(1)), StubBrain())
        assert a.ok, name


def test_round_three_sees_the_same_point_as_round_one() -> None:
    things = shelf()
    target = _target("mug_1")
    got = trial(target, things, StubBrain(), seed=5)
    one, _, three = got
    assert one.cone == three.cone


def test_a_yes_settles_it() -> None:
    things = shelf()
    target = _target("mug_2")
    cone = point_at(target, things, random.Random(5))
    person = Person(target, things, random.Random(5))
    a = point_and_ask(target, things, person, StubBrain(), cone)
    assert a.ok
    assert a.asks >= 1
    assert person.answers == a.asks


def test_questions_are_capped_and_then_it_must_commit() -> None:
    things = shelf()
    target = _target("bottle_4")
    cone = point_at(target, things, random.Random(2))
    person = Person(target, things, random.Random(2))
    a = point_and_ask(target, things, person, StubBrain(), cone, max_asks=1)
    assert a.asks <= 1
    assert a.picked  # it committed to something rather than giving up


class _AlwaysAsks(StubBrain):
    """Never commits by itself, so every attempt runs the questions out."""

    def step(self, things, cone, said, asked):  # type: ignore[override]
        free = [c.id for c in cone.candidates if c.id not in {i for i, _ in asked}]
        return Decision("ask", free[0], "keep asking")


def test_the_final_commit_is_the_models_call_not_ours() -> None:
    """When the questions run out the brain decides, with only `choose` available. If the
    harness tie-broke here, part of every score would be the harness's."""

    class OwnIdea(_AlwaysAsks):
        def commit(self, things, cone, said, asked):  # type: ignore[override]
            return Decision("choose", "remote", "my own idea")

    things = shelf()
    target = _target("bottle_1")
    cone = point_at(target, things, random.Random(9))
    person = Person(target, things, random.Random(9))
    a = point_and_ask(target, things, person, OwnIdea(), cone, max_asks=0)
    assert a.asks == 0  # no questions allowed at all: straight to the commit
    assert a.picked == "remote"
    assert "my own idea" in " ".join(a.log)


def test_the_question_cap_is_exact() -> None:
    """Up to N questions must mean N, not N + 1 - the mean-asks column depends on it."""
    things = shelf()
    target = _target("bottle_1")
    for cap in (0, 1, 2, 3):
        cone = point_at(target, things, random.Random(9))
        person = Person(target, things, random.Random(9))
        a = point_and_ask(target, things, person, _AlwaysAsks(), cone, max_asks=cap)
        # It stops early if a question happens to land on the right thing.
        assert a.asks <= cap
        assert person.answers == a.asks


def test_an_unreachable_brain_at_the_commit_is_recorded() -> None:
    class DiesAtTheEnd(_AlwaysAsks):
        def commit(self, things, cone, said, asked):  # type: ignore[override]
            raise RuntimeError("token factory down")

    things = shelf()
    target = _target("bottle_1")
    cone = point_at(target, things, random.Random(9))
    person = Person(target, things, random.Random(9))
    a = point_and_ask(target, things, person, DiesAtTheEnd(), cone, max_asks=0)
    assert a.failed and not a.ok


def test_a_thing_that_is_not_on_the_shelf_scores_zero() -> None:
    class Hallucinates(StubBrain):
        def from_words(self, things, said):  # type: ignore[override]
            return Decision("choose", "teapot", "a thing it invented")

    things = shelf()
    target = _target("phone")
    a = words_only(target, things, Person(target, things, random.Random(1)), Hallucinates())
    assert a.invalid and not a.ok


def test_an_unreachable_brain_is_recorded_not_scored() -> None:
    class Dead(StubBrain):
        def from_words(self, things, said):  # type: ignore[override]
            raise RuntimeError("no key")

    things = shelf()
    target = _target("phone")
    a = words_only(target, things, Person(target, things, random.Random(1)), Dead())
    assert a.failed and not a.ok
    assert tally([a])[0].scored == 0


def test_words_are_counted_as_the_person_spoke_them() -> None:
    things = shelf()
    target = _target("scissors")
    person = Person(target, things, random.Random(1))
    said = person.say(2)
    assert said == "cutting"  # the shortest phrase that fits
    assert person.words_used == 1
    assert person.say(0) == ""


def test_the_word_budget_is_obeyed() -> None:
    things = shelf()
    for t in things:
        person = Person(t, things, random.Random(1))
        assert len(person.say(2).split()) <= 2
        assert len(person.say(1).split()) <= 1


def test_tally_keeps_rounds_apart_and_averages_per_attempt() -> None:
    rows = tally(
        [
            Attempt("a", "x", "x", words=2, asks=1),
            Attempt("a", "x", "y", words=0, asks=3),
            Attempt("b", "x", "x", words=1, asks=0),
        ]
    )
    a, b = rows
    assert (a.round, a.n, a.correct, a.pct) == ("a", 2, 1, 50.0)
    assert a.mean_words == 1.0 and a.mean_asks == 2.0
    assert (b.round, b.pct) == ("b", 100.0)


def test_the_scoreboard_splits_named_from_identical() -> None:
    things = shelf()
    attempts = []
    for i, t in enumerate(things):
        attempts += trial(t, things, StubBrain(), seed=100 + i)
    text = scoreboard(attempts, StubBrain())
    assert "one of a kind" in text
    assert "identical" in text
    assert "point + ask" in text


def test_the_whole_thing_runs(capsys) -> None:
    assert main(["--brain", "stub", "--trials", "4", "--show", "1"]) == 0
    out = capsys.readouterr().out
    assert "SCOREBOARD" in out
    assert "point only" in out and "words only" in out and "point + ask" in out


def test_every_model_call_is_kept_on_the_attempt() -> None:
    """The page can only show the thinking if the harness keeps it. Round 1 asks no model
    and must record none; the other two must record one per robot action."""
    things = shelf()
    target = _target("bottle_1")
    one, two, three = trial(target, things, StubBrain(), seed=5)
    assert one.thoughts == []  # pure geometry
    assert one.model_calls == 0
    assert len(two.thoughts) == 1
    assert two.thoughts[0]["at"] == "from words alone"
    assert len(three.thoughts) == three.asks + (0 if three.asks and three.ok else 1)
    for t in three.thoughts:
        assert t["action"] in ("ask", "choose")
        assert "thinking" in t and "latency_s" in t
