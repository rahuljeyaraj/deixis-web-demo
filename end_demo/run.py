"""Run the end demo's brain and print the scoreboard.

    python -m end_demo.run                        # 30 trials on the new shelf
    python -m end_demo.run --brain stub           # spends no calls (test double)
    python -m end_demo.run --trials 200           # the overnight number
    python -m end_demo.run --shelf rounds         # the harness's 10 things: port check
    python -m end_demo.run --prompt port          # the harness's prompt: port check
    python -m end_demo.run --ask-choice --show-me # what E2 measures against the pair
    python -m end_demo.run --person model         # a second Nemotron session as the person
    python -m end_demo.run --unreliable 0.3       # a third of their yes/no answers wrong
    python -m end_demo.run --job 3                # three runs of the whole job, with arms

Every number is printed under the settings that produced it. A selection score means
nothing without the pointing error, the word budgets and **which tools were on the table** -
the harness measured 98% with exactly two, and adding tools changes behaviour, so the
toolset belongs in the header beside the score and not in a footnote.

To check the port rather than trust it: `--shelf rounds --prompt port` runs the new session
loop on the old scene with the old prompt and the old two tools, which is the only
comparison in which the three-rounds numbers mean anything.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

from end_demo.brain import MODEL, SYSTEM, SYSTEM_PORT, Decider, Nemotron
from end_demo.job import MAX_JOB_QUESTIONS, job
from end_demo.person import ModelPerson, Person
from end_demo.point import CONE_DEG, RATIO, SIGMA_DEG
from end_demo.scene import DIST, STEP_X, STEP_Z, Thing, kind_counts, wanted
from end_demo.scene import shelf as end_shelf
from end_demo.score import notes, scoreboard, timing
from end_demo.session import (
    MAX_QUESTIONS,
    POINT_AND_ASK_BUDGET,
    WORDS_ONLY_BUDGET,
    Attempt,
    trial,
)
from end_demo.stub import StubDecider
from end_demo.tools import Toolset
from end_demo.world import World

#: Published baselines, for the header.
PUBLISHED_POINTING = "51-69%"
PUBLISHED_ERROR_DEG = 25


def shelf_for(name: str) -> list[Thing]:
    """The end demo's twelve things, or the harness's ten, for the port check."""
    if name == "rounds":
        from rounds.scene import shelf as rounds_shelf

        # The harness's ten things, as end-demo Things, so the port can be measured against
        # the numbers in three-rounds.md rather than assumed to match them. Its columns run
        # -2..2; these run 0..5, and `pos` centres both, so the geometry is the same.
        out = []
        for t in rounds_shelf():
            out.append(
                Thing(
                    id=t.id,
                    kind=t.kind,
                    seen=t.seen,
                    says=t.says,
                    col=t.col + 2,
                    row=t.row,
                )
            )
        return out
    return end_shelf()


def args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="the end demo's brain, headless")
    p.add_argument("--trials", type=int, default=30, help="how many targets (default 30)")
    p.add_argument("--brain", choices=("nemotron", "stub"), default="nemotron")
    p.add_argument("--shelf", choices=("end", "rounds"), default="end")
    p.add_argument("--prompt", choices=("end", "port"), default="end")
    p.add_argument("--person", choices=("scripted", "model"), default="scripted")
    p.add_argument(
        "--answers",
        choices=("word", "point", "both"),
        default="both",
        help="how they answer a question: yes/no, a second point, or both (default both)",
    )
    p.add_argument(
        "--unreliable",
        type=float,
        default=0.0,
        help="fraction of their yes/no answers that come out wrong (the word only)",
    )
    p.add_argument(
        "--silence", type=float, default=0.0, help="fraction of questions they do not answer"
    )
    p.add_argument(
        "--terse",
        action="store_true",
        help="the person says as little as possible (the harness's behaviour; port check)",
    )
    p.add_argument("--ask-choice", action="store_true", help="offer ask_choice (plan E2)")
    p.add_argument("--show-me", action="store_true", help="offer ask_show_me (plan E2)")
    p.add_argument("--cannot", action="store_true", help="offer cannot")
    p.add_argument(
        "--point-error",
        type=float,
        default=SIGMA_DEG,
        help=f"pointing error, degrees per axis (default {SIGMA_DEG})",
    )
    p.add_argument("--words-2", type=int, default=WORDS_ONLY_BUDGET, help="round 2 budget")
    p.add_argument("--words-3", type=int, default=POINT_AND_ASK_BUDGET, help="round 3 budget")
    p.add_argument("--max-questions", type=int, default=MAX_QUESTIONS)
    p.add_argument("--no-thinking", action="store_true", help="turn reasoning mode off")
    p.add_argument("--no-stream", action="store_true", help="do not stream the reasoning")
    p.add_argument("--job", type=int, default=0, help="runs of the whole job, instead of rounds")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--show", type=int, default=2, help="transcripts to print in full")
    p.add_argument("--json", type=Path, default=None, help="write every attempt here")
    return p.parse_args(argv)


def make_brain(a: argparse.Namespace) -> Decider:
    if a.brain == "stub":
        return StubDecider()
    return Nemotron(thinking=not a.no_thinking, stream=not a.no_stream)


def person_factory(a: argparse.Namespace, model: Decider):
    """How a person is made for one round. The model person gets its own session."""

    def make(target, things, rng, sigma):
        common = dict(
            style=a.answers,
            unreliable=a.unreliable,
            silence=a.silence,
            terse=a.terse,
        )
        if a.person == "model":
            return ModelPerson(target, things, rng, sigma, model=model, **common)
        return Person(target, things, rng, sigma, **common)

    return make


def header(a: argparse.Namespace, brain: Decider, things: list[Thing], kit: Toolset) -> list[str]:
    same = ", ".join(f"{n} {k}s" for k, n in kind_counts(things).items() if n > 1)
    reasoning = (
        "" if a.brain == "stub" else (", reasoning off" if a.no_thinking else ", reasoning on")
    )
    streaming = "" if a.brain == "stub" else (", whole" if a.no_stream else ", streamed")
    prompt_name = "the harness's, for the port check" if a.prompt == "port" else "the end demo's"
    lines = [
        "THE END DEMO - which thing on the shelf does the person mean?",
        "",
        f"  brain          {brain.name}{reasoning}{streaming}",
        f"  prompt         {prompt_name}",
        f"  shelf          {len(things)} things ({a.shelf}), {DIST} m away, "
        f"{STEP_X * 100:.0f} cm apart across and {STEP_Z * 100:.0f} cm up",
        f"  identical      {same} - no word tells them apart",
        f"  point error    {a.point_error:.1f} deg per axis "
        f"(published mean, harder setup: {PUBLISHED_ERROR_DEG} deg)",
        f"  cone / ratio   {CONE_DEG:.0f} deg, ask below {RATIO:.0f}:1 (the core's own rule)",
        f"  tools          {kit.label}",
        f"  word budgets   round 2: {a.words_2}, round 3: {a.words_3} (yes/no is free)",
        f"  questions      up to {a.max_questions} per attempt in round 3",
        f"  the person     {a.person}, answers by {a.answers}"
        + (", says as little as possible" if a.terse else ""),
    ]
    if a.unreliable:
        lines.append(
            f"  unreliable     {a.unreliable:.0%} of their yes/no answers come out wrong "
            "(the word only - their pointing stays honest)"
        )
    if a.silence:
        lines.append(f"  silent         {a.silence:.0%} of questions get no answer at all")
    lines += [f"  trials         {a.trials}, seed {a.seed}", ""]
    return lines


def run(a: argparse.Namespace) -> tuple[list[Attempt], Decider]:
    things = shelf_for(a.shelf)
    brain = make_brain(a)
    kit = Toolset.selection(choice=a.ask_choice, show_me=a.show_me, cannot=a.cannot)
    system = SYSTEM_PORT if a.prompt == "port" else SYSTEM
    print("\n".join(header(a, brain, things, kit)), flush=True)
    make = person_factory(a, brain)
    order = wanted(things)
    random.Random(a.seed).shuffle(order)
    attempts: list[Attempt] = []
    shown = 0
    for i in range(a.trials):
        target = order[i % len(order)]
        got = trial(
            target,
            things,
            brain,
            seed=a.seed * 1000 + i,
            make_person=make,
            sigma_deg=a.point_error,
            budget2=a.words_2,
            budget3=a.words_3,
            max_questions=a.max_questions,
            toolset=kit,
            system=system,
        )
        attempts += got
        if shown < a.show:
            shown += 1
            print(f"--- trial {i + 1}: the person wants {target.id} ({target.seen}) ---")
            for att in got:
                print(f"\n  [{att.round}] {'RIGHT' if att.ok else 'WRONG'}")
                for line in att.log:
                    print(f"    {line}")
            print()
        elif not a.json:
            print(f"\rtrial {i + 1}/{a.trials}", end="", file=sys.stderr)
    if a.trials > a.show:
        print(file=sys.stderr)
    return attempts, brain


def run_job(a: argparse.Namespace) -> tuple[list[Attempt], Decider]:
    """The whole job, with the arms. The showpiece - not part of the measurement."""
    things = shelf_for(a.shelf)
    brain = make_brain(a)
    system = SYSTEM_PORT if a.prompt == "port" else SYSTEM
    print("\n".join(header(a, brain, things, Toolset.job())), flush=True)
    print("  NOT a measurement: this is the loop, not the score.\n")
    make = person_factory(a, brain)
    order = wanted(things)
    random.Random(a.seed).shuffle(order)
    out: list[Attempt] = []
    for i in range(a.job):
        target = order[i % len(order)]
        world = World(things=list(things))
        person = make(target, things, random.Random(a.seed * 1000 + i), a.point_error)
        print(f"--- job {i + 1}: the person wants {target.id} ({target.seen}) ---")
        att = job(
            target,
            world,
            brain,
            person,
            budget=a.words_3,
            max_questions=MAX_JOB_QUESTIONS,
            system=system,
        )
        out.append(att)
        for line in att.log:
            print(f"    {line}")
        print(f"    => {'handed over' if att.delivered else 'NOT handed over'}\n")
    done = sum(a_.delivered for a_ in out)
    print(f"  handed over: {done}/{len(out)}")
    return out, brain


def cost(brain: Decider) -> list[str]:
    u = brain.usage
    if not (u.prompt or u.completion):
        return [f"  model: {u.calls} calls"]
    out = [
        f"  model          {u.calls} calls, {u.prompt + u.completion} tokens, "
        f"${u.cost_usd:.4f} ({MODEL})"
    ]
    if u.out_of_room:
        out.append(
            f"  out of room    {u.out_of_room} calls spent their whole reasoning budget and "
            "were retried with reasoning off (still Nemotron, never a word list)"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    a = args(argv)
    attempts, brain = run_job(a) if a.job else run(a)
    if not a.job:
        print("\n".join(scoreboard(attempts)))
        print()
        print(f"  published, pointing alone: {PUBLISHED_POINTING}")
        for line in notes(attempts):
            print(line)
    print()
    for line in timing(attempts):
        print(line)
    for line in cost(brain):
        print(line)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(
            json.dumps(
                {
                    "settings": vars(a) | {"json": str(a.json)},
                    "attempts": [asdict(x) for x in attempts],
                },
                indent=1,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\n  every attempt written to {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
