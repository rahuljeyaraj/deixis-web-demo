"""Run the three rounds and print the scoreboard.

    python -m rounds.run                     # 30 trials, Nemotron on Token Factory
    python -m rounds.run --brain stub        # no calls spent (word matcher, test double)
    python -m rounds.run --trials 200        # the overnight number
    python -m rounds.run --show 3            # three transcripts in full

Every number is printed under the settings that produced it, because a selection score
means nothing without the pointing error and the word budgets it was measured at.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

from rounds.brain import MODEL, Brain, Nemotron, StubBrain
from rounds.pointer import CONE_DEG, RATIO, SIGMA_DEG
from rounds.rounds import (
    MAX_ASKS,
    POINT_AND_ASK_BUDGET,
    WORDS_ONLY_BUDGET,
    Attempt,
    tally,
    trial,
)
from rounds.scene import DIST, STEP_X, STEP_Z, kind_counts, shelf

#: Published baselines, for the header. Pointing-only selection, and the mean angular error
#: of the head-hand line measured with a time-of-flight camera at up to 4 m.
PUBLISHED_POINTING = "51-69%"
PUBLISHED_ERROR_DEG = 25


def args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="the three rounds")
    p.add_argument("--trials", type=int, default=30, help="how many targets (default 30)")
    p.add_argument("--brain", choices=("nemotron", "stub"), default="nemotron")
    p.add_argument(
        "--point-error",
        type=float,
        default=SIGMA_DEG,
        help=f"pointing error, degrees per axis (default {SIGMA_DEG})",
    )
    p.add_argument("--words-2", type=int, default=WORDS_ONLY_BUDGET, help="round 2 budget")
    p.add_argument("--words-3", type=int, default=POINT_AND_ASK_BUDGET, help="round 3 budget")
    p.add_argument("--max-asks", type=int, default=MAX_ASKS)
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="turn Nemotron's reasoning mode off (faster, no trace to show)",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--show", type=int, default=2, help="transcripts to print in full")
    p.add_argument("--json", type=Path, default=None, help="write every attempt here")
    return p.parse_args(argv)


def header(a: argparse.Namespace, brain: Brain, things: list) -> list[str]:
    kinds = kind_counts(things)
    same = ", ".join(f"{n} {k}s" for k, n in kinds.items() if n > 1)
    return [
        "THE THREE ROUNDS - which thing on the shelf does the person mean?",
        "",
        f"  brain          {brain.name}"
        + ("" if a.brain == "stub" else f", reasoning {'off' if a.no_thinking else 'on'}"),
        f"  shelf          {len(things)} things, {DIST} m away, "
        f"{STEP_X * 100:.0f} cm apart across and {STEP_Z * 100:.0f} cm up",
        f"  identical      {same} - no word tells them apart",
        f"  point error    {a.point_error:.1f} deg per axis "
        f"(published mean, harder setup: {PUBLISHED_ERROR_DEG} deg)",
        f"  cone / ratio   {CONE_DEG:.0f} deg, ask below {RATIO:.0f}:1 (the core's own rule)",
        f"  word budgets   round 2: {a.words_2}, round 3: {a.words_3} (yes/no is free)",
        f"  questions       up to {a.max_asks} per attempt in round 3",
        f"  trials         {a.trials}, seed {a.seed}",
        "",
    ]


def run(a: argparse.Namespace) -> tuple[list[Attempt], Brain]:
    things = shelf()
    brain: Brain = StubBrain() if a.brain == "stub" else Nemotron(thinking=not a.no_thinking)
    print("\n".join(header(a, brain, things)), flush=True)
    order = list(things)
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
            sigma_deg=a.point_error,
            budget2=a.words_2,
            budget3=a.words_3,
            max_asks=a.max_asks,
        )
        attempts += got
        if shown < a.show:
            shown += 1
            print(f"--- trial {i + 1}: the person wants {target.id} ({target.seen}) ---")
            for att in got:
                mark = "RIGHT" if att.ok else "WRONG"
                print(f"\n  [{att.round}] {mark}")
                for line in att.log:
                    print(f"    {line}")
            print()
        elif not a.json:
            print(f"\rtrial {i + 1}/{a.trials}", end="", file=sys.stderr)
    if a.trials > a.show:
        print(file=sys.stderr)
    return attempts, brain


def _table(attempts: list[Attempt], width: int) -> list[str]:
    return [
        f"  {r.round:<{width}}  {r.correct:>4}/{r.scored:<4}  {r.pct:>5.0f}%  "
        f"{r.mean_words:>6.1f}  {r.mean_asks:>5.1f}"
        for r in tally(attempts)
    ]


def scoreboard(attempts: list[Attempt], brain: Brain) -> str:
    rows = tally(attempts)
    width = max(len(r.round) for r in rows)
    head = f"  {'round':<{width}}  {'right':>9}  {'score':>6}  {'words':>6}  {'asks':>5}"
    lines = ["", "SCOREBOARD", "", head]
    lines += _table(attempts, width)
    # The split that carries the argument: words can only work on a thing that has a name
    # of its own. Blending the two hides exactly what this project is about.
    unique = [a for a in attempts if a.unique]
    same = [a for a in attempts if not a.unique]
    if unique and same:
        lines += ["", "  when the thing is one of a kind (it has a name):", head]
        lines += _table(unique, width)
        lines += ["", "  when it is one of several identical things (no name exists):", head]
        lines += _table(same, width)
    lines.append("")
    lines.append(f"  published, pointing alone: {PUBLISHED_POINTING}")
    bad = [r for r in rows if r.invalid or r.failed]
    for r in bad:
        note = []
        if r.invalid:
            note.append(f"{r.invalid} named something not on the shelf")
        if r.failed:
            note.append(f"{r.failed} could not reach the model (not scored)")
        lines.append(f"  {r.round}: " + "; ".join(note))
    u = brain.usage
    if u.prompt or u.completion:
        lines.append(
            f"  model: {u.calls} calls, {u.prompt + u.completion} tokens, "
            f"${u.cost_usd:.4f} ({MODEL})"
        )
    else:
        lines.append(f"  model: {u.calls} calls")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    a = args(argv)
    attempts, brain = run(a)
    print(scoreboard(attempts, brain))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps([asdict(x) for x in attempts], indent=1), encoding="utf-8")
        print(f"\n  every attempt written to {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
