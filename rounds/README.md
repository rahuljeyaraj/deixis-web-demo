# The three rounds

A headless proof of concept: **can a person pick one thing off a shelf when they have
almost no words for it?**

No arm, no 3D, no page. Just the three rounds a judge will try at the booth, run a hundred
times over so the numbers mean something.

```
python -m rounds.run                  # 30 trials, Nemotron on Nebius Token Factory
python -m rounds.run --brain stub     # spends no calls (word matcher, a test double)
python -m rounds.run --trials 200     # the overnight number
python -m rounds.run --show 3         # three transcripts in full
```

## Why this, and why first

The demo today is three cubes on a table. Cubes have names and are never ambiguous, so the
one question this project exists to answer — *which one do you mean?* — never comes up.

This replaces the cubes with a shelf across the room holding four identical bottles, two
identical mugs and a bracket nobody has a word for. Now:

- **Words alone cannot do it.** No word separates one bottle from the next.
- **Pointing alone cannot do it.** A point is a cone, not a line. People miss by **25
  degrees on average**, and systems that select from pointing alone get **51–69%**.

Neither channel is enough. Putting them together, and letting the robot **ask**, is the
claim. This harness measures it.

## The three rounds

| | pointing | words | may ask |
|---|---|---|---|
| **1. point only** | yes | — | — |
| **2. words only** | — | 4 words | — |
| **3. point + ask** | yes | 2 words | yes |

Round 1 is how published pointing systems work. Round 2 is how every voice assistant works.
Round 3 is ours.

**The design is paired.** Inside one trial all three rounds want the same thing, and rounds
1 and 3 are handed *the same point, with the same wobble*, drawn once. So round 3 cannot win
on a luckier aim — the only thing it has that round 1 does not is two words and the right to
ask. The difference between those two numbers is attributable.

## Asking by pointing

The part nobody builds. When the model is unsure it calls `ask_about(id)`: the gripper
hovers over one thing and the robot asks *this one?* The person answers yes or no.

That matters because the person we are building for cannot answer "did you mean the blue mug
or the grey one?" — that demands words back. A nod they have. So **every question the robot
asks must be answerable by pointing or a nod**, and the word count never rises.

## Honesty rules this harness keeps

- **Every number is printed with the settings that produced it.** A score without the
  pointing error and the word budgets is meaningless.
- **`--point-error` is the one difficulty dial**, and it is in the header of every run. The
  default (3° per axis) puts round 1 inside the published 51–69%, which is the check that the
  rig is fair rather than tuned.
- **Scored strictly.** A model that names something not on the shelf scores zero and is
  counted as `invalid`, never quietly retried. A model that cannot be reached is recorded and
  **not scored** — the robot says it cannot, it never falls back on a word list.
- **Words are counted as the person spoke them.** Yes and no are free, because a nod is free.
- **The scoreboard splits one-of-a-kind things from identical ones**, because words only
  ever work on the first kind, and blending the two hides the whole point.

## The pieces

| file | what it is |
|---|---|
| `scene.py` | the shelf: ten things, six of them impossible to tell apart in words |
| `pointer.py` | a point is a cone: aim, wobble, and who is inside it |
| `person.py` | someone who knows what they want and has two words for it |
| `brain.py` | Nemotron 3.5 Lightning, and a stub word matcher as a test double |
| `rounds.py` | the three rounds, and the scoring |
| `run.py` | the command line and the scoreboard |

Tests: `pytest tests/rounds` — 23 of them, no network, no calls spent.

## What this is not

- Not the demo. The demo is this with the arm, the shelf and the cone drawn on screen.
- Not a medical claim. Nobody with aphasia has tried it.
- Not a test of the arm. Selection only: *which thing*, not whether the gripper can lift it.
