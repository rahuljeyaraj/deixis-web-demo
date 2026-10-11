# The end demo: *hand me that one*

The brain of the finished web demo, headless. No arm, no 3D, no page — just the loop
between a person who cannot say what things are called and a robot that has to work out
which thing they mean, run a few hundred times over so the numbers mean something.

```
python -m end_demo.run                          # 30 trials, Nemotron on Token Factory
python -m end_demo.run --brain stub             # spends no calls (a test double)
python -m end_demo.run --trials 200             # the overnight number
python -m end_demo.run --ask-choice --show-me   # with the two SCA questions added
python -m end_demo.run --unreliable 0.3         # a third of their yes/no answers wrong
python -m end_demo.run --person model           # a second Nemotron session as the person
python -m end_demo.run --job 3                  # the whole job, with both arms
python -m end_demo.run --shelf rounds --prompt port --answers word --terse   # port check
```

Tests: `pytest tests/end_demo` — no network, no calls spent.

## Who it is for

Anomia — losing the names of things — is present in **everyone who has aphasia**, and about
**one in three stroke survivors** has aphasia. The textbook description: they can pick up a
pencil and write with it, but cannot say the word "pencil". Their thinking is untouched.

What survives is pointing, yes/no, and understanding. And the words that survive are *that,
this, here, there, it, one* — exactly what the deixis core was built for.

**Nobody with aphasia has tried this.** The claim is about how few words the interface
needs, which is measurable without them. Nothing here is a clinical claim.

## The shelf

Twelve things, two rows of six, 2.5 m away. Four identical pill bottles, two identical
mugs, a bracket nobody has a word for, and six things that do have names.

The scene defeats naming **for everyone**. There is no word that separates bottle 2 from
bottle 3, for a fluent judge as completely as for the person we built this for. That is the
honest way to show the problem: put the barrier in the world, not in the person. The demo
never asks anyone to pretend to have a disability — the research on disability simulation is
clear that it increases stigma rather than empathy.

The six one-of-a-kind things are there deliberately, because **words-only must be able to
score 100% on something** or the comparison is a stitch-up.

## Two arms, and why they are not scenery

The person sits at the right-hand end of the table.

- **right_arm** stands beside them. It reaches the right of the shelf *and* can put a thing
  in their hand.
- **left_arm** stands at the far end. It reaches further left, but **cannot reach the
  person**.

So a thing on the left of the shelf has to be picked up by the left arm and **passed
across** before it can be handed over. That is why the person needs to be able to say
*you*, and it is the one thing a single arm cannot show.

## The pieces

| file | what it is |
|---|---|
| `scene.py` | the shelf and the two arms; `reshuffle` for the booth |
| `point.py` | the cone, re-exported from `rounds/pointer.py` so there is only ever one |
| `world.py` | what the robot sees, what it did, and why an action was refused |
| `say.py` | what it says, writes and gestures — all three, every time |
| `confirm.py` | how an answer is read: the hand first, and never act on a contradiction |
| `tools.py` | the tools, and which of them are on the table when |
| `brain.py` | Nemotron 3.5 Lightning as a session: one transcript, streamed |
| `person.py` | the simulated person, scripted or a second Nemotron session |
| `session.py` | the three rounds, and the loop that enforces R4 and R5 |
| `job.py` | the whole job: work it out, pick it up, pass it across, hand it over |
| `score.py` | the scoreboard, with N on every line and the split always shown |
| `stub.py` | test doubles. **Not fallbacks** — see below |
| `run.py` | the command line |

## The rules this code keeps

**The robot asks by pointing.** Every question it can ask is answerable with a point, a
nod, or nothing: *this one?*, *this one or this one?*, *show me*. It never asks "did you
mean the mug or the bottle?", because that makes the person produce a word they do not
have. The manner comes from **Supported Conversation for Adults with Aphasia**, which is an
evidence-based method and reads almost like a specification.

**The hand outranks the word.** A point at the thing the gripper is over is a yes. A point
at something else is not a refusal to re-ask about — it is the new answer.

**A contradiction is never acted on.** In severe aphasia a yes or no can be wrong or
inconsistent; there is a clinical instrument whose only job is sorting reliable from
unreliable responders. So when the word and the hand disagree, the loop picks neither, says
so, and asks again. Every one is counted and reported.

**No word matcher, ever.** Nemotron answers or the robot says it cannot. `stub.py` holds
test doubles so the loop can be tested and exercised without spending calls; nothing in
`end_demo` can reach them, and a run that uses one says so in its header and means nothing
as a score.

**The model makes every decision, including the last one.** When the questions run out, the
model chooses with only `choose` on the table. The loop never breaks that tie itself: it
used to, and part of every score was the harness's judgement rather than the model's.

**No coordinates.** The core's standing invariant is that the language model never writes a
number for a place. Every place in every tool is a reference.

**The prompt describes only the tools that exist.** Built from the toolset by
`brain.system_for`. A prompt that advertises a tool the model has not been given produces
calls to a tool that does not exist — measured on the first live run.

**Every number carries its N, and the split is always shown.** Words score 100% on a thing
that has a name and about a third on one of four identical bottles, so a blended figure
hides the whole argument.

## How Nemotron must be called

Three things, all measured on Token Factory on 11 Oct 2026, all of which fail silently:

1. **`tool_choice="auto"`, never `"required"`.** With `"required"` this model never emits a
   stop: `finish_reason` is `"length"` every time and it repeats one call 7–23 times.
2. **Reasoning needs 4000 `max_tokens`.** At 2000 it overran 3 of 40 calls, and an overrun
   produces **no tool call at all**. An overrun is retried once with reasoning off — still
   Nemotron — and recorded.
3. **Keep `reasoning_content`.** It is what the page shows. Traces average 3,000 characters.

**Reasoning is on for every decision** (rahul, 11 Oct). It halves how often the person is
asked — 0.1 questions instead of 1.3 when the thing has a name — and every needless
question is a nod somebody had to give. It costs seconds, so the reasoning is **streamed**
and the wait reads as work rather than a hang.

## What this is not

- Not the demo. The demo is this with the shelf, the arms and the cone drawn on screen.
- Not a measurement of the gripper. Selection only: *which thing*, not whether it can lift it.
- Not a medical claim of any kind.

Plans, results and session handoffs live in `deixis-planning`, never here.
