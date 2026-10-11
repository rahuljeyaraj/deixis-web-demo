# Limitations

What this web demo doesn't do, or does differently from a real robot. Each item says why.

## L1. The grip is helped

The jaws really close on the block, and in 50 random moves the block was between both jaws
every time (0 grip assists). But once the jaws close, a **weld** (a fixed joint in the
physics) holds the block, and the held block stops touching the gripper. A real SO-101 holds a
block by squeezing it; in the simulator a squeezed block can slowly slide or be pushed when
it's let go, which reads as a bug on camera.

If the block isn't between the jaws as expected (over 8 mm off, or tipped over 15°), the arm
lets go and says so instead of carrying it.

Technical note: the WebAssembly bindings (`@mujoco/mujoco` 3.14.0) can't switch one weld on
and off (`eq_active` can't be written), so the scene has one weld, pointed at the block being
picked, and it is switched with the global "equality" disable flag.

## L2. The arm reaches a half ring only

The SO-101 is small (upper arm about 113 mm, lower arm about 135 mm). It can pick a block up
and put it down from **140 to 280 mm** around its base, measured with the real planner
(`web/tests/probes/reach2.probe.test.ts`; closer than 140 mm the gripper can touch the arm's own
shoulder, `inner.probe.test.ts`). It stacks all three blocks anywhere on that ring. The lighter half ring on the table shows this. The
core knows a rectangle (500 × 250 mm); the half ring is the robot's own knowledge, checked in
the browser before anything moves, as a real robot would. A point outside it gets "I can't
reach there. Please point somewhere closer."

## L3. Scripted moves, not a learned policy

The pick and place is scripted: inverse kinematics (damped least squares) and smooth joint
moves, straight down and straight up. No learned policy (such as ACT or a vision-language-action
model) runs. The demo is about telling a robot **where**; how the robot moves is its own
business.

## L4. Accuracy

In the headless test (50 random moves), a block lands a median 0.57 mm and at most 1.43 mm from
where it was sent, flat. Stacking ("put that on that") puts the top block within 3 mm of the one
below. These are simulator numbers; a real arm's error would be larger.

## L5. A day's share of the AI

The hosted demo is for the hackathon judges: its address is given only to them. Nemotron runs
on our Nebius key, with a safety limit in case the address leaks: **2,000 commands a day** in
all (about $1), and 500 a day from one network, so one person can't use up everyone's day.
No judge should reach either. Past them **nothing answers**: the robot says so and the page
offers the box for a visitor's own Nebius key, which skips the daily caps. There is no
stand-in. A simple word matcher used to answer in Nemotron's place; it was taken out on
11 Oct 2026, because a demo whose claim is "the model decides" must not quietly be answered
by a word list. One command can be sent every 1.5 s; the page waits by itself.

The counts live in the server's memory: they start again at midnight UTC and when the server
restarts. A network is a scrambled form of the visitor's address, never logged.

Within those limits Nemotron answers **every** command, "yes", "stop" and the answer to the
robot's own question included: nothing is decided by matching words behind the model's back,
and nothing else on the server understands a person at all. The one thing the model is never
asked to judge is where a finger pointed: when an answer is only pointed at ("this one"), the model
says so and the core works out which block, because the words genuinely do not carry it.

## L6. A plan is four steps at most, and one re-plan

One sentence can be several steps ("stack them all up", "swap the red and the blue"): Nemotron
answers with up to **four** tool calls from one model call, and the page runs them in order.
A longer plan is refused out loud rather than attempted badly — the literature is clear that
reliability falls away and hallucinations pile up over longer plans. "Stack all six into one
tower" gets "That's more steps than I can plan at once. Ask me for up to four."

When a step turns out to be impossible, the arm's own reason goes back to Nemotron, which
plans the rest another way. That happens **once** per command, so a model that keeps failing
cannot run away; after that the robot says the reason and stops there.

A single move that cannot be done is not re-planned: "I can't reach there. Please move the dot
closer." is a better answer than anything a model that may not write coordinates could offer.

## L7. The whole plan is previewed from a guess; every step is measured again

The plan is shown before anything moves, with a ghost block at each block's final place. The
places past the first step are worked out against the table as the steps before them *will*
have left it — a prediction by the page, not a measurement.

When the arm reaches a step, the core measures that step again against the real table, and
that is the answer the arm acts on. So the preview can be a little out (a block that ends up
a millimetre from where the ghost was), but nothing is ever acted on without being measured
first. One "yes" covers the whole plan; one "yes" per step would ruin the demo.

## L8. The painted squares hold one block each

The three coloured squares are 70 mm of paint on the mat: flat, with no physics, nothing to
grip, and a finger pointing at one goes through to the table under it. "In the red square" is
the middle of that square, so a square holds **one** block in the middle; a second block into
the same square goes beside the first and half on the paint. "Put all three in the blue
square" is not something this table can do.

A block and its square both answer to "the red one". The core prefers an exact match, and the
page reports "red one" as one of the block's own names, so "the red one" is the block, "the
red square" is the paint, and nothing has to be asked.

## L9. "The spot it left" lasts until something is put there

A swap needs one block to go where another one was, and no reference the core has could say
that. So the page reports the place the last step emptied as a thing called "the spot it
left", and Nemotron can name it. There is only ever one, and it is gone as soon as a block
stands there. This is robot-side knowledge reported as a thing, which is what the core's L6
expects — not a change to the core.

More items arrive as the finger, speech and the core are added (plan section 9).
