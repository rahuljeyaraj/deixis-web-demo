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
and put it down from **130 to 280 mm** around its base, measured with the real planner
(`web/tests/probes/reach2.probe.test.ts`). The lighter half ring on the table shows this. The
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

More items arrive as the finger, speech and the core are added (plan section 9).
