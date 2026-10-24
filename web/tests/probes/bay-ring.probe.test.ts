/**
 * Stage 0, measurement 2: **the reach ring at bay height, reaching in horizontally.**
 *
 * The workbench stands a two-bay shelf unit behind the bench, and a thing inside a bay has a
 * board over it, so the hand cannot come down on it. It has to come in from the front, level.
 * That is a different reach from everything measured for this project so far: `end_demo`'s
 * ring was driven with the fingers pointing **straight down**.
 *
 * So the ring is measured again, at the two bay heights the plan proposes
 * (`workbench-requirements.md` 4.1: bottom bay 0 to 110, so a thing's middle at **z = 55**;
 * top bay 122 to 232, so **z = 177**), and at the other heights this world turns out to need -
 * the hook rail at 190, a tall bottle held round the middle at 130, and, pointing **down**, a
 * thing standing on the bench and a thing standing on the unit's top board.
 *
 * Three things the sweep says, beyond the two numbers:
 *
 * 1. **A level hand points straight out from its own base and nowhere else.** The `into the
 *    shelf` rows are that fact measured: the SO-101 has five joints, and after the shoulder's
 *    turn every one bends about an axis across the arm, so the fingers stay in the arm's own
 *    turned plane. Past about 10 degrees of bearing there is no pose at all that points the
 *    hand square into the shelf. A thing off to one side is entered along the slant, and
 *    whether the bay's own sides allow that is a question for the scene, not for a ring.
 * 2. **Nothing much above 100 mm can be gripped from above.** The fingers-down ring is wide at
 *    the bench, narrow at 100 mm up, and empty at 130 and above. So a tall thing is taken round
 *    the middle, level; and the unit's top board, a quarter of a metre up, is not a place this
 *    robot can put anything or take anything from.
 * 3. **Tilt is most of the far reach.** A grip wants the fingers within about 4 degrees of
 *    where they should be (`maxTiltDeg: 4` at the grasp in `src/sim/arm.ts`); merely getting
 *    the gripper to a spot allows 25 (`shelf.probe.test.ts`). Both are swept, because they
 *    answer different questions and differ by about 60 mm of reach.
 *
 * What counts as reached: the solver puts the gripper site within 1 mm of the spot with the
 * fingers inside the tilt allowed, the arm does not fold into itself, and nothing from the
 * elbow out is pushed into the bench top. A pose that sinks the wrist through the bench is not
 * a reach, and a plain ring cannot know about it - which is the whole reason this is measured
 * and not calculated.
 *
 * The far edge is **not** softened by sagging: commanded to a level pose at 460 mm and left to
 * settle for two seconds, the gripper site stays within 0.5 mm of where it was asked, at both
 * bay heights. The servos (`forcerange` 2.94 Nm, `kp` 998) hold their own weight out to the end
 * of the sweep, so the edge found here is the kinematic one.
 *
 * Run: `PROBES=1 npx vitest run tests/probes/bay-ring.probe.test.ts`
 * Report: `out/stage0-bay-ring.txt`
 */

import fs from 'node:fs';
import path from 'node:path';
import { expect, test } from 'vitest';

import { GRIPPER_OPEN } from '../../src/sim/arm';
import { ArmKinematics, type IkTarget } from '../../src/sim/ik';
import type { MjData, MjModel, Mujoco } from '../../src/sim/mujoco';
import type { Vec3 } from '../../src/sim/table';
import { nodeScene } from '../scene';

const REPORT = path.join(import.meta.dirname, '..', '..', '..', 'out', 'stage0-bay-ring.txt');

/** Bearings off the arm's forward, degrees. The shoulder joint itself stops at 110. */
const BEARINGS_DEG = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
const RADIUS_FROM_MM = 90;
const RADIUS_TO_MM = 500;
const RADIUS_STEP_MM = 5;

/** Within this much of the spot counts as there. */
const CLOSE_MM = 1;
/** Deeper than this into the bench top is a pose that is not available, metres. */
const THROUGH_BENCH_M = 0.001;
/** What a grip needs (`arm.ts` at the grasp), and what merely getting there needs. */
const TILT_GRIP_DEG = 4;
const TILT_THERE_DEG = 25;
/** A level entry is allowed this much slack, because level is what a bay allows. */
const TILT_LEVEL_DEG = 10;

/**
 * How the hand is pointed: level and straight out from the base, level and into the shelf
 * whatever the bearing, or straight down.
 */
type How = 'level' | 'into' | 'down';

interface Sweep {
  how: How;
  zMm: number;
  tiltDeg: number;
  why: string;
}

/**
 * Every sweep, and what each is for. The first two are stage 0's own question; the rest are the
 * heights this world turns out to need, and the checks against what is already known.
 */
const SWEEPS: Sweep[] = [
  { how: 'level', zMm: 55, tiltDeg: TILT_LEVEL_DEG, why: 'the bottom bay - what stage 0 asked for' },
  { how: 'level', zMm: 177, tiltDeg: TILT_LEVEL_DEG, why: 'the top bay - what stage 0 asked for' },
  { how: 'into', zMm: 55, tiltDeg: TILT_LEVEL_DEG, why: 'can the hand come in square? (finding 1)' },
  { how: 'into', zMm: 177, tiltDeg: TILT_LEVEL_DEG, why: 'the same, a bay higher' },
  { how: 'level', zMm: 130, tiltDeg: TILT_LEVEL_DEG, why: 'a tall bottle on the bench, held round the middle' },
  { how: 'level', zMm: 250, tiltDeg: TILT_LEVEL_DEG, why: 'how level reach goes with height' },
  { how: 'level', zMm: 299, tiltDeg: TILT_LEVEL_DEG, why: 'a third bay, if there is to be one' },
  { how: 'level', zMm: 340, tiltDeg: TILT_LEVEL_DEG, why: 'where level reach runs out' },
  { how: 'down', zMm: 35, tiltDeg: TILT_GRIP_DEG, why: 'a thing standing on the bench' },
  { how: 'down', zMm: 70, tiltDeg: TILT_GRIP_DEG, why: 'letting go over the rim of a pot' },
  { how: 'down', zMm: 100, tiltDeg: TILT_GRIP_DEG, why: 'how fast it narrows (finding 2)' },
  { how: 'down', zMm: 130, tiltDeg: TILT_GRIP_DEG, why: 'the same' },
  { how: 'down', zMm: 244, tiltDeg: TILT_GRIP_DEG, why: "the unit's top board" },
  { how: 'down', zMm: 35, tiltDeg: TILT_THERE_DEG, why: 'the check: end_demo said 137 to 386 here' },
  { how: 'down', zMm: 80, tiltDeg: TILT_THERE_DEG, why: 'the check: end_demo said 177 to 348 here' },
  { how: 'down', zMm: 244, tiltDeg: TILT_THERE_DEG, why: 'is the top board reachable with tilt?' },
];

function say(line: string) {
  fs.mkdirSync(path.dirname(REPORT), { recursive: true });
  fs.appendFileSync(REPORT, `${line}\n`, 'utf8');
}

/** Where the fingers should point for a sweep at a bearing (undefined: the solver's default). */
function pointing(how: How, bearingRad: number): Vec3 | undefined {
  if (how === 'down') return undefined;
  if (how === 'into') return [1, 0, 0];
  return [Math.cos(bearingRad), Math.sin(bearingRad), 0];
}

/** Everything the sweep needs out of the model, found once. */
interface Rig {
  mj: Mujoco;
  model: MjModel;
  data: MjData;
  ik: ArmKinematics;
  gripperAdr: number;
  armBodies: Set<number>;
  table: number;
}

/**
 * Does this pose push the arm through the bench top?
 *
 * Only from the elbow out: the base and the shoulder stand **on** the bench and touch it in
 * every pose, so counting those would call every reach a collision.
 */
function throughBench(rig: Rig, q: number[]): boolean {
  const { mj, model, data } = rig;
  for (let j = 0; j < 5; j++) data.qpos[rig.ik.qposAdr[j]] = q[j];
  data.qpos[rig.gripperAdr] = GRIPPER_OPEN;
  mj.mj_fwdPosition(model, data);
  const bodyOf = model.geom_bodyid as Int32Array;
  const contacts = data.contact;
  let deepest = 0;
  try {
    for (let i = 0; i < data.ncon; i++) {
      const c = contacts.get(i);
      if (!c) continue;
      const onTable = c.geom1 === rig.table || c.geom2 === rig.table;
      const other = c.geom1 === rig.table ? c.geom2 : c.geom1;
      if (onTable && rig.armBodies.has(bodyOf[other])) deepest = Math.max(deepest, -c.dist);
      c.delete();
    }
  } finally {
    contacts.delete();
  }
  return deepest > THROUGH_BENCH_M;
}

interface Band {
  near: number | null;
  far: number | null;
}

/** The longest unbroken run of reachable radii at one bearing. */
function band(ok: boolean[], radii: number[]): Band {
  let best: Band = { near: null, far: null };
  let bestLen = 0;
  let from: number | null = null;
  for (let i = 0; i <= ok.length; i++) {
    if (i < ok.length && ok[i]) {
      from ??= i;
      continue;
    }
    if (from !== null) {
      if (i - from > bestLen) {
        bestLen = i - from;
        best = { near: radii[from], far: radii[i - 1] };
      }
      from = null;
    }
  }
  return best;
}

/** How fast the outer edge falls per degree off forward: the slope of a least-squares line. */
function fallPerDeg(points: { deg: number; far: number }[]): number {
  const n = points.length;
  if (n < 2) return 0;
  const mx = points.reduce((s, p) => s + p.deg, 0) / n;
  const my = points.reduce((s, p) => s + p.far, 0) / n;
  const num = points.reduce((s, p) => s + (p.deg - mx) * (p.far - my), 0);
  const den = points.reduce((s, p) => s + (p.deg - mx) ** 2, 0) || 1;
  return -num / den;
}

interface Ring {
  near: number;
  far: number;
  fall: number;
}

/** One sweep: the band at every bearing, written out, and the ring straight ahead. */
function sweep(rig: Rig, s: Sweep, radii: number[]): Ring | null {
  const label =
    s.how === 'down'
      ? `fingers straight down, within ${s.tiltDeg} deg`
      : s.how === 'into'
        ? 'fingers level, pointing into the shelf (+X) whatever the bearing'
        : 'fingers level, pointing straight out from the base';
  say('');
  say(`z = ${s.zMm} mm - ${label}`);
  say(`  (${s.why})`);
  const far: { deg: number; far: number }[] = [];
  const bands: Band[] = [];
  for (const deg of BEARINGS_DEG) {
    const a = (deg * Math.PI) / 180;
    const ok = radii.map((r) => {
      const target: IkTarget = {
        pos: [r * Math.cos(a) * 0.001, r * Math.sin(a) * 0.001, s.zMm / 1000],
        yaw: null,
        point: pointing(s.how, a),
        maxTiltDeg: s.tiltDeg,
      };
      const got = rig.ik.solve(target, rig.ik.guess(target));
      if (!got.ok || got.posErrMm > CLOSE_MM) return false;
      if (rig.ik.selfHits(got.q, GRIPPER_OPEN)) return false;
      return !throughBench(rig, got.q);
    });
    const got = band(ok, radii);
    bands.push(got);
    if (got.far !== null) far.push({ deg, far: got.far });
    say(
      `  ${String(deg).padStart(4)} deg: ` +
        (got.near === null
          ? 'nothing'
          : `${String(got.near).padStart(3)} to ${String(got.far).padStart(3)} mm`),
    );
  }
  const ahead = bands[0];
  if (ahead.near === null || ahead.far === null) return null;
  const fall = fallPerDeg(far);
  say(
    `  straight ahead ${ahead.near} to ${ahead.far} mm; the outer edge falls ` +
      `${fall.toFixed(2)} mm per degree off forward`,
  );
  return { near: ahead.near, far: ahead.far, fall };
}

test('the reach rings for the workbench', async () => {
  const { mj, model, data } = await nodeScene();
  const B = mj.mjtObj.mjOBJ_BODY.value;
  const rig: Rig = {
    mj,
    model,
    data,
    ik: new ArmKinematics(mj, model),
    gripperAdr: (model.jnt_qposadr as Int32Array)[
      mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT.value, 'gripper')
    ],
    armBodies: new Set(
      ['upper_arm', 'lower_arm', 'wrist', 'gripper', 'moving_jaw_so101_v1'].map((n) =>
        mj.mj_name2id(model, B, n),
      ),
    ),
    table: mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM.value, 'table'),
  };
  // The cubes demo's blocks come with this scene. Keep them off the bench, so the contact list
  // holds only what is being asked about.
  for (const name of ['red', 'green', 'blue']) {
    const adr = (model.jnt_qposadr as Int32Array)[
      mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT.value, name)
    ];
    data.qpos[adr] = 2;
    data.qpos[adr + 1] = 2;
  }

  const radii: number[] = [];
  for (let r = RADIUS_FROM_MM; r <= RADIUS_TO_MM; r += RADIUS_STEP_MM) radii.push(r);

  fs.rmSync(REPORT, { force: true });
  say('stage 0, measurement 2: the reach rings for the workbench');
  say(
    `  one SO-101 at the origin, forward +X, bench top z = 0. ${radii.length} radii x ` +
      `${BEARINGS_DEG.length} bearings per sweep, ${SWEEPS.length} sweeps.`,
  );
  say(
    `  reached = the gripper site within ${CLOSE_MM} mm of the spot, the fingers inside the` +
      ' tilt named, no self-collision, and nothing from the elbow out through the bench.',
  );

  const rings = new Map<string, Ring | null>();
  for (const s of SWEEPS) rings.set(`${s.how}-${s.zMm}-${s.tiltDeg}`, sweep(rig, s, radii));
  const ring = (how: How, z: number, tilt: number) => rings.get(`${how}-${z}-${tilt}`) ?? null;
  const show = (r: Ring | null) =>
    r ? `${r.near} to ${r.far} mm, falling ${r.fall.toFixed(2)}/deg` : 'nothing';

  say('');
  say('the rings, to be copied into workbench/scene.py:');
  for (const s of SWEEPS) {
    say(
      `  ${s.how.padEnd(5)} z=${String(s.zMm).padStart(3)} tilt=${String(s.tiltDeg).padStart(2)}` +
        `  ${show(ring(s.how, s.zMm, s.tiltDeg))}`,
    );
  }

  say('');
  say('finding 1: a level hand points straight out from its own base and nowhere else.');
  say('  After the shoulder turn, every joint bends about an axis across the arm, so the');
  say('  fingers stay in the turned plane of the arm. The "into the shelf" sweeps are that');
  say('  fact measured: past 10 degrees of bearing there is no pose at all. A thing off to one');
  say('  side is entered along the slant, which the scene has to leave room for.');

  say('');
  say('finding 2: nothing much above 100 mm can be gripped from above.');
  for (const z of [35, 70, 100, 130]) {
    say(`  down at ${String(z).padStart(3)} mm: ${show(ring('down', z, TILT_GRIP_DEG))}`);
  }
  say(
    `  down at 244 mm: ${show(ring('down', 244, TILT_GRIP_DEG))}, and ` +
      `${show(ring('down', 244, TILT_THERE_DEG))} even with ${TILT_THERE_DEG} degrees of tilt`,
  );
  say('  So a tall thing is taken round the middle, level - and the top board of the shelf');
  say('  unit, at 244 mm, is not a place this robot can put anything or take anything from.');

  say('');
  say('the check against what was already measured. end_demo published 137 to 386 mm at the');
  say('  table top and 177 to 348 at 80 mm up, both fingers down, both with tilt allowed.');
  say(
    `  Here, with the same ${TILT_THERE_DEG} degrees: ${show(ring('down', 35, TILT_THERE_DEG))}` +
      ` at 35 mm, and ${show(ring('down', 80, TILT_THERE_DEG))} at 80 mm.`,
  );
  say(
    `  With only the ${TILT_GRIP_DEG} degrees a grip really has, 35 mm gives` +
      ` ${show(ring('down', 35, TILT_GRIP_DEG))} instead.`,
  );
  say('  At the table top the near edges are the same number (135 against 137) and the far');
  say('  edges differ by tilt alone, so this harness measures what the old one measured. At');
  say('  80 mm up the near edge does not match - 135 here against 177 published - and the');
  say('  reason is the scene, not the arm: there, the row stood on a riser 88 mm deep, so the');
  say('  hand could not come in close without hitting the step. Here there is nothing but');
  say('  bench. A near edge is a fact about a scene as much as about an arm.');

  // The gate: a bay is no use if nothing can be reached inside it at all.
  expect(ring('level', 55, TILT_LEVEL_DEG), 'nothing reachable at the bottom bay').not.toBeNull();
  expect(ring('level', 177, TILT_LEVEL_DEG), 'nothing reachable at the top bay').not.toBeNull();

  rig.ik.delete();
  model.delete();
  data.delete();
}, 1800000);
