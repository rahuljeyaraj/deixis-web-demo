/**
 * Stage 2: **what does the gripper joint mean in millimetres, and where are the jaws?**
 *
 * Two numbers the body code was guessing at, and guessing wrongly:
 *
 * - how far apart the jaws stand for a given joint value. A straight line from nothing at 0 to
 *   107 mm at the joint's limit is the obvious guess and it is wrong, because the jaws are not
 *   shut at nought - the cubes demo found them touching a 30 mm block at 0.18 (`arm.ts`).
 * - **how far they reach ahead of the site** the solver aims. Everything here is aimed by
 *   putting `gripperframe` somewhere, and if the open jaws stand out in front of it then coming
 *   in until the site reaches a jar means the jaws reached the jar some way back - and pushed
 *   it, which is what a jar being found 22 mm further into its bay than it started looks like.
 *
 * Both are read off the model: the jaws' own geoms, projected onto the site's own axes.
 *
 * Run: `PROBES=1 npx vitest run tests/probes/stage2-jaws.probe.test.ts`
 * Report: `out/stage2-jaws.txt`
 */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { expect, test } from 'vitest';

import { loadScene, mujoco, unpack } from '../../src/sim/mujoco';
import type { PackFile } from '../../src/sim/mujoco';

const WEB = path.join(import.meta.dirname, '..', '..');
const PACK = path.join(WEB, 'public', 'so101.pack');
const SCENE = path.join(WEB, 'model', 'workbench_scene.xml');
const REPORT = path.join(WEB, '..', 'out', 'stage2-jaws.txt');

function say(line: string) {
  fs.mkdirSync(path.dirname(REPORT), { recursive: true });
  fs.appendFileSync(REPORT, `${line}\n`, 'utf8');
}

function files(): PackFile[] {
  const packed = unpack(new Uint8Array(zlib.gunzipSync(fs.readFileSync(PACK))));
  return [
    ...packed.filter((f) => !f.path.endsWith('_scene.xml')),
    { path: 'workbench_scene.xml', bytes: new Uint8Array(fs.readFileSync(SCENE)) },
  ];
}

test('what the gripper joint means, and where the jaws are', async () => {
  fs.rmSync(REPORT, { force: true });
  const mj = await mujoco();
  const { model, data } = loadScene(mj, files(), 'workbench_scene.xml');
  mj.mj_resetDataKeyframe(model, data, 0);

  const O = mj.mjtObj;
  const site = mj.mj_name2id(model, O.mjOBJ_SITE.value, 'arm_gripperframe');
  const gripper = mj.mj_name2id(model, O.mjOBJ_BODY.value, 'arm_gripper');
  const jaw = mj.mj_name2id(model, O.mjOBJ_BODY.value, 'arm_moving_jaw_so101_v1');
  const q = (model.jnt_qposadr as Int32Array)[mj.mj_name2id(model, O.mjOBJ_JOINT.value, 'arm_gripper')];
  const bodyOf = model.geom_bodyid as Int32Array;

  say('stage 2: what the gripper joint means, and where the jaws stand.');
  say('  measured on the jaws\' own geoms, projected onto the gripperframe site\'s axes -');
  say('  "along" is the way the fingers point, "across" is the way the jaws open.');
  say('');
  say(`  ${'joint'.padEnd(7)} ${'apart'.padStart(8)} ${'fixed jaw'.padStart(11)} ${'moving jaw'.padStart(11)} ${'reach ahead'.padStart(12)}`);

  const rows: { q: number; apart: number; ahead: number; behind: number }[] = [];
  for (const v of [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.745]) {
    data.qpos[q] = v;
    mj.mj_forward(model, data);
    const p = data.site_xpos as Float64Array;
    const R = data.site_xmat as Float64Array;
    const o: [number, number, number] = [p[3 * site], p[3 * site + 1], p[3 * site + 2]];
    // The site's axes: x is the way the fingers point, z the way the jaws open.
    const along: [number, number, number] = [R[9 * site], R[9 * site + 3], R[9 * site + 6]];
    const across: [number, number, number] = [R[9 * site + 2], R[9 * site + 5], R[9 * site + 8]];
    const xg = data.geom_xpos as Float64Array;
    const size = model.geom_size as Float64Array;
    let fixed = Infinity;
    let moving = -Infinity;
    let ahead = -Infinity;
    let behind = Infinity;
    for (let g = 0; g < model.ngeom; g++) {
      const b = bodyOf[g];
      if (b !== gripper && b !== jaw) continue;
      const d: [number, number, number] = [
        xg[3 * g] - o[0],
        xg[3 * g + 1] - o[1],
        xg[3 * g + 2] - o[2],
      ];
      // `geom_rbound` is the radius of the sphere round the whole geom, whatever shape it is -
      // the mesh jaws have no useful `size`, which is what made the first run of this report
      // say the jaws reach 6 mm ahead of the site when they reach ten times that.
      const r = (model.geom_rbound as Float64Array)[g];
      const a = d[0] * across[0] + d[1] * across[1] + d[2] * across[2];
      const l = d[0] * along[0] + d[1] * along[1] + d[2] * along[2];
      if (b === gripper && a < 0) fixed = Math.min(fixed, a);
      if (b === jaw) moving = Math.max(moving, a);
      ahead = Math.max(ahead, l + r);
      behind = Math.min(behind, l - r);
    }
    const apart = (moving - fixed) * 1000;
    rows.push({ q: v, apart, ahead: ahead * 1000, behind: behind * 1000 });
    say(
      `  ${v.toFixed(3).padEnd(7)} ${apart.toFixed(0).padStart(6)}mm ${(fixed * 1000).toFixed(0).padStart(9)}mm ` +
        `${(moving * 1000).toFixed(0).padStart(9)}mm ` +
        `${(behind * 1000).toFixed(0).padStart(6)} to ${(ahead * 1000).toFixed(0)}mm`,
    );
  }

  // A straight line through the two ends, which is what the body code needs.
  const lo = rows[0];
  const hi = rows[rows.length - 1];
  const slope = (hi.apart - lo.apart) / (hi.q - lo.q);
  say('');
  say(
    `  a straight line through it: apart = ${lo.apart.toFixed(0)} + ${slope.toFixed(0)} x joint,` +
      ` so joint = (apart - ${lo.apart.toFixed(0)}) / ${slope.toFixed(0)}.`,
  );
  const worst = Math.max(...rows.map((r) => Math.abs(lo.apart + slope * r.q - r.apart)));
  say(`  the worst that line is out by, over the whole range: ${worst.toFixed(0)} mm.`);
  say(
    `  along the fingers the jaws run from ${Math.min(...rows.map((r) => r.behind)).toFixed(0)} to` +
      ` ${Math.max(...rows.map((r) => r.ahead)).toFixed(0)} mm of the site, so a thing is held` +
      ` about ${(
        (Math.min(...rows.map((r) => r.behind)) + Math.max(...rows.map((r) => r.ahead))) /
        2
      ).toFixed(0)} mm along the fingers from it.`,
  );

  expect(rows[rows.length - 1].apart, 'the jaws should open about 107 mm').toBeGreaterThan(90);

  // ---------------------------------------------------------------- and where to aim
  //
  // The two numbers above say what the jaws do. This says where to put them: the hand is driven
  // at a jar, level, at a range of offsets **along the fingers**, and the jaws are closed. The
  // offset that holds the jar without shoving it is the one the body code wants.
  say('');
  say('where to aim, along the fingers: the hand driven at a jar and the jaws closed.');
  say('  along the fingers down the side, across the opening along the top;');
  say('  each cell is how many jaws ended on the jar and how far it was shoved.');

  const { ArmKinematics } = await import('../../src/sim/ik');
  const { slideOf, thing } = await import('../../src/sim/bench');
  const { jawsAt, gripMiddle } = await import('../../src/sim/rig');
  const ik = new ArmKinematics(mj, model, 'arm_');
  ik.ridesOn('slide');
  const jar = thing('jar1');
  const slide = slideOf(jar.slideTo!);
  ik.rideTo(slide);
  const ctrl = [
    mj.mj_name2id(model, O.mjOBJ_ACTUATOR.value, 'slide'),
    ...['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'].map(
      (n) => mj.mj_name2id(model, O.mjOBJ_ACTUATOR.value, `arm_${n}`),
    ),
  ];
  const jarBody = mj.mj_name2id(model, O.mjOBJ_BODY.value, 'jar1');
  const jarGeoms: number[] = [];
  for (let g = 0; g < model.ngeom; g++) if (bodyOf[g] === jarBody) jarGeoms.push(g);
  const wide = jawsAt(jar.gripMm + 18);
  const shut = jawsAt(jar.gripMm - 7);
  const best = { off: 0, across: 0, moved: Infinity, touched: false };

  const ACROSS = [-30, -15, 0, 15, 30, 45];
  say(`  ${''.padStart(7)} ${ACROSS.map((a) => `${a}mm`.padStart(11)).join(' ')}`);
  for (const offMm of [-20, -10, 0, 10, 20]) {
   const row: string[] = [];
   for (const acrossMm of ACROSS) {
    mj.mj_resetDataKeyframe(model, data, 0);
    mj.mj_forward(model, data);
    const X = jar.at[1] / 1000;
    const Y = -(jar.at[0] - 350) / 1000;
    const point: [number, number, number] = [1, 0, 0];
    const middle = gripMiddle(jar.gripMm + 18) + acrossMm / 1000;
    const aim = ik.solve({
      pos: [X, Y, jar.gripZ / 1000],
      point,
      yaw: Math.PI / 2,
      yawTurns: 2,
      maxTiltDeg: 6,
    });
    const opens = ik.fk(aim.q).open;
    const got = ik.solve({
      pos: [
        X + offMm / 1000 - middle * opens[0],
        Y - middle * opens[1],
        jar.gripZ / 1000 - middle * opens[2],
      ],
      point,
      yaw: Math.PI / 2,
      yawTurns: 2,
      maxTiltDeg: 6,
    }, aim.q);
    if (got.posErrMm > 1) {
      row.push('out of reach'.padStart(11));
      continue;
    }
    // Put the arm straight at the grip with the jaws open, let it settle, then close them.
    for (let j = 0; j < 5; j++) data.qpos[ik.qposAdr[j]] = got.q[j];
    data.qpos[(model.jnt_qposadr as Int32Array)[mj.mj_name2id(model, O.mjOBJ_JOINT.value, 'slide')]] = slide;
    data.ctrl[ctrl[0]] = slide;
    for (let j = 0; j < 5; j++) data.ctrl[ctrl[j + 1]] = got.q[j];
    data.ctrl[ctrl[6]] = wide;
    data.qpos[q] = wide;
    mj.mj_forward(model, data);
    const was = [data.xpos[3 * jarBody], data.xpos[3 * jarBody + 1], data.xpos[3 * jarBody + 2]];
    for (let i = 0; i < 300; i++) {
      data.ctrl[ctrl[6]] = wide + (shut - wide) * Math.min(1, i / 200);
      mj.mj_step(model, data);
    }
    const now = [data.xpos[3 * jarBody], data.xpos[3 * jarBody + 1], data.xpos[3 * jarBody + 2]];
    const moved = Math.hypot(now[0] - was[0], now[1] - was[1], now[2] - was[2]) * 1000;
    const touched = new Set<number>();
    const cs = data.contact;
    for (let i = 0; i < data.ncon; i++) {
      const c = cs.get(i);
      if (!c) continue;
      const other = jarGeoms.includes(c.geom1) ? c.geom2 : jarGeoms.includes(c.geom2) ? c.geom1 : -1;
      if (other >= 0 && (bodyOf[other] === gripper || bodyOf[other] === jaw)) touched.add(bodyOf[other]);
      c.delete();
    }
    cs.delete();
    row.push(`${touched.size === 2 ? 'both' : touched.size} ${moved.toFixed(0)}mm`.padStart(11));
    if (touched.size === 2 && moved < best.moved) {
      best.off = offMm;
      best.across = acrossMm;
      best.moved = moved;
      best.touched = true;
    }
   }
   say(`  ${`${offMm}mm`.padStart(7)} ${row.join(' ')}`);
  }
  say('');
  say(
    best.touched
      ? `  aim ${best.off} mm along the fingers and ${best.across} mm further along the opening` +
        ` than the arithmetic says: both jaws on it, and it moved ${best.moved.toFixed(0)} mm` +
        ' being gripped.'
      : '  no offset along the fingers got both jaws onto it.',
  );
  expect(best.touched, 'some offset should grip a jar').toBe(true);

  model.delete();
  data.delete();
}, 600000);
