/**
 * Stage 2, the reach half: **can the solver actually put the hand where a grip needs it?**
 *
 * The rings stage 0 swept say a bay is reachable. A ring is a claim about one point; a grip is
 * a claim about a whole straight line in to that point, with the fingers pointing one way and
 * the jaws turned one way, and this is where that claim is checked - thing by thing, at the
 * carriage position the layout chose, with the arm's own self-collision test on every pose.
 *
 * It reports how far short the hand ends up when it cannot, because "no" is not an answer you
 * can act on and "no, by 40 mm" is.
 *
 * Run: `PROBES=1 npx vitest run tests/probes/stage2-reach.probe.test.ts`
 * Report: `out/stage2-reach.txt`
 */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { expect, test } from 'vitest';

import { THINGS, slideOf } from '../../src/sim/bench';
import { ArmKinematics } from '../../src/sim/ik';
import { loadScene, mujoco, unpack } from '../../src/sim/mujoco';
import type { PackFile } from '../../src/sim/mujoco';
import type { Vec3 } from '../../src/sim/table';

const WEB = path.join(import.meta.dirname, '..', '..');
const PACK = path.join(WEB, 'public', 'so101.pack');
const SCENE = path.join(WEB, 'model', 'workbench_scene.xml');
const REPORT = path.join(WEB, '..', 'out', 'stage2-reach.txt');

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

const DOWN: Vec3 = [0, 0, -1];

test('where the hand can be put, for every grip in the world', async () => {
  fs.rmSync(REPORT, { force: true });
  const mj = await mujoco();
  const { model, data } = loadScene(mj, files(), 'workbench_scene.xml');
  mj.mj_resetDataKeyframe(model, data, 0);
  mj.mj_forward(model, data);
  const ik = new ArmKinematics(mj, model, 'arm_');
  ik.ridesOn('slide');

  say('stage 2, the reach half: can the hand be put where each grip needs it?');
  say('  solved at the carriage position the layout chose, fingers pointing the way the grip');
  say('  needs, and every pose checked against the arm folding into itself.');
  say('');
  say(`  ${'thing'.padEnd(13)} ${'way in'.padEnd(6)} ${'jaws'.padEnd(9)} at the grip   backed off`);

  const stuck: string[] = [];
  for (const t of THINGS) {
    if (!t.graspable || t.slideTo === null) continue;
    ik.rideTo(slideOf(t.slideTo));
    const level = t.gripHow === 'level';
    const X = t.at[1] / 1000;
    const Y = -(t.at[0] - 350) / 1000;
    const z = level ? t.gripZ / 1000 : (t.rises[0] + 7) / 1000;
    const grip: Vec3 = [X, Y, z];
    // Level, the hand comes in along the line from where the arm stands; from above, it comes
    // straight down. Both the same as `rig.ts` plans them.
    const r = Math.hypot(X - ik.base[0], Y - ik.base[1]) || 1;
    const point: Vec3 = level ? [(X - ik.base[0]) / r, (Y - ik.base[1]) / r, 0] : DOWN;
    // The same backing-off `rig.ts` plans: the first of 100, 70 and 50 mm that works, because
    // straight out along the line is sometimes into the arm's own dead circle.
    const offs: Vec3[] = level
      ? [0.1, 0.07, 0.05].map((d) => [X - d * point[0], Y - d * point[1], z] as Vec3)
      : [[X, Y, z + 0.09]];
    const long = -((t.yaw * Math.PI) / 180) + (t.room[0] >= t.room[1] ? 0 : Math.PI / 2);
    const across = Math.atan2(point[1], point[0]) + Math.PI / 2;
    const tries: { name: string; yaw: number | null }[] = level
      ? [{ name: 'across', yaw: across }]
      : [
          { name: 'across', yaw: long + Math.PI / 2 },
          { name: 'along', yaw: long },
          { name: 'free', yaw: null },
        ];
    let best = '';
    let bestErr = Infinity;
    let name = '-';
    for (const how of tries) {
      const a = ik.solve({ pos: grip, point, yaw: how.yaw, yawTurns: 2, maxTiltDeg: 6 });
      let b = a;
      for (const off of offs) {
        b = ik.solve({ pos: off, point, yaw: how.yaw, yawTurns: 2, maxTiltDeg: level ? 14 : 30 }, a.q);
        if (b.ok) break;
      }
      const hits = a.ok && ik.selfHits(a.q, 1.1);
      const err = a.posErrMm + b.posErrMm + (hits ? 1000 : 0);
      if (err < bestErr) {
        bestErr = err;
        name = how.name;
        const show = (r: typeof a) =>
          `${r.ok ? 'ok ' : 'NO '}${r.posErrMm.toFixed(1)}mm ${r.angErrDeg.toFixed(0)}deg ` +
          `${r.yawErrDeg.toFixed(0)}yaw`;
        best = `${show(a).padEnd(22)}${show(b)}${hits ? '  (folds into itself)' : ''}`;
      }
      if (err < 2) break;
    }
    say(`  ${t.id.padEnd(13)} ${t.gripHow.padEnd(6)} ${name.padEnd(9)} ${best}`);
    if (bestErr >= 2) stuck.push(t.id);
  }

  say('');
  say(`  ${stuck.length ? `no way in to: ${stuck.join(', ')}` : 'every grip has a way in.'}`);

  expect(stuck, 'every graspable thing should have a way in').toEqual([]);
  model.delete();
  data.delete();
}, 600000);
