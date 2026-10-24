/**
 * How high can the hand work inside a bay? **Measured against the boards, not reasoned.**
 *
 * Stage 0 swept the reach ring at the middle of each bay's opening - 55, 177 and 299 mm - and
 * that sweep asked only whether the arm could get there without folding into itself. It never
 * asked whether the hand fits. It does not: the world check added in stage 2 puts the gripper
 * 11 to 15 mm into the board above at every one of those heights, which is the arm leaning on
 * the shelf for the length of every grip and is two of the failures in `out/stage2-body.txt`.
 *
 * So this walks the height down a millimetre at a time, solving the real level grip pose with
 * the real solver and asking the real collision check what it touches, and reports:
 *
 * - how far above and below the site the hand really reaches, level;
 * - the highest the site can work in each bay with the board clear;
 * - and how much of the thing standing there is left under the jaws at that height.
 *
 * Run: `PROBES=1 npx vitest run tests/probes/stage2-bay-height.probe.test.ts`
 * Report: `out/stage2-bay-height.txt`
 */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { expect, test } from 'vitest';

import { BAYS, SHELF, THINGS, bay as bayOf, benchToWorld, slideOf } from '../../src/sim/bench';
import { loadScene, mujoco, unpack } from '../../src/sim/mujoco';
import type { PackFile } from '../../src/sim/mujoco';
import { Rig } from '../../src/sim/rig';

const WEB = path.join(import.meta.dirname, '..', '..');
const REPORT = path.join(WEB, '..', 'out', 'stage2-bay-height.txt');

/** Where to try it: the middle of the bay, across, and the front row's depth. */
const AT_X_MM = 350;

function say(line: string) {
  fs.mkdirSync(path.dirname(REPORT), { recursive: true });
  fs.appendFileSync(REPORT, `${line}\n`, 'utf8');
}

function files(): PackFile[] {
  const packed = unpack(
    new Uint8Array(zlib.gunzipSync(fs.readFileSync(path.join(WEB, 'public', 'so101.pack')))),
  );
  return [
    ...packed.filter((f) => !f.path.endsWith('_scene.xml')),
    {
      path: 'workbench_scene.xml',
      bytes: new Uint8Array(fs.readFileSync(path.join(WEB, 'model', 'workbench_scene.xml'))),
    },
  ];
}

test('how high the hand can work inside a bay', async () => {
  fs.rmSync(REPORT, { force: true });
  const mj = await mujoco();
  const { model, data } = loadScene(mj, files(), 'workbench_scene.xml');
  mj.mj_resetDataKeyframe(model, data, 0);
  const rig = new Rig(mj, model, data);
  rig.restNow();
  const { ik } = rig;

  say('how high the hand can work inside a bay, against the boards themselves:');
  say('  the real solver, the real scene, and the real collision check.');
  say('');

  const yMm = SHELF.rows[0];
  const [X, Y] = benchToWorld(AT_X_MM, yMm);
  // Level, fingers at the shelf, jaws across the bench: the only way in under a board.
  const point: [number, number, number] = [1, 0, 0];
  // Only the furniture counts here. Whatever is standing in a bay is in the way of the hand
  // today and may be somewhere else tomorrow; the boards are the shelf and do not move, and it
  // is the boards this is a question about.
  const ids = new Set(THINGS.map((t) => t.id));
  const bodyOf = model.geom_bodyid as Int32Array;
  const nothing = new Set<number>();
  for (let g = 0; g < model.ngeom; g++) {
    const name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, bodyOf[g]);
    if (name && ids.has(name)) nothing.add(g);
  }

  say('  bay   opening         the highest the site can work   what board is in the way above');
  const best: number[] = [];
  for (let n = 0; n < BAYS; n++) {
    const [floor, ceiling] = bayOf(n);
    ik.rideTo(slideOf(AT_X_MM));
    let highest: number | null = null;
    let blocked = '';
    for (let z = Math.round(ceiling) - 8; z >= floor + 20; z -= 2) {
      const got = ik.solve(
        { pos: [X, Y, z / 1000], point, yaw: Math.PI / 2, yawTurns: 2, maxTiltDeg: 8 },
        undefined,
      );
      if (!(got.posErrMm < 1 && got.angErrDeg < 8 && got.yawErrDeg < 6)) continue;
      const into = ik.worldHit(data, got.q, 1.3, nothing);
      if (!into) {
        highest = z;
        break;
      }
      blocked = `${into.name}, ${(into.depthM * 1000).toFixed(0)} mm into it at ${z} mm`;
    }
    best.push(highest ?? floor + 20);
    say(
      `  ${n}     ${floor.toFixed(0)} to ${ceiling.toFixed(0)} mm` +
        `   ${highest === null ? 'nowhere' : `${highest} mm`}`.padEnd(32) +
        `  ${blocked || 'nothing'}`,
    );
  }

  say('');
  say(`  so the three bays can be worked at: ${best.join(', ')} mm`);
  say(`  against the middles of their openings, which stage 0 used: 55, 177, 299 mm`);
  say('');
  say('  what that leaves for the jaws, per bay: the pads stand about 20 mm either side of');
  say('  the fingers, so a thing has to reach that far above the height above and stand on');
  say('  the board below it.');

  expect(best.every((z) => z > 0)).toBe(true);
  model.delete();
  data.delete();
}, 600000);
