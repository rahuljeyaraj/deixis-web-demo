/**
 * Stage 2: **where can the arm hold a thing while the carriage runs?**
 *
 * Sliding with something in the jaws is not free. Whatever is held has to clear everything
 * standing on the bench - the tallest is the bottle of oil at 148 mm - and the arm has to be
 * able to get there and stay there, which near the base it very often cannot: stage 0 found
 * that reaching **down** dies above 100 mm, so there is no pose at all with the fingers
 * pointing at the bench from 240 mm up.
 *
 * So this sweeps the question rather than guessing at it: for a hand pointing down, and for a
 * hand pointing level into the shelf, how high and how far out can it be put, and how far off
 * does the hand end up pointing?
 *
 * Run: `PROBES=1 npx vitest run tests/probes/stage2-carry.probe.test.ts`
 * Report: `out/stage2-carry.txt`
 */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { expect, test } from 'vitest';

import { slideOf } from '../../src/sim/bench';
import { ArmKinematics } from '../../src/sim/ik';
import { loadScene, mujoco, unpack } from '../../src/sim/mujoco';
import type { PackFile } from '../../src/sim/mujoco';
import type { Vec3 } from '../../src/sim/table';

const WEB = path.join(import.meta.dirname, '..', '..');
const PACK = path.join(WEB, 'public', 'so101.pack');
const SCENE = path.join(WEB, 'model', 'workbench_scene.xml');
const REPORT = path.join(WEB, '..', 'out', 'stage2-carry.txt');

/** How high a held thing has to be to clear the bench: the oil is 148 mm, the pot 59. */
const CLEARS_THE_BENCH_MM = 160;

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

test('where a thing can be held while the carriage runs', async () => {
  fs.rmSync(REPORT, { force: true });
  const mj = await mujoco();
  const { model, data } = loadScene(mj, files(), 'workbench_scene.xml');
  mj.mj_resetDataKeyframe(model, data, 0);
  mj.mj_forward(model, data);
  const ik = new ArmKinematics(mj, model, 'arm_');
  ik.ridesOn('slide');
  ik.rideTo(slideOf(350));

  say('stage 2: where can the arm hold something while the carriage runs?');
  say('  the hand at a place in front of the carriage, pointing one of two ways. "off" is how');
  say('  far the fingers end up from the way they were asked to point - whatever is held tips');
  say('  with them.');
  say('');

  const DOWN: Vec3 = [0, 0, -1];
  const LEVEL: Vec3 = [1, 0, 0];
  const best: Record<string, { y: number; z: number; off: number } | null> = {
    down: null,
    level: null,
  };

  for (const [name, point] of [
    ['down', DOWN],
    ['level', LEVEL],
  ] as const) {
    say(`  fingers ${name}:`);
    say(`    ${'out (mm)'.padEnd(10)} ${[140, 170, 200, 230, 260].map((y) => `y=${y}`.padStart(11)).join(' ')}`);
    for (const z of [280, 250, 220, 190, 160]) {
      const row: string[] = [];
      for (const y of [140, 170, 200, 230, 260]) {
        const pos: Vec3 = [y / 1000, ik.base[1], z / 1000];
        const got = ik.solve({ pos, point, yaw: null, maxTiltDeg: 80 });
        const folds = got.posErrMm < 1 && ik.selfHits(got.q, 0);
        row.push(
          folds
            ? '     folds'.padStart(11)
            : got.posErrMm < 1
              ? `${got.angErrDeg.toFixed(0)}deg off`.padStart(11)
              : `${got.posErrMm.toFixed(0)}mm short`.padStart(11),
        );
        if (got.posErrMm < 1 && !folds && z >= CLEARS_THE_BENCH_MM) {
          const cur = best[name];
          if (!cur || got.angErrDeg < cur.off - 0.5 || (got.angErrDeg < cur.off + 0.5 && z > cur.z)) {
            best[name] = { y, z, off: got.angErrDeg };
          }
        }
      }
      say(`    z = ${String(z).padEnd(6)} ${row.join(' ')}`);
    }
    say('');
  }

  for (const name of ['down', 'level'] as const) {
    const b = best[name];
    say(
      b
        ? `  fingers ${name}: best is ${b.y} mm out at ${b.z} mm up, fingers ${b.off.toFixed(0)} degrees off.`
        : `  fingers ${name}: nowhere clears the bench.`,
    );
  }
  say(`  (a held thing has to be above ${CLEARS_THE_BENCH_MM} mm to clear the bottle of oil.)`);

  expect(best.down, 'there should be somewhere to hold a thing taken from above').not.toBeNull();
  expect(best.level, 'there should be somewhere to hold a thing taken from a bay').not.toBeNull();
  model.delete();
  data.delete();
}, 600000);
