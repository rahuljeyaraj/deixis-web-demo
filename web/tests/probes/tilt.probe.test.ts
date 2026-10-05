/**
 * How far must the fingers tilt from straight down to hold a block at a height, by distance
 * from the base? (rahul, 6 Oct: "why not pick it from the side?") For each distance and each
 * height of the block's bottom (table, on 1, 2, 3 blocks), the smallest tilt that solves.
 *   PROBES=1 npx vitest run tests/probes/tilt.probe.test.ts --silent=false
 */

import { test } from 'vitest';
import { ArmKinematics } from '../../src/sim/ik';
import { BASE_MM, tableToWorld } from '../../src/sim/table';
import { nodeScene } from '../scene';

const TILTS = [4, 10, 20, 30, 45, 60, 75, 90];

test('tilt needed', async () => {
  const { mj, model } = await nodeScene();
  const ik = new ArmKinematics(mj, model);
  const names = ['pan', 'lift', 'elbow', 'wrist', 'roll'];
  for (const z of [0, 30, 60, 90]) {
    const row: string[] = [];
    for (let r = 120; r <= 300; r += 20) {
      const [x, y] = tableToWorld(BASE_MM.x, BASE_MM.y - r); // straight out from the base
      const pos: [number, number, number] = [x, y, z / 1000 + 0.015];
      let found = '-';
      for (const t of TILTS) {
        const s = ik.solve({ pos, yaw: null, maxTiltDeg: t });
        if (s.ok) {
          const atLimit = s.q
            .map((v, j) => (Math.abs(v - ik.lo[j]) < 0.02 || Math.abs(v - ik.hi[j]) < 0.02 ? names[j] : ''))
            .filter(Boolean);
          found = `${Math.round(s.angErrDeg)}°${atLimit.length ? ` (${atLimit.join('+')} at limit)` : ''}`;
          break;
        }
      }
      row.push(`r${r}: ${found}`);
    }
    console.log(`bottom at ${z} mm: ${row.join(', ')}`);
  }
  // Which joint stops straight-down close to the base?
  for (const r of [140, 180, 220]) {
    const [x, y] = tableToWorld(BASE_MM.x, BASE_MM.y - r);
    const s = ik.solve({ pos: [x, y, 0.075], yaw: null, maxTiltDeg: 0 });
    console.log(`r${r} z75 straight down: ok=${s.ok} tilt ${s.angErrDeg.toFixed(1)}° pos err ${s.posErrMm.toFixed(1)} mm`,
      s.q.map((v, j) => `${names[j]} ${v.toFixed(2)} [${ik.lo[j].toFixed(2)}, ${ik.hi[j].toFixed(2)}]`).join(' '));
  }
});
