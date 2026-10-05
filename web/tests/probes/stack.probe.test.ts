/**
 * Where can the arm put a block on a stack? For places on the mat, at 0, 30 and 60 mm up
 * (on the table, on one block, on two), the share that `canMove` plans.
 *   PROBES=1 npx vitest run tests/probes/stack.probe.test.ts --silent=false
 */

import { test } from 'vitest';
import { Arm } from '../../src/sim/arm';
import { BASE_MM, REACH_MM, onMat } from '../../src/sim/table';
import { nodeScene } from '../scene';

test('stack reach', async () => {
  const { mj, model, data } = await nodeScene();
  const arm = new Arm(mj, model, data);
  arm.goHomeNow();
  arm.putBlock('yellow', 470, 240);
  arm.putBlock('green', 30, 240);
  arm.putBlock('blue', 20, 230);
  for (let i = 0; i < 100; i++) mj.mj_step(model, data);
  for (const z of [0, 30, 60]) {
    const rows: string[] = [];
    for (let r = REACH_MM.min; r <= REACH_MM.max; r += 25) {
      let ok = 0;
      let n = 0;
      for (let a = -1.2; a <= 1.21; a += 0.3) {
        const x = BASE_MM.x + r * Math.sin(a);
        const y = BASE_MM.y - r * Math.cos(a);
        if (!onMat(x, y)) continue;
        // The block to move sits elsewhere on the mat, flat.
        arm.putBlock('red', BASE_MM.x - 150 * Math.sin(a), BASE_MM.y - 150 * Math.cos(a), 0);
        mj.mj_forward(model, data);
        n++;
        if (typeof arm.canMove('red', { xMm: x, yMm: y, zMm: z }) !== 'string') ok++;
      }
      rows.push(`r ${r}: ${ok}/${n}`);
    }
    console.log(`z ${z} mm: ${rows.join(', ')}`);
  }
});
