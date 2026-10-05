/**
 * How close to the base can the arm pick a block cleanly (not hitting its own shoulder)?
 * For each distance, blocks at 9 angles and 4 turns: the share with a plan whose arm sinks
 * at most 1 mm into itself, and the deepest touch of the plans with the default tolerance.
 *   PROBES=1 npx vitest run tests/probes/inner.probe.test.ts --silent=false
 */
import { test } from 'vitest';
import { Arm } from '../../src/sim/arm';
import { BASE_MM, onMat } from '../../src/sim/table';
import { nodeScene } from '../scene';

test('inner edge', async () => {
  const { mj, model, data } = await nodeScene();
  const arm = new Arm(mj, model, data);
  arm.goHomeNow();
  arm.putBlock('yellow', 470, 240);
  arm.putBlock('blue', 30, 240);
  for (let r = 125; r <= 170; r += 5) {
    let n = 0;
    let clean = 0;
    let worst = 0;
    for (let a = -1.2; a <= 1.21; a += 0.3) {
      for (const yaw of [0, 0.4, 0.8, 1.2]) {
        const x = BASE_MM.x + r * Math.sin(a);
        const y = BASE_MM.y - r * Math.cos(a);
        if (y > BASE_MM.y) continue;
        arm.putBlock('red', x, y, yaw);
        mj.mj_forward(model, data);
        n++;
        // The place: straight out at 220 mm, clear of everything.
        const plan = arm.canMove('red', { xMm: BASE_MM.x, yMm: BASE_MM.y - 220, zMm: 0 });
        if (typeof plan === 'string') continue;
        let deep = 0;
        for (const s of plan) deep = Math.max(deep, arm.ik.selfDepth(s.to.slice(0, 5), s.to[5]));
        worst = Math.max(worst, deep);
        if (deep <= 0.001) clean++;
      }
    }
    console.log(`r ${r}: clean ${clean}/${n}, deepest planned touch ${(worst * 1000).toFixed(1)} mm${onMat(BASE_MM.x, BASE_MM.y - r) ? '' : ' (off the mat)'}`);
  }
});
