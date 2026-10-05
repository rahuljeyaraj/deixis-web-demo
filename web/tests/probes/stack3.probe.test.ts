import { test } from 'vitest';
import { Arm, type BlockId } from '../../src/sim/arm';
import { nodeScene } from '../scene';

test('three high, step by step', async () => {
  const { mj, model, data } = await nodeScene();
  const arm = new Arm(mj, model, data);
  const run = async (p: Promise<unknown>) => {
    for (let t = 0; arm.busy && t < 30; t += 0.005) { arm.update(); mj.mj_step(model, data); }
    for (let i = 0; i < 100; i++) mj.mj_step(model, data);
    return p;
  };
  const show = (label: string) =>
    console.log(label, (['red', 'green', 'blue'] as BlockId[]).map((id) => {
      const b = arm.block(id);
      return `${id} (${b.tableMm[0].toFixed(1)}, ${b.tableMm[1].toFixed(1)}) top ${b.heightMm.toFixed(1)}`;
    }).join(' | '));
  for (const [gx, gy] of [[300, 90], [250, 60], [330, 50]]) {
    arm.goHomeNow();
    arm.putBlock('red', 180, 120, 0);
    arm.putBlock('green', gx, gy, 0);
    arm.putBlock('blue', 360, 150, 0);
    arm.putBlock('yellow', 470, 240, 0);
    for (let i = 0; i < 100; i++) mj.mj_step(model, data);
    console.log(await run(arm.move('red', { xMm: gx, yMm: gy, zMm: 30 })));
    show('after red on green:');
    const red = arm.block('red');
    console.log(await run(arm.move('blue', { xMm: red.tableMm[0], yMm: red.tableMm[1], zMm: 60 })));
    show('after blue on red:  ');
  }
});
