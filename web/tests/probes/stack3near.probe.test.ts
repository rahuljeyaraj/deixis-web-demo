/**
 * Three high, near and far, run in the physics: does each stack stand, flat?
 *   PROBES=1 npx vitest run tests/probes/stack3near.probe.test.ts --silent=false
 */
import { test } from 'vitest';
import { Arm } from '../../src/sim/arm';
import { BASE_MM } from '../../src/sim/table';
import { nodeScene } from '../scene';

test('three high near and far', async () => {
  const { mj, model, data } = await nodeScene();
  const arm = new Arm(mj, model, data);
  const run = async (p: Promise<unknown>) => {
    for (let t = 0; arm.busy && t < 30; t += 0.005) { arm.update(); mj.mj_step(model, data); }
    for (let i = 0; i < 200; i++) mj.mj_step(model, data);
    return p;
  };
  const tilt = (id: 'red' | 'green' | 'blue') => {
    const b = arm.blocks.get(id)!;
    const q = data.qpos as Float64Array;
    const [w, x, y] = [q[b.qposAdr + 3], q[b.qposAdr + 4], q[b.qposAdr + 5]];
    // angle between the block's z axis and up
    const zz = 1 - 2 * (x * x + y * y);
    return (Math.acos(Math.min(1, Math.max(-1, zz))) * 180) / Math.PI;
  };
  for (const [r, a] of [[150, 0], [165, 0.5], [180, -0.6], [200, 0.9], [230, 0], [260, -0.4], [275, 0.2]]) {
    const gx = BASE_MM.x + r * Math.sin(a);
    const gy = BASE_MM.y - r * Math.cos(a);
    arm.goHomeNow();
    arm.putBlock('green', gx, gy, 0.3);
    arm.putBlock('red', BASE_MM.x + 200 * Math.sin(a > 0 ? a - 0.7 : a + 0.7), BASE_MM.y - 200 * Math.cos(a > 0 ? a - 0.7 : a + 0.7), 0);
    arm.putBlock('blue', BASE_MM.x + 160 * Math.sin(a > 0 ? a - 1.2 : a + 1.2), BASE_MM.y - 160 * Math.cos(a > 0 ? a - 1.2 : a + 1.2), 1.0);
    arm.putBlock('yellow', 470, 240, 0);
    for (let i = 0; i < 100; i++) mj.mj_step(model, data);
    const r1 = (await run(arm.move('red', { xMm: gx, yMm: gy, zMm: 30 }))) as { ok: boolean; reason?: string };
    const red = arm.block('red');
    const r2 = (await run(arm.move('blue', { xMm: red.tableMm[0], yMm: red.tableMm[1], zMm: 60 }))) as { ok: boolean; reason?: string };
    const g = arm.block('green'), rd = arm.block('red'), bl = arm.block('blue');
    const off = (b: typeof g) => Math.hypot(b.tableMm[0] - gx, b.tableMm[1] - gy).toFixed(1);
    console.log(`r${r} a${a}: red ${r1.ok ? 'ok' : r1.reason}, blue ${r2.ok ? 'ok' : r2.reason} | tops ${g.heightMm.toFixed(0)}/${rd.heightMm.toFixed(0)}/${bl.heightMm.toFixed(0)} off ${off(g)}/${off(rd)}/${off(bl)} mm, tilt ${tilt('green').toFixed(1)}/${tilt('red').toFixed(1)}/${tilt('blue').toFixed(1)}°`);
  }
});
