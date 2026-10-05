/**
 * The arm's pick and place (plan W3): random blocks, random places on the reach mat. Every
 * move must land within 3 mm of the place, flat, without dropping the block.
 */

import { describe, expect, test } from 'vitest';
import { Arm, BLOCK_IDS, type BlockId } from '../src/sim/arm';
import { BASE_MM, REACH_MM, onMat } from '../src/sim/table';
import { nodeScene } from './scene';

function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

function tiltDeg(q: Float64Array, i: number): number {
  const [, x, y] = [q[i], q[i + 1], q[i + 2], q[i + 3]];
  // The block's z axis against the world's: its zz element is 1 - 2(x^2 + y^2).
  const zz = 1 - 2 * (x * x + y * y);
  // A cube lying on any face is flat, so take the smallest angle to any of its axes.
  return (Math.acos(Math.min(1, Math.abs(zz))) * 180) / Math.PI;
}

async function setup() {
  const { mj, model, data } = await nodeScene();
  const arm = new Arm(mj, model, data);
  const step = (s: number) => {
    for (let i = 0, n = Math.round(s / model.opt.timestep); i < n; i++) {
      arm.update();
      mj.mj_step(model, data);
    }
  };
  const run = async (p: Promise<unknown>) => {
    for (let t = 0; arm.busy && t < 30; t += 0.1) step(0.1);
    step(0.5);
    return p;
  };
  return { mj, model, data, arm, step, run };
}

function spot(rand: () => number, taken: [number, number][]): [number, number] {
  for (;;) {
    const r = REACH_MM.min + 10 + rand() * (REACH_MM.max - REACH_MM.min - 20);
    const a = (rand() - 0.5) * Math.PI * 0.95;
    const x = BASE_MM.x + r * Math.sin(a);
    const y = BASE_MM.y - r * Math.cos(a);
    if (onMat(x, y) && taken.every(([u, v]) => Math.hypot(u - x, v - y) > 60)) return [x, y];
  }
}

describe('pick and place', () => {
  test('50 random moves land within 3 mm, flat', async () => {
    const { data, arm, step, run } = await setup();
    const rand = rng(7);
    const errors: number[] = [];
    let assisted = 0;
    for (let trial = 0; trial < 50; trial++) {
      arm.goHomeNow();
      const taken: [number, number][] = [];
      for (const id of BLOCK_IDS) {
        const [x, y] = spot(rand, taken);
        taken.push([x, y]);
        arm.putBlock(id, x, y, rand() * Math.PI);
      }
      step(0.3);
      const id: BlockId = BLOCK_IDS[Math.floor(rand() * 4)];
      const [px, py] = spot(rand, taken);
      const res = (await run(arm.move(id, { xMm: px, yMm: py, zMm: 0 }))) as {
        ok: boolean;
        assisted?: boolean;
      };
      expect(res.ok, `trial ${trial}: ${JSON.stringify(res)}`).toBe(true);
      if (res.assisted) assisted++;
      const b = arm.block(id);
      const err = Math.hypot(b.tableMm[0] - px, b.tableMm[1] - py);
      errors.push(err);
      expect(err, `trial ${trial}: ${id} landed ${err.toFixed(1)} mm off`).toBeLessThan(3);
      expect(b.heightMm, `trial ${trial}: ${id} not on the table`).toBeCloseTo(30, 0);
      expect(tiltDeg(data.xquat as Float64Array, 4 * arm.blocks.get(id)!.body)).toBeLessThan(5);
    }
    errors.sort((a, b) => a - b);
    console.log(
      `50 moves: median ${errors[25].toFixed(2)} mm, worst ${errors[49].toFixed(2)} mm; grip assisted ${assisted}`,
    );
  }, 600_000);

  test('put that on that: stacks flat', async () => {
    const { arm, step, run } = await setup();
    arm.goHomeNow();
    arm.putBlock('red', 150, 120, 0.4);
    arm.putBlock('green', 320, 110, 0.2);
    arm.putBlock('yellow', 250, 40);
    arm.putBlock('blue', 400, 200);
    step(0.3);
    const g = arm.block('green');
    const res = await run(arm.move('red', { xMm: g.tableMm[0], yMm: g.tableMm[1], zMm: 30 }));
    expect(res).toMatchObject({ ok: true });
    const r = arm.block('red');
    expect(r.heightMm).toBeCloseTo(60, 0);
    expect(Math.hypot(r.tableMm[0] - g.tableMm[0], r.tableMm[1] - g.tableMm[1])).toBeLessThan(3);
  }, 120_000);

  test('out of reach: refuses before moving', async () => {
    const { arm } = await setup();
    arm.goHomeNow();
    arm.putBlock('red', 200, 100);
    expect(arm.canMove('red', { xMm: 10, yMm: 5, zMm: 0 })).toBe('cant_reach'); // a far corner
    expect(arm.canMove('red', { xMm: 250, yMm: 230, zMm: 0 })).toBe('cant_reach'); // at the base
    expect(arm.busy).toBe(false);
  });
});
