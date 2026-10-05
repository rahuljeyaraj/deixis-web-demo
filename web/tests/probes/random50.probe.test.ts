/**
 * The arm test's 50 random moves, logged (WATCH=n traces move n).
 *   PROBES=1 WATCH=31 npx vitest run tests/probes/random50.probe.test.ts --silent=false
 */
import { test } from 'vitest';
import { Arm, BLOCK_IDS, type BlockId } from '../../src/sim/arm';
import { BASE_MM, REACH_MM, onMat } from '../../src/sim/table';
import { nodeScene } from '../scene';

function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
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

test('random 50, logged', async () => {
  const { mj, model, data } = await nodeScene();
  const arm = new Arm(mj, model, data);
  const watch = Number(process.env.WATCH ?? -1);
  const rand = rng(7);
  for (let trial = 0; trial < 50; trial++) {
    arm.goHomeNow();
    const taken: [number, number][] = [];
    for (const id of BLOCK_IDS) {
      const [x, y] = spot(rand, taken);
      taken.push([x, y]);
      arm.putBlock(id, x, y, rand() * Math.PI);
    }
    for (let i = 0; i < 60; i++) { arm.update(); mj.mj_step(model, data); }
    const id: BlockId = BLOCK_IDS[Math.floor(rand() * BLOCK_IDS.length)];
    const [px, py] = spot(rand, taken);
    const from = arm.block(id);
    const res = arm.move(id, { xMm: px, yMm: py, zMm: 0 });
    let k = 0;
    for (let t = 0; arm.busy && t < 30; t += 0.005, k++) {
      arm.update();
      mj.mj_step(model, data);
      if (trial === watch && k === 320) {
        const cs = data.contact;
        for (let i = 0; i < data.ncon; i++) {
          const c = cs.get(i)!;
          const bo = model.geom_bodyid as Int32Array;
          const nm = (g: number) => `${mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, bo[g])}/${mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM.value, g) ?? g}`;
          console.log(`contact ${nm(c.geom1)} - ${nm(c.geom2)} dist ${(c.dist * 1000).toFixed(1)}`);
          c.delete();
        }
        cs.delete();
        const plan = 0; void plan;
      }
      if (trial === watch && k % 20 === 0) {
        const p = arm.ik.fk(arm.armQ());
        const b = arm.block(id);
        console.log(`t ${t.toFixed(2)} z ${(p.pos[2] * 1000).toFixed(0)} open ${(Math.atan2(p.open[1], p.open[0]) * 180 / Math.PI).toFixed(0)} tilt ${(Math.acos(-p.point[2]) * 180 / Math.PI).toFixed(0)} grip ${data.ctrl[5].toFixed(2)} q ${arm.armQ().map((v) => v.toFixed(2)).join(' ')} | ${id} ${b.tableMm.map((v) => v.toFixed(0))} z${b.heightMm.toFixed(0)} yaw ${(b.yaw * 180 / Math.PI).toFixed(0)} ncon ${data.ncon}`);
      }
    }
    for (let i = 0; i < 100; i++) mj.mj_step(model, data);
    const r = await res;
    const b = arm.block(id);
    const err = Math.hypot(b.tableMm[0] - px, b.tableMm[1] - py);
    if (!r.ok || err > 3 || trial === watch) {
      const rr = (x: number, y: number) => Math.hypot(x - BASE_MM.x, y - BASE_MM.y).toFixed(0);
      console.log(`trial ${trial}: ${id} ${JSON.stringify(r)} err ${err.toFixed(1)} | from ${from.tableMm.map((v) => v.toFixed(0))} r${rr(...from.tableMm)} yaw ${(from.yaw * 180 / Math.PI).toFixed(0)} to ${px.toFixed(0)},${py.toFixed(0)} r${rr(px, py)} | others ${BLOCK_IDS.filter((o) => o !== id).map((o) => { const q = arm.block(o).tableMm; return `${o} ${q.map((v) => v.toFixed(0))} d${Math.hypot(q[0] - from.tableMm[0], q[1] - from.tableMm[1]).toFixed(0)}/${Math.hypot(q[0] - px, q[1] - py).toFixed(0)}`; }).join(' ')}`);
    }
  }
});
