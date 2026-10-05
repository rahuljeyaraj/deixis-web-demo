/**
 * Blocks close together: random moves next to, onto and between blocks. Does the moved block
 * land, and do the others stay put? (rahul, 6 Oct: stacks next to other blocks.)
 *   PROBES=1 npx vitest run tests/probes/crowd.probe.test.ts --silent=false
 */
import { test } from 'vitest';
import { Arm, BLOCK_IDS, type BlockId } from '../../src/sim/arm';
import { BASE_MM, REACH_MM, onMat } from '../../src/sim/table';
import { blockedBy } from '../../src/flow/things';
import { nodeScene } from '../scene';

function rng(seed: number) {
  return () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
}

test('crowded moves', async () => {
  const { mj, model, data } = await nodeScene();
  const arm = new Arm(mj, model, data);
  const rand = rng(7);
  const watch = Number(process.env.WATCH ?? -1);
  let trial = 0;
  const run = async (p: Promise<{ ok: boolean; reason?: string }>) => {
    let k = 0;
    for (let t = 0; arm.busy && t < 30; t += 0.005, k++) {
      arm.update();
      mj.mj_step(model, data);
      if (trial === watch && k % 20 === 0) {
        const site = arm.ik.fk(arm.armQ());
        const g = data.qpos[arm.blocks.get('red')!.qposAdr - 0]; void g;
        console.log(`t ${t.toFixed(2)} site z ${(site.pos[2] * 1000).toFixed(0)} open ${(Math.atan2(site.open[1], site.open[0]) * 180 / Math.PI).toFixed(0)}° tilt ${(Math.acos(-site.point[2]) * 180 / Math.PI).toFixed(0)}° grip ${data.ctrl[5]?.toFixed(2)} | ` +
          BLOCK_IDS.map((id) => { const b = arm.block(id); return `${id} ${b.tableMm.map((v) => v.toFixed(0)).join(',')} z${b.heightMm.toFixed(0)} yaw ${(b.yaw * 180 / Math.PI).toFixed(0)}`; }).join(' | '));
      }
    }
    for (let i = 0; i < 100; i++) mj.mj_step(model, data);
    return p;
  };
  const counts = { planned: 0, refused: 0, ok: 0, slipped: 0, missed: 0, knocked: 0 };
  for (trial = 0; trial < Number(process.env.TRIALS ?? 60); trial++) {
    arm.goHomeNow();
    const spots: [number, number][] = [];
    while (spots.length < BLOCK_IDS.length) {
      const r = REACH_MM.min + 20 + rand() * (REACH_MM.max - REACH_MM.min - 40);
      const a = (rand() - 0.5) * 2.0;
      const x = BASE_MM.x + r * Math.sin(a);
      const y = BASE_MM.y - r * Math.cos(a);
      if (onMat(x, y) && spots.every(([u, v]) => Math.hypot(u - x, v - y) > 45) &&
          spots.some(([u, v]) => Math.hypot(u - x, v - y) < 70) || (onMat(x, y) && spots.length === 0)) spots.push([x, y]);
    }
    BLOCK_IDS.forEach((id, i) => arm.putBlock(id, spots[i][0], spots[i][1], rand() * Math.PI));
    for (let i = 0; i < 100; i++) mj.mj_step(model, data);
    const mover: BlockId = BLOCK_IDS[Math.floor(rand() * BLOCK_IDS.length)];
    const others = BLOCK_IDS.filter((b) => b !== mover);
    const target = others[Math.floor(rand() * others.length)];
    const t = arm.block(target).tableMm;
    const kind = rand();
    let place;
    let what;
    if (kind < 0.4) {
      const dir = [[40, 0], [-40, 0], [0, 40], [0, -40]][Math.floor(rand() * 4)];
      place = { xMm: t[0] + dir[0], yMm: t[1] + dir[1], zMm: 0 };
      what = `next to ${target}`;
    } else if (kind < 0.8) {
      place = { xMm: t[0], yMm: t[1], zMm: 30 };
      what = `on ${target}`;
    } else {
      const o = arm.block(others.find((b) => b !== target)!).tableMm;
      place = { xMm: (t[0] + o[0]) / 2, yMm: (t[1] + o[1]) / 2, zMm: 0 };
      what = `between ${target} and another`;
    }
    const all = BLOCK_IDS.map((id) => arm.block(id));
    if (!onMat(place.xMm, place.yMm) || (place.zMm === 0 && blockedBy(place.xMm, place.yMm, all, [mover]))) {
      counts.refused++;
      continue;
    }
    const before = new Map(others.map((id) => [id, arm.block(id).tableMm]));
    const plan = arm.canMove(mover, place);
    if (typeof plan === 'string') { counts.refused++; continue; }
    counts.planned++;
    const res = await run(arm.move(mover, place));
    const b = arm.block(mover);
    const off = Math.hypot(b.tableMm[0] - place.xMm, b.tableMm[1] - place.yMm);
    const moved = others.filter((id) => { const p = before.get(id)!; const n = arm.block(id).tableMm; return Math.hypot(n[0] - p[0], n[1] - p[1]) > 2; });
    if (!res.ok) counts.slipped++;
    else if (off > 4 || Math.abs(b.heightMm - place.zMm - 30) > 2) counts.missed++;
    else counts.ok++;
    if (moved.length) counts.knocked++;
    if (!res.ok || off > 4 || moved.length) {
      const near = (p: [number, number], skip: BlockId[]) => others.filter((o) => !skip.includes(o)).map((o) => { const q = before.get(o)!; return `${o} ${Math.hypot(q[0] - p[0], q[1] - p[1]).toFixed(0)}`; }).join(' ');
      const from = spots[BLOCK_IDS.indexOf(mover)];
      console.log(`trial ${trial}: ${mover} ${what}: ${res.ok ? 'ok' : res.reason}, off ${off.toFixed(1)} mm, top ${b.heightMm.toFixed(0)}, knocked ${moved.join(',') || '-'} | from r${Math.hypot(from[0] - BASE_MM.x, from[1] - BASE_MM.y).toFixed(0)} near: ${near(from, [])} | to r${Math.hypot(place.xMm - BASE_MM.x, place.yMm - BASE_MM.y).toFixed(0)} near: ${near([place.xMm, place.yMm], [])}`);
    }
  }
  console.log('crowd:', JSON.stringify(counts));
});
