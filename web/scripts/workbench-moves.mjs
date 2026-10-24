// Stage 2's checklist, the page half: does the arm really do it when it is told to, through
// the same buttons and the same pointer a judge will use?
//
//   (start `npm run dev` first)
//   node scripts/workbench-moves.mjs [url] [out folder]
//
// The probe `tests/probes/stage2-body.probe.test.ts` measures the moves against the physics.
// This drives them through the page instead - a real mouse at a real pixel, the real buttons -
// and keeps a picture of each one, which is what stage 2 asks to be shown.

import fs from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

const url = process.argv[2] ?? 'http://localhost:5173/workbench.html';
const out = path.resolve(
  process.argv[3] ?? path.join(import.meta.dirname, '..', '..', 'out', 'stage2-moves'),
);
fs.mkdirSync(out, { recursive: true });

const browser = await chromium.launch({
  channel: 'chrome',
  args: ['--use-angle=d3d11', '--enable-unsafe-swiftshader', '--mute-audio'],
});
const page = await browser.newPage({ viewport: { width: 1312, height: 900 } });
page.on('pageerror', (e) => console.log('pageerror:', e.message));
await page.goto(url);
await page.waitForFunction(() => 'rig' in window, null, { timeout: 60000 });
await page.waitForTimeout(1200);

/** Put the real pointer on a visible part of a thing, the way the spot check does. */
async function pointAt(id) {
  const at = await page.evaluate((want) => {
    const { mj, model, view, THREE, reach } = window;
    const G = mj.mjtObj.mjOBJ_GEOM.value;
    const ids = new Set(reach.things.map((t) => t.id));
    const mine = view.pickable.filter((o) => {
      const name = mj.mj_id2name(model, G, o.userData.geom);
      const base = name ? name.replace(/_\d+$/, '') : '';
      return ids.has(base) && base === want;
    });
    if (!mine.length) return null;
    view.camera.updateMatrixWorld();
    view.scene.updateMatrixWorld();
    const box = new THREE.Box3();
    for (const m of mine) box.union(new THREE.Box3().setFromObject(m));
    const eye = view.camera.position;
    const ray = new THREE.Raycaster();
    const at = new THREE.Vector3();
    const N = 6;
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++)
        for (let k = 0; k < N; k++) {
          const f = (n) => (n + 0.5) / N;
          at.set(
            box.min.x + f(i) * (box.max.x - box.min.x),
            box.min.y + f(j) * (box.max.y - box.min.y),
            box.min.z + f(k) * (box.max.z - box.min.z),
          );
          ray.set(eye, at.clone().sub(eye).normalize());
          const hit = ray.intersectObjects(view.pickable, false)[0];
          if (!hit || !mine.includes(hit.object)) continue;
          const s = hit.point.clone().project(view.camera);
          if (Math.abs(s.x) > 0.9 || Math.abs(s.y) > 0.9) continue;
          const c = document.querySelector('#scene').getBoundingClientRect();
          return { x: c.left + ((s.x + 1) / 2) * c.width, y: c.top + ((1 - s.y) / 2) * c.height };
        }
    return null;
  }, id);
  if (!at) throw new Error(`cannot see ${id} from here`);
  await page.mouse.move(at.x, at.y);
  await page.waitForTimeout(150);
  return at;
}

/** Put the pointer on a bare patch of bench, given in bench millimetres. */
async function pointAtBench(xMm, yMm) {
  const at = await page.evaluate(([x, y]) => {
    const { view, THREE, reach } = window;
    view.camera.updateMatrixWorld();
    const world = new THREE.Vector3(y / 1000, (reach.bench.centreX - x) / 1000, 0);
    const s = world.project(view.camera);
    const c = document.querySelector('#scene').getBoundingClientRect();
    return { x: c.left + ((s.x + 1) / 2) * c.width, y: c.top + ((1 - s.y) / 2) * c.height };
  }, [xMm, yMm]);
  await page.mouse.move(at.x, at.y);
  await page.waitForTimeout(150);
  return at;
}

const said = async () => (await page.locator('#doing').innerText()).trim();

/** Press a button and wait until the arm has stopped saying it is busy. */
async function press(id, name) {
  const before = await said();
  await page.click(`#${id}`);
  await page.waitForFunction(
    (was) => {
      const t = document.querySelector('#doing').textContent.trim();
      return t !== was && !t.endsWith('…');
    },
    before,
    { timeout: 60000 },
  );
  const line = await said();
  await page.screenshot({ path: path.join(out, `${name}.png`) });
  return line;
}

const rows = [];
const step = async (what, fn) => {
  const line = await fn();
  rows.push([what, line]);
  console.log(`  ${what.padEnd(34)} ${line}`);
};

console.log('stage 2, through the page: the arm told to do things with a mouse\n');

await step('pick up the middle jar', async () => {
  await pointAt('jar2');
  return press('take', '1-holding-a-jar');
});
await step('put it on the bench in front', async () => {
  await pointAtBench(300, 250);
  return press('put', '2-put-on-the-bench');
});
await step('pick up the tin of blades', async () => {
  await pointAt('blades');
  return press('take', '3-holding-the-blades');
});
await step('drop it in the pot', async () => press('pot', '4-in-the-pot'));
await step('pick up the bottle of oil', async () => {
  await pointAt('oil');
  return press('take', '5-holding-the-oil');
});
await step('put it in the middle bay', async () => {
  await pointAt('tin1');
  return press('put', '6-refused-too-tall');
});
await step('park', async () => press('park', '7-parked'));

fs.writeFileSync(
  path.join(out, 'said.txt'),
  rows.map(([a, b]) => `${a.padEnd(34)} ${b}`).join('\n') + '\n',
  'utf8',
);
console.log(`\n  pictures in ${out}`);
await browser.close();
