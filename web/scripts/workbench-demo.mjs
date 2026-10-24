// A clip of the arm being told to do something, driven the way a person will drive it:
// the pointer stays on the thing and a key says what to do with it.
//
//   (start `npm run dev` first)
//   node scripts/workbench-demo.mjs [url] [out folder]
//
// Records the whole thing as a video, because a still cannot show an arm moving. The pointer
// never leaves the thing it is talking about - that is the demo, and it is why the keys exist
// rather than buttons.

import fs from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

const url = process.argv[2] ?? 'http://localhost:5173/workbench.html';
const out = path.resolve(
  process.argv[3] ?? path.join(import.meta.dirname, '..', '..', 'out', 'stage2-demo'),
);
fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(out, { recursive: true });

const browser = await chromium.launch({
  channel: 'chrome',
  args: ['--use-angle=d3d11', '--enable-unsafe-swiftshader', '--mute-audio'],
});
const context = await browser.newContext({
  viewport: { width: 1100, height: 820 },
  recordVideo: { dir: out, size: { width: 1100, height: 820 } },
});
const page = await context.newPage();
page.on('pageerror', (e) => console.log('pageerror:', e.message));
await page.goto(url);
await page.waitForFunction(() => 'rig' in window, null, { timeout: 60000 });
await page.waitForTimeout(1500);

/** Put the real pointer on a part of a thing the camera can see. */
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
  await page.mouse.move(at.x, at.y, { steps: 20 });
  await page.waitForTimeout(500);
}

/** Put the pointer on a bare patch of bench, in bench millimetres. */
async function pointAtBench(xMm, yMm) {
  const at = await page.evaluate(([x, y]) => {
    const { view, THREE, reach } = window;
    view.camera.updateMatrixWorld();
    const world = new THREE.Vector3(y / 1000, (reach.bench.centreX - x) / 1000, 0);
    const s = world.project(view.camera);
    const c = document.querySelector('#scene').getBoundingClientRect();
    return { x: c.left + ((s.x + 1) / 2) * c.width, y: c.top + ((1 - s.y) / 2) * c.height };
  }, [xMm, yMm]);
  await page.mouse.move(at.x, at.y, { steps: 20 });
  await page.waitForTimeout(500);
}

const said = async () => (await page.locator('#doing').innerText()).trim();

/** Press a key and wait until the arm has finished saying what it is doing. */
async function key(k, name) {
  const before = await said();
  await page.keyboard.press(k);
  await page.waitForFunction(
    (was) => {
      const t = document.querySelector('#doing').textContent.trim();
      return t !== was && !t.endsWith('…');
    },
    before,
    { timeout: 60000 },
  );
  await page.waitForTimeout(400);
  const line = await said();
  if (name) await page.screenshot({ path: path.join(out, `${name}.png`) });
  console.log(`  ${k.toUpperCase()}  ${line}`);
  return line;
}

console.log('a clip: the pointer stays on the thing, a key says what to do with it\n');

await pointAt('screwdriver1');
await page.screenshot({ path: path.join(out, '1-pointing.png') });
console.log('  pointing at a screwdriver');
await key('p', '2-picked-up');

await pointAtBench(350, 150);
console.log('  pointing at the bench in front of the person');
await key('d', '3-put-down');

await pointAt('blades');
console.log('  pointing at the tin of blades');
await key('p', '4-holding-the-blades');
await key('o', '5-into-the-pot');

await key('k', '6-parked');

await page.waitForTimeout(800);
await context.close();
await browser.close();

const clip = fs.readdirSync(out).find((f) => f.endsWith('.webm'));
if (clip) {
  fs.renameSync(path.join(out, clip), path.join(out, 'workbench-stage2.webm'));
  console.log(`\n  clip: ${path.join(out, 'workbench-stage2.webm')}`);
}
