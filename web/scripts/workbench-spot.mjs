// Stage 1's checklist, the pointer half: does the laser spot sit correctly on a flat board, a
// curved jar and a hook, and is the name under it right for every thing in the world?
//
//   (start `npm run dev` first)
//   node scripts/workbench-spot.mjs [url] [out folder]
//
// It does not move a mouse at a guessed pixel. For each target it asks the page where that
// geom is in the world, projects it through the page's own camera to a point on the canvas,
// and moves the real pointer there - so a miss is a miss in the thing being tested rather than
// in the test. Each target gets a cropped picture and the line the page wrote underneath.

import fs from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

const url = process.argv[2] ?? 'http://localhost:5173/workbench.html';
const out = path.resolve(process.argv[3] ?? path.join(import.meta.dirname, '..', '..', 'out', 'stage1-spot'));
fs.mkdirSync(out, { recursive: true });

// Each target names a geom in the scene and what the page ought to call it. The three the
// checklist asks for by name are first: a flat board, a curved jar, a hook.
const TARGETS = [
  ['shelf_board0', 'the board over the bottom bay', 'a flat board'],
  ['jar2', 'a jar of screws', 'a curved jar'],
  ['mug1_5', 'a mug', "a bead of the handle - part of the mug"],
  ['shelf_board2', 'the top of the shelf unit', 'the lid of the unit'],
  ['bench', 'the bench top', 'the bench itself'],
  ['offcut1', 'an offcut of wood', 'a flat top, straight on'],
  ['oil', 'a bottle of oil', 'a tall curved bottle'],
  ['pot', 'a pot of loose screws', 'the rim of the pot'],
  ['odds', 'a box of odds and ends', 'inside the top bay'],
  ['hammer_2', 'a hammer', "a part, not the thing's own geom"],
  ['glue1_3', 'a tube of glue, capped', 'the cap, which is part of the tube'],
  ['tape', 'a roll of masking tape', 'a roll lying flat'],
  ['shelf_left', 'the side of the shelf unit', 'an upright'],
  ['tin2', 'a tin of varnish', 'one of three that are the same'],
  ['clamp_1', 'a G clamp', 'the jaw of the clamp'],
  ['pliers_3', 'a pair of pliers', 'the pivot'],
  ['knife_3', 'a craft knife', 'the blade'],
  ['screwdriver1_2', 'a screwdriver', 'the shaft'],
];

const browser = await chromium.launch({
  channel: 'chrome',
  args: ['--use-angle=d3d11', '--enable-unsafe-swiftshader', '--mute-audio'],
});
const page = await browser.newPage({ viewport: { width: 1312, height: 900 } });
page.on('pageerror', (e) => console.log('pageerror:', e.message));
await page.goto(url);
await page.waitForFunction(() => 'view' in window && 'model' in window, null, { timeout: 60000 });
await page.waitForTimeout(1500);

const rows = [];
for (const [geom, want, why] of TARGETS) {
  // A point on that geom the camera can really see, projected onto the canvas.
  //
  // Not its centre: a geom's centre is often behind something else - the middle board is
  // behind the box standing on it, the bench is behind the offcut - and aiming there tests
  // nothing but the aim. So points are sampled over the geom's own box and the first one the
  // page's own ray reaches is used.
  const at = await page.evaluate((name) => {
    const { mj, model, view, THREE } = window;
    const id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM.value, name);
    if (id < 0) return null;
    const mesh = view.pickable.find((o) => o.userData.geom === id);
    if (!mesh) return null;
    view.camera.updateMatrixWorld();
    view.scene.updateMatrixWorld();
    const box = new THREE.Box3().setFromObject(mesh);
    const eye = view.camera.position;
    const ray = new THREE.Raycaster();
    const at = new THREE.Vector3();
    // The visible point nearest the middle of the geom, so the spot lands on the body of a
    // jar rather than on the sliver of its foot that happens to be sampled first.
    const middle = box.getCenter(new THREE.Vector3());
    const N = 6;
    let best = null;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        for (let k = 0; k < N; k++) {
          const f = (n) => (n + 0.5) / N;
          at.set(
            box.min.x + f(i) * (box.max.x - box.min.x),
            box.min.y + f(j) * (box.max.y - box.min.y),
            box.min.z + f(k) * (box.max.z - box.min.z),
          );
          ray.set(eye, at.clone().sub(eye).normalize());
          const hit = ray.intersectObjects(view.pickable, false)[0];
          if (!hit || hit.object.userData.geom !== id) continue;
          const s = hit.point.clone().project(view.camera);
          // Inside the canvas, and not right at its edge: a point that projects off it moves
          // the pointer out of the page, which clears the spot and reads as a miss.
          if (Math.abs(s.x) > 0.92 || Math.abs(s.y) > 0.92) continue;
          const off = hit.point.distanceTo(middle);
          if (!best || off < best.off) best = { off, s };
        }
      }
    }
    if (!best) return null;
    const canvas = document.querySelector('#scene').getBoundingClientRect();
    return {
      x: canvas.left + ((best.s.x + 1) / 2) * canvas.width,
      y: canvas.top + ((1 - best.s.y) / 2) * canvas.height,
      canvas: { x: canvas.left, y: canvas.top, w: canvas.width, h: canvas.height },
    };
  }, geom);
  if (!at) {
    rows.push({ geom, why, got: 'NOT VISIBLE FROM HERE', want, ok: false });
    continue;
  }
  await page.mouse.move(at.x, at.y);
  await page.waitForTimeout(220);
  const got = (await page.locator('#under').innerText()).trim();
  const name = got.split('·')[0].trim();
  const ok = want === null ? true : name === want;
  rows.push({ geom, why, got, want, ok });

  // A crop round the spot, big enough to see how the circle lies on the surface.
  const half = 150;
  await page.screenshot({
    path: path.join(out, `${geom}.png`),
    clip: {
      x: Math.max(at.canvas.x, at.x - half),
      y: Math.max(at.canvas.y, at.y - half),
      width: Math.min(2 * half, at.canvas.x + at.canvas.w - (at.x - half)),
      height: Math.min(2 * half, at.canvas.y + at.canvas.h - (at.y - half)),
    },
  });
}

// One picture with the spot on a jar, full size, for the record.
await page.screenshot({ path: path.join(out, 'page.png') });

const report = [
  "stage 1, the pointer: the laser spot and the name under it",
  `  ${url}`,
  '',
  ...rows.map(
    (r) =>
      `  ${r.ok ? 'ok  ' : 'WRONG'} ${r.geom.padEnd(16)} ${String(r.got).padEnd(44)} (${r.why})`,
  ),
  '',
  `  ${rows.filter((r) => r.ok).length} of ${rows.length} named as expected.`,
  `  pictures in ${out}`,
];
fs.writeFileSync(path.join(out, 'report.txt'), report.join('\n') + '\n', 'utf8');
console.log(report.join('\n'));
await browser.close();
process.exit(rows.every((r) => r.ok) ? 0 : 1);
