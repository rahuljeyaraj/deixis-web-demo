// Where can what it is holding be put down? A picture per thing, through the page itself.
//
//   (start `npm run dev` first)
//   node scripts/workbench-where.mjs [url] [out folder]
//
// "Wherever i tried to place thing it keeps on saying there is no way to put it there" (rahul,
// 24 Oct) is what a reach limit feels like when it cannot be seen, and the limits here are real
// and strange: a jar is held upright by a level hand, and a level hand at jar height cannot work
// closer in than 280 mm from the rail, so a jar can go in the bays and along the back of the
// bench and nowhere near the person. The page lays a patch of light on every spot `rig.canPut`
// says yes to. This keeps a picture of that, and the count, so neither can quietly stop being
// true - and so there is something to put in front of him rather than describe.

import fs from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

const url = process.argv[2] ?? 'http://localhost:5173/workbench.html';
const out = path.resolve(
  process.argv[3] ?? path.join(import.meta.dirname, '..', '..', 'out', 'stage2-where'),
);
fs.mkdirSync(out, { recursive: true });

/** One of each kind of hold: from above on the bench, level out of a bay, and a flat tin. */
const TRY = ['hammer', 'jar1', 'blades'];

const browser = await chromium.launch({
  channel: 'chrome',
  args: ['--use-angle=d3d11', '--enable-unsafe-swiftshader', '--mute-audio'],
});
const page = await browser.newPage({ viewport: { width: 1312, height: 900 } });
page.on('pageerror', (e) => console.log('pageerror:', e.message));
await page.goto(`${url}?view=bays`);
await page.waitForFunction(() => 'rig' in window, null, { timeout: 60000 });
await page.waitForTimeout(1500);

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

const lines = [];
for (const id of TRY) {
  // Through the page, not through the console: the patches are lit by the page's own hook when
  // a move settles, so a `rig.take` called from outside never lights them. A real pointer on a
  // part of the thing the camera can really see, and a real key - which is how it is driven,
  // because a hand cannot point and click at once.
  // From whichever view can see it. The close view is square on to the bays and the default one
  // stands back over the whole bench, and nothing in this world is well seen from both.
  let found = false;
  for (const view of ['bays', 'default', 'over']) {
    await page.click(`button[data-view="${view}"]`);
    await page.waitForTimeout(300);
    found = await pointAt(id).then(
      () => true,
      () => false,
    );
    if (found) break;
  }
  if (!found) throw new Error(`cannot see ${id} from any view`);
  await page.keyboard.press('p');
  await page.waitForTimeout(11000);
  // Picked at close quarters, where the pointer can find a part of the thing the camera really
  // sees; photographed from where a person stands, which is where the patches read as a field.
  await page.click('button[data-view="default"]');
  await page.waitForTimeout(400);
  const got = await page.evaluate(() => ({
    holding: window.rig.holding,
    places: window.rig.holding ? window.rig.wherePut(20).length : 0,
    lit: window.view.scene.children.filter((c) => c.isInstancedMesh).map((c) => c.count),
    doing: document.querySelector('#doing')?.textContent ?? '',
  }));
  lines.push(
    `  ${id.padEnd(8)} holding ${String(got.holding).padEnd(8)}` +
      ` ${String(got.places).padStart(4)} places, ${got.lit.join('/')} patches drawn` +
      `   "${got.doing}"`,
  );
  console.log(lines[lines.length - 1]);
  await page.screenshot({ path: path.join(out, `${id}.png`) });
  await page.evaluate(() => window.rig.park());
  await page.waitForTimeout(5000);
  await page.click('button[data-view="bays"]');
  await page.waitForTimeout(400);
}

fs.writeFileSync(
  path.join(out, 'where.txt'),
  ['where a held thing can be put down, through the page:', ...lines, ''].join('\n'),
  'utf8',
);
await browser.close();
