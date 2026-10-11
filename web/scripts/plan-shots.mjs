// Pictures of the live wiring panel and the plan strip (advanced-commands-plan.md 5.1, 5.2).
//   (start `uv run webdemo` and `npm run dev` first, or point this at the built page)
//   node scripts/plan-shots.mjs [url] [out folder]
//
// 1. A typed command, then the whole page at a 1366 x 768 laptop and at 1512 wide: the panel
//    fills in from the server's real trace, under the scene, the full width of the page.
// 2. The plan strip, drawn from a made-up plan (the strip is a pure function of one, so it
//    can be checked without the model), with a step done, one running and one still to do,
//    and again as a re-planned strip with a step that could not be done.
// 3. A narrow window (860 px), where the rows stack: the panel is never folded away.
//
// It measures what a person has to read: whether any two lane names overlap, how wide the
// column of words is, and how many rows wrap onto a second line. A layout that pushes the
// controls off a 1366 x 768 laptop, or crushes the names together, fails here and not on the
// day (rahul, 11 Oct: "it is all crushed together and it is very difficult to read").
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

const url = process.argv[2] ?? 'http://localhost:5173/';
const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
const out = path.resolve(process.argv[3] ?? path.join(import.meta.dirname, '..', '..', 'out', 'plan', stamp));
fs.mkdirSync(out, { recursive: true });

const browser = await chromium.launch({
  channel: 'chrome',
  // --mute-audio and ?quiet: the robot writes but never speaks while shots are taken.
  args: ['--use-angle=d3d11', '--enable-unsafe-swiftshader', '--use-fake-device-for-media-stream',
    '--mute-audio'],
});
const report = { url, when: new Date().toISOString(), shots: [] };

/** A plan as the flow builds one, for the strip alone. */
const FAKE_PLAN = {
  replanned: false,
  steps: [
    { label: 'red beside green', state: 'done' },
    { label: 'blue where the other one was', state: 'doing' },
    { label: 'red where the other one was', state: 'todo' },
  ].map((s) => ({ ...s, call: { name: 'move', args: {} }, resolved: null, live: false, move: null })),
};
const FAKE_REPLAN = {
  replanned: true,
  steps: [
    { label: 'blue onto green', state: 'failed', why: "There's already a block on that one." },
    { label: 'blue into the blue square', state: 'todo' },
  ].map((s) => ({ ...s, call: { name: 'move', args: {} }, resolved: null, live: false, move: null })),
};

async function open(width, height) {
  const page = await browser.newPage({ viewport: { width, height } });
  page.on('pageerror', (e) => console.log('pageerror:', e.message));
  page.on('console', (m) => m.type() === 'error' && console.log('console error:', m.text()));
  await page.goto(url + '?start=skip&hands=demo&quiet=1');
  await page.waitForFunction(() => window.__sim, null, { timeout: 60000 });
  return page;
}

/** Is the diagram readable? The things a person's eye actually trips over. */
async function readable(page) {
  return page.evaluate(() => {
    const box = (el) => el.getBoundingClientRect();
    // The ink, not the box: each name is centred in a box one lane wide, so touching boxes
    // mean nothing. What matters is whether the letters of two names run together.
    const ink = (node) => {
      const r = document.createRange();
      r.selectNodeContents(node);
      return r.getBoundingClientRect();
    };
    const lanes = [...document.querySelectorAll('.wiring-lane')].map((el) => {
      // A `small` that is hidden at this width measures 0 x 0 at the origin: not ink.
      const parts = [ink(el.firstChild), ...[...el.querySelectorAll('small')].map(ink)].filter(
        (r) => r.width > 0,
      );
      return {
        name: el.firstChild.textContent,
        left: Math.round(Math.min(...parts.map((p) => p.left))),
        right: Math.round(Math.max(...parts.map((p) => p.right))),
      };
    });
    lanes.sort((a, b) => a.left - b.left);
    const collide = [];
    for (let i = 1; i < lanes.length; i++) {
      // Six clear pixels between two names, or they read as one word.
      if (lanes[i].left < lanes[i - 1].right + 6) collide.push(`${lanes[i - 1].name} / ${lanes[i].name}`);
    }
    const rows = [...document.querySelectorAll('.wiring-row')];
    const says = rows.map((r) => r.querySelector('.wiring-say')).filter(Boolean);
    const sayWidth = says.length ? Math.round(box(says[0]).width) : null;
    // A row of one line is about 20 px; anything taller has wrapped.
    const wrapped = says.filter((el) => box(el).height > 24).length;
    const starts = new Set(says.map((el) => Math.round(box(el).left)));
    return {
      lanes: lanes.map((l) => l.name),
      collide,
      sayWidth,
      wrapped,
      rows: rows.length,
      wordsStartAtOneX: starts.size <= 1,
      fontPx: says.length ? getComputedStyle(says[0]).fontSize : null,
    };
  });
}

async function shot(page, name, note) {
  const png = path.join(out, `${name}.png`);
  await page.screenshot({ path: png, fullPage: true });
  const boxes = {};
  for (const sel of ['#stage', '#controls', '#wiring', '#plan-strip', '#strip', '.wiring-rows', '#see-inside']) {
    const el = page.locator(sel);
    const box = (await el.count()) ? await el.boundingBox() : null;
    boxes[sel] = box && [Math.round(box.x), Math.round(box.y), Math.round(box.width), Math.round(box.height)];
  }
  const read = await readable(page);
  report.shots.push({ name, note, viewport: page.viewportSize(), boxes, read });
  console.log(`${name}: ${note}`);
  for (const [sel, box] of Object.entries(boxes)) console.log(`   ${sel}: ${box ? box.join(', ') : 'not shown'}`);
  console.log(`   ${read.rows} rows, words ${read.sayWidth} px wide at ${read.fontPx}, ${read.wrapped} wrapped`);
  if (read.collide.length) console.log(`   LANE NAMES COLLIDE: ${read.collide.join(', ')}`);
  if (!read.wordsStartAtOneX) console.log('   THE WORDS DO NOT START AT ONE X');
  // The controls must stay on the first screen: the diagram is below it, on purpose.
  const h = page.viewportSize().height;
  if (boxes['#controls'] && boxes['#controls'][1] + boxes['#controls'][3] > h) {
    console.log(`   CONTROLS OFF THE FIRST SCREEN (bottom ${boxes['#controls'][1] + boxes['#controls'][3]} > ${h})`);
  }
}

/** One typed command, through the real server (the word matcher is enough for the trace). */
async function command(page, text) {
  await page.click('#type-toggle');
  await page.fill('#type-box', text);
  await page.press('#type-box', 'Enter');
  await page.waitForTimeout(2500);
}

async function drawPlan(page, plan) {
  await page.evaluate((plan) => window.__sim.plan(plan), plan);
  await page.waitForTimeout(150);
}

for (const [width, height, name] of [
  [1366, 657, 'laptop-1366'],
  [1512, 860, 'wide-1512'],
  [860, 900, 'narrow-860'],
]) {
  const page = await open(width, height);
  await command(page, 'put the red one here');
  await shot(page, `${name}-command`, 'a typed command, and the panel filled from its trace');
  await drawPlan(page, FAKE_PLAN);
  await shot(page, `${name}-plan`, 'the plan strip: one step done, one running, one to do');
  await drawPlan(page, FAKE_REPLAN);
  await shot(page, `${name}-replan`, 'a step that could not be done, and the plan that came instead');
  await page.close();
}

fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 1));
console.log(`\n${out}`);
await browser.close();
