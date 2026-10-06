// The look check (measure list M34; web demo items 5 and 10), in a real Chrome.
//   (start `uv run webdemo` with AI=rules, and `npm run dev`, first)
//   node scripts/look-check.mjs [url] [out folder]
//
// 1. Layout at a 1366 x 768 laptop (Chrome's page is about 1366 x 657): after a typed
//    command, is the strip "What the robot heard and saw" on screen without scrolling?
// 2. The colour screenshot test: on each block (red, yellow, blue), the amber dot, the thin
//    amber hover outline and the dashed blue guess, seen normally and through Chrome's
//    colour-blind simulations (protanopia, deuteranopia, tritanopia, achromatopsia).
//    A person looks at the pictures (index.html): on any block where a mark is hard to see,
//    that block changes colour (and the chips with it).
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

const url = process.argv[2] ?? 'http://localhost:5173/';
const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
const out = path.resolve(process.argv[3] ?? path.join(import.meta.dirname, '..', '..', 'out', 'look', stamp));
fs.mkdirSync(out, { recursive: true });

const VISIONS = ['none', 'protanopia', 'deuteranopia', 'tritanopia', 'achromatopsia'];
const BLOCKS = ['red', 'yellow', 'blue'];
const browser = await chromium.launch({ channel: 'chrome', args: ['--use-fake-device-for-media-stream'] });
const report = { url, when: new Date().toISOString(), layout: {}, pictures: [] };

// ---- 1. layout at 1366 x 657
{
  const page = await browser.newPage({ viewport: { width: 1366, height: 657 } });
  page.on('pageerror', (e) => console.log('pageerror:', e.message));
  await page.goto(url + '?start=skip');
  await page.waitForFunction(() => window.__sim, null, { timeout: 60000 });
  await page.click('#type-toggle');
  await page.fill('#type-box', 'put the red one here');
  await page.press('#type-box', 'Enter');
  await page.waitForTimeout(2500);
  const box = await page.locator('#strip').boundingBox();
  const stage = await page.locator('#stage').boundingBox();
  report.layout = {
    viewport: [1366, 657],
    stage: stage && [Math.round(stage.width), Math.round(stage.height)],
    strip_bottom: box && Math.round(box.y + box.height),
    strip_on_screen: !!box && box.y + box.height <= 657,
  };
  await page.screenshot({ path: path.join(out, 'layout-1366x657.png') });
  await page.close();
}

// ---- 2. the marks on each block, through each vision
const page = await browser.newPage({ viewport: { width: 1312, height: 800 } });
const cdp = await page.context().newCDPSession(page);
await page.goto(url + '?start=skip');
await page.waitForFunction(() => window.__sim, null, { timeout: 60000 });
await page.evaluate(() => {
  const { arm, flow } = window.__sim;
  arm.goHomeNow();
  arm.putBlock('red', 180, 100, 0);
  arm.putBlock('yellow', 245, 110, 0);
  arm.putBlock('blue', 300, 80, 0);
  flow.reset();
});
await page.waitForTimeout(1500);
const stage = await page.locator('#stage').boundingBox();
for (const block of BLOCKS) {
  await page.evaluate((b) => window.__sim.markBlock(b), block);
  await page.waitForTimeout(300);
  for (const vision of VISIONS) {
    await cdp.send('Emulation.setEmulatedVisionDeficiency', { type: vision });
    await page.waitForTimeout(150);
    const name = `${block}-${vision}.png`;
    await page.screenshot({ path: path.join(out, name), clip: stage });
    report.pictures.push({ block, vision, file: name });
  }
  await cdp.send('Emulation.setEmulatedVisionDeficiency', { type: 'none' });
}
await browser.close();

const rows = BLOCKS.map(
  (b) => `<tr><th>${b}</th>${VISIONS.map((v) => `<td><img src="${b}-${v}.png" width="420"><br>${v}</td>`).join('')}</tr>`,
).join('\n');
fs.writeFileSync(
  path.join(out, 'index.html'),
  `<!doctype html><meta charset="utf-8"><title>Look check</title>
<h1>Look check (M34)</h1>
<p>On each block: the amber dot (you), the thin amber outline (what the dot is on) and the dashed blue ring (the robot's guess). Is every mark easy to see, in every column? If not on a block, that block changes colour.</p>
<p>Layout at 1366 x 657: the strip ends at y = ${report.layout.strip_bottom} (${report.layout.strip_on_screen ? 'on screen' : 'BELOW THE FOLD'}). <a href="layout-1366x657.png">picture</a></p>
<table>${rows}</table>`,
);
fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 1));
console.log(JSON.stringify(report.layout));
console.log(`pictures: ${report.pictures.length} in ${out}`);
