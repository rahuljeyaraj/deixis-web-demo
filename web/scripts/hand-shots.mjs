// Screenshots of the finger pointer with the made-up hand (development only):
//   node scripts/hand-shots.mjs [url] [outdir]
import { chromium } from '@playwright/test';

const [url = 'http://localhost:5173/?hands=demo', out = '../out'] = process.argv.slice(2);
const browser = await chromium.launch({ channel: 'chrome' });
const page = await browser.newPage({ viewport: { width: 1312, height: 800 } });
page.on('pageerror', (e) => console.log('pageerror:', e.message));
page.on('console', (m) => m.type() === 'error' && console.log('console:', m.text()));
await page.goto(url);
await page.waitForFunction(() => window.__sim?.pointer.frames > 10, null, { timeout: 30000 });

// Wait for the dot to hold on the red block, then on the empty spot.
await page.waitForFunction(() => window.__sim.pointer.hit?.block === 'red', null, { timeout: 15000 });
await page.waitForTimeout(500);
await page.screenshot({ path: `${out}/hand-on-red.png` });
const onRed = await page.evaluate(() => ({ hit: window.__sim.pointer.hit, red: window.__sim.arm.block('red').tableMm }));
await page.waitForFunction(() => {
  const h = window.__sim.pointer.hit;
  return h && !h.block && window.__sim.pointer.samples.latest.t > 0;
}, null, { timeout: 15000, polling: 50 });
await page.waitForTimeout(700);
await page.screenshot({ path: `${out}/hand-on-spot.png` });
const stats = await page.evaluate(() => {
  const p = window.__sim.pointer;
  const s = p.samples.between(0, 1e9);
  return { n: s.length, pointing: s.filter((x) => x.pointing).length, withPoint: s.filter((x) => x.point).length, last: s[s.length - 1] };
});
console.log('on red:', JSON.stringify(onRed));
console.log('samples:', JSON.stringify(stats));
await browser.close();
