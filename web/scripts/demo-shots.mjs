// Screenshots of the mouse test for a quick look (development only):
//   node scripts/demo-shots.mjs [url] [outdir]
import { chromium } from '@playwright/test';

const [url = 'http://localhost:5173/?start=skip', out = '../out'] = process.argv.slice(2);
const browser = await chromium.launch({ channel: 'chrome' });
const page = await browser.newPage({ viewport: { width: 1312, height: 800 } });
page.on('pageerror', (e) => console.log('pageerror:', e.message));
page.on('console', (m) => m.type() === 'error' && console.log('console:', m.text()));
await page.goto(url);
await page.waitForFunction(() => window.__sim, null, { timeout: 30000 });
await page.waitForTimeout(800);
await page.screenshot({ path: `${out}/a-start.png` });

// Screen position of a table point (mm, height mm) in page pixels.
const screenOf = (x, y, z = 15) =>
  page.evaluate(
    ([x, y, z]) => {
      const { view } = window.__sim;
      const c = view.canvas.getBoundingClientRect();
      const v = view.camera.position.clone();
      v.set((250 - y) / 1000, (x - 250) / 1000, z / 1000).project(view.camera);
      return [c.left + ((v.x + 1) / 2) * c.width, c.top + ((1 - v.y) / 2) * c.height];
    },
    [x, y, z],
  );
const red = await page.evaluate(() => window.__sim.arm.block('red').tableMm);
const [rx, ry] = await screenOf(red[0], red[1], 30);
await page.mouse.click(rx, ry);
const target = red[0] < 250 ? [330, 60] : [170, 60];
const [tx, ty] = await screenOf(target[0], target[1], 0);
await page.mouse.click(tx, ty);
await page.waitForTimeout(400);
await page.screenshot({ path: `${out}/b0-chosen.png` });
await page.waitForTimeout(2600);
await page.screenshot({ path: `${out}/b-moving.png` });
await page.waitForFunction(() => !window.__sim.arm.busy, null, { timeout: 30000 });
await page.waitForTimeout(500);
await page.screenshot({ path: `${out}/c-done.png` });
const after = await page.evaluate(() => window.__sim.arm.block('red').tableMm);
console.log('red', red.map((v) => v.toFixed(0)), '->', after.map((v) => v.toFixed(1)), 'target', target);
console.log('status:', await page.locator('#status').textContent());
await browser.close();
