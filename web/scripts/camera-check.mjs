// Checks the camera path loads (development only): Chrome's fake camera (a test pattern, no
// hand), the "Use my camera" button, MediaPipe's runtime and model from our own server.
//   node scripts/camera-check.mjs [url]
import { chromium } from '@playwright/test';

const [url = 'http://localhost:5173/'] = process.argv.slice(2);
const browser = await chromium.launch({
  channel: 'chrome',
  args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
});
const context = await browser.newContext({ viewport: { width: 1312, height: 800 } });
await context.grantPermissions(['camera'], { origin: new URL(url).origin });
const page = await context.newPage();
page.on('pageerror', (e) => console.log('pageerror:', e.message));
page.on('console', (m) => ['error', 'warning'].includes(m.type()) && console.log(`console.${m.type()}:`, m.text().slice(0, 200)));
page.on('response', (r) => r.status() >= 400 && console.log('http', r.status(), r.url()));
await page.goto(url);
await page.waitForFunction(() => window.__sim, null, { timeout: 30000 });
const t0 = Date.now();
await page.click('#camera-btn');
await page.waitForFunction(() => window.__sim.pointer.frames > 30, null, { timeout: 60000 });
const info = await page.evaluate(() => {
  const tr = window.__sim.tracker();
  const s = window.__sim.pointer.samples;
  const latest = s.latest;
  return { delegate: tr?.delegate, frames: window.__sim.pointer.frames, samples: s.length, latestT: latest?.t, now: performance.now() / 1000 };
});
console.log(`camera on in ${((Date.now() - t0) / 1000).toFixed(1)} s:`, JSON.stringify(info));
await page.waitForTimeout(1000);
const fps = await page.evaluate(async () => {
  const a = window.__sim.pointer.frames;
  await new Promise((r) => setTimeout(r, 2000));
  return (window.__sim.pointer.frames - a) / 2;
});
console.log(`hand frames per second: ${fps}`);
await page.screenshot({ path: '../out/camera-check.png' });
console.log('status:', await page.locator('#status').textContent());
await browser.close();
