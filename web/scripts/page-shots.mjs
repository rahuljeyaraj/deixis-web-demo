// The page's states in a real Chrome (plan W10; development only: `npm run dev` and
// `uv run webdemo` must run). No microphone allowed first (so the page offers typing), then
// the microphone but no camera (so the mouse points).
//   node scripts/page-shots.mjs [url]
// Pictures: docs/pictures/09-start.png, 10-typed-preview.png, 11-question.png,
// 12-models.png, 13-no-camera.png.
import path from 'node:path';
import { chromium } from '@playwright/test';

const url = process.argv[2] ?? 'http://localhost:5173/';
const pictures = path.resolve(import.meta.dirname, '..', '..', 'docs', 'pictures');
const shot = (page, name, fullPage = false) => page.screenshot({ path: path.join(pictures, name), fullPage });

const browser = await chromium.launch({ channel: 'chrome', args: ['--use-fake-device-for-media-stream'] });
const viewport = { width: 1312, height: 940 };

async function open(context) {
  const page = await context.newPage();
  page.on('pageerror', (e) => console.log('pageerror:', e.message));
  page.on('console', (m) => m.type() === 'error' && console.log('console.error:', m.text().slice(0, 200)));
  await page.goto(url);
  await page.waitForFunction(() => !document.getElementById('start-mouse').disabled, null, { timeout: 30000 });
  return page;
}

/** The blocks where the server tests have them. */
const layout = (page) =>
  page.evaluate(() => {
    const { arm, flow } = window.__sim;
    arm.goHomeNow();
    arm.putBlock('red', 180, 100, 0);
    arm.putBlock('yellow', 225, 100, 0);
    arm.putBlock('blue', 300, 90, 0);
    flow.reset();
  });

/** Click a table point (mm), z above the table. */
async function clickTable(page, x, y, z = 0) {
  const at = await page.evaluate(
    ([x, y, z]) => {
      const { view } = window.__sim;
      const cam = view.camera;
      cam.updateMatrixWorld();
      const v = cam.position.clone().set((250 - y) / 1000, (x - 250) / 1000, z / 1000).project(cam);
      const r = view.canvas.getBoundingClientRect();
      return [r.left + ((v.x + 1) / 2) * r.width, r.top + ((1 - v.y) / 2) * r.height];
    },
    [x, y, z],
  );
  await page.mouse.click(at[0], at[1]);
  await page.waitForTimeout(1200); // a person's pace: clicks 0.3 s apart look like one gesture
}

async function type(page, text, until) {
  await page.fill('#type-box', text);
  await page.press('#type-box', 'Enter');
  await page.waitForFunction((k) => window.__sim.flow.state.kind === k, until, { timeout: 15000 });
  await page.waitForTimeout(400);
}

const flowLine = async (page) =>
  `${await page.evaluate(() => window.__sim.flow.state.kind)} | ${await page.locator('#bubble').textContent()} | ${await page.locator('#status').textContent()}`;

// --- 1. No microphone allowed: the start card, then typing. ---
{
  const context = await browser.newContext({ viewport });
  const page = await open(context);
  await page.waitForTimeout(500);
  await shot(page, '09-start.png');

  await page.click('#start-mouse');
  await page.waitForFunction(() => document.body.classList.contains('typing'), null, { timeout: 10000 });
  console.log('no mic:', await page.locator('#status').textContent());
  await layout(page);
  await page.waitForTimeout(600);
  await clickTable(page, 180, 100, 30); // that: the red block
  await clickTable(page, 330, 50); // here: an empty spot
  await type(page, 'put that here', 'preview');
  console.log('typed:', await flowLine(page));
  await page.click('#details summary');
  await page.mouse.move(5, 5);
  await page.waitForTimeout(300);
  await shot(page, '10-typed-preview.png', true);
  await page.click('#details summary');

  await page.click('#go-btn');
  await page.waitForFunction(() => !window.__sim.arm.busy && window.__sim.flow.state.kind === 'idle', null, {
    timeout: 30000,
  });
  const red = await page.evaluate(() => window.__sim.arm.block('red').tableMm.map((v) => v.toFixed(1)));
  console.log(`moved: red at (${red.join(', ')}), asked for (330, 50)`);

  // Between two blocks: "This one or that one?", answered with a click.
  await layout(page);
  await page.waitForTimeout(600);
  await clickTable(page, 202.5, 100, 30);
  await clickTable(page, 330, 50);
  await type(page, 'put that here', 'asking');
  console.log('between:', await flowLine(page));
  await page.mouse.move(5, 5);
  await page.waitForTimeout(300);
  await shot(page, '11-question.png');
  await clickTable(page, 225, 100, 30); // the yellow one
  await page.waitForFunction(() => window.__sim.flow.state.kind === 'preview', null, { timeout: 15000 });
  console.log('answered:', await flowLine(page));

  await page.click('#models summary');
  await page.waitForTimeout(300);
  await shot(page, '12-models.png', true);
  console.log('models:', (await page.locator('#models-list li').allTextContents()).length, 'lines');
  await context.close();
}

// --- 2. The microphone, but no camera: the mouse points. ---
{
  const context = await browser.newContext({ viewport });
  await context.grantPermissions(['microphone'], { origin: new URL(url).origin });
  const page = await open(context);
  await page.click('#start-camera');
  await page.waitForFunction(() => document.getElementById('talk-btn').classList.contains('ready'), null, {
    timeout: 15000,
  });
  await page.waitForTimeout(800);
  console.log('no camera:', await page.locator('#status').textContent());
  await shot(page, '13-no-camera.png');
  await context.close();
}
await browser.close();
