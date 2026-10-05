// The day's limits and the own-key box (plan 8.1, 8.2, W11) in a real Chrome (development
// only). Start the server with no budget left, so the banner shows without spending anything:
//   $env:CAP_TOTAL_PER_DAY=0; uv run webdemo        (and `npm run dev`)
//   node scripts/key-shots.mjs [url] [--real-key]
// Pictures: out/pictures/14-simple-mode.png, 15-own-key.png, 16-bad-key.png (and with
// `--real-key`, 17-own-key-saved.png).
// `--real-key` also pastes the key from ../.env as a visitor's own key (one Nemotron call,
// about $0.0005) and checks that the page says it works. The key is never printed.
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

const args = process.argv.slice(2);
const url = args.find((a) => !a.startsWith('--')) ?? 'http://localhost:5173/';
const realKey = args.includes('--real-key');
const pictures = path.resolve(import.meta.dirname, '..', '..', 'out', 'pictures');

const browser = await chromium.launch({ channel: 'chrome' });
const context = await browser.newContext({ viewport: { width: 1312, height: 940 } }); // no mic: typing
const page = await context.newPage();
page.on('pageerror', (e) => console.log('pageerror:', e.message));
await page.goto(url);
await page.waitForFunction(() => !document.getElementById('start-mouse').disabled, null, { timeout: 30000 });
await page.click('#start-mouse');
await page.waitForFunction(() => document.body.classList.contains('typing'), null, { timeout: 10000 });

const text = (sel) => page.locator(sel).textContent();
const banner = async () => ((await page.locator('#banner').isVisible()) ? await text('#banner-text') : '(none)');
const nemotronLine = () => text('#models-list li[data-id="nemotron"]');

/** Type a command and wait for its reply. */
async function type(words) {
  await page.evaluate(() => (window.__seen = window.__sim.lastReply())); // compared in the page
  await page.fill('#type-box', words);
  await page.press('#type-box', 'Enter');
  await page.waitForFunction(() => window.__sim.lastReply() !== window.__seen, null, { timeout: 20000 });
  await page.waitForTimeout(500);
}

// 1. Simple mode: the banner from /api/info, and a typed command the word matcher answers.
await page.waitForFunction(() => !document.getElementById('banner').hidden, null, { timeout: 10000 });
await type('go home');
console.log('banner:', await banner());
console.log('understood by:', await text('#details-body section:nth-of-type(2) p'));
await page.screenshot({ path: path.join(pictures, '14-simple-mode.png') });

// 2. "Add your own Nebius key" opens the footer's panel.
await page.click('#banner-key');
await page.waitForTimeout(600);
console.log('models, Nemotron:', await nemotronLine());
console.log('panel:', await text('#key-left'));
await page.locator('footer').screenshot({ path: path.join(pictures, '15-own-key.png') });

// 3. A key Nebius refuses: forgotten, and the word matcher answers.
await page.fill('#key-box', 'sk-not-a-real-key-0123456789abcdef');
await page.click('#key-save');
console.log('saved:', await text('#key-status'), '| box empty:', (await page.inputValue('#key-box')) === '');
console.log('banner with a key:', await banner());
await type('go home');
console.log('bad key:', await text('#key-status'), '|', await text('#status'));
console.log('banner after:', await banner());
await page.screenshot({ path: path.join(pictures, '16-bad-key.png'), fullPage: true });

// 4. (--real-key) A good key: Nemotron answers this visitor although the budget is used up.
if (realKey) {
  const env = fs.readFileSync(path.resolve(import.meta.dirname, '..', '..', '.env'), 'utf8');
  const key = /^NEBIUS_API_KEY=(.+)$/m.exec(env)?.[1].trim().replace(/^["']|["']$/g, '');
  if (!key) throw new Error('no NEBIUS_API_KEY in .env');
  await page.fill('#key-box', key);
  await page.click('#key-save');
  await type('go home');
  console.log('good key:', await text('#key-status'));
  console.log('understood by:', await text('#details-body section:nth-of-type(2) p'));
  console.log('banner:', await banner(), '| models, Nemotron:', await nemotronLine());
  const html = await page.content();
  console.log('key in the page:', html.includes(key) ? 'YES (bad)' : 'no');
  await page.locator('footer').screenshot({ path: path.join(pictures, '17-own-key-saved.png') });
}

await browser.close();
