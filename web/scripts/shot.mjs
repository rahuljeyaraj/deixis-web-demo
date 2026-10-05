// Screenshot helper for development: node ../out/shot.mjs <url> <png> [waitMs]
import { chromium } from '@playwright/test';
const [url, png, wait = '6000'] = process.argv.slice(2);
const browser = await chromium.launch({ channel: 'chrome', args: ['--use-angle=d3d11', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1312, height: 800 } });
page.on('console', (m) => console.log('console:', m.type(), m.text()));
page.on('pageerror', (e) => console.log('pageerror:', e.message));
page.on('response', (r) => { if (r.status() >= 400) console.log('http', r.status(), r.url()); });
page.on('requestfailed', (r) => console.log('failed', r.url(), r.failure()?.errorText));
await page.goto(url);
await page.waitForTimeout(Number(wait));
await page.screenshot({ path: png });
console.log('status:', await page.locator('#status').textContent());
await browser.close();
