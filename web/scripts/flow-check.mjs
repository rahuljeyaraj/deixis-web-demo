// The whole flow in a real Chrome (development only; `npm run dev` and `uv run webdemo` must
// run): a fake microphone says "put that ... here" (a test voice), the mouse points at the red
// block during "that" and at an empty spot during "here", Talk is held with the space bar.
// Then the preview ("This one, here?", the glow, the ghost), Go, and the block at the spot.
// Screenshots: out/pictures/07-preview.png and 08-moved.png.
//   node scripts/flow-check.mjs [url]
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

const url = process.argv[2] ?? 'http://localhost:5173/';
const root = path.resolve(import.meta.dirname, '..', '..');
const out = path.join(root, 'out');
const pictures = path.join(root, 'out', 'pictures');
fs.mkdirSync(out, { recursive: true });

// --- the fake microphone: 1 s, "put that", 1.4 s, "here", 3.2 s (Chrome loops it) ---
const voice = JSON.parse(fs.readFileSync(path.join(root, 'tests', 'fixtures', 'voice.json'), 'utf8'));
const item = (text) => voice.items.find((i) => i.voice.includes('David') && i.text === text);
const pcmOf = (it) => {
  const src = fs.readFileSync(path.join(root, '.cache', 'voice', it.wav));
  let at = 12;
  while (src.toString('ascii', at, at + 4) !== 'data') at += 8 + src.readUInt32LE(at + 4);
  return src.subarray(at + 8, at + 8 + src.readUInt32LE(at + 4));
};
const gap = (s) => Buffer.alloc(Math.round(s * 16000) * 2);
const putThat = item('put that');
const here = item('here');
const parts = [gap(1), pcmOf(putThat), gap(1.4), pcmOf(here), gap(3.2)];
const body = Buffer.concat(parts);
const period = body.length / 32000;
// Seconds from the start of the loop: when "that" and "here" begin.
const tThat = 1 + putThat.words[1].t0;
const tHere = 1 + putThat.duration + 1.4 + here.words[0].t0;
const firstSound = 1 + putThat.first_sound;
const header = Buffer.alloc(44);
header.write('RIFF', 0, 'ascii');
header.writeUInt32LE(36 + body.length, 4);
header.write('WAVEfmt ', 8, 'ascii');
header.writeUInt32LE(16, 16);
header.writeUInt16LE(1, 20);
header.writeUInt16LE(1, 22);
header.writeUInt32LE(16000, 24);
header.writeUInt32LE(32000, 28);
header.writeUInt16LE(2, 32);
header.writeUInt16LE(16, 34);
header.write('data', 36, 'ascii');
header.writeUInt32LE(body.length, 40);
const micFile = path.join(out, 'fake-mic-flow.wav');
fs.writeFileSync(micFile, Buffer.concat([header, body]));

const browser = await chromium.launch({
  channel: 'chrome',
  args: [
    '--use-fake-device-for-media-stream',
    '--use-fake-ui-for-media-stream',
    `--use-file-for-fake-audio-capture=${micFile}`,
    '--autoplay-policy=no-user-gesture-required',
  ],
});
const context = await browser.newContext({ viewport: { width: 1312, height: 940 } });
await context.grantPermissions(['microphone'], { origin: new URL(url).origin });
const page = await context.newPage();
page.on('pageerror', (e) => console.log('pageerror:', e.message));
page.on('console', (m) => m.type() === 'error' && console.log('console.error:', m.text().slice(0, 200)));
await page.goto(url);
await page.waitForFunction(() => window.__sim, null, { timeout: 30000 });

// The blocks where the server tests have them; the spot is empty and on the mat.
const SPOT = [330, 50];
await page.evaluate(() => {
  const { arm, flow } = window.__sim;
  arm.goHomeNow();
  arm.putBlock('red', 180, 100, 0);
  arm.putBlock('green', 225, 100, 0);
  arm.putBlock('blue', 300, 90, 0);
  flow.reset();
});
await page.waitForTimeout(800);

/** Page coordinates of a table point (mm), z above the table. */
const screen = (x, y, z = 0) =>
  page.evaluate(
    ([x, y, z]) => {
      const { view } = window.__sim;
      const cam = view.camera;
      cam.updateMatrixWorld();
      // table mm -> world m (src/sim/table.ts)
      const v = cam.position.clone().set((250 - y) / 1000, (x - 250) / 1000, z / 1000).project(cam);
      const r = view.canvas.getBoundingClientRect();
      return [r.left + ((v.x + 1) / 2) * r.width, r.top + ((1 - v.y) / 2) * r.height];
    },
    [x, y, z],
  );
const onRed = await screen(180, 100, 30);
const onSpot = await screen(...SPOT, 0);
const rest = await screen(250, -40, 0);
await page.mouse.move(...rest);

// The microphone, and where its loop is: wait for the first sound of "put that".
await page.click('#start-mouse'); // the start card: the mouse points, the microphone listens
await page.waitForFunction(() => window.__sim.mic()?.chunks > 10, null, { timeout: 20000 });
const loopStart = await page.evaluate(async (firstSound) => {
  // The first sound after 3 s of quiet (inside the sentence there is at most 2 s).
  const m = window.__sim.mic();
  let lastLoud = performance.now();
  for (;;) {
    await new Promise((r) => setTimeout(r, 10));
    const t = performance.now();
    if (m.level < 0.003) continue;
    if (t - lastLoud > 3000) return t / 1000 - firstSound - 0.06; // the chunk came a bit late
    lastLoud = t;
  }
}, firstSound);
const pageNow = () => page.evaluate(() => performance.now() / 1000);
const waitUntil = async (t) => {
  const d = t - (await pageNow());
  if (d > 0) await page.waitForTimeout(d * 1000);
};
const next = loopStart + period; // the next "put that ... here"
await waitUntil(next + tThat - 1.6);
await page.mouse.move(...onRed, { steps: 8 });
await waitUntil(next + firstSound - 0.7);
await page.keyboard.down('Space');
await waitUntil(next + tHere - 0.9);
await page.mouse.move(...onSpot, { steps: 10 });
await waitUntil(next + tHere + 1.0);
await page.keyboard.up('Space');
await page.waitForFunction(() => window.__sim.flow.state.kind !== 'thinking' && window.__sim.lastReply(), null, {
  timeout: 15000,
});
const reply = await page.evaluate(() => window.__sim.lastReply());
const clip = await page.evaluate(() => {
  const c = window.__sim.lastClip();
  return { t0: c.t0, duration: c.duration, wav: Array.from(new Uint8Array(c.wav)) };
});
fs.writeFileSync(path.join(out, 'flow-check-clip.wav'), Buffer.from(clip.wav));
const rel = (t) => (t - next).toFixed(2);
console.log(`loop ${period.toFixed(2)} s; "that" at ${tThat.toFixed(2)}, "here" at ${tHere.toFixed(2)} into it`);
console.log(`clip from ${rel(clip.t0)} to ${rel(clip.t0 + clip.duration)} s into the loop`);
console.log('words:', reply.words.map((w) => `${w.text}@${rel(w.t0)}`).join(' '));
console.log('agent:', reply.agent.source, reply.agent.model, JSON.stringify(reply.agent.calls));
console.log('timings:', JSON.stringify(reply.timings));
console.log('core:', reply.result?.status, reply.result?.ask?.question ?? '');
const state = await page.evaluate(() => window.__sim.flow.state.kind);
console.log('flow:', state, '| bubble:', await page.locator('#bubble').textContent());
await page.mouse.move(...rest);
await page.waitForTimeout(300);
await page.screenshot({ path: path.join(pictures, '07-preview.png') });
if (state === 'preview') {
  await page.click('#go-btn');
  await page.waitForFunction(() => !window.__sim.arm.busy && window.__sim.flow.state.kind === 'idle', null, {
    timeout: 30000,
  });
  await page.waitForTimeout(500);
  const red = await page.evaluate(() => window.__sim.arm.block('red').tableMm);
  console.log(`red block at (${red.map((v) => v.toFixed(1)).join(', ')}); asked for (${SPOT.join(', ')})`);
  await page.screenshot({ path: path.join(pictures, '08-moved.png') });
}
await browser.close();
