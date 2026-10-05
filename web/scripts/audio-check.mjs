// Checks the voice path in a real Chrome (development only; the dev server must be running):
//   1. a tone made inside the page at a known audio time lands at the matching page time in
//      the clip (the worklet's frame numbers, the pre-roll, the resampler);
//   2. the audio clock against the page clock: the mapping from getOutputTimestamp against
//      the moment the page sees the audio clock pass a time, and the drift over a minute;
//   3. a spoken command from Chrome's fake microphone (a test voice clip): hold the space bar,
//      let go, and save the clip to ../out/audio-check-clip.wav.
//   node scripts/audio-check.mjs [url] [--minute]
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

const args = process.argv.slice(2);
const url = args.find((a) => !a.startsWith('--')) ?? 'http://localhost:5173/';
const minute = args.includes('--minute');
const root = path.resolve(import.meta.dirname, '..', '..');
const out = path.join(root, 'out');
fs.mkdirSync(out, { recursive: true });

// The fake microphone: 1 s of silence, "put that here", 2 s of silence (Chrome loops it).
const voice = JSON.parse(fs.readFileSync(path.join(root, 'tests', 'fixtures', 'voice.json'), 'utf8'));
const item = voice.items.find((i) => i.id === 'david_00');
const src = fs.readFileSync(path.join(root, '.cache', 'voice', item.wav));
// SAPI writes an 18-byte fmt chunk: find the data chunk instead of assuming 44 bytes.
let at = 12;
while (src.toString('ascii', at, at + 4) !== 'data') at += 8 + src.readUInt32LE(at + 4);
const pcm = src.subarray(at + 8, at + 8 + src.readUInt32LE(at + 4));
const pad = (s) => Buffer.alloc(Math.round(s * 16000) * 2);
const body = Buffer.concat([pad(1), pcm, pad(2)]);
const header = Buffer.alloc(44);
header.write('RIFF', 0, 'ascii');
header.writeUInt32LE(36 + body.length, 4);
header.write('WAVEfmt ', 8, 'ascii');
header.writeUInt32LE(16, 16);
header.writeUInt16LE(1, 20); // PCM
header.writeUInt16LE(1, 22); // mono
header.writeUInt32LE(16000, 24);
header.writeUInt32LE(32000, 28);
header.writeUInt16LE(2, 32);
header.writeUInt16LE(16, 34);
header.write('data', 36, 'ascii');
header.writeUInt32LE(body.length, 40);
const micFile = path.join(out, 'fake-mic.wav');
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
const context = await browser.newContext({ viewport: { width: 1312, height: 900 } });
await context.grantPermissions(['microphone'], { origin: new URL(url).origin });
const page = await context.newPage();
page.on('pageerror', (e) => console.log('pageerror:', e.message));
page.on('console', (m) => ['error', 'warning'].includes(m.type()) && console.log(`console.${m.type()}:`, m.text().slice(0, 200)));
await page.goto(url);
await page.waitForFunction(() => window.__sim, null, { timeout: 30000 });
await page.click('#talk-btn'); // the first press asks for the microphone
await page.waitForFunction(() => window.__sim.mic()?.chunks > 20, null, { timeout: 20000 });
const setup = await page.evaluate(() => {
  const m = window.__sim.mic();
  return { rate: m.ctx.sampleRate, baseLatency: m.ctx.baseLatency, outputLatency: m.ctx.outputLatency, inputLatency: m.inputLatency };
});
console.log('audio:', JSON.stringify(setup));

// 1 and 2: a tone at a known audio time; when does the page see the audio clock reach it?
const tone = await page.evaluate(async () => {
  const m = window.__sim.mic();
  const ctx = m.ctx;
  m.source.disconnect(); // only the tone goes in
  const yieldNow = () => new Promise((r) => { const ch = new MessageChannel(); ch.port1.onmessage = r; ch.port2.postMessage(0); });
  const results = [];
  for (let k = 0; k < 5; k++) {
    const c0 = ctx.currentTime + 0.6;
    const osc = new OscillatorNode(ctx, { frequency: 700 });
    const gain = new GainNode(ctx, { gain: 0.5 });
    osc.connect(gain).connect(m.node);
    m.begin(performance.now() / 1000);
    osc.start(c0);
    osc.stop(c0 + 0.3);
    // The page sees the audio clock pass c0 (the audio thread rendered that block).
    let seen = 0;
    while (!seen) {
      if (ctx.currentTime >= c0) seen = performance.now() / 1000;
      await yieldNow();
    }
    await new Promise((r) => setTimeout(r, 700));
    const clip = await m.end(performance.now() / 1000);
    const i = clip.samples.findIndex((s) => Math.abs(s) > 0.25);
    const inClip = clip.t0 + i / clip.rate;
    const mapped = m.clock.toPage(c0);
    // The mapping is for the microphone: it takes off the input latency too, which a tone
    // made inside the page doesn't have; add it back to compare with the render time.
    const rendered = mapped + m.inputLatency;
    results.push({ inClipVsMapMs: (inClip - mapped) * 1000, mapVsSeenMs: (rendered - seen) * 1000 });
    osc.disconnect();
    gain.disconnect();
  }
  m.source.connect(m.node);
  return results;
});
const fmt = (xs) => xs.map((x) => x.toFixed(1)).join(', ');
console.log('tone in the clip minus the mapped time, ms:', fmt(tone.map((r) => r.inClipVsMapMs)));
console.log('mapped time minus when the page saw the audio clock pass it, ms:', fmt(tone.map((r) => r.mapVsSeenMs)));

// 3: speak (the fake microphone loops "put that here"): hold the space bar for 4.5 s.
await page.keyboard.down('Space');
await page.waitForTimeout(4500);
await page.keyboard.up('Space');
await page.waitForFunction(() => window.__sim.lastClip()?.duration > 4, null, { timeout: 5000 });
const clip = await page.evaluate(() => {
  const c = window.__sim.lastClip();
  return { t0: c.t0, duration: c.duration, peakDb: c.peakDb, rmsDb: c.rmsDb, wav: Array.from(new Uint8Array(c.wav)) };
});
fs.writeFileSync(path.join(out, 'audio-check-clip.wav'), Buffer.from(clip.wav));
console.log(`clip: ${clip.duration.toFixed(2)} s, peak ${clip.peakDb.toFixed(1)} dB, rms ${clip.rmsDb.toFixed(1)} dB -> out/audio-check-clip.wav`);
// The server's words (W6), if the server runs: times as seconds into the clip.
const reply = await page
  .waitForFunction(() => window.__sim.lastReply(), null, { timeout: 10000 })
  .then((h) => h.jsonValue())
  .catch(() => null);
if (reply) {
  const words = reply.words.map((w) => `${w.text}@${(w.t0 - clip.t0).toFixed(2)}`).join(' ');
  console.log(`words (s into the clip): ${words}; speech ${reply.timings.asr_s} s, total ${reply.timings.total_s} s`);
} else {
  console.log('no words: is the server running (uv run webdemo)?');
}
console.log('status:', await page.locator('#status').textContent());

if (minute) {
  await page.waitForTimeout(60000);
}
const drift = await page.evaluate(() => {
  const m = window.__sim.mic();
  return { pairs: m.clock.size, driftPpm: m.clock.driftPpm, residualMs: m.clock.residualMs };
});
console.log(`clock: ${drift.pairs} pairs, drift ${drift.driftPpm.toFixed(1)} ppm, scatter ${drift.residualMs.toFixed(2)} ms`);
await page.screenshot({ path: path.join(out, 'audio-check.png') });
await browser.close();
