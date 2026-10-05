/**
 * The voice side (plan W5): the WAV encoder, the resampler, the audio clock mapped onto the
 * page clock, the clip with its pre-roll and tail, push-to-talk, and the mouse as a pointer.
 */

import { describe, expect, test, vi } from 'vitest';
import { AudioRing, ClipRecorder } from '../src/audio/clip';
import { ClockMap } from '../src/audio/clock';
import { PushToTalk, type StopReason } from '../src/audio/talk';
import { SPEECH_RATE, decodeWav, encodeWav, levels, resample } from '../src/audio/wav';
import { MousePointer } from '../src/hand/mouse';

const tone = (rate: number, hz: number, s: number, amp = 0.5, at = 0) =>
  Float32Array.from({ length: Math.round(rate * s) }, (_, i) =>
    i / rate >= at ? amp * Math.sin(2 * Math.PI * hz * (i / rate - at)) : 0,
  );

/** RMS of the middle half (away from the ends). */
const midRms = (x: Float32Array) => {
  const m = x.subarray(Math.floor(x.length / 4), Math.floor((3 * x.length) / 4));
  return Math.sqrt(m.reduce((a, s) => a + s * s, 0) / m.length);
};

/** The first sample whose magnitude passes half of `amp`. */
const onset = (x: Float32Array, amp: number) => x.findIndex((s) => Math.abs(s) > amp / 2);

describe('WAV', () => {
  test('a mono 16 kHz 16-bit header, and the samples read back', () => {
    const x = Float32Array.from([0, 0.5, -0.5, 1, -1, 2, -2]);
    const buf = encodeWav(x);
    const v = new DataView(buf);
    const text = (at: number) => String.fromCharCode(...new Uint8Array(buf, at, 4));
    expect([text(0), text(8), text(12), text(36)]).toEqual(['RIFF', 'WAVE', 'fmt ', 'data']);
    expect(v.getUint16(20, true)).toBe(1); // PCM
    expect(v.getUint16(22, true)).toBe(1); // mono
    expect(v.getUint32(24, true)).toBe(16000);
    expect(v.getUint16(34, true)).toBe(16);
    expect(v.getUint32(40, true)).toBe(x.length * 2);
    expect(buf.byteLength).toBe(44 + x.length * 2);
    const back = decodeWav(buf);
    expect(back.rate).toBe(16000);
    // Within one step of 16 bits; outside -1..1 is clipped.
    [0, 0.5, -0.5, 1, -1, 1, -1].forEach((s, i) => expect(back.samples[i]).toBeCloseTo(s, 4));
  });

  test('levels', () => {
    expect(levels(tone(16000, 440, 1, 0.5)).peakDb).toBeCloseTo(-6.02, 1);
    expect(levels(tone(16000, 440, 1, 0.5)).rmsDb).toBeCloseTo(-9.03, 1);
    expect(levels(new Float32Array(10)).peakDb).toBe(-Infinity);
  });
});

describe('resampling to 16 kHz', () => {
  test.each([48000, 44100])('%i Hz: a 1 kHz tone keeps its level and pitch', (rate) => {
    const y = resample(tone(rate, 1000, 1), rate);
    expect(y.length).toBe(Math.floor(rate * (16000 / rate)));
    expect(midRms(y)).toBeCloseTo(0.5 / Math.SQRT2, 2);
    // Zero crossings in the middle half: 2 per cycle, so 1000 over 0.5 s.
    const m = y.subarray(4000, 12000);
    let zc = 0;
    for (let i = 1; i < m.length; i++) if (m[i - 1] < 0 !== m[i] < 0) zc++;
    expect(Math.abs(zc - 1000)).toBeLessThanOrEqual(2);
  });

  test.each([48000, 44100])('%i Hz: sound above 8 kHz is removed, not folded down', (rate) => {
    const y = resample(tone(rate, 11000, 1), rate);
    expect(20 * Math.log10(midRms(y) / (0.5 / Math.SQRT2))).toBeLessThan(-40);
  });

  test.each([48000, 44100])('%i Hz: a sound starts at the same time after resampling', (rate) => {
    const at = 0.4321;
    const y = resample(tone(rate, 500, 1, 0.5, at), rate);
    expect(Math.abs(onset(y, 0.5) / SPEECH_RATE - at)).toBeLessThan(0.0015);
  });
});

describe('the audio clock on the page clock', () => {
  // The page clock runs 80 ppm fast against the audio clock, 1234.5 s apart, with readings
  // jittered by up to 3 ms (as getOutputTimestamp's are).
  const truth = (c: number) => 1234.5 + c * (1 + 80e-6);
  const jitter = (i: number) => 0.003 * Math.sin(i * 12.9898);

  test('a single pair: offset only', () => {
    const m = new ClockMap();
    m.add({ context: 2, page: truth(2) });
    expect(m.toPage(2)).toBeCloseTo(truth(2), 9);
    expect(m.driftPpm).toBe(0);
  });

  test('a minute of pairs every 2 s: the line through them', () => {
    const m = new ClockMap();
    for (let i = 0; i <= 30; i++) m.add({ context: 2 * i, page: truth(2 * i) + jitter(i) });
    for (const c of [0, 17.3, 60, 75]) expect(Math.abs(m.toPage(c) - truth(c))).toBeLessThan(0.002);
    expect(m.driftPpm).toBeGreaterThan(0);
    expect(m.driftPpm).toBeLessThan(200);
    expect(m.residualMs).toBeLessThan(3);
    for (const c of [3, 44.4]) expect(m.toContext(m.toPage(c))).toBeCloseTo(c, 9);
  });

  test('a stale pair (older context time) is skipped', () => {
    const m = new ClockMap();
    m.add({ context: 10, page: truth(10) });
    m.add({ context: 9, page: 0 });
    expect(m.size).toBe(1);
    m.clear();
    expect(m.size).toBe(0);
  });
});

describe('the clip', () => {
  const RATE = 48000;
  /** The audio clock started at page time 100 s, with no drift. */
  const clockAt100 = () => {
    const m = new ClockMap();
    m.add({ context: 0, page: 100 });
    return m;
  };
  /** Feeds `signal` in 2048-frame chunks from where the ring ends up to (excluding) `to`. */
  const feed = (rec: ClipRecorder, signal: Float32Array, to: number) => {
    for (let f = rec.ring.end; f < to; f += 2048) rec.push(f, signal.slice(f, Math.min(f + 2048, to)));
  };

  test('the ring fills gaps with silence and keeps about a second', () => {
    const ring = new AudioRing(1000, 1);
    ring.push(0, new Float32Array(500).fill(1));
    ring.push(700, new Float32Array(300).fill(2)); // 500..700 never arrived
    const s = ring.slice(400, 800);
    expect([s[0], s[99], s[100], s[299], s[300], s[399]]).toEqual([1, 1, 0, 0, 2, 2]);
    for (let f = 1000; f < 5000; f += 250) ring.push(f, new Float32Array(250));
    expect(ring.end).toBe(5000);
    expect(ring.start).toBeGreaterThanOrEqual(3750);
    expect(ring.start).toBeLessThanOrEqual(4000);
    ring.keep(4000);
    for (let f = 5000; f < 9000; f += 250) ring.push(f, new Float32Array(250));
    expect(ring.start).toBe(4000);
  });

  test('pre-roll, tail, and a sound at a known page time lands there', async () => {
    // A beep at page time 103.250 s; Talk held from 103.0 s to 104.5 s.
    const signal = tone(RATE, 700, 8, 0.5, 3.25);
    const rec = new ClipRecorder(RATE, clockAt100());
    feed(rec, signal, RATE * 3.1); // the mic ran before the press
    rec.begin(103.0);
    feed(rec, signal, RATE * 6);
    const clip = await rec.end(104.5);
    expect(clip.rate).toBe(16000);
    expect(clip.t0).toBeCloseTo(102.5, 3); // 0.5 s of pre-roll
    expect(clip.duration).toBeCloseTo(104.9 - 102.5, 3); // to 0.4 s after the release
    const beep = clip.t0 + onset(clip.samples, 0.5) / clip.rate;
    expect(Math.abs(beep - 103.25)).toBeLessThan(0.002);
    expect(decodeWav(clip.wav).samples.length).toBe(clip.samples.length);
    expect(clip.cut).toBe(false);
  });

  test('a press right after the mic starts gets only what the ring has', async () => {
    const signal = tone(RATE, 700, 3, 0.5);
    const rec = new ClipRecorder(RATE, clockAt100());
    feed(rec, signal, 4096);
    rec.begin(100.05);
    feed(rec, signal, RATE * 2);
    const clip = await rec.end(101);
    expect(clip.t0).toBeCloseTo(100, 6);
  });

  test('waits for the tail, then cuts with what came if it never arrives', async () => {
    vi.useFakeTimers();
    try {
      const rec = new ClipRecorder(RATE, clockAt100());
      const signal = tone(RATE, 700, 3, 0.5);
      feed(rec, signal, RATE);
      rec.begin(100.6);
      let done = false;
      const p = rec.end(100.8).then((c) => ((done = true), c));
      await vi.advanceTimersByTimeAsync(500);
      expect(done).toBe(false);
      await vi.advanceTimersByTimeAsync(1100); // past the 1.5 s wait
      const clip = await p;
      expect(clip.t0).toBeCloseTo(100.1, 3);
      expect(clip.duration).toBeCloseTo(0.9, 2); // up to the last sample received
    } finally {
      vi.useRealTimers();
    }
  });

  test('a second press during the first clip’s tail records its own clip', async () => {
    const signal = tone(RATE, 700, 10, 0.5);
    const rec = new ClipRecorder(RATE, clockAt100());
    feed(rec, signal, RATE * 2);
    rec.begin(101.8);
    feed(rec, signal, RATE * 3);
    const first = rec.end(103);
    rec.begin(103.2); // pressed again before the first tail (to 103.4) arrived
    feed(rec, signal, RATE * 5);
    const a = await first;
    const b = await rec.end(104.5);
    expect(a.t0).toBeCloseTo(101.3, 3);
    expect(a.duration).toBeCloseTo(2.1, 3);
    expect(b.t0).toBeCloseTo(102.7, 3);
    expect(b.duration).toBeCloseTo(2.2, 3);
  });

  test('a clip longer than 12 s is cut at 12 s', async () => {
    const signal = new Float32Array(RATE * 16);
    const rec = new ClipRecorder(RATE, clockAt100());
    feed(rec, signal, RATE);
    rec.begin(100.9);
    feed(rec, signal, RATE * 16);
    const clip = await rec.end(115);
    expect(clip.cut).toBe(true);
    expect(clip.duration).toBeCloseTo(12, 3);
  });
});

describe('push-to-talk', () => {
  const make = () => {
    const log: string[] = [];
    const ptt = new PushToTalk({
      start: (t) => log.push(`start ${t}`),
      stop: (t, why: StopReason) => log.push(`stop ${t} ${why}`),
    });
    return { ptt, log };
  };

  test('hold to talk, let go to stop', () => {
    const { ptt, log } = make();
    ptt.press(1);
    ptt.release(3);
    expect(log).toEqual(['start 1', 'stop 3 release']);
    expect(ptt.listening).toBe(false);
  });

  test('a tap keeps listening until the next tap', () => {
    const { ptt, log } = make();
    ptt.press(1);
    ptt.release(1.1);
    expect(ptt.listening).toBe(true);
    ptt.press(4);
    ptt.release(4.1);
    expect(log).toEqual(['start 1', 'stop 4 tap']);
    ptt.press(6); // a new listen, held this time
    ptt.release(7);
    expect(log.slice(2)).toEqual(['start 6', 'stop 7 release']);
  });

  test('it stops by itself before the clip gets too long', () => {
    const { ptt, log } = make();
    ptt.press(0);
    ptt.tick(10.9);
    ptt.tick(11);
    expect(log).toEqual(['start 0', 'stop 11 long']);
    ptt.release(12); // the button was still held: nothing new starts
    ptt.press(13);
    ptt.release(14);
    expect(log.slice(2)).toEqual(['start 13', 'stop 14 release']);
  });

  test('Esc while the button is held: stops, and the release starts nothing', () => {
    const { ptt, log } = make();
    ptt.press(0);
    ptt.cancel(1);
    ptt.release(2);
    ptt.cancel(3); // not listening: nothing
    ptt.press(4);
    ptt.release(4.1);
    ptt.press(5); // ends the tapped listen
    ptt.press(5.05); // a second key while one is down: ignored
    ptt.release(5.1);
    expect(log).toEqual(['start 0', 'stop 1 cancel', 'start 4', 'stop 5 tap']);
  });

  test('a tapped listen that runs too long leaves the next press working', () => {
    const { ptt, log } = make();
    ptt.press(0);
    ptt.release(0.1);
    ptt.tick(11);
    ptt.press(20);
    ptt.release(21);
    expect(log).toEqual(['start 0', 'stop 11 long', 'start 20', 'stop 21 release']);
  });
});

describe('the mouse as a pointer', () => {
  const hitAt = (u: number, v: number) => (u < 0.9 ? { x: u * 500, y: (1 - v) * 250, z: 0, block: null } : null);

  test('the cursor is sampled even when it stays still', () => {
    const m = new MousePointer(hitAt);
    m.sample(1); // not over the canvas yet
    m.move(0.5, 0.5);
    for (let i = 1; i <= 10; i++) m.sample(1 + i / 30);
    m.move(0.95, 0.5); // the ray misses
    m.sample(2);
    const s = m.samples.between(0, 3);
    expect(s.length).toBe(12);
    expect(s[0].point).toBeNull();
    expect(s[5]).toEqual({ t: 1 + 5 / 30, point: [250, 125], pointing: true });
    expect(s[11].point).toBeNull();
  });

  test('a click is a 0.3 s hold at the click', () => {
    const m = new MousePointer(hitAt);
    const h = m.click(5, { x: 100, y: 50, z: 30, block: 'red' });
    expect(h).toEqual({ t0: 4.85, t1: 5.15, mean: [100, 50], spread: [0, 0], frame: 'table', source: 'click' });
    m.click(30, { x: 1, y: 2, z: 0, block: null });
    expect(m.clicksBetween(0, 100).length).toBe(1); // older than 20 s: gone
    expect(m.clicksBetween(29, 29.9).length).toBe(1);
  });
});
