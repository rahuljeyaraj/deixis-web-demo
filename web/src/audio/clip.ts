/**
 * Cutting the clip for one command (plan 4.2, 4.7). The microphone's samples arrive in chunks
 * numbered by their first frame on the AudioContext's clock. The ring keeps the last second,
 * so a word said just before Talk is pressed isn't lost: the clip starts 0.5 s before the
 * press (the pre-roll) and ends 0.4 s after the release (the tail). Its `t0` is the page-clock
 * time of its first sample, so the server can put the words on the page clock.
 */

import type { ClockMap } from './clock';
import { SPEECH_RATE, encodeWav, levels, resample } from './wav';

interface Chunk {
  frame: number;
  data: Float32Array;
}

/** The microphone's recent samples, by frame number on the context's clock. */
export class AudioRing {
  private chunks: Chunk[] = [];
  /** While recording, everything from this frame on is kept. */
  private keepFrom: number | null = null;

  constructor(
    readonly rate: number,
    readonly keepS = 1,
  ) {}

  /** The frame after the last sample received. */
  get end(): number {
    const c = this.chunks[this.chunks.length - 1];
    return c ? c.frame + c.data.length : 0;
  }

  /** The first frame still held. */
  get start(): number {
    return this.chunks[0]?.frame ?? this.end;
  }

  push(frame: number, data: Float32Array): void {
    this.chunks.push({ frame, data });
    const cut = Math.min(this.end - Math.round(this.keepS * this.rate), this.keepFrom ?? Infinity);
    while (this.chunks.length > 1 && this.chunks[0].frame + this.chunks[0].data.length <= cut) {
      this.chunks.shift();
    }
  }

  /** Keep everything from `frame` on (null: back to the last second only). */
  keep(frame: number | null): void {
    this.keepFrom = frame;
  }

  /** Frames f0 (inclusive) to f1 (exclusive); frames not received are silence. */
  slice(f0: number, f1: number): Float32Array {
    const out = new Float32Array(Math.max(0, f1 - f0));
    for (const c of this.chunks) {
      const a = Math.max(f0, c.frame);
      const b = Math.min(f1, c.frame + c.data.length);
      if (a < b) out.set(c.data.subarray(a - c.frame, b - c.frame), a - f0);
    }
    return out;
  }
}

export interface Clip {
  /** Mono 16 kHz 16-bit WAV, for `POST /api/command`. */
  wav: ArrayBuffer;
  /** The same samples as floats. */
  samples: Float32Array;
  rate: number;
  /** Page-clock seconds (`performance.now() / 1000`) of the first sample. */
  t0: number;
  duration: number;
  peakDb: number;
  rmsDb: number;
  /** The clip reached the longest allowed length and was cut there. */
  cut: boolean;
}

export interface ClipOptions {
  preRollS: number;
  tailS: number;
  /** The server refuses longer clips (plan 7.2). */
  maxS: number;
  /** How long to wait for the tail's samples before using what has arrived. */
  waitS: number;
}

export const CLIP_DEFAULTS: ClipOptions = { preRollS: 0.5, tailS: 0.4, maxS: 12, waitS: 1.5 };

export class ClipRecorder {
  readonly ring: AudioRing;
  readonly opts: ClipOptions;
  /** The clip being recorded: its first frame. */
  private startFrame: number | null = null;
  /** Clips released and waiting for their tail: their first frames. */
  private finishing = new Set<number>();
  private waiters: { frame: number; done: () => void }[] = [];

  constructor(
    readonly rate: number,
    private readonly clock: ClockMap,
    opts: Partial<ClipOptions> = {},
  ) {
    this.opts = { ...CLIP_DEFAULTS, ...opts };
    this.ring = new AudioRing(rate, Math.max(1, this.opts.preRollS + 0.5));
  }

  get recording(): boolean {
    return this.startFrame !== null;
  }

  /** A chunk from the microphone. */
  push(frame: number, data: Float32Array): void {
    this.ring.push(frame, data);
    const end = this.ring.end;
    for (const w of this.waiters.filter((w) => end >= w.frame)) w.done();
  }

  /** Talk pressed at page time `t`: the clip starts `preRollS` earlier (if the ring has it). */
  begin(t: number): void {
    const want = this.frameAt(t - this.opts.preRollS);
    this.startFrame = Math.max(want, this.ring.start);
    this.keep();
  }

  /**
   * Talk released at page time `t`: wait for the tail, then cut the clip. Talk may be pressed
   * again while this waits; the next clip is recorded meanwhile.
   */
  async end(t: number): Promise<Clip> {
    if (this.startFrame === null) throw new Error('not recording');
    const start = this.startFrame;
    this.startFrame = null;
    this.finishing.add(start);
    const longest = start + Math.round(this.opts.maxS * this.rate);
    const want = this.frameAt(t + this.opts.tailS);
    const stop = Math.min(want, longest);
    if (this.ring.end < stop) await this.waitFor(stop);
    const raw = this.ring.slice(start, Math.min(stop, Math.max(this.ring.end, start)));
    this.finishing.delete(start);
    this.keep();
    const samples = resample(raw, this.rate, SPEECH_RATE);
    return {
      wav: encodeWav(samples, SPEECH_RATE),
      samples,
      rate: SPEECH_RATE,
      t0: this.clock.toPage(start / this.rate),
      duration: samples.length / SPEECH_RATE,
      ...levels(samples),
      cut: want > longest,
    };
  }

  /** Forget the clip being recorded (stop, or a page reset). */
  cancel(): void {
    this.startFrame = null;
    this.keep();
  }

  private waitFor(frame: number): Promise<void> {
    return new Promise((resolve) => {
      const w = {
        frame,
        done: () => {
          clearTimeout(timer);
          this.waiters = this.waiters.filter((x) => x !== w);
          resolve();
        },
      };
      const timer = setTimeout(w.done, this.opts.waitS * 1000);
      this.waiters.push(w);
    });
  }

  /** The ring keeps everything from the oldest clip still needed. */
  private keep(): void {
    const starts = [...this.finishing, ...(this.startFrame === null ? [] : [this.startFrame])];
    this.ring.keep(starts.length ? Math.min(...starts) : null);
  }

  private frameAt(page: number): number {
    return Math.round(this.clock.toContext(page) * this.rate);
  }
}
