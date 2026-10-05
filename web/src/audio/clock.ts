/**
 * The audio clock on the page clock (plan 4.7). The microphone's samples are numbered on the
 * AudioContext's clock; the hand frames and the words must share `performance.now()`. Pairs of
 * (context time, page time) come from `AudioContext.getOutputTimestamp()` every few seconds; a
 * straight line through the recent pairs maps one clock to the other, and its slope is the
 * drift between them (the session report gives the measured number).
 */

export interface ClockPair {
  /** Seconds on the AudioContext's clock. */
  context: number;
  /** Seconds on the page clock (`performance.now() / 1000`) when that sample was captured. */
  page: number;
}

export class ClockMap {
  private pairs: ClockPair[] = [];
  private slope = 1;
  private offset = 0;

  /** `keep`: how many recent pairs the line goes through. */
  constructor(private readonly keep = 30) {}

  get size(): number {
    return this.pairs.length;
  }

  add(p: ClockPair): void {
    // A pair with an older context time than the last one is a stale reading: skip it.
    const last = this.pairs[this.pairs.length - 1];
    if (last && p.context <= last.context) return;
    this.pairs.push(p);
    if (this.pairs.length > this.keep) this.pairs.shift();
    this.fit();
  }

  /** Forget the pairs (a rough first pair, once real ones arrive). */
  clear(): void {
    this.pairs = [];
  }

  /** Context seconds to page seconds. */
  toPage(context: number): number {
    return this.offset + this.slope * context;
  }

  /** Page seconds to context seconds. */
  toContext(page: number): number {
    return (page - this.offset) / this.slope;
  }

  /** How much faster the page clock runs than the audio clock, in parts per million. */
  get driftPpm(): number {
    return (this.slope - 1) * 1e6;
  }

  /** The pairs' scatter around the line, in milliseconds (the readings' own jitter). */
  get residualMs(): number {
    if (this.pairs.length < 3) return 0;
    const sq = this.pairs.reduce((a, p) => a + (p.page - this.toPage(p.context)) ** 2, 0);
    return Math.sqrt(sq / this.pairs.length) * 1000;
  }

  private fit(): void {
    const ps = this.pairs;
    const last = ps[ps.length - 1];
    const span = last.context - ps[0].context;
    // Under 5 s of pairs the slope is mostly jitter: keep 1 and fit the offset alone.
    if (ps.length < 3 || span < 5) {
      this.slope = 1;
      this.offset = ps.reduce((a, p) => a + (p.page - p.context), 0) / ps.length;
      return;
    }
    const mc = ps.reduce((a, p) => a + p.context, 0) / ps.length;
    const mp = ps.reduce((a, p) => a + p.page, 0) / ps.length;
    let num = 0;
    let den = 0;
    for (const p of ps) {
      num += (p.context - mc) * (p.page - mp);
      den += (p.context - mc) ** 2;
    }
    this.slope = num / den;
    this.offset = mp - this.slope * mc;
  }
}
