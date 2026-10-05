/**
 * The last 20 s of pointer samples (plan 4.2): the core's `Sample` shape, in table mm, `t` in
 * seconds on the page clock (`performance.now() / 1000`). Raw: the core filters and finds the
 * holds itself.
 */

export interface Sample {
  t: number;
  /** Table mm, or null: no hand, or the ray missed the table and the blocks. */
  point: [number, number] | null;
  pointing: boolean;
}

export class SampleBuffer {
  private readonly items: Sample[] = [];

  constructor(readonly keepS = 20) {}

  push(s: Sample): void {
    // Frames arrive in order; a late one (out of order) is dropped.
    const last = this.items[this.items.length - 1];
    if (last && s.t <= last.t) return;
    this.items.push(s);
    const cut = s.t - this.keepS;
    let n = 0;
    while (n < this.items.length && this.items[n].t < cut) n++;
    if (n) this.items.splice(0, n);
  }

  /** Samples with t0 <= t <= t1, oldest first. */
  between(t0: number, t1: number): Sample[] {
    return this.items.filter((s) => s.t >= t0 && s.t <= t1);
  }

  get length(): number {
    return this.items.length;
  }

  get latest(): Sample | undefined {
    return this.items[this.items.length - 1];
  }

  clear(): void {
    this.items.length = 0;
  }
}
