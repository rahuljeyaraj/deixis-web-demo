/**
 * A One Euro filter (Casiez et al., 2012) for the dot the page draws only. The samples sent to
 * the core stay raw; the core runs its own filter (plan 4.6).
 */

class LowPass {
  private y: number | null = null;
  filter(x: number, alpha: number): number {
    this.y = this.y === null ? x : alpha * x + (1 - alpha) * this.y;
    return this.y;
  }
  get last(): number | null {
    return this.y;
  }
  reset(): void {
    this.y = null;
  }
}

function alpha(cutoff: number, dt: number): number {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

export class OneEuro {
  private readonly x = new LowPass();
  private readonly dx = new LowPass();
  private t: number | null = null;

  constructor(
    private readonly minCutoff = 1.5,
    private readonly beta = 0.02,
    private readonly dCutoff = 1.0,
  ) {}

  /** Filter one value at time t (seconds). */
  filter(value: number, t: number): number {
    const prev = this.x.last;
    if (this.t === null || prev === null || t <= this.t) {
      this.t = t;
      return this.x.filter(value, 1);
    }
    const dt = t - this.t;
    this.t = t;
    const d = this.dx.filter((value - prev) / dt, alpha(this.dCutoff, dt));
    return this.x.filter(value, alpha(this.minCutoff + this.beta * Math.abs(d), dt));
  }

  reset(): void {
    this.x.reset();
    this.dx.reset();
    this.t = null;
  }
}
