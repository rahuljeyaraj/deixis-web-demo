/**
 * The mouse as the pointer (plan 5.3, mouse mode): while Talk is on, the cursor is sampled
 * like a fingertip (30 a second, even when it doesn't move, so the core can find holds), and
 * each click becomes a hold of its own (`source: "click"`, 0.3 s long at the click).
 */

import type { Hit } from '../sim/pick';
import { type Hold, SampleBuffer } from './samples';

export const CLICK_HOLD_S = 0.3;

export class MousePointer {
  readonly samples = new SampleBuffer();
  private clicks: Hold[] = [];
  /** The cursor as canvas fractions (0..1 from the left and the top), or null when outside. */
  private at: [number, number] | null = null;
  /** What the cursor was on at the last sample (the hover outline). */
  hit: Hit | null = null;

  /** `locate`: the raycast for a canvas point (`pointAt` on the page). */
  constructor(
    private readonly locate: (u: number, v: number) => Hit | null,
    readonly keepS = 20,
  ) {}

  move(u: number, v: number): void {
    this.at = u >= 0 && u <= 1 && v >= 0 && v <= 1 ? [u, v] : null;
  }

  leave(): void {
    this.at = null;
  }

  /** One sample of the cursor at page time `t`. */
  sample(t: number): void {
    const hit = this.at ? this.locate(this.at[0], this.at[1]) : null;
    this.hit = hit;
    this.samples.push({ t, point: hit ? [hit.x, hit.y] : null, pointing: true });
  }

  /** A click at page time `t` on `hit`: a hold centred on the click. */
  click(t: number, hit: Hit): Hold {
    const hold: Hold = {
      t0: t - CLICK_HOLD_S / 2,
      t1: t + CLICK_HOLD_S / 2,
      mean: [hit.x, hit.y],
      spread: [0, 0],
      frame: 'table',
      source: 'click',
    };
    this.clicks = this.clicks.filter((h) => h.t1 >= t - this.keepS);
    this.clicks.push(hold);
    return hold;
  }

  /** Clicks whose hold overlaps t0..t1, oldest first. */
  clicksBetween(t0: number, t1: number): Hold[] {
    return this.clicks.filter((h) => h.t1 >= t0 && h.t0 <= t1);
  }
}
