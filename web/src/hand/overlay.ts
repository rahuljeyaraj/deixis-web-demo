/**
 * The hand drawn over the scene (plan 4.6): a translucent "ghost hand" from the 21 landmarks
 * and a bright fingertip dot, both through the same mapping as the cursor (mirror plus active
 * box). The dot is smoothed for drawing only.
 */

import { type ActiveBox, toCanvas } from './cursor';
import { INDEX_TIP } from './pose';
import { OneEuro } from './smooth';
import type { HandObs } from './tracker';

const CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
]; // prettier-ignore

export class HandOverlay {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly fx = new OneEuro();
  private readonly fy = new OneEuro();

  constructor(readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
  }

  resize(width: number, height: number): void {
    const dpr = Math.min(window.devicePixelRatio, 2);
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  clear(): void {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.fx.reset();
    this.fy.reset();
  }

  /** Draw the chosen hand (or nothing) for a frame at time t (seconds). */
  draw(hand: HandObs | null, t: number, box: ActiveBox, camAspect: number): void {
    const ctx = this.ctx;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);
    if (!hand) {
      this.fx.reset();
      this.fy.reset();
      return;
    }
    const aspect = w / h;
    const pts = hand.image.map(([x, y]) => {
      const [u, v] = toCanvas(x, y, box, camAspect, aspect);
      return [u * w, v * h] as const;
    });

    // The ghost hand: soft white bones and joints.
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'rgb(255 255 255 / 0.45)';
    ctx.lineWidth = 7;
    ctx.beginPath();
    for (const [a, b] of CONNECTIONS) {
      ctx.moveTo(pts[a][0], pts[a][1]);
      ctx.lineTo(pts[b][0], pts[b][1]);
    }
    ctx.stroke();
    ctx.fillStyle = 'rgb(255 255 255 / 0.6)';
    for (const [x, y] of pts) {
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // The fingertip dot: filled while pointing, a ring when not.
    const tip = pts[INDEX_TIP];
    const x = this.fx.filter(tip[0], t);
    const y = this.fy.filter(tip[1], t);
    const pointing = hand.pose.pointing;
    ctx.beginPath();
    ctx.arc(x, y, pointing ? 11 : 9, 0, Math.PI * 2);
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgb(255 255 255 / 0.95)';
    ctx.fillStyle = pointing ? 'rgb(255 64 129 / 0.95)' : 'rgb(255 64 129 / 0.0)';
    ctx.shadowColor = 'rgb(255 64 129 / 0.8)';
    ctx.shadowBlur = pointing ? 16 : 0;
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
}
