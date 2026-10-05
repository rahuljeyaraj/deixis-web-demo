/**
 * The hand drawn on the page (plan 4.6, as changed with rahul on 5 Oct):
 *
 * - `HandOverlay`, over the scene: the fingertip dot, through the same mapping as the cursor
 *   (mirror plus active box), smoothed for drawing only. A faint "ghost hand" from the 21
 *   landmarks is optional (off: the 3D scene reads better without it).
 * - `CameraInset`, over the small mirrored camera window in a corner: the same dot on the real
 *   fingertip, and a thin frame showing the part of the camera that covers the table (the
 *   active box). With no video (the made-up hand) it also draws the hand.
 *
 * The full-size faded webcam behind the scene was dropped: it flattened the table and the arm.
 */

import { type ActiveBox, boxRect, toCanvas } from './cursor';
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

const PINK = '255 64 129';

/** A 2D canvas sized to its element, in CSS pixels. */
class Layer {
  readonly ctx: CanvasRenderingContext2D;

  constructor(readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
  }

  get width(): number {
    return this.canvas.clientWidth;
  }

  get height(): number {
    return this.canvas.clientHeight;
  }

  fit(): void {
    const dpr = Math.min(window.devicePixelRatio, 2);
    const w = Math.round(this.width * dpr);
    const h = Math.round(this.height * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  clear(): void {
    this.ctx.clearRect(0, 0, this.width, this.height);
  }
}

function drawBones(ctx: CanvasRenderingContext2D, pts: (readonly [number, number])[], scale: number): void {
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgb(255 255 255 / 0.5)';
  ctx.lineWidth = 7 * scale;
  ctx.beginPath();
  for (const [a, b] of CONNECTIONS) {
    ctx.moveTo(pts[a][0], pts[a][1]);
    ctx.lineTo(pts[b][0], pts[b][1]);
  }
  ctx.stroke();
  ctx.fillStyle = 'rgb(255 255 255 / 0.65)';
  for (const [x, y] of pts) {
    ctx.beginPath();
    ctx.arc(x, y, 3.5 * scale, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** The dot: filled and glowing while pointing, a ring when not. */
function drawDot(ctx: CanvasRenderingContext2D, x: number, y: number, pointing: boolean, r: number): void {
  ctx.beginPath();
  ctx.arc(x, y, pointing ? r : r * 0.8, 0, Math.PI * 2);
  ctx.lineWidth = Math.max(2, r * 0.27);
  ctx.strokeStyle = 'rgb(255 255 255 / 0.95)';
  ctx.fillStyle = `rgb(${PINK} / ${pointing ? 0.95 : 0})`;
  ctx.shadowColor = `rgb(${PINK} / 0.8)`;
  ctx.shadowBlur = pointing ? r * 1.4 : 0;
  ctx.fill();
  ctx.stroke();
  ctx.shadowBlur = 0;
}

export class HandOverlay {
  private readonly layer: Layer;
  private readonly fx = new OneEuro();
  private readonly fy = new OneEuro();

  constructor(
    canvas: HTMLCanvasElement,
    /** Also draw a faint ghost hand around the dot. */
    public ghost = false,
  ) {
    this.layer = new Layer(canvas);
  }

  resize(): void {
    this.layer.fit();
  }

  clear(): void {
    this.layer.clear();
    this.fx.reset();
    this.fy.reset();
  }

  /** Draw the chosen hand (or nothing) for a frame at time t (seconds). */
  draw(hand: HandObs | null, t: number, box: ActiveBox, camAspect: number): void {
    const { ctx, width: w, height: h } = this.layer;
    this.layer.clear();
    if (!hand) {
      this.fx.reset();
      this.fy.reset();
      return;
    }
    const pts = hand.image.map(([x, y]) => {
      const [u, v] = toCanvas(x, y, box, camAspect, w / h);
      return [u * w, v * h] as const;
    });
    if (this.ghost) drawBones(ctx, pts, 1);
    const tip = pts[INDEX_TIP];
    drawDot(ctx, this.fx.filter(tip[0], t), this.fy.filter(tip[1], t), hand.pose.pointing, 11);
  }
}

export class CameraInset {
  private readonly layer: Layer;

  constructor(
    canvas: HTMLCanvasElement,
    /** Draw the hand itself too (there's no video behind: the made-up hand). */
    public drawHand = false,
  ) {
    this.layer = new Layer(canvas);
  }

  resize(): void {
    this.layer.fit();
  }

  /**
   * Draw the active box and the fingertip over the mirrored camera image. `sceneAspect` is
   * the big view's shape (the active box keeps it).
   */
  draw(hand: HandObs | null, box: ActiveBox, camAspect: number, sceneAspect: number): void {
    const { ctx, width: w, height: h } = this.layer;
    this.layer.clear();
    // Mirrored, like the video under it.
    const at = (x: number, y: number) => [(1 - x) * w, y * h] as const;

    const r = boxRect(box, camAspect, sceneAspect);
    const [bx, by] = at(r.x + r.w, r.y);
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgb(255 255 255 / 0.85)';
    ctx.strokeRect(bx, by, r.w * w, r.h * h);
    ctx.setLineDash([]);

    if (!hand) return;
    const pts = hand.image.map(([x, y]) => at(x, y));
    if (this.drawHand) drawBones(ctx, pts, 0.45);
    const [tx, ty] = pts[INDEX_TIP];
    drawDot(ctx, tx, ty, hand.pose.pointing, 6);
  }
}
