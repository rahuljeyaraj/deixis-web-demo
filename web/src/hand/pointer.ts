/**
 * The finger as the pointer (plan 4.2, 4.6): each hand frame is drawn (ghost hand, dot) and
 * becomes one sample in table mm, from a raycast through the fingertip's canvas point. The
 * camera tracker, a recorded replay and the synthetic hand all feed `onFrame`.
 */

import type { BlockId } from '../sim/arm';
import { type Hit, pointAt } from '../sim/pick';
import type { TableView } from '../sim/render';
import { type ActiveBox, DEFAULT_BOX, onCanvas, toCanvas } from './cursor';
import type { CameraInset, HandOverlay } from './overlay';
import { INDEX_TIP } from './pose';
import { HandRecorder } from './recorder';
import { SampleBuffer } from './samples';
import { type HandFrame, chooseHand } from './tracker';

export class FingerPointer {
  readonly samples = new SampleBuffer();
  readonly recorder = new HandRecorder();
  box: ActiveBox = { ...DEFAULT_BOX };
  /** What is under the fingertip now (null: no hand, off the canvas, or off the table). */
  hit: Hit | null = null;
  pointing = false;
  frames = 0;
  /** The small camera window's dot and frame, once the camera (or the made-up hand) runs. */
  inset: CameraInset | null = null;

  constructor(
    private readonly view: TableView,
    private readonly overlay: HandOverlay,
    private readonly blockOfGeom: Map<number, BlockId>,
  ) {}

  onFrame(f: HandFrame): void {
    this.frames++;
    this.recorder.add(f);
    const hand = chooseHand(f.hands);
    const camAspect = f.width / f.height;
    const c = this.view.canvas;
    const canvasAspect = c ? c.clientWidth / c.clientHeight : 16 / 9;
    this.overlay.draw(hand, f.t, this.box, camAspect);
    this.inset?.draw(hand, this.box, camAspect, canvasAspect);
    this.hit = null;
    this.pointing = !!hand?.pose.pointing;
    if (hand) {
      const [x, y] = hand.image[INDEX_TIP];
      const at = toCanvas(x, y, this.box, camAspect, canvasAspect);
      if (onCanvas(at)) this.hit = pointAt(this.view, at[0], at[1], this.blockOfGeom);
    }
    this.samples.push({
      t: f.t,
      point: this.hit ? [this.hit.x, this.hit.y] : null,
      pointing: this.pointing,
    });
  }

  /** No hand any more (the camera stopped). */
  lost(t: number): void {
    this.overlay.clear();
    this.hit = null;
    this.pointing = false;
    this.samples.push({ t, point: null, pointing: false });
  }
}
