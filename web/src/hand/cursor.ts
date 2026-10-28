/**
 * The fingertip as a cursor (plan 4.6): the middle part of the camera image (the active box)
 * maps to the whole scene canvas, mirrored like a mirror, so nobody has to reach to the
 * frame's edges where MediaPipe loses fingers.
 *
 * The box keeps the canvas's shape, so moving the hand 1 cm sideways or up moves the cursor
 * the same distance on screen.
 */

export interface ActiveBox {
  /** The box's width as a fraction of the camera image's width (the gain is 1 / this). */
  width: number;
  /** The box's centre, as fractions of the camera image (0..1 from the left and the top). */
  cx: number;
  cy: number;
}

export const DEFAULT_BOX: ActiveBox = { width: 0.7, cx: 0.5, cy: 0.5 };

/** The box in camera fractions: left, top, width, height. */
export function boxRect(
  box: ActiveBox,
  camAspect: number,
  canvasAspect: number,
): { x: number; y: number; w: number; h: number } {
  let w = box.width;
  let h = (w * camAspect) / canvasAspect;
  if (h > 0.95) {
    w *= 0.95 / h;
    h = 0.95;
  }
  return { x: box.cx - w / 2, y: box.cy - h / 2, w, h };
}

/**
 * A point in the camera image (MediaPipe's normalised landmark: 0..1 from the left and the
 * top of the unmirrored image) to canvas fractions (0..1 from the left and the top). Points
 * outside the active box land outside 0..1.
 */
export function toCanvas(
  px: number,
  py: number,
  box: ActiveBox,
  camAspect: number,
  canvasAspect: number,
): [number, number] {
  const r = boxRect(box, camAspect, canvasAspect);
  const mx = 1 - px; // mirrored: your hand moves right, the dot moves right
  const rx = 1 - (r.x + r.w); // the box's left edge in the mirrored image
  return [(mx - rx) / r.w, (py - r.y) / r.h];
}

/** The inverse of `toCanvas`: canvas fractions to the camera image point that maps there. */
export function fromCanvas(
  u: number,
  v: number,
  box: ActiveBox,
  camAspect: number,
  canvasAspect: number,
): [number, number] {
  const r = boxRect(box, camAspect, canvasAspect);
  const rx = 1 - (r.x + r.w);
  return [1 - (u * r.w + rx), v * r.h + r.y];
}

export function onCanvas(f: [number, number]): boolean {
  return f[0] >= 0 && f[0] <= 1 && f[1] >= 0 && f[1] <= 1;
}
