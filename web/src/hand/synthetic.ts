/**
 * A made-up hand for screenshots and replays without a camera (plan W4, W13): rahul's pointing
 * hand recorded on 1 Oct 2026 (MediaPipe world landmarks, from deixis-doodle
 * tests/runtime/data/hands_1oct.json, point frame 0), moved so its index fingertip sits at a
 * chosen camera point. The pose is classified from the real world landmarks, so it points.
 *
 * It is labelled synthetic wherever it shows; the real look comes from a real webcam.
 */

import { type Landmark3, INDEX_TIP, WRIST, classifyPointing } from './pose';
import type { HandFrame, HandObs } from './tracker';

export const RAHUL_POINT_1OCT: Landmark3[] = [
  [-0.0624, -0.0295, 0.038], [-0.0314, -0.0329, 0.0364], [-0.011, -0.0196, 0.041],
  [0.0066, -0.007, 0.0382], [0.0165, 0.004, 0.0332], [0.0056, -0.0105, 0.0034],
  [0.0221, -0.0005, 0.0041], [0.0423, 0.0145, 0.0159], [0.056, 0.0286, 0.0306],
  [0.0019, -0.0021, -0.0021], [0.0164, 0.0108, 0.008], [0.0156, 0.0066, 0.0332],
  [0.0123, 0.0016, 0.0561], [-0.0031, 0.0072, -0.0022], [0.0061, 0.012, 0.0107],
  [0.0028, 0.0135, 0.0334], [-0.003, 0.0114, 0.0532], [-0.0134, 0.0077, 0.0045],
  [-0.0044, 0.0176, 0.0152], [-0.0081, 0.0186, 0.0359], [-0.012, 0.0172, 0.0443],
]; // prettier-ignore

const POSE = classifyPointing(RAHUL_POINT_1OCT);

/**
 * The hand with its index tip at camera point (x, y) (unmirrored, 0..1), the finger pointing
 * `upDeg` degrees from straight up in the image, about `heightFrac` of the image tall.
 */
export function syntheticHand(
  x: number,
  y: number,
  camAspect: number,
  upDeg = -25,
  heightFrac = 0.32,
): HandObs {
  const w = RAHUL_POINT_1OCT;
  const tip = w[INDEX_TIP];
  const dir = [tip[0] - w[WRIST][0], tip[1] - w[WRIST][1]];
  const len = Math.hypot(dir[0], dir[1]);
  // Turn the wrist-to-tip line to point up (image -y), then by upDeg.
  const want = ((-90 + upDeg) * Math.PI) / 180;
  const turn = want - Math.atan2(dir[1], dir[0]);
  const [c, s] = [Math.cos(turn), Math.sin(turn)];
  const k = heightFrac / len; // metres to image heights
  const image = w.map(([px, py, pz]) => {
    const dx = px - tip[0];
    const dy = py - tip[1];
    return [x + ((c * dx - s * dy) * k) / camAspect, y + (s * dx + c * dy) * k, pz] as Landmark3;
  });
  return { image, world: w, handedness: 'Right', score: 0.99, pose: POSE };
}

/** One frame with the hand at a camera point, at page time t. */
export function syntheticFrame(t: number, x: number, y: number, width = 1280, height = 720): HandFrame {
  return { t, width, height, hands: [syntheticHand(x, y, width / height)] };
}
