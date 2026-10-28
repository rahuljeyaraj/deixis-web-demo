/**
 * The pointing pose (plan 4.1, 6.3): a port of deixis-doodle's `classify_pointing`
 * (src/pointdraw/vision/hands.py, commit 7a7e9cc), with the same thresholds. Tested against
 * Python's own answers on the same cases (tests/fixtures/pointing-cases.json).
 *
 * It uses MediaPipe's 3D world landmarks (metres, hand-centred), so it doesn't depend on how
 * the camera sees the hand.
 */

export type Landmark3 = [number, number, number];

/** MediaPipe hand landmark indices. */
export const WRIST = 0;
export const THUMB_TIP = 4;
export const INDEX_TIP = 8;
export const FINGERS = {
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
} as const;
export type Finger = keyof typeof FINGERS;

/**
 * Thresholds for finger states. A finger is scored on two cues:
 * - reach: |wrist → tip| / |wrist → PIP|, above about 1.2 when straight, below 1.0 curled;
 * - bend: total bend at the PIP and DIP joints, near 0 degrees when straight.
 */
export interface PoseParams {
  extended_reach: number;
  extended_bend_deg: number;
  curled_reach: number;
  curled_bend_deg: number;
  /** The middle finger must curl; ring or pinky may lag. */
  min_curled_of_ring_pinky: number;
  /** An index bent a little (neither straight nor curled) counts if it reaches past the middle. */
  half_index_ok: boolean;
  /** A hooked index counts if it reaches this much further than the middle (0: off). */
  hooked_index_margin: number;
}

export const DEFAULT_POSE: PoseParams = {
  extended_reach: 1.12,
  extended_bend_deg: 60.0,
  curled_reach: 1.02,
  curled_bend_deg: 90.0,
  min_curled_of_ring_pinky: 1,
  half_index_ok: true,
  hooked_index_margin: 0.15,
};

export interface FingerState {
  reach: number;
  bend_deg: number;
  extended: boolean;
  curled: boolean;
}

export interface PoseResult {
  pointing: boolean;
  fingers: Record<Finger, FingerState>;
}

type V = [number, number, number];

function sub(a: ArrayLike<number>, b: ArrayLike<number>): V {
  return [a[0] - b[0], a[1] - b[1], (a[2] ?? 0) - (b[2] ?? 0)];
}

function norm(v: V): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function angleDeg(u: V, v: V): number {
  const c = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (norm(u) * norm(v) + 1e-12);
  return (Math.acos(Math.max(-1, Math.min(1, c))) * 180) / Math.PI;
}

export function fingerState(lm: ArrayLike<number>[], finger: Finger, p = DEFAULT_POSE): FingerState {
  const [mcp, pip, dip, tip] = FINGERS[finger].map((i) => lm[i]);
  const wrist = lm[WRIST];
  const reach = norm(sub(tip, wrist)) / (norm(sub(pip, wrist)) + 1e-12);
  const bend = angleDeg(sub(pip, mcp), sub(dip, pip)) + angleDeg(sub(dip, pip), sub(tip, dip));
  return {
    reach,
    bend_deg: bend,
    extended: reach >= p.extended_reach && bend <= p.extended_bend_deg,
    curled: reach <= p.curled_reach || bend >= p.curled_bend_deg,
  };
}

/**
 * Pointing = index extended (or bent a little but reaching past the middle finger, or hooked
 * but reaching well past it), middle curled, and ring or pinky curled. The thumb is ignored:
 * people point with the thumb in or out. `lm` is 21 world landmarks (or image landmarks).
 */
export function classifyPointing(lm: ArrayLike<number>[], p = DEFAULT_POSE): PoseResult {
  const fingers = {
    index: fingerState(lm, 'index', p),
    middle: fingerState(lm, 'middle', p),
    ring: fingerState(lm, 'ring', p),
    pinky: fingerState(lm, 'pinky', p),
  };
  const others = Number(fingers.ring.curled) + Number(fingers.pinky.curled);
  const { index, middle } = fingers;
  const half = p.half_index_ok && !index.curled && index.reach > middle.reach;
  const hook = p.hooked_index_margin > 0 && index.reach - middle.reach >= p.hooked_index_margin;
  const pointing =
    (index.extended || half || hook) && middle.curled && others >= p.min_curled_of_ring_pinky;
  return { pointing, fingers };
}
