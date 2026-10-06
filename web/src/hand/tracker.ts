/**
 * Hand tracking in the browser (plan 4.1, 4.7): the webcam and MediaPipe Hand Landmarker in
 * VIDEO mode, two hands, the GPU delegate with a CPU fallback. The video never leaves the
 * browser.
 *
 * Every frame's time is on the page clock, `performance.now()` in seconds: the frame's capture
 * time when the browser gives it (`requestVideoFrameCallback` metadata), else its expected
 * display time, else the time it was handed to MediaPipe.
 */

import { FilesetResolver, HandLandmarker, type HandLandmarkerResult } from '@mediapipe/tasks-vision';
import { type Landmark3, type PoseResult, classifyPointing } from './pose';

export interface HandObs {
  /** 21 landmarks, normalised to the unmirrored camera image (x, y in 0..1; z relative). */
  image: Landmark3[];
  /** 21 landmarks in metres, hand-centred (for the pose). */
  world: Landmark3[];
  /** MediaPipe's label, which assumes an unmirrored image. */
  handedness: string;
  score: number;
  pose: PoseResult;
}

export interface HandFrame {
  /** Seconds on the page clock. */
  t: number;
  width: number;
  height: number;
  hands: HandObs[];
}

const WASM_PATH = '/mediapipe';
const MODEL_PATH = '/models/hand_landmarker.task';

/** The camera we ask for (with the microphone, in one prompt: web demo item 6). */
export const VIDEO_CONSTRAINTS: MediaTrackConstraints = {
  width: { ideal: 1280 },
  height: { ideal: 720 },
  facingMode: 'user',
};

type Loaded = { landmarker: HandLandmarker; delegate: 'GPU' | 'CPU' };
let loading: Promise<Loaded> | null = null;

type VideoFrameMeta = { captureTime?: number; expectedDisplayTime?: number };
type VideoWithRvfc = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, meta: VideoFrameMeta) => void) => number;
};

/** The pointing hand if any, else the most confident one. */
export function chooseHand(hands: HandObs[]): HandObs | null {
  return hands.find((h) => h.pose.pointing) ?? [...hands].sort((a, b) => b.score - a.score)[0] ?? null;
}

export function framesFromResult(r: HandLandmarkerResult, t: number, width: number, height: number): HandFrame {
  const hands: HandObs[] = r.landmarks.map((lms, i) => {
    const image = lms.map((l) => [l.x, l.y, l.z] as Landmark3);
    const worldLms = r.worldLandmarks[i];
    const world = worldLms?.length ? worldLms.map((l) => [l.x, l.y, l.z] as Landmark3) : image;
    const cat = r.handedness[i]?.[0];
    return {
      image,
      world,
      handedness: cat?.categoryName ?? 'Unknown',
      score: cat?.score ?? 1,
      pose: classifyPointing(world),
    };
  });
  return { t, width, height, hands };
}

export class HandTracker {
  onFrame: ((f: HandFrame) => void) | null = null;
  /** Which delegate MediaPipe ended up on. */
  readonly delegate: 'GPU' | 'CPU';
  private running = false;
  private lastMs = -1;

  private constructor(
    readonly video: HTMLVideoElement,
    private readonly landmarker: HandLandmarker,
    delegate: 'GPU' | 'CPU',
    private readonly stream: MediaStream,
  ) {
    this.delegate = delegate;
  }

  /**
   * Download MediaPipe and the hand model, and make the landmarker, without the camera: the
   * page starts this while the start card shows, so "Use my finger" starts at once (item 9).
   * Safe to call more than once.
   */
  static preload(): Promise<Loaded> {
    loading ??= HandTracker.load().catch((e) => {
      loading = null; // try again on the next call
      throw e;
    });
    return loading;
  }

  private static async load(): Promise<Loaded> {
    const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
    const options = (delegate: 'GPU' | 'CPU') => ({
      baseOptions: { modelAssetPath: MODEL_PATH, delegate },
      runningMode: 'VIDEO' as const,
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    try {
      return { landmarker: await HandLandmarker.createFromOptions(fileset, options('GPU')), delegate: 'GPU' };
    } catch (e) {
      console.warn('hand tracking: no GPU delegate, using the CPU', e);
      return { landmarker: await HandLandmarker.createFromOptions(fileset, options('CPU')), delegate: 'CPU' };
    }
  }

  /**
   * Start on the camera: `stream` if the page already has it (one prompt for the camera and
   * the microphone), else ask for the camera alone. Throws if the camera is denied, missing
   * or busy (the page then offers the mouse).
   */
  static async start(video: HTMLVideoElement, stream?: MediaStream): Promise<HandTracker> {
    const [s, { landmarker, delegate }] = await Promise.all([
      stream ?? navigator.mediaDevices.getUserMedia({ video: VIDEO_CONSTRAINTS, audio: false }),
      HandTracker.preload(),
    ]);
    video.srcObject = s;
    video.muted = true;
    video.playsInline = true;
    await video.play();
    const tracker = new HandTracker(video, landmarker, delegate, s);
    tracker.running = true;
    tracker.next();
    return tracker;
  }

  stop(): void {
    this.running = false;
    for (const track of this.stream.getTracks()) track.stop();
    this.landmarker.close();
    loading = null;
  }

  private next(): void {
    if (!this.running) return;
    const v = this.video as VideoWithRvfc;
    if (v.requestVideoFrameCallback) {
      v.requestVideoFrameCallback((now, meta) => this.process(meta.captureTime ?? meta.expectedDisplayTime ?? now));
    } else {
      requestAnimationFrame(() => this.process(performance.now()));
    }
  }

  private process(frameMs: number): void {
    if (!this.running) return;
    const v = this.video;
    // A frame time on another clock origin (seen with requestAnimationFrame in headless
    // Chrome) would break fusion with the words; trust it only if it's within a second.
    const nowMs = performance.now();
    const tMs = Math.abs(frameMs - nowMs) < 1000 ? frameMs : nowMs;
    if (v.readyState >= 2 && v.videoWidth > 0) {
      // VIDEO mode needs strictly increasing milliseconds.
      const ms = Math.max(Math.round(tMs), this.lastMs + 1);
      this.lastMs = ms;
      const result = this.landmarker.detectForVideo(v, ms);
      this.onFrame?.(framesFromResult(result, tMs / 1000, v.videoWidth, v.videoHeight));
    }
    this.next();
  }
}
