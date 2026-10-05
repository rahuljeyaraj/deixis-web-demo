/**
 * The page. For now: the arm, the table and the blocks; a mouse test of the arm (click a
 * block, then click a spot); the finger as a pointer (plan W4: the webcam, the fingertip dot,
 * samples in table mm); and the voice (plan W5: push-to-talk, the clip on the page clock).
 * The words and the deixis core come next (plan W6 on).
 *
 * `?hands=demo` replays a made-up pointing hand instead of the camera (screenshots, tests).
 */

import { MicCapture } from './audio/capture';
import type { Clip } from './audio/clip';
import { PushToTalk } from './audio/talk';
import { fromCanvas } from './hand/cursor';
import { MousePointer } from './hand/mouse';
import { CameraInset, HandOverlay } from './hand/overlay';
import { FingerPointer } from './hand/pointer';
import { syntheticFrame } from './hand/synthetic';
import { HandTracker } from './hand/tracker';
import { Arm, BLOCK_IDS, type BlockId } from './sim/arm';
import { fetchPack, loadScene, mujoco } from './sim/mujoco';
import { type Hit, pointAt } from './sim/pick';
import { TableView } from './sim/render';
import { BASE_MM, BLOCK_MM, REACH_MM, onMat, tableToWorld } from './sim/table';

/** Random spots on the reach mat, at least 60 mm apart. */
function layout(n: number, rand = Math.random): [number, number][] {
  const out: [number, number][] = [];
  while (out.length < n) {
    const r = REACH_MM.min + 15 + rand() * (REACH_MM.max - REACH_MM.min - 30);
    const a = (rand() - 0.5) * Math.PI * 0.9;
    const x = BASE_MM.x + r * Math.sin(a);
    const y = BASE_MM.y - r * Math.cos(a);
    if (onMat(x, y) && out.every(([u, v]) => Math.hypot(u - x, v - y) > 60)) out.push([x, y]);
  }
  return out;
}

async function start(): Promise<void> {
  const status = document.getElementById('status')!;
  const stage = document.getElementById('stage')!;
  const canvas = document.getElementById('scene') as HTMLCanvasElement;
  const handCanvas = document.getElementById('hand') as HTMLCanvasElement;
  const video = document.getElementById('cam') as HTMLVideoElement;
  const camBox = document.getElementById('cam-box')!;
  const camDots = document.getElementById('cam-dots') as HTMLCanvasElement;
  const cameraBtn = document.getElementById('camera-btn') as HTMLButtonElement;
  const talkBtn = document.getElementById('talk-btn') as HTMLButtonElement;
  const now = () => performance.now() / 1000;
  const say =(text: string) => (status.textContent = text);
  const [mj, files] = await Promise.all([
    mujoco(),
    fetchPack('/so101.pack', (f) => say(`Loading the arm… ${Math.round(f * 100)} %`)),
  ]);
  const { model, data } = loadScene(mj, files);
  const arm = new Arm(mj, model, data);
  const view = new TableView(canvas, mj, model);
  const blockOfGeom = new Map<number, BlockId>([...arm.blocks.values()].map((b) => [b.geom, b.id]));

  const reset = () => {
    arm.goHomeNow();
    layout(BLOCK_IDS.length).forEach(([x, y], i) =>
      arm.putBlock(BLOCK_IDS[i], x, y, Math.random() * Math.PI),
    );
  };
  reset();

  const overlay = new HandOverlay(handCanvas);
  const pointer = new FingerPointer(view, overlay, blockOfGeom);
  const inset = new CameraInset(camDots);
  const fit = () => {
    view.resize(canvas.clientWidth, canvas.clientHeight);
    overlay.resize();
    inset.resize();
  };
  new ResizeObserver(() => inset.resize()).observe(camDots);
  new ResizeObserver(fit).observe(canvas);
  fit();

  // Mouse test: click a block, then a spot (or another block to stack on it).
  let chosen: BlockId | null = null;
  const choose = (id: BlockId | null) => {
    if (chosen) view.glow(arm.blocks.get(chosen)!.geom, false);
    chosen = id;
    if (id) view.glow(arm.blocks.get(id)!.geom, true);
  };
  const hitOf = (e: MouseEvent): Hit | null => {
    const r = canvas.getBoundingClientRect();
    return pointAt(view, (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height, blockOfGeom);
  };
  canvas.addEventListener('click', async (e) => {
    const hit = hitOf(e);
    if (hit) mouse.click(now(), hit);
    if (arm.busy || !hit) return;
    if (!chosen) {
      if (hit.block) {
        choose(hit.block);
        say(`The ${hit.block} block. Now click where it should go.`);
      }
      return;
    }
    const id = chosen;
    const onto = hit.block && hit.block !== id ? arm.block(hit.block) : null;
    const place = onto
      ? { xMm: onto.tableMm[0], yMm: onto.tableMm[1], zMm: onto.heightMm }
      : { xMm: hit.x, yMm: hit.y, zMm: 0 };
    const [gx, gy] = tableToWorld(place.xMm, place.yMm);
    view.showGhost([gx, gy, (place.zMm + BLOCK_MM / 2) / 1000], arm.blocks.get(id)!.geom);
    say(`Moving the ${id} block…`);
    const res = await arm.move(id, place);
    choose(null);
    view.showGhost(null);
    say(
      res.ok
        ? 'Done. Click another block.'
        : res.reason === 'cant_reach'
          ? "I can't reach there. Please pick a spot on the light area."
          : `That didn't work (${res.reason}).`,
    );
  });
  // The finger (plan W4).
  const fingerHelp = 'Point with one finger: the pink dot follows it. Keys: L record, [ ] gain.';
  let tracker: HandTracker | null = null;
  const startCamera = async (): Promise<string> => {
    try {
      tracker = await HandTracker.start(video);
      tracker.onFrame = (f) => pointer.onFrame(f);
      camBox.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
      pointer.inset = inset;
      stage.classList.add('camera-on');
      return fingerHelp;
    } catch (e) {
      console.error(e);
      return `No camera (${e instanceof Error ? e.message : e}); the mouse points instead.`;
    }
  };

  // The mouse as the pointer while the camera is off (plan 5.3): sampled like a fingertip
  // (always, so the 2.5 s before the speech are there too), and each click a hold.
  const mouse = new MousePointer((u, v) => pointAt(view, u, v, blockOfGeom));
  canvas.addEventListener('pointermove', (e) => {
    const r = canvas.getBoundingClientRect();
    mouse.move((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
  });
  canvas.addEventListener('pointerleave', () => mouse.leave());
  setInterval(() => tracker || mouse.sample(now()), 1000 / 30);

  // The voice (plan W5): push-to-talk; the clip with its pre-roll and tail, on the page clock.
  let mic: MicCapture | null = null;
  let micStarting: Promise<string> | null = null;
  let lastClip: Clip | null = null;
  const startMic = (): Promise<string> =>
    (micStarting ??= MicCapture.start().then(
      (m) => {
        mic = m;
        talkBtn.classList.add('ready');
        return 'Hold Talk (or the space bar) and speak.';
      },
      (e) => {
        micStarting = null;
        console.error(e);
        return `No microphone (${e instanceof Error ? e.message : e}).`;
      },
    ));
  const talk = new PushToTalk({
    start: (t) => {
      mic?.begin(t);
      talkBtn.classList.add('on');
      say('Listening… let go to stop (after a quick tap: tap again).');
    },
    stop: (t, reason) => {
      talkBtn.classList.remove('on');
      talkBtn.style.setProperty('--level', '0');
      if (!mic) return;
      if (reason === 'cancel') {
        mic.recorder.cancel();
        say('Stopped listening.');
        return;
      }
      void mic.end(t).then((clip) => {
        lastClip = clip;
        const long = clip.cut || reason === 'long' ? ' That was long, so I stopped listening.' : '';
        say(`Heard ${clip.duration.toFixed(1)} s of sound (loudest ${clip.peakDb.toFixed(0)} dB).${long}`);
      });
    },
  });
  /** A Talk press with no microphone yet asks for it; that press doesn't record. */
  const pressTalk = () => {
    if (mic) talk.press(now());
    else void startMic().then(say);
  };
  talkBtn.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    talkBtn.setPointerCapture(e.pointerId);
    pressTalk();
  });
  talkBtn.addEventListener('pointerup', () => talk.release(now()));
  talkBtn.addEventListener('pointercancel', () => talk.release(now()));
  talkBtn.addEventListener('contextmenu', (e) => e.preventDefault());

  cameraBtn.addEventListener('click', async () => {
    cameraBtn.disabled = true;
    say('Starting the camera and the microphone…');
    const [cam, voice] = await Promise.all([startCamera(), startMic()]);
    if (!tracker) cameraBtn.disabled = false;
    say(`${cam} ${voice}`);
  });
  if (new URLSearchParams(location.search).get('hands') === 'demo') {
    // No video: the small window shows the made-up hand itself, labelled.
    inset.drawHand = true;
    pointer.inset = inset;
    camBox.classList.add('synthetic');
    stage.classList.add('camera-on');
    startSyntheticHand(view, arm, pointer);
  }

  const typing = (e: KeyboardEvent) =>
    e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement;
  window.addEventListener('keyup', (e) => {
    if (e.code === 'Space' && !typing(e)) talk.release(now());
  });
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !typing(e)) {
      e.preventDefault(); // no page scroll, no "click" on a focused button
      if (!e.repeat) pressTalk();
      return;
    }
    if (e.key === 'Escape') {
      arm.stop();
      talk.cancel(now());
    }
    if (e.key === 'r' && !arm.busy) reset();
    if (e.key === '[' || e.key === ']') {
      const w = pointer.box.width + (e.key === '[' ? -0.05 : 0.05);
      pointer.box.width = Math.min(1, Math.max(0.3, Math.round(w * 100) / 100));
      say(`Hand gain: the middle ${Math.round(pointer.box.width * 100)} % of the camera fills the scene.`);
    }
    if (e.key === 'l') {
      if (pointer.recorder.recording) {
        say(`Saved ${pointer.recorder.stopAndSave()} hand frames (landmarks only, no video).`);
      } else {
        pointer.recorder.start(pointer.box, canvas.clientWidth / canvas.clientHeight);
        say('Recording your hand (landmarks only, no video). Press L again to save.');
      }
    }
  });

  // Physics in real time: 5 ms steps, as many as the frame took (at most 0.1 s' worth).
  const dt = model.opt.timestep;
  let last = performance.now();
  let owed = 0;
  const loop = () => {
    // performance.now(), not the frame's timestamp: some browsers (headless Chrome) give
    // frame times on another origin, which would stall the physics.
    const now = performance.now();
    owed += Math.max(0, Math.min((now - last) / 1000, 0.1));
    last = now;
    for (; owed >= dt; owed -= dt) {
      arm.update();
      mj.mj_step(model, data);
    }
    talk.tick(now / 1000);
    if (talk.listening && mic) talkBtn.style.setProperty('--level', Math.min(1, mic.level * 8).toFixed(2));
    view.sync(data);
    view.render();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
  say('Click a block, then click where it should go. R: new blocks. Esc: stop.');
  Object.assign(window, {
    __sim: {
      mj, model, data, view, arm, pointer, mouse, talk,
      tracker: () => tracker,
      mic: () => mic,
      lastClip: () => lastClip,
      startMic,
    },
  });
}

/**
 * A made-up pointing hand, looping (`?hands=demo`): to the red block, hold, to an empty spot
 * on the mat, hold, back. It feeds the same pointer as the camera.
 */
function startSyntheticHand(view: TableView, arm: Arm, pointer: FingerPointer): void {
  const cam = 1280 / 720;
  const canvasOf = (x: number, y: number, zMm: number): [number, number] => {
    const [X, Y] = tableToWorld(x, y);
    const v = view.camera.position.clone().set(X, Y, zMm / 1000).project(view.camera);
    return [(v.x + 1) / 2, (1 - v.y) / 2];
  };
  const toCam = ([u, v]: [number, number]) => {
    const c = view.canvas!;
    return fromCanvas(u, v, pointer.box, cam, c.clientWidth / c.clientHeight);
  };
  const emptySpot = (): [number, number] => {
    const taken = BLOCK_IDS.map((id) => arm.block(id).tableMm);
    for (let r = REACH_MM.min + 20; r < REACH_MM.max; r += 10) {
      for (let a = -1.2; a <= 1.2; a += 0.1) {
        const x = BASE_MM.x + r * Math.sin(a);
        const y = BASE_MM.y - r * Math.cos(a);
        if (onMat(x, y) && taken.every(([u, w]) => Math.hypot(u - x, w - y) > 70)) return [x, y];
      }
    }
    return [BASE_MM.x, BASE_MM.y - 200];
  };
  const lerp = (a: number[], b: number[], s: number) => {
    const e = s * s * (3 - 2 * s);
    return [a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e] as [number, number];
  };
  const t0 = performance.now() / 1000;
  setInterval(() => {
    const t = performance.now() / 1000;
    const red = arm.block('red').tableMm;
    const spot = emptySpot();
    const rest = toCam([0.5, 0.8]);
    const onRed = toCam(canvasOf(red[0], red[1], BLOCK_MM));
    const onSpot = toCam(canvasOf(spot[0], spot[1], 0));
    const k = (t - t0) % 6;
    const jitter = () => (Math.random() - 0.5) * 0.002;
    let p: [number, number];
    if (k < 0.8) p = lerp(rest, onRed, k / 0.8);
    else if (k < 2.2) p = [onRed[0] + jitter(), onRed[1] + jitter()];
    else if (k < 3.0) p = lerp(onRed, onSpot, (k - 2.2) / 0.8);
    else if (k < 4.4) p = [onSpot[0] + jitter(), onSpot[1] + jitter()];
    else if (k < 5.2) p = lerp(onSpot, rest, (k - 4.4) / 0.8);
    else p = rest;
    pointer.onFrame(syntheticFrame(t, p[0], p[1]));
  }, 1000 / 30);
}

start().catch((e) => {
  document.getElementById('status')!.textContent =
    `Could not start: ${e instanceof Error ? e.message : e}`;
  console.error(e);
});
