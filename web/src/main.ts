/**
 * The page. For now: the arm, the table and the blocks; a mouse test of the arm (click a
 * block, then click a spot); and the finger as a pointer (plan W4: the webcam, the ghost hand,
 * the fingertip dot, samples in table mm). Speech and the deixis core come next (plan W5 on).
 *
 * `?hands=demo` replays a made-up pointing hand instead of the camera (screenshots, tests).
 */

import { fromCanvas } from './hand/cursor';
import { HandOverlay } from './hand/overlay';
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
  const cameraBtn = document.getElementById('camera-btn') as HTMLButtonElement;
  const say = (text: string) => (status.textContent = text);
  const [mj, files] = await Promise.all([
    mujoco(),
    fetchPack('/so101.pack', (f) => say(`Loading the arm… ${Math.round(f * 100)} %`)),
  ]);
  const { model, data } = loadScene(mj, files);
  const arm = new Arm(mj, model, data);
  // Transparent, so the webcam shows through behind the table.
  const view = new TableView(canvas, mj, model, { transparent: true });
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
  const fit = () => {
    view.resize(canvas.clientWidth, canvas.clientHeight);
    overlay.resize(canvas.clientWidth, canvas.clientHeight);
  };
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
    if (arm.busy) return;
    const hit = hitOf(e);
    if (!hit) return;
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
  const fingerHelp = 'Point with one finger: the pink dot follows it. Keys: L record, V camera look, [ ] gain.';
  let tracker: HandTracker | null = null;
  cameraBtn.addEventListener('click', async () => {
    cameraBtn.disabled = true;
    say('Starting the camera…');
    try {
      tracker = await HandTracker.start(video);
      tracker.onFrame = (f) => pointer.onFrame(f);
      stage.classList.add('camera-on');
      say(fingerHelp);
    } catch (e) {
      cameraBtn.disabled = false;
      const why = e instanceof Error ? e.message : String(e);
      say(`No camera (${why}). The mouse still works: click a block, then a spot.`);
      console.error(e);
    }
  });
  if (new URLSearchParams(location.search).get('hands') === 'demo') {
    startSyntheticHand(view, arm, pointer);
    cameraBtn.hidden = true;
  }

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') arm.stop();
    if (e.key === 'r' && !arm.busy) reset();
    if (e.key === 'v') stage.classList.toggle('cam-on-top');
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
    view.sync(data);
    view.render();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
  say('Click a block, then click where it should go. R: new blocks. Esc: stop.');
  Object.assign(window, { __sim: { mj, model, data, view, arm, pointer, tracker: () => tracker } });
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
