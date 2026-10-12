/**
 * A look at the end demo's scene, and nothing else (plan E14's first half).
 *
 * The brain of the end demo is finished and measured, but it has no page yet. This is the
 * smallest page that answers "can I see it?": it compiles `shelf_scene.xml`, loads the
 * resting pose, lets the physics settle, and draws it from where the person's eyes are.
 *
 * It drives nothing. No arm control, no pointing, no Nemotron - those are items E10 to E19.
 * What it is for is checking by eye that the shelf the numbers describe is the shelf that is
 * there: twelve things in two rows, four identical bottles, two identical mugs, an arm on
 * each side, and the spot where a thing is handed over on the person's left.
 *
 *   ?view=eye | over | side   where to look from (default: eye)
 *   ?settle=0                 draw the declared pose without stepping the physics
 *
 * Drag to turn the view; the scroll wheel moves in and out (plan E17 wants the camera
 * movable, so a judge can see that nothing is hidden behind the arm).
 */

import * as THREE from 'three';

import { fetchPack, loadScene, mujoco } from './sim/mujoco';
import { TableView } from './sim/render';

/**
 * Table millimetres to world metres, in the **shelf** scene's frame: the origin is on the
 * table top at the middle of the person's edge, +X away from them towards the shelf, +Y to
 * their left. `scripts/make_shelf_scene.py` writes the same conversion, and the generated
 * file's header says so.
 *
 * This is not `table.ts`'s `tableToWorld`: that one is the cubes demo's frame, whose origin
 * is the single arm's base and whose +Y is the other way about. The two scenes keep their own
 * frames on purpose, and nothing may mix them.
 */
function shelfToWorld(xMm: number, yMm: number): [number, number] {
  return [yMm / 1000, -(xMm - 250) / 1000];
}

/** Where the person's eyes are, and the middle of the shelf they are looking at (scene.py). */
const EYE_MM = [250, -300, 350] as const;
const LOOK_MM = [250, 350, 40] as const;

/** The three views. Each is a place to look from and a place to look at, in world metres. */
const VIEWS = {
  eye: { from: [...shelfToWorld(EYE_MM[0], EYE_MM[1]), EYE_MM[2] / 1000] },
  over: { from: [0.3, 0, 0.95] },
  side: { from: [0.1, -0.75, 0.3] },
} as const;

const AT: [number, number, number] = [...shelfToWorld(LOOK_MM[0], LOOK_MM[1]), LOOK_MM[2] / 1000];

async function main(): Promise<void> {
  const canvas = document.querySelector<HTMLCanvasElement>('#scene')!;
  const status = document.querySelector<HTMLElement>('#status')!;
  const say = (s: string) => {
    status.textContent = s;
  };

  const params = new URLSearchParams(location.search);
  const which = (params.get('view') ?? 'eye') as keyof typeof VIEWS;
  const settle = params.get('settle') !== '0';

  say('Loading the arms…');
  const [mj, files] = await Promise.all([
    mujoco(),
    fetchPack('/so101.pack', (f) => say(`Loading the arms… ${Math.round(f * 100)}%`)),
  ]);
  const { model, data } = loadScene(mj, files, 'shelf_scene.xml');

  // The resting pose. MuJoCo's default has both arms stretched flat along +X - straight
  // through the front row - so without this the right arm lies on the remote.
  mj.mj_resetDataKeyframe(model, data, 0);
  // The keyframe says where the joints *are*; the motors say where they are *held*. Without
  // this the arms drift from the resting pose to wherever a control of zero puts them, which
  // is upright over the shelf, and the pose in the scene file turns out to mean nothing.
  const ctrl = data.ctrl as Float64Array;
  const keyQpos = model.key_qpos as Float64Array;
  for (let i = 0; i < model.nu; i++) ctrl[i] = keyQpos[i];
  mj.mj_forward(model, data);
  if (settle) for (let i = 0; i < 600; i++) mj.mj_step(model, data);

  const view = new TableView(canvas, mj, model, { bare: true });
  view.sync(data);

  // Where the camera sits, as a distance and two angles about the point it looks at, so a
  // drag can move it without the three.js orbit controls.
  const from = new THREE.Vector3(...VIEWS[which].from);
  const at = new THREE.Vector3(...AT);
  const off = from.clone().sub(at);
  let radius = off.length();
  let around = Math.atan2(off.y, off.x);
  let up = Math.asin(off.z / radius);

  const aim = () => {
    view.camera.position.set(
      at.x + radius * Math.cos(up) * Math.cos(around),
      at.y + radius * Math.cos(up) * Math.sin(around),
      at.z + radius * Math.sin(up),
    );
    view.camera.lookAt(at);
  };

  let dragging = false;
  let last: [number, number] = [0, 0];
  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    last = [e.clientX, e.clientY];
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointerup', () => {
    dragging = false;
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    around -= (e.clientX - last[0]) * 0.006;
    up = Math.max(-0.2, Math.min(1.4, up + (e.clientY - last[1]) * 0.006));
    last = [e.clientX, e.clientY];
    aim();
  });
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      radius = Math.max(0.25, Math.min(2.0, radius * (1 + Math.sign(e.deltaY) * 0.08)));
      aim();
    },
    { passive: false },
  );

  for (const b of document.querySelectorAll<HTMLButtonElement>('button[data-view]')) {
    b.addEventListener('click', () => {
      const v = new THREE.Vector3(...VIEWS[b.dataset.view as keyof typeof VIEWS].from).sub(at);
      radius = v.length();
      around = Math.atan2(v.y, v.x);
      up = Math.asin(v.z / radius);
      aim();
    });
  }

  const resize = () => {
    const w = canvas.clientWidth;
    const h = Math.round((w * 9) / 16);
    canvas.style.height = `${h}px`;
    view.resize(w, h);
    aim();
  };
  window.addEventListener('resize', resize);
  resize();

  // One frame at a time: nothing is moving, so there is nothing to animate. A draw after
  // every change keeps the page cheap enough to leave open.
  const draw = () => {
    view.sync(data);
    view.render();
    requestAnimationFrame(draw);
  };
  draw();

  say(
    `${model.nbody} bodies, ${model.nu} joints to drive, ${model.neq} grips. ` +
      `Drag to turn, scroll to move in and out.`,
  );
  // Handy from the console while checking the scene by eye.
  Object.assign(window as unknown as Record<string, unknown>, { mj, model, data, view });
}

main().catch((e: unknown) => {
  const status = document.querySelector<HTMLElement>('#status');
  if (status) status.textContent = `Could not load the scene: ${String((e as Error)?.message ?? e)}`;
  throw e;
});
