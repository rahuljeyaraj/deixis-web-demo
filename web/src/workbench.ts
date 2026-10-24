/**
 * A look at the workbench, and nothing else (stage 1 of `workbench-requirements.md`).
 *
 * The smallest page that answers "can I see it?": it compiles `workbench_scene.xml`, loads the
 * arms' resting pose, and draws the world from where a person standing at the bench would see
 * it. No physics, no voice, no model - those are stages 2 to 4.
 *
 * What it does have is the pointer, because the pointer is part of the world rather than part of
 * the plumbing:
 *
 * - **a free camera** (FR12), with a default view and three presets. Orbiting is not a
 *   convenience here: the back row of the bottom bay is hidden from the default view, and
 *   finding it by moving the camera is the thing Act 4 is built on.
 * - **the laser spot** (FR7, FR8): a ray from wherever the camera is, through the cursor, onto
 *   the first thing it meets - and a filled circle lying on that surface at whatever angle the
 *   surface runs, on a curved jar as readily as on a flat board. The world is solid, so the
 *   depth is never guessed: the hit is already x, y and height.
 * - **the name under the spot** (FR9), quietly, so a person knows what they are about to refer
 *   to before they speak.
 *
 *   ?settle=1   step the physics for a second, to see what is not yet held in place
 *   ?view=default | over | side | bays
 *
 * The names come from `workbench_reach.json`, which the generator writes from the same Python
 * the scene comes from, so a thing cannot be called one thing here and another in the model.
 */

import * as THREE from 'three';

import reach from '../model/workbench_reach.json';
import { fetchPack, loadScene, mujoco } from './sim/mujoco';
import type { MjModel, Mujoco } from './sim/mujoco';
import { LOOK, TableView } from './sim/render';

/**
 * Bench millimetres to world metres, in the **workbench** scene's frame: the origin is on the
 * bench top at the middle of the person's edge, +X away from them towards the shelf, +Y to
 * their left. `scripts/make_workbench_scene.py` writes the same conversion and the generated
 * file's header says so.
 *
 * This is neither `table.ts`'s `tableToWorld` (the cubes demo, one arm at the origin) nor the
 * end demo's, whose centre line is 250. Each scene keeps its own frame, and nothing may mix
 * them.
 */
function benchToWorld(xMm: number, yMm: number): [number, number] {
  return [yMm / 1000, -(xMm - reach.bench.centreX) / 1000];
}

/**
 * Where the person's eye is for the default view, and what it looks at. Both come from the
 * generated file rather than being written here, because the fourth jar's place across the
 * shelf is worked out from this eye (`behind_from_the_eye` in `workbench/scene.py`): a second
 * copy that drifted would quietly un-hide the thing Act 4 is built on.
 */
const EYE_MM = reach.eye;
const LOOK_MM = reach.lookAt;

/**
 * The four views. Each is a place to look from, in world metres.
 *
 * `default` is the one that matters: from there the jars in the front row hide the one behind
 * them completely, and the board above cuts off any look over their tops. Going **up** does not
 * find it - going **round** does, which is what the free camera is for and what
 * `tests/probes/visible.probe.test.ts` measures. `bays` is lower and nearer, square on to the
 * two openings; `side` is the orbit that finds the back row.
 */
const VIEWS = {
  default: [...benchToWorld(EYE_MM[0], EYE_MM[1]), EYE_MM[2] / 1000],
  over: [0.30, 0, 1.15],
  // Round to one side, by the amount that finds the back row: from here the jar behind the
  // middle one is 38 per cent visible, against nothing at all from the default view
  // (`tests/probes/visible.probe.test.ts`). It is a small move, which is the point - a judge
  // gets there by dragging, and this button is only a shortcut to it.
  side: [...benchToWorld(120, -350), 0.40],
  bays: [...benchToWorld(350, -120), 0.26],
} as const;

type ViewName = keyof typeof VIEWS;

const AT: [number, number, number] = [
  ...benchToWorld(LOOK_MM[0], LOOK_MM[1]),
  LOOK_MM[2] / 1000,
];

/**
 * How big the laser spot is, and how it is kept on the surface it lies on, metres.
 *
 * 5 mm was the first try and it is a dot: on a 32 mm jar, at the size this page is looked at,
 * it is about six pixels and reads as a speck rather than as a pointer (NFR1). 9 mm is a laser
 * spot on a jar.
 *
 * At that size a flat circle cannot lie on a curved thing without its rim going inside: the rim
 * sinks by about r squared over twice the radius of the curve, which on the tightest curve here
 * - a glue tube at 13 mm - is 3.1 mm. Floating the whole spot 3 mm off would make it hover on a
 * flat board. So it is lifted a little and drawn with a depth offset instead, which lets the
 * rim be inside the jar and still be drawn: the circle reads as painted on the surface, which
 * is what a laser spot does.
 */
const SPOT_R_M = 0.009;
const SPOT_LIFT_M = 0.0008;

/** The fixed parts of the world, named as a person would name them (FR9). */
const FURNITURE: Record<string, string> = {
  bench: 'the bench top',
  shelf_left: 'the side of the shelf unit',
  shelf_right: 'the side of the shelf unit',
  shelf_back: 'the back of the shelf unit',
  shelf_board0: 'the board over the bottom bay',
  shelf_board1: 'the board over the middle bay',
  shelf_board2: 'the top of the shelf unit',
  left_stand: 'the left arm',
  right_stand: 'the right arm',
};

/**
 * What to call whatever a geom belongs to.
 *
 * A thing's own geom carries its id; its extra shapes - a hammer's head, a jar's lid - are
 * `id_1`, `id_2`, and must give the same name, because a person pointing at the head of the
 * hammer is pointing at the hammer. The arms' geoms are mostly unnamed inside so101.xml, so
 * for those the walk goes up the body tree until it reaches a body this page knows.
 */
function naming(mj: Mujoco, model: MjModel): (geom: number) => string | null {
  const G = mj.mjtObj.mjOBJ_GEOM.value;
  const B = mj.mjtObj.mjOBJ_BODY.value;
  const whatOf = new Map(reach.things.map((t) => [t.id, t.what]));
  const bodyOf = model.geom_bodyid as Int32Array;
  const parentOf = model.body_parentid as Int32Array;

  const named = (name: string | null): string | null => {
    if (!name) return null;
    if (FURNITURE[name]) return FURNITURE[name];
    const base = name.replace(/_\d+$/, '');
    return whatOf.get(base) ?? null;
  };

  const cache = new Map<number, string | null>();
  return (geom: number) => {
    const had = cache.get(geom);
    if (had !== undefined) return had;
    let label = named(mj.mj_id2name(model, G, geom));
    for (let b = bodyOf[geom]; !label && b > 0; b = parentOf[b]) {
      label = named(mj.mj_id2name(model, B, b));
    }
    cache.set(geom, label);
    return label;
  };
}

/** The laser spot: a filled amber circle with a dark edge, lying on whatever it is put on. */
function makeSpot(): THREE.Group {
  const spot = new THREE.Group();
  const disc = (radius: number, colour: number, lift: number) => {
    const m = new THREE.Mesh(
      new THREE.CircleGeometry(radius, 36),
      new THREE.MeshBasicMaterial({
        color: colour,
        transparent: true,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: -4,
        polygonOffsetUnits: -8,
      }),
    );
    m.position.z = lift;
    m.renderOrder = colour === LOOK.edge ? 3 : 4;
    m.raycast = () => {}; // the ray goes to the world, never to the spot
    spot.add(m);
  };
  disc(SPOT_R_M * 1.3, LOOK.edge, 0);
  disc(SPOT_R_M, LOOK.amber, 0.0002);
  spot.visible = false;
  return spot;
}

async function main(): Promise<void> {
  const canvas = document.querySelector<HTMLCanvasElement>('#scene')!;
  const status = document.querySelector<HTMLElement>('#status')!;
  const under = document.querySelector<HTMLElement>('#under')!;
  const say = (s: string) => {
    status.textContent = s;
  };

  const params = new URLSearchParams(location.search);
  const start = (params.get('view') ?? 'default') as ViewName;
  const settle = params.get('settle') === '1';

  say('Loading the arms…');
  const [mj, files] = await Promise.all([
    mujoco(),
    fetchPack('/so101.pack', (f) => say(`Loading the arms… ${Math.round(f * 100)}%`)),
  ]);
  const { model, data } = loadScene(mj, files, 'workbench_scene.xml');

  // The resting pose. MuJoCo's default has both arms stretched flat along +X - straight through
  // the shelf unit - so without this they lie in the bays.
  mj.mj_resetDataKeyframe(model, data, 0);
  // The keyframe says where the joints *are*; the motors say where they are *held*. Without
  // this the arms drift to wherever a control of zero puts them, which is upright over the
  // bench, and the pose in the scene file turns out to mean nothing.
  const ctrl = data.ctrl as Float64Array;
  const keyQpos = model.key_qpos as Float64Array;
  for (let i = 0; i < model.nu; i++) ctrl[i] = keyQpos[i];
  mj.mj_forward(model, data);
  // Stage 1 draws the pose the numbers describe and does not step the physics: two of the
  // nineteen things hang on hooks, and nothing holds them there yet (hanging is stage 2).
  if (settle) for (let i = 0; i < 200; i++) mj.mj_step(model, data);

  const view = new TableView(canvas, mj, model, { bare: true });
  view.sync(data);
  const nameOf = naming(mj, model);
  const spot = makeSpot();
  view.scene.add(spot);

  // Where the camera sits, as a distance and two angles about the point it looks at, so a drag
  // can move it without the three.js orbit controls.
  const at = new THREE.Vector3(...AT);
  let radius = 0;
  let around = 0;
  let up = 0;

  const aim = () => {
    view.camera.position.set(
      at.x + radius * Math.cos(up) * Math.cos(around),
      at.y + radius * Math.cos(up) * Math.sin(around),
      at.z + radius * Math.sin(up),
    );
    view.camera.lookAt(at);
  };

  const goTo = (name: ViewName) => {
    const v = new THREE.Vector3(...VIEWS[name]).sub(at);
    radius = v.length();
    around = Math.atan2(v.y, v.x);
    up = Math.asin(v.z / radius);
    aim();
    for (const b of document.querySelectorAll<HTMLButtonElement>('button[data-view]')) {
      b.setAttribute('aria-pressed', String(b.dataset.view === name));
    }
  };
  goTo(start);

  // ------------------------------------------------------------------ the pointer

  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const normal = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const UP = new THREE.Vector3(0, 0, 1);

  /** Put the spot where a ray from the camera through the cursor first meets the world. */
  const pointAt = (clientX: number, clientY: number) => {
    const r = canvas.getBoundingClientRect();
    ndc.set(((clientX - r.left) / r.width) * 2 - 1, -(((clientY - r.top) / r.height) * 2 - 1));
    view.camera.updateMatrixWorld();
    view.scene.updateMatrixWorld();
    ray.setFromCamera(ndc, view.camera);
    const hit = ray.intersectObjects(view.pickable, false)[0];
    if (!hit) {
      spot.visible = false;
      under.textContent = '';
      return;
    }
    // The surface's own direction, in the world. `face.normal` is in the mesh's frame, and the
    // meshes here are posed by a matrix copied straight out of the simulation, so it has to be
    // turned by that matrix - not by the mesh's (unused) rotation.
    normal.set(0, 0, 1);
    if (hit.normal) normal.copy(hit.normal);
    else if (hit.face) normal.copy(hit.face.normal).transformDirection(hit.object.matrixWorld);
    spot.visible = true;
    spot.position.copy(hit.point).addScaledVector(normal, SPOT_LIFT_M);
    spot.quaternion.copy(quat.setFromUnitVectors(UP, normal));

    const geom = hit.object.userData.geom as number | undefined;
    const label = geom === undefined ? null : nameOf(geom);
    const [x, y] = [hit.point.y, hit.point.x];
    const atMm = `${(reach.bench.centreX - x * 1000).toFixed(0)}, ${(y * 1000).toFixed(0)}, ${(hit.point.z * 1000).toFixed(0)}`;
    under.innerHTML = label
      ? `<b>${label}</b> &middot; ${atMm} mm`
      : `nothing named here &middot; ${atMm} mm`;
  };

  // ------------------------------------------------------------------- the camera

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
    if (!dragging) {
      pointAt(e.clientX, e.clientY);
      return;
    }
    around -= (e.clientX - last[0]) * 0.006;
    up = Math.max(-0.25, Math.min(1.45, up + (e.clientY - last[1]) * 0.006));
    last = [e.clientX, e.clientY];
    aim();
  });
  canvas.addEventListener('pointerleave', () => {
    spot.visible = false;
    under.textContent = '';
  });
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      radius = Math.max(0.3, Math.min(2.4, radius * (1 + Math.sign(e.deltaY) * 0.08)));
      aim();
    },
    { passive: false },
  );

  for (const b of document.querySelectorAll<HTMLButtonElement>('button[data-view]')) {
    b.addEventListener('click', () => goTo(b.dataset.view as ViewName));
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

  // One frame after another, so that the frame rate can be read off while orbiting (NFR3).
  let frames = 0;
  let since = performance.now();
  let fps = 0;
  const draw = () => {
    view.render();
    frames += 1;
    const now = performance.now();
    if (now - since > 500) {
      fps = Math.round((frames * 1000) / (now - since));
      frames = 0;
      since = now;
      say(
        `${reach.things.length} things, ${model.nbody} bodies, ${model.nu} joints to drive. ` +
          `${fps} frames a second. Drag to turn, scroll to move in and out.`,
      );
    }
    requestAnimationFrame(draw);
  };
  draw();

  // Handy from the console while checking the scene by eye, and what
  // `scripts/workbench-spot.mjs` drives the pointer with: it aims at a part of a thing the
  // camera can really see, which takes the same ray this page casts.
  Object.assign(window as unknown as Record<string, unknown>, {
    mj,
    model,
    data,
    view,
    reach,
    THREE,
  });
}

main().catch((e: unknown) => {
  const status = document.querySelector<HTMLElement>('#status');
  if (status) status.textContent = `Could not load the scene: ${String((e as Error)?.message ?? e)}`;
  throw e;
});
