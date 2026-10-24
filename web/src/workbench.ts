/**
 * The workbench: a look at it (stage 1), and a body that can be told to do something with it
 * (stage 2 of `workbench-requirements.md`).
 *
 * It compiles `workbench_scene.xml`, puts the arm in its resting pose, draws the world from
 * where a person standing at the bench would see it, and **steps the physics**, so a thing that
 * is picked up is really picked up and a thing put down really stays there. No voice and no
 * model yet - those are stages 3 and 4; until then the pointer and five buttons stand in for
 * what will be said out loud.
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
 * - **the arm's own words** when it will not do something (FR15): out of reach, never picked
 *   up, too tall for that bay. The refusal is the robot's, not the page's - the page only
 *   prints what `rig.ts` hands back.
 * - **where what it is holding can go** (rahul, 24 Oct): the moment something is picked up, a
 *   patch of light lands on every spot on the bench and in every bay where the arm could really
 *   put it down. "Wherever i tried to place thing it keeps on saying there is no way to put it
 *   there" is what a reach limit feels like when it cannot be seen - and the limits here are
 *   real and strange. A jar is held upright by a level hand, and a level hand at jar height
 *   cannot work closer in than 280 mm, so a jar can go in the bays and along the back of the
 *   bench and nowhere near the person. Shown, that reads as the arm's reach. Unshown, it reads
 *   as a broken robot. The patches come from `rig.canPut`, the same answer the arm will give,
 *   so the two cannot disagree.
 *
 *   ?still=1    don't step the physics (stage 1's view of the world, for a clean picture)
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
import { Rig, type Done } from './sim/rig';

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
 * flat board.
 *
 * So the spot is lifted a hair and **drawn without asking the depth buffer** (`depthTest`), on
 * top of everything. A polygon offset was tried first and is not enough: it shifts a fixed
 * amount, and on a jar seen at a glancing angle - or on a steep part of a tilted hammer head -
 * the rim still sank inside the thing and the circle vanished, which is what rahul saw. Not
 * asking is also honest here rather than a cheat: the spot is put where a ray from the camera
 * first meets the world, so by construction there is nothing in front of it to hide it.
 */
const SPOT_R_M = 0.009;
const SPOT_LIFT_M = 0.0008;

/**
 * The patches that show where a held thing can be put down: how big each one is and how many
 * there may be.
 *
 * One patch per spot swept, laid flat on the bench or on a bay's board. 20 mm apart with 17 mm
 * squares leaves a hair between them, so the grid reads as a field of light rather than as a
 * solid sheet - and a field of light reads as "about here", which is the truth: the sweep is in
 * 20 mm steps and a spot 10 mm from one of them was never asked about.
 */
const PATCH_STEP_MM = 20;
const PATCH_SIDE_M = 0.017;
const MOST_PATCHES = 4000;

/** The fixed parts of the world, named as a person would name them (FR9). */
const FURNITURE: Record<string, string> = {
  bench_near: 'the bench top',
  bench_far: 'the bench top',
  shelf_left: 'the side of the shelf unit',
  shelf_right: 'the side of the shelf unit',
  shelf_back: 'the back of the shelf unit',
  shelf_board0: 'the board over the bottom bay',
  shelf_board1: 'the board over the middle bay',
  shelf_board2: 'the top of the shelf unit',
  rail: 'the rail the arm runs on',
  rail_guide_near: 'the rail the arm runs on',
  rail_guide_far: 'the rail the arm runs on',
  carriage: 'the arm',
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

/**
 * Which thing each geom belongs to, by geom id - `jar2_3` is part of `jar2`.
 *
 * The page needs this as well as the name, because two things can share a name and only one of
 * them is the one being pointed at. That is the whole demo.
 */
function owners(mj: Mujoco, model: MjModel): (geom: number) => string | null {
  const G = mj.mjtObj.mjOBJ_GEOM.value;
  const ids = new Set(reach.things.map((t) => t.id));
  const cache = new Map<number, string | null>();
  return (geom: number) => {
    const had = cache.get(geom);
    if (had !== undefined) return had;
    const name = mj.mj_id2name(model, G, geom);
    const base = name ? name.replace(/_\d+$/, '') : '';
    const owner = ids.has(base) ? base : null;
    cache.set(geom, owner);
    return owner;
  };
}

/**
 * The field of patches, as one instanced mesh: cheap to draw however many there are.
 *
 * Drawn without asking the depth buffer, like the laser spot, and for the same reason - a patch
 * lying on a board is in the same plane as the board - but **behind** the spot, so pointing at a
 * place still shows where the spot is.
 */
function makePatches(): THREE.InstancedMesh {
  const mesh = new THREE.InstancedMesh(
    new THREE.PlaneGeometry(PATCH_SIDE_M, PATCH_SIDE_M),
    new THREE.MeshBasicMaterial({
      color: 0x51d16a,
      transparent: true,
      opacity: 0.42,
      depthWrite: false,
      depthTest: false,
      side: THREE.DoubleSide,
    }),
    MOST_PATCHES,
  );
  mesh.count = 0;
  mesh.renderOrder = 2;
  mesh.frustumCulled = false;
  mesh.raycast = () => {}; // the ray goes to the world, never to the patches
  return mesh;
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
        depthTest: false,
        // Belt and braces: a circle lying on a surface should be drawn whichever way round its
        // normal came out, and a back face that is culled is a spot that is not there.
        side: THREE.DoubleSide,
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
  const doing = document.querySelector<HTMLElement>('#doing')!;
  /** What the spot is on: a thing to pick up, and a place to put one. */
  const pointed: { at: string | null; place: { xMm: number; yMm: number; zMm: number } | null } = {
    at: null,
    place: null,
  };
  const say = (s: string) => {
    status.textContent = s;
  };

  const params = new URLSearchParams(location.search);
  const start = (params.get('view') ?? 'default') as ViewName;
  const still = params.get('still') === '1';

  say('Loading the arms…');
  const [mj, files] = await Promise.all([
    mujoco(),
    fetchPack('/so101.pack', (f) => say(`Loading the arms… ${Math.round(f * 100)}%`)),
  ]);
  const { model, data } = loadScene(mj, files, 'workbench_scene.xml');

  // The resting pose. MuJoCo's default has the arm stretched flat along +X - straight through
  // the shelf unit - so without this it lies in the bays.
  //
  // `restNow` also sets every control to match, which matters: the keyframe says where the
  // joints *are* and the motors say where they are *held*, and without the second the arm
  // drifts to wherever a control of nought puts it the moment the physics starts.
  mj.mj_resetDataKeyframe(model, data, 0);
  const rig = new Rig(mj, model, data);
  rig.restNow();

  const view = new TableView(canvas, mj, model, { bare: true });
  view.sync(data);
  const nameOf = naming(mj, model);
  const thingOf = owners(mj, model);
  const spot = makeSpot();
  view.scene.add(spot);
  const patches = makePatches();
  view.scene.add(patches);

  /**
   * Light up everywhere the held thing can be put down - or put the light out.
   *
   * Swept once, when something is picked up or let go, rather than every frame: it is some
   * hundreds of reach sums and a footprint test against every other thing, which is nothing
   * once and too much sixty times a second.
   */
  const where = new THREE.Matrix4();
  const showWherePut = () => {
    const places = rig.holding ? rig.wherePut(PATCH_STEP_MM) : [];
    patches.count = Math.min(places.length, MOST_PATCHES);
    for (let i = 0; i < patches.count; i++) {
      const p = places[i];
      const [X, Y] = benchToWorld(p.xMm, p.yMm);
      where.makeTranslation(X, Y, p.zMm / 1000 + 0.0012);
      patches.setMatrixAt(i, where);
    }
    patches.instanceMatrix.needsUpdate = true;
    patches.visible = patches.count > 0;
    return places.length;
  };

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
    // The surface's own direction, **in the world**. Both `normal` and `face.normal` come back
    // in the mesh's own frame, and the meshes here are posed by a matrix copied straight out of
    // the simulation, so either has to be turned by that matrix - not by the mesh's (unused)
    // rotation.
    //
    // Turning only one of the two is what made the spot disappear into things (rahul, 24 Oct).
    // `hit.normal` is the one three.js fills in and so the one that was used, and it was the
    // one left untouched: on anything whose body carries a turn - a tin, a mug, a hammer - the
    // circle was laid at the wrong angle, and where that angle pointed away from the camera it
    // was culled as a back face and vanished. It read as the spot sinking into the thing.
    normal.set(0, 0, 1);
    const raw = hit.normal ?? hit.face?.normal;
    if (raw) normal.copy(raw).transformDirection(hit.object.matrixWorld);
    spot.visible = true;
    spot.position.copy(hit.point).addScaledVector(normal, SPOT_LIFT_M);
    spot.quaternion.copy(quat.setFromUnitVectors(UP, normal));

    const geom = hit.object.userData.geom as number | undefined;
    const label = geom === undefined ? null : nameOf(geom);
    const [x, y] = [hit.point.y, hit.point.x];
    // What the buttons act on: the thing under the spot, and the place under it.
    pointed.at = geom === undefined ? null : thingOf(geom);
    pointed.place = {
      xMm: reach.bench.centreX - x * 1000,
      yMm: y * 1000,
      zMm: hit.point.z * 1000,
    };
    showButtons();
    const atMm = `${(reach.bench.centreX - x * 1000).toFixed(0)}, ${(y * 1000).toFixed(0)}, ${(hit.point.z * 1000).toFixed(0)}`;
    under.innerHTML = label
      ? `<b>${label}</b> &middot; ${atMm} mm`
      : `nothing named here &middot; ${atMm} mm`;
  };

  // ----------------------------------------------------------------- the buttons
  //
  // Standing in for what will be said out loud in stage 4: point at a thing, pick it up; point
  // at a place, put it down there. Every one of them goes to `rig.ts` and prints whatever comes
  // back, refusals included, because the refusal is the robot's answer and not the page's.

  const button = (id: string) => document.querySelector<HTMLButtonElement>(`#${id}`)!;
  const buttons = {
    take: button('take'),
    put: button('put'),
    pot: button('pot'),
    park: button('park'),
    stop: button('stop'),
  };
  const tell = (line: string, no = false) => {
    doing.textContent = line;
    doing.classList.toggle('no', no);
  };
  const showButtons = () => {
    const held = rig.holding !== null;
    buttons.take.disabled = held || rig.busy || !pointed.at;
    buttons.put.disabled = !held || rig.busy || !pointed.place;
    buttons.pot.disabled = !held || rig.busy;
    buttons.park.disabled = rig.busy;
    buttons.stop.disabled = !rig.busy;
  };
  const after = (what: string) => (d: Done) => {
    const spots = showWherePut();
    const andWhere = d.ok && rig.holding ? ` ${spots} places it can go.` : '';
    tell(d.ok ? `${what}.${andWhere}` : `It will not: ${d.why}.`, !d.ok);
    showButtons();
  };
  // The buttons do the same as the keys. They are there to say what the keys are, and because
  // a judge who has not read anything will click something.
  buttons.take.addEventListener('click', () => KEYS.p.go());
  buttons.put.addEventListener('click', () => KEYS.d.go());
  buttons.pot.addEventListener('click', () => KEYS.o.go());
  buttons.park.addEventListener('click', () => KEYS.k.go());
  buttons.stop.addEventListener('click', () => {
    rig.stop();
    tell('Stopped.', true);
    showButtons();
  });
  /**
   * The keys, which are the real way to drive this and the buttons are the labels for them.
   *
   * **A hand cannot point and click a button at the same time** (rahul, 24 Oct). Pointing is the
   * whole demo: the mouse stays on the thing being talked about and the other hand says what to
   * do with it. In stage 4 that is the space bar and a sentence; until the microphone is in, it
   * is one key per sentence, pressed while the pointer stays where it is.
   */
  const KEYS: Record<string, { say: string; go: () => void }> = {
    p: {
      say: 'pick that up',
      go: () => {
        const id = pointed.at;
        if (!id || rig.holding || rig.busy) return;
        const name = reach.things.find((t) => t.id === id)?.what ?? id;
        tell(`Picking up ${name}…`);
        showButtons();
        void rig.take(id).then(after(`Holding ${name}`));
      },
    },
    d: {
      say: 'put it down there',
      go: () => {
        if (!pointed.place || !rig.holding || rig.busy) return;
        tell('Putting it down…');
        showButtons();
        void rig.putOn(pointed.place).then(after('Put down'));
      },
    },
    o: {
      say: 'drop it in the pot',
      go: () => {
        if (!rig.holding || rig.busy) return;
        tell('Dropping it in the pot…');
        showButtons();
        void rig.dropInto('pot').then(after('Dropped in the pot'));
      },
    },
    k: {
      say: 'park',
      go: () => {
        if (rig.busy) return;
        tell('Parking…');
        showButtons();
        void rig.park().then(after('Parked'));
      },
    },
  };
  window.addEventListener('keydown', (e) => {
    // Stop is a key of its own and it never waits for anything (FR27).
    if (e.key === 'Escape' || e.key === ' ') {
      e.preventDefault();
      rig.stop();
      tell('Stopped.', true);
      showButtons();
      return;
    }
    const key = KEYS[e.key.toLowerCase()];
    if (!key) return;
    e.preventDefault();
    key.go();
  });
  showButtons();

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
  //
  // The physics runs on the clock rather than one step a frame, so the arm moves at the same
  // speed on a fast machine as on a slow one; a frame that took too long is cut short rather
  // than caught up, because catching up on a long stall is how a simulation explodes.
  const STEP = model.opt.timestep;
  const MOST = 0.05; // of simulated time in one frame
  let drawnAt = performance.now();
  let frames = 0;
  let since = performance.now();
  let fps = 0;
  const draw = () => {
    const nowMs = performance.now();
    let owed = Math.min(MOST, (nowMs - drawnAt) / 1000);
    drawnAt = nowMs;
    if (!still) {
      while (owed > 0) {
        rig.update();
        mj.mj_step(model, data);
        owed -= STEP;
      }
      view.sync(data);
    }
    view.render();
    frames += 1;
    const now = performance.now();
    if (now - since > 500) {
      fps = Math.round((frames * 1000) / (now - since));
      frames = 0;
      since = now;
      say(
        `${reach.things.length} things, ${model.nbody} bodies, ${model.nu} joints to drive. ` +
          `${fps} frames a second. Drag to turn, scroll to move in and out; ` +
          `space or Esc stops the arm at once.`,
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
    rig,
    THREE,
  });
}

main().catch((e: unknown) => {
  const status = document.querySelector<HTMLElement>('#status');
  if (status) status.textContent = `Could not load the scene: ${String((e as Error)?.message ?? e)}`;
  throw e;
});
