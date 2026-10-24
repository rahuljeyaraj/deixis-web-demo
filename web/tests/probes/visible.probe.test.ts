/**
 * Stage 1's checklist, measured: **what can actually be seen, from where?**
 *
 * Three of the items at the end of stage 1 are about seeing, and none of them is safe to settle
 * by eye:
 *
 * - the back row is invisible from the default view, and findable by orbiting;
 * - the three jars are indistinguishable from each other from the default view;
 * - both arms rest somewhere that leaves the middle of the frame clear.
 *
 * All three are the same measurement. For each thing, rays are cast from the eye at a grid of
 * points over that thing's own bounding box, and the share of them that reach **it** rather
 * than something in front of it is how visible it is. The rays go against the real scene, with
 * the real shelf and the real arms in it, through the same three.js raycaster the page's laser
 * spot uses - so this measures what a judge will see, not what a bounding sphere suggests.
 *
 * It then sweeps the arms' resting pose and reports the one that hides least, because "out of
 * the way" is a number here and not a matter of taste. The end demo measured its resting pose
 * for the same reason, and found that facing forward left 5 mm of clearance.
 *
 * Run: `PROBES=1 npx vitest run tests/probes/visible.probe.test.ts`
 * Report: `out/stage1-visible.txt`
 */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import * as THREE from 'three';
import { expect, test } from 'vitest';

import { loadScene, mujoco, unpack } from '../../src/sim/mujoco';
import type { MjData, MjModel, Mujoco, PackFile } from '../../src/sim/mujoco';
import { TableView } from '../../src/sim/render';

const WEB = path.join(import.meta.dirname, '..', '..');
const PACK = path.join(WEB, 'public', 'so101.pack');
const SCENE = path.join(WEB, 'model', 'workbench_scene.xml');
const REPORT = path.join(WEB, '..', 'out', 'stage1-visible.txt');

interface Reach {
  bench: { centreX: number };
  eye: [number, number, number];
  things: { id: string; what: string; at: [number, number, number]; bay: number | null }[];
}
const REACH = JSON.parse(
  fs.readFileSync(path.join(WEB, 'model', 'workbench_reach.json'), 'utf8'),
) as Reach;

/**
 * The page's default view, and the one you get by orbiting up a little, in bench millimetres.
 *
 * Taken from the generated file rather than written here, because the fourth jar's place is
 * worked out from this eye (`behind_from_the_eye` in `workbench/scene.py`): a copy of the eye
 * that drifted from the real one would quietly stop measuring the thing it is here to measure.
 */
const EYE_MM = REACH.eye;
/**
 * And the views a judge gets by orbiting a little. Up, and round to either side.
 *
 * Up is the obvious one and it is not the one that works: with the jars tall enough to fill
 * their bay, raising the eye only looks further down onto the lids. What finds the back row is
 * going **round**, which is what a person does when they want to see past something - and it is
 * the move the free camera is there for.
 */
const ORBITS: { name: string; at: readonly number[] }[] = [
  { name: 'up 120 mm', at: [REACH.eye[0], REACH.eye[1], REACH.eye[2] + 120] },
  { name: 'round to the left', at: [40, -420, 420] },
  { name: 'round to the right', at: [660, -420, 420] },
];

/** Poses to try for the arms at rest: the shoulder's turn, then how the rest of it folds. */
const PAN_RAD = [-1.8, -1.4, -1.0, -0.5, 0.0, 0.5, 1.0, 1.4, 1.8];
const FOLDS: { name: string; q: [number, number, number, number] }[] = [
  { name: 'folded back', q: [-1.0, 1.69, 1.0, 0.0] },
  { name: 'folded down', q: [-1.5, 1.69, 1.2, 0.0] },
  { name: 'crouched', q: [-1.4, 1.69, 1.65, 0.0] },
  { name: 'nose down', q: [-1.74, 1.69, 1.0, 0.0] },
  { name: 'curled', q: [-1.74, 1.69, 1.65, 0.0] },
  { name: 'laid back', q: [1.0, -1.69, -1.0, 0.0] },
  { name: 'stood up', q: [-0.6, 1.3, 0.9, 0.0] },
];

/** How many rays per thing: a 4 x 4 x 4 grid over its own box, less the middle of each face. */
const GRID = 4;

function say(line: string) {
  fs.mkdirSync(path.dirname(REPORT), { recursive: true });
  fs.appendFileSync(REPORT, `${line}\n`, 'utf8');
}

function benchToWorld(xMm: number, yMm: number): [number, number] {
  return [yMm / 1000, -(xMm - REACH.bench.centreX) / 1000];
}

function files(): PackFile[] {
  const packed = unpack(new Uint8Array(zlib.gunzipSync(fs.readFileSync(PACK))));
  return [
    ...packed.filter((f) => !f.path.endsWith('_scene.xml')),
    { path: 'workbench_scene.xml', bytes: new Uint8Array(fs.readFileSync(SCENE)) },
  ];
}

interface Rig {
  mj: Mujoco;
  model: MjModel;
  data: MjData;
  view: TableView;
  /** The qpos addresses of each arm's five joints. */
  arms: number[][];
  /** Which thing each geom belongs to, by geom id. */
  ownerOf: Map<number, string>;
  /** Every geom of a thing, so its own box can be worked out. */
  boxOf: Map<string, THREE.Box3>;
}

function rig(mj: Mujoco, model: MjModel, data: MjData, view: TableView): Rig {
  const J = mj.mjtObj.mjOBJ_JOINT.value;
  const G = mj.mjtObj.mjOBJ_GEOM.value;
  const qa = model.jnt_qposadr as Int32Array;
  const joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'];
  const arms = ['left_', 'right_'].map((p) =>
    joints.map((n) => qa[mj.mj_name2id(model, J, `${p}${n}`)]),
  );
  const ids = new Set(REACH.things.map((t) => t.id));
  const ownerOf = new Map<number, string>();
  for (let g = 0; g < model.ngeom; g++) {
    const name = mj.mj_name2id && mj.mj_id2name(model, G, g);
    if (!name) continue;
    const base = name.replace(/_\d+$/, '');
    if (ids.has(base)) ownerOf.set(g, base);
  }
  return { mj, model, data, view, arms, ownerOf, boxOf: new Map() };
}

/** Each thing's box in the world, from the meshes that draw it. */
function boxes(r: Rig): void {
  r.boxOf.clear();
  for (const mesh of r.view.pickable) {
    const geom = (mesh as THREE.Mesh).userData.geom as number | undefined;
    const owner = geom === undefined ? undefined : r.ownerOf.get(geom);
    if (!owner) continue;
    mesh.updateMatrixWorld();
    const box = new THREE.Box3().setFromObject(mesh);
    const had = r.boxOf.get(owner);
    r.boxOf.set(owner, had ? had.union(box) : box);
  }
}

/**
 * The share of a thing that can be seen from a point.
 *
 * Rays are cast at a grid over the thing's own box, and counted twice: once against the whole
 * scene, and once against **only that thing**. The answer is the first over the second.
 *
 * Dividing matters. A craft knife is a sliver inside a box that is mostly air, so most rays at
 * its box miss it however clear the view is; counting those as "hidden" marked a knife lying
 * alone in the open as 22 per cent visible. What is wanted is how much of what *could* be seen
 * of a thing is actually seen, and that is this ratio.
 */
function visible(r: Rig, id: string, eye: THREE.Vector3): number {
  const box = r.boxOf.get(id);
  if (!box) return 0;
  const mine = r.view.pickable.filter((o) => {
    const geom = (o as THREE.Mesh).userData.geom as number | undefined;
    return geom !== undefined && r.ownerOf.get(geom) === id;
  });
  const ray = new THREE.Raycaster();
  const at = new THREE.Vector3();
  let seen = 0;
  let could = 0;
  for (let i = 0; i < GRID; i++) {
    for (let j = 0; j < GRID; j++) {
      for (let k = 0; k < GRID; k++) {
        const f = (n: number) => (n + 0.5) / GRID;
        at.set(
          box.min.x + f(i) * (box.max.x - box.min.x),
          box.min.y + f(j) * (box.max.y - box.min.y),
          box.min.z + f(k) * (box.max.z - box.min.z),
        );
        ray.set(eye, at.clone().sub(eye).normalize());
        if (!ray.intersectObjects(mine, false).length) continue;
        could += 1;
        const hit = ray.intersectObjects(r.view.pickable, false)[0];
        const geom = hit?.object.userData.geom as number | undefined;
        if (geom !== undefined && r.ownerOf.get(geom) === id) seen += 1;
      }
    }
  }
  return could ? seen / could : 0;
}

/** Set both arms to a pose (the left one mirrored) and bring the drawing up to date. */
function pose(r: Rig, pan: number, fold: number[]): void {
  for (const [arm, adr] of r.arms.entries()) {
    const q = [arm === 0 ? pan : -pan, ...fold];
    for (let j = 0; j < 5; j++) r.data.qpos[adr[j]] = q[j];
  }
  r.mj.mj_forward(r.model, r.data);
  r.view.sync(r.data);
  // `sync` writes each mesh's own matrix and marks it stale; nothing turns that into a world
  // matrix until somebody asks. The renderer asks every frame, and there are no frames here -
  // so without this the arms stay at the origin and every pose measures the same, which is
  // exactly what the first run of this probe reported.
  r.view.scene.updateMatrixWorld(true);
  boxes(r);
}

function eyeAt(mm: readonly number[]): THREE.Vector3 {
  const [X, Y] = benchToWorld(mm[0], mm[1]);
  return new THREE.Vector3(X, Y, mm[2] / 1000);
}

test('what can be seen from the default view', async () => {
  const mj = await mujoco();
  const { model, data } = loadScene(mj, files(), 'workbench_scene.xml');
  mj.mj_resetDataKeyframe(model, data, 0);
  mj.mj_forward(model, data);
  const view = new TableView(null, mj as Mujoco, model, { bare: true });
  const r = rig(mj, model, data, view);
  const eye = eyeAt(EYE_MM);
  const orbits = ORBITS.map((o) => ({ name: o.name, eye: eyeAt(o.at) }));

  fs.rmSync(REPORT, { force: true });
  say("stage 1's checklist, measured: what can be seen, from where?");
  say(
    `  ${GRID ** 3} rays per thing, over its own box, from the eye at ` +
      `(${EYE_MM.join(', ')}) mm. A ray counts when it reaches that thing first.`,
  );

  // ------------------------------------------------- the arms' resting pose
  say('');
  say('the resting pose: how much of the world does each one hide?');
  say(`  (the share of all ${REACH.things.length} things that can be seen, averaged, and how`);
  say('   many are less than a fifth visible)');
  const score = () => {
    const shares = REACH.things.map((t) => visible(r, t.id, eye));
    return {
      mean: shares.reduce((s, v) => s + v, 0) / shares.length,
      lost: shares.filter((v) => v < 0.2).length,
    };
  };
  let best: { pan: number; fold: string; mean: number; lost: number } | null = null;
  for (const fold of FOLDS) {
    const row: string[] = [];
    for (const pan of PAN_RAD) {
      pose(r, pan, fold.q);
      const got = score();
      row.push(`${pan.toFixed(1)}: ${(got.mean * 100).toFixed(0)}%/${got.lost}`);
      if (!best || got.mean > best.mean) best = { pan, fold: fold.name, ...got };
    }
    say(`  ${fold.name.padEnd(12)} ${row.join('  ')}`);
  }
  const b = best!;
  say('');
  say(
    `  the best of them: ${b.fold}, pan ${b.pan.toFixed(2)} - ` +
      `${(b.mean * 100).toFixed(0)}% of the world visible, ${b.lost} things mostly hidden.`,
  );

  // ----------------------------------- the pose the scene actually rests in
  //
  // The sweep suggests; the scene decides. Several poses sit within a point of each other and
  // a number cannot say which of them reads as an arm at rest rather than an arm about to do
  // something, so the choice stays in the generator - and this is the gate on it.
  const key = model.key_qpos as Float64Array;
  const used: [number, number, number, number, number] = [
    key[0],
    key[1],
    key[2],
    key[3],
    key[4],
  ];
  pose(r, used[0], used.slice(1));
  const mine = score();
  say('');
  say(
    `the pose the scene rests in (REST_ARM and REST_PAN_RAD): pan ${used[0].toFixed(2)}, ` +
      `fold ${used.slice(1).map((v) => v.toFixed(2)).join(' ')}`,
  );
  say(
    `  ${(mine.mean * 100).toFixed(0)}% of the world visible, ${mine.lost} things mostly ` +
      `hidden - ${((mine.mean - b.mean) * 100).toFixed(1)} points off the best found.`,
  );

  // ----------------------------------------- every thing, at the pose in use
  say('');
  say(`every thing at that pose, from the default view and from each orbit:`);
  say(`  ${'thing'.padEnd(12)} ${'here'.padStart(5)} ${orbits.map((o) => o.name.padStart(18)).join(' ')}`);
  const share = new Map<string, number>();
  const orbitShare = new Map<string, number[]>();
  for (const t of REACH.things) {
    const now = visible(r, t.id, eye);
    const round = orbits.map((o) => visible(r, t.id, o.eye));
    share.set(t.id, now);
    orbitShare.set(t.id, round);
    say(
      `  ${t.id.padEnd(12)} ${(now * 100).toFixed(0).padStart(4)}% ` +
        `${round.map((v) => `${(v * 100).toFixed(0).padStart(17)}%`).join(' ')}`,
    );
  }

  // --------------------------------------------------------- the checklist
  say('');
  say('the checklist items this settles:');
  const jar4 = share.get('jar4')!;
  const jar4Round = orbitShare.get('jar4')!;
  const found = Math.max(...jar4Round);
  say(
    `  the back row is hidden from the default view: jar4 is ${(jar4 * 100).toFixed(0)}% visible` +
      ` there, and ${jar4Round.map((v, i) => `${(v * 100).toFixed(0)}% ${orbits[i].name}`).join(', ')}.`,
  );
  say('  So it is found by going round, not by going up - which is the move to coach.');
  const jars = [1, 2, 3].map((n) => share.get(`jar${n}`)!);
  say(
    `  the three jars in front are all in plain view: ${jars
      .map((v) => `${(v * 100).toFixed(0)}%`)
      .join(', ')} - nothing about the view tells them apart.`,
  );
  const lost = REACH.things.filter((t) => share.get(t.id)! < 0.2).map((t) => t.id);
  say(`  the arms leave the frame clear: ${lost.length ? lost.join(', ') : 'nothing'} mostly hidden.`);
  const worst = [...share.entries()].sort((a, c) => a[1] - c[1])[0];
  say(`  the least visible thing is ${worst[0]}, at ${(worst[1] * 100).toFixed(0)}%.`);

  // jar4 is the whole of Act 4: it has to be hidden here and findable by moving.
  expect(jar4, 'jar4 should be hidden from the default view').toBeLessThan(0.05);
  expect(found, 'jar4 should appear from some small orbit').toBeGreaterThan(0.15);
  for (const n of [1, 2, 3]) {
    expect(share.get(`jar${n}`), `jar${n} should be in plain view`).toBeGreaterThan(0.3);
  }
  // Every thing has to be recognisable from the default view without moving the camera.
  for (const [id, seen] of share) {
    if (id === 'jar4') continue; // the one thing that is meant to be hidden
    expect(seen, `${id} should be visible enough to point at`).toBeGreaterThan(0.3);
  }
  // And the pose in use has to be a real contender, not just whatever was typed in.
  expect(mine.mean, 'the resting pose should be close to the best found').toBeGreaterThan(
    b.mean - 0.03,
  );

  model.delete();
  data.delete();
}, 1800000);
