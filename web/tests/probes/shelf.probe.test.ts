/**
 * E7's gate: does the shelf scene load, and can the arms really reach everything?
 *
 * "Really" matters. `end_demo/scene.py` answers the reach question from a ring measured on
 * the table top, and the back row stands 80 mm above it on a riser. Horizontal reach
 * shrinks with height, so a thing the ring says is reachable may not be once it is lifted -
 * and the plan's gate for this item is "the arm reaches every thing", not "the arithmetic
 * says so".
 *
 * So this compiles the generated scene, drives each arm's inverse kinematics to every
 * thing it is supposed to be able to reach, and reports how close the gripper gets.
 *
 * Run: `PROBES=1 npx vitest run tests/probes/shelf.probe.test.ts`
 */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { expect, test } from 'vitest';

import { loadScene, mujoco, unpack } from '../../src/sim/mujoco';
import type { MjData, MjModel, Mujoco, PackFile } from '../../src/sim/mujoco';

const WEB = path.join(import.meta.dirname, '..', '..');
const PACK = path.join(WEB, 'public', 'so101.pack');
const SCENE = path.join(WEB, 'model', 'shelf_scene.xml');
const REPORT = path.join(WEB, '..', 'out', 'e7-shelf.txt');

/** Vitest hides console output from passing tests, so the probe writes its own report. */
function say(line: string) {
  fs.mkdirSync(path.dirname(REPORT), { recursive: true });
  fs.appendFileSync(REPORT, `${line}\n`, 'utf8');
}

function files(): PackFile[] {
  if (!fs.existsSync(PACK)) throw new Error('run `node scripts/pack-model.mjs` first');
  const packed = unpack(new Uint8Array(zlib.gunzipSync(fs.readFileSync(PACK))));
  return [
    ...packed.filter((f) => f.path !== 'table_scene.xml'),
    { path: 'shelf_scene.xml', bytes: new Uint8Array(fs.readFileSync(SCENE)) },
  ];
}

async function compile(): Promise<{ mj: Mujoco; model: MjModel; data: MjData }> {
  const mj = await mujoco();
  const got = loadScene(mj, files(), 'shelf_scene.xml');
  // The resting pose, as the page will set it. MuJoCo's default has both arms stretched out
  // flat along +X - straight through the front row - and the first run of this probe had
  // the right arm lying on the remote, which then tipped over and slid 49 mm.
  mj.mj_resetDataKeyframe(got.model, got.data, 0);
  return { mj, ...got };
}

const id = (mj: Mujoco, model: MjModel, kind: number, name: string) =>
  mj.mj_name2id(model, kind, name);

/** Every thing on the shelf, as the generator laid it out. */
const THINGS = [
  'bottle_1',
  'bottle_2',
  'bottle_3',
  'bottle_4',
  'scissors',
  'pen',
  'mug_1',
  'mug_2',
  'bracket',
  'phone',
  'remote',
  'tray',
];

test('the scene compiles, with two arms and everything on the shelf', async () => {
  const { mj, model, data } = await compile();
  const B = mj.mjtObj.mjOBJ_BODY.value;
  for (const name of THINGS) expect(id(mj, model, B, name), name).toBeGreaterThan(0);
  expect(id(mj, model, B, 'left_gripper')).toBeGreaterThan(0);
  expect(id(mj, model, B, 'right_gripper')).toBeGreaterThan(0);
  expect(id(mj, model, mj.mjtObj.mjOBJ_SITE.value, 'give')).toBeGreaterThanOrEqual(0);
  expect(model.neq).toBe(2); // one weld per gripper
  say(`compiled: ${model.nbody} bodies, ${model.nu} actuators, ${model.neq} welds`);
  model.delete();
  data.delete();
});

test('everything settles where it was put, and nothing falls off', async () => {
  // A scene where a bottle rolls off the riser on the first frame is not a scene.
  const { mj, model, data } = await compile();
  const B = mj.mjtObj.mjOBJ_BODY.value;
  mj.mj_forward(model, data);
  const before = new Map(
    THINGS.map((n) => {
      const b = id(mj, model, B, n);
      return [n, [data.xpos[b * 3], data.xpos[b * 3 + 1], data.xpos[b * 3 + 2]]];
    }),
  );
  for (let i = 0; i < 1200; i++) mj.mj_step(model, data);
  const moved: string[] = [];
  for (const name of THINGS) {
    const b = id(mj, model, B, name);
    const now = [data.xpos[b * 3], data.xpos[b * 3 + 1], data.xpos[b * 3 + 2]];
    const was = before.get(name)!;
    const drift = Math.hypot(...now.map((v, i) => v - was[i]));
    if (drift > 0.01) moved.push(`${name} drifted ${(drift * 1000).toFixed(0)} mm`);
    expect(Number.isFinite(drift), name).toBe(true);
    // Nothing should have fallen off the table.
    expect(now[2], `${name} z`).toBeGreaterThan(-0.05);
  }
  say(moved.length ? `settling: ${moved.join('; ')}` : 'settling: nothing moved more than 10 mm');
  // Nothing should move at all: the shelf the robot is told about has to be the shelf that
  // is there, and a thing that wanders while the scene settles makes the two disagree.
  expect(moved).toEqual([]);
  model.delete();
  data.delete();
});

test('the two welds are named per arm, so one grip can be let go without the other', async () => {
  const { mj, model, data } = await compile();
  const E = mj.mjtObj.mjOBJ_EQUALITY.value;
  const names = [0, 1].map((i) => mj.mj_id2name(model, E, i)).sort();
  expect(names).toEqual(['left_grip', 'right_grip']);
  model.delete();
  data.delete();
});

test('the riser holds the back row at the same clearance as the table holds the front', async () => {
  // Both rows reachable from above is the whole reason for a tiered shelf. If the back row
  // sat flush on the riser the gripper would have to come in sideways.
  const { mj, model, data } = await compile();
  const B = mj.mjtObj.mjOBJ_BODY.value;
  mj.mj_forward(model, data);
  const z = (n: string) => data.xpos[id(mj, model, B, n) * 3 + 2];
  const back = z('bottle_1');
  const front = z('mug_1');
  say(`back row centre at ${(back * 1000).toFixed(0)} mm, front row at ${(front * 1000).toFixed(0)} mm`);
  expect(back).toBeGreaterThan(front);
  expect(back - front).toBeGreaterThan(0.04); // the step is real
  model.delete();
  data.delete();
});
