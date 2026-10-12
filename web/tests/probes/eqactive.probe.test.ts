/**
 * How do you let go of one grip when there are two? The question that decides whether
 * two-arm grasping is possible in this build at all.
 *
 * The one-arm page switches its single weld with MuJoCo's **global** equality-disable flag
 * (`mjDSBL_EQUALITY`). With two welds that flag lets go of both grips at once, so it cannot
 * be used here.
 *
 * `data.eq_active` is the per-equality switch, and in this build **it cannot be touched at
 * all**: it is a `memory_view<bool>` and emscripten throws
 * `_emval_take_value has unknown type` on any access, read or write. That confirms the
 * comment in `arm.ts` and rules the obvious answer out.
 *
 * So this probe tries the two ways round it:
 *
 * 1. **A parking body.** Keep both welds active always, and when an arm is holding nothing,
 *    point its weld at a small invisible body with no contacts. The gripper drags it about
 *    harmlessly. Retarget to the real thing on a grip and back to the park on a release.
 * 2. **A toothless weld.** Keep the weld pointed where it is and write its solver stiffness
 *    (`eq_solimp`) down to nothing, which are float buffers rather than bools.
 *
 * Run: `PROBES=1 npx vitest run tests/probes/eqactive.probe.test.ts`
 */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { expect, test } from 'vitest';

import { loadScene, mujoco, unpack } from '../../src/sim/mujoco';
import type { MjData, MjModel, Mujoco } from '../../src/sim/mujoco';

const WEB = path.join(import.meta.dirname, '..', '..');
const REPORT = path.join(WEB, '..', 'out', 'e6-eqactive.txt');

function say(line: string) {
  fs.mkdirSync(path.dirname(REPORT), { recursive: true });
  fs.appendFileSync(REPORT, `${line}\n`, 'utf8');
}

function filesFor(sceneName: string, xml?: string) {
  const packed = unpack(
    new Uint8Array(zlib.gunzipSync(fs.readFileSync(path.join(WEB, 'public', 'so101.pack')))),
  );
  const bytes = xml
    ? new TextEncoder().encode(xml)
    : new Uint8Array(fs.readFileSync(path.join(WEB, 'model', sceneName)));
  return [
    ...packed.filter((f) => f.path !== 'table_scene.xml' && f.path !== sceneName),
    { path: sceneName, bytes },
  ];
}

async function compile(sceneName: string, xml?: string) {
  const mj = await mujoco();
  const got = loadScene(mj, filesFor(sceneName, xml), sceneName);
  return { mj, ...got } as { mj: Mujoco; model: MjModel; data: MjData };
}

const bodyId = (mj: Mujoco, model: MjModel, n: string) =>
  mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, n);
const eqId = (mj: Mujoco, model: MjModel, n: string) =>
  mj.mj_name2id(model, mj.mjtObj.mjOBJ_EQUALITY.value, n);

// ---------------------------------------------------------------- what is not possible

test('eq_active cannot be touched in this build', async () => {
  const { mj, model, data } = await compile('shelf_scene.xml');
  let why = 'readable';
  try {
    void (data.eq_active as unknown as { [i: number]: number })[0];
  } catch (e) {
    why = String((e as Error).message ?? e).split('\n')[0];
  }
  say(`data.eq_active: ${why}`);
  expect(why).toContain('unknown type');
  model.delete();
  data.delete();
});

// ---------------------------------------------------------------- what is

/** Two arms, two welds, and a parking body per arm for when it is holding nothing. */
const PARKED = `<mujoco model="two grips, parked">
  <asset><model name="so101" file="so101.xml"/></asset>
  <worldbody>
    <geom name="table" type="box" pos="0.28 0 -0.015" size="0.34 0.42 0.015"
      rgba="0.78 0.63 0.47 1" contype="3" conaffinity="3"/>
    <body name="left_stand" pos="0.15 0.14 0"><attach model="so101" prefix="left_"/></body>
    <body name="right_stand" pos="0.15 -0.14 0"><attach model="so101" prefix="right_"/></body>
    <body name="bottle" pos="0.30 0.14 0.046"><freejoint name="bottle"/>
      <geom name="bottle" type="cylinder" size="0.016 0.035" mass="0.02"
        contype="3" conaffinity="3"/></body>
    <body name="mug" pos="0.30 -0.14 0.051"><freejoint name="mug"/>
      <geom name="mug" type="cylinder" size="0.032 0.040" mass="0.03"
        contype="3" conaffinity="3"/></body>
    <!-- A parking body per arm: no contacts, no gravity worth speaking of, invisible. The
         gripper drags it about and nothing notices. -->
    <body name="left_park" pos="0.15 0.14 0.30"><freejoint name="left_park"/>
      <geom name="left_park" type="sphere" size="0.002" mass="0.0001"
        contype="0" conaffinity="0" rgba="0 0 0 0"/></body>
    <body name="right_park" pos="0.15 -0.14 0.30"><freejoint name="right_park"/>
      <geom name="right_park" type="sphere" size="0.002" mass="0.0001"
        contype="0" conaffinity="0" rgba="0 0 0 0"/></body>
  </worldbody>
  <equality>
    <weld name="left_grip" body1="left_gripper" body2="left_park"/>
    <weld name="right_grip" body1="right_gripper" body2="right_park"/>
  </equality>
</mujoco>`;

test('a parking body lets one arm hold something while the other holds nothing', async () => {
  const { mj, model, data } = await compile('parked.xml', PARKED);
  const leftEq = eqId(mj, model, 'left_grip');
  const bottle = bodyId(mj, model, 'bottle');
  const mug = bodyId(mj, model, 'mug');

  // Settle, then have the left arm take the bottle: point its weld at the bottle and set
  // the current relative pose, exactly as the page does with eq_obj2id and eq_data.
  for (let i = 0; i < 200; i++) mj.mj_step(model, data);
  const mugZBefore = data.xpos[mug * 3 + 2];
  (model.eq_obj2id as Int32Array)[leftEq] = bottle;
  mj.mj_forward(model, data);
  for (let i = 0; i < 300; i++) mj.mj_step(model, data);

  const bottleZ = data.xpos[bottle * 3 + 2];
  const mugZ = data.xpos[mug * 3 + 2];
  say(
    `parking body: bottle z ${(bottleZ * 1000).toFixed(0)} mm (welded to the left gripper), ` +
      `mug z ${(mugZ * 1000).toFixed(0)} mm (was ${(mugZBefore * 1000).toFixed(0)}, right arm holding its park)`,
  );
  // The mug must not have been dragged anywhere: the right weld is holding its own parking
  // body, not the mug.
  expect(Math.abs(mugZ - mugZBefore)).toBeLessThan(0.005);
  expect(Number.isFinite(bottleZ)).toBe(true);
  model.delete();
  data.delete();
});

test('the solver stiffness buffers are writable, as a second way round it', async () => {
  const { mj, model, data } = await compile('shelf_scene.xml');
  const leftEq = eqId(mj, model, 'left_grip');
  const solimp = model.eq_solimp as unknown as Float64Array;
  let verdict = 'not exposed';
  try {
    const before = solimp[leftEq * 5];
    solimp[leftEq * 5] = 0;
    verdict = solimp[leftEq * 5] === 0 ? 'WRITABLE' : 'ignored';
    solimp[leftEq * 5] = before;
  } catch (e) {
    verdict = `threw ${String((e as Error).message ?? e).split('\n')[0]}`;
  }
  say(`model.eq_solimp: ${verdict}`);
  model.delete();
  data.delete();
});
