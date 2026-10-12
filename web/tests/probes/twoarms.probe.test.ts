/**
 * E6, the spike the plan allowed one hour for. Two questions, both now answered.
 *
 * **Can MuJoCo give us two arms?** Yes - with `<attach model="..." prefix="...">`, not with
 * `<replicate>`. This WASM build (3.14.0) does not have `replicate` at all: a bare
 * `<replicate>` is an "unrecognized element", and `count`, `n` and `copies` are all
 * rejected as attributes. `attach` works both at the top level and inside a body, and it
 * is the better answer anyway: it gives readable prefixes (`left_`, `right_`) instead of
 * `_0` and `_1`, and `so101.xml` stays byte-for-byte the file Menagerie ships.
 *
 * Two `<include>`s of the same file fail, correctly: "File 'so101.xml' already included".
 * A generated prefixed copy was the fallback and is not needed; it also trips over
 * `<default>` class names, which `attach` handles by itself.
 *
 * **Will the grip assist hold things that are not cubes?** Yes. The single `<weld>` holds a
 * box, a cylinder and a capsule without creeping, and it can be retargeted from one shape
 * to another through `eq_obj2id`, which is what the page does per object. So the shelf can
 * have bottles and mugs (cylinders) and a pen (capsule) without promising a grip we cannot
 * deliver.
 *
 * These are kept as tests rather than notes because both answers are load-bearing for the
 * scene, and both would fail silently if a MuJoCo upgrade changed them.
 *
 * Run: `PROBES=1 npx vitest run tests/probes/twoarms.probe.test.ts`
 */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { expect, test } from 'vitest';

import { loadScene, mujoco, unpack } from '../../src/sim/mujoco';
import type { MjData, MjModel, Mujoco, PackFile } from '../../src/sim/mujoco';

const PACK = path.join(import.meta.dirname, '..', '..', 'public', 'so101.pack');

function packFiles(): PackFile[] {
  if (!fs.existsSync(PACK)) throw new Error('run `node scripts/pack-model.mjs` first');
  return unpack(new Uint8Array(zlib.gunzipSync(fs.readFileSync(PACK))));
}

/** Compile an XML of our own against the packed model files. */
async function compile(xml: string): Promise<{ mj: Mujoco; model: MjModel; data: MjData }> {
  const mj = await mujoco();
  const files = [
    ...packFiles().filter((f) => f.path !== 'table_scene.xml'),
    { path: 'spike.xml', bytes: new TextEncoder().encode(xml) },
  ];
  return { mj, ...loadScene(mj, files, 'spike.xml') };
}

async function tryCompile(xml: string) {
  try {
    const got = await compile(xml);
    got.model.delete();
    got.data.delete();
    return '';
  } catch (e) {
    return String((e as Error).message ?? e).split('\n')[0];
  }
}

function names(mj: Mujoco, model: MjModel, kind: number, n: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < n; i++) out.push(mj.mj_id2name(model, kind, i));
  return out;
}

const bodyId = (mj: Mujoco, model: MjModel, name: string) =>
  mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, name);

/** The scene the shelf demo needs: two arms, by attachment, a metre apart.
 *
 * The wrapper bodies are `left_stand` and `right_stand`, not `left_base`: the prefix is
 * applied to the attached model's own root body, which is called `base`, so a wrapper named
 * `left_base` collides with it ("repeated name 'left_base' in body"). */
const TWO_ARMS = `<mujoco model="two arms">
  <asset><model name="so101" file="so101.xml"/></asset>
  <worldbody>
    <body name="left_stand" pos="0 -0.45 0"><attach model="so101" prefix="left_"/></body>
    <body name="right_stand" pos="0 0.45 0"><attach model="so101" prefix="right_"/></body>
  </worldbody>
</mujoco>`;

// ---------------------------------------------------------------- two arms

test('attach gives two arms with readable prefixes, from one unchanged so101.xml', async () => {
  const { mj, model, data } = await compile(TWO_ARMS);
  const bodies = names(mj, model, mj.mjtObj.mjOBJ_BODY.value, model.nbody);
  expect(bodies.filter((n) => n.endsWith('gripper')).sort()).toEqual([
    'left_gripper',
    'right_gripper',
  ]);
  model.delete();
  data.delete();
});

test('each attached arm has its own actuators and its own joints', async () => {
  const two = await compile(TWO_ARMS);
  const one = await compile('<mujoco model="one"><include file="so101.xml"/></mujoco>');
  // Exactly twice the moving parts: nothing is shared, so one arm cannot drive the other.
  expect(two.model.nu).toBe(one.model.nu * 2);
  expect(two.model.njnt).toBe(one.model.njnt * 2);
  const A = two.mj.mjtObj.mjOBJ_ACTUATOR.value;
  const actuators = names(two.mj, two.model, A, two.model.nu);
  expect(actuators.filter((n) => n.startsWith('left_')).length).toBe(one.model.nu);
  expect(actuators.filter((n) => n.startsWith('right_')).length).toBe(one.model.nu);
  for (const got of [two, one]) {
    got.model.delete();
    got.data.delete();
  }
});

test('driving one arm leaves the other where it was', async () => {
  const { mj, model, data } = await compile(TWO_ARMS);
  const A = mj.mjtObj.mjOBJ_ACTUATOR.value;
  const pan = mj.mj_name2id(model, A, 'left_shoulder_pan');
  expect(pan).toBeGreaterThanOrEqual(0);
  const right = bodyId(mj, model, 'right_gripper');
  mj.mj_forward(model, data);
  const before = [...Array(3)].map((_, i) => data.xpos[right * 3 + i]);
  data.ctrl[pan] = 0.8;
  for (let i = 0; i < 1500; i++) mj.mj_step(model, data);
  const after = [...Array(3)].map((_, i) => data.xpos[right * 3 + i]);
  const moved = Math.hypot(...after.map((v, i) => v - before[i]));
  expect(moved).toBeLessThan(0.01);
  model.delete();
  data.delete();
});

test('the arms stand where their bodies put them', async () => {
  const { mj, model, data } = await compile(TWO_ARMS);
  mj.mj_forward(model, data);
  const left = bodyId(mj, model, 'left_gripper');
  const right = bodyId(mj, model, 'right_gripper');
  const gap = Math.abs(data.xpos[right * 3 + 1] - data.xpos[left * 3 + 1]);
  expect(gap).toBeGreaterThan(0.85);
  expect(gap).toBeLessThan(0.95);
  model.delete();
  data.delete();
});

test('each arm can have its own grip weld', async () => {
  // The page switches the grip with the global equality-disable flag, which is only safe
  // while the scene has one equality. With two arms it has two, so the page will need
  // per-equality control - worth knowing now rather than when the second arm is wired up.
  const xml = `<mujoco model="two arms, two grips">
  <asset><model name="so101" file="so101.xml"/></asset>
  <worldbody>
    <body name="left_stand" pos="0 -0.45 0"><attach model="so101" prefix="left_"/></body>
    <body name="right_stand" pos="0 0.45 0"><attach model="so101" prefix="right_"/></body>
    <body name="bottle" pos="0.2 -0.4 0.3"><freejoint/>
      <geom type="cylinder" size="0.016 0.035" mass="0.02"/></body>
    <body name="mug" pos="0.2 0.4 0.3"><freejoint/>
      <geom type="cylinder" size="0.04 0.045" mass="0.03"/></body>
  </worldbody>
  <equality>
    <weld name="left_grip" body1="left_gripper" body2="bottle"/>
    <weld name="right_grip" body1="right_gripper" body2="mug"/>
  </equality>
</mujoco>`;
  const { mj, model, data } = await compile(xml);
  expect(model.neq).toBe(2);
  const E = mj.mjtObj.mjOBJ_EQUALITY.value;
  expect(names(mj, model, E, model.neq).sort()).toEqual(['left_grip', 'right_grip']);
  model.delete();
  data.delete();
});

// ---------------------------------------------------------------- what does not work

test('replicate is not in this build, in any spelling', async () => {
  const tries = [
    '<replicate count="2" sep="_" offset="0 0.9 0">',
    '<replicate count="2">',
    '<replicate>',
    '<replicate n="2">',
    '<replicate copies="2">',
  ];
  for (const open of tries) {
    const why = await tryCompile(
      `<mujoco model="r">${open}<include file="so101.xml"/></replicate></mujoco>`,
    );
    expect(why, open).toMatch(/unrecognized (attribute|element)/);
  }
});

test('including the same arm twice fails, as it should', async () => {
  const why = await tryCompile(
    '<mujoco model="twice"><include file="so101.xml"/><include file="so101.xml"/></mujoco>',
  );
  expect(why).toMatch(/already included/);
});

// ---------------------------------------------------------------- the grip, on real shapes

const SHAPES: [string, string][] = [
  ['box', 'type="box" size="0.015 0.015 0.015"'],
  ['cylinder (a bottle)', 'type="cylinder" size="0.016 0.035"'],
  ['cylinder (a mug)', 'type="cylinder" size="0.04 0.045"'],
  ['capsule (a pen)', 'type="capsule" size="0.007 0.05"'],
];

test.each(SHAPES)('a weld holds a %s without creeping', async (_name, geom) => {
  const xml = `<mujoco model="grip">
  <include file="so101.xml"/>
  <worldbody>
    <body name="thing" pos="0.17 -0.08 0.3">
      <freejoint name="thing"/>
      <geom name="thing" ${geom} mass="0.02" friction="1.2 0.01 0.001" condim="4"
        solref="0.004 1" contype="2" conaffinity="2"/>
    </body>
  </worldbody>
  <equality><weld name="grip" body1="gripper" body2="thing"/></equality>
</mujoco>`;
  const { mj, model, data } = await compile(xml);
  const gripper = bodyId(mj, model, 'gripper');
  const thing = bodyId(mj, model, 'thing');
  const at = (b: number) => [data.xpos[b * 3], data.xpos[b * 3 + 1], data.xpos[b * 3 + 2]];
  mj.mj_forward(model, data);
  const before = Math.hypot(...at(thing).map((v, i) => v - at(gripper)[i]));
  for (let i = 0; i < 2000; i++) mj.mj_step(model, data);
  const after = Math.hypot(...at(thing).map((v, i) => v - at(gripper)[i]));
  expect(Number.isFinite(after)).toBe(true);
  expect(after).toBeLessThan(before + 0.005);
  model.delete();
  data.delete();
});

test('one weld can be retargeted from a cylinder to a capsule', async () => {
  const xml = `<mujoco model="retarget">
  <include file="so101.xml"/>
  <worldbody>
    <body name="bottle" pos="0.17 -0.08 0.3"><freejoint/>
      <geom type="cylinder" size="0.016 0.035" mass="0.02"/></body>
    <body name="pen" pos="0.17 0.08 0.3"><freejoint/>
      <geom type="capsule" size="0.007 0.05" mass="0.01"/></body>
  </worldbody>
  <equality><weld name="grip" body1="gripper" body2="bottle"/></equality>
</mujoco>`;
  const { mj, model, data } = await compile(xml);
  expect(model.neq).toBe(1);
  const pen = bodyId(mj, model, 'pen');
  model.eq_obj2id[0] = pen;
  mj.mj_forward(model, data);
  for (let i = 0; i < 200; i++) mj.mj_step(model, data);
  expect(Number.isFinite(data.xpos[pen * 3])).toBe(true);
  model.delete();
  data.delete();
});
