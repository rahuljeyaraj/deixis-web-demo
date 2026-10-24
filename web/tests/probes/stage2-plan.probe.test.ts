/**
 * Stage 2, the planning half: **does a move get planned at all, from a world at rest?**
 *
 * The body probe drives every move for real, which takes minutes and mixes two questions: can
 * the arm plan this, and does the plan survive contact with the physics? This asks only the
 * first, from a fresh untouched world each time, so a failure here is geometry and a failure
 * there is not.
 *
 * Run: `PROBES=1 npx vitest run tests/probes/stage2-plan.probe.test.ts`
 * Report: `out/stage2-plan.txt`
 */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { expect, test } from 'vitest';

import { THINGS } from '../../src/sim/bench';
import { loadScene, mujoco, unpack } from '../../src/sim/mujoco';
import type { PackFile } from '../../src/sim/mujoco';
import { Rig } from '../../src/sim/rig';

const WEB = path.join(import.meta.dirname, '..', '..');
const PACK = path.join(WEB, 'public', 'so101.pack');
const SCENE = path.join(WEB, 'model', 'workbench_scene.xml');
const REPORT = path.join(WEB, '..', 'out', 'stage2-plan.txt');

function say(line: string) {
  fs.mkdirSync(path.dirname(REPORT), { recursive: true });
  fs.appendFileSync(REPORT, `${line}\n`, 'utf8');
}

function files(): PackFile[] {
  const packed = unpack(new Uint8Array(zlib.gunzipSync(fs.readFileSync(PACK))));
  return [
    ...packed.filter((f) => !f.path.endsWith('_scene.xml')),
    { path: 'workbench_scene.xml', bytes: new Uint8Array(fs.readFileSync(SCENE)) },
  ];
}

test('every thing can have a way in planned for it', async () => {
  fs.rmSync(REPORT, { force: true });
  const mj = await mujoco();
  const { model, data } = loadScene(mj, files(), 'workbench_scene.xml');

  say('stage 2, planning: can a way in be worked out for each thing, from a world at rest?');
  say('  the arm is put back in its resting pose before each one, so nothing carries over.');
  say('');

  const stuck: string[] = [];
  for (const t of THINGS) {
    if (!t.graspable) continue;
    // A world of its own each time: the keyframe, then the resting pose.
    mj.mj_resetDataKeyframe(model, data, 0);
    const rig = new Rig(mj, model, data);
    rig.restNow();
    // Planning happens inside `take` before it returns, so the queue says at once whether
    // there is a plan. The promise is **not** awaited: a move that was planned only finishes
    // when somebody steps the physics, and nobody here does.
    let refused: string | null = null;
    void rig.take(t.id).then((d) => {
      refused = d.why ?? null;
    });
    await Promise.resolve(); // let a refusal's promise land
    const why = rig.busy ? 'planned' : (refused ?? 'refused, no reason');
    say(`  ${t.id.padEnd(13)} ${t.gripHow.padEnd(6)} ${why}${rig.trouble ? ` | ${rig.trouble}` : ''}`);
    if (why !== 'planned') stuck.push(`${t.id}: ${why}`);
    rig.stop();
  }

  say('');
  say(`  ${stuck.length ? `could not plan: ${stuck.join(', ')}` : 'every thing can be planned for.'}`);
  expect(stuck, 'every graspable thing should have a plan').toEqual([]);
  model.delete();
  data.delete();
}, 600000);
