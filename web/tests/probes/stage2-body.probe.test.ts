/**
 * Stage 2's checklist, measured: **can it really pick each thing up, carry it and put it down?**
 *
 * Nothing here is mimed. The real scene is stepped at its own timestep, the real solver plans
 * every pose, the jaws really close, the weld really takes the thing, and where it ends up is
 * read back out of the simulation. A thing that ends more than a centimetre from where it was
 * aimed has not been put down, whatever the move said.
 *
 * What it settles, each an item of stage 2:
 *
 * - every graspable thing can be picked up, from the bench and from a bay;
 * - a thing put back comes to rest where it was put, and **stays** there;
 * - a thing can be dropped into the pot (FR16);
 * - the refusals are refusals: the pot is never picked up, a bottle too tall for a bay is
 *   refused before anything moves, and a place off the end of the rail is refused;
 * - stop is instant (FR27): nothing moves after it.
 *
 * Run: `PROBES=1 npx vitest run tests/probes/stage2-body.probe.test.ts`
 * Report: `out/stage2-body.txt`
 */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { expect, test } from 'vitest';

import { THINGS } from '../../src/sim/bench';
import { loadScene, mujoco, unpack } from '../../src/sim/mujoco';
import type { MjData, MjModel, Mujoco, PackFile } from '../../src/sim/mujoco';
import { Rig, type Done, type Place } from '../../src/sim/rig';

const WEB = path.join(import.meta.dirname, '..', '..');
const PACK = path.join(WEB, 'public', 'so101.pack');
const SCENE = path.join(WEB, 'model', 'workbench_scene.xml');
const REPORT = path.join(WEB, '..', 'out', 'stage2-body.txt');

/** How long a move may take before it is called stuck, seconds of simulated time. */
const PATIENCE_S = 30;
/** How far a thing may end from where it was aimed and still count as put down, mm. */
const CLOSE_MM = 12;

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

interface World {
  mj: Mujoco;
  model: MjModel;
  data: MjData;
  rig: Rig;
}

async function build(): Promise<World> {
  const mj = await mujoco();
  const { model, data } = loadScene(mj, files(), 'workbench_scene.xml');
  mj.mj_resetDataKeyframe(model, data, 0);
  const rig = new Rig(mj, model, data);
  rig.restNow();
  return { mj, model, data, rig };
}

/**
 * Step until the move finishes, or until patience runs out.
 *
 * The move's promise settles on a microtask, and a microtask cannot run inside a loop that
 * never yields - so the loop yields as soon as the arm is idle, which is exactly when the
 * promise has something to say.
 */
async function run(w: World, move: Promise<Done>): Promise<Done> {
  const landed: { done?: Done } = {};
  void move.then((d) => {
    landed.done = d;
  });
  const until = w.data.time + PATIENCE_S;
  while (!landed.done && w.data.time < until) {
    w.rig.update();
    w.mj.mj_step(w.model, w.data);
    if (!w.rig.busy) await Promise.resolve();
  }
  return landed.done ?? { ok: false, why: 'busy' };
}

/** Let the world settle, so that "it stays there" means something. */
function settle(w: World, seconds = 1.5): void {
  const n = Math.round(seconds / w.model.opt.timestep);
  for (let i = 0; i < n; i++) {
    w.rig.update();
    w.mj.mj_step(w.model, w.data);
  }
}

/**
 * Stand the **whole world** back where the scene was generated with it, and still it.
 *
 * Every move used to be measured in whatever world the move before it left behind. A failed
 * put-down ends in `park`, which opens the jaws wherever the hand happens to be, so a thing
 * would be left in the middle of the bench - and every later grip was then refused for running
 * into it, correctly, in a scene that was wrecked. The tail of this report meant nothing.
 *
 * One move per row, each from the world as laid out. The free joint's seven numbers are where a
 * thing is and which way round; the six after them are how fast it is going, and a thing put
 * back while still moving carries on from where it was dropped.
 */
function putBack(w: World): void {
  const J = w.mj.mjtObj.mjOBJ_JOINT.value;
  const laid = w.model.key_qpos as Float64Array;
  for (const t of THINGS) {
    const joint = w.mj.mj_name2id(w.model, J, t.id);
    if (joint < 0) continue;
    const adr = (w.model.jnt_qposadr as Int32Array)[joint];
    const dof = (w.model.jnt_dofadr as Int32Array)[joint];
    for (let k = 0; k < 7; k++) w.data.qpos[adr + k] = laid[adr + k];
    for (let k = 0; k < 6; k++) w.data.qvel[dof + k] = 0;
  }
  w.mj.mj_forward(w.model, w.data);
}

function offMm(w: World, id: string, place: Place): number {
  const at = w.rig.state(id).atMm;
  return Math.hypot(at[0] - place.xMm, at[1] - place.yMm);
}

test('the body: picking up, carrying, putting down, and refusing', async () => {
  fs.rmSync(REPORT, { force: true });
  say("stage 2's checklist, measured: what can the arm really do?");
  say('  the real scene, stepped at its own timestep; the real solver; the real jaws.');
  say('');

  const w = await build();
  // Let the world come to rest before anything is measured. A scene that has just been reset
  // has everything exactly where the generator put it, which is not quite where the physics
  // will hold it, and the first grip of the run was being compared against the former.
  settle(w, 1);
  // ONLY=jar1,tin2 runs a few of them, for when something is being chased down.
  const only = (process.env.ONLY ?? '').split(',').filter(Boolean);
  const graspable = THINGS.filter((t) => t.graspable && (!only.length || only.includes(t.id)));

  say('picking each thing up and putting it back where it was:');
  say(`  ${'thing'.padEnd(13)} ${'way in'.padEnd(6)} took  put   ended  why`);
  const failed: string[] = [];
  for (const t of graspable) {
    const was = w.rig.state(t.id).atMm;
    const took = await run(w, w.rig.take(t.id));
    let put: Done = { ok: false, why: 'nothing is being held' };
    let off = Number.NaN;
    if (took.ok) {
      const back: Place = { xMm: was[0], yMm: was[1], zMm: t.rises[0] };
      put = await run(w, w.rig.putOn(back));
      settle(w);
      off = offMm(w, t.id, back);
    }
    await run(w, w.rig.park());
    // One move per row: the world goes back as it was laid out before the next thing is tried.
    putBack(w);
    settle(w, 0.5);
    const line =
      `  ${t.id.padEnd(13)} ${t.gripHow.padEnd(6)} ${(took.ok ? 'yes' : 'no ').padEnd(5)} ` +
      `${(put.ok ? 'yes' : 'no ').padEnd(5)} ${Number.isNaN(off) ? '  -  ' : `${off.toFixed(0).padStart(3)}mm`} ` +
      `${took.why ?? put.why ?? (took.assisted ? '(the weld took it before both jaws did)' : '')}` +
      `${w.rig.trouble ? ` | ${w.rig.trouble}` : ''}`;
    say(line);
    if (!took.ok || !put.ok || off > CLOSE_MM) failed.push(t.id);
  }

  say('');
  say('dropping a thing into the pot (FR16):');
  const pot = THINGS.find((t) => t.id === 'pot')!;
  const tookForPot = await run(w, w.rig.take('blades'));
  const dropped = tookForPot.ok ? await run(w, w.rig.dropInto('pot')) : tookForPot;
  settle(w, 2);
  const inPot = w.rig.state('blades').atMm;
  const intoPot =
    dropped.ok && Math.hypot(inPot[0] - pot.at[0], inPot[1] - pot.at[1]) < 60 && inPot[2] > 10;
  say(
    `  the tin of blades: ${dropped.ok ? 'let go over the rim' : dropped.why}, and it came to` +
      ` rest ${Math.hypot(inPot[0] - pot.at[0], inPot[1] - pot.at[1]).toFixed(0)} mm from the` +
      ` pot's middle, ${inPot[2].toFixed(0)} mm up - ${intoPot ? 'in it' : 'not in it'}.`,
  );
  await run(w, w.rig.park());

  say('');
  say('the refusals, and whether anything moved (FR15):');
  const refusals: [string, () => Promise<Done>, string][] = [
    ['the pot of screws', () => w.rig.take('pot'), 'that is never picked up'],
    [
      'the bottle of oil into the middle bay',
      async () => {
        const took = await run(w, w.rig.take('oil'));
        if (!took.ok) return took;
        const no = await run(w, w.rig.putOn({ xMm: 300, yMm: 385, zMm: 122 }));
        await run(w, w.rig.park());
        return no;
      },
      'it is too tall for that bay',
    ],
    [
      'a place out past the end of the bench',
      async () => {
        const took = await run(w, w.rig.take('knife'));
        if (!took.ok) return took;
        const no = await run(w, w.rig.putOn({ xMm: 350, yMm: -400, zMm: 0 }));
        await run(w, w.rig.park());
        return no;
      },
      'there is no way to put it there',
    ],
  ];
  const wrong: string[] = [];
  for (const [what, ask, want] of refusals) {
    const got = await ask();
    say(`  ${what.padEnd(38)} ${got.ok ? 'DID IT' : got.why}`);
    if (got.ok || got.why !== want) wrong.push(what);
  }

  say('');
  say('stop, mid-move (FR27):');
  const before = { ...w.rig.state('hammer') };
  const moving = w.rig.take('hammer');
  for (let i = 0; i < 400; i++) {
    w.rig.update();
    w.mj.mj_step(w.model, w.data);
  }
  const atStop = w.rig.armQ();
  w.rig.stop();
  const stopped = await moving;
  settle(w, 0.5);
  const after = w.rig.armQ();
  const drift = Math.max(...after.map((v, j) => Math.abs(v - atStop[j])));
  say(
    `  stopped in the middle of reaching for the hammer: ${stopped.why}; the joints moved` +
      ` ${((drift * 180) / Math.PI).toFixed(1)} degrees more after it, and the hammer` +
      ` ${Math.hypot(...w.rig.state('hammer').atMm.map((v, k) => v - before.atMm[k])).toFixed(0)} mm.`,
  );

  say('');
  say(`  ${graspable.length - failed.length} of ${graspable.length} things picked up, carried and put back.`);
  if (failed.length) say(`  not: ${failed.join(', ')}`);

  // ------------------------------------------------------------- the gates
  expect(failed, 'every graspable thing should survive a pick up and a put down').toEqual([]);
  expect(intoPot, 'the tin of blades should end up in the pot').toBe(true);
  expect(wrong, 'each refusal should be the refusal it is meant to be').toEqual([]);
  expect(stopped.why, 'a stopped move says so').toBe('stopped');
  expect((drift * 180) / Math.PI, 'stop is instant').toBeLessThan(3);

  w.model.delete();
  w.data.delete();
}, 1800000);
