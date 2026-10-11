/**
 * The flow in the browser (plan W9) on the server's real replies for the plan's scenes
 * (`tests/fixtures/replies.json`, written by the server tests `tests/test_flow_replies.py`):
 * each reply goes into the flow, and the real arm in MuJoCo does the rest.
 */

import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, test } from 'vitest';
import type { CoreResponse, Thing } from '../src/core';
import {
  type EventData,
  Flow,
  type FlowEvent,
  type FlowView,
  type Plan,
  SAY,
  askLine,
  guessLine,
} from '../src/flow/flow';
import { type PlanLine, planLine } from '../src/ui/plan';
import { blockedBy, distanceToBlock, footprint, thingsFor } from '../src/flow/things';
import { Arm, BLOCK_IDS, type BlockId, type BlockState } from '../src/sim/arm';
import { nodeScene } from './scene';

interface Step {
  kind: 'command' | 'answer' | 'go';
  text?: string;
  reply?: { result: CoreResponse | null; say: string | null; error: string | null };
}
const FIXTURE = JSON.parse(
  fs.readFileSync(path.join(import.meta.dirname, 'fixtures', 'replies.json'), 'utf8'),
) as { things: Thing[]; here: [number, number]; scenes: Record<string, Step[]> };

class Recorder implements FlowView {
  said: string[] = [];
  glowing: BlockId[] = [];
  ghostsOf: BlockId[] = [];
  plans: (PlanLine | null)[] = [];
  events: { kind: FlowEvent; data?: EventData }[] = [];
  speak(text: string) {
    this.said.push(text);
  }
  glow(ids: BlockId[]) {
    this.glowing = ids;
  }
  ghosts(items: { block: BlockId }[]) {
    this.ghostsOf = items.map((g) => g.block);
  }
  plan(plan: Plan | null) {
    this.plans.push(plan ? planLine(plan) : null);
  }
  event(kind: FlowEvent, data?: EventData) {
    this.events.push({ kind, data });
  }
  /** The one ghost block shown, when exactly one is (as it was before plans). */
  get ghostOf(): BlockId | null {
    return this.ghostsOf.length === 1 ? this.ghostsOf[0] : null;
  }
}

describe('the flow on the server replies', async () => {
  const { mj, model, data } = await nodeScene();
  const arm = new Arm(mj, model, data);
  const step = (s: number) => {
    for (let i = 0, n = Math.round(s / model.opt.timestep); i < n; i++) {
      arm.update();
      mj.mj_step(model, data);
    }
  };
  const settle = async <T>(p: Promise<T>): Promise<T> => {
    for (let t = 0; arm.busy && t < 30; t += 0.1) step(0.1);
    step(0.3);
    return p;
  };
  const centre = (t: Thing): [number, number] => [t.box_mm[0] + t.box_mm[2] / 2, t.box_mm[1] + t.box_mm[3] / 2];

  /** The blocks where the server's scene had them, the arm at home; then the scene's steps. */
  async function play(name: string) {
    arm.goHomeNow();
    for (const t of FIXTURE.things) arm.putBlock(t.id as BlockId, ...centre(t), 0);
    step(0.5);
    const view = new Recorder();
    let clock = 100;
    const flow = new Flow(arm, view, { confirmFirst: true, now: () => clock++ });
    for (const s of FIXTURE.scenes[name]) {
      if (s.kind === 'go') await settle(flow.go());
      else {
        flow.thinking();
        await flow.handle(s.reply!);
      }
    }
    return { flow, view };
  }

  const at = (id: BlockId) => arm.block(id);
  const near = (b: BlockState, x: number, y: number) => Math.hypot(b.tableMm[0] - x, b.tableMm[1] - y);

  test('put that here: preview, "Move the red block here?", yes, "Done."', async () => {
    const { flow, view } = await play('put that here');
    expect(view.said).toEqual(['Move the red block here?', SAY.done]);
    expect(flow.state.kind).toBe('idle');
    expect(near(at('red'), ...FIXTURE.here)).toBeLessThan(3);
    expect(at('red').heightMm).toBeCloseTo(30, 0);
    expect(flow.lastThing).toBe('red');
  });

  test('put the blue one next to that: to the right of the green block', async () => {
    await play('put the blue one next to that');
    const blue = at('blue');
    const green = at('green');
    expect(blue.tableMm[0] - green.tableMm[0]).toBeGreaterThan(30 + 5);
    expect(Math.abs(blue.tableMm[1] - green.tableMm[1])).toBeLessThan(3);
  });

  test('put that on that: stacked on the blue block, flat', async () => {
    const { flow } = await play('put that on that');
    const red = at('red');
    expect(near(red, ...at('blue').tableMm)).toBeLessThan(4);
    expect(red.heightMm).toBeCloseTo(60, 0);
    // The blue block now has one on top: it can't be picked.
    const things = flow.things();
    expect(things.find((t) => t.id === 'blue')!.selectable).toBe(false);
    expect(things.find((t) => t.id === 'red')!.height_mm).toBeCloseTo(60, 0);
  });

  test('a bit left: the ghost moves 20 mm to the left, then yes', async () => {
    const { view } = await play('a bit left');
    expect(view.said).toEqual([guessLine('red', null), guessLine('red', null), SAY.done]);
    expect(near(at('red'), FIXTURE.here[0] - 20, FIXTURE.here[1])).toBeLessThan(3);
  });

  test('outside the area: the question, and nothing moves', async () => {
    const { flow, view } = await play('outside the area');
    const before = centre(FIXTURE.things.find((t) => t.id === 'red')!);
    expect(view.said).toEqual(["I can't reach there. Please move the dot closer."]);
    expect(flow.state.kind).toBe('asking');
    expect(flow.request().state.question).toBe(view.said[0]);
    expect(near(at('red'), ...before)).toBeLessThan(0.5);
  });

  test('between two blocks: "The red one or the green one?", both outlined; a click on green', async () => {
    const steps = FIXTURE.scenes['between two blocks, then a click'];
    arm.goHomeNow();
    for (const t of FIXTURE.things) arm.putBlock(t.id as BlockId, ...centre(t), 0);
    step(0.5);
    const view = new Recorder();
    const flow = new Flow(arm, view, { confirmFirst: true, now: () => 1 });
    flow.thinking();
    await flow.handle(steps[0].reply!);
    expect(view.said).toEqual(['The red one or the green one?']);
    expect(flow.request().state.question).toBe('The red one or the green one?');
    expect([...view.glowing].sort()).toEqual(['green', 'red']);
    expect(flow.candidateOf('green')).not.toBeNull();
    expect(flow.candidateOf('blue')).toBeNull();
    await flow.handle(steps[1].reply!);
    expect(flow.state.kind).toBe('preview');
    expect(view.glowing).toEqual(['green']);
    await settle(flow.go());
    expect(near(at('green'), ...FIXTURE.here)).toBeLessThan(3);
  });

  test('no pointing: "Where? Point at the spot."', async () => {
    const { view, flow } = await play('no pointing');
    expect(view.said).toEqual(['Where? Point at the spot.']);
    expect(flow.state.kind).toBe('asking');
  });

  test('not understood: says so and keeps what was there', async () => {
    const { view, flow } = await play('not understood');
    expect(view.said[0]).toMatch(/^Sorry, I didn't get that/);
    expect(flow.state.kind).toBe('idle');
  });

  test('stop during a move freezes the arm; undo puts a block back', async () => {
    const { flow } = await play('put that here');
    // Undo: back to where the server's scene had it.
    await settle(flow.undo());
    const home = centre(FIXTURE.things.find((t) => t.id === 'red')!);
    expect(near(at('red'), ...home)).toBeLessThan(3);
    // Again, and stop half way.
    const view = new Recorder();
    const f2 = new Flow(arm, view, { confirmFirst: true, now: () => 1 });
    await f2.handle(FIXTURE.scenes['put that here'][0].reply!);
    const going = f2.go();
    step(1.2);
    await f2.handle(FIXTURE.scenes['stop'][0].reply!);
    expect(arm.busy).toBe(false);
    expect((await going)?.reason).toBe('stopped');
    expect(view.said.at(-1)).toBe(SAY.stopped);
  });
});

describe('robot-side checks before a preview', async () => {
  const { mj, model, data } = await nodeScene();
  const arm = new Arm(mj, model, data);
  const reply = (args: Record<string, unknown>): { result: CoreResponse } => ({
    result: { status: 'ok', calls: [{ index: 0, name: 'move', args }] },
  });
  const thing = (id: string) => ({ frame: 'table', kind: 'thing', thing: id, source: 'gesture', mean_mm: [0, 0] });
  const spot = (x: number, y: number, z = 0, target?: string) => ({
    frame: 'table', kind: 'box', source: 'gesture', mean_mm: [x, y], box_mm: [x - 15, y - 15, 30, 30], z_mm: z,
    ...(target ? { evidence: { target } } : {}),
  });

  async function fresh() {
    arm.goHomeNow();
    arm.putBlock('red', 180, 100, 0);
    arm.putBlock('green', 225, 100, 0);
    arm.putBlock('blue', 300, 90, 0);
    for (let i = 0; i < 100; i++) mj.mj_step(model, data);
    const view = new Recorder();
    return { view, flow: new Flow(arm, view, { confirmFirst: true, now: () => 1 }) };
  }

  test('out of reach (inside the area, off the mat)', async () => {
    const { view, flow } = await fresh();
    await flow.handle(reply({ object: thing('red'), place: spot(30, 30) }));
    expect(view.said).toEqual([SAY.cantReach]);
    expect(flow.state.kind).toBe('idle');
  });

  test('a block in the way, named', async () => {
    const { view, flow } = await fresh();
    await flow.handle(reply({ object: thing('red'), place: spot(305, 95) }));
    expect(view.said).toEqual(['The blue block is in the way there.']);
  });

  test('next to a block is fine; a turned block counts with its real corners', () => {
    const square = { tableMm: [100, 100] as [number, number], yaw: 0 };
    expect(distanceToBlock(100, 100, square)).toBe(0);
    expect(distanceToBlock(135, 100, square)).toBeCloseTo(20, 6);
    const turned = { tableMm: [100, 100] as [number, number], yaw: Math.PI / 4 };
    expect(distanceToBlock(135, 100, turned)).toBeCloseTo(35 - 15 * Math.SQRT2, 6); // its corner
    // 32 mm centre to centre, side by side: 2 mm apart, not in the way.
    const blocks = BLOCK_IDS.map((id) => arm.block(id));
    const blue = arm.block('blue').tableMm;
    expect(blockedBy(blue[0] + 32, blue[1], blocks, [])).toBeNull();
    expect(blockedBy(blue[0] + 25, blue[1], blocks, [])?.id).toBe('blue');
  });

  test('three high: done right where the arm reaches, else refused before moving', async () => {
    const settle = async <T>(p: Promise<T>): Promise<T> => {
      for (let t = 0; arm.busy && t < 30; t += 0.1) {
        for (let i = 0; i < 20; i++) {
          arm.update();
          mj.mj_step(model, data);
        }
      }
      for (let i = 0; i < 60; i++) mj.mj_step(model, data);
      return p;
    };
    let stacked = 0;
    let refused = 0;
    for (const [gx, gy] of [[300, 90], [250, 30], [330, 60], [360, 110]] as [number, number][]) {
      const { view, flow } = await fresh();
      arm.putBlock('green', gx, gy, 0);
      arm.putBlock('red', 180, 120, 0);
      arm.putBlock('blue', 360, 160, 0);
      for (let i = 0; i < 100; i++) mj.mj_step(model, data);
      await flow.handle(reply({ object: thing('red'), place: spot(gx, gy, 30, 'green') }));
      if (flow.state.kind !== 'preview') continue;
      const first = await settle(flow.go());
      expect(first?.ok, `red onto green at ${gx}, ${gy}: ${first?.reason}`).toBe(true);
      const red = arm.block('red');
      expect(red.heightMm).toBeCloseTo(60, 0);
      await flow.handle(reply({ object: thing('blue'), place: spot(red.tableMm[0], red.tableMm[1], 60, 'red') }));
      if (flow.state.kind !== 'preview') {
        expect(view.said.at(-1)).toBe(SAY.tooHigh);
        refused++;
        continue;
      }
      const second = await settle(flow.go());
      expect(second?.ok, `blue onto red at ${gx}, ${gy}: ${second?.reason}`).toBe(true);
      // Never knocked off: on top of the stack, 90 mm up.
      expect(arm.block('blue').heightMm).toBeCloseTo(90, 0);
      expect(Math.hypot(arm.block('blue').tableMm[0] - gx, arm.block('blue').tableMm[1] - gy)).toBeLessThan(6);
      expect(flow.things().filter((t) => t.selectable).map((t) => t.id)).toEqual(['blue']);
      stacked++;
    }
    console.log(`three high: ${stacked} stacked, ${refused} refused`);
    // Since the solver lets the hand tilt as it lifts away (6 Oct), all four places stack,
    // including the pick right next to a stack, which used to knock it over.
    expect(stacked).toBe(4);
  });

  test('on itself', async () => {
    const { view, flow } = await fresh();
    await flow.handle(reply({ object: thing('red'), place: spot(180, 100, 30, 'red') }));
    expect(view.said).toEqual([SAY.onItself]);
  });

  test('a refused correction keeps the guess on screen (item 8)', async () => {
    const { view, flow } = await fresh();
    await flow.handle(reply({ object: thing('red'), place: spot(260, 110) }));
    expect(flow.state.kind).toBe('preview');
    const adjust = (where: unknown): { result: CoreResponse } => ({
      result: { status: 'ok', calls: [{ index: 0, name: 'adjust', args: { place: where } }] },
    });
    flow.thinking();
    await flow.handle(adjust(spot(300, 90))); // onto the blue block's spot
    expect(view.said.at(-1)).toBe('The blue block is in the way there.');
    expect(flow.state.kind).toBe('preview');
    expect(view.ghostOf).toBe('red');
    expect(view.glowing).toEqual(['red']);
  });

  test('what the robot understood, said back; the choices named', () => {
    expect(guessLine('red', null)).toBe('Move the red block here?');
    expect(guessLine('red', 'blue')).toBe('Put the red block on the blue one?');
    const cand = (thing: string) => ({ frame: 'table', kind: 'thing', thing, source: 'gesture', mean_mm: [0, 0] });
    const ask = (things: string[], reason = 'ambiguous_object') =>
      ({ reason, question: 'This one or that one?', candidates: things.map(cand) }) as never;
    expect(askLine(ask(['red', 'green', 'blue']))).toBe('The red one, the green one or the blue one?');
    expect(askLine(ask(['red'], 'no_object'))).toBe('This one or that one?');
  });

  test('the footprint of a turned block, and a free place', () => {
    expect(footprint({ tableMm: [100, 100], yaw: 0 })).toEqual([85, 85, 30, 30]);
    const turned = footprint({ tableMm: [100, 100], yaw: Math.PI / 4 });
    expect(turned[2]).toBeCloseTo(42.4, 1);
    const blocks = BLOCK_IDS.map((id) => arm.block(id));
    expect(blockedBy(250, 50, blocks, [])).toBeNull();
    expect(thingsFor(blocks, () => 0).every((t) => t.selectable)).toBe(true);
  });
});
