/**
 * Plans of several steps in the browser (advanced-commands-plan.md), with the real arm in
 * MuJoCo: the swap that is impossible in two moves, a tower, each block on its own colour, a
 * plan that ends at home, stopping in the middle, and the re-plan after a step the arm says
 * it cannot do.
 *
 * The command itself is the server's real reply (`tests/fixtures/replies.json`, written by
 * `tests/test_flow_replies.py`): the whole plan as Nemotron wrote it, and the core's answer
 * for its first step. The later steps are measured when the arm reaches them, against a table
 * only the browser knows, so `Stub` stands in for `/api/step` here - it answers from the
 * `things` the page sends it, finding things by label the way the core does. That is the
 * point being tested: the page hands over the table as it is at that moment, and acts on what
 * comes back, not on what it guessed while showing the plan.
 */

import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, test } from 'vitest';
import type { CoreResponse, ResolvedCall, Thing, Where } from '../src/core';
import {
  type EventData,
  Flow,
  type FlowEvent,
  type FlowReply,
  type FlowServer,
  type FlowView,
  type Ghost,
  type Plan,
  type RawCall,
  SAY,
  type StepReply,
} from '../src/flow/flow';
import { Arm, type BlockId, type BlockState } from '../src/sim/arm';
import { planLine } from '../src/ui/plan';
import { nodeScene } from './scene';

interface Scene {
  kind: 'command' | 'go';
  text?: string;
  reply?: FlowReply;
}
const FIXTURE = JSON.parse(
  fs.readFileSync(path.join(import.meta.dirname, 'fixtures', 'replies.json'), 'utf8'),
) as { things: Thing[]; paint: Thing[]; here: [number, number]; scenes: Record<string, Scene[]> };

const centre = (t: Thing): [number, number] => [t.box_mm[0] + t.box_mm[2] / 2, t.box_mm[1] + t.box_mm[3] / 2];

class Recorder implements FlowView {
  said: string[] = [];
  glowing: BlockId[] = [];
  shown: Ghost[] = [];
  strips: (string | null)[] = [];
  events: { kind: FlowEvent; data?: EventData }[] = [];
  traced: string[] = [];
  speak(text: string) {
    this.said.push(text);
  }
  glow(ids: BlockId[]) {
    this.glowing = ids;
  }
  ghosts(items: Ghost[]) {
    if (items.length) this.shown = items;
  }
  plan(plan: Plan | null) {
    const line = plan && plan.steps.length > 1 ? planLine(plan) : null;
    this.strips.push(line && `${line.title} ${line.steps.map((s) => `${s.text} ${s.mark}`).join(' | ')}`);
  }
  trace(rows: { say: string }[]) {
    this.traced.push(...rows.map((r) => r.say));
  }
  event(kind: FlowEvent, data?: EventData) {
    this.events.push({ kind, data });
  }
  /** The last plan strip that was shown (the steps and their marks). */
  get strip(): string | null {
    return [...this.strips].reverse().find((s) => s !== null) ?? null;
  }
}

// ---------------------------------------------------------------- the core, stood in for

/** Find a thing by what was said, the way `deixis.scene.find` does: exactly first. */
function byLabel(things: Thing[], said: string): Thing {
  const norm = (s: string) => s.trim().toLowerCase().replace(/^(the|a|an)\s+/, '');
  const words = (s: string) => new Set(norm(s).split(/[\s_-]+/).filter(Boolean).map((w) => w.replace(/s$/, '')));
  const want = norm(said);
  const exact = things.filter((t) => norm(t.label) === want || Object.values(t.attrs).some((v) => norm(v) === want));
  const hits = exact.length
    ? exact
    : things.filter((t) => {
        const have = words([t.label, ...Object.values(t.attrs)].join(' '));
        return [...words(said)].filter((w) => !['one', 'thing', 'block'].includes(w)).every((w) => have.has(w));
      });
  if (hits.length !== 1) throw new Error(`“${said}” found ${hits.length} things, not 1`);
  return hits[0];
}

function where(mean: [number, number], zMm: number, evidence: Record<string, unknown>): Where {
  return { frame: 'table', kind: 'box', mean_mm: mean, z_mm: zMm || null, source: 'geometry', evidence };
}

/**
 * One raw `move` call against the table it is given: exactly the two answers the core gives
 * for the relations these plans use (`on`, `inside`, `next_to`).
 */
function measure(call: RawCall, things: Thing[], index: number): ResolvedCall {
  const object = byLabel(things, String((call.args.object as { label: string }).label));
  const place = call.args.place as { relation: string; target: { label: string } };
  const target = byLabel(things, place.target.label);
  const ev = { relation: place.relation, target: target.id };
  const [cx, cy] = centre(target);
  let mean: [number, number] = [cx, cy];
  let zMm = 0;
  if (place.relation === 'on') {
    zMm = target.height_mm; // stacked on top
  } else if (place.relation === 'next_to') {
    // The core's own order: to the right, to the left, in front, behind; the first that is
    // free. The paint is flat, so it is never in the way.
    const d = 15 + 10 + 15; // half a block, the profile's gap, half a block
    const sides: [number, number][] = [[cx + d, cy], [cx - d, cy], [cx, cy - d], [cx, cy + d]]; // prettier-ignore
    const clear = ([x, y]: [number, number]) =>
      things.every((t) => {
        if (t.id === object.id || !t.height_mm) return true;
        const [tx, ty] = centre(t);
        return Math.max(Math.abs(tx - x), Math.abs(ty - y)) >= 30;
      });
    mean = sides.find(clear) ?? sides[0];
  }
  return {
    index,
    name: 'move',
    args: {
      object: { ...where(centre(object), 0, { rule: 'label' }), thing: object.id, label: object.label },
      place: where(mean, zMm, ev),
    },
  };
}

function okReply(call: ResolvedCall): StepReply {
  return { result: { status: 'ok', calls: [call] } as CoreResponse, say: null, error: null };
}

/** The server for the steps the page asks about as it goes. */
class Stub implements FlowServer {
  /** Every scene it was given, in order: proof that the table it was sent had moved on. */
  asked: { index: number; things: Thing[] }[] = [];
  /** The plan it answers a re-plan with, if any. */
  instead: RawCall[] | null = null;
  replans = 0;

  constructor(private plan: RawCall[]) {}

  async step(index: number, things: Thing[]): Promise<StepReply> {
    this.asked.push({ index, things });
    return okReply(measure(this.plan[index], things, index));
  }

  async replan(_index: number, _why: string, things: Thing[]): Promise<FlowReply> {
    this.replans++;
    if (!this.instead) return { result: null, say: 'I had no better idea.', error: null };
    this.plan = this.instead;
    return { ...okReply(measure(this.plan[0], things, 0)), plan: { calls: this.plan, steps: this.plan.length } };
  }
}

describe('a plan of several steps, in the browser', async () => {
  const { mj, model, data } = await nodeScene();
  const arm = new Arm(mj, model, data);
  const tick = (s: number) => {
    for (let i = 0, n = Math.round(s / model.opt.timestep); i < n; i++) {
      arm.update();
      mj.mj_step(model, data);
    }
  };
  /** Step the physics until the promise settles: a plan is several arm moves, one after another. */
  const pump = async <T>(p: Promise<T>): Promise<T> => {
    let done = false;
    const out = p.then(
      (v) => {
        done = true;
        return v;
      },
      (e) => {
        done = true;
        throw e;
      },
    );
    for (let i = 0; !done && i < 900; i++) {
      tick(0.1);
      await new Promise((r) => setTimeout(r, 0));
    }
    tick(0.2);
    return out;
  };

  /** The blocks where the server's scene had them, the arm at home, then the command. */
  async function start(name: string) {
    arm.goHomeNow();
    for (const t of FIXTURE.things) arm.putBlock(t.id as BlockId, ...centre(t), 0);
    tick(0.5);
    const scene = FIXTURE.scenes[name];
    const reply = scene[0].reply!;
    const view = new Recorder();
    const stub = new Stub(reply.plan!.calls);
    let clock = 100;
    const flow = new Flow(arm, view, { confirmFirst: true, now: () => clock++, server: stub });
    flow.thinking();
    await pump(flow.handle(reply));
    return { flow, view, stub };
  }

  const at = (id: BlockId) => arm.block(id);
  const near = (b: BlockState, x: number, y: number) => Math.hypot(b.tableMm[0] - x, b.tableMm[1] - y);
  const was = (id: string) => centre(FIXTURE.things.find((t) => t.id === id)!);

  test('swap the red and the blue: three moves, and they really change places', async () => {
    // Command 10. Two blocks, one gripper: you must park one somewhere free first. No word
    // matcher can invent that third move, and no single `move` call can say it.
    const { flow, view, stub } = await start('swap the red and the blue');
    expect(view.said).toEqual(['Three steps, like this?']);
    expect(flow.state.kind).toBe('preview');
    // Two ghost blocks, each at the place that block ends up in.
    expect(view.shown.map((g) => g.block).sort()).toEqual(['blue', 'red']);
    expect(view.strip).toBe(
      'Nemotron’s plan: 1 red beside green ○ | 2 blue where the other one was ○ | 3 red where the other one was ○',
    );

    await pump(flow.go());
    expect(near(at('red'), ...was('blue'))).toBeLessThan(6);
    expect(near(at('blue'), ...was('red'))).toBeLessThan(6);
    expect(at('red').heightMm).toBeCloseTo(30, 0);
    expect(view.said.at(-1)).toBe(SAY.done);
    // Steps 2 and 3 were each measured again, against the table as it was by then.
    expect(stub.asked.map((a) => a.index)).toEqual([1, 2, 1, 2]); // shown, then run
    const spots = stub.asked.map((a) => a.things.find((t) => t.id === 'left-spot')!);
    expect(centre(spots[2])).toEqual(expect.arrayContaining([expect.any(Number)]));
    // When step 2 ran, the spot the red one left was the real one the arm had just made.
    expect(Math.hypot(...sub(centre(spots[2]), was('red')))).toBeLessThan(6);
    // And when step 3 ran, it was the one the blue one had just left.
    expect(Math.hypot(...sub(centre(spots[3]), was('blue')))).toBeLessThan(6);
  });

  test('the panel gets a row for each step, and two for each move of the arm', async () => {
    // The arm's lane was empty on every command until 11 Oct: the last link in the chain,
    // the one that actually moves the block, said nothing.
    const { flow, view } = await start('swap the red and the blue');
    view.traced.length = 0;
    await pump(flow.go());
    expect(view.traced).toEqual([
      'step 1 of 3',
      'pick up the red block, put it at (265, 100) mm',
      'done, the block is there',
      'step 2 of 3',
      'pick up the blue block, put it at (180, 100) mm',
      'done, the block is there',
      'step 3 of 3',
      'pick up the red block, put it at (300, 90) mm',
      'done, the block is there',
    ]);
  });

  test('stack them all up: a tower of three, built in the right order', async () => {
    const { flow, view } = await start('stack them all up');
    expect(view.said).toEqual(['Two steps, like this?']);
    await pump(flow.go());
    expect(at('green').heightMm).toBeCloseTo(60, 0);
    expect(at('blue').heightMm).toBeCloseTo(90, 0);
    expect(near(at('blue'), ...at('red').tableMm)).toBeLessThan(8);
    // The two under the top one cannot be picked up any more.
    const things = flow.things();
    expect(things.find((t) => t.id === 'red')!.selectable).toBe(false);
    expect(things.find((t) => t.id === 'green')!.selectable).toBe(false);
    expect(things.find((t) => t.id === 'blue')!.selectable).toBe(true);
  });

  test('put each block on its own colour: three blocks, three squares', async () => {
    // Command 15, and the first use of the core's `inside` relation in any demo.
    const { flow, view } = await start('put each block on its own colour');
    expect(view.said).toEqual(['Three steps, like this?']);
    expect(view.shown.map((g) => g.block).sort()).toEqual(['blue', 'green', 'red']);
    await pump(flow.go());
    for (const id of ['red', 'green', 'blue'] as BlockId[]) {
      const square = flow.things().find((t) => t.id === `${id}-square`)!;
      expect(near(at(id), ...centre(square))).toBeLessThan(6);
    }
    expect(view.strips.at(-1)).toBe(null); // the strip goes once the plan is done
    expect(view.strips.some((s) => s?.includes('1 red into the red square ✓'))).toBe(true);
  });

  test('a plan may end at home', async () => {
    const { flow, view } = await start('put the red one on the blue one then go home');
    expect(view.strip).toBe('Nemotron’s plan: 1 red onto blue ○ | 2 go home ○');
    await pump(flow.go());
    expect(at('red').heightMm).toBeCloseTo(60, 0);
    expect(arm.armQ()[1]).toBeCloseTo(-1.45, 1); // the home pose
    expect(view.said.at(-1)).toBe(SAY.done);
  });

  test('everything except the blue one: two steps, and the blue one stays put', async () => {
    // Command 18: a rule cannot do "except".
    const { flow, view } = await start('everything except the blue one goes in the red square');
    expect(view.said).toEqual(['Two steps, like this?']);
    const blue = at('blue').tableMm.slice() as [number, number];
    await pump(flow.go());
    // One block goes in the middle of the square; the second goes beside the first, since
    // the middle is taken. The blue one does not move: a rule cannot do "except".
    const square = flow.things().find((t) => t.id === 'red-square')!;
    expect(near(at('red'), ...centre(square))).toBeLessThan(6);
    expect(near(at('green'), ...at('red').tableMm)).toBeLessThan(55);
    expect(near(at('blue'), ...blue)).toBeLessThan(2);
  });

  test('stop between the steps, and the rest of the plan is dropped', async () => {
    const { flow, view, stub } = await start('put each block on its own colour');
    const running = flow.go();
    // Stop while the first step is in the air.
    tick(0.4);
    flow.stop();
    await pump(running);
    expect(flow.state.kind).toBe('idle');
    expect(view.events.some((e) => e.kind === 'stop')).toBe(true);
    // Nothing was asked for after the stop, and no new plan was asked for either.
    expect(stub.replans).toBe(0);
    expect(view.said).not.toContain(SAY.done);
  });

  test('a step the arm cannot do is handed back to Nemotron, once', async () => {
    const { flow, stub } = await start('stack them all up');
    // Blue onto green is impossible once something else is on green. The stand-in offers
    // another plan instead, which is what the real server asks Nemotron for.
    stub.instead = [{ name: 'move', args: { object: { label: 'blue one' }, place: { relation: 'inside', target: { label: 'blue square' } } } }];
    // Put the red one on green behind the plan's back, so step 2 finds green covered.
    await pump(flow.go());
    expect(stub.replans).toBe(0); // nothing wrong with this one

    const second = await start('stack them all up');
    second.stub.instead = stub.instead;
    // Cover the red block, so "green onto red" is refused when it is run.
    arm.putBlock('blue', ...at('red').tableMm, 0, 30);
    tick(0.5);
    await pump(second.flow.go());
    expect(second.stub.replans).toBe(1);
    expect(second.view.said).toContain(SAY.thinkAgain);
    expect(second.view.events.map((e) => e.kind)).toContain('replan');
    const square = second.flow.things().find((t) => t.id === 'blue-square')!;
    expect(near(at('blue'), ...centre(square))).toBeLessThan(8);
  });

  test('no better idea: the arm says the reason and stops there', async () => {
    const second = await start('stack them all up');
    arm.putBlock('blue', ...at('red').tableMm, 0, 30);
    tick(0.5);
    await pump(second.flow.go());
    expect(second.stub.replans).toBe(1);
    expect(second.view.said).toContain('I had no better idea.');
    expect(second.view.said).not.toContain(SAY.done);
  });
});

function sub(a: [number, number], b: [number, number]): [number, number] {
  return [a[0] - b[0], a[1] - b[1]];
}
