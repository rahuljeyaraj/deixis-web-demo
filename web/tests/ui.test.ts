/**
 * The page's panels (plan W10) on the server's real replies (`tests/fixtures/replies.json`):
 * the thinking strip's words, holds and links, the wiring panel's rows and lanes, the plan
 * strip, and the one summary line under the panel.
 */

import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, test } from 'vitest';
import type { HeardWord } from '../src/api';
import type { CoreResponse } from '../src/core';
import type { Plan, PlanStep, StepState } from '../src/flow/flow';
import { costLine, summaryLine, timingsLine, understoodBy } from '../src/ui/summary';
import { MARK, planLine } from '../src/ui/plan';
import { layoutWords, stripModel } from '../src/ui/strip';
import { LANES, lanePercent, laneOf, rowShape, timeLabel } from '../src/ui/wiring';

const FIXTURE = JSON.parse(fs.readFileSync(path.join(import.meta.dirname, 'fixtures', 'replies.json'), 'utf8')) as {
  scenes: Record<string, { kind: string; reply?: { result: CoreResponse | null } }[]>;
};
const result = (scene: string) => FIXTURE.scenes[scene][0].reply!.result;

/** Typed words as the server times them: "that" and "here" at the clicks (28.0, 29.2). */
const words = (...ws: [string, number][]): HeardWord[] => ws.map(([text, t0]) => ({ text, t0, t1: t0 + 0.2 }));

describe('the thinking strip', () => {
  test('put that here: two pointing words, each tied to its own hold', () => {
    const m = stripModel(words(['put', 27.75], ['that', 28.0], ['here', 29.2]), result('put that here'));
    expect(m.words.map((w) => w.pointing)).toEqual([false, true, true]);
    expect(m.holds).toHaveLength(2);
    expect(m.links).toEqual([
      { word: 1, hold: 0 },
      { word: 2, hold: 1 },
    ]);
    expect(m.t0).toBeLessThan(27.75);
    expect(m.t1).toBeGreaterThan(29.4);
  });

  test('the blue one next to that: one link, from "that"', () => {
    const m = stripModel(
      words(['put', 27.0], ['the', 27.2], ['blue', 27.4], ['one', 27.6], ['next', 27.8], ['to', 27.9], ['that', 28.0]),
      result('put the blue one next to that'),
    );
    expect(m.links).toEqual([{ word: 6, hold: 0 }]);
    expect(m.words.filter((w) => w.pointing).map((w) => w.text)).toEqual(['that']);
  });

  test('a question: "that" is tied to the hold between the two blocks', () => {
    const m = stripModel(words(['put', 27.75], ['that', 28.0], ['here', 29.2]), result('between two blocks, then a click'));
    expect(m.words.filter((w) => w.pointing).map((w) => w.text)).toEqual(['that', 'here']);
    expect(m.links).toEqual([{ word: 1, hold: 0 }]);
    expect(m.holds[0].t0).toBeCloseTo(27.85);
  });

  test('after a clicked answer, "here" is tied to its hold too', () => {
    const [ask, answer] = FIXTURE.scenes['between two blocks, then a click'];
    const m = stripModel(words(['put', 27.75], ['that', 28.0], ['here', 29.2]), ask.reply!.result, answer.reply!.result);
    expect(m.links.map((l) => m.words[l.word].text).sort()).toEqual(['here', 'that']);
    expect(new Set(m.links.map((l) => l.hold)).size).toBe(2);
  });

  test('not understood: the words only', () => {
    const m = stripModel(words(['hello', 30.0], ['robot', 30.25]), null);
    expect(m.holds).toEqual([]);
    expect(m.links).toEqual([]);
    expect(m.words.every((w) => !w.pointing)).toBe(true);
  });

  test('words never overlap, and stay inside the strip', () => {
    const m = stripModel(words(['put', 1.0], ['that', 1.01], ['right', 1.02], ['here', 1.03]), null);
    const x = (t: number) => 100 + (t - m.t0) * 100;
    const spots = layoutWords(m, x, () => 50, 260);
    for (let i = 1; i < spots.length; i++) expect(spots[i].x).toBeGreaterThanOrEqual(spots[i - 1].x + 50);
    expect(spots.at(-1)!.x + 50).toBeLessThanOrEqual(260);
  });
});

describe('the line under the panel', () => {
  const agent = {
    source: 'nemotron' as const, model: 'nvidia/Nemotron-3_5-Lightning', calls: [], raw: [],
    errors: [], attempts: 1,
  }; // prettier-ignore
  const t = { asr_s: 0.41, agent_s: 1.2, core_s: 0.012, total_s: 1.7 };

  test('who understood, and how long it took', () => {
    expect(understoodBy({ ...agent, source: 'nemotron' })).toContain('Nemotron 3.5 Lightning');
    expect(understoodBy({ ...agent, source: 'nemotron' }, 'own')).toContain('with your own key');
    // The only thing that is not the model: the person clicking one of the robot's choices.
    expect(understoodBy({ ...agent, source: 'answer' })).toContain('Your answer');
    expect(understoodBy(undefined)).toContain('answer');
    expect(timingsLine(t, { ...agent, source: 'nemotron' })).toBe(
      'Parakeet 0.41 s · Nemotron 1.20 s · core 0.012 s · server in all 1.70 s',
    );
    // A clicked answer took no model at all, so the line names the answer, not a model.
    expect(timingsLine(t, { ...agent, source: 'answer' })).toContain('answer 1.20 s');
  });

  test('tokens and cost at list price (item 11)', () => {
    const paid = { ...agent, usage: { prompt: 7051, cached: 4192, completion: 64 } };
    expect(costLine(paid, 0.00043842)).toBe('7,051 tokens in (4,192 cached), 64 out: $0.00044 at list price');
    expect(costLine({ ...agent, usage: {} }, 0)).toBeNull();
  });

  test('all three as the one line the details box used to be a boxful of', () => {
    const paid = { ...agent, usage: { prompt: 7051, cached: 4192, completion: 64 } };
    const line = summaryLine({ agent: paid, timings: t, cost_usd: 0.00043842 });
    expect(line.split(' — ')).toEqual([
      'Understood by NVIDIA Nemotron 3.5 Lightning (nvidia/Nemotron-3_5-Lightning), on Nebius Token Factory',
      'Parakeet 0.41 s · Nemotron 1.20 s · core 0.012 s · server in all 1.70 s',
      '7,051 tokens in (4,192 cached), 64 out: $0.00044 at list price',
    ]);
    // A clicked answer asks no model at all: no tokens, nothing to leave a dangling dash.
    expect(summaryLine({ timings: { total_s: 0.01 } })).toBe(
      'Understood by Your answer to the robot’s question (no language model needed) — server in all 0.01 s',
    );
  });
});

describe('the live wiring panel (advanced-commands-plan.md 5.1)', () => {
  test('the lanes are the ones the server names, left to right', () => {
    expect(LANES.map((l) => l.id)).toEqual(['you', 'mediapipe', 'parakeet', 'page', 'nemotron', 'core', 'arm']);
    // Every lane says where it runs, under its name, short enough for one lane's width.
    expect(LANES.map((l) => l.runs)).toEqual([
      'at the table', 'your browser', 'our server', 'your browser', 'Nebius', 'over MCP', 'MuJoCo',
    ]);
    for (const lane of LANES) expect(lane.runs.length).toBeLessThanOrEqual(13);
    expect(laneOf('nemotron')).toBe(4);
    expect(laneOf('core')).toBe(5);
    // Nemotron's lane sits between the page's and the core's, as the learn diagrams have it.
    expect(lanePercent(0)).toBeCloseTo(100 / 14);
    expect(lanePercent(6)).toBeCloseTo(1300 / 14);
  });

  test('a row knows which way its arrow points, and what it cost', () => {
    const asked = rowShape({ from: 'page', to: 'nemotron', say: 'your words + the table', detail: { a: 1 } });
    expect([asked.left, asked.right, asked.rightwards, asked.single]).toEqual([3, 4, true, false]);
    expect(asked.opens).toBe(true);
    const back = rowShape({ from: 'nemotron', to: 'page', say: 'a plan: 3 tool calls', ms: 613 });
    expect([back.rightwards, back.time]).toEqual([false, '0.61 s']);
    // A row a part sends to itself (a step marker) is a dot, not an arrow.
    const note = rowShape({ from: 'page', to: 'page', say: 'step 2 of 3', kind: 'step' });
    expect([note.single, note.kind, note.opens]).toEqual([true, 'step', false]);
  });

  test('only the rows that cost something show a time', () => {
    expect(timeLabel(undefined)).toBe('');
    expect(timeLabel(0.4)).toBe('');
    expect(timeLabel(12)).toBe('12 ms');
    expect(timeLabel(99.6)).toBe('100 ms');
    expect(timeLabel(1904)).toBe('1.90 s');
  });
});

describe('the plan strip (advanced-commands-plan.md 5.2)', () => {
  const step = (label: string, state: StepState, why?: string): PlanStep => ({
    call: { name: 'move', args: {} }, label, resolved: null, live: false, move: null, state, why,
  }); // prettier-ignore

  test('each step lights as it runs', () => {
    const plan: Plan = {
      steps: [step('red onto blue', 'done'), step('green onto red', 'doing'), step('go home', 'todo')],
      replanned: false,
    };
    const line = planLine(plan);
    expect(line.title).toBe('Nemotron’s plan:');
    expect(line.steps.map((s) => `${s.text} ${s.mark}`)).toEqual([
      '1 red onto blue ✓',
      '2 green onto red ●',
      '3 go home ○',
    ]);
  });

  test('a step that could not be done, and the plan that came instead', () => {
    const plan: Plan = {
      steps: [step('blue onto green', 'failed', "There's already a block on that one.")],
      replanned: true,
    };
    const line = planLine(plan);
    expect(line.title).toBe('Nemotron re-planned:');
    expect(line.steps[0].mark).toBe(MARK.failed);
    expect(line.steps[0].why).toContain('already a block');
  });
});
