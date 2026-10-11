/**
 * The page's panels (plan W10) on the server's real replies (`tests/fixtures/replies.json`):
 * the thinking strip's words, holds and links, and the details panel's lines.
 */

import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, test } from 'vitest';
import type { HeardWord } from '../src/api';
import type { CoreResponse } from '../src/core';
import { callLine, coreJson, costLine, pickLine, timingsLine, understoodBy } from '../src/ui/details';
import { layoutWords, stripModel } from '../src/ui/strip';

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

describe('the details panel', () => {
  test('a tool call reads as one line, with references, not coordinates', () => {
    const line = callLine({
      name: 'move',
      args: { object: { deixis: 'that', word_index: 1 }, place: { deixis: 'here', word_index: 2 } },
    });
    expect(line).toBe('move(object={"deixis":"that","word_index":1}, place={"deixis":"here","word_index":2})');
  });

  test("the pick that answered the robot's question reads as a line too", () => {
    // Named: the model chose it. Pointed: the model handed it to the core, and the line says so.
    expect(pickLine({ choice: 1, word_index: null })).toBe('pick(choice=1)');
    expect(pickLine({ choice: null, word_index: 0 })).toContain('pick(word_index=0)');
    expect(pickLine({ choice: null, word_index: 0 })).toContain('the core says which');
  });

  test("the core's answer without empty fields or its working notes", () => {
    const json = coreJson(result('put that here')!);
    const back = JSON.parse(json);
    expect(back.status).toBe('ok');
    expect(back.calls[0].args.place.mean_mm).toEqual([330, 50]);
    expect(back).not.toHaveProperty('log');
    expect(back).not.toHaveProperty('ask');
    expect(json).not.toContain('null');
  });

  test('who understood, and how long it took', () => {
    const agent = { model: 'nvidia/Nemotron-3_5-Lightning', calls: [], raw: [], errors: [], attempts: 1 };
    expect(understoodBy({ ...agent, source: 'nemotron' })).toContain('Nemotron 3.5 Lightning');
    expect(understoodBy({ ...agent, source: 'rules', errors: ['timeout'] })).toContain('timeout');
    expect(understoodBy(undefined)).toContain('answer');
    const t = { asr_s: 0.41, agent_s: 1.2, core_s: 0.012, total_s: 1.7 };
    expect(timingsLine(t, { ...agent, source: 'nemotron' })).toBe(
      'Parakeet 0.41 s · Nemotron 1.20 s · core 0.012 s · server in all 1.70 s',
    );
    expect(timingsLine(t)).toContain('word matcher 1.20 s');
  });

  test('tokens and cost at list price (item 11)', () => {
    const agent = {
      source: 'nemotron' as const, model: 'm', calls: [], raw: [], errors: [], attempts: 1,
      usage: { prompt: 7051, cached: 4192, completion: 64 },
    };
    expect(costLine(agent, 0.00043842)).toBe('7,051 tokens in (4,192 cached), 64 out: $0.00044 at list price');
    expect(costLine({ ...agent, usage: {} }, 0)).toBeNull();
  });
});
