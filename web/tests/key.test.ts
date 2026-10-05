/**
 * The visitor's own key and the day's limits (plan 8.1, 8.2, W11): what the page shows, the
 * key in the command's header only, and waiting when the server says one at a time.
 */

import fs from 'node:fs';
import path from 'node:path';
import { afterEach, describe, expect, test, vi } from 'vitest';
import { type Limits, sendCommand } from '../src/api';
import { keyProblem, limitsShown } from '../src/ui/key';

const ours: Limits = {
  mode: 'ours',
  left_today: 27,
  per_day: 30,
  why: null,
  banner: null,
  off: null,
  own_key_helps: false,
};
const usedUp: Limits = {
  ...ours,
  mode: 'simple',
  left_today: 0,
  why: 'total',
  banner: 'Today’s AI budget is used up, so a simple word matcher is answering.',
  off: 'off right now: today’s budget is used up',
  own_key_helps: true,
};

describe('the own-key box', () => {
  test('a key is one long word', () => {
    expect(keyProblem('')).toContain('Paste');
    expect(keyProblem('  sk-short ')).toContain('short');
    expect(keyProblem('sk-abc def ghi jkl mno pqr')).toContain('spaces');
    expect(keyProblem('sk-ünïcödé-0123456789abcdef')).toContain('unusual');
    expect(keyProblem(`sk-${'x'.repeat(5000)}`)).toContain('long');
    expect(keyProblem('  eyJhbGciOiJIUzI1NiJ9.abcdefghijklmnop  ')).toBeNull();
  });

  test('the key lives only in memory: never in storage or a cookie', () => {
    for (const file of ['src/ui/key.ts', 'src/api.ts', 'src/main.ts']) {
      const src = fs.readFileSync(path.join(import.meta.dirname, '..', file), 'utf8');
      expect(src).not.toMatch(/\b(localStorage|sessionStorage|indexedDB)\s*[.[]|document\.cookie/);
    }
  });
});

describe('the banner and the models line', () => {
  test('our key, with commands left', () => {
    expect(limitsShown(ours, false)).toEqual({
      banner: null,
      offerKey: false,
      off: null,
      left: 'Our key: 27 of 30 commands left for you today.',
    });
  });

  test('simple mode: the banner, the offer of a key, the note on Nemotron', () => {
    const s = limitsShown(usedUp, false);
    expect(s.banner).toContain('simple word matcher');
    expect(s.offerKey).toBe(true);
    expect(s.off).toContain('used up');
  });

  test('with their own key, no banner and Nemotron is on', () => {
    expect(limitsShown(usedUp, true)).toMatchObject({ banner: null, offerKey: false, off: null });
  });

  test('when the server runs the word matcher only, a key would not help', () => {
    const rules: Limits = { ...usedUp, why: 'rules', own_key_helps: false };
    expect(limitsShown(rules, false).offerKey).toBe(false);
  });
});

describe('sending a command', () => {
  afterEach(() => vi.unstubAllGlobals());

  const ok = () => new Response(JSON.stringify({ words: [], limits: ours }), { status: 200 });
  const busy = (wait_s: number) =>
    new Response(JSON.stringify({ detail: 'one command at a time, please', wait_s }), {
      status: 429,
      headers: { 'Retry-After': '1' },
    });

  test('the own key goes in a header, and only when there is one', async () => {
    const fetch = vi.fn(async () => ok());
    vi.stubGlobal('fetch', fetch);
    await sendCommand({ text: 'yes' }, undefined, 'sk-visitor-9999999999');
    await sendCommand({ text: 'yes' }, undefined, null);
    const headers = fetch.mock.calls.map((c) => (c as unknown as [string, RequestInit])[1].headers);
    expect(headers).toEqual([{ 'X-Nebius-Key': 'sk-visitor-9999999999' }, {}]);
  });

  test('told to wait, it waits that long and sends the same command again', async () => {
    const fetch = vi.fn().mockResolvedValueOnce(busy(0.05)).mockResolvedValueOnce(ok());
    vi.stubGlobal('fetch', fetch);
    const waits: number[] = [];
    const t0 = performance.now();
    const reply = await sendCommand({ text: 'yes' }, undefined, null, (s) => waits.push(s));
    expect(performance.now() - t0).toBeGreaterThanOrEqual(90);
    expect(waits).toEqual([0.05]);
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(fetch.mock.calls[1][1].body).toBe(fetch.mock.calls[0][1].body);
    expect(reply.limits.mode).toBe('ours');
  });

  test('it gives up after two waits', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => busy(0.01)));
    await expect(sendCommand({ text: 'yes' })).rejects.toThrow('one command at a time');
  });
});
