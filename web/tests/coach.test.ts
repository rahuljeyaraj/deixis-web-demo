/** The coach line (final/web-demo.md items 2, 13, 14): the next step, or the problem. */

import { describe, expect, test } from 'vitest';
import { COACH, Coach } from '../src/ui/coach';

describe('the coach line', () => {
  test('move the mouse, then talk, then quiet; after the first move, your own way', () => {
    const c = new Coach();
    expect(c.line(0)).toBeNull(); // before the page is ready
    c.started(0);
    expect(c.line(0.2)?.text).toBe(COACH.moveMouse);
    c.hover(true);
    expect(c.line(0.6)?.text).toBe(COACH.talk);
    c.sent();
    expect(c.line(1.1)).toBeNull();
    expect(c.moved(2)).toBe(true);
    expect(c.line(2.5)?.text).toBe(COACH.ownWay);
    expect(c.moved(3)).toBe(false); // only once
    expect(c.line(20.1)).toBeNull();
  });

  test('problems first: a silent clip, no microphone, the server down', () => {
    const c = new Coach();
    c.started(0);
    c.tell(COACH.noMic, 0);
    expect(c.line(1)).toEqual({ text: COACH.noMic });
    expect(c.line(11)?.text).toBe(COACH.moveMouse); // it passes
    c.server(false);
    expect(c.line(12)?.text).toBe(COACH.serverDown);
    c.server(true);
    c.tell(COACH.silent, 12);
    expect(c.line(13)?.text).toBe(COACH.silent);
  });
});
