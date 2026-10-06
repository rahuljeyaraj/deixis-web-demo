/** The coach line (final/web-demo.md items 2, 7, 13, 14, 19): the next step, or the problem. */

import { describe, expect, test } from 'vitest';
import { COACH, Coach } from '../src/ui/coach';

describe('the coach line', () => {
  test('finger: move the dot, then talk, then quiet; after the first move, your own way', () => {
    const c = new Coach();
    expect(c.line(0)).toBeNull(); // before the start card is left
    c.started('finger', 0);
    c.hand(true, 0.1);
    expect(c.line(0.2)?.text).toBe(COACH.moveFinger);
    c.hover(true);
    c.hand(true, 0.5);
    expect(c.line(0.6)?.text).toBe(COACH.talk);
    c.sent();
    c.hand(true, 1.0);
    expect(c.line(1.1)).toBeNull();
    expect(c.moved(2)).toBe(true);
    c.hand(true, 2);
    expect(c.line(2.5)?.text).toBe(COACH.ownWay);
    expect(c.moved(3)).toBe(false); // only once
    c.hand(true, 20);
    expect(c.line(20.1)).toBeNull();
  });

  test('no hand for 3 s after "Use my finger" (item 19); it clears when a hand comes', () => {
    const c = new Coach();
    c.started('finger', 10);
    expect(c.line(12)?.text).toBe(COACH.moveFinger);
    expect(c.line(13.1)?.text).toBe(COACH.noHand);
    c.hand(true, 13.2);
    expect(c.line(13.3)?.text).toBe(COACH.moveFinger);
  });

  test('problems first: no camera with "Try again", a silent clip, the server down', () => {
    const c = new Coach();
    c.started('mouse', 0);
    c.tell(COACH.noCamera, 0, 'retry-camera');
    expect(c.line(1)).toEqual({ text: COACH.noCamera, action: 'retry-camera' });
    expect(c.line(11)?.text).toBe(COACH.moveMouse); // it passes
    c.server(false);
    expect(c.line(12)?.text).toBe(COACH.serverDown);
    c.server(true);
    c.tell(COACH.silent, 12);
    expect(c.line(13)?.text).toBe(COACH.silent);
  });
});
