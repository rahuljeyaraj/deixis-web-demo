/**
 * The workbench frame's own arithmetic: where a thing **is**, and what that means for a grip.
 *
 * These are the sums that replaced reading the world's layout. Everything about a grip used to
 * come out of `workbench_reach.json`, which says where each thing **started**: a jar carried
 * from the bottom bay to the top one was still described as a bottom-bay jar, so the arm
 * reached into the bottom bay for it and swept the real one along with its forearm (rahul,
 * 24 Oct - "i picked one jar from bottom shelf and put it in top, then when i picked up the
 * same jar from top, it picked up it from bottom").
 *
 * They are small enough to be obviously right and important enough to be held still.
 */

import { describe, expect, test } from 'vitest';

import { RING, SHELF, THINGS, bayAt, bayGripZ, ringNear, thing } from '../src/sim/bench';

describe('which bay a spot is in', () => {
  test('worked out from across, away and up together', () => {
    const [front] = SHELF.rows;
    expect(bayAt(300, front, 0)).toBe(0);
    expect(bayAt(300, front, 122)).toBe(1);
    expect(bayAt(300, front, 244)).toBe(2);
  });

  test('the bench top in front of the unit is no bay, at any height', () => {
    expect(bayAt(300, 170, 0)).toBeNull();
    expect(bayAt(300, 340, 0)).toBeNull();
  });

  test('nor is the bench past the end of the unit', () => {
    // The bottle of oil stands on the bench at the same height as the bottom bay's floor.
    // Without the test across, it read as a thing in that bay far too tall for it - which is
    // the one thing the oil is not.
    const oil = thing('oil');
    expect(oil.bay).toBeNull();
    expect(bayAt(oil.at[0], oil.at[1], 0)).toBeNull();
  });

  test('a thing that has been moved is found in the bay it is in now', () => {
    const jar = thing('jar1');
    expect(jar.bay).toBe(0); // where it started
    expect(bayAt(jar.at[0], jar.at[1], 244)).toBe(2); // and where it might be put
  });
});

describe('the nearest ring that was really swept', () => {
  test('every bay height is answered by a ring of its own', () => {
    for (let n = 0; n < SHELF.bays.length; n++) {
      expect(ringNear('level', bayGripZ(n))).toBe(bayGripZ(n));
    }
  });

  test('a grip worked out from a live place takes the nearest of them', () => {
    expect(ringNear('level', 40)).toBe(55);
    expect(ringNear('level', 165)).toBe(177);
    expect(ringNear('down', 7)).toBe(35);
    expect(ringNear('down', 63)).toBe(70);
  });

  test('and nothing at all where none was swept, rather than a guess', () => {
    // Reaching **down** dies above 100 mm: a thing whose grip from above would be up there has
    // no way in from above, and that is an answer and not a figure to interpolate.
    expect(ringNear('down', 150)).toBeNull();
    expect(ringNear('level', 600)).toBeNull();
  });

  test('every ring named is one the generator wrote', () => {
    for (const key of Object.keys(RING)) {
      const [how, z] = key.split(' ');
      expect(ringNear(how, Number(z))).toBe(Number(z));
    }
  });
});

describe('how high up its own side a thing is gripped', () => {
  test('a level grip carries the figure; a grip from above does not', () => {
    expect(thing('jar1').gripAbove).toBe(55);
    expect(thing('tin1').gripAbove).toBe(55);
    expect(thing('oil').gripAbove).toBe(130);
    expect(thing('screwdriver1').gripAbove).toBeNull();
  });

  test('it is a height on the thing, not a height above the bench', () => {
    // The point of it: a tin in the middle bay is gripped 55 mm up itself, which is 177 mm
    // above the bench today and 55 above it if it is ever stood on the bench.
    for (const t of THINGS) {
      if (t.gripHow !== 'level') continue;
      expect(t.gripAbove).toBe(t.gripZ - t.rises[0]);
      expect(t.gripAbove! + t.rises[0]).toBeLessThan(t.rises[1]);
    }
  });
});

describe('where the room a thing takes up actually sits', () => {
  test('nothing long here stands in the middle of its own footprint', () => {
    // A screwdriver's handle is all at one end, so a footprint drawn about the middle of one
    // covers 50 mm of bare bench and misses 50 mm of screwdriver. That is what the occupancy
    // test needs and what it did not have: as a circle about its middle instead, a 193 mm
    // screwdriver is a 96 mm disc that overlaps everything within four inches.
    const driver = thing('screwdriver1');
    expect(driver.room[0]).toBeGreaterThan(190);
    expect(driver.box[0]).toBeGreaterThan(40);
    // The hammer is the other shape of the same fault: its head is heavy rather than long, so
    // its footprint is only a little off its middle but 111 mm deep where its handle is 22.
    const hammer = thing('hammer');
    expect(Math.abs(hammer.box[0])).toBeGreaterThan(5);
    expect(hammer.room[0]).toBeGreaterThan(200);
    expect(hammer.room[1]).toBeGreaterThan(100);
  });

  test('and a round thing stands in the middle of its own', () => {
    for (const t of THINGS) {
      if (!t.round) continue;
      expect(Math.abs(t.box[0])).toBeLessThan(1);
      expect(Math.abs(t.box[1])).toBeLessThan(1);
    }
  });
});
