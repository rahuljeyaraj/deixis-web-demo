/**
 * The workbench frame, and what stands in it (`workbench-requirements.md` 4, FR5).
 *
 * Bench frame: millimetres, **x across to the person's right, y away from them, z up from the
 * bench top**, centre line x = 350. World: metres, z up, origin on the bench top in the middle
 * of the person's edge, +X away from the person towards the shelf, +Y to their left.
 *
 * This is **not** `table.ts`, which is the cubes demo's frame (one arm at the origin, centre
 * line 250), and nothing may mix the two: a number in one frame means somewhere else in the
 * other. Each scene keeps its own.
 *
 * Every number here is read from `workbench_reach.json`, which
 * `scripts/make_workbench_scene.py` writes from `workbench/scene.py` - the same Python the
 * physics scene is generated from. Nothing in this file is typed out a second time, because a
 * second copy is a copy that can drift, and a drift between the arm's idea of the shelf and
 * the shelf it reaches into is invisible until it knocks something over.
 */

import reach from '../../model/workbench_reach.json';

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

/** One thing in the world, as the generator describes it. */
export interface Thing {
  id: string;
  /** What to call it, in the fewest words that name it (FR9). */
  what: string;
  /** Things of the same kind cannot be told apart by any word. Four jars, two tins, ... */
  kind: string;
  /** Which bay it stands in, or null for the bench top. */
  bay: number | null;
  /** Its **centre**, bench mm. */
  at: Vec3;
  /** Its first shape's box in its own frame: across, deep, up. */
  size: Vec3;
  /** Which way a cylinder's length runs before it is turned: up, across or deep. */
  axis: string;
  /** How it is turned on the bench, degrees. */
  yaw: number;
  /** How much room it takes, across and deep, with every part counted. */
  room: Vec2;
  /** Where that room sits, from its own middle: a hammer's head is all at one end. */
  box: Vec2;
  /** Its lowest and highest points above the bench. */
  rises: Vec2;
  /** The same from every side: a jar, a tin, a pot - not a mug, which has a handle. */
  round: boolean;
  /** How wide the part the gripper closes on is. */
  gripMm: number;
  /** How the hand has to come at it: `down` from above, `level` in under a board. */
  gripHow: string;
  /** The height the reach ring for that grip was swept at. */
  gripZ: number;
  /**
   * How far above its **own bottom** the jaws close on it, mm; null for a thing taken from
   * above, which is gripped just above whatever it rests on.
   *
   * This is what lets a grip be worked out from where a thing **now** stands. `gripZ` is a
   * height above the bench, so a jar carried up to the top bay is still described by a height
   * 244 mm below where it is.
   */
  gripAbove: number | null;
  /** False for a thing nobody would ask for but anybody might point at. */
  wanted: boolean;
  /** False for a thing never picked up at all, only put into. */
  graspable: boolean;
  /** Where the carriage should slide to to work on it, bench mm, or null for nowhere. */
  slideTo: number | null;
}

export const THINGS = reach.things as Thing[];

/** The bench top: the person stands at the near edge, the shelf unit on the far one. */
export const BENCH = reach.bench;
/** The rail the carriage rides: how far back it runs and how far along it can travel. */
export const RAIL = reach.rail;
/** The shelf unit: its footprint, its three bays as (floor, ceiling), and its two rows. */
export const SHELF = reach.shelf;
/** Where a thing is put down for the person to take, world metres (x, y). */
export const GIVE = reach.give as Vec2;
/** The reach rings stage 0 measured, by `<how> <height>`: how close in, and how far out. */
export const RING = reach.ringMm as unknown as Record<string, [number, number]>;

const byId = new Map(THINGS.map((t) => [t.id, t]));

export function thing(id: string): Thing {
  const got = byId.get(id);
  if (!got) throw new Error(`no thing called ${id}`);
  return got;
}

/** Bench mm (x, y) to world metres (X, Y). The only place this conversion is written here. */
export function benchToWorld(x: number, y: number): Vec2 {
  return [y / 1000, -(x - BENCH.centreX) / 1000];
}

/** World metres (X, Y) back to bench mm (x, y). */
export function worldToBench(X: number, Y: number): Vec2 {
  return [BENCH.centreX - Y * 1000, X * 1000];
}

/**
 * Where the carriage's slide joint has to be, metres, for the arm to stand at a place across
 * the bench.
 *
 * The joint runs along world **-Y**, because world Y is minus the bench's x; the generator
 * writes the same conversion for the keyframe and says why there.
 */
export function slideOf(xMm: number): number {
  return (xMm - BENCH.centreX) / 1000;
}

/** And back: where on the bench the carriage is, from the slide joint. */
export function slideAt(metres: number): number {
  return metres * 1000 + BENCH.centreX;
}

/** How far along the rail the carriage may go, bench mm. */
export function onRail(xMm: number): number {
  return Math.min(RAIL.to, Math.max(RAIL.from, xMm));
}

/** A bay's floor and the underside of the board over it, mm. */
export function bay(n: number): Vec2 {
  return SHELF.bays[n] as Vec2;
}

/** How many bays the unit has. */
export const BAYS = SHELF.bays.length;

/** The height the hand works at inside a bay: the middle of its clear opening, mm. */
export function bayGripZ(n: number): number {
  return SHELF.gripZ[n];
}

/**
 * Which bay a spot is in - **worked out, not remembered** - or null for the bench top.
 *
 * `zMm` is the height a thing's bottom rests at. All three of across, away and up have to
 * agree: the bottle of oil stands on the bench past the end of the unit at the same height as
 * the bottom bay's floor, and without the first two tests it would read as a thing in that bay.
 *
 * The bottom bay's floor **is** the bench top, so a thing standing on the bench behind the
 * unit's front edge is in the bottom bay, which is the truth: there is a board over it.
 */
export function bayAt(xMm: number, yMm: number, zMm: number): number | null {
  if (xMm < SHELF.from[0] || xMm > SHELF.to[0]) return null;
  if (yMm < SHELF.from[1] || yMm > SHELF.to[1]) return null;
  for (let n = 0; n < BAYS; n++) {
    const [floor, ceiling] = bay(n);
    if (zMm > floor - 6 && zMm < ceiling - 6) return n;
  }
  return null;
}

/**
 * Where a thing's grip ring says the arm can work, from a carriage position: how close in and
 * how far out, mm. Stage 0 swept each of these at the height it is used at
 * (`out/stage0-bay-ring.txt`); asking for one that was never swept is a programming mistake
 * rather than a refusal, because it means a thing is being gripped at a height nobody measured.
 */
export function ringFor(how: string, z: number): [number, number] {
  const got = RING[`${how} ${z}`];
  if (!got) throw new Error(`no ring was swept for ${how} at ${z} mm`);
  return got;
}

/**
 * The nearest height a ring was swept at for this way of coming at a thing, mm.
 *
 * The rings are sparse on purpose - every one of them is a height something in this world is
 * really gripped at and nothing is interpolated - so a grip worked out from where a thing now
 * stands has to be answered by the nearest of them. Null when there is none within reason:
 * reaching **down** dies above 100 mm, and a thing whose grip would be up there has no way in
 * from above at all, which is an answer rather than a guess.
 */
export function ringNear(how: string, zMm: number): number | null {
  let best: number | null = null;
  for (const key of Object.keys(RING)) {
    const [itsHow, its] = key.split(' ');
    if (itsHow !== how) continue;
    const z = Number(its);
    if (best === null || Math.abs(z - zMm) < Math.abs(best - zMm)) best = z;
  }
  return best !== null && Math.abs(best - zMm) <= 45 ? best : null;
}

/** How much of the outward reach is lost per degree the shoulder turns off forward, mm. */
const FALL_MM_PER_DEG = 0.48;
/** How far either side of forward the shoulder can work, degrees. */
const PAN_LIMIT_DEG = 100;

/** How much room to spare a carriage position has for working at a spot, mm; negative: none. */
export function marginFrom(slideX: number, xMm: number, yMm: number, how: string, z: number): number {
  const bearing = Math.abs((Math.atan2(-(xMm - slideX), yMm - RAIL.y) * 180) / Math.PI);
  if (bearing > PAN_LIMIT_DEG) return -(bearing - PAN_LIMIT_DEG) * 10;
  const [near, far] = ringFor(how, z);
  const r = Math.hypot(xMm - slideX, yMm - RAIL.y);
  return Math.min(r - near, far - FALL_MM_PER_DEG * bearing - r);
}

/**
 * Where to slide to, to work at a spot with the most room to spare - or null for nowhere.
 *
 * **Not simply "straight in front of it".** The arm drags a dead circle about with it: reaching
 * down it cannot work closer in than 135 mm, and the place a thing is handed over at is 130 mm
 * from the rail. Parked square in front of that, the arm is standing on top of it. Standing off
 * to one side puts it at arm's length, which is what `workbench/scene.py` does for every thing
 * in the world and what this is the other half of.
 *
 * Searched rather than solved, in 5 mm steps: the ring's outer edge falls with the bearing, so
 * the best place on the rail is not always the nearest one.
 */
export function slideFor(xMm: number, yMm: number, how: string, z: number): number | null {
  let best: number | null = null;
  let most = 0;
  for (let here = RAIL.from; here <= RAIL.to; here += 5) {
    const margin = marginFrom(here, xMm, yMm, how, z);
    if (margin > most) {
      best = here;
      most = margin;
    }
  }
  return best;
}
