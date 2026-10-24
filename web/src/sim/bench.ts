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
