/**
 * The deixis table frame and the MuJoCo world (plan 4.4).
 *
 * Table frame: millimetres, origin at the near-left corner of the work area as the person
 * sees it, x to the person's right, y away from the person (toward the arm).
 * World: metres, z up, origin at the arm's base on the table top. The arm's forward
 * (shoulder_pan = 0) is +x, so the person sits on the +x side, facing -x.
 */

/** The arm's base in the table frame: the middle of the work area's far edge. */
export const BASE_MM = { x: 250, y: 250 } as const;

/**
 * The work area the core knows (`area_mm` in the profile): x, y, w, h.
 *
 * The near edge is at y = -40, not 0: the area is 290 mm deep, 10 mm more than the arm's
 * longest reach (REACH_MM.max = 280), so the whole reach mat fits inside it. Until 7 Oct it
 * was 250 mm deep, which is less than the reach, so the mat's outer arc was cut off flat at
 * the near edge (rahul saw it in the render). The base stays at y = 250 so every table
 * coordinate already logged still means the same place.
 */
export const AREA_MM = { x: 0, y: -40, w: 500, h: 290 } as const;

/**
 * The half ring around the base where the arm can both pick a block up and put one down, mm.
 * Measured on 5 Oct (tests/probes/reach2.probe.test.ts): places work from 130 to 310 mm, picks of a block
 * turned 0.3 rad from 130 to 280 mm. On 6 Oct the inner edge moved out to 140 mm: closer in,
 * the gripper can touch the arm's own shoulder (tests/probes/inner.probe.test.ts).
 */
export const REACH_MM = { min: 140, max: 280 } as const;

/** The blocks are cubes of this size. */
export const BLOCK_MM = 30;

/** The painted squares are this big (advanced-commands-plan.md 3.1). */
export const SQUARE_MM = 70;

/**
 * Three flat coloured squares painted on the reach mat: somewhere to put a block that is not
 * another block ("put each one on its own colour", "everything in the blue square"), and the
 * first use of the core's `inside` relation.
 *
 * They are painted, not objects: no physics, no weld, nothing to grip. They sit 210 mm from
 * the arm's base, so every corner is between 161 and 259 mm from it - well inside the pick
 * and place ring (REACH_MM) - and 100 mm of bare mat is left between them.
 */
export const SQUARES = [
  { id: 'red-square', color: 'red', xMm: 79.3, yMm: 127.7 },
  { id: 'green-square', color: 'green', xMm: 250, yMm: 40 },
  { id: 'blue-square', color: 'blue', xMm: 420.7, yMm: 127.7 },
] as const;

export type Square = (typeof SQUARES)[number];

/** How far a block's centre must stay from a square's centre to be clear of it. */
export const CLEAR_OF_SQUARE_MM = (SQUARE_MM + BLOCK_MM) / 2 + 5;

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

/** Table mm (x, y) to world metres (X, Y) on the table top. */
export function tableToWorld(x: number, y: number): Vec2 {
  return [(BASE_MM.y - y) / 1000, (x - BASE_MM.x) / 1000];
}

/** World metres (X, Y) to table mm (x, y). */
export function worldToTable(X: number, Y: number): Vec2 {
  return [BASE_MM.x + Y * 1000, BASE_MM.y - X * 1000];
}

/** Distance from the arm's base axis, mm. */
export function radiusMm(x: number, y: number): number {
  return Math.hypot(x - BASE_MM.x, y - BASE_MM.y);
}

/** Inside the drawn reach mat: the half ring on the person's side of the base, in the area. */
export function onMat(x: number, y: number): boolean {
  const r = radiusMm(x, y);
  const inArea = x >= AREA_MM.x && x <= AREA_MM.x + AREA_MM.w && y >= AREA_MM.y;
  return inArea && y <= BASE_MM.y && r >= REACH_MM.min && r <= REACH_MM.max;
}
