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

/** The work area the core knows (`area_mm` in the profile): x, y, w, h. */
export const AREA_MM = { x: 0, y: 0, w: 500, h: 250 } as const;

/**
 * The half ring around the base where the arm can both pick a block up and put one down, mm.
 * Measured on 5 Oct (tests/probes/reach2.probe.test.ts): places work from 130 to 310 mm, picks of a block
 * turned 0.3 rad from 130 to 280 mm. On 6 Oct the inner edge moved out to 140 mm: closer in,
 * the gripper can touch the arm's own shoulder (tests/probes/inner.probe.test.ts).
 */
export const REACH_MM = { min: 140, max: 280 } as const;

/** The blocks are cubes of this size. */
export const BLOCK_MM = 30;

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
