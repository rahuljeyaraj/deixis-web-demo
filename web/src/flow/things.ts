/**
 * The blocks for the core, built from the simulation at every command (plan 6.2), and the
 * robot-side check that a place is free.
 */

import type { Thing } from '../core';
import type { BlockState } from '../sim/arm';
import { BLOCK_MM } from '../sim/table';

/** The axis-aligned box around a cube's corners (it may be turned), table mm. */
export function footprint(b: Pick<BlockState, 'tableMm' | 'yaw'>): [number, number, number, number] {
  // In the table frame the cube is turned by its yaw too (the frames differ by 90 degrees).
  const h = BLOCK_MM / 2;
  const c = Math.abs(Math.cos(b.yaw));
  const s = Math.abs(Math.sin(b.yaw));
  const half = h * (c + s);
  const r = (v: number) => Math.round(v * 10) / 10;
  return [r(b.tableMm[0] - half), r(b.tableMm[1] - half), r(2 * half), r(2 * half)];
}

/** The block right under `b`, if `b` sits on another one. */
export function under(b: BlockState, all: BlockState[]): BlockState | null {
  return (
    all.find(
      (o) =>
        o.id !== b.id &&
        Math.hypot(o.tableMm[0] - b.tableMm[0], o.tableMm[1] - b.tableMm[1]) < BLOCK_MM * 0.6 &&
        Math.abs(b.heightMm - o.heightMm - BLOCK_MM) < BLOCK_MM * 0.4,
    ) ?? null
  );
}

export function thingsFor(blocks: BlockState[], movedAt: (id: string) => number): Thing[] {
  const below = new Set(blocks.map((b) => under(b, blocks)?.id).filter(Boolean));
  return blocks.map((b) => ({
    id: b.id,
    label: `${b.id} block`,
    // The core matches every word said against the label and these values, so "the yellow
    // box" and "the yellow cube" find the yellow block too (rahul, 5 Oct).
    attrs: { color: b.id, also: 'box cube' },
    box_mm: footprint(b),
    height_mm: Math.round(b.heightMm * 10) / 10,
    t: movedAt(b.id),
    // A block with another on top can't be picked up.
    selectable: !below.has(b.id),
  }));
}

/** Distance (mm) from a point to a block's real (turned) square footprint; 0 inside it. */
export function distanceToBlock(x: number, y: number, b: Pick<BlockState, 'tableMm' | 'yaw'>): number {
  // Into the block's own axes (any sign of turn works: a square is symmetric).
  const dx = x - b.tableMm[0];
  const dy = y - b.tableMm[1];
  const c = Math.cos(b.yaw);
  const s = Math.sin(b.yaw);
  const u = Math.abs(c * dx + s * dy) - BLOCK_MM / 2;
  const v = Math.abs(-s * dx + c * dy) - BLOCK_MM / 2;
  return Math.hypot(Math.max(u, 0), Math.max(v, 0));
}

/**
 * Which other block is in the way of a block put down at (x, y), if any. The block put down
 * may end up turned any way, so it counts as its inner circle (15 mm) and needs `clearMm`
 * more; the others count with their real, turned squares. `ignore`: the block being moved.
 */
export function blockedBy(
  x: number,
  y: number,
  blocks: BlockState[],
  ignore: string[],
  clearMm = 1,
): BlockState | null {
  let worst: BlockState | null = null;
  let closest = BLOCK_MM / 2 + clearMm;
  for (const b of blocks) {
    if (ignore.includes(b.id)) continue;
    const d = distanceToBlock(x, y, b);
    if (d < closest) {
      closest = d;
      worst = b;
    }
  }
  return worst;
}
