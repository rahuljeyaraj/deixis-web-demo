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
    attrs: { color: b.id },
    box_mm: footprint(b),
    height_mm: Math.round(b.heightMm * 10) / 10,
    t: movedAt(b.id),
    // A block with another on top can't be picked up.
    selectable: !below.has(b.id),
  }));
}

/**
 * Which other block a 30 mm block put at (x, y) would overlap, if any. `ignore`: the block
 * being moved, and the one it goes on top of.
 */
export function blockedBy(
  x: number,
  y: number,
  blocks: BlockState[],
  ignore: string[],
  marginMm = 2,
): BlockState | null {
  const h = BLOCK_MM / 2;
  for (const b of blocks) {
    if (ignore.includes(b.id)) continue;
    const [bx, by, bw, bh] = footprint(b);
    const overlapX = Math.min(x + h, bx + bw) - Math.max(x - h, bx);
    const overlapY = Math.min(y + h, by + bh) - Math.max(y - h, by);
    if (overlapX > marginMm && overlapY > marginMm) return b;
  }
  return null;
}
