/**
 * The scene for the core, built from the simulation at every command (plan 6.2), and the
 * robot-side check that a place is free.
 *
 * Three kinds of thing go to the core. The blocks, which can be picked up. The three painted
 * squares, which cannot (`selectable: false`) but can be named and put things in. And, while
 * a plan is running, the spot the last step emptied: that is what lets one block go where
 * another one was, which is what a swap needs (advanced-commands-plan.md section 4,
 * command 10). All three are just things the robot side reports, as the core's L6 expects:
 * no change to the core.
 */

import type { Thing } from '../core';
import type { BlockState, Place } from '../sim/arm';
import { BLOCK_MM, SQUARES, SQUARE_MM, type Vec3, tableToWorld } from '../sim/table';

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
    // The core matches every word said against the label and these values, so "the green
    // box" and "the green cube" find the green block too (rahul, 5 Oct). `one` is there
    // because "the red one" now names two things: the block and the red square. The core
    // takes an exact match over a word-by-word one, so an exact "red one" on the block
    // keeps the demo's oldest phrase meaning the block it has always meant.
    attrs: { color: b.id, one: `${b.id} one`, also: 'box cube' },
    box_mm: footprint(b),
    height_mm: Math.round(b.heightMm * 10) / 10,
    t: movedAt(b.id),
    // A block with another on top can't be picked up.
    selectable: !below.has(b.id),
  }));
}

/**
 * The painted squares, as the core sees them. Not selectable: a square is a place to put a
 * block, never a thing to pick up. They carry no bare colour word of their own, so "the red
 * one" still finds the red block and only "the red square" finds the square.
 */
export function squareThings(): Thing[] {
  return SQUARES.map((s) => ({
    id: s.id,
    label: `${s.color} square`,
    attrs: { also: 'patch area zone' },
    box_mm: [s.xMm - SQUARE_MM / 2, s.yMm - SQUARE_MM / 2, SQUARE_MM, SQUARE_MM],
    height_mm: 0,
    t: 0,
    selectable: false,
  }));
}

/** The label the core knows the spot a step emptied by. One at a time, so it never ties. */
export const SPOT_LABEL = 'the spot it left';

/**
 * The spot a block has just been moved away from, as a thing the core can be asked about
 * ("put the blue one where the red one was"). Not selectable, and gone again as soon as
 * something is put there.
 */
export function spotThing(xMm: number, yMm: number, t: number): Thing {
  return {
    id: 'left-spot',
    label: SPOT_LABEL,
    attrs: { also: 'gap' },
    box_mm: [xMm - BLOCK_MM / 2, yMm - BLOCK_MM / 2, BLOCK_MM, BLOCK_MM],
    height_mm: 0,
    t,
    selectable: false,
  };
}

/** A thing's centre in the table frame. */
export function centreOf(t: Thing): [number, number] {
  return [t.box_mm[0] + t.box_mm[2] / 2, t.box_mm[1] + t.box_mm[3] / 2];
}

/**
 * Everything the core is told about, in one list: the blocks, the three painted squares, and
 * the spot the last step of a plan emptied, if there is one.
 */
export function sceneFor(
  blocks: BlockState[],
  movedAt: (id: string) => number,
  spot: [number, number] | null,
  t: number,
): Thing[] {
  const scene = [...thingsFor(blocks, movedAt), ...squareThings()];
  return spot ? [...scene, spotThing(spot[0], spot[1], t)] : scene;
}

/**
 * The blocks as they will be once `id` has been put down at `place`: used to show a whole
 * plan before the arm starts, so step 2 is planned against what step 1 will have made.
 *
 * It is a guess, and it is treated as one. When the arm reaches a step, the core measures it
 * again against the real table (`/api/step`), so a guess that turns out wrong is caught and
 * not acted on.
 */
export function blocksAfter(blocks: BlockState[], id: string, place: Place): BlockState[] {
  return blocks.map((b) =>
    b.id === id
      ? {
          ...b,
          tableMm: [place.xMm, place.yMm] as [number, number],
          heightMm: place.zMm + BLOCK_MM,
          center: [...tableToWorld(place.xMm, place.yMm), (place.zMm + BLOCK_MM / 2) / 1000] as Vec3,
        }
      : b,
  );
}

/**
 * The spot a block leaves behind when it moves away from `from`: a place the core can be
 * asked about by name ("the spot it left"), which is how one block goes where another one
 * was. Null when a block is standing there after the move, because then it is not a spot.
 */
export function spotLeft(from: [number, number], after: BlockState[]): [number, number] | null {
  const taken = after.some(
    (b) => Math.hypot(b.tableMm[0] - from[0], b.tableMm[1] - from[1]) < BLOCK_MM * 0.6,
  );
  return taken ? null : from;
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
