/**
 * Pointing into the scene (plan 4.3): a ray from the camera through a screen point, against
 * the blocks and the table. The fingertip and the mouse both use it. On a block it gives a
 * point on the block, so its x, y fall inside the block's footprint (a homography on the table
 * plane would land behind a block); on the table it gives the table point.
 */

import * as THREE from 'three';
import type { BlockId } from './arm';
import type { TableView } from './render';
import { worldToTable } from './table';

export interface Hit {
  /** Table frame, mm. */
  x: number;
  y: number;
  /** Height of the hit point above the table, mm. */
  z: number;
  /** The block the ray hit first, if any. */
  block: BlockId | null;
}

const ray = new THREE.Raycaster();
const ndc = new THREE.Vector2();

/**
 * The hit for a point given as fractions of the canvas (0..1 from the left and from the top),
 * or null if the ray misses the table and the blocks.
 */
export function pointAt(view: TableView, fx: number, fy: number, blockOfGeom: Map<number, BlockId>): Hit | null {
  ndc.set(fx * 2 - 1, -(fy * 2 - 1));
  ray.setFromCamera(ndc, view.camera);
  for (const h of ray.intersectObjects(view.pickable, false)) {
    const geom = h.object.userData.geom as number | undefined;
    const block = geom === undefined ? null : (blockOfGeom.get(geom) ?? null);
    if (geom !== undefined && block === null && !view.isTable(geom)) continue; // the arm itself
    const [x, y] = worldToTable(h.point.x, h.point.y);
    return { x, y, z: h.point.z * 1000, block };
  }
  return null;
}
