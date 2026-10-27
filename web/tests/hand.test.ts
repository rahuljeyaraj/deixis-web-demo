/**
 * The mouse as a pointer: the sample buffer, and the raycast against the 4-corner homography
 * on the table. (The pointing pose and the camera's active box went with the camera, 27 Oct.)
 */

import fs from 'node:fs';
import path from 'node:path';
import * as THREE from 'three';
import { describe, expect, test } from 'vitest';
import { SampleBuffer } from '../src/hand/samples';
import { Arm, type BlockId } from '../src/sim/arm';
import { solveLinear } from '../src/sim/ik';
import { pointAt } from '../src/sim/pick';
import { TableView } from '../src/sim/render';
import { AREA_MM, BLOCK_MM, tableToWorld, worldToTable } from '../src/sim/table';
import { nodeScene } from './scene';

describe('sample buffer', () => {
  test('keeps 20 s, in order, and finds a window', () => {
    const b = new SampleBuffer();
    for (let i = 0; i <= 900; i++) b.push({ t: i / 30, point: [i, 0], pointing: true }); // 30 s
    expect(b.latest?.t).toBe(30);
    expect(b.between(0, 9.9)).toEqual([]);
    expect(b.length).toBe(601); // 10 s .. 30 s
    b.push({ t: 29, point: null, pointing: false }); // late: dropped
    expect(b.latest?.t).toBe(30);
    expect(b.between(20, 21).length).toBe(31);
  });
});

/** A homography from 4 or more point pairs (direct linear transform, h33 = 1). */
function fitHomography(src: [number, number][], dst: [number, number][]): number[] {
  const A: number[][] = [];
  const rhs: number[] = [];
  src.forEach(([x, y], i) => {
    const [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    rhs.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    rhs.push(v);
  });
  return [...solveLinear(A, rhs), 1];
}

function applyHomography(H: number[], x: number, y: number): [number, number] {
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}

describe('raycast (plan 4.3)', async () => {
  const { mj, model, data } = await nodeScene();
  const arm = new Arm(mj, model, data);
  arm.goHomeNow();
  const view = new TableView(null, mj, model);
  const blockOfGeom = new Map<number, BlockId>([...arm.blocks.values()].map((b) => [b.geom, b.id]));

  /** Canvas fractions of a world point. */
  const screenOf = (x: number, y: number, zMm = 0): [number, number] => {
    const [X, Y] = tableToWorld(x, y);
    view.camera.updateMatrixWorld();
    const v = new THREE.Vector3(X, Y, zMm / 1000).project(view.camera);
    return [(v.x + 1) / 2, (1 - v.y) / 2];
  };

  test('on the table it equals the 4-corner homography within 0.01 mm', () => {
    // Blocks parked far out of the way, so every ray below reaches the table (or the arm).
    arm.putBlock('red', 480, 240);
    arm.putBlock('green', 470, 240);
    arm.putBlock('blue', 30, 240);
    view.sync(data);
    const corners: [number, number][] = [
      [AREA_MM.x, AREA_MM.y],
      [AREA_MM.x + AREA_MM.w, AREA_MM.y],
      [AREA_MM.x + AREA_MM.w, AREA_MM.y + AREA_MM.h],
      [AREA_MM.x, AREA_MM.y + AREA_MM.h],
    ];
    const H = fitHomography(corners.map(([x, y]) => screenOf(x, y)), corners);
    let worst = 0;
    let n = 0;
    for (let fy = 0.3; fy <= 0.98; fy += 0.04) {
      for (let fx = 0.05; fx <= 0.95; fx += 0.05) {
        const hit = pointAt(view, fx, fy, blockOfGeom);
        if (!hit || hit.block) continue;
        const [hx, hy] = applyHomography(H, fx, fy);
        worst = Math.max(worst, Math.hypot(hit.x - hx, hit.y - hy));
        n++;
      }
    }
    expect(n).toBeGreaterThan(150);
    console.log(`raycast vs homography: ${n} table points, worst ${worst.toExponential(2)} mm`);
    expect(worst).toBeLessThan(0.01);
  });

  test('pointing at the top of a block lands on the block, where the homography would miss', () => {
    arm.putBlock('red', 250, 110, 0);
    view.sync(data);
    const red = arm.block('red');
    const [fx, fy] = screenOf(red.tableMm[0], red.tableMm[1], BLOCK_MM); // its top face
    const hit = pointAt(view, fx, fy, blockOfGeom)!;
    expect(hit.block).toBe('red');
    expect(Math.abs(hit.x - red.tableMm[0])).toBeLessThan(BLOCK_MM / 2);
    expect(Math.abs(hit.y - red.tableMm[1])).toBeLessThan(BLOCK_MM / 2);
    // The table point under the same pixel is behind the block, outside its footprint.
    view.camera.updateMatrixWorld();
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2(fx * 2 - 1, 1 - fy * 2), view.camera);
    const onTable = ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), new THREE.Vector3())!;
    const behindMm = (red.center[0] - onTable.x) * 1000; // world -x is away from the person
    expect(behindMm).toBeGreaterThan(BLOCK_MM / 2);
  });

  test('a ray that misses the table is null', () => {
    expect(pointAt(view, 0.5, 0.0, blockOfGeom)).toBeNull(); // the sky above the table
  });

  test("the server tests' view (tests/fixtures/view.json) is this view", () => {
    // The Python tests (tests/test_arm_core.py) point a made-up fingertip through the same
    // camera and raycast it the same way. This fixture pins the camera and a grid of real
    // raycasts; run with UPDATE_FIXTURES=1 after changing the camera to write it again.
    const blocks: [BlockId, number, number][] = [
      ['red', 180, 100],
      ['green', 225, 100],
      ['blue', 300, 90],
    ];
    for (const [id, x, y] of blocks) arm.putBlock(id, x, y, 0);
    for (let i = 0; i < 200; i++) mj.mj_step(model, data); // let them settle on the table
    view.sync(data);
    const cases: { canvas: [number, number]; hit: { x: number; y: number; z: number; block: string | null } | null }[] = [];
    const add = (fx: number, fy: number) => {
      const h = pointAt(view, fx, fy, blockOfGeom);
      const r = (v: number) => Math.round(v * 1e4) / 1e4;
      cases.push({ canvas: [r(fx), r(fy)], hit: h && { x: r(h.x), y: r(h.y), z: r(h.z), block: h.block } });
    };
    for (let fy = 0.3; fy < 0.99; fy += 0.05) for (let fx = 0.05; fx < 0.99; fx += 0.1) add(fx, fy);
    for (const [id] of blocks) {
      const b = arm.block(id);
      add(...screenOf(b.tableMm[0], b.tableMm[1], BLOCK_MM));
    }
    const [ex, ey] = worldToTable(view.camera.position.x, view.camera.position.y);
    const fwd = view.camera.getWorldDirection(new THREE.Vector3());
    const view_ = {
      about: 'The page camera in table mm, and real raycasts with 3 blocks (yaw 0). Written by web/tests/hand.test.ts.',
      eye_mm: [ex, ey, view.camera.position.z * 1000].map((v) => Math.round(v * 1e4) / 1e4),
      // The direction in table axes: world X is table -y, world Y is table +x.
      forward: [fwd.y, -fwd.x, fwd.z].map((v) => Math.round(v * 1e8) / 1e8),
      fov_deg: view.camera.fov,
      aspect: view.camera.aspect,
      blocks: blocks.map(([id]) => ({ id, center_mm: arm.block(id).tableMm.map((v) => Math.round(v * 1e3) / 1e3) })),
      block_mm: BLOCK_MM,
      cases,
    };
    const file = path.join(import.meta.dirname, '..', '..', 'tests', 'fixtures', 'view.json');
    if (process.env.UPDATE_FIXTURES || !fs.existsSync(file)) fs.writeFileSync(file, JSON.stringify(view_, null, 1) + '\n');
    const saved = JSON.parse(fs.readFileSync(file, 'utf8'));
    expect(saved.eye_mm).toEqual(view_.eye_mm);
    expect(saved.forward).toEqual(view_.forward);
    expect([saved.fov_deg, saved.aspect]).toEqual([view_.fov_deg, view_.aspect]);
    expect(saved.cases.length).toBe(cases.length);
    saved.cases.forEach((c: (typeof cases)[number], i: number) => {
      expect(c.canvas).toEqual(cases[i].canvas);
      if (!c.hit || !cases[i].hit) return expect(c.hit).toEqual(cases[i].hit);
      expect(c.hit.block).toBe(cases[i].hit!.block);
      expect(Math.hypot(c.hit.x - cases[i].hit!.x, c.hit.y - cases[i].hit!.y)).toBeLessThan(0.01);
    });
  });
});
