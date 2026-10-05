import { test } from 'vitest';
import { ArmKinematics } from '../../src/sim/ik';
import { nodeScene } from '../scene';

// The resting pose should keep the arm off the work area, so every block is in view.
test('probe: home pose candidates', async () => {
  const { mj, model } = await nodeScene();
  const ik = new ArmKinematics(mj, model);
  const body = (q: number[]) => {
    const d = ik['scratch'] as any; // the wrist body's position too
    ik.fk(q);
    const w = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, 'wrist');
    return [d.xpos[3 * w], d.xpos[3 * w + 2]].map((v: number) => (v * 1000).toFixed(0)).join(',');
  };
  for (const q of [[0, -1.74, -0.6, 0, 0], [0, -1.74, -1.0, 0.6, 0], [0, -1.5, -0.8, 0.9, 0], [0, -1.74, -0.3, 0.8, 0], [0, -1.6, 0.2, 1.4, 0], [0, -1.74, 0, 1.0, 0], [0, -1.3, -1.2, 1.2, 0]]) {
    const p = ik.fk(q);
    console.log(JSON.stringify(q), 'site', p.pos.map((v) => (v * 1000).toFixed(0)).join(','), 'wrist x,z', body(q), 'point', p.point.map((v) => v.toFixed(2)).join(','));
  }
});
