import { test } from 'vitest';
import { ArmKinematics } from '../../src/sim/ik';
import { nodeScene } from '../scene';

test('probe: jaw geometry in the site frame', async () => {
  const { mj, model, data } = await nodeScene();
  const ik = new ArmKinematics(mj, model);
  const r = ik.solve({ pos: [0.2, 0, 0.05], yaw: 0 });
  const G = mj.mjtObj.mjOBJ_GEOM.value;
  const site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE.value, 'gripperframe');
  const gid = (n: string) => mj.mj_name2id(model, G, n);
  const names = ['fixed_jaw_sph_tip1', 'fixed_jaw_box6', 'moving_jaw_sph_tip1', 'moving_jaw_box3'];
  const gq = (model.jnt_qposadr as Int32Array)[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT.value, 'gripper')];
  for (const g of [-0.17, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]) {
    for (let j = 0; j < 5; j++) data.qpos[ik.qposAdr[j]] = r.q[j];
    data.qpos[gq] = g;
    mj.mj_kinematics(model, data);
    const sp = data.site_xpos as Float64Array, sR = data.site_xmat as Float64Array;
    const local = (n: string) => {
      const i = gid(n); const p = data.geom_xpos as Float64Array;
      const d = [p[3*i]-sp[3*site], p[3*i+1]-sp[3*site+1], p[3*i+2]-sp[3*site+2]];
      // site frame coordinates: R^T d (R row-major, columns are the axes)
      return [0,1,2].map((c) => (sR[9*site+c]*d[0] + sR[9*site+3+c]*d[1] + sR[9*site+6+c]*d[2]) * 1000).map((v) => v.toFixed(1)).join(',');
    };
    console.log('gripper', g, names.map((n) => `${n}=[${local(n)}]`).join('  '));
  }
});
