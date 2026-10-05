import { test } from 'vitest';
import { nodeScene } from '../scene';

test('probe: ways to switch a weld', async () => {
  const { mj, model } = await nodeScene();
  const tryIt = (name: string, f: () => unknown) => { try { console.log(name, '->', f()); } catch (e) { console.log(name, 'FAILS', String(e).slice(0, 120)); } };
  tryIt('eq_active0', () => Array.from(model.eq_active0 as Uint8Array));
  tryIt('eq_obj2id', () => Array.from(model.eq_obj2id as Int32Array));
  tryIt('eq_obj2id write', () => { (model.eq_obj2id as Int32Array)[0] = 13; return Array.from(model.eq_obj2id as Int32Array); });
  tryIt('opt.disableflags', () => model.opt.disableflags);
  tryIt('opt.disableflags write', () => { const o = model.opt; o.disableflags = o.disableflags | mj.mjtDisableBit.mjDSBL_EQUALITY.value; return model.opt.disableflags; });
  tryIt('eq_data view', () => (model.eq_data as Float64Array).constructor.name + ' ' + (model.eq_data as Float64Array).length);
  tryIt('geom_contype', () => (model.geom_contype as Int32Array).constructor.name);
});
