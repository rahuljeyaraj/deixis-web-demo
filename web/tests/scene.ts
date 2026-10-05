/** Load the packed scene in Node for the tests (the same pack the page downloads). */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { loadScene, mujoco, unpack } from '../src/sim/mujoco';

const PACK = path.join(import.meta.dirname, '..', 'public', 'so101.pack');

export async function nodeScene() {
  if (!fs.existsSync(PACK)) throw new Error('run `node scripts/pack-model.mjs` first');
  const mj = await mujoco();
  const files = unpack(new Uint8Array(zlib.gunzipSync(fs.readFileSync(PACK))));
  return { mj, ...loadScene(mj, files) };
}
