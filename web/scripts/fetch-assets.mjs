// Puts the hand tracker's files where the page loads them from (plan 4.1, W4), so the page
// needs no CDN:
//   public/mediapipe/  MediaPipe's WebAssembly runtime, copied from node_modules
//   public/models/hand_landmarker.task  the Hand Landmarker model (Apache-2.0), downloaded once
// Both are gitignored; `npm run dev` and `npm run build` run this first.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const web = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const pub = path.join(web, 'public');

const wasmSrc = path.join(web, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const wasmDst = path.join(pub, 'mediapipe');
fs.mkdirSync(wasmDst, { recursive: true });
for (const f of fs.readdirSync(wasmSrc)) {
  if (!/^vision_wasm(_nosimd)?_internal\.(js|wasm)$/.test(f)) continue;
  const src = path.join(wasmSrc, f);
  const dst = path.join(wasmDst, f);
  if (!fs.existsSync(dst) || fs.statSync(dst).size !== fs.statSync(src).size) fs.copyFileSync(src, dst);
}

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task';
const model = path.join(pub, 'models', 'hand_landmarker.task');
if (!fs.existsSync(model)) {
  fs.mkdirSync(path.dirname(model), { recursive: true });
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`could not download the hand model (${res.status})`);
  fs.writeFileSync(model, Buffer.from(await res.arrayBuffer()));
  console.log(`hand model: ${(fs.statSync(model).size / 1e6).toFixed(1)} MB -> public/models/`);
}
