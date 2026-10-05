/**
 * Loading MuJoCo (the official WebAssembly bindings) and the packed arm model (plan 6.1).
 * Works in the browser and in Node (the tests), so nothing here touches the DOM.
 */

import loadMujoco from '@mujoco/mujoco';
import type { MainModule, MjData, MjModel } from '@mujoco/mujoco';

export type Mujoco = MainModule;
export type { MjData, MjModel };

export interface PackFile {
  path: string;
  bytes: Uint8Array;
}

let module: Promise<Mujoco> | null = null;

/** The MuJoCo module, loaded once. */
export function mujoco(): Promise<Mujoco> {
  module ??= loadMujoco();
  return module;
}

/** Split an unzipped pack (scripts/pack-model.mjs) into its files. */
export function unpack(buf: Uint8Array): PackFile[] {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const headerLen = view.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(buf.subarray(4, 4 + headerLen))) as {
    files: { path: string; offset: number; size: number }[];
  };
  const start = 4 + headerLen;
  return header.files.map((f) => ({
    path: f.path,
    bytes: buf.subarray(start + f.offset, start + f.offset + f.size),
  }));
}

/**
 * Download the pack in the browser and unzip it, reporting progress (0 to 1). The file has
 * no .gz name on purpose: dev servers would label it Content-Encoding gzip and the browser
 * would unzip it first. Either way works: it is unzipped here only if it still starts with
 * the gzip magic bytes.
 */
export async function fetchPack(
  url: string,
  onProgress?: (fraction: number) => void,
): Promise<PackFile[]> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`could not download the arm model (${res.status})`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let seen = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    seen += value.byteLength;
    if (total) onProgress?.(Math.min(1, seen / total));
  }
  let buf = new Uint8Array(seen);
  let at = 0;
  for (const c of chunks) {
    buf.set(c, at);
    at += c.byteLength;
  }
  if (buf[0] === 0x1f && buf[1] === 0x8b) {
    const gunzip = new DecompressionStream('gzip') as unknown as ReadableWritablePair<
      Uint8Array,
      Uint8Array
    >;
    const unzipped = new Blob([buf]).stream().pipeThrough(gunzip);
    buf = new Uint8Array(await new Response(unzipped).arrayBuffer());
  }
  return unpack(buf);
}

/** Compile the scene from the pack's files. */
export function loadScene(
  mj: Mujoco,
  files: PackFile[],
  scene = 'table_scene.xml',
): { model: MjModel; data: MjData } {
  const vfs = new mj.MjVFS();
  try {
    for (const f of files) vfs.addBuffer(f.path, f.bytes);
    const model = mj.MjModel.mj_loadXML(scene, vfs);
    return { model, data: new mj.MjData(model) };
  } finally {
    vfs.delete();
  }
}
