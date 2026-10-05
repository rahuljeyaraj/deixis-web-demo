// Packs the arm model into one gzip file the page downloads (plan 6.1): public/so101.pack.
//
// Format, before gzip: a 4-byte little-endian header length, a JSON header
// {"files": [{"path", "offset", "size"}]}, then the files' bytes back to back.
// Paths are where each file goes in MuJoCo's virtual file system: our table_scene.xml,
// Menagerie's so101.xml and assets/*.stl, all at the root (so101.xml's meshdir is "assets").

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

const web = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const model = path.join(web, 'model');
const out = path.join(web, 'public', 'so101.pack');

const entries = [
  ['table_scene.xml', path.join(model, 'table_scene.xml')],
  ['so101.xml', path.join(model, 'so101', 'so101.xml')],
  ...fs
    .readdirSync(path.join(model, 'so101', 'assets'))
    .filter((f) => f.endsWith('.stl') && !f.includes('camera_mount')) // the mount is removed (NOTICE.md)
    .sort()
    .map((f) => [`assets/${f}`, path.join(model, 'so101', 'assets', f)]),
];

const files = [];
const blobs = [];
let offset = 0;
for (const [p, src] of entries) {
  const bytes = fs.readFileSync(src);
  files.push({ path: p, offset, size: bytes.length });
  blobs.push(bytes);
  offset += bytes.length;
}
const header = Buffer.from(JSON.stringify({ files }), 'utf8');
const len = Buffer.alloc(4);
len.writeUInt32LE(header.length, 0);
const packed = zlib.gzipSync(Buffer.concat([len, header, ...blobs]), { level: 9 });
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, packed);
console.log(`packed ${files.length} files, ${(offset / 1e6).toFixed(1)} MB -> ${(packed.length / 1e6).toFixed(1)} MB: ${path.relative(web, out)}`);
