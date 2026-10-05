/**
 * The page. For now: the arm, the table and the blocks, and a mouse test of the arm (click a
 * block, then click a spot). Speech, the finger and the deixis core come next (plan W4 on).
 */

import { Arm, BLOCK_IDS, type BlockId } from './sim/arm';
import { fetchPack, loadScene, mujoco } from './sim/mujoco';
import { type Hit, pointAt } from './sim/pick';
import { TableView } from './sim/render';
import { BASE_MM, BLOCK_MM, REACH_MM, onMat, tableToWorld } from './sim/table';

/** Random spots on the reach mat, at least 60 mm apart. */
function layout(n: number, rand = Math.random): [number, number][] {
  const out: [number, number][] = [];
  while (out.length < n) {
    const r = REACH_MM.min + 15 + rand() * (REACH_MM.max - REACH_MM.min - 30);
    const a = (rand() - 0.5) * Math.PI * 0.9;
    const x = BASE_MM.x + r * Math.sin(a);
    const y = BASE_MM.y - r * Math.cos(a);
    if (onMat(x, y) && out.every(([u, v]) => Math.hypot(u - x, v - y) > 60)) out.push([x, y]);
  }
  return out;
}

async function start(): Promise<void> {
  const status = document.getElementById('status')!;
  const canvas = document.getElementById('scene') as HTMLCanvasElement;
  const say = (text: string) => (status.textContent = text);
  const [mj, files] = await Promise.all([
    mujoco(),
    fetchPack('/so101.pack', (f) => say(`Loading the arm… ${Math.round(f * 100)} %`)),
  ]);
  const { model, data } = loadScene(mj, files);
  const arm = new Arm(mj, model, data);
  const view = new TableView(canvas, mj, model);
  const blockOfGeom = new Map<number, BlockId>([...arm.blocks.values()].map((b) => [b.geom, b.id]));

  const reset = () => {
    arm.goHomeNow();
    layout(BLOCK_IDS.length).forEach(([x, y], i) =>
      arm.putBlock(BLOCK_IDS[i], x, y, Math.random() * Math.PI),
    );
  };
  reset();

  const fit = () => view.resize(canvas.clientWidth, canvas.clientHeight);
  new ResizeObserver(fit).observe(canvas);
  fit();

  // Mouse test: click a block, then a spot (or another block to stack on it).
  let chosen: BlockId | null = null;
  const choose = (id: BlockId | null) => {
    if (chosen) view.glow(arm.blocks.get(chosen)!.geom, false);
    chosen = id;
    if (id) view.glow(arm.blocks.get(id)!.geom, true);
  };
  const hitOf = (e: MouseEvent): Hit | null => {
    const r = canvas.getBoundingClientRect();
    return pointAt(view, (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height, blockOfGeom);
  };
  canvas.addEventListener('click', async (e) => {
    if (arm.busy) return;
    const hit = hitOf(e);
    if (!hit) return;
    if (!chosen) {
      if (hit.block) {
        choose(hit.block);
        say(`The ${hit.block} block. Now click where it should go.`);
      }
      return;
    }
    const id = chosen;
    const onto = hit.block && hit.block !== id ? arm.block(hit.block) : null;
    const place = onto
      ? { xMm: onto.tableMm[0], yMm: onto.tableMm[1], zMm: onto.heightMm }
      : { xMm: hit.x, yMm: hit.y, zMm: 0 };
    const [gx, gy] = tableToWorld(place.xMm, place.yMm);
    view.showGhost([gx, gy, (place.zMm + BLOCK_MM / 2) / 1000], arm.blocks.get(id)!.geom);
    say(`Moving the ${id} block…`);
    const res = await arm.move(id, place);
    choose(null);
    view.showGhost(null);
    say(
      res.ok
        ? 'Done. Click another block.'
        : res.reason === 'cant_reach'
          ? "I can't reach there. Please pick a spot on the light area."
          : `That didn't work (${res.reason}).`,
    );
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') arm.stop();
    if (e.key === 'r' && !arm.busy) reset();
  });

  // Physics in real time: 5 ms steps, as many as the frame took (at most 0.1 s' worth).
  const dt = model.opt.timestep;
  let last = performance.now();
  let owed = 0;
  const loop = () => {
    // performance.now(), not the frame's timestamp: some browsers (headless Chrome) give
    // frame times on another origin, which would stall the physics.
    const now = performance.now();
    owed += Math.max(0, Math.min((now - last) / 1000, 0.1));
    last = now;
    for (; owed >= dt; owed -= dt) {
      arm.update();
      mj.mj_step(model, data);
    }
    view.sync(data);
    view.render();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
  say('Click a block, then click where it should go. R: new blocks. Esc: stop.');
  Object.assign(window, { __sim: { mj, model, data, view, arm } });
}

start().catch((e) => {
  document.getElementById('status')!.textContent =
    `Could not start: ${e instanceof Error ? e.message : e}`;
  console.error(e);
});
