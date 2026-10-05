/** The page: for now the W2 spike (the arm, the table and the blocks at the seated view). */

import { fetchPack, loadScene, mujoco } from './sim/mujoco';
import { TableView } from './sim/render';

const HOME = [0, -1.57, 1.45, 0.75, 0, 0.2]; // arm up and folded, gripper nearly closed

async function start(): Promise<void> {
  const status = document.getElementById('status')!;
  const canvas = document.getElementById('scene') as HTMLCanvasElement;
  const [mj, files] = await Promise.all([
    mujoco(),
    fetchPack('/so101.pack', (f) => (status.textContent = `Loading the arm… ${Math.round(f * 100)} %`)),
  ]);
  const { model, data } = loadScene(mj, files);
  for (let i = 0; i < 6; i++) {
    data.qpos[i] = HOME[i];
    data.ctrl[i] = HOME[i];
  }
  mj.mj_forward(model, data);

  const view = new TableView(canvas, mj, model);
  const fit = () => view.resize(canvas.clientWidth, canvas.clientHeight);
  new ResizeObserver(fit).observe(canvas);
  fit();

  const dt = model.opt.timestep;
  let last = performance.now();
  let frames = 0;
  let fpsT = last;
  const loop = (now: number) => {
    let budget = Math.min((now - last) / 1000, 0.1);
    last = now;
    while (budget >= dt) {
      mj.mj_step(model, data);
      budget -= dt;
    }
    view.sync(data);
    view.render();
    frames++;
    if (now - fpsT > 1000) {
      status.textContent = `${frames} fps`;
      frames = 0;
      fpsT = now;
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
  Object.assign(window, { __sim: { mj, model, data, view } });
}

start().catch((e) => {
  document.getElementById('status')!.textContent = `Could not start: ${e instanceof Error ? e.message : e}`;
  console.error(e);
});
