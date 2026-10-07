import { test } from 'vitest';
import { Arm } from '../../src/sim/arm';
import { BASE_MM } from '../../src/sim/table';
import { nodeScene } from '../scene';

test('probe: pick and place reach maps', async () => {
  const { mj, model, data } = await nodeScene();
  const arm = new Arm(mj, model, data);
  arm.goHomeNow();
  arm.putBlock('green', 40, 230); arm.putBlock('blue', 470, 20);
  const maps: Record<string, string[]> = { place: [], pick: [] };
  const radii: Record<string, number[][]> = { place: [], pick: [] };
  for (let y = 240; y >= 0; y -= 10) {
    let lp = `${String(y).padStart(3)} `, lk = lp;
    for (let x = 0; x <= 500; x += 10) {
      const r = Math.hypot(x - BASE_MM.x, y - BASE_MM.y);
      arm.putBlock('red', 250, 70);
      const okPlace = typeof arm.canMove('red', { xMm: x, yMm: y, zMm: 0 }) !== 'string';
      arm.putBlock('red', x, y, 0.3);
      const okPick = typeof arm.canMove('red', { xMm: 250, yMm: 70, zMm: 0 }) !== 'string';
      lp += okPlace ? '#' : '.'; lk += okPick ? '#' : '.';
      radii.place.push([r, okPlace ? 1 : 0]); radii.pick.push([r, okPick ? 1 : 0]);
    }
    maps.place.push(lp); maps.pick.push(lk);
  }
  for (const k of ['place', 'pick']) {
    console.log(k + '\n' + maps[k].join('\n'));
    for (let r0 = 60; r0 < 360; r0 += 10) {
      const sel = radii[k].filter(([r]) => r >= r0 && r < r0 + 10);
      if (sel.length) console.log(k, 'r', r0, 'ok', (sel.reduce((s, [, o]) => s + o, 0) / sel.length).toFixed(2));
    }
  }
});
