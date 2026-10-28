/**
 * Records hand frames for replays (plan W4, W13): one JSON line per camera frame, landmarks
 * only (never the video), saved as a file the person downloads. The first line describes the
 * session (the active box, the canvas shape) so a replay maps the cursor the same way.
 */

import type { ActiveBox } from './cursor';
import type { HandFrame } from './tracker';

export class HandRecorder {
  private lines: string[] = [];
  private on = false;

  get recording(): boolean {
    return this.on;
  }

  get frames(): number {
    return Math.max(0, this.lines.length - 1);
  }

  start(box: ActiveBox, canvasAspect: number): void {
    this.lines = [JSON.stringify({ kind: 'session', version: 1, box, canvasAspect, started: new Date().toISOString() })];
    this.on = true;
  }

  add(f: HandFrame): void {
    if (!this.on) return;
    const round = (v: number) => Math.round(v * 1e5) / 1e5;
    this.lines.push(
      JSON.stringify({
        t: round(f.t),
        w: f.width,
        h: f.height,
        hands: f.hands.map((h) => ({
          handedness: h.handedness,
          score: round(h.score),
          image: h.image.map((p) => p.map(round)),
          world: h.world.map((p) => p.map(round)),
        })),
      }),
    );
  }

  /** Stop and download the file. Returns the number of frames saved. */
  stopAndSave(name = `hands-${new Date().toISOString().replace(/[:.]/g, '-')}.jsonl`): number {
    this.on = false;
    const n = this.frames;
    const url = URL.createObjectURL(new Blob([this.lines.join('\n') + '\n'], { type: 'application/x-ndjson' }));
    const a = Object.assign(document.createElement('a'), { href: url, download: name });
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    this.lines = [];
    return n;
  }
}
