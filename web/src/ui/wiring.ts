/**
 * The live wiring panel (advanced-commands-plan.md 5.1): what each part of the demo said to
 * the next, filling in row by row as a command runs.
 *
 * It is here because a judge could otherwise finish the whole demo without ever seeing the
 * word "Nemotron": both details panels are collapsed, and the only place the model's name
 * appeared unprompted was the banner that says it is off. So this panel is always open, it
 * fills in live rather than being a log read afterwards, and it is one colour per part -
 * Nemotron red, the core green, the arm purple, the models on this computer blue, you amber -
 * so a glance says *red decided, green measured*.
 *
 * Plain words on the row; the real JSON on click. Nothing hidden, nothing shouted.
 *
 * The grey shadow row - what the simple word matcher alone made of the same words - is the
 * strongest single piece of evidence on the page, and used to be two clicks deep.
 */

import type { TraceRow } from '../flow/flow';

export interface Lane {
  id: string;
  /** As the panel heads the column. */
  name: string;
  /** Where it runs, for the title text. */
  where: string;
}

/** The lanes, left to right, as `server/webdemo/trace.py` names them. */
export const LANES: Lane[] = [
  { id: 'you', name: 'You', where: 'at the table' },
  { id: 'mediapipe', name: 'MediaPipe', where: 'your browser: where your fingertip is' },
  { id: 'parakeet', name: 'Parakeet', where: 'our server: your words, with their times' },
  { id: 'page', name: 'The page', where: 'your browser: the robot side' },
  { id: 'nemotron', name: 'Nemotron', where: 'Nebius Token Factory: what to do' },
  { id: 'core', name: 'deixis core', where: 'over MCP: where' },
  { id: 'arm', name: 'The arm', where: 'your browser: MuJoCo' },
];

const INDEX = new Map(LANES.map((l, i) => [l.id, i]));

export function laneOf(id: string): number {
  return INDEX.get(id) ?? INDEX.size - 1;
}

/** One row, ready to draw: which lanes, which way, and the words. */
export interface RowShape {
  /** The lane the arrow starts at, and the one it ends at (left to right on screen). */
  from: number;
  to: number;
  /** Leftmost and rightmost lane the arrow covers. */
  left: number;
  right: number;
  /** The arrow points right (`from` is left of `to`). */
  rightwards: boolean;
  /** The arrow is only within one lane (something that part did by itself). */
  single: boolean;
  say: string;
  /** "0.61 s", "12 ms", or empty when the row cost nothing worth saying. */
  time: string;
  kind: string;
  /** The row has JSON behind it. */
  opens: boolean;
}

export function rowShape(row: TraceRow): RowShape {
  const from = laneOf(row.from);
  const to = laneOf(row.to);
  return {
    from,
    to,
    left: Math.min(from, to),
    right: Math.max(from, to),
    rightwards: to >= from,
    single: from === to,
    say: row.say,
    time: timeLabel(row.ms),
    kind: row.kind ?? '',
    opens: row.detail !== undefined && row.detail !== null,
  };
}

/** How long it took, on the rows that cost something: the model's call and each arm move. */
export function timeLabel(ms: number | undefined): string {
  if (ms === undefined || ms < 1) return '';
  return ms >= 100 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

/** Where a lane's line sits across the track, as a percentage. */
export function lanePercent(i: number): number {
  return ((i + 0.5) / LANES.length) * 100;
}

const SVG = 'http://www.w3.org/2000/svg';

/**
 * The panel. `add` appends rows as they happen; `clear` starts a new command; `replay` steps
 * through the rows it has at about one a second, for the film and for a judge who wants a
 * second look without saying it again.
 */
export class Wiring {
  private readonly track: HTMLElement;
  private readonly list: HTMLElement;
  private rows: TraceRow[] = [];
  private timer = 0;

  constructor(
    private readonly root: HTMLElement,
    private readonly replayButton?: HTMLButtonElement,
  ) {
    this.track = root.querySelector('.wiring-lanes')!;
    this.list = root.querySelector('.wiring-rows')!;
    this.drawLaneNames();
    this.replayButton?.addEventListener('click', () => this.replay());
  }

  /** A new command: start the rows again. */
  clear(): void {
    window.clearInterval(this.timer);
    this.rows = [];
    this.list.replaceChildren();
    this.root.classList.remove('replaying');
  }

  add(rows: TraceRow[] | undefined): void {
    if (!rows?.length) return;
    window.clearInterval(this.timer);
    this.root.classList.remove('replaying');
    for (const row of rows) {
      this.rows.push(row);
      this.list.append(this.draw(row));
    }
    this.list.lastElementChild?.scrollIntoView({ block: 'nearest' });
  }

  /** Step through the rows again, about one a second. */
  replay(): void {
    window.clearInterval(this.timer);
    const rows = [...this.rows];
    if (!rows.length) return;
    this.list.replaceChildren();
    this.root.classList.add('replaying');
    let i = 0;
    const tick = () => {
      if (i >= rows.length) {
        window.clearInterval(this.timer);
        this.root.classList.remove('replaying');
        return;
      }
      this.list.append(this.draw(rows[i++]));
    };
    tick();
    this.timer = window.setInterval(tick, 850);
  }

  private drawLaneNames(): void {
    this.track.replaceChildren(
      ...LANES.map((lane, i) => {
        const el = document.createElement('span');
        el.className = 'wiring-lane';
        el.dataset.lane = lane.id;
        el.style.left = `${lanePercent(i)}%`;
        el.textContent = lane.name;
        el.title = `${lane.name} - ${lane.where}`;
        return el;
      }),
    );
  }

  private draw(row: TraceRow): HTMLElement {
    const s = rowShape(row);
    const li = document.createElement('li');
    li.className = 'wiring-row';
    if (s.kind) li.dataset.kind = s.kind;
    li.dataset.from = row.from;

    const track = document.createElement('span');
    track.className = 'wiring-track';
    for (let i = 0; i < LANES.length; i++) {
      const line = document.createElement('i');
      line.className = 'wiring-rail';
      line.style.left = `${lanePercent(i)}%`;
      track.append(line);
    }
    if (!s.single && s.kind !== 'step') track.append(arrow(s));
    else if (s.kind !== 'step') track.append(blip(s));

    const say = document.createElement('span');
    say.className = 'wiring-say';
    say.textContent = s.say;

    li.append(track, say);
    if (s.time) {
      const time = document.createElement('span');
      time.className = 'wiring-time';
      time.textContent = s.time;
      li.append(time);
    }
    if (s.opens) {
      li.classList.add('opens');
      const open = document.createElement('button');
      open.type = 'button';
      open.className = 'wiring-open';
      open.textContent = 'JSON';
      const pre = document.createElement('pre');
      pre.hidden = true;
      pre.textContent = JSON.stringify(row.detail, null, 2);
      open.addEventListener('click', () => {
        pre.hidden = !pre.hidden;
        open.textContent = pre.hidden ? 'JSON' : 'hide';
      });
      li.append(open, pre);
    }
    return li;
  }
}

/** The arrow from one lane to another, in the sending lane's colour. */
function arrow(s: RowShape): SVGSVGElement {
  const a = lanePercent(s.from);
  const b = lanePercent(s.to);
  const svg = document.createElementNS(SVG, 'svg');
  svg.setAttribute('class', 'wiring-arrow');
  svg.setAttribute('viewBox', '0 0 100 10');
  svg.setAttribute('preserveAspectRatio', 'none');
  const line = document.createElementNS(SVG, 'line');
  line.setAttribute('x1', String(a));
  line.setAttribute('x2', String(b));
  line.setAttribute('y1', '5');
  line.setAttribute('y2', '5');
  svg.append(line);
  // The head: a small triangle at the receiving end, drawn in screen units so it keeps its
  // shape however wide the panel is.
  const head = document.createElementNS(SVG, 'polygon');
  const dir = s.rightwards ? -1 : 1;
  head.setAttribute('points', `${b},5 ${b + dir * 1.6},2.2 ${b + dir * 1.6},7.8`);
  svg.append(head);
  return svg;
}

/** A row a part sent to itself (the word matcher, a note): a dot on that lane. */
function blip(s: RowShape): SVGSVGElement {
  const svg = document.createElementNS(SVG, 'svg');
  svg.setAttribute('class', 'wiring-arrow');
  svg.setAttribute('viewBox', '0 0 100 10');
  svg.setAttribute('preserveAspectRatio', 'none');
  const dot = document.createElementNS(SVG, 'rect');
  dot.setAttribute('x', String(lanePercent(s.from) - 1.2));
  dot.setAttribute('y', '3.4');
  dot.setAttribute('width', '2.4');
  dot.setAttribute('height', '3.2');
  dot.setAttribute('rx', '1');
  svg.append(dot);
  return svg;
}
