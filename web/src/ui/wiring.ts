/**
 * The live wiring panel (advanced-commands-plan.md 5.1): what each part of the demo said to
 * the next, filling in row by row as a command runs.
 *
 * It is here because a judge could otherwise finish the whole demo without ever seeing the
 * word "Nemotron": the details panels were collapsed, and the only place the model's name
 * appeared unprompted was the banner that says it is off. So this panel is always open (never
 * folded away, rahul 11 Oct), it fills in live rather than being a log read afterwards, and it
 * is one colour per part - Nemotron red, the core green, the arm purple, the models on this
 * computer blue, you amber - so a glance says *red decided, green measured*.
 *
 * It is under the scene and the full width of the page. Beside it, seven lanes in a 460 px
 * column put the names on two crushed lines and nobody could read them.
 *
 * Plain words on the row; the real JSON on click. Nothing hidden, nothing shouted.
 */

import type { TraceRow } from '../flow/flow';

export interface Lane {
  id: string;
  /** As the panel heads the column. */
  name: string;
  /** Where it runs, on the second line under the name: short enough for one lane's width. */
  runs: string;
  /** What it does, in the column's tooltip. */
  does: string;
}

/**
 * The lanes, left to right, as `server/webdemo/trace.py` names them. Each one says where it
 * runs under its name: three of the seven are on your own computer, and that is the point.
 */
export const LANES: Lane[] = [
  { id: 'you', name: 'You', runs: 'at the table', does: 'you point, and you speak' },
  { id: 'mediapipe', name: 'MediaPipe', runs: 'your browser', does: 'where your fingertip is' },
  { id: 'parakeet', name: 'Parakeet', runs: 'our server', does: 'your words, with their times' },
  { id: 'page', name: 'The page', runs: 'your browser', does: 'the robot side: it runs the plan' },
  { id: 'nemotron', name: 'Nemotron', runs: 'Nebius', does: 'Nebius Token Factory: what to do' },
  { id: 'core', name: 'deixis core', runs: 'over MCP', does: 'it measures where, and answers' },
  { id: 'arm', name: 'The arm', runs: 'MuJoCo', does: 'your browser: it does the moving' },
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

/**
 * The panel. `add` appends rows as they happen; `clear` starts a new command; `replay` steps
 * through the rows it has at about one a second, for the film and for a judge who wants a
 * second look without saying it again.
 */
export class Wiring {
  private readonly track: HTMLElement;
  private readonly list: HTMLElement;
  private readonly foot: HTMLElement | null;
  private rows: TraceRow[] = [];
  private timer = 0;

  constructor(
    private readonly root: HTMLElement,
    private readonly replayButton?: HTMLButtonElement,
  ) {
    this.track = root.querySelector('.wiring-lanes')!;
    this.list = root.querySelector('.wiring-rows')!;
    this.foot = root.querySelector('.wiring-foot');
    this.drawLaneNames();
    this.replayButton?.addEventListener('click', () => this.replay());
  }

  /** A new command: start the rows again. */
  clear(): void {
    window.clearInterval(this.timer);
    this.rows = [];
    this.list.replaceChildren();
    this.root.classList.remove('replaying');
    if (this.foot) this.foot.textContent = '';
  }

  /** Who answered, how long it took and what it cost: one line under the rows. */
  summary(line: string | null): void {
    if (this.foot) this.foot.textContent = line ?? '';
  }

  add(rows: TraceRow[] | undefined): void {
    if (!rows?.length) return;
    window.clearInterval(this.timer);
    this.root.classList.remove('replaying');
    for (const row of rows) {
      this.rows.push(row);
      this.list.append(this.draw(row));
    }
    // No scrollIntoView: the list does not scroll inside itself any more, and pulling the
    // page down to it would take the scene away from someone watching the arm.
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
        el.append(lane.name);
        const runs = document.createElement('small');
        runs.textContent = lane.runs;
        el.append(runs);
        el.title = `${lane.name} - ${lane.does}`;
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
    for (const [i, lane] of LANES.entries()) {
      const line = document.createElement('i');
      line.className = 'wiring-rail';
      line.dataset.lane = lane.id; // each rail a pale shade of its own part's colour
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

/**
 * The arrow from one lane to another, in the sending lane's colour: a dot where it starts, a
 * line, a head where it lands.
 *
 * Plain elements placed by percentage, not an SVG: the panel is as wide as the page, and an
 * SVG stretched to that width turns a round dot into a wide pill and the head into a wedge.
 * These keep their size whatever the panel's width.
 */
function arrow(s: RowShape): HTMLElement {
  const a = lanePercent(s.from);
  const b = lanePercent(s.to);
  const wrap = document.createElement('span');
  wrap.className = 'wiring-arrow';
  const line = document.createElement('i');
  line.className = 'wiring-line';
  line.style.left = `${Math.min(a, b)}%`;
  line.style.width = `${Math.abs(b - a)}%`;
  const tail = document.createElement('i');
  tail.className = 'wiring-dot';
  tail.style.left = `${a}%`;
  const head = document.createElement('i');
  head.className = 'wiring-tip';
  head.dataset.dir = s.rightwards ? 'right' : 'left';
  head.style.left = `${b}%`;
  wrap.append(line, tail, head);
  return wrap;
}

/** A row a part sent to itself (a note it made): a square on that lane. */
function blip(s: RowShape): HTMLElement {
  const wrap = document.createElement('span');
  wrap.className = 'wiring-arrow';
  const self = document.createElement('i');
  self.className = 'wiring-self';
  self.style.left = `${lanePercent(s.from)}%`;
  wrap.append(self);
  return wrap;
}
