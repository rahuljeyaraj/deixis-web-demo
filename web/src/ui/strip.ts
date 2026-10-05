/**
 * The thinking strip (plan 5.2, "the robot's mind"): the words on a timeline with the pointing
 * words lit up, the holds (where the finger or a click rested) as bars on the same timeline,
 * and a line from each lit word to the hold the core used for it (the `Where`'s evidence
 * `word_t` and `hold_t`).
 */

import type { HeardWord } from '../api';
import type { CoreResponse } from '../core';

/** The core's pointing words (`deixis.refs.DEMONSTRATIVES`, without "so"). */
const POINTING = new Set(['here', 'there', 'this', 'that', 'these', 'those']);

export interface StripWord extends HeardWord {
  pointing: boolean;
}

export interface StripHold {
  t0: number;
  t1: number;
  source: string;
}

export interface StripModel {
  words: StripWord[];
  holds: StripHold[];
  /** A pointing word and the hold the core used for it (indexes into `words`, `holds`). */
  links: { word: number; hold: number }[];
  /** The timeline's span, seconds on the page clock. */
  t0: number;
  t1: number;
}

const norm = (w: string) => w.toLowerCase().replace(/[^a-z']/g, '');

/** Every `evidence` in the core's answer that ties a word to a hold. */
function evidenceOf(v: unknown, out: { word_t: [number, number]; hold_t: [number, number] }[] = []) {
  if (Array.isArray(v)) {
    for (const x of v) evidenceOf(x, out);
  } else if (v && typeof v === 'object') {
    const e = (v as { evidence?: Record<string, unknown> }).evidence;
    if (e && Array.isArray(e.word_t) && Array.isArray(e.hold_t)) {
      out.push({ word_t: e.word_t as [number, number], hold_t: e.hold_t as [number, number] });
    }
    for (const x of Object.values(v)) evidenceOf(x, out);
  }
  return out;
}

/**
 * The strip for one command: its words, and the core's answer (null: not understood). After
 * the person answers a question, pass the answer's result too: its links are added.
 */
export function stripModel(words: HeardWord[], ...results: (CoreResponse | null)[]): StripModel {
  const ws: StripWord[] = words.map((w) => ({ ...w, pointing: POINTING.has(norm(w.text)) }));
  const holds: StripHold[] = [];
  for (const h of results.flatMap((r) => r?.holds ?? [])) {
    if (!holds.some((k) => Math.abs(k.t0 - h.t0) < 0.02 && Math.abs(k.t1 - h.t1) < 0.02)) {
      holds.push({ t0: h.t0, t1: h.t1, source: h.source ?? 'gesture' });
    }
  }
  const links: StripModel['links'] = [];
  for (const e of evidenceOf(results)) {
    const word = nearest(ws, e.word_t[0], 0.05);
    if (word < 0) continue;
    let hold = holds.findIndex((h) => Math.abs(h.t0 - e.hold_t[0]) < 0.02 && Math.abs(h.t1 - e.hold_t[1]) < 0.02);
    if (hold < 0) {
      holds.push({ t0: e.hold_t[0], t1: e.hold_t[1], source: 'gesture' });
      hold = holds.length - 1;
    }
    ws[word].pointing = true;
    if (!links.some((l) => l.word === word && l.hold === hold)) links.push({ word, hold });
  }
  const times = [...ws.flatMap((w) => [w.t0, w.t1]), ...holds.flatMap((h) => [h.t0, h.t1])];
  const t0 = times.length ? Math.min(...times) : 0;
  const t1 = times.length ? Math.max(...times) : 1;
  const pad = Math.max(0.15, (t1 - t0) * 0.04);
  return { words: ws, holds, links, t0: t0 - pad, t1: t1 + pad };
}

function nearest(words: HeardWord[], t: number, within: number): number {
  let best = -1;
  for (let i = 0; i < words.length; i++) {
    const d = Math.abs(words[i].t0 - t);
    if (d <= within && (best < 0 || d < Math.abs(words[best].t0 - t))) best = i;
  }
  return best;
}

/**
 * Where each word goes: at its time, pushed right only as far as the word before needs; then,
 * if the last ones run past `maxX`, pulled back left as far as the word after needs.
 */
export function layoutWords(
  model: StripModel,
  x: (t: number) => number,
  width: (w: StripWord) => number,
  maxX = Infinity,
  gap = 6,
): { x: number; w: number }[] {
  let end = -Infinity;
  const spots = model.words.map((w) => {
    const ww = width(w);
    const at = Math.max(x(w.t0), end + gap);
    end = at + ww;
    return { x: at, w: ww };
  });
  let start = maxX + gap;
  for (let i = spots.length - 1; i >= 0; i--) {
    spots[i].x = Math.min(spots[i].x, start - gap - spots[i].w);
    start = spots[i].x;
  }
  return spots;
}

const SVG = 'http://www.w3.org/2000/svg';
const LABEL_W = 78;
const PAD = 14;
const WORD_Y = 30;
const HOLD_Y = 70;
const HOLD_H = 12;
export const STRIP_H = 96;

function el<K extends keyof SVGElementTagNameMap>(
  name: K,
  attrs: Record<string, string | number>,
  text?: string,
): SVGElementTagNameMap[K] {
  const e = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (text !== undefined) e.textContent = text;
  return e;
}

let measurer: CanvasRenderingContext2D | null = null;
function textWidth(text: string, font: string): number {
  measurer ??= document.createElement('canvas').getContext('2d');
  if (!measurer) return text.length * 8.5;
  measurer.font = font;
  return measurer.measureText(text).width;
}

/** Draw the strip into `svg` (its own width): a model, or a line of text (listening, ...). */
export function drawStrip(svg: SVGSVGElement, what: StripModel | string): void {
  const W = Math.max(320, svg.clientWidth || svg.parentElement?.clientWidth || 800);
  svg.replaceChildren();
  svg.setAttribute('viewBox', `0 0 ${W} ${STRIP_H}`);
  svg.setAttribute('height', String(STRIP_H));
  if (typeof what === 'string') {
    // A state (listening, thinking, a hint), not something the person said: no row labels.
    // What the robot says goes in its speech bubble, never here (rahul, 5 Oct).
    svg.append(el('text', { x: W / 2, y: STRIP_H / 2 + 5, class: 'note', 'text-anchor': 'middle' }, what));
    return;
  }
  svg.append(
    el('text', { x: 0, y: WORD_Y + 5, class: 'row-label' }, 'you said'),
    el('text', { x: 0, y: HOLD_Y + HOLD_H - 1, class: 'row-label' }, 'you pointed'),
  );
  const left = LABEL_W;
  const right = W - PAD;
  const m = what;
  const x = (t: number) => left + ((t - m.t0) / (m.t1 - m.t0)) * (right - left);

  // The time line under the holds, a tick each second.
  const axisY = HOLD_Y + HOLD_H / 2;
  svg.append(el('line', { x1: left, x2: right, y1: axisY, y2: axisY, class: 'axis' }));
  for (let t = Math.ceil(m.t0); t <= m.t1; t++) {
    svg.append(el('line', { x1: x(t), x2: x(t), y1: axisY - 4, y2: axisY + 4, class: 'tick' }));
  }
  const fontOf = (w: StripWord) => `${w.pointing ? 700 : 400} 16px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  const spots = layoutWords(m, x, (w) => textWidth(w.text, fontOf(w)) + (w.pointing ? 12 : 0), right);

  const holdX = (h: StripHold) => [x(h.t0), Math.max(x(h.t1), x(h.t0) + 6)] as const;
  for (const l of m.links) {
    const s = spots[l.word];
    const [a, b] = holdX(m.holds[l.hold]);
    svg.append(
      el('path', {
        d: `M ${s.x + s.w / 2} ${WORD_Y + 10} C ${s.x + s.w / 2} ${WORD_Y + 26}, ${(a + b) / 2} ${HOLD_Y - 16}, ${(a + b) / 2} ${HOLD_Y - 1}`,
        class: 'link',
      }),
    );
  }
  m.holds.forEach((h) => {
    const [a, b] = holdX(h);
    svg.append(el('rect', { x: a, y: HOLD_Y, width: b - a, height: HOLD_H, rx: 4, class: `hold ${h.source}` }));
  });
  m.words.forEach((w, i) => {
    const s = spots[i];
    if (w.pointing) {
      svg.append(el('rect', { x: s.x, y: WORD_Y - 15, width: s.w, height: 25, rx: 7, class: 'pill' }));
      svg.append(el('text', { x: s.x + 6, y: WORD_Y + 3, class: 'word pointing' }, w.text));
    } else {
      svg.append(el('text', { x: s.x, y: WORD_Y + 3, class: 'word' }, w.text));
    }
  });
}
