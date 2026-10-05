/**
 * The flow in the browser (plan 5.4): a small state machine between the server's answers and
 * the arm. idle -> thinking -> (preview | asking | acting | idle).
 *
 * A resolved move is checked robot-side first (the block can be picked, the place is free,
 * the arm can reach the block and the place); then the block glows, a ghost block shows the
 * place, and the robot asks "This one, here?". On yes (or Go) the arm runs the pick and place.
 * The core's `Where`s are the only places used: the page never makes up a coordinate.
 */

import type { Ask, CoreResponse, ResolvedCall, Thing, Where } from '../core';
import { isWhere } from '../core';
import type { Arm, BlockId, BlockState, MoveResult, Place } from '../sim/arm';
import { BLOCK_IDS } from '../sim/arm';
import { onMat } from '../sim/table';
import { blockedBy, thingsFor, under } from './things';

export type FlowState =
  | { kind: 'idle' }
  | { kind: 'thinking'; before: FlowState }
  | { kind: 'preview'; block: BlockId; place: Place; where: Where; onto: BlockId | null }
  | { kind: 'asking'; ask: Ask }
  | { kind: 'acting'; block: BlockId | null };

/** What the flow shows and says; the page draws it (tests record it). */
export interface FlowView {
  /** The robot says it: the speech bubble and the voice. */
  speak(text: string): void;
  /** Exactly these blocks glow (none: []). */
  glow(ids: BlockId[]): void;
  /** The ghost of `block` at `place`, or no ghost. */
  ghost(block: BlockId | null, place?: Place): void;
  /** Something changed (redraw buttons, the state line). */
  changed?(state: FlowState): void;
}

export interface FlowOptions {
  /** Ask "This one, here?" and wait for yes before moving (open decision D3: keep). */
  confirmFirst: boolean;
  now: () => number;
}

export const SAY = {
  confirm: 'This one, here?',
  cantReach: "I can't reach there. Please point somewhere closer.",
  inTheWay: 'The {block} block is in the way there.',
  tooHigh: "That stack is too high for me there. I can stack higher a bit further from me.",
  coveredObject: "There's a block on top of that one. Move that one first.",
  occupiedTarget: "There's already a block on that one.",
  onItself: "I can't put a block on itself.",
  nothingToGo: 'There is nothing to do yet. Point at a block and say “put that here”.',
  nothingToUndo: 'There is nothing to undo.',
  stopped: 'Stopped.',
  ok: 'OK.',
  slipped: 'Oops, I dropped it.',
  failed: "That didn't work.",
} as const;

interface Moved {
  block: BlockId;
  from: Place;
  to: Where;
}

export class Flow {
  state: FlowState = { kind: 'idle' };
  /** The block moved last ("it", and "a bit left" after a move). */
  lastThing: BlockId | null = null;
  private history: Moved[] = [];
  private readonly movedAt = new Map<string, number>();

  constructor(
    private readonly arm: Arm,
    private readonly view: FlowView,
    private readonly opts: FlowOptions = { confirmFirst: true, now: () => performance.now() / 1000 },
  ) {}

  blocks(): BlockState[] {
    return BLOCK_IDS.map((id) => this.arm.block(id));
  }

  /** The blocks as the core sees them, now. */
  things(): Thing[] {
    return thingsFor(this.blocks(), (id) => this.movedAt.get(id) ?? 0);
  }

  /** The core's context and the page state, for the next command (plan 4.2, 5.4). */
  request(): { context: Record<string, unknown>; state: { previewing: boolean; question: string | null } } {
    const s = this.settled();
    const previous = s.kind === 'preview' ? s.where : (this.history.at(-1)?.to ?? null);
    return {
      context: {
        ...(this.lastThing ? { last_thing: this.lastThing } : {}),
        ...(previous ? { previous: { place: previous } } : {}),
      },
      state: {
        previewing: s.kind === 'preview',
        question: s.kind === 'asking' ? s.ask.question : null,
      },
    };
  }

  /** A command is on its way to the server. */
  thinking(): void {
    if (this.state.kind !== 'thinking') this.set({ kind: 'thinking', before: this.state });
  }

  /** The server's answer: a question, or calls to act on in order. */
  async handle(reply: { result: CoreResponse | null; say?: string | null; error?: string | null }): Promise<void> {
    const r = reply.result;
    if (!r) {
      this.set(this.settled()); // not understood: keep the preview or question as it was
      if (reply.say) this.view.speak(reply.say);
      return;
    }
    if (r.status === 'ask' && r.ask) {
      this.clearShow();
      this.set({ kind: 'asking', ask: r.ask });
      const ids = r.ask.candidates.map((c) => c.thing).filter((t): t is BlockId => isBlock(t));
      this.view.glow(ids);
      this.view.speak(r.ask.question);
      return;
    }
    this.set(this.settled());
    for (const call of r.calls) await this.act(call);
    if (this.state.kind === 'thinking') this.set({ kind: 'idle' });
  }

  /** The candidate (by its index in the open question) a click picked, or null. */
  candidateOf(block: BlockId): number | null {
    if (this.state.kind !== 'asking') return null;
    const i = this.state.ask.candidates.findIndex((c) => c.thing === block);
    return i >= 0 ? i : null;
  }

  /** Yes, Go, Enter: run the previewed move. */
  async go(): Promise<MoveResult | null> {
    const s = this.state;
    if (s.kind !== 'preview') {
      this.view.speak(SAY.nothingToGo);
      return null;
    }
    const from = this.placeOf(s.block);
    this.set({ kind: 'acting', block: s.block });
    const res = await this.arm.move(s.block, s.place);
    this.clearShow();
    if (res.ok) {
      this.lastThing = s.block;
      this.movedAt.set(s.block, this.opts.now());
      this.history.push({ block: s.block, from, to: s.where });
    } else {
      this.view.speak(res.reason === 'stopped' ? SAY.stopped : res.reason === 'slipped' ? SAY.slipped : SAY.failed);
    }
    this.set({ kind: 'idle' });
    return res;
  }

  /** No, cancel: drop the preview or the question. */
  cancel(): void {
    this.clearShow();
    this.set({ kind: 'idle' });
    this.view.speak(SAY.ok);
  }

  /** Stop: freeze the arm where it is and drop everything waiting. */
  stop(): void {
    this.arm.stop();
    this.clearShow();
    this.set({ kind: 'idle' });
  }

  async home(): Promise<MoveResult> {
    this.clearShow();
    this.set({ kind: 'acting', block: null });
    const res = await this.arm.home();
    this.set({ kind: 'idle' });
    return res;
  }

  /** Put the last moved block back where it was (straight away, no preview). */
  async undo(): Promise<MoveResult | null> {
    const last = this.history.at(-1);
    if (!last || this.arm.busy) {
      if (!last) this.view.speak(SAY.nothingToUndo);
      return null;
    }
    this.clearShow();
    this.set({ kind: 'acting', block: last.block });
    const res = await this.arm.move(last.block, last.from);
    if (res.ok) {
      this.history.pop();
      this.movedAt.set(last.block, this.opts.now());
    }
    this.set({ kind: 'idle' });
    return res;
  }

  /** New blocks: forget what was moved. */
  reset(): void {
    this.clearShow();
    this.history = [];
    this.lastThing = null;
    this.movedAt.clear();
    this.set({ kind: 'idle' });
  }

  // ------------------------------------------------------------------ calls

  private async act(call: ResolvedCall): Promise<void> {
    const a = call.args;
    switch (call.name) {
      case 'move': {
        const obj = a.object;
        const place = a.place;
        if (!isWhere(obj) || !isWhere(place) || !isBlock(obj.thing)) return;
        this.preview(obj.thing, place);
        return;
      }
      case 'adjust': {
        // The same block, one step from the preview (or from where it was just put).
        const s = this.state;
        const block = s.kind === 'preview' ? s.block : this.lastThing;
        if (isWhere(a.place) && block) this.preview(block, a.place);
        return;
      }
      case 'confirm':
        await this.go();
        return;
      case 'cancel':
        this.cancel();
        return;
      case 'stop':
        this.stop();
        this.view.speak(SAY.stopped);
        return;
      case 'home':
        await this.home();
        return;
      case 'ask_user':
        if (typeof a.question === 'string') this.view.speak(a.question);
        return;
    }
  }

  /** The robot-side checks, then the preview (or the move itself, without the yes). */
  private preview(block: BlockId, where: Where): void {
    const mean = where.mean_mm;
    if (!mean) return;
    const blocks = this.blocks();
    const onto = (where.z_mm ?? 0) > 0 ? targetOf(where) : null;
    const place: Place = { xMm: mean[0], yMm: mean[1], zMm: where.z_mm ?? 0 };
    const why = (() => {
      if (blocks.some((b) => under(b, blocks)?.id === block)) return SAY.coveredObject;
      if (onto === block) return SAY.onItself;
      if (onto && blocks.some((b) => b.id !== block && under(b, blocks)?.id === onto)) return SAY.occupiedTarget;
      // The light mat is where the arm can both pick and place (plan 4.4); off it, say so,
      // even where the arm could just stretch.
      if (!onMat(place.xMm, place.yMm)) return SAY.cantReach;
      const inTheWay = onto ? null : blockedBy(place.xMm, place.yMm, blocks, [block]);
      if (inTheWay) return SAY.inTheWay.replace('{block}', inTheWay.id);
      if (typeof this.arm.canMove(block, place) === 'string') return place.zMm > 0 ? SAY.tooHigh : SAY.cantReach;
      return null;
    })();
    if (why) {
      this.clearShow();
      this.set({ kind: 'idle' });
      this.view.speak(why);
      return;
    }
    this.view.glow([block]);
    this.view.ghost(block, place);
    this.set({ kind: 'preview', block, place, where, onto });
    if (this.opts.confirmFirst) this.view.speak(SAY.confirm);
    else void this.go();
  }

  private placeOf(id: BlockId): Place {
    const b = this.arm.block(id);
    return { xMm: b.tableMm[0], yMm: b.tableMm[1], zMm: Math.max(0, b.heightMm - 30) };
  }

  private clearShow(): void {
    this.view.glow([]);
    this.view.ghost(null);
  }

  /** The state underneath "thinking". */
  private settled(): FlowState {
    return this.state.kind === 'thinking' ? this.state.before : this.state;
  }

  private set(s: FlowState): void {
    this.state = s;
    this.view.changed?.(s);
  }
}

function isBlock(v: unknown): v is BlockId {
  return typeof v === 'string' && (BLOCK_IDS as readonly string[]).includes(v);
}

/** The block a place "on that" sits on (the core's evidence names it). */
function targetOf(where: Where): BlockId | null {
  const t = where.evidence?.target;
  return isBlock(t) ? t : null;
}
