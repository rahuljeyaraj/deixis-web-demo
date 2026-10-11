/**
 * The flow in the browser (plan 5.4): a small state machine between the server's answers and
 * the arm. idle -> thinking -> (preview | asking | acting | idle).
 *
 * One command can be a **plan** of up to four steps (advanced-commands-plan.md section 6).
 * Nemotron writes the steps; this runs them:
 *
 * 1. The whole plan is shown first. The server has measured step 1; each later step is
 *    measured against the table as the steps before it will have left it (`blocksAfter`), so
 *    a ghost block can be put at every block's final place. One "yes" covers the lot - one
 *    "yes" per step would ruin the demo (open decision D3).
 * 2. Then the steps run in order, and every step past the first is measured **again**, by the
 *    core, against the real table at that moment (`server.step`). A guess made while showing
 *    the plan is never the thing that is acted on.
 * 3. A step the arm says it cannot do goes back to Nemotron with the arm's own reason, once
 *    per command, and the new plan takes over (`server.replan`). That is recovery: the model
 *    is told what went wrong and plans around it.
 *
 * A single move is a plan of one step, and behaves exactly as it always did: the block is
 * outlined, a ghost block shows the place, and the robot says what it understood ("Move the
 * red block here?"). The core's `Where`s are still the only places used: the page never makes
 * up a coordinate, and the core never changes.
 */

import type { Ask, CoreResponse, ResolvedCall, Thing, Where } from '../core';
import { isWhere } from '../core';
import type { Arm, BlockId, BlockState, MoveResult, Place } from '../sim/arm';
import { BLOCK_IDS } from '../sim/arm';
import { onMat } from '../sim/table';
import { blockedBy, blocksAfter, sceneFor, spotLeft, under } from './things';

export type FlowState =
  | { kind: 'idle' }
  | { kind: 'thinking'; before: FlowState }
  | { kind: 'preview'; plan: Plan }
  | { kind: 'asking'; ask: Ask }
  | { kind: 'acting'; block: BlockId | null };

/** A tool call as the agent wrote it: words, never coordinates. */
export interface RawCall {
  name: string;
  args: Record<string, unknown>;
}

/** A move the core has measured: ready for the arm's own checks. */
export interface Move {
  block: BlockId;
  place: Place;
  where: Where;
  /** The block it goes on top of, when it does. */
  onto: BlockId | null;
}

export type StepState = 'todo' | 'doing' | 'done' | 'failed';

/** One step of a plan, from the model's call to what became of it. */
export interface PlanStep {
  call: RawCall;
  /** Short words for the plan strip: "red onto blue", "go home". */
  label: string;
  /** What the core measured, if anything needed measuring. */
  resolved: ResolvedCall | null;
  /** True when `resolved` came from the real table, not from a guess at it. */
  live: boolean;
  move: Move | null;
  state: StepState;
  /** Why it was refused or failed. */
  why?: string;
}

export interface Plan {
  steps: PlanStep[];
  /** This plan came from Nemotron after a step of the one before it could not be done. */
  replanned: boolean;
}

/** What the flow shows and says; the page draws it (tests record it). */
export interface FlowView {
  /** The robot says it: the speech bubble and the voice. */
  speak(text: string): void;
  /** Exactly these blocks glow (none: []). */
  glow(ids: BlockId[]): void;
  /** Exactly these ghost blocks are shown (none: []). */
  ghosts(items: Ghost[]): void;
  /** Something changed (redraw buttons, the state line). */
  changed?(state: FlowState): void;
  /** The plan strip: the steps and how each one is going; null hides it. */
  plan?(plan: Plan | null): void;
  /** Rows for the wiring panel, from the page's own parts. */
  trace?(rows: TraceRow[]): void;
  /** What happened, for the server's log (M30, M31): a guess shown, refused, yes, done... */
  event?(kind: FlowEvent, data?: EventData): void;
}

export interface Ghost {
  block: BlockId;
  place: Place;
}

export interface EventData {
  why?: string;
  block?: string;
  step?: number;
  steps?: number;
}

/** One row of the wiring panel, as `server/webdemo/trace.py` writes them. */
export interface TraceRow {
  from: string;
  to: string;
  say: string;
  detail?: unknown;
  ms?: number;
  kind?: string;
}

export type FlowEvent =
  | 'preview'
  | 'refused'
  | 'go'
  | 'done'
  | 'failed'
  | 'cancel'
  | 'stop'
  | 'undo'
  | 'plan'
  | 'step_failed'
  | 'replan'
  | 'declined';

/** The server, for the parts of a plan the page asks for as it goes. */
export interface FlowServer {
  /** Measure step `index` of the plan the last command set off, against this scene. */
  step(index: number, things: Thing[], context: Record<string, unknown>): Promise<StepReply>;
  /** Step `index` cannot be done: ask Nemotron for the rest another way. */
  replan(index: number, why: string, things: Thing[], context: Record<string, unknown>): Promise<FlowReply>;
}

export interface StepReply {
  result: CoreResponse | null;
  say?: string | null;
  error?: string | null;
}

/** The server's answer to a command (or to a re-plan). */
export interface FlowReply extends StepReply {
  plan?: { calls: RawCall[]; steps: number } | null;
}

export interface FlowOptions {
  /** Ask "This one, here?" and wait for yes before moving (open decision D3: keep). */
  confirmFirst: boolean;
  now: () => number;
  /** Without one, only the first step of a plan is used (and nothing is ever re-planned). */
  server?: FlowServer;
}

export const SAY = {
  confirm: 'Move the {block} block here?',
  confirmOnto: 'Put the {block} block on the {onto} one?',
  plan: '{n} steps, like this?',
  done: 'Done.',
  cantReach: "I can't reach there. Please move the dot closer.",
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
  declined: 'I only move the blocks on this table.',
  notABlock: 'I can only pick up the blocks. The squares are painted on.',
  cantPlan: "I couldn't work out how to do all of that.",
  thinkAgain: 'Let me think again.',
  oneAtATime: 'I can only nudge one move at a time.',
} as const;

/** How a plan's length is said: "Three steps, like this?". */
const COUNT = ['no', 'One', 'Two', 'Three', 'Four'];

interface Moved {
  block: BlockId;
  from: Place;
  to: Where;
}

export class Flow {
  state: FlowState = { kind: 'idle' };
  /** The block moved last ("it", and "a bit left" after a move). */
  lastThing: BlockId | null = null;
  /**
   * Where the last block moved away from, while nothing stands there. The core is told about
   * it as a thing called "the spot it left", which is how one block goes where another one
   * was - what a swap needs, and something no coordinate-free reference could say before.
   */
  spot: [number, number] | null = null;
  private history: Moved[] = [];
  private readonly movedAt = new Map<string, number>();
  /** Stop was pressed: a running plan gives up between steps, never waiting for the network. */
  private stopping = false;

  constructor(
    private readonly arm: Arm,
    private readonly view: FlowView,
    private readonly opts: FlowOptions = { confirmFirst: true, now: () => performance.now() / 1000 },
  ) {}

  blocks(): BlockState[] {
    return BLOCK_IDS.map((id) => this.arm.block(id));
  }

  /** The scene as the core sees it, now: the blocks, the painted squares, the empty spot. */
  things(): Thing[] {
    return sceneFor(this.blocks(), (id) => this.movedAt.get(id) ?? 0, this.spot, this.opts.now());
  }

  /** The core's context and the page state, for the next command (plan 4.2, 5.4). */
  request(): { context: Record<string, unknown>; state: { previewing: boolean; question: string | null } } {
    const s = this.settled();
    return {
      context: this.context(),
      state: {
        previewing: s.kind === 'preview',
        question: s.kind === 'asking' ? askLine(s.ask) : null, // as the robot said it
      },
    };
  }

  /** A command is on its way to the server. */
  thinking(): void {
    if (this.state.kind !== 'thinking') this.set({ kind: 'thinking', before: this.state });
  }

  /** The server's answer: a question, one call to act on, or a plan to show and then run. */
  async handle(reply: FlowReply): Promise<void> {
    const r = reply.result;
    this.stopping = false;
    if (!r) {
      this.set(this.settled()); // not understood: keep the preview or question as it was
      if (reply.say) this.view.speak(reply.say);
      return;
    }
    if (r.status === 'ask' && r.ask) {
      this.showAsk(r.ask);
      return;
    }
    this.set(this.settled());
    const calls = reply.plan?.calls ?? [];
    if (calls.length > 1 && this.opts.server) {
      await this.show(calls, r, false);
      return;
    }
    // One call: exactly as it has always been.
    for (const call of r.calls) await this.act(call);
    if (this.state.kind === 'thinking') this.set({ kind: 'idle' });
  }

  /** The candidate (by its index in the open question) a click picked, or null. */
  candidateOf(block: BlockId): number | null {
    if (this.state.kind !== 'asking') return null;
    const i = this.state.ask.candidates.findIndex((c) => c.thing === block);
    return i >= 0 ? i : null;
  }

  /** Yes, Go, Enter: run what is previewed - one move, or the whole plan. */
  async go(): Promise<MoveResult | null> {
    const s = this.state;
    if (s.kind !== 'preview') {
      this.view.speak(SAY.nothingToGo);
      return null;
    }
    const plan = s.plan;
    this.view.event?.('go', { block: firstBlock(plan) ?? undefined, steps: plan.steps.length });
    this.clearShow();
    const out = await this.runPlan(plan);
    this.set({ kind: 'idle' });
    return out;
  }

  /** No, cancel: drop the preview or the question. */
  cancel(): void {
    this.clearShow();
    this.set({ kind: 'idle' });
    this.view.speak(SAY.ok);
    this.view.event?.('cancel');
  }

  /** Stop: freeze the arm where it is and drop everything waiting, a running plan included. */
  stop(): void {
    this.stopping = true;
    this.arm.stop();
    this.clearShow();
    this.set({ kind: 'idle' });
    this.view.event?.('stop');
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
      this.spot = null;
      this.view.event?.('undo', { block: last.block });
    }
    this.set({ kind: 'idle' });
    return res;
  }

  /** New blocks: forget what was moved. */
  reset(): void {
    this.clearShow();
    this.history = [];
    this.lastThing = null;
    this.spot = null;
    this.movedAt.clear();
    this.set({ kind: 'idle' });
  }

  // ------------------------------------------------------------------ one call

  private async act(call: ResolvedCall): Promise<void> {
    const a = call.args;
    switch (call.name) {
      case 'move': {
        if (notABlock(call)) {
          this.refuse(null, SAY.notABlock);
          return;
        }
        const move = this.asMove(objectOf(call), placeOf(call));
        if (!move) return;
        const why = this.checkMove(move, this.blocks(), true);
        if (why) {
          this.refuse(move.block, why);
          return;
        }
        this.showPlan(onePlan(call, move));
        return;
      }
      case 'adjust': {
        // The same block, one step from the preview (or from where it was just put).
        if (this.state.kind === 'preview' && !this.previewed()) {
          this.view.speak(SAY.oneAtATime); // a plan of several moves is not nudged
          return;
        }
        const block = this.previewed()?.block ?? this.lastThing;
        if (!isWhere(a.place) || !block) return;
        const move = this.asMove(block, a.place);
        if (!move) return;
        const why = this.checkMove(move, this.blocks(), true);
        if (why) this.refuse(block, why);
        else this.showPlan(onePlan(call, move));
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
      case 'undo':
        await this.undo();
        return;
      case 'decline':
        // Anything that isn't about moving these blocks: one short line, no chat, no guess.
        this.clearShow();
        this.set({ kind: 'idle' });
        this.view.speak(SAY.declined);
        this.view.event?.('declined');
        return;
      case 'ask_user':
        if (typeof a.question === 'string') this.view.speak(a.question);
        return;
    }
  }

  // ------------------------------------------------------------------ showing a plan

  /**
   * Show the whole plan and wait for one "yes": step 1 as the server measured it, and each
   * later step measured against the table as the steps before it will have left it.
   */
  private async show(calls: RawCall[], first: CoreResponse, replanned: boolean): Promise<void> {
    const plan: Plan = { steps: calls.map(blankStep), replanned };
    let blocks = this.blocks();
    let spot = this.spot;
    const ghosts: Ghost[] = [];
    for (let k = 0; k < plan.steps.length; k++) {
      const step = plan.steps[k];
      if (step.call.name !== 'move') continue;
      let resolved = k === 0 ? (first.calls[0] ?? null) : null;
      if (!resolved) {
        const scene = sceneFor(blocks, (id) => this.movedAt.get(id) ?? 0, spot, this.opts.now());
        const reply = await this.opts.server!.step(k, scene, planContext(ghosts, this.lastThing));
        if (reply.result?.status === 'ask' && reply.result.ask) {
          this.showAsk(reply.result.ask);
          return;
        }
        resolved = reply.result?.status === 'ok' ? (reply.result.calls[0] ?? null) : null;
        if (!resolved) {
          this.cannotPlan(reply.say || reply.error);
          return;
        }
      }
      step.resolved = resolved;
      step.live = k === 0;
      const move = notABlock(resolved) ? null : this.asMove(objectOf(resolved), placeOf(resolved));
      if (!move) {
        this.cannotPlan(notABlock(resolved) ? SAY.notABlock : null);
        return;
      }
      // Checked against the table as this step will find it, so an impossible plan is
      // caught before anything moves - and handed back to Nemotron if it can try again.
      const why = this.checkMove(move, blocks, false);
      if (why) {
        if (await this.showAnotherWay(plan, k, why)) return;
        this.refuse(move.block, why);
        return;
      }
      step.move = move;
      step.label = stepLabel(move);
      const was = blocks.find((b) => b.id === move.block)!.tableMm;
      const after = blocksAfter(blocks, move.block, move.place);
      spot = spotLeft(was, after);
      blocks = after;
      const seen = ghosts.findIndex((g) => g.block === move.block);
      if (seen >= 0) ghosts[seen] = { block: move.block, place: move.place };
      else ghosts.push({ block: move.block, place: move.place });
    }
    this.showPlan(plan, ghosts);
  }

  /** Outline the blocks, show the ghosts, say what it understood, and wait for yes. */
  private showPlan(plan: Plan, ghosts?: Ghost[]): void {
    const places = ghosts ?? movesOf(plan).map((m) => ({ block: m.block, place: m.place }));
    this.view.glow([...new Set(places.map((g) => g.block))]);
    this.view.ghosts(places);
    const steps = plan.steps.length;
    this.view.plan?.(steps > 1 ? plan : null);
    this.set({ kind: 'preview', plan });
    this.view.event?.(steps > 1 ? 'plan' : 'preview', { block: firstBlock(plan) ?? undefined, steps });
    if (!this.opts.confirmFirst) {
      void this.go();
      return;
    }
    const one = steps === 1 ? plan.steps[0].move : null;
    this.view.speak(one ? guessLine(one.block, one.onto) : SAY.plan.replace('{n}', COUNT[steps] ?? String(steps)));
  }

  // ------------------------------------------------------------------ running a plan

  /** Run the steps in order; one re-plan if a step turns out to be impossible. */
  private async runPlan(plan: Plan): Promise<MoveResult | null> {
    let current = plan;
    let last: MoveResult | null = null;
    for (;;) {
      const run = await this.runSteps(current);
      last = run.last ?? last;
      if (run.failed === null) break;
      const reply = current.replanned ? null : await this.askAgain(run.failed, run.why);
      if (!reply) {
        this.view.speak(run.why);
        break;
      }
      current = this.planFrom(reply);
      this.view.plan?.(current);
    }
    this.view.plan?.(null);
    return last;
  }

  private async runSteps(plan: Plan): Promise<{ last: MoveResult | null; failed: number | null; why: string }> {
    let last: MoveResult | null = null;
    const steps = plan.steps.length;
    for (let k = 0; k < steps; k++) {
      const step = plan.steps[k];
      if (step.state === 'done') continue;
      if (this.stopping) return { last, failed: null, why: '' };
      this.mark(plan, k, 'doing');
      if (steps > 1) this.view.trace?.([stepRow(k, steps)]);
      // A step that carries a move is a move, whatever the model called it: "a bit left" is
      // an `adjust` whose answer from the core is a place to put the same block.
      if (step.move || step.call.name === 'move') {
        // Measured again, by the core, against the table as it is now: the guess made while
        // the plan was shown is never what the arm acts on.
        const stale = step.live ? null : await this.remeasure(step, k);
        if (stale) {
          this.mark(plan, k, 'failed', stale);
          return { last, failed: k, why: stale };
        }
        if (this.stopping) return { last, failed: null, why: '' };
        const move = step.move!;
        const why = this.checkMove(move, this.blocks(), true);
        if (why) {
          this.mark(plan, k, 'failed', why);
          this.view.event?.('refused', { why, block: move.block, step: k });
          return { last, failed: k, why };
        }
        last = await this.doMove(move);
        if (!last.ok) {
          // The arm itself: a dropped block or a stop. Nothing to re-plan around - the table
          // is not where anyone thought it was - so say what happened and stop there.
          const said = last.reason === 'stopped' ? SAY.stopped : last.reason === 'slipped' ? SAY.slipped : SAY.failed;
          this.mark(plan, k, 'failed', said);
          this.view.event?.('failed', { why: last.reason, step: k });
          this.view.speak(said);
          return { last, failed: null, why: said };
        }
      } else if (step.call.name === 'home') {
        last = await this.arm.home();
      } else if (step.call.name === 'undo') {
        last = (await this.undo()) ?? last;
      }
      this.mark(plan, k, 'done');
    }
    if (!this.stopping) this.view.speak(SAY.done);
    return { last, failed: null, why: '' };
  }

  /** Have the core measure a step again, against the real table. Null if it worked. */
  private async remeasure(step: PlanStep, k: number): Promise<string | null> {
    const reply = await this.opts.server!.step(k, this.things(), this.context());
    if (this.stopping) return null;
    const r = reply.result;
    if (r?.status !== 'ok' || !r.calls[0]) return reply.say || reply.error || SAY.cantPlan;
    const move = this.asMove(objectOf(r.calls[0]), placeOf(r.calls[0]));
    if (!move) return SAY.cantPlan;
    step.resolved = r.calls[0];
    step.live = true;
    step.move = move;
    step.label = stepLabel(move);
    return null;
  }

  /**
   * A step cannot be done: hand Nemotron the plan, the step and the arm's own reason, and
   * take a new plan. Once per command, and only for a plan of several steps - for a single
   * move, "I can't reach there. Please move the dot closer." is a better answer than anything
   * a model that may not write coordinates could offer.
   */
  private async askAgain(failed: number, why: string): Promise<FlowReply | null> {
    if (!this.opts.server || this.stopping) return null;
    this.view.event?.('step_failed', { step: failed, why });
    this.view.speak(SAY.thinkAgain);
    const reply = await this.opts.server.replan(failed, why, this.things(), this.context());
    const r = reply.result;
    if (!reply.plan?.calls.length || r?.status !== 'ok' || !r.calls.length) {
      if (reply.say) this.view.speak(reply.say);
      return null;
    }
    this.view.event?.('replan', { step: failed, steps: reply.plan.calls.length });
    return reply;
  }

  /** A plan refused while it was being shown: show the re-planned one instead. True if so. */
  private async showAnotherWay(plan: Plan, k: number, why: string): Promise<boolean> {
    if (plan.replanned || plan.steps.length < 2) return false;
    const reply = await this.askAgain(k, why);
    if (!reply) return false;
    this.view.plan?.(null);
    await this.show(reply.plan!.calls, reply.result!, true);
    return true;
  }

  private planFrom(reply: FlowReply): Plan {
    const plan: Plan = { steps: (reply.plan?.calls ?? []).map(blankStep), replanned: true };
    const first = reply.result?.calls[0] ?? null;
    const move = first && this.asMove(objectOf(first), placeOf(first));
    if (first && move) {
      plan.steps[0].resolved = first;
      plan.steps[0].live = true;
      plan.steps[0].move = move;
      plan.steps[0].label = stepLabel(move);
    }
    return plan;
  }

  private async doMove(move: Move): Promise<MoveResult> {
    const from = this.placeOf(move.block);
    this.set({ kind: 'acting', block: move.block });
    const res = await this.arm.move(move.block, move.place);
    if (res.ok) {
      this.lastThing = move.block;
      this.movedAt.set(move.block, this.opts.now());
      this.history.push({ block: move.block, from, to: move.where });
      this.spot = spotLeft([from.xMm, from.yMm], this.blocks());
      this.view.event?.('done', { block: move.block });
    }
    return res;
  }

  private mark(plan: Plan, k: number, state: StepState, why?: string): void {
    plan.steps[k].state = state;
    plan.steps[k].why = why;
    if (plan.steps.length > 1) this.view.plan?.(plan);
  }

  // ------------------------------------------------------------------ the robot-side checks

  /** The core's `Where`s as a move, or null when they are not a block and a place. */
  private asMove(block: BlockId | null, where: Where | null): Move | null {
    if (!block || !where?.mean_mm) return null;
    const onto = (where.z_mm ?? 0) > 0 ? targetOf(where) : null;
    const place: Place = { xMm: where.mean_mm[0], yMm: where.mean_mm[1], zMm: where.z_mm ?? 0 };
    return { block, place, where, onto };
  }

  /**
   * What is wrong with a move, robot side, before it is shown or done: a covered block, a
   * block on itself, a target that already has one, a place off the reach mat, a block in the
   * way, or a pose the planner cannot solve. Null when it is fine.
   *
   * `blocks` is the table it is checked against: the real one (`live`) before acting, or the
   * one the steps before it will have made while a plan is being shown. The arm's own planner
   * is only asked about the real table - there is no point solving for a guess.
   */
  private checkMove(move: Move, blocks: BlockState[], live: boolean): string | null {
    const { block, place, onto } = move;
    if (blocks.some((b) => under(b, blocks)?.id === block)) return SAY.coveredObject;
    if (onto === block) return SAY.onItself;
    if (onto && blocks.some((b) => b.id !== block && under(b, blocks)?.id === onto)) return SAY.occupiedTarget;
    // The light mat is where the arm can both pick and place (plan 4.4); off it, say so,
    // even where the arm could just stretch.
    if (!onMat(place.xMm, place.yMm)) return SAY.cantReach;
    const inTheWay = onto ? null : blockedBy(place.xMm, place.yMm, blocks, [block]);
    if (inTheWay) return SAY.inTheWay.replace('{block}', inTheWay.id);
    if (live && typeof this.arm.canMove(block, place) === 'string') {
      return place.zMm > 0 ? SAY.tooHigh : SAY.cantReach;
    }
    return null;
  }

  private refuse(block: BlockId | null, why: string): void {
    this.view.event?.('refused', { why, ...(block ? { block } : {}) });
    if (this.state.kind === 'preview') {
      // A correction that can't be done: say why, and keep the guess (item 8).
      this.view.speak(why);
      return;
    }
    this.clearShow();
    this.set({ kind: 'idle' });
    this.view.speak(why);
  }

  private cannotPlan(say: string | null | undefined): void {
    this.clearShow();
    this.set({ kind: 'idle' });
    this.view.speak(say || SAY.cantPlan);
  }

  private showAsk(ask: Ask): void {
    this.clearShow();
    this.set({ kind: 'asking', ask });
    this.view.glow(ask.candidates.map((c) => c.thing).filter((t): t is BlockId => isBlock(t)));
    this.view.speak(askLine(ask));
  }

  // ------------------------------------------------------------------ odds and ends

  /** The one move being previewed, when exactly one is. */
  private previewed(): Move | null {
    const s = this.state;
    if (s.kind !== 'preview') return null;
    const moves = movesOf(s.plan);
    return moves.length === 1 ? moves[0] : null;
  }

  private context(): Record<string, unknown> {
    const previous = this.previewed()?.where ?? this.history.at(-1)?.to ?? null;
    return {
      ...(this.lastThing ? { last_thing: this.lastThing } : {}),
      ...(previous ? { previous: { place: previous } } : {}),
    };
  }

  private placeOf(id: BlockId): Place {
    const b = this.arm.block(id);
    return { xMm: b.tableMm[0], yMm: b.tableMm[1], zMm: Math.max(0, b.heightMm - 30) };
  }

  private clearShow(): void {
    this.view.glow([]);
    this.view.ghosts([]);
    this.view.plan?.(null);
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

function blankStep(call: RawCall): PlanStep {
  return { call, label: otherLabel(call.name), resolved: null, live: false, move: null, state: 'todo' };
}

function onePlan(call: ResolvedCall, move: Move): Plan {
  const step = blankStep({ name: call.name, args: call.args });
  step.move = move;
  step.live = true;
  step.resolved = call;
  step.label = stepLabel(move);
  return { steps: [step], replanned: false };
}

function movesOf(plan: Plan): Move[] {
  return plan.steps.map((s) => s.move).filter((m): m is Move => m !== null);
}

function firstBlock(plan: Plan): BlockId | null {
  return movesOf(plan)[0]?.block ?? null;
}

/** "It" while a plan is being planned: the block the steps so far have moved. */
function planContext(moved: Ghost[], lastThing: BlockId | null): Record<string, unknown> {
  const last = moved.at(-1)?.block ?? lastThing;
  return last ? { last_thing: last } : {};
}

/** A move of something that is not a block: the paint, named as the thing to pick up. */
function notABlock(call: ResolvedCall): boolean {
  const obj = call.args.object;
  return isWhere(obj) && !!obj.thing && !isBlock(obj.thing);
}

function objectOf(call: ResolvedCall): BlockId | null {
  const obj = call.args.object;
  return isWhere(obj) && isBlock(obj.thing) ? obj.thing : null;
}

function placeOf(call: ResolvedCall): Where | null {
  const place = call.args.place;
  return isWhere(place) ? place : null;
}

function stepRow(index: number, steps: number): TraceRow {
  return { from: 'page', to: 'page', say: `step ${index + 1} of ${steps}`, kind: 'step' };
}

/** What the robot understood, said back: "Move the red block here?" (web demo item 1). */
export function guessLine(block: BlockId, onto: BlockId | null): string {
  return (onto ? SAY.confirmOnto : SAY.confirm).replace('{block}', block).replace('{onto}', onto ?? '');
}

/** A step of a plan in a few words, for the strip: "red onto blue", "blue into the red square". */
export function stepLabel(move: Move): string {
  const ev = move.where.evidence ?? {};
  const rel = typeof ev.relation === 'string' ? ev.relation : null;
  const target = plainName(ev.target);
  if (rel === 'on') return `${move.block} onto ${target}`;
  if (rel === 'inside') {
    return target === 'the spot it left' ? `${move.block} where the other one was` : `${move.block} into the ${target}`;
  }
  if (rel === 'between') return `${move.block} between ${target} and ${plainName(ev.target2)}`;
  if (rel === 'next_to') return `${move.block} beside ${target}`;
  if (rel) return `${move.block} ${rel.replace(/_/g, ' ')} ${target}`;
  return `${move.block} where you pointed`;
}

/** A thing's id as it is said: "red-square" is "red square", "left-spot" the spot it left. */
function plainName(id: unknown): string {
  const s = typeof id === 'string' ? id : '';
  return s === 'left-spot' ? 'the spot it left' : s.replace(/-/g, ' ');
}

/** The steps that are not moves, in the same few words. */
export function otherLabel(name: string): string {
  if (name === 'home') return 'go home';
  if (name === 'undo') return 'put it back';
  return name;
}

/**
 * The robot's question, naming what it is between: "The red one or the green one?" (item 1).
 * A block and a painted square both answer to "the red one", so when the two are the
 * candidates it says which is which: "The red block or the red square?". Any other question
 * comes out as the core worded it.
 */
export function askLine(ask: Ask): string {
  const names = ask.candidates.map((c) => (c.thing ? oneOf(c.label ?? c.thing) : null));
  if (ask.reason !== 'ambiguous_object' || names.length < 2 || names.some((n) => n === null)) {
    return ask.question;
  }
  const said = names as string[];
  const kinds = new Set(said.map((n) => n.split(' ').at(-1)));
  const list = kinds.size > 1 ? said : said.map((n) => n.replace(/\s\w+$/, ' one'));
  const all = list.length === 2 ? list.join(' or ') : `${list.slice(0, -1).join(', ')} or ${list.at(-1)}`;
  return all.charAt(0).toUpperCase() + all.slice(1) + '?';
}

/** "red block" -> "the red block"; a bare id -> "the red block". */
function oneOf(label: string): string {
  return /\s/.test(label) ? `the ${label}` : `the ${label} block`;
}

function isBlock(v: unknown): v is BlockId {
  return typeof v === 'string' && (BLOCK_IDS as readonly string[]).includes(v);
}

/** The block a place "on that" sits on (the core's evidence names it). */
function targetOf(where: Where): BlockId | null {
  const t = where.evidence?.target;
  return isBlock(t) ? t : null;
}
