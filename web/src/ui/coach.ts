/**
 * The coach line (final/web-demo.md): one plain line in the scene that says what to do next,
 * or what is wrong, in this order of importance:
 *
 * 1. a problem: the server is down (item 14), no microphone, a silent clip (item 13);
 * 2. the next step: "Move the mouse onto a block." (item 2), then, once the cursor is on a
 *    block, "Hold the space bar and say “put that … here”.", then quiet; after the first
 *    move, "Now say it your own way." for a while.
 *
 * Pure state: the page calls the methods and draws `line(now)`.
 */

export interface Line {
  text: string;
}

export const COACH = {
  moveMouse: 'Move the mouse onto a block.',
  talk: 'Hold the space bar and say “put that … here”.',
  ownWay: 'Now say it your own way.',
  noMic: 'No microphone, so type instead: click a block and a spot, then type “put that here”.',
  silent: 'I didn’t hear anything. Check that your microphone is on.',
  serverDown: 'The robot’s brain isn’t answering right now. The video shows it working.',
} as const;

const SHOW_S = 10; // a passing problem or "say it your own way" stays this long

type Step = 'off' | 'move' | 'talk' | 'quiet' | 'own';

export class Coach {
  private step: Step = 'off';
  private ownUntil = 0;
  private problem: { line: Line; until: number } | null = null;
  private down = false;

  /** The page is ready: the mouse points. */
  started(now: number): void {
    void now;
    if (this.step === 'off' || this.step === 'move') this.step = 'move';
  }

  /** The cursor is on a block, or not. */
  hover(onBlock: boolean): void {
    if (onBlock && this.step === 'move') this.step = 'talk';
  }

  /** A command went to the server. */
  sent(): void {
    if (this.step === 'move' || this.step === 'talk') this.step = 'quiet';
  }

  /** The arm finished a move. The first time: "Now say it your own way." */
  moved(now: number): boolean {
    if (this.step === 'own' || this.ownUntil > 0) return false;
    this.step = 'own';
    this.ownUntil = now + SHOW_S;
    return true;
  }

  /** A passing problem, shown for a while (or until `clear`). */
  tell(text: string, now: number): void {
    this.problem = { line: { text }, until: now + SHOW_S };
  }

  clear(): void {
    this.problem = null;
  }

  /** The server answered (true) or couldn't be reached (false). */
  server(ok: boolean): void {
    this.down = !ok;
  }

  line(now: number): Line | null {
    if (this.down) return { text: COACH.serverDown };
    if (this.problem && now < this.problem.until) return this.problem.line;
    this.problem = null;
    if (this.step === 'move') return { text: COACH.moveMouse };
    if (this.step === 'talk') return { text: COACH.talk };
    if (this.step === 'own' && now < this.ownUntil) return { text: COACH.ownWay };
    return null;
  }
}

/** A phone or a small tablet: the demo needs a laptop or desktop browser (item 15). */
export function onPhone(w: Window = window): boolean {
  const coarse = w.matchMedia?.('(pointer: coarse)').matches ?? false;
  const fine = w.matchMedia?.('(any-pointer: fine)').matches ?? true;
  return (coarse && !fine) || Math.min(w.screen.width, w.screen.height) < 500;
}
