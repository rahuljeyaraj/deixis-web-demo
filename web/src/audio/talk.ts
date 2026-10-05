/**
 * Push-to-talk (plan 5.2): the Talk button and the space bar. Hold to talk, let go to stop.
 * A short press (a tap) keeps listening until the next tap instead: that is how touch screens
 * work, and it frees the mouse to point while talking (mouse mode, plan 5.3). Listening stops
 * by itself before the clip gets too long for the server.
 */

export type StopReason = 'release' | 'tap' | 'long' | 'cancel';

export interface TalkHandlers {
  start(t: number): void;
  stop(t: number, reason: StopReason): void;
}

export class PushToTalk {
  listening = false;
  /** Listening after a tap: the next press stops it. */
  latched = false;
  /** The button (or the space bar) is down. */
  private held = false;
  private pressT = 0;
  private startT = 0;
  /** The held press already ended its listen: its release does nothing. */
  private spent = false;

  constructor(
    private readonly on: TalkHandlers,
    /** A press shorter than this is a tap. */
    readonly tapS = 0.35,
    /** The longest listen: 12 s of clip minus the 0.5 s pre-roll and the 0.4 s tail. */
    readonly maxS = 11,
  ) {}

  press(t: number): void {
    if (this.held) return; // a second key or button while one is down
    this.held = true;
    this.pressT = t;
    if (this.listening) {
      // Only a latched listen is still on at a press: this press ends it.
      this.spent = true;
      this.finish(t, 'tap');
      return;
    }
    this.spent = false;
    this.listening = true;
    this.latched = false;
    this.startT = t;
    this.on.start(t);
  }

  release(t: number): void {
    if (!this.held) return;
    this.held = false;
    if (this.spent || !this.listening) return;
    if (t - this.pressT < this.tapS) this.latched = true;
    else this.finish(t, 'release');
  }

  /** Call often (each frame): stops a listen that has gone on too long. */
  tick(t: number): void {
    if (this.listening && t - this.startT >= this.maxS) this.end(t, 'long');
  }

  /** Stop listening now (Esc, Stop); the handlers decide to drop the clip. */
  cancel(t: number): void {
    if (this.listening) this.end(t, 'cancel');
  }

  private end(t: number, reason: StopReason): void {
    this.spent = this.held;
    this.finish(t, reason);
  }

  private finish(t: number, reason: StopReason): void {
    this.listening = false;
    this.latched = false;
    this.on.stop(t, reason);
  }
}
