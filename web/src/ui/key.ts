/**
 * The visitor's own Nebius key (plan 8.2) and the day's limits on ours (8.1): the footer's
 * "Use your own Nebius key" panel, the banner when Nemotron is off, and the models list's note.
 *
 * The key lives only in this module's memory: not in `localStorage`, not in the page (the box
 * is emptied on Save), gone when the tab closes. It goes to our server with each command, in
 * the `X-Nebius-Key` header, and the server uses it for that command's Nemotron call only.
 */

import type { CommandReply, Limits } from '../api';
import { setModelOff } from './models';

let key: string | null = null;

/** The visitor's own key, if they gave one. */
export const ownKey = (): string | null => key;

/** Why this text can't be a key (else null). Keys are long and have no spaces. */
export function keyProblem(text: string): string | null {
  const k = text.trim();
  if (!k) return 'Paste a key first.';
  if (!/^[\x21-\x7e]+$/.test(k)) return 'That doesn’t look like a key: it has spaces or unusual characters.';
  if (k.length < 20) return 'That key looks too short.';
  if (k.length > 4096) return 'That key is too long.';
  return null;
}

/** What the page shows for the day's limits and the visitor's key. */
export interface LimitsShown {
  banner: string | null;
  /** "Add your own Nebius key" after the banner. */
  offerKey: boolean;
  /** The note on the models list's Nemotron line. */
  off: string | null;
  /** The panel's line about our key. */
  left: string;
}

export function limitsShown(l: Limits | null, hasKey: boolean): LimitsShown {
  if (hasKey) return { banner: null, offerKey: false, off: null, left: 'Your own key answers your commands now.' };
  if (!l) return { banner: null, offerKey: false, off: null, left: '' };
  const left =
    l.mode === 'off'
      ? 'Our key can’t answer you right now. With your own, the robot understands you again.'
      : `Our key: ${l.left_today} of ${l.per_day} commands left for you today.`;
  return { banner: l.banner, offerKey: l.own_key_helps, off: l.off, left };
}

/** The panel, the banner and the models line, kept in step. */
export class KeyPanel {
  private limits: Limits | null = null;
  /** Offer "add your own key" (off for the judges' demo, item 16). */
  allowOwnKey = true;
  private readonly panel = document.getElementById('own-key') as HTMLDetailsElement;
  private readonly box = document.getElementById('key-box') as HTMLInputElement;
  private readonly status = document.getElementById('key-status')!;
  private readonly leftLine = document.getElementById('key-left')!;
  private readonly banner = document.getElementById('banner')!;
  private readonly bannerText = document.getElementById('banner-text')!;
  private readonly bannerKey = document.getElementById('banner-key') as HTMLButtonElement;

  constructor(private readonly models: HTMLElement) {
    document.getElementById('key-save')!.addEventListener('click', () => this.save());
    document.getElementById('key-forget')!.addEventListener('click', () => this.forget('Your key is forgotten.'));
    this.box.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter') return;
      e.preventDefault();
      this.save();
    });
    this.bannerKey.addEventListener('click', () => this.open());
  }

  /** Open the panel and put the cursor in the box. */
  open(): void {
    this.panel.open = true;
    this.panel.scrollIntoView({ behavior: 'smooth', block: 'end' });
    this.box.focus({ preventScroll: true });
  }

  private save(): void {
    const problem = keyProblem(this.box.value);
    if (problem) return this.tell(problem, true);
    key = this.box.value.trim();
    this.box.value = ''; // the page doesn't keep it; only this module's memory does
    this.tell('Saved for this tab. Your next command uses it.');
    this.render();
  }

  private forget(message: string, bad = false): void {
    key = null;
    this.box.value = '';
    this.tell(message, bad);
    this.render();
  }

  private tell(text: string, bad = false): void {
    this.status.textContent = text;
    this.status.classList.toggle('bad', bad);
  }

  /** The limits from `/api/info`, at the start. */
  show(l: Limits): void {
    this.limits = l;
    this.render();
  }

  /** After each command: the limits now, and whether the visitor's key worked. Returns a line
   * for the page's status, if something about the key needs saying. */
  reply(r: Pick<CommandReply, 'limits' | 'key_refused' | 'agent'>): string | null {
    this.limits = r.limits;
    if (r.key_refused) {
      this.forget('That key didn’t work, so I forgot it. Check it and paste it again.', true);
      return r.limits.mode === 'ours'
        ? 'Your key didn’t work, so our key answered this time.'
        : 'Your key didn’t work, and our key can’t answer you right now.';
    }
    if (key && r.limits.mode === 'own' && r.agent.source === 'nemotron') this.tell('Your key works.');
    this.render();
    return null;
  }

  private render(): void {
    const s = limitsShown(this.limits, key !== null);
    this.panel.classList.toggle('has-key', key !== null);
    this.banner.hidden = !s.banner;
    this.bannerText.textContent = s.banner ?? '';
    this.bannerKey.hidden = !(s.offerKey && this.allowOwnKey);
    this.leftLine.textContent = s.left;
    setModelOff(this.models, 'nemotron', s.off);
  }
}
