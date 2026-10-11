/**
 * The server's API (plan 7.2). One `POST /api/command` per command: the clip (16 kHz WAV) and
 * `meta` (JSON: the clip's start time, the pointer samples and clicks, the blocks, the context
 * and the page state), and the visitor's own Nebius key if they gave one (plan 8.2).
 * `POST /api/answer` when the person clicks one of the robot's choices.
 */

import type { CoreResponse, Thing } from './core';
import type { Hold, Sample } from './hand/samples';

/** A word on the page clock (seconds), as Parakeet heard it (or typed, with times made up). */
export interface HeardWord {
  text: string;
  t0: number;
  t1: number;
}

export interface AgentSummary {
  source: 'nemotron' | 'rules' | 'answer';
  model: string;
  calls: { name: string; args: Record<string, unknown> }[];
  /**
   * How the model answered the robot's open question, when it did (the `pick` tool):
   * `choice` when the person named one of the glowing blocks, `word_index` when they only
   * pointed ("this one") and the core worked out which from the finger.
   */
  pick?: { choice: number | null; word_index: number | null } | null;
  raw: unknown[];
  errors: string[];
  attempts: number;
  /** Token Factory's counts for this command (empty when no model was asked). */
  usage?: { prompt?: number; cached?: number; completion?: number };
}

/** What the simple word rules alone would have done with the same words (item 18). */
export interface WordRules {
  understood: boolean;
  /** In plain words: "move the red one to where you pointed (“here”)", "not understood". */
  words: string;
  calls: { name: string; args: Record<string, unknown> }[];
  /** The same calls as Nemotron's. */
  same: boolean;
}

export interface Timings {
  asr_s?: number;
  agent_s?: number;
  core_s?: number;
  total_s: number;
}

/** How much of our key this visitor may still use today (plan 8.1). */
export interface Limits {
  /** Who answers: our key, the visitor's own key, or the simple word matcher. */
  mode: 'ours' | 'own' | 'simple';
  /** Commands left today with our key. */
  left_today: number;
  per_day: number;
  /** Why simple mode (the visitor's share, their network's, everyone's, no key, AI off). */
  why: 'visitor' | 'address' | 'total' | 'no_key' | 'rules' | null;
  /** The simple-mode banner, from the server. */
  banner: string | null;
  /** The note on the models list's Nemotron line. */
  off: string | null;
  /** Would the visitor's own key help? (Not when the server runs the word matcher only.) */
  own_key_helps: boolean;
}

export interface CommandReply {
  words: HeardWord[];
  word_rules?: WordRules;
  /** This command's tokens at Token Factory's list price, US$. */
  cost_usd?: number;
  agent: AgentSummary;
  result: CoreResponse | null;
  error: string | null;
  /** What the robot says out loud (a question, "didn't get that"), if the server decided. */
  say: string | null;
  timings: Timings;
  limits: Limits;
  /** The visitor's own key was refused by Nebius (the page forgets it). */
  key_refused: boolean;
}

export interface AnswerReply {
  result: CoreResponse | null;
  error: string | null;
  say: string | null;
  timings: Timings;
}

export interface CommandMeta {
  /** Page-clock seconds of the clip's first sample. */
  audio_t0?: number;
  /** A typed sentence instead of a clip. */
  text?: string;
  /** The page clock when sent. */
  t?: number;
  samples?: Sample[];
  holds?: Hold[];
  things?: Thing[];
  context?: Record<string, unknown>;
  state?: { previewing: boolean; question: string | null };
}

async function check<T>(r: Response): Promise<T> {
  if (r.status >= 500) throw new ServerDown(`the server said ${r.status}`);
  if (!r.ok) {
    const detail = await r
      .json()
      .then((b: { detail?: unknown }) => String(b.detail ?? ''))
      .catch(() => '');
    throw new Error(detail || `the server said ${r.status}`);
  }
  return (await r.json()) as T;
}

const sleep = (s: number) => new Promise((done) => setTimeout(done, s * 1000));

/**
 * One command. `key`: the visitor's own Nebius key, sent in a header for this command only.
 * The server allows one command every 1.5 s (R9): when it says wait (429), this waits as long
 * as it says (`wait`, to tell the person) and sends the same command again, twice at most.
 * The words keep their times: they are on the page clock, not the time of sending.
 */
export async function sendCommand(
  meta: CommandMeta,
  wav?: ArrayBuffer,
  key?: string | null,
  wait?: (s: number) => void,
): Promise<CommandReply> {
  const form = new FormData();
  form.append('meta', JSON.stringify(meta));
  if (wav) form.append('audio', new Blob([wav], { type: 'audio/wav' }), 'clip.wav');
  const headers: Record<string, string> = key ? { 'X-Nebius-Key': key } : {};
  for (let tries = 0; ; tries++) {
    const r = await reach('/api/command', { method: 'POST', body: form, headers });
    if (r.status !== 429 || tries >= 2) return check(r);
    const body = (await r.json().catch(() => ({}))) as { wait_s?: number };
    const s = Math.min(5, Math.max(0.05, Number(body.wait_s ?? r.headers.get('Retry-After') ?? 1)));
    wait?.(s);
    await sleep(s + 0.05);
  }
}

/** One line of "What runs this demo" (plan 8.4), as the server lists it. */
export interface ModelLine {
  /** mediapipe, parakeet, nemotron, deixis, mujoco, voice */
  id: string;
  role: string;
  name: string;
  by: string;
  where: string;
  url: string;
  model_id?: string;
  /** Set when this model is off right now (simple mode). */
  off?: string | null;
}

/** `GET /api/info`: the page's name, the core, and every model the demo uses. */
export interface Info {
  title: string;
  version: string;
  core: { version: string; schema_version: string; over_mcp: boolean };
  models: ModelLine[];
  not_used: { name: string; why: string }[];
  limits: Limits;
  /** Show the "use your own Nebius key" box (hidden for the judges' demo, item 16). */
  show_own_key?: boolean;
}

/** The server can't be reached, or answers with a server error (item 14). */
export class ServerDown extends Error {}

export async function getInfo(): Promise<Info> {
  return check(await reach('/api/info'));
}

/** fetch, with a network failure as `ServerDown`. */
async function reach(url: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch (e) {
    throw new ServerDown(e instanceof Error ? e.message : String(e));
  }
}

export interface PageEvent {
  kind: string;
  via?: string;
  why?: string;
  latency_s?: number;
  block?: string;
}

/** What happened on the page (the guess shown, yes, done...), for the server's log only.
 * Never in the way: a failure is ignored. */
export function sendEvent(e: PageEvent): void {
  void fetch('/api/event', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(e),
    keepalive: true,
  }).catch(() => {});
}

export async function sendAnswer(candidate: number): Promise<AnswerReply> {
  return check(
    await reach('/api/answer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ candidate }),
    }),
  );
}
