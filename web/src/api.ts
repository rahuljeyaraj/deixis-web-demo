/**
 * The server's API (plan 7.2). One `POST /api/command` per command: the clip (16 kHz WAV) and
 * `meta` (JSON: the clip's start time, the pointer samples and clicks, the blocks, the context
 * and the page state). `POST /api/answer` when the person clicks one of the robot's choices.
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
  raw: unknown[];
  errors: string[];
  attempts: number;
}

export interface Timings {
  asr_s?: number;
  agent_s?: number;
  core_s?: number;
  total_s: number;
}

export interface CommandReply {
  words: HeardWord[];
  agent: AgentSummary;
  result: CoreResponse | null;
  error: string | null;
  /** What the robot says out loud (a question, "didn't get that"), if the server decided. */
  say: string | null;
  timings: Timings;
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
  if (!r.ok) {
    const detail = await r
      .json()
      .then((b: { detail?: unknown }) => String(b.detail ?? ''))
      .catch(() => '');
    throw new Error(detail || `the server said ${r.status}`);
  }
  return (await r.json()) as T;
}

export async function sendCommand(meta: CommandMeta, wav?: ArrayBuffer): Promise<CommandReply> {
  const form = new FormData();
  form.append('meta', JSON.stringify(meta));
  if (wav) form.append('audio', new Blob([wav], { type: 'audio/wav' }), 'clip.wav');
  return check(await fetch('/api/command', { method: 'POST', body: form }));
}

/** One line of "What runs this demo" (plan 8.4), as the server lists it. */
export interface ModelLine {
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
}

export async function getInfo(): Promise<Info> {
  return check(await fetch('/api/info'));
}

export async function sendAnswer(candidate: number): Promise<AnswerReply> {
  return check(
    await fetch('/api/answer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ candidate }),
    }),
  );
}
