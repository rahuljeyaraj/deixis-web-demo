/**
 * The details panel (plan 5.2, collapsed under the strip): which model understood the words,
 * the tool calls it wrote, the core's answer (the `Where`s or the `Ask`, the same messages the
 * real plotter and the real vacuum act on), and how long each step took.
 */

import type { AgentSummary, HeardWord, Limits, Timings, WordRules } from '../api';
import type { CoreResponse } from '../core';

export interface DetailsInput {
  words?: HeardWord[];
  typed?: boolean;
  agent?: AgentSummary;
  limits?: Limits;
  result: CoreResponse | null;
  error?: string | null;
  timings: Timings;
  word_rules?: WordRules;
  cost_usd?: number;
}

/** Who understood the words, in plain words; `mode`: whose key (plan 8.1, 8.2). */
export function understoodBy(agent: AgentSummary | undefined, mode?: Limits['mode']): string {
  if (!agent) return 'Your answer to the robot’s question (no language model needed).';
  if (agent.source === 'nemotron') {
    const whose = mode === 'own' ? ', with your own key' : '';
    return `NVIDIA Nemotron 3.5 Lightning (${agent.model}), on Nebius Token Factory${whose}.`;
  }
  if (agent.source === 'answer') return 'Your answer to the robot’s question (no language model needed).';
  const why = agent.errors.length ? ` Nemotron didn’t answer (${agent.errors[0]}).` : '';
  return `The simple word matcher (no AI model).${why}`;
}

/** One tool call as a line: `move(object={"deixis":"that","word_index":1}, ...)`. */
export function callLine(c: { name: string; args: Record<string, unknown> }): string {
  const args = Object.entries(c.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`);
  return `${c.name}(${args.join(', ')})`;
}

/** The core's answer without its empty fields and its working notes (`log`). */
export function coreJson(result: CoreResponse): string {
  const { log: _log, holds: _holds, ...rest } = result as CoreResponse & { log?: unknown };
  const drop = (v: unknown): unknown => {
    if (Array.isArray(v)) return v.map(drop);
    if (v && typeof v === 'object') {
      return Object.fromEntries(
        Object.entries(v)
          .filter(([, x]) => x !== null && !(Array.isArray(x) && x.length === 0))
          .map(([k, x]) => [k, drop(x)]),
      );
    }
    return v;
  };
  return JSON.stringify(drop(rest), null, 2);
}

/** Each step's time next to what did it: "Parakeet 0.20 s · Nemotron 1.10 s · core 0.010 s". */
export function timingsLine(t: Timings, agent?: AgentSummary): string {
  const parts: string[] = [];
  if (t.asr_s !== undefined) parts.push(`Parakeet ${t.asr_s.toFixed(2)} s`);
  if (t.agent_s !== undefined) {
    const who = agent?.source === 'nemotron' ? 'Nemotron' : agent?.source === 'answer' ? 'answer' : 'word matcher';
    parts.push(`${who} ${t.agent_s.toFixed(2)} s`);
  }
  if (t.core_s !== undefined) parts.push(`core ${t.core_s.toFixed(3)} s`);
  parts.push(`server in all ${t.total_s.toFixed(2)} s`);
  return parts.join(' · ');
}

/** "7,051 tokens in (4,192 cached), 64 out: $0.00044 at list price" (item 11). */
export function costLine(agent: AgentSummary | undefined, cost: number | undefined): string | null {
  const u = agent?.usage;
  if (!u || !u.prompt) return null;
  const n = (x: number | undefined) => (x ?? 0).toLocaleString('en-GB');
  const cached = u.cached ? ` (${n(u.cached)} cached)` : '';
  const usd = cost === undefined ? '' : `: $${cost < 0.001 ? cost.toFixed(5) : cost.toFixed(4)} at list price`;
  return `${n(u.prompt)} tokens in${cached}, ${n(u.completion)} out${usd}`;
}

function section(title: string, body: HTMLElement | string): HTMLElement {
  const s = document.createElement('section');
  const h = document.createElement('h3');
  h.textContent = title;
  s.append(h);
  if (typeof body === 'string') {
    const p = document.createElement('p');
    p.textContent = body;
    s.append(p);
  } else {
    s.append(body);
  }
  return s;
}

function pre(text: string): HTMLElement {
  const p = document.createElement('pre');
  p.textContent = text;
  return p;
}

/** Fill the panel's body for one command (or a clicked answer). */
export function fillDetails(body: HTMLElement, d: DetailsInput): void {
  const parts: HTMLElement[] = [];
  if (d.words) {
    const said = d.words.map((w) => w.text).join(' ') || '(no words)';
    parts.push(section(d.typed ? 'You typed' : 'Heard (NVIDIA Parakeet, with the time of each word)', said));
  }
  parts.push(section('Understood by', understoodBy(d.agent, d.limits?.mode)));
  if (d.agent && d.agent.calls.length) {
    parts.push(section('The tool calls it wrote (words, never coordinates)', pre(d.agent.calls.map(callLine).join('\n'))));
  }
  if (d.word_rules && d.agent?.source === 'nemotron') {
    const wr = d.word_rules;
    const box = document.createElement('div');
    const p = document.createElement('p');
    p.className = 'rules-line';
    p.textContent = wr.understood
      ? `understood: ${wr.words}${wr.same ? ' (the same as Nemotron)' : ''}`
      : 'not understood';
    box.append(p);
    if (wr.calls.length) {
      const small = pre(wr.calls.map(callLine).join('\n'));
      small.classList.add('small-pre');
      box.append(small);
    }
    parts.push(section('Simple word rules (no AI model), on the same words', box));
  }
  if (d.result) {
    const title =
      d.result.status === 'ask'
        ? 'The deixis core’s answer, over MCP: a question (Ask)'
        : 'The deixis core’s answer, over MCP: the places (Where), as the real plotter and vacuum get them';
    parts.push(section(title, pre(coreJson(d.result))));
  } else if (d.error) {
    parts.push(section('The deixis core', `It refused the request: ${d.error}`));
  }
  parts.push(section('Time', timingsLine(d.timings, d.agent)));
  const cost = costLine(d.agent, d.cost_usd);
  if (cost) parts.push(section('Tokens and cost on Nebius Token Factory', cost));
  body.replaceChildren(...parts);
}
