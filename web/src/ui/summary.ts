/**
 * The one line under the wiring panel: who understood the words, how long each part took, and
 * what this command cost on Nebius Token Factory (item 11).
 *
 * This was a collapsed "Details" box until 11 Oct, with the tool calls, the core's answer and
 * the timings in it. The wiring panel says all of that, in order, with the real JSON a button
 * away, so the box went and only its three facts that are nowhere else stayed.
 */

import type { AgentSummary, Limits, Timings } from '../api';

/** Who understood the words, in plain words; `mode`: whose key (plan 8.1, 8.2). */
export function understoodBy(agent: AgentSummary | undefined, mode?: Limits['mode']): string {
  if (!agent) return 'Your answer to the robot’s question (no language model needed).';
  if (agent.source === 'nemotron') {
    const whose = mode === 'own' ? ', with your own key' : '';
    return `NVIDIA Nemotron 3.5 Lightning (${agent.model}), on Nebius Token Factory${whose}.`;
  }
  return 'Your answer to the robot’s question (no language model needed).';
}

/** Each step's time next to what did it: "Parakeet 0.20 s · Nemotron 1.10 s · core 0.010 s". */
export function timingsLine(t: Timings, agent?: AgentSummary): string {
  const parts: string[] = [];
  if (t.asr_s !== undefined) parts.push(`Parakeet ${t.asr_s.toFixed(2)} s`);
  if (t.agent_s !== undefined) {
    parts.push(`${agent?.source === 'answer' ? 'answer' : 'Nemotron'} ${t.agent_s.toFixed(2)} s`);
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

export interface SummaryInput {
  agent?: AgentSummary;
  limits?: Limits;
  timings: Timings;
  cost_usd?: number;
}

/** The three of them as one line, for under the panel's rows. */
export function summaryLine(d: SummaryInput): string {
  const who = understoodBy(d.agent, d.limits?.mode).replace(/\.$/, '');
  return [`Understood by ${who}`, timingsLine(d.timings, d.agent), costLine(d.agent, d.cost_usd)]
    .filter(Boolean)
    .join(' — ');
}
