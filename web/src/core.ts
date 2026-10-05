/**
 * The deixis core's messages as the page reads them (the core's JSON Schemas
 * `schema/where-1.json`, `ask-1.json`, `response-1.json`; only the fields the page uses).
 */

/** A place or a thing, in the table frame (mm). */
export interface Where {
  version?: string;
  frame: string;
  kind: string;
  mean_mm?: [number, number] | null;
  box_mm?: [number, number, number, number] | null;
  /** Height above the table it rests at ("on that"). */
  z_mm?: number | null;
  thing?: string | null;
  label?: string | null;
  source: string;
  evidence?: Record<string, unknown>;
}

export interface Ask {
  reason: string;
  question: string;
  call?: number | null;
  arg?: string | null;
  candidates: Where[];
  options?: string[];
}

export interface ResolvedCall {
  index: number;
  name: string;
  args: Record<string, unknown>;
}

export interface CoreResponse {
  status: 'ok' | 'ask';
  calls: ResolvedCall[];
  ask?: Ask | null;
  partial?: unknown[];
  holds?: { t0: number; t1: number; mean: [number, number]; source?: string }[];
}

/** A block as the core sees it (plan 6.2, the core's `Thing`). */
export interface Thing {
  id: string;
  label: string;
  attrs: Record<string, string>;
  box_mm: [number, number, number, number];
  height_mm: number;
  t: number;
  selectable: boolean;
}

export function isWhere(v: unknown): v is Where {
  return !!v && typeof v === 'object' && 'frame' in v && 'kind' in v;
}
