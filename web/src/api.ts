/**
 * The server's API (plan 7.2). One `POST /api/command` per command: the clip (16 kHz WAV) and
 * `meta` (JSON: `audio_t0` and, from W8 on, the samples, the blocks and the page state).
 */

/** A word on the page clock (seconds), as Parakeet heard it. */
export interface HeardWord {
  text: string;
  t0: number;
  t1: number;
}

export interface CommandReply {
  words: HeardWord[];
  timings: { asr_s?: number; total_s: number };
}

export interface CommandMeta {
  /** Page-clock seconds of the clip's first sample. */
  audio_t0?: number;
  /** A typed sentence instead of a clip. */
  text?: string;
}

export async function sendCommand(meta: CommandMeta, wav?: ArrayBuffer): Promise<CommandReply> {
  const form = new FormData();
  form.append('meta', JSON.stringify(meta));
  if (wav) form.append('audio', new Blob([wav], { type: 'audio/wav' }), 'clip.wav');
  const r = await fetch('/api/command', { method: 'POST', body: form });
  if (!r.ok) {
    const detail = await r
      .json()
      .then((b: { detail?: unknown }) => String(b.detail ?? ''))
      .catch(() => '');
    throw new Error(detail || `the server said ${r.status}`);
  }
  return (await r.json()) as CommandReply;
}
