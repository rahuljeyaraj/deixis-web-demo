/**
 * Turning the microphone's samples into the clip the server reads (plan 4.2): mono, 16 kHz,
 * 16-bit PCM WAV. The resampler is zero-phase (a symmetric windowed-sinc kernel), so a sound
 * stays at the same time in the clip: the word times depend on it.
 */

export const SPEECH_RATE = 16000;

/**
 * Resample mono `x` from `fromRate` to `toRate`. Each output sample is a windowed-sinc
 * interpolation of the input around its exact time, low-passed below the lower Nyquist rate.
 */
export function resample(x: Float32Array, fromRate: number, toRate = SPEECH_RATE, halfTaps = 24): Float32Array {
  if (fromRate === toRate) return x.slice();
  const ratio = toRate / fromRate;
  const n = Math.floor(x.length * ratio);
  const out = new Float32Array(n);
  // The cut-off, as a fraction of the input rate: 90 % of the lower Nyquist frequency.
  const fc = 0.45 * Math.min(1, ratio);
  // The kernel spans `halfTaps` periods of the cut-off on each side.
  const half = halfTaps / (2 * fc);
  const kernel = (d: number) => {
    if (Math.abs(d) >= half) return 0;
    const s = d === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * d) / (Math.PI * d);
    // Blackman window over [-half, half].
    return s * (0.42 + 0.5 * Math.cos((Math.PI * d) / half) + 0.08 * Math.cos((2 * Math.PI * d) / half));
  };
  // Polyphase: output i sits at input time i * down / up, so its fraction repeats every `up`
  // outputs (1 phase for 48 kHz, 160 for 44.1 kHz). Each phase's weights are worked out once.
  const g = gcd(fromRate, toRate);
  const up = toRate / g;
  const down = fromRate / g;
  const reach = Math.ceil(half) + 1;
  const taps = 2 * reach + 1;
  const weights = new Float64Array(up * taps);
  for (let p = 0; p < up; p++) {
    for (let k = 0; k < taps; k++) weights[p * taps + k] = kernel(k - reach - p / up);
  }
  for (let i = 0; i < n; i++) {
    const pos = i * down;
    const j = Math.floor(pos / up);
    const w0 = (pos % up) * taps;
    const k0 = Math.max(0, reach - j);
    const k1 = Math.min(taps, x.length - j + reach);
    let acc = 0;
    let norm = 0;
    for (let k = k0; k < k1; k++) {
      const w = weights[w0 + k];
      acc += x[j - reach + k] * w;
      norm += w;
    }
    // Normalise by the weights used, so a steady level stays the same near the clip's ends.
    out[i] = norm !== 0 ? acc / norm : 0;
  }
  return out;
}

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b);
}

/** A mono 16-bit PCM WAV file. Samples outside -1..1 are clipped. */
export function encodeWav(samples: Float32Array, rate = SPEECH_RATE): ArrayBuffer {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const text = (at: number, s: string) => [...s].forEach((c, i) => v.setUint8(at + i, c.charCodeAt(0)));
  text(0, 'RIFF');
  v.setUint32(4, 36 + samples.length * 2, true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  v.setUint32(16, 16, true); // fmt chunk size
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true); // bytes per second
  v.setUint16(32, 2, true); // bytes per frame
  v.setUint16(34, 16, true); // bits per sample
  text(36, 'data');
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, Math.round(s < 0 ? s * 0x8000 : s * 0x7fff), true);
  }
  return buf;
}

/** Read back a file written by `encodeWav` (tests, checks). */
export function decodeWav(buf: ArrayBuffer): { rate: number; samples: Float32Array } {
  const v = new DataView(buf);
  const rate = v.getUint32(24, true);
  const n = v.getUint32(40, true) / 2;
  const samples = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s = v.getInt16(44 + i * 2, true);
    samples[i] = s < 0 ? s / 0x8000 : s / 0x7fff;
  }
  return { rate, samples };
}

/** Peak and RMS levels in dB below full scale (-Infinity for silence). */
export function levels(x: Float32Array): { peakDb: number; rmsDb: number } {
  let peak = 0;
  let sum = 0;
  for (const s of x) {
    peak = Math.max(peak, Math.abs(s));
    sum += s * s;
  }
  const db = (a: number) => 20 * Math.log10(a);
  return { peakDb: db(peak), rmsDb: db(Math.sqrt(sum / Math.max(1, x.length))) };
}
