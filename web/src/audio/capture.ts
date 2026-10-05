/**
 * The microphone in the browser (plan 4.2, 4.7): an AudioContext, an AudioWorklet that sends
 * the samples in numbered chunks, the clip recorder, and the audio clock mapped onto the page
 * clock. The microphone runs from "Use my camera and mic" (or the first Talk press) until the
 * page closes, so the pre-roll always has the last second.
 */

import { type Clip, ClipRecorder } from './clip';
import { ClockMap } from './clock';
import workletSource from './capture.worklet.js?raw';

/** How often a new (context time, page time) pair is taken. */
const CLOCK_EVERY_MS = 2000;

/** Input latency when the browser doesn't say (Chrome does: the track's `latency`). */
const DEFAULT_INPUT_LATENCY_S = 0.01;

export class MicCapture {
  readonly clock = new ClockMap();
  readonly recorder: ClipRecorder;
  /** The last chunk's RMS level, 0..1 (for the Talk button's glow). */
  level = 0;
  /** Chunks received (checks). */
  chunks = 0;
  /** True once a pair from `getOutputTimestamp` replaced the first rough one. */
  private timed = false;
  private timer: ReturnType<typeof setInterval>;

  private constructor(
    readonly ctx: AudioContext,
    readonly stream: MediaStream,
    readonly source: MediaStreamAudioSourceNode,
    readonly node: AudioWorkletNode,
    /** Seconds from the sound reaching the microphone to the samples reaching the context. */
    readonly inputLatency: number,
  ) {
    this.recorder = new ClipRecorder(ctx.sampleRate, this.clock);
    node.port.onmessage = (e: MessageEvent<{ frame: number; data: Float32Array }>) => {
      const { frame, data } = e.data;
      this.chunks++;
      let sum = 0;
      for (const s of data) sum += s * s;
      this.level = Math.sqrt(sum / data.length);
      this.recorder.push(frame, data);
    };
    // A rough pair now, so a press in the first moments still finds its frames.
    this.clock.add({ context: ctx.currentTime, page: performance.now() / 1000 - ctx.baseLatency });
    this.timer = setInterval(() => this.takeTime(), CLOCK_EVERY_MS);
    this.takeTime();
  }

  /** Ask for the microphone and start. Throws if it is denied or missing. */
  static async start(): Promise<MicCapture> {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: false,
    });
    const ctx = new AudioContext({ latencyHint: 'interactive' });
    const url = URL.createObjectURL(new Blob([workletSource], { type: 'text/javascript' }));
    try {
      await ctx.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    const source = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, 'deixis-capture', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCountMode: 'explicit',
      channelCount: 1,
    });
    source.connect(node);
    // Its output is silence; connecting it keeps the browser pulling the node.
    node.connect(ctx.destination);
    await ctx.resume();
    const settings = stream.getAudioTracks()[0]?.getSettings() as MediaTrackSettings & { latency?: number };
    return new MicCapture(ctx, stream, source, node, settings?.latency ?? DEFAULT_INPUT_LATENCY_S);
  }

  /** Talk pressed at page time `t`. */
  begin(t = performance.now() / 1000): void {
    this.recorder.begin(t);
  }

  /** Talk released at page time `t`: the clip, once its tail has arrived. */
  end(t = performance.now() / 1000): Promise<Clip> {
    return this.recorder.end(t);
  }

  stop(): void {
    clearInterval(this.timer);
    for (const track of this.stream.getTracks()) track.stop();
    void this.ctx.close();
  }

  /**
   * One clock pair. `getOutputTimestamp` says when the sample at `contextTime` leaves the
   * speakers; that sample was rendered `outputLatency + baseLatency` earlier, and the input
   * mixed into it was captured `inputLatency` before that.
   */
  private takeTime(): void {
    const ctx = this.ctx;
    if (ctx.state !== 'running' || typeof ctx.getOutputTimestamp !== 'function') return;
    const ts = ctx.getOutputTimestamp();
    if (!ts.performanceTime || !ts.contextTime) return; // the output hasn't started yet
    const lag = (ctx.outputLatency || 0) + ctx.baseLatency + this.inputLatency;
    if (!this.timed) {
      this.clock.clear();
      this.timed = true;
    }
    this.clock.add({ context: ts.contextTime, page: ts.performanceTime / 1000 - lag });
  }
}
