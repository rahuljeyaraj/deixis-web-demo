// Runs on the audio thread (an AudioWorklet; plain JavaScript, loaded from a Blob URL so no
// bundler code is injected). It sends the microphone's samples to the page in chunks of 2048
// frames (about 43 ms at 48 kHz), mixed to mono, each with the frame number of its first
// sample on the AudioContext's clock (plan 4.7). Frames with no input are sent as silence, so
// the numbering never skips.

class DeixisCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.size = 2048;
    this.buf = new Float32Array(this.size);
    this.n = 0;
    this.first = 0;
  }

  process(inputs) {
    const chans = inputs[0] || [];
    const len = chans[0] ? chans[0].length : 128;
    if (this.n === 0) this.first = currentFrame;
    for (let i = 0; i < len; i++) {
      let s = 0;
      for (const ch of chans) s += ch[i];
      this.buf[this.n++] = chans.length ? s / chans.length : 0;
      if (this.n === this.size) {
        this.port.postMessage({ frame: this.first, data: this.buf }, [this.buf.buffer]);
        this.buf = new Float32Array(this.size);
        this.n = 0;
        this.first = currentFrame + i + 1;
      }
    }
    return true;
  }
}

registerProcessor('deixis-capture', DeixisCapture);
