"""Speech to timed words on the server (plan 4.2 step 3, 7.3): NVIDIA Parakeet TDT 0.6B v2
(int8, sherpa-onnx, CPU).

    speech = Speech(settings)                    # loads the model and decodes one silent clip
    words = speech.transcribe(wav_bytes, t0)     # list[Word], on the browser's clock

`t0` is the browser-clock time of the clip's first sample (`audio_t0`), so the words land on
the same clock as the fingertip samples; the server never uses its own clock for fusion.

`tokens_to_words`, `LocalAsr`'s offline path, the measured lag and the ONNX Runtime DLL fix are
copied from deixis-doodle (`src/pointdraw/audio/asr_local.py`, `models.py`, `ort.py`, commit
7a7e9cc).
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from deixis.types import Word

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_S = 0.08  # encoder frame (8x subsampling of 10 ms features) for these NeMo models
# Median lag of word start times against ground truth, measured in deixis-doodle's spike R2
# (synthetic voices, 300 ms pre-roll; docs/spikes/r2-asr.md there).
MEASURED_LAG_S = 0.095
MAX_CLIP_S = 12.0  # the browser cuts there too (plan 7.2)

SHERPA_ASR_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"


@dataclass(frozen=True)
class AsrModel:
    """One sherpa-onnx transducer export."""

    key: str
    archive: str  # release asset name without .tar.bz2
    licence: str
    source: str  # upstream model card

    @property
    def url(self) -> str:
        return f"{SHERPA_ASR_RELEASE}/{self.archive}.tar.bz2"

    def files(self, models_dir: Path) -> dict[str, Path]:
        d = Path(models_dir) / self.archive
        return {
            "encoder": d / "encoder.int8.onnx",
            "decoder": d / "decoder.int8.onnx",
            "joiner": d / "joiner.int8.onnx",
            "tokens": d / "tokens.txt",
        }

    def available(self, models_dir: Path) -> bool:
        return all(p.exists() for p in self.files(models_dir).values())


PARAKEET = AsrModel(
    key="parakeet_v2",
    archive="sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
    licence="CC-BY-4.0",
    source="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2",
)


class ClipError(ValueError):
    """The uploaded clip can't be used (not a WAV we read, or too long)."""


def import_sherpa() -> ModuleType:
    """Import `sherpa_onnx`, preferring the pip onnxruntime DLL on Windows.

    Windows 11 ships an old `System32\\onnxruntime.dll`; without help sherpa-onnx binds to it
    and fails with "The requested API version [28] is not available".
    """
    if sys.platform == "win32" and "sherpa_onnx" not in sys.modules:
        import onnxruntime

        capi = os.path.join(os.path.dirname(onnxruntime.__file__), "capi")
        if os.path.isdir(capi):
            os.add_dll_directory(capi)
    import sherpa_onnx

    return sherpa_onnx


def tokens_to_words(
    tokens: list[str],
    timestamps: list[float],
    durations: list[float] | None = None,
    t_end: float | None = None,
) -> list[tuple[str, float, float]]:
    """Group sub-word tokens into (word, t0, t1) with times relative to the audio start.

    sherpa-onnx marks a word start with a leading space (SentencePiece's "▁"). Some models emit
    a bare " " token before the word's first piece. Punctuation-only tokens are dropped.
    """
    words: list[list] = []  # [text, t0, t_last, dur_last]
    new_word = True
    for k, (tok, ts) in enumerate(zip(tokens, timestamps, strict=False)):
        tok = tok.replace("▁", " ")
        dur = durations[k] if durations and k < len(durations) else FRAME_S
        if tok.startswith(" "):
            new_word = True
        piece = tok.strip()
        if not piece:
            continue
        if not any(c.isalnum() for c in piece):
            new_word = True  # punctuation ends a word
            continue
        if new_word or not words:
            words.append([piece, ts, ts, dur])
            new_word = False
        else:
            words[-1][0] += piece
            words[-1][2] = ts
            words[-1][3] = dur
    out: list[tuple[str, float, float]] = []
    for i, (text, t0, t_last, dur) in enumerate(words):
        t1 = t_last + max(dur, FRAME_S)
        if i + 1 < len(words):
            t1 = min(t1, words[i + 1][1])
        elif t_end is not None:
            t1 = min(t1, t_end)
        out.append((text.strip("'"), float(t0), float(max(t1, t0))))
    return [w for w in out if w[0]]


def read_clip(data: bytes, max_s: float = MAX_CLIP_S) -> np.ndarray:
    """A mono 16-bit PCM WAV at 16 kHz (what the page sends) as float32 samples."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            n = w.getnframes()
            if rate != SAMPLE_RATE or channels != 1 or width != 2:
                got = f"{channels} channel(s), {8 * width}-bit, {rate} Hz"
                raise ClipError(f"expected mono 16-bit 16 kHz, got {got}")
            if n > max_s * rate + 1:
                raise ClipError(f"the clip is {n / rate:.1f} s; at most {max_s:.0f} s")
            pcm = np.frombuffer(w.readframes(n), dtype="<i2")
    except (wave.Error, EOFError) as e:
        raise ClipError(f"not a WAV file ({e})") from e
    return pcm.astype(np.float32) / 32768.0


class LocalAsr:
    """One loaded sherpa-onnx Parakeet recognizer (offline: whole utterances)."""

    def __init__(self, models_dir: Path, num_threads: int = 2, lag_s: float = MEASURED_LAG_S):
        self.model = PARAKEET
        if not self.model.available(models_dir):
            raise FileNotFoundError(
                f"{self.model.archive} not found in {models_dir}; run "
                "`uv run python scripts/fetch_models.py` or set ASR_MODELS_DIR"
            )
        self.lag_s = lag_s
        so = import_sherpa()
        files = {k: str(v) for k, v in self.model.files(models_dir).items()}
        t = time.perf_counter()
        self._recognizer = so.OfflineRecognizer.from_transducer(
            **files, model_type="nemo_transducer", num_threads=num_threads,
            decoding_method="greedy_search",
        )  # fmt: skip
        self.load_s = time.perf_counter() - t

    def transcribe(self, x: np.ndarray, t_start: float = 0.0) -> list[Word]:
        """Recognise one utterance (16 kHz). Word times are `t_start` + offset into `x`."""
        dur = len(x) / SAMPLE_RATE
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, x)
        self._recognizer.decode_stream(stream)
        r = stream.result
        durations = list(getattr(r, "durations", []) or [])
        rel = tokens_to_words(list(r.tokens), list(r.timestamps), durations, dur)
        out = []
        for text, t0, t1 in rel:
            a = min(max(t0 - self.lag_s, 0.0), dur)
            b = min(max(t1 - self.lag_s, a), dur)
            out.append(Word(text=text, t0=t_start + a, t1=t_start + b))
        return out


class Speech:
    """Parakeet for the whole process (plan 7.3): loaded once at start and warmed up with one
    silent clip (the first decode is slow). One recognizer serves `workers` decodes at the same
    time (ONNX Runtime runs are thread-safe, and sherpa-onnx lets go of the GIL while it
    decodes; a test checks parallel decodes match); further requests wait their turn."""

    def __init__(self, models_dir: Path, workers: int = 2, threads: int = 2) -> None:
        t = time.perf_counter()
        self._asr = LocalAsr(models_dir, threads)
        self._slots = threading.BoundedSemaphore(workers)
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asr")
        self._asr.transcribe(np.zeros(SAMPLE_RATE, np.float32))
        self.load_s = time.perf_counter() - t
        self.workers = workers
        log.info("speech: Parakeet ready in %.1f s (%d workers)", self.load_s, workers)

    @property
    def model(self) -> AsrModel:
        return PARAKEET

    def transcribe(self, wav: bytes, t0: float) -> list[Word]:
        """The clip's words on the browser clock (`t0`: its first sample). Blocks while all
        workers are busy."""
        x = read_clip(wav)
        with self._slots:
            return self._asr.transcribe(x, t0)

    async def transcribe_async(self, wav: bytes, t0: float) -> list[Word]:
        """`transcribe` on a worker thread, so the server keeps answering meanwhile."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.transcribe, wav, t0)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
