"""Make the voice test set for the arm: its commands spoken by Windows' built-in voices.

    uv run python scripts/make_voice_fixtures.py

Writes 16 kHz mono WAVs to .cache/voice/ (never committed: they are Microsoft voices'
output) and the ground truth, each word's start time from SAPI's SpeakProgress events, to
tests/fixtures/voice.json (committed). Tests that need the WAVs skip when they are missing.
Shaped after deixis-doodle's scripts/make_test_utterances.py (commit 7a7e9cc).

Windows only (System.Speech via PowerShell).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PS1 = ROOT / "scripts" / "make_voice_fixtures.ps1"
GROUND_TRUTH = ROOT / "tests" / "fixtures" / "voice.json"
OUT = ROOT / ".cache" / "voice"

# The arm's phrases (plan 6.4 and the scenes in section 10).
SENTENCES = [
    "put that here",
    "put the blue one next to that",
    "put that on that",
    "move it a bit left",
    "a bit to the right",
    "put it there",
    "yes",
    "no",
    "stop",
    "go home",
    "this one",
    "the red one",
    # Halves of "put that here", joined with a pause by web/scripts/flow-check.mjs.
    "put that",
    "here",
]

VOICES = [("Microsoft David Desktop", "david", 0), ("Microsoft Zira Desktop", "zira", 1)]
DEMONSTRATIVES = frozenset({"here", "there", "this", "that", "these", "those"})


def words_of(text: str) -> list[str]:
    """Lowercase, no punctuation (apostrophes inside words stay)."""
    out = (re.sub(r"[^a-z0-9']+", "", w.lower()).strip("'") for w in text.split())
    return [w for w in out if w]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0, sr


def sound_span(
    x: np.ndarray, sr: int, rel_db: float = -35.0, frame_s: float = 0.005
) -> tuple[float, float]:
    """First and last 5 ms frame within `rel_db` of the loudest frame."""
    n = int(round(frame_s * sr))
    k = len(x) // n
    e = np.mean(np.square(x[: k * n].reshape(k, n), dtype=np.float64), axis=1)
    db = 10 * np.log10(e + 1e-12)
    idx = np.flatnonzero(db >= db.max() + rel_db)
    return float(idx[0] * frame_s), float((idx[-1] + 1) * frame_s)


def synthesise(out_dir: Path) -> list[dict]:
    jobs = [
        {"id": f"{short}_{i:02d}", "text": text, "voice": voice, "rate": rate,
         "wav": str((out_dir / f"{short}_{i:02d}.wav").resolve())}
        for voice, short, rate in VOICES
        for i, text in enumerate(SENTENCES)
    ]  # fmt: skip
    jobs_path = out_dir / "jobs.json"
    events_path = out_dir / "events.json"
    jobs_path.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PS1),
         "-Jobs", str(jobs_path), "-Out", str(events_path)],
        check=True,
    )  # fmt: skip
    events = json.loads(events_path.read_text(encoding="utf-8-sig"))
    by_id = {e["id"]: e for e in events}
    return [{**j, "events": by_id[j["id"]]["words"]} for j in jobs]


def build_item(job: dict) -> dict:
    x, sr = read_wav(Path(job["wav"]))
    first, last = sound_span(x, sr)
    ref = words_of(job["text"])
    events = [e for e in job["events"] if words_of(e["text"])]
    if [words_of(e["text"])[0] for e in events] != ref:
        raise RuntimeError(f"{job['id']}: SAPI words {events} don't match {ref}")
    words = []
    for k, e in enumerate(events):
        t0 = e["ms"] / 1000
        t1 = events[k + 1]["ms"] / 1000 if k + 1 < len(events) else last
        words.append({"i": k, "text": ref[k], "t0": round(t0, 4), "t1": round(t1, 4),
                      "deictic": ref[k] in DEMONSTRATIVES})  # fmt: skip
    return {
        "id": job["id"],
        "voice": job["voice"],
        "text": job["text"],
        "wav": Path(job["wav"]).name,
        "duration": round(len(x) / sr, 4),
        "first_sound": round(first, 4),
        "words": words,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if sys.platform != "win32":
        sys.exit("needs Windows (System.Speech)")
    args.out.mkdir(parents=True, exist_ok=True)
    items = [build_item(j) for j in synthesise(args.out)]
    meta = {
        "description": "The arm's commands in Windows SAPI voices; word starts from "
        "SpeakProgress events. t0/t1 in seconds from the start of the WAV; t1 is the next "
        "word's start (or the end of speech), so it includes any pause.",
        "generator": "scripts/make_voice_fixtures.py",
        "wav_dir": ".cache/voice",
        "sample_rate": 16000,
    }
    GROUND_TRUTH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH.write_text(json.dumps({"meta": meta, "items": items}, indent=1), encoding="utf-8")
    print(f"wrote {GROUND_TRUTH.relative_to(ROOT)} and {len(items)} WAVs in {args.out}")


if __name__ == "__main__":
    main()
