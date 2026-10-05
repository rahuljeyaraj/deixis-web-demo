"""Speech to timed words (plan W6): Parakeet on the arm's commands, the words on the browser
clock, the clip checks, and `/api/command` with a clip."""

from __future__ import annotations

import io
import json
import statistics
import threading
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import VOICE, needs_parakeet, needs_voice
from webdemo.app import create_app
from webdemo.settings import ROOT
from webdemo.speech import ClipError, read_clip, tokens_to_words

GROUND_TRUTH = json.loads((ROOT / "tests" / "fixtures" / "voice.json").read_text())["items"]
LEAD_S = 0.5  # the page's pre-roll
TAIL_S = 0.4  # the page's tail


def wav_bytes(x: np.ndarray, rate: int = 16000, channels: int = 1) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return b.getvalue()


def clip_of(item: dict, seed: int = 0) -> bytes:
    """A fixture voice as the page would send it: pre-roll, the words, the tail, a noise floor."""
    with wave.open(str(VOICE / item["wav"]), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    x = np.concatenate([np.zeros(int(LEAD_S * 16000), np.float32), x,
                        np.zeros(int(TAIL_S * 16000), np.float32)])  # fmt: skip
    x += np.random.default_rng(seed).normal(0, 1e-3, len(x)).astype(np.float32)
    return wav_bytes(x)


# --- tokens to words (copied from deixis-doodle with the code) ---


def test_leading_space_tokens():
    toks = [" P", "ut", " the", " do", "g", " he", "re"]
    ts = [0.08, 0.08, 0.16, 0.56, 0.56, 0.72, 0.72]
    w = tokens_to_words(toks, ts, t_end=1.2)
    assert [x[0] for x in w] == ["Put", "the", "dog", "here"]
    assert [x[1] for x in w] == [0.08, 0.16, 0.56, 0.72]
    assert w[0][2] == pytest.approx(0.16)  # ends where the next word starts
    assert w[-1][2] == pytest.approx(0.80)  # last token + one 80 ms frame


def test_sentencepiece_marker_durations_and_punctuation():
    w = tokens_to_words(["▁here", ",", "▁th", "is"], [0.96, 1.2, 1.5, 1.6], [0.24, 0.3, 0.16, 0.32])
    assert [x[0] for x in w] == ["here", "this"]
    assert w[1][2] == pytest.approx(1.92)
    assert [x[0] for x in tokens_to_words([" yes", ",", "no"], [0.1, 0.2, 0.3])] == ["yes", "no"]


# --- the clip ---


def test_read_clip_takes_what_the_page_sends():
    x = read_clip(wav_bytes(np.full(1600, 0.5, np.float32)))
    assert x.dtype == np.float32 and len(x) == 1600
    assert x[0] == pytest.approx(0.5, abs=1e-4)


@pytest.mark.parametrize(
    ("data", "why"),
    [
        (wav_bytes(np.zeros(4800, np.float32), rate=48000), "16 kHz"),
        (wav_bytes(np.zeros(3200, np.float32), channels=2), "mono"),
        (wav_bytes(np.zeros(16000 * 13, np.float32)), "at most 12 s"),
        (b"RIFF....not a wav", "not a WAV"),
        (b"", "not a WAV"),
    ],
    ids=["48kHz", "stereo", "13s", "garbage", "empty"],
)
def test_read_clip_refuses(data, why):
    with pytest.raises(ClipError, match=why):
        read_clip(data)


# --- Parakeet on the arm's commands ---


@needs_parakeet
@needs_voice
@pytest.mark.parametrize("voice", ["david", "zira"])
def test_put_that_here_on_the_browser_clock(speech, voice):
    """The done-when of W6: the right words, at the right browser-clock times. "that" lands
    within 0.1 s; the sentence's last word comes out up to 0.28 s late (Parakeet's timestamps
    move in 80 ms frames, and the last word lags most, as deixis-doodle's spike R2 found), well
    inside the profile's 0.6 s time sigma."""
    item = next(i for i in GROUND_TRUTH if i["id"] == f"{voice}_00")
    audio_t0 = 1234.5  # the browser's time of the clip's first sample
    words = speech.transcribe(clip_of(item), audio_t0)
    assert [w.text.lower() for w in words] == ["put", "that", "here"]
    that, here = words[1:]
    assert that.t0 == pytest.approx(audio_t0 + LEAD_S + item["words"][1]["t0"], abs=0.1)
    assert here.t0 == pytest.approx(audio_t0 + LEAD_S + item["words"][2]["t0"], abs=0.3)
    assert all(w.t1 > w.t0 for w in words)


@needs_parakeet
@needs_voice
def test_every_arm_command(speech):
    """All 24 clips word for word; the demonstratives' start times (what deixis fuses with the
    finger) a median within 0.15 s and never over 0.4 s (the profile's time sigma is 0.6 s)."""
    errors = []
    for item in GROUND_TRUTH:
        words = speech.transcribe(clip_of(item), 100.0)
        assert [w.text.lower() for w in words] == [w["text"] for w in item["words"]], item["id"]
        for w, truth in zip(words, item["words"], strict=True):
            if truth["deictic"]:
                errors.append(abs(w.t0 - (100.0 + LEAD_S + truth["t0"])))
    assert len(errors) == 14
    assert statistics.median(errors) < 0.15
    assert max(errors) < 0.4


@needs_parakeet
@needs_voice
def test_parallel_decodes_match(speech):
    """Two workers share one recognizer: decodes at the same time give the same words."""
    items = [i for i in GROUND_TRUTH if i["id"] in ("david_01", "zira_03", "david_02", "zira_11")]
    clips = [clip_of(i) for i in items]
    alone = [speech.transcribe(c, 0.0) for c in clips]
    together: list = [None] * len(clips)

    def run(k: int) -> None:
        together[k] = speech.transcribe(clips[k], 0.0)

    threads = [threading.Thread(target=run, args=(k,)) for k in range(len(clips))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert together == alone


# --- the route ---


@needs_parakeet
@needs_voice
def test_command_route_returns_the_words(api):
    item = next(i for i in GROUND_TRUTH if i["id"] == "zira_02")
    r = api.post(
        "/api/command",
        data={"meta": json.dumps({"audio_t0": 50.0})},
        files={"audio": ("clip.wav", clip_of(item), "audio/wav")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert [w["text"].lower() for w in body["words"]] == ["put", "that", "on", "that"]
    assert all(50.0 <= w["t0"] <= w["t1"] <= 50.0 + 3.5 for w in body["words"])
    assert 0 < body["timings"]["asr_s"] <= body["timings"]["total_s"]


@needs_parakeet
def test_command_route_refuses_bad_input(api):
    silence = wav_bytes(np.zeros(1600, np.float32))
    post = api.post
    assert post("/api/command", data={"meta": "{}"}).status_code == 400  # no clip, no text
    assert post("/api/command", data={"meta": "not json"}).status_code == 400
    r = post("/api/command", data={"meta": "{}"}, files={"audio": ("a.wav", silence, "audio/wav")})
    assert r.status_code == 400 and "audio_t0" in r.text
    r = post("/api/command", data={"meta": json.dumps({"audio_t0": 1})},
             files={"audio": ("a.wav", b"nope", "audio/wav")})  # fmt: skip
    assert r.status_code == 400 and "not a WAV" in r.text
    long = wav_bytes(np.zeros(16000 * 13, np.float32))
    r = post("/api/command", data={"meta": json.dumps({"audio_t0": 1})},
             files={"audio": ("a.wav", long, "audio/wav")})  # fmt: skip
    assert r.status_code == 413
    r = post("/api/command", data={"meta": json.dumps({"audio_t0": 1, "pad": "x" * 1_000_001})})
    assert r.status_code == 413


def test_command_route_without_speech(settings, core, tmp_path: Path):
    settings.asr_models_dir = tmp_path  # nothing downloaded there
    settings.asr_enabled = True
    with TestClient(create_app(settings, client=core)) as c:
        assert c.get("/healthz").json()["speech"] == {"loaded": False}
        r = c.post("/api/command", data={"meta": json.dumps({"audio_t0": 1})},
                   files={"audio": ("a.wav", wav_bytes(np.zeros(160)), "audio/wav")})  # fmt: skip
        assert r.status_code == 503
