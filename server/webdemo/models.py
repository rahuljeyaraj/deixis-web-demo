"""Every model the demo uses, for the page's "What runs this demo" panel (plan 8.4).

`/api/info` returns this list, so the page always matches the server.
"""

from __future__ import annotations

from typing import Any


def models_in_use(nemotron_model: str, simple_mode: bool = False) -> list[dict[str, Any]]:
    """The panel's lines in order. `simple_mode`: today's AI budget is used up (plan 8.1)."""
    return [
        {
            "role": "Hand tracking",
            "name": "MediaPipe Hand Landmarker",
            "by": "Google",
            "where": "in your browser; your video never leaves it",
            "url": "https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker",
        },
        {
            "role": "Speech to words, with the time of each word",
            "name": "NVIDIA Parakeet TDT 0.6B v2",
            "by": "NVIDIA",
            "where": "on our server",
            "url": "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2",
        },
        {
            "role": "Understanding what you asked",
            "name": "NVIDIA Nemotron 3.5 Lightning",
            "by": "NVIDIA",
            "where": "on Nebius Token Factory; it turns your words into the robot's tool calls",
            "url": "https://nebius.com/services/token-factory/nemotron",
            "model_id": nemotron_model,
            "off": "off right now: today's budget is used up" if simple_mode else None,
        },
        {
            "role": 'Finding "here" and "that"',
            "name": "the deixis core",
            "by": "open source, MIT",
            "where": (
                "plain code, no AI model; reached over MCP, the same way the real plotter and "
                "the real vacuum reach it"
            ),
            "url": "https://github.com/rahuljeyaraj/deixis",
        },
        {
            "role": "The arm",
            "name": "MuJoCo physics, with the SO-101 arm from MuJoCo Menagerie",
            "by": "Google DeepMind; the arm model by The Robot Studio",
            "where": "in your browser",
            "url": "https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101",
        },
        {
            "role": "The robot's voice",
            "name": "your browser's own voice",
            "by": "your browser",
            "where": "in your browser",
            "url": "https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesis",
        },
    ]


NOT_USED = [
    {
        "name": "NVIDIA Cosmos",
        "why": (
            "Not in this web demo. The real plotter uses Cosmos 3 Nano to name what you drew; "
            "you can see it in the video."
        ),
    }
]
