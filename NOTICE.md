# Notices

This repository is MIT licensed (see `LICENSE`). It uses, and in one case includes, the work
of others under their own licences.

## Included in this repository

- **SO-101 arm model** from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)
  (`robotstudio_so101`, by The Robot Studio and Google DeepMind), Apache License 2.0.
  Copied into `web/model/so101/` at the commit in `web/model/so101/MENAGERIE_COMMIT`, with its
  `LICENSE` file. `web/model/table_scene.xml` (ours) includes it. **One change** (5 Oct 2026):
  the wrist camera mount (`camera_mount` body, its wrist camera and its collision boxes) is
  removed from `so101.xml`, because the demo has no wrist camera and the mount hit the arm's
  shoulder near the base.

## Used, not included

| What | By | Licence | Where it runs |
|---|---|---|---|
| [MuJoCo](https://mujoco.org) and its WebAssembly bindings `@mujoco/mujoco` | Google DeepMind | Apache-2.0 | the browser |
| [three.js](https://threejs.org) | three.js authors | MIT | the browser |
| [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (Hand Landmarker) | Google | Apache-2.0 | the browser |
| [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | NVIDIA | CC-BY-4.0 | the server (downloaded at build, not in this repo) |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and its int8 Parakeet export | k2-fsa | Apache-2.0 | the server |
| NVIDIA Nemotron 3.5 Lightning, on [Nebius Token Factory](https://nebius.com/services/token-factory/nemotron) | NVIDIA, Nebius | NVIDIA open model licence | a hosted API |
| [deixis](https://github.com/rahuljeyaraj/deixis) | Rahul Jeyaraj | MIT | the server, over MCP |
| [FastAPI](https://fastapi.tiangolo.com), uvicorn, pydantic, the MCP Python SDK | their authors | MIT or BSD | the server |

Parakeet is credited on the page itself, in "What runs this demo".

## Read, not copied

[bonustrack/so101-mujoco](https://github.com/bonustrack/so101-mujoco) (Apache-2.0) showed that
the SO-101 runs in the browser with `@mujoco/mujoco` 3.14 and three.js, and how to pack the
model into one file. No code was copied from it.
