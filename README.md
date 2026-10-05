---
title: Put that here
emoji: 🦾
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Point at a block and say "put that here"
---

# deixis web demo

Point at a block and say "put that here": a virtual SO-101 robot arm in your browser, told
*where* by your finger and your voice, through the [`deixis`](https://github.com/rahuljeyaraj/deixis) core.

**Status (5 Oct 2026):** being built. The arm runs in the browser (MuJoCo WebAssembly and
three.js) with a mouse test; the finger, speech and the core come next. The plan is
[docs/plan.md](docs/plan.md); the page layout is
[docs/pictures/01-page-layout.png](docs/pictures/01-page-layout.png). What the demo can't do
is in [LIMITATIONS.md](LIMITATIONS.md); third-party licences are in [NOTICE.md](NOTICE.md).

## Run it on your computer

You need [uv](https://docs.astral.sh/uv/), Node 22 or newer, and the core checked out next to
this folder (`../deixis`).

```powershell
uv sync                          # the server (Python 3.11)
uv run webdemo                   # the server on http://127.0.0.1:8000
cd web
npm install
npm run dev                      # the page on http://localhost:5173 (it asks the server through /api)
```

Copy `.env.example` to `.env` and put a Nebius Token Factory key in it to use Nemotron.

## Tests

```powershell
uv run pytest                    # the server; the core is reached over MCP with no fallback
cd web
npm test                         # the arm, headless (MuJoCo WebAssembly in Node)
npx tsc --noEmit                 # type check
```

The block at the top of this file (between the `---` lines) holds the Hugging Face Space
settings (plan W14).
