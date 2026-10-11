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

**Status (11 Oct 2026):** working. The arm runs in the browser (MuJoCo WebAssembly and
three.js), your fingertip and your voice say where, and the `deixis` core answers over MCP.
One sentence can be a plan of up to four steps ("stack them all up", "swap the red and the
blue"): Nemotron decides *what*, the core decides *where*, and the core measures the table
again between steps. A panel under the scene shows what each part said to the next, live.
What the demo can't do is in [LIMITATIONS.md](LIMITATIONS.md); third-party licences are in
[NOTICE.md](NOTICE.md).

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
uv run pytest -m network         # the real Nemotron, opt-in (needs a key; about $0.02 a run)
cd web
npm test                         # the arm, headless (MuJoCo WebAssembly in Node)
npx tsc --noEmit                 # type check
```

`uv run pytest` answers every command with a fake model, so the `-m network` run is the one
that checks what a judge will see: `tests/test_live_web.py` drives the whole server with the
real key and asserts that **every** command came back from Nemotron, never from the word
matcher.

The block at the top of this file (between the `---` lines) holds the Hugging Face Space
settings (plan W14).
