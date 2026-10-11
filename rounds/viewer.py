"""Turn a run's attempts into one HTML page you can click through.

    python -m rounds.viewer out/r2-main.json

The harness prints transcripts, but a transcript does not show *where things are*, and the
whole difficulty of this scene is where things are. So this draws the shelf as the person
sees it, shades each thing by how much of their point it holds, and steps through the
conversation one line at a time.

It is not the demo. It is a window onto the numbers, so a run can be read rather than
trusted. The data is written into the page, so the file opens on its own with no server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rounds.scene import shelf

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The three rounds</title>
<style>
  :root {
    --bg: #fbfaf7; --card: #ffffff; --ink: #1b1a17; --soft: #5c5952;
    --line: #ddd8cf; --warm: #b4541f; --cool: #2b6ca3; --good: #2f7a43;
    --bad: #b03030; --shelf: #c9a579;
  }
  :root:not([data-theme="light"]) { }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #17181a; --card: #202225; --ink: #f0eee9; --soft: #a9a49b;
      --line: #35383c; --warm: #e5884f; --cool: #6fb0dd; --good: #6cc184;
      --bad: #e07a7a; --shelf: #8a6f4c;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 19px/1.7 system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 24px 16px 64px;
  }
  .wrap { max-width: 920px; margin: 0 auto; }
  h1 { font-size: 27px; margin: 0 0 4px; letter-spacing: -0.01em; }
  .sub { color: var(--soft); margin: 0 0 24px; }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 20px; margin-bottom: 18px;
  }
  .bar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 18px; }
  button, select {
    font: inherit; font-size: 17px; padding: 9px 16px; border-radius: 10px;
    border: 1px solid var(--line); background: var(--card); color: var(--ink);
    cursor: pointer;
  }
  button.main { background: var(--ink); color: var(--bg); border-color: var(--ink); }
  button:disabled { opacity: 0.4; cursor: default; }
  .want { font-size: 21px; margin: 0 0 14px; }
  .want b { color: var(--warm); }
  .round { color: var(--soft); font-size: 16px; letter-spacing: 0.06em;
           text-transform: uppercase; margin: 0 0 10px; }
  svg { width: 100%; height: auto; display: block; }
  .log { margin: 18px 0 0; padding: 0; list-style: none; }
  .log li {
    display: grid; grid-template-columns: 76px 1fr; gap: 14px;
    padding: 7px 0; border-top: 1px solid var(--line);
  }
  .log li:first-child { border-top: 0; }
  .who { color: var(--soft); font-size: 15px; text-align: right; padding-top: 3px; }
  .who.person { color: var(--cool); }
  .who.robot { color: var(--warm); }
  .verdict { font-weight: 600; }
  .verdict.ok { color: var(--good); }
  .verdict.no { color: var(--bad); }
  .legend { color: var(--soft); font-size: 16px; margin-top: 12px; }
  .score { display: flex; flex-wrap: wrap; gap: 22px; margin-top: 4px; }
  .score div { font-size: 16px; color: var(--soft); }
  .score b { display: block; font-size: 25px; color: var(--ink); }
</style>
</head>
<body>
<div class="wrap">
  <h1>The three rounds</h1>
  <p class="sub">Which thing on the shelf does the person mean? __SUB__</p>

  <div class="card">
    <div class="score" id="score"></div>
  </div>

  <div class="bar">
    <select id="pick"></select>
    <button id="back">Back</button>
    <button id="next" class="main">Next step</button>
    <button id="all">Show all</button>
  </div>

  <div class="card">
    <p class="want" id="want"></p>
    <p class="round" id="round"></p>
    <svg id="shelf" viewBox="0 0 860 420" role="img" aria-label="the shelf"></svg>
    <p class="legend">A thing's ring is how much of the person's point it holds. Orange
      outline: what they actually wanted. Filled: what the robot picked.</p>
    <ul class="log" id="log"></ul>
  </div>
</div>
<script>
const DATA = __DATA__;
const SHELF = __SHELF__;
const SV = document.getElementById('shelf');
let trial = 0, step = 0;

function layout(t) {
  const x = 430 + t.col * 150, y = 150 + t.row * 170;
  return {x, y};
}

function draw(att) {
  const weights = Object.fromEntries((att.cone || []).map(c => [c[0], c[1]]));
  const asked = (att.shown_asks || []);
  let s = '';
  for (const row of [0, 1]) {
    const y = 150 + row * 170 + 46;
    s += `<rect x="40" y="${y}" width="780" height="9" rx="4" fill="var(--shelf)"/>`;
  }
  for (const t of SHELF) {
    const {x, y} = layout(t);
    const w = weights[t.id] || 0;
    if (w > 0) {
      s += `<circle cx="${x}" cy="${y}" r="${18 + w * 52}" fill="var(--cool)"
             opacity="${0.10 + w * 0.30}"/>`;
    }
    const isTarget = t.id === att.target;
    const isPicked = att.shown_pick === t.id;
    const ring = isTarget ? 'var(--warm)' : 'var(--line)';
    const fill = isPicked ? (att.ok ? 'var(--good)' : 'var(--bad)') : 'var(--card)';
    s += `<circle cx="${x}" cy="${y}" r="22" fill="${fill}" stroke="${ring}"
           stroke-width="${isTarget ? 4 : 2}"/>`;
    const a = asked.indexOf(t.id);
    if (a >= 0) {
      s += `<text x="${x}" y="${y + 6}" text-anchor="middle" font-size="17"
             fill="${isPicked ? 'var(--card)' : 'var(--ink)'}">${a + 1}</text>`;
    }
    s += `<text x="${x}" y="${y + 46}" text-anchor="middle" font-size="15"
           fill="var(--soft)">${t.id}</text>`;
    if (w > 0) {
      s += `<text x="${x}" y="${y - 34}" text-anchor="middle" font-size="14"
             fill="var(--cool)">${w.toFixed(2)}</text>`;
    }
  }
  SV.innerHTML = s;
}

function render() {
  const att = DATA.attempts[trial];
  const total = att.log.length;
  step = Math.max(0, Math.min(step, total));
  // Replay only as far as the person has stepped, so the picture matches the words.
  const lines = att.log.slice(0, step);
  att.shown_asks = [];
  att.shown_pick = null;
  for (const line of lines) {
    let m = line.match(/points at (\\w+) and asks/);
    if (m) att.shown_asks.push(m[1]);
    m = line.match(/takes the best one: (\\w+)|commits to (\\w+)/);
    if (m) att.shown_pick = m[1] || m[2];
    m = line.match(/robot\\s+picks (\\w+)/);
    if (m) att.shown_pick = m[1];
  }
  if (step >= total && !att.shown_pick) att.shown_pick = att.picked;
  const done = step >= total;
  document.getElementById('want').innerHTML =
    `The person wants <b>${att.target}</b> &mdash; ${att.target_seen}`;
  document.getElementById('round').textContent =
    att.round + (done ? (att.ok ? '  \\u2014 right' : '  \\u2014 wrong') : '');
  draw(att);
  document.getElementById('log').innerHTML = lines.map(line => {
    const who = line.startsWith('person') ? 'person'
      : line.startsWith('robot') ? 'robot' : '';
    const rest = line.replace(/^(person|robot|cone)\\s+/, '');
    const label = line.startsWith('cone') ? 'point' : who;
    return `<li><span class="who ${who}">${label}</span><span>${rest}</span></li>`;
  }).join('');
  document.getElementById('next').disabled = done;
  document.getElementById('back').disabled = step === 0;
}

function fill() {
  const sel = document.getElementById('pick');
  sel.innerHTML = DATA.attempts.map((a, i) =>
    `<option value="${i}">${i + 1}. ${a.round} &mdash; wants ${a.target}</option>`).join('');
  document.getElementById('score').innerHTML = DATA.rounds.map(r =>
    `<div>${r.round}<b>${r.pct}%</b>${r.correct} of ${r.n}</div>`).join('');
}

document.getElementById('next').onclick = () => { step++; render(); };
document.getElementById('back').onclick = () => { step--; render(); };
document.getElementById('all').onclick = () => { step = 999; render(); };
document.getElementById('pick').onchange = e => {
  trial = +e.target.value; step = 0; render();
};
fill();
render();
</script>
</body>
</html>
"""


def build(attempts: list[dict], out: Path, sub: str = "") -> Path:
    """Write the page. `attempts` is the harness's JSON, as `--json` saved it."""
    things = shelf()
    seen = {t.id: t.seen for t in things}
    for a in attempts:
        a["target_seen"] = seen.get(a["target"], "")
    rounds: dict[str, dict] = {}
    for a in attempts:
        r = rounds.setdefault(a["round"], {"round": a["round"], "n": 0, "correct": 0})
        r["n"] += 1
        r["correct"] += int(a["picked"] == a["target"] and not a["invalid"] and not a["failed"])
    for r in rounds.values():
        r["pct"] = round(100 * r["correct"] / r["n"]) if r["n"] else 0
    for a in attempts:
        a["ok"] = a["picked"] == a["target"] and not a["invalid"] and not a["failed"]
    data = {"attempts": attempts, "rounds": list(rounds.values())}
    page = (
        TEMPLATE.replace("__DATA__", json.dumps(data))
        .replace(
            "__SHELF__", json.dumps([{"id": t.id, "col": t.col, "row": t.row} for t in things])
        )
        .replace("__SUB__", sub)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m rounds.viewer out/r2-main.json [out/page.html]")
        return 2
    src = Path(args[0])
    out = Path(args[1]) if len(args) > 1 else src.with_suffix(".html")
    attempts = json.loads(src.read_text(encoding="utf-8"))
    built = build(attempts, out, sub=f"{len(attempts) // 3} trials, from {src.name}")
    print(f"wrote {built}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
