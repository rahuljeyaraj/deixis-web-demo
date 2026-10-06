#!/usr/bin/env bash
# Build and start (or rebuild and restart) the judges' demo on the rented server.
#
#   bash deploy/run.sh <https-name>        # first time, e.g. bash deploy/run.sh demo.example.net
#   bash deploy/run.sh                      # again, with the same name (kept in deploy/.host)
#   bash deploy/run.sh check                # health and the last log lines
#
# Expects the source copied next to each other (rsync, from home):
#   ~/deixis/            the core   (only until deixis==0.1.0 is on PyPI)
#   ~/deixis-web-demo/   this repo
# If ../deixis is there, its wheel is built and the image installs it; otherwise the image
# installs deixis==0.1.0 from PyPI. The key stays in /etc/deixis/webdemo.env (root, 600).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
HOSTFILE="$HERE/.host"

if [ "${1:-}" = "check" ]; then
  host="$(cat "$HOSTFILE")"
  echo "== containers"; sudo docker compose -f "$HERE/compose.yaml" ps
  echo "== health"; curl -fsS "https://$host/healthz" && echo
  echo "== the last commands (words and what the robot did; no audio, no address)"
  sudo docker compose -f "$HERE/compose.yaml" exec -T webdemo tail -n 5 /data/logs/commands.jsonl || true
  exit 0
fi

if [ -n "${1:-}" ]; then echo "$1" > "$HOSTFILE"; fi
[ -s "$HOSTFILE" ] || { echo "give the HTTPS name the first time: bash deploy/run.sh <name>" >&2; exit 1; }
DEMO_HOST="$(cat "$HOSTFILE")"
case "$DEMO_HOST" in
  *deixis*|*rahul*) echo "the name must not say deixis or rahul (certificate logs are public)" >&2; exit 1 ;;
esac
sudo test -s /etc/deixis/webdemo.env || { echo "put the key in /etc/deixis/webdemo.env first" >&2; exit 1; }

rm -f "$REPO"/wheels/deixis-*.whl
if [ -d "$REPO/../deixis" ]; then
  echo "== the core's wheel, from ../deixis"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  (cd "$REPO/../deixis" && uv build --wheel --out-dir "$REPO/wheels")
else
  echo "== no ../deixis: the image installs deixis==0.1.0 from PyPI"
fi

echo "== build and start (the first build downloads Parakeet: a few minutes)"
cd "$HERE"
sudo DEMO_HOST="$DEMO_HOST" docker compose -f compose.yaml up -d --build
sudo docker image prune -f >/dev/null
echo "== waiting for the health check"
for _ in $(seq 1 60); do
  if curl -fsS "https://$DEMO_HOST/healthz" >/dev/null 2>&1; then
    echo "LIVE: https://$DEMO_HOST/"
    exit 0
  fi
  sleep 5
done
echo "not answering yet: sudo docker compose -f $HERE/compose.yaml logs --tail 50" >&2
exit 1
