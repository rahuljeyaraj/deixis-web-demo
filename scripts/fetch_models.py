"""Download Parakeet (sherpa-onnx's int8 export, about 630 MB unpacked) into .cache/models/.

    uv run python scripts/fetch_models.py [--dir DIR]

The Docker build runs this (plan W14). In development a link to another copy works too, for
example deixis-doodle's (on Windows: `mklink /J`).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from webdemo.speech import PARAKEET  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ROOT / ".cache" / "models")
    args = ap.parse_args()
    if PARAKEET.available(args.dir):
        print(f"{PARAKEET.archive} is already in {args.dir}")
        return
    args.dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{PARAKEET.archive}.tar.bz2"
        print(f"downloading {PARAKEET.url}")
        with urllib.request.urlopen(PARAKEET.url) as r, archive.open("wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        with tarfile.open(archive, "r:bz2") as t:
            t.extractall(args.dir, filter="data")
    if not PARAKEET.available(args.dir):
        sys.exit(f"the archive did not contain the expected files: {PARAKEET.files(args.dir)}")
    print(f"done: {args.dir / PARAKEET.archive}")


if __name__ == "__main__":
    main()
