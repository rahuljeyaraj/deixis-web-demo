"""Settings from the environment (or a `.env` file next to `pyproject.toml`), plan 7 and 8.

The Nebius key is a server secret: it is never sent to the browser, logged or echoed back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    nebius_api_key: SecretStr | None = None
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    nemotron_model: str = "nvidia/Nemotron-3_5-Lightning"
    # Only Nemotron understands a person here. The word matcher that used to stand in for
    # it when it could not be reached was taken out on 11 Oct 2026 (rahul); what is left of
    # it is a test double, `tests/wordrules.py`, which the server cannot reach.
    ai: Literal["nemotron"] = "nemotron"

    # How much of our key each visitor may use (plan 8.1, D4: judges only, 2,000 a day and 500
    # per network, rahul 5 Oct); see `limits.py`.
    cap_visitor_per_day: int = 500
    cap_address_per_day: int = 500
    cap_total_per_day: int = 2000
    burst_s: float = 1.5
    # Proxies in front of the server (hosted: the Space's or Caddy's). The visitor's address is
    # then this many places from the right of `X-Forwarded-For`; 0: the connection's address.
    proxy_hops: int = 0

    # How the server reaches the core: over MCP (a `deixis-mcp` child process) or in-process.
    deixis_mode: Literal["mcp", "direct"] = "mcp"
    # Hosted: a crashed `deixis-mcp` falls back to in-process calls (logged). Tests: none.
    deixis_fallback: Literal["direct", "none"] = "direct"
    profile: Path = ROOT / "profiles" / "so101-table.json"

    # Parakeet (plan 7.3): where the sherpa-onnx export lives (`scripts/fetch_models.py`
    # downloads it; in development it can point at deixis-doodle's copy). Off: no speech.
    asr_models_dir: Path = ROOT / ".cache" / "models"
    asr_enabled: bool = True
    asr_workers: int = 2  # decodes at the same time; more requests wait
    asr_threads: int = 4  # ONNX Runtime threads per decode (4 was fastest on rahul's PC)

    # Token Factory list prices, US$ per million tokens (Nemotron 3.5 Lightning, 6 Oct 2026;
    # no cached-token price is published). For the cost line and the log (M33): re-check.
    price_in_per_m: float = 0.06
    price_out_per_m: float = 0.24
    # One JSON line per command and page event (M30, M31, M33); None: off.
    command_log: Path | None = ROOT / "logs" / "commands.jsonl"
    # The "use your own Nebius key" box: hidden for the judges' demo (web demo item 16).
    show_own_key: bool = False

    # The page's name (open decision D6) is read from here, so it changes in one place.
    title: str = "deixis"
    # The built front end; served at / when it exists (in development Vite serves the page).
    web_dist: Path = ROOT / "web" / "dist"

    host: str = "127.0.0.1"
    port: int = 8000
