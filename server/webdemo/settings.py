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

    # How the server reaches the core: over MCP (a `deixis-mcp` child process) or in-process.
    deixis_mode: Literal["mcp", "direct"] = "mcp"
    # Hosted: a crashed `deixis-mcp` falls back to in-process calls (logged). Tests: none.
    deixis_fallback: Literal["direct", "none"] = "direct"
    profile: Path = ROOT / "profiles" / "so101-table.json"

    # The page's name (open decision D6) is read from here, so it changes in one place.
    title: str = "deixis"
    # The built front end; served at / when it exists (in development Vite serves the page).
    web_dist: Path = ROOT / "web" / "dist"

    host: str = "127.0.0.1"
    port: int = 8000
