"""Settings file: $KGRAG_HOME/kgrag.env (default ~/.kgrag/kgrag.env).

MCP clients pass environment variables to servers in different ways (Claude Desktop
and Cursor use an "env" object in JSON, Codex and the ChatGPT desktop app use a TOML
table), and a mistake there silently changes which extractor a collection is built
with. One file read by the server itself keeps every client on the same settings.

Format: KEY=VALUE per line; blank lines and # comments ignored; optional quotes.
Variables already set in the environment win over the file, so a client config can
still override a single value.
"""

from __future__ import annotations

import os
from pathlib import Path


def env_file_path() -> Path:
    home = Path(os.environ.get("KGRAG_HOME", Path.home() / ".kgrag")).expanduser()
    return Path(os.environ.get("KGRAG_ENV_FILE", home / "kgrag.env")).expanduser()


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """Apply the settings file to os.environ without overriding existing variables.
    Returns {key: "file" | "environment"} describing where each file key came from."""
    path = path or env_file_path()
    origin: dict[str, str] = {}
    if not path.is_file():
        return origin
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        if key in os.environ:
            origin[key] = "environment"
        else:
            os.environ[key] = value
            origin[key] = "file"
    return origin
