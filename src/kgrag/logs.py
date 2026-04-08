"""Logging setup. Everything goes to stderr (stdout carries the MCP protocol) and,
optionally, to a file, which is the easiest place to read logs when the server is
launched by Claude Desktop or Cursor.

  KGRAG_LOG_LEVEL   DEBUG | INFO | WARNING | ERROR   (default INFO)
  KGRAG_LOG_FILE    path to also append logs to      (default $KGRAG_HOME/kgrag.log
                                                      when running as the MCP server)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_ROOT = "kgrag"
_configured = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_ROOT}.{name}")


def configure(log_file: str | Path | None = None) -> logging.Logger:
    """Idempotent. Library use (tests, eval) gets stderr logging; the server also
    passes a log file."""
    global _configured
    root = logging.getLogger(_ROOT)
    if _configured:
        return root
    level = getattr(logging, os.environ.get("KGRAG_LOG_LEVEL", "INFO").upper(), logging.INFO)
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s [pid %(process)d] %(message)s")
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(fmt)
    root.addHandler(h)
    path = os.environ.get("KGRAG_LOG_FILE") or log_file
    if path:
        try:
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(Path(path).expanduser(), encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError as e:  # unwritable log path must not stop the server
            root.warning("cannot open log file %s: %s", path, e)
    root.propagate = False
    _configured = True
    return root
