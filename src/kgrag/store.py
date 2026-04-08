"""Per-corpus SQLite storage. One file per corpus under $KGRAG_HOME (default ~/.kgrag).

Embeddings are stored as float32 blobs next to the rows they describe and loaded
into numpy matrices for search, so a corpus is a single portable file.
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY, title TEXT, source TEXT, metadata TEXT, content_hash TEXT UNIQUE,
    added_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, doc_id INTEGER REFERENCES docs(id), idx INTEGER, text TEXT, emb BLOB);
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY, name TEXT, type TEXT, description TEXT, emb BLOB, mention_count INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS aliases (entity_id INTEGER, alias TEXT, PRIMARY KEY (entity_id, alias));
CREATE TABLE IF NOT EXISTS mentions (entity_id INTEGER, chunk_id INTEGER, PRIMARY KEY (entity_id, chunk_id));
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY, src INTEGER, dst INTEGER, label TEXT, description TEXT, weight REAL, emb BLOB,
    UNIQUE (src, label, dst));
CREATE TABLE IF NOT EXISTS rel_evidence (rel_id INTEGER, chunk_id INTEGER, PRIMARY KEY (rel_id, chunk_id));
CREATE INDEX IF NOT EXISTS idx_mentions_chunk ON mentions(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")


def home() -> Path:
    p = Path(os.environ.get("KGRAG_HOME", Path.home() / ".kgrag")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def corpus_path(name: str, root: Path | None = None) -> Path:
    if not _NAME_RE.match(name):
        raise ValueError("corpus name must be 1-64 chars of letters, digits, '-' or '_'")
    return (root or home()) / "corpora" / f"{name}.sqlite"


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # timeout = busy timeout: a second server process waits for the write lock
    # instead of failing with "database is locked"
    db = sqlite3.connect(str(path), check_same_thread=False, timeout=120)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript(SCHEMA)
    cols = {r[1] for r in db.execute("PRAGMA table_info(chunks)")}
    if "extracted" not in cols:  # 0 = graph extraction failed, retry on next ingest
        db.execute("ALTER TABLE chunks ADD COLUMN extracted INTEGER DEFAULT 1")
        db.commit()
    return db


def connect_ro(path: Path) -> sqlite3.Connection:
    """Separate read connection, so queries never share a connection (or a half-done
    transaction) with an ingest running in another thread. WAL lets them run concurrently."""
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False, timeout=30)


def to_blob(v: np.ndarray | None) -> bytes | None:
    return None if v is None else np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes | None) -> np.ndarray | None:
    return None if b is None else np.frombuffer(b, dtype=np.float32)


def stack(blobs: list[bytes | None], dim: int) -> np.ndarray:
    m = np.zeros((len(blobs), dim), dtype=np.float32)
    for i, b in enumerate(blobs):
        if b is not None:
            m[i] = np.frombuffer(b, dtype=np.float32)
    return m
