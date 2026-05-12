"""Embedding backends.

Selected with KGRAG_EMBEDDER:
  auto                         (default) st:BAAI/bge-small-en-v1.5 if sentence-transformers
                               is installed, else hash
  hash                         offline feature-hashing embedder (lexical, no deps)
  st:<model>                   sentence-transformers, e.g. st:BAAI/bge-small-en-v1.5
  openai:<model>               any OpenAI-compatible /embeddings endpoint
                               (OPENAI_BASE_URL, OPENAI_API_KEY; works with Ollama, vLLM)

The embedder spec is recorded per corpus so queries always use the same space
the corpus was indexed with.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import urllib.request
from typing import Protocol

import numpy as np

from .logs import get_logger

log = get_logger("embed")


class Embedder(Protocol):
    spec: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing over word unigrams, bigrams and char 4-grams with
    sublinear tf. Deterministic and dependency-free. It is lexical, not semantic,
    so treat it as a stand-in for tests and offline runs, not a production model."""

    _word = re.compile(r"[a-z0-9]+")
    _stop = set("a an the of in on at to for by with and or is was were are be been it its as from that this which who what where when how".split())

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self.spec = f"hash:{dim}"

    def _feats(self, text: str) -> dict[str, float]:
        words = [w for w in self._word.findall(text.lower()) if w not in self._stop]
        f: dict[str, float] = {}
        for w in words:
            f["w:" + w] = f.get("w:" + w, 0) + 1.0
            if len(w) > 4:
                for i in range(len(w) - 3):
                    g = "c:" + w[i : i + 4]
                    f[g] = f.get(g, 0) + 0.25
        for a, b in zip(words, words[1:]):
            k = f"b:{a}_{b}"
            f[k] = f.get(k, 0) + 1.0
        return {k: 1.0 + math.log(v) if v >= 1 else v for k, v in f.items()}

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for k, v in self._feats(t).items():
                h = int.from_bytes(hashlib.blake2b(k.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += v if (h >> 63) & 1 else -v
        return _l2(out)


class SentenceTransformerEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer  # optional dependency

        # CPU by default: the embedding model is small, and on a Mac the GPU is usually
        # busy with the local LLM (Ollama), which makes the embedder run out of memory.
        # KGRAG_DEVICE=cuda (or mps) opts in to a GPU.
        self._m = SentenceTransformer(model, device=os.environ.get("KGRAG_DEVICE", "cpu"))
        getdim = getattr(self._m, "get_sentence_embedding_dimension", None) or getattr(self._m, "get_embedding_dimension")
        self.dim = int(getdim())
        self.spec = f"st:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = self._m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


class OpenAICompatEmbedder:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.spec = f"openai:{model}"
        self.dim = len(self._call(["dimension probe"])[0])

    def _call(self, texts: list[str]) -> list[list[float]]:
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": texts}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())["data"]
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows: list[list[float]] = []
        for i in range(0, len(texts), 128):
            rows.extend(self._call(texts[i : i + 128]))
        return _l2(np.asarray(rows, dtype=np.float32))


_CACHE: dict[str, Embedder] = {}


DEFAULT_ST_MODEL = "BAAI/bge-small-en-v1.5"


def resolve_spec(spec: str | None) -> str:
    """'auto' (the default) picks a real dense model when sentence-transformers is
    installed and falls back to the hash embedder, with a warning, when it is not."""
    spec = spec or os.environ.get("KGRAG_EMBEDDER") or "auto"
    if spec != "auto":
        return spec
    if importlib.util.find_spec("sentence_transformers") is not None:
        return f"st:{DEFAULT_ST_MODEL}"
    log.warning("sentence-transformers is not installed: using the lexical hash embedder. "
                "pip install 'kgrag-mcp[st]' or set KGRAG_EMBEDDER for dense embeddings.")
    return "hash"


def get_embedder(spec: str | None = None) -> Embedder:
    spec = resolve_spec(spec)
    if spec in _CACHE:
        return _CACHE[spec]
    kind, _, arg = spec.partition(":")
    if kind == "hash":
        e: Embedder = HashingEmbedder(int(arg) if arg else 2048)
    elif kind == "st":
        log.info("loading sentence-transformers model %s", arg or DEFAULT_ST_MODEL)
        e = SentenceTransformerEmbedder(arg or DEFAULT_ST_MODEL)
    elif kind == "openai":
        e = OpenAICompatEmbedder(arg or "text-embedding-3-small")
    else:
        raise ValueError(f"unknown embedder spec {spec!r}")
    _CACHE[spec] = _CACHE[e.spec] = e
    return e
