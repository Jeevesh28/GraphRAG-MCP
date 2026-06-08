"""MCP server (stdio) for agent-managed knowledge collections.

Tools: create, add, list, query.

stdout carries the MCP protocol. Logs go to stderr and $KGRAG_HOME/kgrag.log, and
each tool runs with Python-level stdout redirected to stderr, so a library that
prints cannot corrupt the protocol stream (mcp 2.x also guards the file descriptor).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore
    from mcp.server.fastmcp.exceptions import ToolError  # type: ignore

from . import store
from .corpus import Corpus
from .extract import LLMError
from .logs import configure, get_logger
from .retrieve import RetrievalParams, retrieve
from .settings import env_file_path, load_env_file

log = get_logger("server")
mcp = _Server("kgrag")
_open: dict[str, Corpus] = {}

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".json", ".csv", ".py", ".tex"}
MAX_FILE_BYTES = 5_000_000
MAX_DOCS_PER_CALL = 200
MODES = ("hybrid", "dense", "graph")
FUSIONS = ("weighted", "rrf")


# ------------------------------------------------------------------ helpers
def _default_params() -> dict:
    """Retrieval defaults tuned by kgrag-eval --save-params. Looked up in
    $KGRAG_PARAMS, then $KGRAG_HOME/retrieval_params.json. Per-call params override."""
    for p in (os.environ.get("KGRAG_PARAMS"), store.home() / "retrieval_params.json"):
        if p and Path(p).expanduser().is_file():
            try:
                d = json.loads(Path(p).expanduser().read_text())
                d.pop("k", None)
                d.pop("mode", None)
                return d
            except Exception as e:
                log.warning("ignoring unreadable params file %s: %s", p, e)
    return {}


def _get(name: str) -> Corpus:
    if name not in _open:
        c = Corpus.open(name)
        _open[name] = c
        if c.stats().get("chunks_pending_extraction"):  # resume work interrupted by a restart
            log.info("collection %s has chunks pending extraction; resuming in the background", name)
            c.start_extraction_worker()
    return _open[name]


def _warnings(c: Corpus) -> list[str]:
    w = []
    if c.meta.get("extractor") == "heuristic":
        w.append("This collection uses the offline heuristic extractor (pattern matching, not an LLM). "
                 "Set KGRAG_EXTRACTOR before creating collections for LLM-built graphs.")
    if str(c.meta.get("embedder", "")).startswith("hash"):
        w.append("This collection uses the lexical hash embedder, not a dense embedding model. "
                 "Install sentence-transformers or set KGRAG_EMBEDDER before creating collections.")
    return w


def _tool(fn):
    """Uniform logging, timing and error reporting for every tool.

    Expected problems (bad arguments, unknown collection, LLM provider errors) come
    back to the agent as a ToolError with an actionable message. Anything else is
    logged with a traceback and reported with its type and message, so the agent
    and the user can see what went wrong instead of a generic failure.
    """
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        t0 = time.time()
        brief = {k: (f"<{len(v)} items>" if isinstance(v, list) else v) for k, v in kwargs.items() if k != "query"}
        log.info("tool %s called %s", fn.__name__, brief)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                out = await fn(*args, **kwargs)
            log.info("tool %s ok in %.2fs", fn.__name__, time.time() - t0)
            return out
        except ToolError:
            raise
        except (ValueError, LLMError, FileNotFoundError) as e:
            log.warning("tool %s failed: %s", fn.__name__, e)
            raise ToolError(str(e)) from e
        except Exception as e:
            log.exception("tool %s crashed", fn.__name__)
            raise ToolError(f"internal error ({type(e).__name__}): {e}. See the kgrag log for details.") from e
    return wrapper


def _validate_documents(documents: Any) -> list[dict]:
    if documents is None:
        return []
    if not isinstance(documents, list):
        raise ValueError("documents must be a list of objects like {\"text\": \"...\", \"title\": \"...\"}")
    out = []
    for i, d in enumerate(documents):
        if isinstance(d, str):  # be forgiving: a bare string is a document
            d = {"text": d}
        if not isinstance(d, dict) or not isinstance(d.get("text"), str):
            raise ValueError(f"documents[{i}] must be an object with a string 'text' field")
        if d.get("metadata") is not None and not isinstance(d["metadata"], dict):
            raise ValueError(f"documents[{i}].metadata must be an object")
        out.append(d)
    return out


def _read_paths(paths: list[str] | None) -> tuple[list[dict], list[str]]:
    docs, errors = [], []
    for p in paths or []:
        fp = Path(p).expanduser()
        try:
            if not fp.is_file():
                raise FileNotFoundError("not a file")
            if fp.suffix.lower() not in TEXT_SUFFIXES:
                raise ValueError(f"unsupported file type {fp.suffix or '(none)'}; supported: {', '.join(sorted(TEXT_SUFFIXES))}")
            if fp.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("file larger than 5 MB")
            docs.append({"text": fp.read_text(errors="replace"), "title": fp.name, "source": str(fp.resolve())})
        except Exception as e:
            errors.append(f"{p}: {e}")
    return docs, errors


# -------------------------------------------------------------------- tools
@mcp.tool(name="create")
@_tool
async def create(name: str, description: str = "") -> str:
    """Create a new, empty knowledge collection that you own and curate.

    Make one collection per topic or task (e.g. "rust-async-runtime",
    "competitor-pricing"). The description should say what belongs in it, so you
    and other agents know what to collect and when to search it. Call `list` first
    to reuse an existing collection instead of creating a near-duplicate.

    Args:
        name: 1-64 chars of letters, digits, '-' or '_'.
        description: What this collection is for and what belongs in it.
    """
    c = await asyncio.to_thread(Corpus.create, name, description)
    _open[name] = c
    return json.dumps({"created": c.stats(), "warnings": _warnings(c)})


@mcp.tool(name="add")
@_tool
async def add(collection: str, documents: list[dict | str] | None = None, paths: list[str] | None = None,
              wait: bool = False) -> str:
    """Add documents to a collection. Each document is chunked and embedded right
    away, so it is searchable by vector similarity as soon as this returns. An LLM
    then extracts its entities and relations into the collection's knowledge graph;
    new entities are resolved against the ones already there, so "M. Okonkwo" joins
    the existing "Dr. Mira Okonkwo" node instead of creating a duplicate.

    Graph extraction runs in the background by default, because a local LLM can
    take minutes. Call `list` with the collection to see progress
    (chunks_pending_extraction, extraction_in_progress, last_extraction_error).
    Graph-based answers are complete once nothing is pending.

    Add things you have read and judged worth keeping: fetched pages, notes, file
    contents. Prefer clean text over raw HTML. Exact duplicates are skipped.

    Args:
        collection: Collection name.
        documents: [{"text": str, "title": str (optional), "source": str (optional, e.g. URL),
                     "metadata": object (optional)}]. A bare string is accepted as {"text": ...}.
        paths: Local text files to read and add (txt, md, html, json, csv, py, rst, tex).
        wait: If true, build the graph before returning (can exceed the client's tool
            timeout with a slow model). Default false.
    """
    docs = _validate_documents(documents)
    file_docs, errors = _read_paths(paths)
    docs += file_docs
    if not docs:
        raise ValueError("nothing to add: pass documents and/or readable paths" + (f" ({'; '.join(errors)})" if errors else ""))
    if len(docs) > MAX_DOCS_PER_CALL:
        raise ValueError(f"{len(docs)} documents in one call; send at most {MAX_DOCS_PER_CALL} per call")
    c = _get(collection)
    stats = await asyncio.to_thread(c.add_documents, docs, 900, not wait)
    out = {"ingested": stats, "errors": errors, "collection": c.stats(), "warnings": _warnings(c)}
    if not wait and c.stats().get("chunks_pending_extraction"):
        out["next_step"] = ("Documents are stored and searchable now. The knowledge graph is being built in the "
                            "background; call list with this collection until chunks_pending_extraction is gone "
                            "before relying on multi-hop answers.")
    return json.dumps(out)


@mcp.tool(name="list")
@_tool
async def list_collections(collection: str | None = None, limit: int = 50, offset: int = 0) -> str:
    """List collections, or show what is inside one.

    Without `collection`: every collection with its description and size
    (documents, chunks, entities, relations). With `collection`: its documents
    (newest first, paginated), its most-mentioned entities with their merged
    aliases, and its best-supported relations. Use this before creating a
    collection or adding documents, to avoid duplicates.

    Args:
        collection: Optional collection name to inspect.
        limit: Documents per page (1-200).
        offset: Documents to skip, for paging.
    """
    if collection:
        c = _get(collection)
        limit = max(1, min(int(limit), 200))
        contents = await asyncio.to_thread(c.contents, limit, max(0, int(offset)))
        return json.dumps({"collection": c.stats(), **contents, "warnings": _warnings(c)})
    out = []
    for n in Corpus.list_names():
        try:
            out.append(_get(n).stats())
        except Exception as e:  # a corrupt or foreign file must not hide the others
            log.warning("cannot open collection %s: %s", n, e)
            out.append({"name": n, "error": str(e)})
    return json.dumps({"collections": out, "data_dir": str(store.home())})


@mcp.tool(name="query")
@_tool
async def query(collection: str, query: str, k: int = 5, mode: str = "hybrid", fusion: str | None = None,
                params: dict | None = None) -> str:
    """Search a collection. Hybrid mode combines dense vector similarity with
    traversal of the entity-relation graph, so it also finds passages connected to
    the query through shared entities (multi-hop: "where is the company founded by
    X based?") even when they share no words with the query.

    Returns ranked chunks with text, source, dense and graph scores and mentioned
    entities, plus graph_context: the entities reached and the relation paths that
    led there.

    Args:
        collection: Collection name.
        query: Natural-language question or keywords. Name specific entities when you can.
        k: Number of chunks to return (1-50).
        mode: "hybrid" (default), "dense" (vector similarity only) or "graph" (traversal only).
        fusion: How hybrid mode merges the two rankings: "weighted" (min-max normalized
            alpha * dense + (1 - alpha) * graph) or "rrf" (reciprocal rank fusion).
            Defaults to the tuned setting.
        params: Optional retrieval overrides, e.g. {"alpha": 0.3, "hops": 3, "decay": 0.6}.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if fusion is not None and fusion not in FUSIONS:
        raise ValueError(f"fusion must be one of {FUSIONS}")
    overrides = dict(params or {})
    unknown = set(overrides) - set(RetrievalParams.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown params {sorted(unknown)}; allowed: {sorted(RetrievalParams.__dataclass_fields__)}")
    try:
        p = RetrievalParams.from_dict({**_default_params(), **overrides})
    except (TypeError, ValueError) as e:
        raise ValueError(f"bad params: {e}") from e
    p.k, p.mode = max(1, min(int(k), 50)), mode
    if fusion:
        p.fusion = fusion
    c = _get(collection)
    res = await asyncio.to_thread(retrieve, c, query, p)
    res.pop("ranked_chunk_ids", None)
    res["settings"] = {"mode": p.mode, "fusion": p.fusion, "alpha": p.alpha, "hops": p.hops}
    res.pop("params", None)
    return json.dumps(res)


def main() -> None:
    origin = load_env_file()  # before anything reads KGRAG_* settings
    configure(log_file=store.home() / "kgrag.log")
    log.info("starting kgrag MCP server on stdio (data: %s, retrieval params: %s)",
             store.home(), "tuned" if _default_params() else "built-in defaults")
    if origin:
        log.info("settings file %s: %s", env_file_path(),
                 ", ".join(f"{k} from {v}" for k, v in sorted(origin.items())))
    log.info("extractor=%s embedder=%s (used for new collections)",
             os.environ.get("KGRAG_EXTRACTOR") or ("anthropic default" if os.environ.get("ANTHROPIC_API_KEY") else "heuristic"),
             os.environ.get("KGRAG_EMBEDDER") or "auto")
    mcp.run()


if __name__ == "__main__":
    main()
