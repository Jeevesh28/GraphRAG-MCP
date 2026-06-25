# Knowledge Graph RAG Server for Agents (MCP)

An MCP server that lets agents build and search their own knowledge collections. The agent decides what to collect (`create`, `add`), checks what it has (`list`), and searches with `query`, which fuses dense vector similarity with traversal over an LLM-extracted entity-relation graph. Entity resolution keeps traversal on real relations instead of duplicate nodes.

Runs over stdio and has been tested with Claude Desktop and Cursor, using a local open-source model (Qwen 2.5 7B through Ollama) for extraction.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install ".[st,dev]"
pytest -q
ollama pull qwen2.5:7b
mkdir -p ~/.kgrag && cat > ~/.kgrag/kgrag.env <<'EOT'
KGRAG_EXTRACTOR=openai:qwen2.5:7b
KGRAG_EMBEDDER=st:BAAI/bge-small-en-v1.5
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
KGRAG_WORKERS=2
EOT
```

Register the server with this entry in either client:

```json
{ "mcpServers": { "kgrag": { "command": "/absolute/path/to/.venv/bin/kgrag-mcp" } } }
```

- **Claude Desktop**: `claude_desktop_config.json` (Settings > Developer > Edit Config)
- **Cursor**: `~/.cursor/mcp.json`

Restart the client and ask the agent to use it. Logs are in `~/.kgrag/kgrag.log`.

---

## Architecture

```
 Claude / Cursor ── stdio ──► kgrag-mcp ─┬─ add ──► chunk ─► embed ─► store ──► SQLite (one file per collection)
                                         │                             │
                                         │        background worker ◄──┘
                                         │   LLM extract ─► entity resolution ─► relation dedup
                                         │
                                         └─ query ─► dense top-N ─► link query entities ─► spreading activation ─► fuse ─► chunks + graph paths
```

- **Ingestion**: `add` stores, chunks and embeds synchronously. A background thread then extracts, resolves and commits.
- **Retrieval**: seeds entities from the query, spreads activation over the graph, and fuses with dense similarity. Results include the relation paths behind each hit.

---

## Key decisions

- **Merge guards**: a first-name veto stops "J. Smith" from merging "Jane Smith" and "John Smith".
- **Two-stage ingest**: documents are searchable before the LLM runs, since a local 7B model can take minutes.
- **Query-steered traversal**: edges are weighted by how well their relation label matches the query.
- **Recoverable failures**: LLM calls retry with backoff, and pending chunks are retried on the next add or restart.

---

## Tools and settings

| Tool | Example | Behavior |
|---|---|---|
| `create` | `{"name": "rust-async", "description": "..."}` | New empty collection |
| `add` | `{"collection": "rust-async", "documents": ["..."]}` | Stores now, builds the graph in the background (`wait: true` to block) |
| `add` | `{"collection": "rust-async", "paths": ["~/notes/tokio.md"]}` | Reads local text files |
| `list` | `{}` or `{"collection": "rust-async"}` | Collections, or a collection's entities, relations and progress |
| `query` | `{"collection": "rust-async", "query": "..."}` | Hybrid retrieval with scores and graph paths (`"mode": "dense"` or `"graph"` for one signal) |

| Setting | Default |
|---|---|
| `KGRAG_EXTRACTOR`: `openai:<model>`, `anthropic:<model>`, `heuristic` | Anthropic if `ANTHROPIC_API_KEY` is set, else `heuristic` |
| `KGRAG_EMBEDDER`: `st:<model>`, `openai:<model>`, `hash` | `st:BAAI/bge-small-en-v1.5` when installed |
| `KGRAG_PARAMS`, `KGRAG_WORKERS`, `KGRAG_LLM_TIMEOUT`, `KGRAG_MAX_TOKENS`, `KGRAG_DEVICE` | `~/.kgrag/retrieval_params.json`, `8`, `300` s, `2048`, `cpu` |

---

## Tests and results

31 tests cover normalization, resolution, multi-hop retrieval, concurrent writers, background extraction, the LLM path, and a full stdio session.

`kgrag-eval` reports nDCG@10 on a 50/50 dev/test split. Hybrid settings are tuned on dev, and every number below is on test.

**Synthetic multi-hop** (299 documents, 100 test questions):

| System | nDCG@10 | All evidence in top 10 |
|---|---|---|
| Dense only | 0.383 | 0.22 |
| Hybrid, entity resolution off | 0.444 | 0.42 |
| Hybrid, entity resolution on | **0.642** | **0.86** |

**HotpotQA distractor dev** (100 questions, 50 test):

| System | nDCG@10 | All evidence in top 10 |
|---|---|---|
| Dense only | 0.880 | 0.94 |
| Hybrid, tuned | 0.884 | 0.94 |

The HotpotQA gain is not significant (+0.004 [-0.022, +0.030]). The graph helps when answers depend on entity links that text similarity misses.

Reproduce:

```bash
kgrag-eval --dataset synthetic --seed 1 --embedder st:BAAI/bge-small-en-v1.5 --extractor openai:qwen2.5:7b
```
