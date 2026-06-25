"""End-to-end: spawn the server over stdio and drive it with the MCP client,
exactly as Claude Desktop or Cursor would."""

import json
import os
import sys

import anyio
import pytest

pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

SRC = os.path.join(os.path.dirname(__file__), "..", "src")


def _is_error(result) -> bool:  # mcp 2.x: is_error, mcp 1.x: isError
    return bool(getattr(result, "is_error", None) or getattr(result, "isError", None))


def _json(result) -> dict:
    assert not _is_error(result), result.content[0].text
    return json.loads(result.content[0].text)


def _params(home: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable, args=["-m", "kgrag.server"],
        env={"KGRAG_HOME": home, "KGRAG_EXTRACTOR": "heuristic", "KGRAG_EMBEDDER": "hash", "PYTHONPATH": SRC},
    )


async def _roundtrip(home: str, note_path: str):
    async with stdio_client(_params(home)) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            tools = {t.name for t in (await s.list_tools()).tools}
            assert tools == {"create", "add", "list", "query"}

            created = _json(await s.call_tool("create", {"name": "companies", "description": "Founders and HQs"}))
            assert created["warnings"]  # heuristic + hash are flagged to the agent

            add = _json(await s.call_tool("add", {
                "collection": "companies",
                "documents": [
                    {"title": "d1", "text": "Dr. Mira Okonkwo founded Velantis Corp. in 2011."},
                    {"title": "d2", "text": "Velantis is headquartered in Port Ellery."},
                    {"title": "d3", "text": "Blue Harbor Systems was acquired by Velantis Corporation."},
                    "BHS builds marine sensors.",
                ],
                "paths": [note_path],
                "wait": True,
            }))
            assert add["ingested"]["documents"] == 5 and not add["errors"]

            # default: returns immediately, graph is built in the background
            bg = _json(await s.call_tool("add", {"collection": "companies",
                                                  "documents": ["Graymoor Labs is headquartered in Castlebrook."]}))
            assert bg["ingested"]["graph_extraction"] == "running in background"
            for _ in range(100):
                info = _json(await s.call_tool("list", {"collection": "companies"}))
                if "chunks_pending_extraction" not in info["collection"]:
                    break
                await anyio.sleep(0.1)
            assert info["collection"]["documents"] == 6 and "chunks_pending_extraction" not in info["collection"]

            listed = _json(await s.call_tool("list", {}))
            assert [c["name"] for c in listed["collections"]] == ["companies"]
            inside = _json(await s.call_tool("list", {"collection": "companies"}))
            assert inside["documents_total"] == 6
            velantis = [e for e in inside["top_entities"] if e["name"] == "Velantis"]
            assert velantis and set(velantis[0]["aliases"]) >= {"Velantis", "Velantis Corp."}

            q = _json(await s.call_tool("query", {
                "collection": "companies", "query": "Where is the company Mira Okonkwo founded based?", "k": 3}))
            assert "d2" in [r["title"] for r in q["results"]]
            assert q["graph_context"] and q["settings"]["mode"] == "hybrid"
            q2 = _json(await s.call_tool("query", {"collection": "companies", "query": "marine sensors",
                                                    "fusion": "rrf", "params": {"hops": 3}}))
            assert q2["settings"]["fusion"] == "rrf" and q2["settings"]["hops"] == 3

            # errors come back as tool errors with actionable messages, and the server keeps running
            bad = await s.call_tool("query", {"collection": "nope", "query": "x"})
            assert _is_error(bad) and "does not exist" in bad.content[0].text and "companies" in bad.content[0].text
            bad = await s.call_tool("query", {"collection": "companies", "query": "x", "params": {"bogus": 1}})
            assert _is_error(bad) and "unknown params" in bad.content[0].text
            bad = await s.call_tool("add", {"collection": "companies", "documents": [{"title": "no text"}]})
            assert _is_error(bad) and "documents[0]" in bad.content[0].text
            bad = await s.call_tool("create", {"name": "companies"})
            assert _is_error(bad) and "already exists" in bad.content[0].text
            assert _json(await s.call_tool("list", {}))["collections"][0]["documents"] == 6


def test_stdio_roundtrip(tmp_path):
    note = tmp_path / "note.md"
    note.write_text("Kessler Dynamics is headquartered in Northwick.")
    anyio.run(_roundtrip, str(tmp_path / "home"), str(note))
    log = (tmp_path / "home" / "kgrag.log").read_text()
    assert "tool add ok" in log and "tool query failed" in log


async def _settings_file_roundtrip(home: str):
    params = StdioServerParameters(command=sys.executable, args=["-m", "kgrag.server"],
                                   env={"KGRAG_HOME": home, "PYTHONPATH": SRC})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            created = _json(await s.call_tool("create", {"name": "fromfile"}))
            assert created["created"]["extractor"] == "heuristic"
            assert created["created"]["embedder"].startswith("hash")


def test_server_reads_settings_file(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "kgrag.env").write_text("KGRAG_EXTRACTOR=heuristic\nKGRAG_EMBEDDER=hash:512\n")
    anyio.run(_settings_file_roundtrip, str(home))
    assert "KGRAG_EXTRACTOR from file" in (home / "kgrag.log").read_text()
