"""Exercise the LLM extraction path against a local fake Anthropic / OpenAI server:
request format, response parsing, retries, fail-fast, rollback, and retry of failed chunks.
This checks the plumbing, not the model's extraction quality."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from kgrag.corpus import Corpus
from kgrag.extract import LLMError, LLMExtractor
from kgrag.retrieve import retrieve

GOOD = {
    "entities": [
        {"name": "Dr. Mira Okonkwo", "type": "person", "description": "Founder of Velantis."},
        {"name": "Velantis Corp.", "type": "organization", "description": "Robotics company."},
    ],
    "relations": [{"source": "Velantis Corp.", "target": "Dr. Mira Okonkwo", "relation": "was founded by", "description": ""}],
}


class Fake:
    def __init__(self):
        self.plan: list = []      # per-request behaviour: "ok", "429", "500", "404", "garbage"
        self.requests: list = []

    def handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append((self.path, dict(self.headers), body))
                mode = fake.plan.pop(0) if fake.plan else "ok"
                if mode in ("429", "500", "404"):
                    self.send_response(int(mode))
                    self.send_header("retry-after", "0")
                    self.end_headers()
                    self.wfile.write(b'{"error":{"message":"nope"}}')
                    return
                text = "sorry, not json" if mode == "garbage" else "```json\n" + json.dumps(GOOD) + "\n```"
                if self.path.endswith("/v1/messages"):
                    out = {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}
                else:
                    out = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return H


@pytest.fixture
def fake(monkeypatch, tmp_path):
    f = Fake()
    srv = HTTPServer(("127.0.0.1", 0), f.handler())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", base + "/v1")
    monkeypatch.setenv("KGRAG_HOME", str(tmp_path))
    monkeypatch.setenv("KGRAG_EMBEDDER", "hash")
    monkeypatch.setattr("kgrag.extract.time.sleep", lambda s: None)
    monkeypatch.setattr("urllib.request.getproxies", lambda: {})  # talk to localhost directly
    yield f
    srv.shutdown()


def test_anthropic_request_and_parse(fake):
    ex = LLMExtractor("anthropic", "claude-haiku-5-5").extract("Dr. Mira Okonkwo founded Velantis Corp.")
    path, headers, body = fake.requests[0]
    headers = {k.lower(): v for k, v in headers.items()}
    assert path == "/v1/messages"
    assert headers["x-api-key"] == "test-key" and headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-haiku-5-5" and "temperature" not in body
    assert [e.name for e in ex.entities] == ["Dr. Mira Okonkwo", "Velantis Corp."]
    assert ex.relations[0].relation == "was founded by"


def test_openai_compatible_path(fake):
    ex = LLMExtractor("openai", "llama3.1").extract("text")
    assert fake.requests[0][0] == "/v1/chat/completions"
    assert len(ex.entities) == 2


def test_retries_rate_limit_then_succeeds(fake):
    fake.plan = ["429", "500", "ok"]
    ex = LLMExtractor("anthropic", "m").extract("text")
    assert len(fake.requests) == 3 and len(ex.entities) == 2


def test_bad_model_fails_fast_with_body(fake):
    fake.plan = ["404"]
    with pytest.raises(LLMError, match="HTTP 404.*nope"):
        LLMExtractor("anthropic", "no-such-model").extract("text")
    assert len(fake.requests) == 1  # 404 is not retried


def test_ingest_with_llm_builds_resolved_graph(fake):
    c = Corpus.create("llm", extractor="anthropic:claude-haiku-5-5")
    st = c.add_documents([{"title": "a", "text": "Dr. Mira Okonkwo founded Velantis Corp."}])
    assert st["entities_created"] == 2
    rel = c.db.execute("SELECT s.name, r.label, d.name FROM relations r JOIN entities s ON s.id=r.src "
                       "JOIN entities d ON d.id=r.dst").fetchone()
    assert rel == ("Dr. Mira Okonkwo", "founded", "Velantis Corp.")


def test_provider_error_keeps_documents_searchable_and_pending(fake):
    c = Corpus.create("rb", extractor="anthropic:m")
    fake.plan = ["404"]
    with pytest.raises(LLMError, match="HTTP 404"):
        c.add_documents([{"title": "a", "text": "Dr. Mira Okonkwo founded Velantis Corp."}])
    st = c.stats()
    assert st["documents"] == 1 and st["chunks_pending_extraction"] == 1 and st["entities"] == 0
    assert retrieve(c, "Velantis", {"mode": "dense"})["results"][0]["title"] == "a"  # dense works already
    # once the provider works, the next add (even a duplicate) builds the pending graph
    st2 = c.add_documents([{"title": "a", "text": "Dr. Mira Okonkwo founded Velantis Corp."}])
    assert st2["skipped_duplicate"] == 1 and st2["chunks_extracted"] == 1
    assert c.stats()["entities"] == 2 and "chunks_pending_extraction" not in c.stats()


def test_transient_failure_is_retried_on_next_add(fake, monkeypatch):
    monkeypatch.setenv("KGRAG_WORKERS", "1")
    c = Corpus.create("retry", extractor="anthropic:m")
    docs = [{"title": f"d{i}", "text": f"Document number {i} about Velantis Corp."} for i in range(8)]
    fake.plan = ["ok"] + ["500"] * 6  # chunk 2 exhausts its retries on server errors
    st = c.add_documents(docs)
    assert st["extraction_failed"] == 1
    assert c.stats()["chunks_pending_extraction"] == 1
    st2 = c.add_documents([{"title": "new", "text": "Another note."}])
    assert st2["chunks_extracted"] == 2  # the earlier failure plus the new chunk
    assert "chunks_pending_extraction" not in c.stats()


def test_unusable_output_does_not_abort_and_is_not_retried(fake, monkeypatch):
    monkeypatch.setenv("KGRAG_WORKERS", "1")
    c = Corpus.create("bad", extractor="anthropic:m")
    docs = [{"title": f"d{i}", "text": f"Paragraph {i} about Velantis Corp."} for i in range(6)]
    fake.plan = ["garbage"]  # the very first (preflight) chunk returns non-JSON
    st = c.add_documents(docs)
    assert st["documents"] == 6 and st["extraction_unusable_output"] == 1
    assert "chunks_pending_extraction" not in c.stats()  # not queued for pointless retries
    n_before = len(fake.requests)
    c.add_documents([{"title": "more", "text": "Another note."}])
    assert len(fake.requests) == n_before + 1  # only the new chunk was sent


def test_truncated_json_is_recovered():
    from kgrag.extract import _repair_truncated

    full = json.dumps(GOOD)
    cut = full[: full.index('"relations"') + 40]  # cut inside the relations list
    obj = _repair_truncated(cut)
    assert [e["name"] for e in obj["entities"]] == ["Dr. Mira Okonkwo", "Velantis Corp."]
    cut2 = full[: full.index("Velantis Corp.") + 5]  # cut inside the second entity's name
    obj2 = _repair_truncated(cut2)
    assert [e["name"] for e in obj2["entities"]] == ["Dr. Mira Okonkwo"]
    assert _repair_truncated("sorry, no json here") is None


def test_background_add_returns_before_extraction_and_finishes(fake, monkeypatch):
    import threading
    import time as _t

    from kgrag import extract

    release = threading.Event()  # the fixture stubs time.sleep, so block on an event instead
    orig = extract.LLMExtractor.extract
    monkeypatch.setattr(extract.LLMExtractor, "extract", lambda self, text: (release.wait(10), orig(self, text))[1])
    c = Corpus.create("bg", extractor="anthropic:m")
    t0 = _t.time()
    st = c.add_documents([{"title": f"d{i}", "text": f"Dr. Mira Okonkwo founded Velantis Corp. Note {i}."}
                          for i in range(4)], background=True)
    assert _t.time() - t0 < 2 and st["documents"] == 4 and st["graph_extraction"] == "running in background"
    s1 = c.stats()
    assert s1["chunks_pending_extraction"] == 4 and s1["extraction_in_progress"] is True and s1["entities"] == 0
    assert retrieve(c, "Velantis", {"mode": "dense"})["results"]  # searchable before the graph exists
    release.set()
    Corpus._workers[str(c.path)].join(timeout=30)
    st = c.stats()
    assert "chunks_pending_extraction" not in st and st["entities"] == 2 and st["relations"] == 1
