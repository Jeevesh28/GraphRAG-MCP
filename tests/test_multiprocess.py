"""Claude Desktop and Cursor each launch their own server process against the same
data directory. These tests run two real processes against one collection."""

import multiprocessing as mp
import os
import time

import pytest

from kgrag.corpus import Corpus
from kgrag.retrieve import retrieve


def _slow_add(home: str, docs: list[dict], barrier) -> dict:
    os.environ.update(KGRAG_HOME=home, KGRAG_EXTRACTOR="heuristic", KGRAG_EMBEDDER="hash")
    from kgrag import extract

    orig = extract.HeuristicExtractor.extract

    def slow(self, text):  # make extraction slow so the two ingests overlap
        time.sleep(0.2)
        return orig(self, text)

    extract.HeuristicExtractor.extract = slow
    c = Corpus.open("shared")
    barrier.wait()
    return c.add_documents(docs)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("KGRAG_HOME", str(tmp_path))
    monkeypatch.setenv("KGRAG_EXTRACTOR", "heuristic")
    monkeypatch.setenv("KGRAG_EMBEDDER", "hash")
    Corpus.create("shared")
    return str(tmp_path)


def test_concurrent_adds_from_two_processes(home):
    a = [{"title": f"a{i}", "text": f"Velantis Corp. opened office number {i} in Port Ellery."} for i in range(4)]
    b = [{"title": f"b{i}", "text": f"Dr. Mira Okonkwo spoke about Velantis at event {i}."} for i in range(4)]
    dup = {"title": "same", "text": "Velantis is headquartered in Port Ellery."}
    ctx = mp.get_context("spawn")
    with ctx.Manager() as m:
        barrier = m.Barrier(2)
        with ctx.Pool(2) as pool:
            ra = pool.apply_async(_slow_add, (home, a + [dup], barrier))
            rb = pool.apply_async(_slow_add, (home, b + [dup], barrier))
            sa, sb = ra.get(timeout=120), rb.get(timeout=120)
    # both ingests succeeded, and the document both sent was stored once
    assert sa["documents"] + sb["documents"] == 9
    assert sa.get("skipped_duplicate", 0) + sb.get("skipped_duplicate", 0) == 1
    c = Corpus.open("shared")
    assert c.stats()["documents"] == 9
    # the entity both processes mentioned is a single node, not one per process
    rows = c.db.execute("SELECT e.id FROM entities e JOIN aliases a ON a.entity_id=e.id "
                        "WHERE a.alias LIKE 'Velantis%' GROUP BY e.id").fetchall()
    assert len(rows) == 1
    ids = [r[0] for r in c.db.execute("SELECT id FROM entities")]
    assert len(ids) == len(set(ids))


def test_index_sees_writes_from_another_process(home):
    reader = Corpus.open("shared")  # e.g. Claude Desktop's server
    writer = Corpus.open("shared")  # e.g. Cursor's server (separate connection and cache)
    writer.add_documents([{"title": "first", "text": "Graymoor Labs is headquartered in Castlebrook."}])
    assert retrieve(reader, "Graymoor Labs", {"k": 5})["results"][0]["title"] == "first"
    writer.add_documents([{"title": "second", "text": "Kessler Dynamics is headquartered in Northwick."}])
    titles = [r["title"] for r in retrieve(reader, "Kessler Dynamics Northwick", {"k": 5})["results"]]
    assert titles[0] == "second"  # no restart needed


def test_contents_lists_documents_entities_relations(home):
    c = Corpus.open("shared")
    c.add_documents([{"title": "d1", "text": "Dr. Mira Okonkwo founded Velantis Corp."},
                     {"title": "d2", "text": "M. Okonkwo founded Velantis."}])
    out = c.contents(limit=1)
    assert out["documents_total"] == 2 and len(out["documents"]) == 1 and out["documents"][0]["title"] == "d2"
    mira = [e for e in out["top_entities"] if "Okonkwo" in e["name"]][0]
    assert set(mira["aliases"]) == {"Dr. Mira Okonkwo", "M. Okonkwo"} and mira["mentions"] == 2
    assert out["top_relations"][0]["support"] == 2  # the founded edge, deduplicated across both docs
