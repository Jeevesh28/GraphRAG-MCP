import pytest

from kgrag.corpus import Corpus
from kgrag.resolve import EntityResolver, canonical_relation
from kgrag.retrieve import retrieve
from kgrag.text import acronym, chunk_text, normalize_name, split_sentences


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("KGRAG_HOME", str(tmp_path))
    monkeypatch.setenv("KGRAG_EXTRACTOR", "heuristic")
    monkeypatch.setenv("KGRAG_EMBEDDER", "hash")


# ------------------------------------------------------------------ text
def test_sentence_split_keeps_abbreviations():
    s = split_sentences("Dr. Mira Okonkwo founded Velantis Corp. in 2011. It grew fast. M. Okonkwo left.")
    assert s == ["Dr. Mira Okonkwo founded Velantis Corp. in 2011.", "It grew fast.", "M. Okonkwo left."]


def test_normalize_and_acronym():
    assert normalize_name("Dr. Mira Okonkwo") == "mira okonkwo"
    assert normalize_name("Velantis Corp.") == normalize_name("Velantis Corporation") == "velantis"
    assert acronym("Blue Harbor Systems") == "bhs"


def test_chunking_respects_size():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, max_chars=300)
    assert len(chunks) > 5 and all(len(c) <= 330 for c in chunks)


# ------------------------------------------------------------ resolution
def _resolve_all(names_types):
    r = EntityResolver("full")
    ids, nxt = [], 1
    for n, t in names_types:
        eid, created = r.resolve(n, t, nxt)
        nxt += created
        ids.append(eid)
    return ids


def test_merges_surface_variants():
    ids = _resolve_all([("Dr. Mira Okonkwo", "person"), ("Mira Okonkwo", "other"), ("M. Okonkwo", "person"),
                        ("Velantis Corp.", "organization"), ("Velantis", "other"), ("Velantis Corporation", "organization"),
                        ("Blue Harbor Systems", "organization"), ("BHS", "other")])
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] == ids[4] == ids[5]
    assert ids[6] == ids[7]
    assert len(set(ids)) == 3


def test_first_name_veto_blocks_transitive_merge():
    ids = _resolve_all([("Jane Smith", "person"), ("J. Smith", "person"), ("John Smith", "person")])
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]


def test_ambiguous_initial_does_not_merge():
    ids = _resolve_all([("Jane Smith", "person"), ("Jack Smith", "person"), ("J. Smith", "person")])
    assert ids[2] not in (ids[0], ids[1])


def test_type_conflict_blocks_merge():
    ids = _resolve_all([("Jordan", "person"), ("Jordan", "location")])
    assert ids[0] != ids[1]


def test_passive_relation_is_flipped():
    assert canonical_relation(2, "was founded by", 1) == (1, "founded", 2)
    assert canonical_relation(1, "Headquartered in", 3) == (1, "headquartered in", 3)


# ------------------------------------------------------- corpus + retrieval
DOCS = [
    {"title": "d1", "text": "Dr. Mira Okonkwo founded Velantis Corp. in 2011."},
    {"title": "d2", "text": "Velantis is headquartered in Port Ellery."},
    {"title": "d3", "text": "Tunde Okonkwo founded Graymoor Labs."},
    {"title": "d4", "text": "Graymoor Labs is headquartered in Castlebrook."},
    {"title": "d5", "text": "Blue Harbor Systems was acquired by Velantis Corporation."},
    {"title": "d6", "text": "Kessler Dynamics is headquartered in Northwick."},
    {"title": "d7", "text": "Arno Weiss founded Kessler Dynamics."},
]


def test_graph_dedup_and_multihop_retrieval():
    c = Corpus.create("t")
    st = c.add_documents(DOCS)
    assert st["documents"] == 7
    names = {n for (n,) in c.db.execute("SELECT name FROM entities")}
    assert "Velantis" in names and "Velantis Corp." not in names
    # passive "was acquired by" became an active edge from Velantis
    rel = c.db.execute("SELECT s.name, r.label, d.name FROM relations r JOIN entities s ON s.id=r.src "
                       "JOIN entities d ON d.id=r.dst WHERE r.label='acquired'").fetchone()
    assert rel == ("Velantis", "acquired", "Blue Harbor Systems")

    q = "In which city is the company founded by Mira Okonkwo headquartered?"
    hybrid = [r["title"] for r in retrieve(c, q, {"k": 3})["results"]]
    assert "d2" in hybrid  # the bridge doc never mentions Mira
    out = retrieve(c, q, {"k": 3})
    assert any("headquartered in" in " ".join(g["path"]) for g in out["graph_context"])


def test_duplicate_documents_are_skipped_and_incremental_add_links():
    c = Corpus.create("t2")
    c.add_documents(DOCS[:2])
    st = c.add_documents([DOCS[0], {"title": "d8", "text": "M. Okonkwo studied at Harlow Institute."}])
    assert st["skipped_duplicate"] == 1
    # the new abbreviated mention attached to the existing node
    n = c.db.execute("SELECT COUNT(*) FROM entities WHERE name LIKE '%Okonkwo%'").fetchone()[0]
    assert n == 1


def test_resolution_ablation_creates_duplicates():
    full = Corpus.create("full", resolution="full")
    none = Corpus.create("none", resolution="none")
    full.add_documents(DOCS)
    none.add_documents(DOCS)
    assert none.stats()["entities"] > full.stats()["entities"]


def test_corpus_name_validation():
    with pytest.raises(ValueError):
        Corpus.create("../escape")


def test_server_loads_tuned_params(tmp_path, monkeypatch):
    import json as _json

    from kgrag import server

    (tmp_path / "retrieval_params.json").write_text(_json.dumps({"alpha": 0.15, "hops": 3, "k": 99, "mode": "graph"}))
    p = server._default_params()
    assert p == {"alpha": 0.15, "hops": 3}  # k and mode always come from the tool call


def test_server_resumes_pending_extraction_after_restart(monkeypatch):
    from kgrag import server

    c = Corpus.create("crash")
    c._store([{"title": "d1", "text": "Dr. Mira Okonkwo founded Velantis Corp."}], 900)  # stored, then "crashed"
    assert c.stats()["chunks_pending_extraction"] == 1
    monkeypatch.setattr(server, "_open", {})  # a fresh server process
    reopened = server._get("crash")
    Corpus._workers[str(reopened.path)].join(timeout=30)
    st = reopened.stats()
    assert "chunks_pending_extraction" not in st and st["entities"] == 2


def test_graph_context_paths_trace_from_query_entities():
    c = Corpus.create("paths")
    c.add_documents([{"text": "Dr. Mira Okonkwo founded Velantis Corp."},
                     {"text": "Blue Harbor Systems was acquired by Velantis Corporation."},
                     {"text": "BHS builds marine sensors."}])
    # semantic seeding of every entity (as tuned params can do in a tiny collection)
    out = retrieve(c, "What does the company acquired by the firm Mira Okonkwo founded make?",
                   {"seed_semantic_k": 50, "seed_semantic_min": -1.0})
    paths = {g["entity"]: g["path"] for g in out["graph_context"]}
    hops = paths["Blue Harbor Systems"]
    assert len(hops) == 2
    assert hops[0].startswith("Dr. Mira Okonkwo --founded--> Velantis")
    assert "--acquired--> Blue Harbor Systems" in hops[1]


def test_settings_file_fills_missing_env_only(tmp_path, monkeypatch):
    from kgrag.settings import load_env_file

    f = tmp_path / "kgrag.env"
    f.write_text("# comment\nKGRAG_EXTRACTOR=openai:qwen2.5:7b\nexport OPENAI_BASE_URL=\"http://localhost:11434/v1\"\n"
                 "KGRAG_EMBEDDER=st:x\n")
    monkeypatch.delenv("KGRAG_EXTRACTOR", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("KGRAG_EMBEDDER", "hash")  # set by the client: must win
    origin = load_env_file(f)
    import os
    assert os.environ["KGRAG_EXTRACTOR"] == "openai:qwen2.5:7b"
    assert os.environ["OPENAI_BASE_URL"] == "http://localhost:11434/v1"
    assert os.environ["KGRAG_EMBEDDER"] == "hash"
    assert origin == {"KGRAG_EXTRACTOR": "file", "OPENAI_BASE_URL": "file", "KGRAG_EMBEDDER": "environment"}
