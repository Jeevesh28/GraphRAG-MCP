"""Corpus: ingestion pipeline (chunk -> embed -> extract -> resolve -> graph) and the
in-memory index used for retrieval."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import store
from .embed import get_embedder
from .extract import ExtractionCache, extract_many, get_extractor
from .resolve import EntityRecord, EntityResolver, canonical_relation
from .logs import get_logger
from .text import acronym, chunk_text, normalize_name

log = get_logger("corpus")


def _canonical_name(aliases: list[str], counts: Counter) -> str:
    """Prefer the fullest, non-abbreviated surface form; break ties by frequency."""
    def key(a: str):
        toks = a.split()
        abbreviated = any(len(t.rstrip(".")) == 1 for t in toks) or (len(toks) == 1 and a.isupper())
        return (not abbreviated, len(normalize_name(a).split()), counts[a], -len(a))
    return max(aliases, key=key)


@dataclass
class Index:
    """Everything retrieval needs, held in memory as numpy arrays and dicts."""
    version: int
    chunk_ids: np.ndarray
    chunk_emb: np.ndarray
    chunk_doc: np.ndarray
    ent_ids: np.ndarray
    ent_emb: np.ndarray
    ent_names: list[str]
    ent_row: dict[int, int]
    alias_norm: dict[str, set[int]]  # normalized alias -> entity ids
    alias_loose: dict[str, set[int]]  # "m okonkwo" and acronym keys -> entity ids (weaker links)
    ent_chunks: dict[int, list[int]]
    chunk_ents: dict[int, list[int]]
    adj: dict[int, list[tuple[int, int, float]]]  # eid -> [(nbr, rel_row, weight)]
    rel_ids: np.ndarray
    rel_emb: np.ndarray
    rel_triples: list[tuple[int, str, int]]
    rel_evidence: dict[int, list[int]] = field(default_factory=dict)  # rel_row -> chunk ids


class Corpus:
    _locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem
        self.db = store.connect(path)
        meta = dict(self.db.execute("SELECT k, v FROM meta").fetchall())
        self.meta = meta
        self.embedder = get_embedder(meta.get("embedder"))
        self._extractor = None
        self._index: Index | None = None

    # ---------------------------------------------------------------- lifecycle
    @classmethod
    def create(cls, name: str, description: str = "", root: Path | None = None, embedder: str | None = None,
               extractor: str | None = None, resolution: str = "full", exist_ok: bool = False) -> "Corpus":
        path = store.corpus_path(name, root)
        if path.exists() and not exist_ok:
            raise ValueError(f"collection {name!r} already exists; add to it, or pick another name")
        emb = get_embedder(embedder)
        ex = get_extractor(extractor)
        db = store.connect(path)
        for k, v in {"description": description, "embedder": emb.spec, "dim": str(emb.dim),
                     "extractor": ex.spec, "resolution": resolution, "version": "0"}.items():
            db.execute("INSERT OR IGNORE INTO meta VALUES (?, ?)", (k, v))
        db.commit()
        db.close()
        return cls(path)

    @classmethod
    def open(cls, name: str, root: Path | None = None) -> "Corpus":
        path = store.corpus_path(name, root)
        if not path.exists():
            have = cls.list_names(root)
            raise ValueError(f"collection {name!r} does not exist. Existing: {', '.join(have) or 'none'}. "
                             f"Create it first.")
        return cls(path)

    @staticmethod
    def list_names(root: Path | None = None) -> list[str]:
        d = (root or store.home()) / "corpora"
        return sorted(p.stem for p in d.glob("*.sqlite")) if d.exists() else []

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = get_extractor(self.meta.get("extractor"))
        return self._extractor

    def stats(self) -> dict:
        db = store.connect_ro(self.path)
        q = lambda s: db.execute(s).fetchone()[0]  # noqa: E731
        out = {
            "name": self.name,
            "description": self.meta.get("description", ""),
            "documents": q("SELECT COUNT(*) FROM docs"),
            "chunks": q("SELECT COUNT(*) FROM chunks"),
            "entities": q("SELECT COUNT(*) FROM entities"),
            "relations": q("SELECT COUNT(*) FROM relations"),
            "embedder": self.meta.get("embedder"),
            "extractor": self.meta.get("extractor"),
            "resolution": self.meta.get("resolution"),
        }
        pending = q("SELECT COUNT(*) FROM chunks WHERE extracted=0")
        if pending:
            out["chunks_pending_extraction"] = pending
            out["extraction_in_progress"] = self.extraction_running()
        err = db.execute("SELECT v FROM meta WHERE k='last_extraction_error'").fetchone()
        if err:
            out["last_extraction_error"] = err[0]
        db.close()
        return out

    # ---------------------------------------------------------------- ingestion
    def add_documents(self, docs: list[dict], chunk_chars: int = 900, background: bool = False) -> dict:
        """Add documents. docs: [{"text": str, "title"?: str, "source"?: str, "metadata"?: dict}]

        Stage 1 (always synchronous, a few seconds): dedup, chunk, embed, store. The
        documents are searchable by dense similarity as soon as this returns.
        Stage 2 (graph): LLM extraction, entity resolution, relations. With
        background=False it runs before returning and provider errors raise here.
        With background=True it runs in a worker thread, so a slow local model never
        hits the MCP client's tool timeout; progress shows in stats() as
        chunks_pending_extraction and extraction_in_progress.
        """
        stats = self._store(docs, chunk_chars)
        if background:
            self.start_extraction_worker()
            stats["graph_extraction"] = "running in background"
        else:
            with self._locks[str(self.path)]:
                for k, v in self.process_pending().items():
                    stats[k] = stats.get(k, 0) + v if isinstance(v, int) else v
        return stats

    # ---------------------------------------------------------------- stage 1
    def _store(self, docs: list[dict], chunk_chars: int) -> dict:
        stats: Counter = Counter()
        db = store.connect_ro(self.path)
        try:
            known = {h for (h,) in db.execute("SELECT content_hash FROM docs")}
        finally:
            db.close()
        prepared, items = [], []
        for d in docs:
            text = (d.get("text") or "").strip()
            if not text:
                stats["skipped_empty"] += 1
                continue
            h = hashlib.sha256(text.encode()).hexdigest()
            if h in known:
                stats["skipped_duplicate"] += 1
                continue
            known.add(h)  # also dedups repeats inside one batch
            title = str(d.get("title") or "")
            doc = {"title": title, "source": str(d.get("source") or ""), "metadata": d.get("metadata") or {},
                   "hash": h, "chunks": []}
            for i, ch in enumerate(chunk_text(text, chunk_chars)):
                item = {"idx": i, "text": ch, "ext": f"{title}\n\n{ch}" if title else ch}
                doc["chunks"].append(item)
                items.append(item)
            prepared.append(doc)
        if not items:
            return dict(stats)
        for item, e in zip(items, self.embedder.embed([x["ext"] for x in items])):  # outside the write lock
            item["emb"] = e
        db = self.db
        try:
            db.execute("BEGIN IMMEDIATE")
            for doc in prepared:
                if db.execute("SELECT 1 FROM docs WHERE content_hash=?", (doc["hash"],)).fetchone():
                    stats["skipped_duplicate"] += 1  # another process stored it meanwhile
                    continue
                doc_id = db.execute("INSERT INTO docs (title, source, metadata, content_hash) VALUES (?,?,?,?)",
                                    (doc["title"], doc["source"], json.dumps(doc["metadata"]), doc["hash"])).lastrowid
                stats["documents"] += 1
                for item in doc["chunks"]:
                    db.execute("INSERT INTO chunks (doc_id, idx, text, emb, extracted) VALUES (?,?,?,?,0)",
                               (doc_id, item["idx"], item["text"], store.to_blob(item["emb"])))
                    stats["chunks"] += 1
            db.execute("INSERT OR REPLACE INTO meta VALUES ('version', ?)", (str(self._db_version(db) + 1),))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        self._index = None
        log.info("collection %s: stored %s", self.name, dict(stats))
        return dict(stats)

    # ---------------------------------------------------------------- stage 2
    def process_pending(self) -> dict:
        """Extract and add to the graph every chunk still waiting (extracted=0): new
        chunks and earlier transient failures. Two phases, so several server
        processes (Claude Desktop and Cursor each run their own) can share one data
        directory: the slow LLM work holds no lock, then one BEGIN IMMEDIATE
        transaction re-checks each chunk, allocates ids and resolves entities against
        the graph as it is at that moment. A failure rolls the transaction back and
        leaves the chunks pending (still searchable densely)."""
        stats: Counter = Counter()
        work = self._prepare_pending(stats)
        if not work:
            return dict(stats)
        try:
            return self._commit_pending(work, stats)
        except BaseException:
            self.db.rollback()
            raise

    def _prepare_pending(self, stats: Counter) -> list[dict]:
        db = store.connect_ro(self.path)
        try:
            rows = db.execute("SELECT c.id, c.text, d.title FROM chunks c JOIN docs d ON d.id=c.doc_id "
                              "WHERE c.extracted=0 ORDER BY c.id").fetchall()
        finally:
            db.close()
        work = [{"cid": cid, "text": txt, "ext": f"{title}\n\n{txt}" if title else txt} for cid, txt, title in rows]
        if not work:
            return work
        cache = ExtractionCache(store.home() / "extraction_cache.sqlite")
        t0 = time.time()
        extractions, failed = extract_many(self.extractor, [x["ext"] for x in work], cache)
        log.info("extracted %d chunks in %.1fs with %s (%d failed)", len(work), time.time() - t0, self.extractor.spec, len(failed))
        mentions = [(i, m) for i, ex in enumerate(extractions) for m in ex.entities]
        m_embs = self.embedder.embed([f"{m.name}. {m.description}" for _, m in mentions]) if mentions else []
        for i, item in enumerate(work):
            item["extraction"] = extractions[i]
            item["failed"] = failed.get(i)  # None | "transient" | "output"
            item["mention_embs"] = []
        for (i, _), e in zip(mentions, m_embs):
            work[i]["mention_embs"].append(e)
        return work

    def _commit_pending(self, work: list[dict], stats: Counter) -> dict:
        db = self.db
        db.execute("BEGIN IMMEDIATE")  # take the write lock; other processes wait (busy timeout)
        live = []
        for item in work:  # skip chunks another process finished while we were extracting
            row = db.execute("SELECT extracted FROM chunks WHERE id=?", (item["cid"],)).fetchone()
            if row and row[0] == 0:
                live.append(item)
        # extracted: 1 done, 0 transient failure (retried on the next pass),
        # 2 unusable model output (not retried; the chunk is still searchable densely)
        code = {None: 1, "transient": 0, "output": 2}
        db.executemany("UPDATE chunks SET extracted=? WHERE id=?", [(code[x["failed"]], x["cid"]) for x in live])
        stats["chunks_extracted"] = sum(x["failed"] is None for x in live)
        n_tr = sum(x["failed"] == "transient" for x in live)
        n_out = sum(x["failed"] == "output" for x in live)
        if n_tr:
            stats["extraction_failed"] = n_tr
        if n_out:
            stats["extraction_unusable_output"] = n_out
        self._build_graph([(x["cid"], x["text"], x["ext"]) for x in live], [x["extraction"] for x in live], stats,
                          mention_embs=[e for x in live for e in x["mention_embs"]])
        db.execute("INSERT OR REPLACE INTO meta VALUES ('version', ?)", (str(self._db_version(db) + 1),))
        db.execute("DELETE FROM meta WHERE k='last_extraction_error'")
        db.commit()
        self._index = None
        log.info("collection %s: graph %s", self.name, dict(stats))
        return dict(stats)

    # ---------------------------------------------------------- background worker
    _workers: dict[str, threading.Thread] = {}
    _again: dict[str, bool] = {}
    _worker_lock = threading.Lock()

    def start_extraction_worker(self) -> bool:
        """Start (or nudge) this process's background extraction for the collection.
        Returns True if a new worker thread was started."""
        key = str(self.path)
        with Corpus._worker_lock:
            t = Corpus._workers.get(key)
            if t is not None and t.is_alive():
                Corpus._again[key] = True  # it will take another pass before exiting
                return False
            Corpus._again[key] = False
            t = threading.Thread(target=self._worker_loop, name=f"kgrag-extract-{self.name}", daemon=True)
            Corpus._workers[key] = t
            t.start()
            return True

    def _worker_loop(self) -> None:
        key = str(self.path)
        worker = Corpus(self.path)  # own connection: sqlite connections are not shared across threads
        while True:
            progress = False
            try:
                with Corpus._locks[key]:
                    st = worker.process_pending()
                progress = st.get("chunks_extracted", 0) > 0 or st.get("extraction_unusable_output", 0) > 0
            except Exception as e:
                log.exception("background extraction for %s failed", self.name)
                worker._set_meta("last_extraction_error", f"{type(e).__name__}: {e}")
            with Corpus._worker_lock:
                pending = worker._pending_count()
                if Corpus._again.get(key):
                    Corpus._again[key] = False
                    continue
                if pending == 0 or not progress:
                    Corpus._workers.pop(key, None)
                    worker.db.close()
                    if pending:
                        log.warning("%s: %d chunks still pending extraction; retried on the next add or restart",
                                    self.name, pending)
                    return

    def _pending_count(self) -> int:
        db = store.connect_ro(self.path)
        try:
            return db.execute("SELECT COUNT(*) FROM chunks WHERE extracted=0").fetchone()[0]
        finally:
            db.close()

    def _set_meta(self, k: str, v: str) -> None:
        try:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (k, v))
            self.db.commit()
        except Exception:  # recording an error must never raise a second one
            log.exception("could not record %s", k)

    def extraction_running(self) -> bool:
        t = Corpus._workers.get(str(self.path))
        return t is not None and t.is_alive()

    @staticmethod
    def _db_version(db) -> int:
        row = db.execute("SELECT v FROM meta WHERE k='version'").fetchone()
        return int(row[0]) if row else 0

    def _load_resolver(self) -> EntityResolver:
        r = EntityResolver(self.meta.get("resolution", "full"))
        aliases = defaultdict(set)
        for eid, a in self.db.execute("SELECT entity_id, alias FROM aliases"):
            aliases[eid].add(a)
        for eid, typ, emb in self.db.execute("SELECT id, type, emb FROM entities"):
            rec = EntityRecord(eid, typ, emb=store.from_blob(emb))
            for a in aliases[eid]:
                rec.add_alias(a)
            r.load(rec)
        return r

    def _build_graph(self, chunks, extractions, stats: Counter, mention_embs=None) -> None:
        """Runs inside the write transaction. mention_embs: one per extracted entity
        mention, in order (precomputed outside the lock); computed here if absent."""
        db = self.db
        resolver = self._load_resolver()
        before = set(resolver.entities)
        next_id = (db.execute("SELECT COALESCE(MAX(id), 0) FROM entities").fetchone()[0]) + 1

        # embed every mention once, used for description-backed fuzzy matching
        flat = [(ci, m) for ci, ex in enumerate(extractions) for m in ex.entities]
        m_embs = mention_embs if mention_embs is not None else (
            self.embedder.embed([f"{m.name}. {m.description}" for _, m in flat]) if flat else [])

        alias_counts: dict[int, Counter] = defaultdict(Counter)
        descs: dict[int, list[str]] = defaultdict(list)
        touched: set[int] = set()
        local_maps: list[dict[str, int]] = [dict() for _ in chunks]
        for (ci, m), emb in zip(flat, m_embs):
            eid, created = resolver.resolve(m.name, m.type, next_id, emb)
            if created:
                next_id += 1
            local_maps[ci][m.name] = eid
            alias_counts[eid][m.name] += 1
            if m.description:
                descs[eid].append(m.description)
            touched.add(eid)
            stats["entity_mentions"] += 1
        stats["entities_created"] = len(set(resolver.entities) - before)
        stats["mentions_merged_fuzzy"] = resolver.merges

        # write entities (new + touched)
        existing_desc = dict(db.execute("SELECT id, description FROM entities"))
        for eid in touched:
            rec = resolver.entities[eid]
            for a in rec.aliases:
                db.execute("INSERT OR IGNORE INTO aliases VALUES (?,?)", (eid, a))
            all_aliases = [a for (a,) in db.execute("SELECT alias FROM aliases WHERE entity_id=?", (eid,))]
            name = _canonical_name(all_aliases, alias_counts[eid])
            desc_parts = [p for p in (existing_desc.get(eid) or "").split(" | ") if p]
            for dsc in descs[eid]:
                if dsc not in desc_parts and sum(map(len, desc_parts)) < 600:
                    desc_parts.append(dsc)
            desc = " | ".join(desc_parts)
            if eid in existing_desc:
                db.execute("UPDATE entities SET name=?, type=?, description=?, mention_count=mention_count+? WHERE id=?",
                           (name, rec.type, desc, sum(alias_counts[eid].values()), eid))
            else:
                db.execute("INSERT INTO entities (id, name, type, description, mention_count) VALUES (?,?,?,?,?)",
                           (eid, name, rec.type, desc, sum(alias_counts[eid].values())))
        # refresh entity embeddings for touched entities
        rows = db.execute(f"SELECT id, name, description FROM entities WHERE id IN ({','.join('?' * len(touched))})", list(touched)).fetchall() if touched else []
        if rows:
            ev = self.embedder.embed([f"{n}. {d[:400]}" for _, n, d in rows])
            db.executemany("UPDATE entities SET emb=? WHERE id=?", [(store.to_blob(v), r[0]) for v, r in zip(ev, rows)])

        # mentions and relations
        new_rel_rows = []
        for (cid, _, _), ex, lmap in zip(chunks, extractions, local_maps):
            for eid in set(lmap.values()):
                db.execute("INSERT OR IGNORE INTO mentions VALUES (?,?)", (eid, cid))
            for r in ex.relations:
                s, t = lmap.get(r.source), lmap.get(r.target)
                if s is None or t is None:
                    continue
                s, label, t = canonical_relation(s, r.relation, t)
                if s == t or not label:
                    stats["self_loops_dropped"] += 1
                    continue
                row = db.execute("SELECT id FROM relations WHERE src=? AND label=? AND dst=?", (s, label, t)).fetchone()
                if row:
                    rid = row[0]
                    db.execute("UPDATE relations SET weight=weight+1 WHERE id=?", (rid,))
                    stats["relations_deduplicated"] += 1
                else:
                    rid = db.execute("INSERT INTO relations (src, dst, label, description, weight) VALUES (?,?,?,?,1)",
                                     (s, t, label, r.description)).lastrowid
                    new_rel_rows.append(rid)
                    stats["relations_created"] += 1
                db.execute("INSERT OR IGNORE INTO rel_evidence VALUES (?,?)", (rid, cid))
        if new_rel_rows:
            # embed the relation label alone: traversal steering asks "is this the kind of
            # edge the query is about?", and endpoint names would drown that signal
            rows = db.execute(f"SELECT id, label FROM relations WHERE id IN ({','.join('?' * len(new_rel_rows))})", new_rel_rows).fetchall()
            rv = self.embedder.embed([lab for _, lab in rows])
            db.executemany("UPDATE relations SET emb=? WHERE id=?", [(store.to_blob(v), r[0]) for v, r in zip(rv, rows)])

    # ---------------------------------------------------------------- index
    def index(self) -> Index:
        """In-memory retrieval index, rebuilt when the stored version changes. The
        version is read from the database on every call (one indexed lookup), so an
        ingest by another process (e.g. Cursor while Claude Desktop is querying) is
        picked up on the next query."""
        db, dim = store.connect_ro(self.path), self.embedder.dim
        v = self._db_version(db)
        if self._index is not None and self._index.version == v:
            db.close()
            return self._index
        crow = db.execute("SELECT id, doc_id, emb FROM chunks ORDER BY id").fetchall()
        erow = db.execute("SELECT id, name, emb FROM entities ORDER BY id").fetchall()
        rrow = db.execute("SELECT id, src, label, dst, weight, emb FROM relations ORDER BY id").fetchall()
        alias_norm: dict[str, set[int]] = defaultdict(set)
        alias_loose: dict[str, set[int]] = defaultdict(set)
        for eid, a in db.execute("SELECT entity_id, alias FROM aliases"):
            n = normalize_name(a)
            if not n:
                continue
            alias_norm[n].add(eid)
            toks = n.split()
            if len(toks) >= 2 and len(toks[0]) > 1:
                alias_loose[f"{toks[0][0]} {' '.join(toks[1:])}"].add(eid)
            ac = acronym(a)
            if len(ac) >= 2:
                alias_loose[ac].add(eid)
        ent_chunks: dict[int, list[int]] = defaultdict(list)
        chunk_ents: dict[int, list[int]] = defaultdict(list)
        for eid, cid in db.execute("SELECT entity_id, chunk_id FROM mentions"):
            ent_chunks[eid].append(cid)
            chunk_ents[cid].append(eid)
        adj: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
        for row_i, (_, s, _, d, w, _) in enumerate(rrow):
            adj[s].append((d, row_i, w))
            adj[d].append((s, row_i, w))  # traverse both directions
        rel_ev: dict[int, list[int]] = defaultdict(list)
        rid_row = {r[0]: i for i, r in enumerate(rrow)}
        for rid, cid in db.execute("SELECT rel_id, chunk_id FROM rel_evidence"):
            rel_ev[rid_row[rid]].append(cid)
        self._index = Index(
            version=v,
            chunk_ids=np.array([r[0] for r in crow], dtype=np.int64),
            chunk_emb=store.stack([r[2] for r in crow], dim),
            chunk_doc=np.array([r[1] for r in crow], dtype=np.int64),
            ent_ids=np.array([r[0] for r in erow], dtype=np.int64),
            ent_emb=store.stack([r[2] for r in erow], dim),
            ent_names=[r[1] for r in erow],
            ent_row={r[0]: i for i, r in enumerate(erow)},
            alias_norm=dict(alias_norm),
            alias_loose=dict(alias_loose),
            ent_chunks=dict(ent_chunks),
            chunk_ents=dict(chunk_ents),
            adj=dict(adj),
            rel_ids=np.array([r[0] for r in rrow], dtype=np.int64),
            rel_emb=store.stack([r[5] for r in rrow], dim),
            rel_triples=[(r[1], r[2], r[3]) for r in rrow],
            rel_evidence=dict(rel_ev),
        )
        db.close()
        return self._index

    def contents(self, limit: int = 50, offset: int = 0, top_entities: int = 20) -> dict:
        """Documents (paginated, newest first) and the most-mentioned entities."""
        db = store.connect_ro(self.path)
        try:
            total = db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
            docs = [{"doc_id": r[0], "title": r[1], "source": r[2], "chunks": r[3], "added_at": r[4]} for r in db.execute(
                "SELECT d.id, d.title, d.source, COUNT(c.id), d.added_at FROM docs d LEFT JOIN chunks c ON c.doc_id=d.id "
                "GROUP BY d.id ORDER BY d.id DESC LIMIT ? OFFSET ?", (limit, offset))]
            ents = [{"name": r[0], "type": r[1], "mentions": r[2], "aliases": r[3].split("\x1f") if r[3] else []} for r in db.execute(
                "SELECT e.name, e.type, e.mention_count, GROUP_CONCAT(a.alias, char(31)) FROM entities e "
                "LEFT JOIN aliases a ON a.entity_id=e.id GROUP BY e.id ORDER BY e.mention_count DESC, e.id LIMIT ?", (top_entities,))]
            rels = [{"source": r[0], "relation": r[1], "target": r[2], "support": r[3]} for r in db.execute(
                "SELECT s.name, r.label, d.name, r.weight FROM relations r JOIN entities s ON s.id=r.src "
                "JOIN entities d ON d.id=r.dst ORDER BY r.weight DESC, r.id LIMIT ?", (top_entities,))]
        finally:
            db.close()
        return {"documents_total": total, "offset": offset, "documents": docs, "top_entities": ents, "top_relations": rels}

    def chunk_records(self, chunk_ids: list[int]) -> dict[int, dict]:
        if not chunk_ids:
            return {}
        db = store.connect_ro(self.path)
        rows = db.execute(
            f"SELECT c.id, c.text, d.id, d.title, d.source FROM chunks c JOIN docs d ON d.id=c.doc_id WHERE c.id IN ({','.join('?' * len(chunk_ids))})",
            chunk_ids).fetchall()
        db.close()
        return {r[0]: {"chunk_id": r[0], "text": r[1], "doc_id": r[2], "title": r[3], "source": r[4]} for r in rows}

    def doc_meta(self) -> dict[int, dict]:
        return {r[0]: {"title": r[1], "metadata": json.loads(r[2] or "{}")} for r in self.db.execute("SELECT id, title, metadata FROM docs")}

