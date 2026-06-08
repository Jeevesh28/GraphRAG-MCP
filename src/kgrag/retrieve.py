"""Hybrid retrieval: dense chunk similarity fused with spreading activation over the
entity-relation graph.

Graph side, step by step:
  1. Seed entities from the query: alias n-gram matches (exact, after normalization),
     nearest entity embeddings, and entities mentioned in the top dense chunks.
  2. Spread activation for `hops` steps. Each step passes `decay` of a node's
     activation to its neighbours, weighted by log(1 + mention count) times
     exp(rel_beta * sim(query, relation label)), normalized so the best edge gets
     the full share. Traversal therefore follows edges whose relation matches the query ("headquartered in" for a
     "where is it based" question), so traversal follows the relevant edges.
  3. Score chunks by the activation of the entities they mention (damped for hub
     entities that appear everywhere) plus the flow over relations they are
     evidence for.
Fusion is either weighted min-max (alpha * dense + (1 - alpha) * graph) or RRF.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from .corpus import Corpus, Index
from .text import normalize_name


@dataclass
class RetrievalParams:
    mode: str = "hybrid"          # dense | graph | hybrid
    k: int = 10
    fusion: str = "weighted"      # weighted | rrf
    alpha: float = 0.4            # dense share in weighted / rrf fusion
    rrf_k: int = 60
    hops: int = 2
    decay: float = 0.6
    seed_semantic_k: int = 3
    seed_semantic_min: float = 0.35
    seed_from_dense: int = 2      # top dense chunks whose entities become weak seeds
    seed_dense_weight: float = 0.3
    rel_beta: float = 4.0         # steering sharpness: edge share ~ exp(rel_beta * sim(query, relation))
    hub_penalty: float = 0.5      # chunk score divides by (#chunks mentioning entity) ** hub_penalty
    edge_weight: float = 1.0      # weight of relation-evidence term in chunk graph score
    dense_pool: int = 100

    @classmethod
    def from_dict(cls, d: dict | None) -> "RetrievalParams":
        p = cls()
        for k, v in (d or {}).items():
            if hasattr(p, k):
                setattr(p, k, type(getattr(p, k))(v))
        return p


def _ngrams(toks: list[str], max_n: int = 6):
    for n in range(min(max_n, len(toks)), 0, -1):
        for i in range(len(toks) - n + 1):
            yield " ".join(toks[i : i + n])


def link_entities(idx: Index, query: str, qv: np.ndarray, p: RetrievalParams) -> dict[int, float]:
    seeds: dict[int, float] = {}
    toks = normalize_name(query).split()
    for g in _ngrams(toks):
        if len(g) < 3:
            continue
        exact = idx.alias_norm.get(g, set())
        for eid in exact:
            seeds[eid] = max(seeds.get(eid, 0.0), 1.0)
        # looser keys: "M. Okonkwo" <-> "Mira Okonkwo", "BHS" <-> "Blue Harbor Systems".
        # Ambiguous keys split their weight across candidates.
        gt = g.split()
        keys = [g] if len(gt) == 1 else ([f"{gt[0][0]} {' '.join(gt[1:])}"] if len(gt) >= 2 and len(gt[0]) > 1 else [])
        for key in keys:
            loose = idx.alias_loose.get(key, set()) - exact
            for eid in loose:
                seeds[eid] = max(seeds.get(eid, 0.0), 0.9 / len(loose))
    if p.seed_semantic_k and len(idx.ent_ids):
        sims = idx.ent_emb @ qv
        for r in np.argsort(-sims)[: p.seed_semantic_k]:
            if sims[r] >= p.seed_semantic_min:
                eid = int(idx.ent_ids[r])
                seeds[eid] = max(seeds.get(eid, 0.0), float(sims[r]))
    return seeds


def spread(idx: Index, seeds: dict[int, float], qv: np.ndarray, p: RetrievalParams):
    act = dict(seeds)
    parent: dict[int, tuple[int, int]] = {}
    edge_flow: dict[int, float] = defaultdict(float)
    rel_sim = (idx.rel_emb @ qv) if len(idx.rel_ids) else np.zeros(0)
    frontier = dict(seeds)
    for _ in range(p.hops):
        nxt: dict[int, float] = defaultdict(float)
        for u, a in frontier.items():
            nbrs = idx.adj.get(u)
            if not nbrs or a < 1e-4:
                continue
            w = np.array([np.log1p(wt) * np.exp(p.rel_beta * float(rel_sim[row])) for _, row, wt in nbrs])
            w = w / w.max()
            best: dict[int, float] = {}
            for (v, row, _), wi in zip(nbrs, w):
                f = a * p.decay * wi
                edge_flow[row] += f
                if seeds.get(v, 0.0) >= 1.0:  # exact query entities are already maximally active
                    continue
                nxt[v] += f
                if v not in seeds and f > best.get(v, 0):
                    best[v] = f
                    if v not in parent or f > act.get(v, 0):
                        parent[v] = (u, row)
        for v, f in nxt.items():
            act[v] = act.get(v, 0.0) + f
        frontier = nxt
    return act, parent, edge_flow


def graph_scores(idx: Index, act: dict[int, float], edge_flow: dict[int, float], p: RetrievalParams) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for eid, a in act.items():
        cl = idx.ent_chunks.get(eid, [])
        if not cl:
            continue
        damp = len(cl) ** p.hub_penalty
        for cid in cl:
            scores[cid] += a / damp
    if p.edge_weight:
        for row, f in edge_flow.items():
            for cid in idx.rel_evidence.get(row, []):
                scores[cid] += p.edge_weight * f
    return scores


def _minmax(d: dict[int, float]) -> dict[int, float]:
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def retrieve(corpus: Corpus, query: str, params: RetrievalParams | dict | None = None, explain: bool = True) -> dict:
    p = params if isinstance(params, RetrievalParams) else RetrievalParams.from_dict(params)
    idx = corpus.index()
    if len(idx.chunk_ids) == 0:
        return {"results": [], "params": asdict(p)}
    qv = corpus.embedder.embed([query])[0]

    dsims = idx.chunk_emb @ qv
    pool = np.argsort(-dsims)[: p.dense_pool]
    dense = {int(idx.chunk_ids[i]): float(dsims[i]) for i in pool}

    graph: dict[int, float] = {}
    seeds: dict[int, float] = {}
    act: dict[int, float] = {}
    parent: dict[int, tuple[int, int]] = {}
    if p.mode in ("graph", "hybrid"):
        seeds = link_entities(idx, query, qv, p)
        if p.seed_from_dense:
            for i in pool[: p.seed_from_dense]:
                for eid in idx.chunk_ents.get(int(idx.chunk_ids[i]), []):
                    seeds.setdefault(eid, p.seed_dense_weight)
        act, parent, flow = spread(idx, seeds, qv, p)
        graph = graph_scores(idx, act, flow, p)

    if p.mode == "dense":
        fused = dense
    elif p.mode == "graph":
        fused = dict(graph)
        # fall back to dense order for chunks the graph never reached
        for cid, s in dense.items():
            fused.setdefault(cid, -1.0 + s * 1e-3)
    elif p.fusion == "rrf":
        rd = {c: r for r, c in enumerate(sorted(dense, key=dense.get, reverse=True))}
        rg = {c: r for r, c in enumerate(sorted(graph, key=graph.get, reverse=True))}
        fused = {c: p.alpha / (p.rrf_k + rd.get(c, 10**6)) + (1 - p.alpha) / (p.rrf_k + rg.get(c, 10**6)) for c in set(rd) | set(rg)}
    else:
        nd, ng = _minmax(dense), _minmax(graph)
        fused = {c: p.alpha * nd.get(c, 0.0) + (1 - p.alpha) * ng.get(c, 0.0) for c in set(nd) | set(ng)}

    top = sorted(fused, key=fused.get, reverse=True)[: p.k]
    out: dict = {"ranked_chunk_ids": top}
    if not explain:
        return out
    recs = corpus.chunk_records(top)
    names = dict(zip(idx.ent_ids.tolist(), idx.ent_names))
    results = []
    for cid in top:
        r = recs[cid]
        r.update(score=round(fused[cid], 4), dense=round(dense.get(cid, 0.0), 4), graph=round(graph.get(cid, 0.0), 4),
                 entities=[names[e] for e in idx.chunk_ents.get(cid, [])][:12])
        results.append(r)

    # Explanation paths: shortest relation path from the entities the query names
    # (exact alias matches), falling back to the strongest seed. This is independent of
    # which entities happened to be seeds, so a small collection where every entity is
    # also a semantic seed still gets readable paths. It does not affect ranking.
    anchors = [e for e, w in seeds.items() if w >= 0.9] or sorted(seeds, key=seeds.get, reverse=True)[:1]
    bfs_parent: dict[int, tuple[int, int]] = {}
    frontier, seen = list(anchors), set(anchors)
    for _ in range(max(p.hops, 1)):
        nxt = []
        for u in frontier:
            for v, row, _ in idx.adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    bfs_parent[v] = (u, row)
                    nxt.append(v)
        frontier = nxt

    def path(eid: int) -> list[str]:
        hops, cur = [], eid
        while cur in bfs_parent:
            u, row = bfs_parent[cur]
            s, lab, d = idx.rel_triples[row]
            hops.append(f"{names[s]} --{lab}--> {names[d]}")
            cur = u
        return hops[::-1]

    top_ents = sorted(act, key=act.get, reverse=True)[:8]
    out.update(
        results=results,
        seed_entities=[names[e] for e in sorted(seeds, key=seeds.get, reverse=True)[:8]],
        graph_context=[{"entity": names[e], "activation": round(act[e], 3), "path": path(e)} for e in top_ents],
        params=asdict(p),
    )
    return out
