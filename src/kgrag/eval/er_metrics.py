"""Entity-resolution quality against gold mentions (synthetic benchmark only).

Each gold mention is a (document, surface form, true entity). It is matched to the
graph node that the document's chunk links to under the same normalized name.
B-cubed precision says how pure the predicted nodes are (low = wrong merges);
B-cubed recall says how completely each true entity was gathered into one node
(low = duplicates left behind). Coverage is the share of gold mentions the extractor
found at all; missed mentions are excluded from B-cubed and reported separately.
"""

from __future__ import annotations

from collections import defaultdict

from ..corpus import Corpus
from ..text import normalize_name


def bcubed(corpus: Corpus, docs: list[dict]) -> dict:
    db = corpus.db
    title_chunks: dict[str, list[int]] = defaultdict(list)
    for cid, title in db.execute("SELECT c.id, d.title FROM chunks c JOIN docs d ON d.id=c.doc_id"):
        title_chunks[title].append(cid)
    chunk_ents: dict[int, set[int]] = defaultdict(set)
    for eid, cid in db.execute("SELECT entity_id, chunk_id FROM mentions"):
        chunk_ents[cid].add(eid)
    ent_norms: dict[int, set[str]] = defaultdict(set)
    for eid, a in db.execute("SELECT entity_id, alias FROM aliases"):
        ent_norms[eid].add(normalize_name(a))

    items: list[tuple[int, str]] = []  # (predicted node, gold entity)
    total = 0
    for d in docs:
        for surface, gold in d.get("gold_mentions", []):
            total += 1
            n = normalize_name(surface)
            cands = {e for cid in title_chunks.get(d["title"], []) for e in chunk_ents[cid] if n in ent_norms[e]}
            if len(cands) == 1:
                items.append((cands.pop(), gold))
    if not items:
        return {"coverage": 0.0}
    by_pred: dict[int, list[str]] = defaultdict(list)
    by_gold: dict[str, list[int]] = defaultdict(list)
    for p, g in items:
        by_pred[p].append(g)
        by_gold[g].append(p)
    prec = rec = 0.0
    for p, g in items:
        same = sum(1 for x in by_pred[p] if x == g)  # items sharing both pred node and gold entity
        prec += same / len(by_pred[p])
        rec += same / len(by_gold[g])
    P, R = prec / len(items), rec / len(items)
    return {
        "bcubed_precision": P,
        "bcubed_recall": R,
        "bcubed_f1": 2 * P * R / (P + R) if P + R else 0.0,
        "coverage": len(items) / total,
        "gold_entities": len({g for _, g in items}),
        "predicted_nodes": len({p for p, _ in items}),
    }
