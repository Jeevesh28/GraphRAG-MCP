"""Evaluate and tune hybrid retrieval with nDCG.

    kgrag-eval --dataset synthetic
    kgrag-eval --dataset hotpot --path hotpot_dev_distractor_v1.json --n 300 \
               --embedder st:BAAI/bge-small-en-v1.5 --extractor anthropic:claude-haiku-5-5

Protocol:
  * queries are split 50/50 into dev and test with a fixed seed
  * hybrid parameters are tuned by random search on dev (objective: mean nDCG@10)
  * every number in the final table is on test, which tuning never saw
  * the corpus is built twice, with and without entity resolution, to measure how
    much graph quality alone moves retrieval (same tuning budget for both)
  * a paired bootstrap gives a 95% CI for tuned-hybrid minus dense on test
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..corpus import Corpus
from ..embed import resolve_spec
from ..logs import configure
from ..retrieve import RetrievalParams, retrieve
from . import hotpot, synthetic
from .er_metrics import bcubed
from .metrics import full_support_at_k, mrr, ndcg_at_k, recall_at_k

SPACE = {
    "fusion": ["weighted", "rrf"],
    "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "hops": [1, 2, 3],
    "decay": [0.3, 0.5, 0.7, 0.9],
    "rel_beta": [0.0, 2.0, 4.0, 8.0],
    "hub_penalty": [0.0, 0.5, 1.0],
    "edge_weight": [0.0, 0.5, 1.0, 2.0],
    "seed_semantic_k": [0, 3, 5],
    "seed_semantic_min": [0.25, 0.35, 0.5],
    "seed_from_dense": [0, 2, 5],
    "seed_dense_weight": [0.1, 0.3, 0.5],
}


def build(docs, name: str, root: Path, embedder: str, extractor: str, resolution: str) -> tuple[Corpus, float]:
    t = time.time()
    # eval corpora are disposable; the extraction cache next to them is kept, so a
    # rerun with the same --workdir costs no LLM calls
    for suffix in ("", "-wal", "-shm"):
        (root / "corpora" / f"{name}.sqlite{suffix}").unlink(missing_ok=True)
    c = Corpus.create(name, root=root, embedder=embedder, extractor=extractor, resolution=resolution, exist_ok=False)
    for i in range(0, len(docs), 200):
        c.add_documents(docs[i : i + 200])
    return c, time.time() - t


def rank_docs(corpus: Corpus, chunk_title: dict[int, str], query: str, p: RetrievalParams, depth: int = 10) -> list[str]:
    p.k = 60
    ids = retrieve(corpus, query, p, explain=False)["ranked_chunk_ids"]
    seen, out = set(), []
    for cid in ids:
        t = chunk_title[cid]
        if t not in seen:
            seen.add(t)
            out.append(t)
            if len(out) == depth:
                break
    return out


def evaluate(corpus, chunk_title, queries, p: RetrievalParams, k: int = 10) -> dict:
    per = []
    for q in queries:
        ranked = rank_docs(corpus, chunk_title, q["query"], RetrievalParams(**asdict(p)), depth=max(k, 20))
        per.append({"id": q["id"], "type": q["type"], "ndcg": ndcg_at_k(ranked, q["qrels"], k),
                    "recall": recall_at_k(ranked, q["qrels"], k), "support": full_support_at_k(ranked, q["qrels"], k),
                    "mrr": mrr(ranked, q["qrels"])})
    agg = {m: float(np.mean([x[m] for x in per])) for m in ("ndcg", "recall", "support", "mrr")}
    by_type = defaultdict(list)
    for x in per:
        by_type[x["type"]].append(x["ndcg"])
    agg["ndcg_by_type"] = {t: float(np.mean(v)) for t, v in sorted(by_type.items())}
    agg["per_query"] = per
    return agg


def tune(corpus, chunk_title, dev, trials: int, seed: int, k: int) -> tuple[RetrievalParams, float, list]:
    rng = random.Random(seed)
    best_p, best = RetrievalParams(), evaluate(corpus, chunk_title, dev, RetrievalParams(), k)["ndcg"]
    history = [{"params": "default", "dev_ndcg": best}]
    for _ in range(trials):
        cfg = {k_: rng.choice(v) for k_, v in SPACE.items()}
        p = RetrievalParams.from_dict(cfg)
        s = evaluate(corpus, chunk_title, dev, p, k)["ndcg"]
        history.append({"params": cfg, "dev_ndcg": s})
        if s > best:
            best, best_p = s, p
    return best_p, best, history


def bootstrap_diff(a: list[float], b: list[float], n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    a, b = np.array(a), np.array(b)
    d = a - b
    rng = np.random.default_rng(seed)
    samples = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["synthetic", "hotpot"], default="synthetic")
    ap.add_argument("--path", help="HotpotQA json (for --dataset hotpot)")
    ap.add_argument("--n", type=int, default=300, help="questions to sample (hotpot)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embedder", default=os.environ.get("KGRAG_EMBEDDER", "auto"))
    ap.add_argument("--extractor", default=os.environ.get("KGRAG_EXTRACTOR", "heuristic"))
    ap.add_argument("--resolution", default="full,none", help="comma list of full|exact_norm|none")
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--workdir", help="where to build eval corpora (default: temp dir)")
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--save-params", help="write the tuned params (resolution=full run) here, e.g. "
                    "~/.kgrag/retrieval_params.json, so the MCP server uses them")
    a = ap.parse_args(argv)

    configure()
    a.embedder = resolve_spec(a.embedder)
    if a.dataset == "synthetic":
        docs, queries = synthetic.generate(a.seed)
    else:
        if not a.path:
            ap.error("--path is required for hotpot")
        docs, queries = hotpot.load(a.path, a.n, a.seed)
    rng = random.Random(a.seed)
    qs = list(queries)
    rng.shuffle(qs)
    dev, test = qs[: len(qs) // 2], qs[len(qs) // 2 :]
    root = Path(a.workdir or tempfile.mkdtemp(prefix="kgrag-eval-"))
    os.environ["KGRAG_HOME"] = str(root)  # keeps the extraction cache with the eval corpora
    print(f"[eval] {a.dataset}: {len(docs)} docs, {len(dev)} dev / {len(test)} test queries; "
          f"embedder={a.embedder} extractor={a.extractor}", file=sys.stderr)

    report: dict = {"dataset": a.dataset, "embedder": a.embedder, "extractor": a.extractor, "k": a.k,
                    "n_docs": len(docs), "n_dev": len(dev), "n_test": len(test), "runs": {}}
    dense_test = None
    for mode in a.resolution.split(","):
        name = f"eval-{a.dataset}-{mode}-{a.seed}"
        corpus, secs = build(docs, name, root, a.embedder, a.extractor, mode)
        idx = corpus.index()
        dm = corpus.doc_meta()
        chunk_title = {int(c): dm[int(d)]["title"] for c, d in zip(idx.chunk_ids, idx.chunk_doc)}
        st = corpus.stats()
        print(f"[eval] built {name} in {secs:.1f}s: {st['entities']} entities, {st['relations']} relations", file=sys.stderr)

        run: dict = {"graph": {"entities": st["entities"], "relations": st["relations"], "build_seconds": round(secs, 1)}}
        if any("gold_mentions" in d for d in docs):
            run["graph"]["resolution_quality"] = bcubed(corpus, docs)
        if dense_test is None:
            dense_test = evaluate(corpus, chunk_title, test, RetrievalParams(mode="dense"), a.k)
            report["dense"] = {k: v for k, v in dense_test.items() if k != "per_query"}
        run["graph_only_default"] = evaluate(corpus, chunk_title, test, RetrievalParams(mode="graph"), a.k)
        run["hybrid_default"] = evaluate(corpus, chunk_title, test, RetrievalParams(), a.k)
        t0 = time.time()
        best_p, best_dev, hist = tune(corpus, chunk_title, dev, a.trials, a.seed, a.k)
        print(f"[eval] tuned {mode} in {time.time() - t0:.0f}s, dev nDCG@{a.k}={best_dev:.4f}", file=sys.stderr)
        run["tuned_params"] = asdict(best_p)
        run["tuned_dev_ndcg"] = best_dev
        run["hybrid_tuned"] = evaluate(corpus, chunk_title, test, best_p, a.k)
        g = RetrievalParams(**asdict(best_p))
        g.mode = "graph"
        run["graph_only_tuned"] = evaluate(corpus, chunk_title, test, g, a.k)
        diff = bootstrap_diff([x["ndcg"] for x in run["hybrid_tuned"]["per_query"]], [x["ndcg"] for x in dense_test["per_query"]])
        run["hybrid_minus_dense_ndcg"] = {"mean": diff[0], "ci95": [diff[1], diff[2]]}
        run["tuning_history_top10"] = sorted(hist, key=lambda h: -h["dev_ndcg"])[:10]
        for key in ("graph_only_default", "hybrid_default", "hybrid_tuned", "graph_only_tuned"):
            run[key].pop("per_query")
        report["runs"][mode] = run

    Path(a.out).write_text(json.dumps(report, indent=2))
    if a.save_params:
        mode = "full" if "full" in report["runs"] else next(iter(report["runs"]))
        sp = Path(a.save_params).expanduser()
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(report["runs"][mode]["tuned_params"], indent=2))
        print(f"tuned params ({mode}) written to {sp}", file=sys.stderr)
    print(render(report))
    print(f"\nfull results: {a.out}")


def render(r: dict) -> str:
    k = r["k"]
    lines = [f"## {r['dataset']} | embedder={r['embedder']} | extractor={r['extractor']} | test queries={r['n_test']}", "",
             f"| system | resolution | nDCG@{k} | Recall@{k} | all-evidence@{k} | MRR |", "|---|---|---|---|---|---|"]
    d = r["dense"]
    lines.append(f"| dense only | n/a | {d['ndcg']:.3f} | {d['recall']:.3f} | {d['support']:.3f} | {d['mrr']:.3f} |")
    for mode, run in r["runs"].items():
        for key, label in (("graph_only_tuned", "graph only (tuned)"), ("hybrid_default", "hybrid (default params)"), ("hybrid_tuned", "hybrid (tuned)")):
            m = run[key]
            lines.append(f"| {label} | {mode} | {m['ndcg']:.3f} | {m['recall']:.3f} | {m['support']:.3f} | {m['mrr']:.3f} |")
    lines += ["", "Graph size and tuned-hybrid gain over dense (paired bootstrap, 95% CI):", ""]
    for mode, run in r["runs"].items():
        dd = run["hybrid_minus_dense_ndcg"]
        lines.append(f"- {mode}: {run['graph']['entities']} entities, {run['graph']['relations']} relations; "
                     f"delta nDCG = {dd['mean']:+.3f} [{dd['ci95'][0]:+.3f}, {dd['ci95'][1]:+.3f}]")
    if any("resolution_quality" in run["graph"] for run in r["runs"].values()):
        lines += ["", "Entity resolution vs gold mentions (B-cubed):", "",
                  "| resolution | precision | recall | F1 | mention coverage | gold entities | predicted nodes |",
                  "|---|---|---|---|---|---|---|"]
        for mode, run in r["runs"].items():
            q = run["graph"].get("resolution_quality")
            if q and "bcubed_f1" in q:
                lines.append(f"| {mode} | {q['bcubed_precision']:.3f} | {q['bcubed_recall']:.3f} | {q['bcubed_f1']:.3f} | "
                             f"{q['coverage']:.3f} | {q['gold_entities']} | {q['predicted_nodes']} |")
    types = list(r["dense"]["ndcg_by_type"])
    lines += ["", f"nDCG@{k} by question type:", "", "| type | dense | " + " | ".join(f"hybrid tuned ({m})" for m in r["runs"]) + " |",
              "|---|---|" + "---|" * len(r["runs"])]
    for t in types:
        lines.append(f"| {t} | {r['dense']['ndcg_by_type'][t]:.3f} | " +
                     " | ".join(f"{run['hybrid_tuned']['ndcg_by_type'].get(t, 0):.3f}" for run in r["runs"].values()) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
