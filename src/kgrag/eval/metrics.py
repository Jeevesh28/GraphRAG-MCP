"""Ranking metrics over document ids with graded relevance."""

from __future__ import annotations

import math


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """Exponential gain (2^rel - 1), standard in IR. qrels: doc -> graded relevance."""
    gains = [(2 ** qrels.get(d, 0)) - 1 for d in ranked[:k]]
    ideal = sorted(((2 ** r) - 1 for r in qrels.values()), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    rel = {d for d, r in qrels.items() if r > 0}
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0


def full_support_at_k(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """1 if every relevant doc is in the top k: the multi-hop 'all evidence found' rate."""
    rel = {d for d, r in qrels.items() if r > 0}
    return float(bool(rel) and rel <= set(ranked[:k]))


def mrr(ranked: list[str], qrels: dict[str, int]) -> float:
    for i, d in enumerate(ranked):
        if qrels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0
