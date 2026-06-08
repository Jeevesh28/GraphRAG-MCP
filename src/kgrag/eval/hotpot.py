"""Load a HotpotQA-format file (e.g. hotpot_dev_distractor_v1.json) as a retrieval benchmark.

Corpus: the union of all context paragraphs of the sampled questions (deduplicated by
title), so every question competes against every other question's distractors.
Qrels: the two supporting-fact paragraphs, both relevance 1.

Download the file yourself (e.g. from the HotpotQA site) and pass its path; the
loader never touches the network.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def load(path: str | Path, n: int = 300, seed: int = 0):
    data = json.loads(Path(path).read_text())
    rng = random.Random(seed)
    rng.shuffle(data)
    data = data[:n]
    docs: dict[str, dict] = {}
    queries = []
    for ex in data:
        for title, sents in ex["context"]:
            if title not in docs:
                docs[title] = {"title": title, "text": f"{title}. " + "".join(sents)}
        sup = {t for t, _ in ex["supporting_facts"]}
        queries.append({"id": ex["_id"], "type": f"hotpot_{ex.get('type', 'q')}", "query": ex["question"],
                        "qrels": {t: 1 for t in sup}})
    return list(docs.values()), queries
