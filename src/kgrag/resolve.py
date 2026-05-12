"""Entity resolution and relation canonicalization during graph construction.

Every extracted mention is matched against existing entities before it becomes a
node. Without this step, "Dr. Mira Okonkwo", "Mira Okonkwo" and "M. Okonkwo"
become three nodes, each holding a third of her edges, and multi-hop traversal
dead-ends at whichever copy the query happened to link to.

Matching, in order of confidence:
  1. normalized key equality (case, accents, punctuation, honorifics, corp suffixes)
  2. acronym <-> expansion ("BHS" <-> "Blue Harbor Systems")
  3. initial + surname ("M. Okonkwo" <-> "Mira Okonkwo")
  4. token containment ("Velantis" <-> "Velantis Robotics"), only if unambiguous
  5. near-identical spelling (typos), optionally backed by description embeddings

Guards that keep merges precise:
  - type compatibility: fuzzy rules need agreeing types; an identical name merges
    across types except person/location ("Jordan" the person vs the country)
  - first-name veto: "Jane Smith" and "John Smith" never merge, and a cluster
    containing one can never absorb the other through a shared "J. Smith"
  - ambiguity: if the best two candidates score about the same, no merge
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from .text import ORG_SUFFIXES, acronym, normalize_name, normalize_relation

_GENERIC = {"of", "and", "the", "for", "in", "on", "a", "an"} | ORG_SUFFIXES
_PASSIVE_IRREGULAR = {"made", "built", "written", "led", "run", "owned", "held", "taught", "won", "bought", "sold", "given", "drawn", "known", "chosen", "taken"}


@dataclass
class EntityRecord:
    id: int
    type: str
    aliases: set[str] = field(default_factory=set)  # raw surface forms
    norms: set[str] = field(default_factory=set)
    emb: np.ndarray | None = None

    def add_alias(self, name: str) -> None:
        self.aliases.add(name)
        n = normalize_name(name)
        if n:
            self.norms.add(n)


def _types_compatible(a: str, b: str) -> bool:
    return a == b or "other" in (a, b) or not a or not b


def _types_compatible_exact(a: str, b: str) -> bool:
    """For identical names, LLMs often disagree on organization vs product vs concept
    ("Velantis" the company in one chunk, the brand in another). Only person and
    location are treated as hard types, so "Jordan" the person and "Jordan" the
    country stay apart while the company/brand split does not fragment the graph."""
    if _types_compatible(a, b):
        return True
    return not ({a, b} & {"person", "location"})


def _initial_match(a: list[str], b: list[str]) -> bool:
    """["m", "okonkwo"] vs ["mira", "okonkwo"]: same surname, first-name initial agrees,
    and at least one side is abbreviated."""
    if len(a) < 2 or len(b) < 2 or a[-1] != b[-1]:
        return False
    fa, fb = a[0], b[0]
    if len(fa) == 1 or len(fb) == 1:
        return fa[0] == fb[0]
    return False


def _first_name_conflict(a: list[str], b: list[str]) -> bool:
    """Same surname, both first names spelled out, and they differ."""
    return len(a) >= 2 and len(b) >= 2 and a[-1] == b[-1] and len(a[0]) > 1 and len(b[0]) > 1 and a[0] != b[0]


class EntityResolver:
    def __init__(self, mode: str = "full", fuzzy_threshold: float = 0.92, emb_threshold: float = 0.85):
        if mode not in {"full", "exact_norm", "none"}:
            raise ValueError(mode)
        self.mode = mode
        self.fuzzy_threshold = fuzzy_threshold
        self.emb_threshold = emb_threshold
        self.entities: dict[int, EntityRecord] = {}
        self._by_raw: dict[str, int] = {}
        self._by_norm: dict[str, set[int]] = {}
        self._by_token: dict[str, set[int]] = {}
        self._by_acr: dict[str, set[int]] = {}
        self.merges = 0  # mentions folded into an existing entity by non-exact rules

    # -- index maintenance
    def load(self, rec: EntityRecord) -> None:
        self.entities[rec.id] = rec
        for a in rec.aliases:
            self._index_alias(rec.id, a)

    def _index_alias(self, eid: int, name: str) -> None:
        self._by_raw[name.strip()] = eid
        n = normalize_name(name)
        if not n:
            return
        self._by_norm.setdefault(n, set()).add(eid)
        for t in n.split():
            if t not in _GENERIC and len(t) > 1:
                self._by_token.setdefault(t, set()).add(eid)
        ac = acronym(name)
        if ac:
            self._by_acr.setdefault(ac, set()).add(eid)

    # -- scoring
    def _score(self, name: str, mtype: str, memb: np.ndarray | None, ent: EntityRecord) -> float:
        if not _types_compatible_exact(mtype, ent.type):
            return 0.0
        n = normalize_name(name)
        mt = n.split()
        # veto first: any alias with a conflicting full first name
        for en in ent.norms:
            if _first_name_conflict(mt, en.split()):
                return 0.0
        if n in ent.norms:
            return 1.0
        if not _types_compatible(mtype, ent.type):  # fuzzy rules need agreeing types
            return 0.0
        best = 0.0
        m_acr = acronym(name)
        for en in ent.norms:
            et = en.split()
            if (len(mt) == 1 and mt[0] == acronym(en)) or (len(et) == 1 and m_acr and et[0] == m_acr):
                best = max(best, 0.95)
            if _initial_match(mt, et):
                best = max(best, 0.9)
            ms = [t for t in mt if t not in _GENERIC]
            es = [t for t in et if t not in _GENERIC]
            if ms and es and (set(ms) < set(es) or set(es) < set(ms)):
                # containment counts only when the shorter side keeps the distinctive token
                short = ms if len(ms) < len(es) else es
                long_ = es if short is ms else ms
                if short[0] == long_[0] or short[-1] == long_[-1]:
                    best = max(best, 0.8)
            r = SequenceMatcher(None, n, en).ratio()
            if r >= self.fuzzy_threshold and min(len(n), len(en)) >= 5:
                best = max(best, 0.85)
            elif r >= 0.8 and memb is not None and ent.emb is not None:
                if float(memb @ ent.emb) >= self.emb_threshold:
                    best = max(best, 0.8)
        return best

    def _candidates(self, name: str) -> set[int]:
        n = normalize_name(name)
        cands: set[int] = set(self._by_norm.get(n, ()))
        toks = [t for t in n.split() if t not in _GENERIC and len(t) > 1]
        for t in toks:
            cands |= self._by_token.get(t, set())
        if len(toks) == 1:
            cands |= self._by_acr.get(toks[0], set())
        ac = acronym(name)
        if ac:
            cands |= self._by_norm.get(ac, set())
        return cands

    def _cluster_conflict(self, eid: int, name: str) -> bool:
        mt = normalize_name(name).split()
        return any(_first_name_conflict(mt, en.split()) for en in self.entities[eid].norms)

    # -- public
    def resolve(self, name: str, mtype: str, new_id: int, emb: np.ndarray | None = None) -> tuple[int, bool]:
        """Return (entity_id, created). new_id is used if a new entity is needed."""
        raw = name.strip()
        if self.mode == "none":
            if raw in self._by_raw:
                return self._by_raw[raw], False
            return self._create(new_id, raw, mtype, emb), True
        if self.mode == "exact_norm":
            hits = [e for e in self._by_norm.get(normalize_name(raw), ()) if _types_compatible_exact(mtype, self.entities[e].type)]
            if hits:
                eid = min(hits)
                self._attach(eid, raw, mtype)
                return eid, False
            return self._create(new_id, raw, mtype, emb), True

        scored = sorted(((self._score(raw, mtype, emb, self.entities[c]), c) for c in self._candidates(raw)), reverse=True)
        scored = [(s, c) for s, c in scored if s >= 0.8]
        if scored:
            top_s, top_c = scored[0]
            ambiguous = top_s < 1.0 and len(scored) > 1 and scored[1][0] >= top_s - 0.05
            if not ambiguous and not self._cluster_conflict(top_c, raw):
                if top_s < 1.0:
                    self.merges += 1
                self._attach(top_c, raw, mtype)
                return top_c, False
        return self._create(new_id, raw, mtype, emb), True

    def _create(self, eid: int, name: str, mtype: str, emb: np.ndarray | None) -> int:
        rec = EntityRecord(eid, mtype or "other", emb=emb)
        rec.add_alias(name)
        self.entities[eid] = rec
        self._index_alias(eid, name)
        return eid

    def _attach(self, eid: int, name: str, mtype: str) -> None:
        rec = self.entities[eid]
        if rec.type == "other" and mtype and mtype != "other":
            rec.type = mtype
        if name not in rec.aliases:
            rec.add_alias(name)
            self._index_alias(eid, name)


def canonical_relation(src: int, label: str, dst: int) -> tuple[int, str, int]:
    """Normalize a relation label and flip passive voice so that
    (Velantis, "was founded by", Mira) and (Mira, "founded", Velantis) dedupe."""
    words = normalize_relation(label).split()
    if len(words) >= 2 and words[-1] == "by":
        verb = words[-2]
        if verb.endswith("ed") or verb in _PASSIVE_IRREGULAR:
            words = words[:-1]
            return dst, " ".join(words), src
    return src, " ".join(words), dst
