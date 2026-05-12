"""Entity / relation extraction from chunks.

Selected with KGRAG_EXTRACTOR:
  anthropic:<model>   Anthropic Messages API (ANTHROPIC_API_KEY)
  openai:<model>      any OpenAI-compatible /chat/completions endpoint
                      (OPENAI_BASE_URL, OPENAI_API_KEY; works with Ollama, vLLM)
  heuristic           offline pattern extractor (capitalized spans + connecting verb
                      phrase). Fine for tests and templated text, weak on real prose.

Default: anthropic:claude-haiku-5-5 when ANTHROPIC_API_KEY is set, else heuristic.
Results are cached on disk by (extractor, chunk text) so rebuilding a corpus or
re-running an eval does not pay for the same extraction twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .logs import get_logger
from .text import HONORIFICS, ORG_SUFFIXES, split_sentences

log = get_logger("extract")

ENTITY_TYPES = ["person", "organization", "location", "product", "event", "concept", "other"]


class LLMError(RuntimeError):
    """The provider could not be used: bad key or model, unreachable, repeated timeouts."""


class LLMOutputError(LLMError):
    """The provider answered, but the output for this chunk is not usable JSON. This is
    about the chunk, not the configuration, so it never aborts an ingest by itself."""


def _LLM_TIMEOUT() -> float:  # seconds per LLM request; raise for slow local models
    return float(os.environ.get("KGRAG_LLM_TIMEOUT", "300"))


@dataclass
class EntityMention:
    name: str
    type: str = "other"
    description: str = ""


@dataclass
class RelationMention:
    source: str
    target: str
    relation: str
    description: str = ""


@dataclass
class Extraction:
    entities: list[EntityMention] = field(default_factory=list)
    relations: list[RelationMention] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"entities": [asdict(e) for e in self.entities], "relations": [asdict(r) for r in self.relations]})

    @classmethod
    def from_obj(cls, obj: dict) -> "Extraction":
        ents = []
        for e in obj.get("entities", []) or []:
            if isinstance(e, dict) and str(e.get("name", "")).strip():
                t = str(e.get("type", "other")).lower().strip()
                ents.append(EntityMention(str(e["name"]).strip(), t if t in ENTITY_TYPES else "other", str(e.get("description", "")).strip()))
        rels = []
        for r in obj.get("relations", []) or []:
            if isinstance(r, dict) and r.get("source") and r.get("target") and r.get("relation"):
                rels.append(RelationMention(str(r["source"]).strip(), str(r["target"]).strip(), str(r["relation"]).strip(), str(r.get("description", "")).strip()))
        # every relation endpoint must exist as an entity
        known = {e.name for e in ents}
        for r in rels:
            for n in (r.source, r.target):
                if n not in known:
                    ents.append(EntityMention(n))
                    known.add(n)
        return cls(ents, rels)


# ------------------------------------------------------------------ LLM extractor

PROMPT = """Extract a knowledge graph from the text below.

Return ONLY a JSON object:
{{"entities": [{{"name": str, "type": one of {types}, "description": str}}],
  "relations": [{{"source": str, "target": str, "relation": str, "description": str}}]}}

Rules:
- Entities are specific named things (people, organizations, places, products, events, named concepts). Skip generic nouns.
- Use the most complete name that appears in the text (e.g. "Dr. Mira Okonkwo", not "she" or "Okonkwo").
- "description": one short sentence about the entity based only on this text.
- Relations are directed facts stated in the text. "relation" is a short lowercase verb phrase in active voice (e.g. "founded", "headquartered in", "acquired", "invented").
- "source" and "target" must exactly match entity names in your list.
- Resolve pronouns to the entity they refer to.

Text:
\"\"\"
{text}
\"\"\""""


def _parse_json(s: str) -> dict:
    s = s.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:  # a non-JSON reply is a failure, not an empty graph
            raise json.JSONDecodeError("no JSON object in model output", s, 0)
        return json.loads(m.group(0))


def _repair_truncated(s: str) -> dict | None:
    """Recover the complete part of a JSON object that was cut off mid-way (output
    cap reached). Cuts back to the last complete element and closes the open brackets.
    Returns None if nothing usable is found."""
    s = re.sub(r"^```(?:json)?", "", s.strip()).strip()
    start = s.find("{")
    if start < 0:
        return None
    s = s[start:]
    ends = [i for i, ch in enumerate(s) if ch in "}]"]
    for pos in reversed(ends[-400:]):
        prefix = s[: pos + 1]
        stack, in_str, esc = [], False, False
        for ch in prefix:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    break
                stack.pop()
        if in_str:
            continue
        candidate = prefix + "".join("}" if b == "{" else "]" for b in reversed(stack))
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("entities"):
            return obj
    return None


class LLMExtractor:
    def __init__(self, provider: str, model: str):
        self.provider, self.model = provider, model
        self.spec = f"{provider}:{model}"

    def _post(self, url: str, payload: dict, headers: dict, retries: int = 5) -> dict:
        """POST with exponential backoff on rate limits, overload and network errors.
        Other HTTP errors raise immediately with the response body in the message,
        so a wrong model name or key shows up as one clear error."""
        body = json.dumps(payload).encode()
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **headers})
            try:
                with urllib.request.urlopen(req, timeout=_LLM_TIMEOUT()) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code in (408, 409, 429, 500, 502, 503, 504, 529) and attempt < retries:
                    wait = float(e.headers.get("retry-after") or 0) or min(60.0, 2.0 ** attempt + random.random())
                    log.info("HTTP %d from %s, retry %d/%d in %.1fs", e.code, url, attempt + 1, retries, wait)
                    time.sleep(wait)
                    continue
                raise LLMError(f"HTTP {e.code} from {url}: {detail}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                timed_out = isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError)
                # a timeout means the model is slow or looping on this input; one more try
                # is worth it, five would stall the whole batch for half an hour
                if attempt < (1 if timed_out else retries):
                    log.info("%s from %s, retrying", "timeout" if timed_out else e, url)
                    time.sleep(min(60.0, 2.0 ** attempt + random.random()))
                    continue
                raise LLMError(f"cannot reach {url}: {e}") from None
        raise LLMError("unreachable")

    def _complete(self, prompt: str) -> tuple[str, bool]:
        """Returns (text, truncated)."""
        if self.provider == "anthropic":
            base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
            out = self._post(
                f"{base}/v1/messages",
                {"model": self.model, "max_tokens": 8192, "messages": [{"role": "user", "content": prompt}]},
                {"x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""), "anthropic-version": "2023-06-01"},
            )
            text = "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
            return text, out.get("stop_reason") == "max_tokens"
        base = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}
        # cap the output: small local models occasionally loop and repeat JSON until
        # the context fills, which shows up as timeouts and stalls a whole batch
        payload = {"model": self.model, "temperature": 0, "max_tokens": int(os.environ.get("KGRAG_MAX_TOKENS", "2048")),
                   "messages": [{"role": "user", "content": prompt}]}
        if self._json_mode:
            try:
                out = self._post(f"{base}/chat/completions", {**payload, "response_format": {"type": "json_object"}}, headers)
            except LLMError as e:
                if "HTTP 400" not in str(e):
                    raise
                self._json_mode = False  # server does not support JSON mode; rely on the prompt
                out = self._post(f"{base}/chat/completions", payload, headers)
        else:
            out = self._post(f"{base}/chat/completions", payload, headers)
        ch = out["choices"][0]
        return ch["message"]["content"] or "", ch.get("finish_reason") == "length"

    _json_mode = True

    def extract(self, text: str) -> Extraction:
        raw, truncated = self._complete(PROMPT.format(types=ENTITY_TYPES, text=text))
        try:
            obj = _parse_json(raw)
        except json.JSONDecodeError as e:
            obj = _repair_truncated(raw)
            if obj is None:
                raise LLMOutputError(f"unparseable JSON{' (output truncated)' if truncated else ''}: {e}") from None
            log.info("recovered %d entities from %s model output",
                     len(obj.get("entities") or []), "truncated" if truncated else "malformed")
        if not isinstance(obj, dict):
            raise LLMOutputError("model did not return a JSON object")
        return Extraction.from_obj(obj)


# ------------------------------------------------------------ heuristic extractor

_CAP = r"(?:[A-Z][a-zA-Z0-9'\-]*\.?|[A-Z]\.)"
_CONNECT = r"(?:of|and|de|van|von|da|&)"
_SPAN = re.compile(rf"{_CAP}(?:\s+(?:{_CONNECT}\s+)?{_CAP})*")
_SENT_START_STOP = {"The", "A", "An", "In", "On", "At", "By", "It", "Its", "This", "That", "These", "Those", "After", "Before", "When", "While", "Today", "Later", "He", "She", "They", "His", "Her", "Their", "According", "Since", "During", "Both", "Each", "Born", "Founded", "Based", "Analysts", "Local", "Further", "Details", "However", "Also"}
_LOC_PREP = re.compile(r"\b(?:in|at|near|from|to)\s*$")
_REL_STOP = {"the", "a", "an", "its", "his", "her", "their", "which", "who", "that", "also", "and", "later", "now", "then", "was", "is", "were", "are"}


class HeuristicExtractor:
    """Offline extractor: capitalized name spans are entities; the verb phrase
    between two entities in the same sentence is the relation label."""

    spec = "heuristic"

    def _entities(self, sent: str) -> list[tuple[int, int, str]]:
        spans = []
        for m in _SPAN.finditer(sent):
            words = m.group(0).split()
            start = m.start()
            while words and words[0].rstrip(".") in _SENT_START_STOP:
                start += len(words[0]) + 1
                words = words[1:]
            if not words:
                continue
            name = " ".join(words).rstrip(",;:")
            if name.endswith(".") and name.rstrip(".").split()[-1].lower() not in ORG_SUFFIXES | HONORIFICS and not re.search(r"\b[A-Z]\.$", name):
                name = name[:-1]
            if len(name) < 2 or name.lower() in {"i", "mr", "dr"}:
                continue
            spans.append((start, start + len(name), name))
        return spans

    def _type(self, name: str, left: str) -> str:
        toks = [t.lower().strip(".") for t in name.split()]
        if toks[0] in HONORIFICS - {"the"} or re.match(r"^[A-Z]\.\s", name):
            return "person"
        if toks[-1] in ORG_SUFFIXES or toks[-1] in {"systems", "labs", "industries", "technologies", "university", "institute", "bank", "foundation"}:
            return "organization"
        if _LOC_PREP.search(left):
            return "location"
        return "other"

    def extract(self, text: str) -> Extraction:
        ents: dict[str, EntityMention] = {}
        rels: list[RelationMention] = []
        for sent in split_sentences(text):
            spans = self._entities(sent)
            for s, _, name in spans:
                t = self._type(name, sent[:s])
                if name not in ents or ents[name].type == "other":
                    ents[name] = EntityMention(name, t, sent[:200])
            for (s1, e1, n1), (s2, _, n2) in zip(spans, spans[1:]):
                between = sent[e1:s2]
                between = re.sub(r",[^,]*,", " ", between)  # drop appositives ", a robotics firm,"
                words = [w for w in re.findall(r"[a-zA-Z\-]+", between.lower())]
                while words and words[0] in _REL_STOP:
                    words = words[1:]
                while words and words[-1] in _REL_STOP:
                    words = words[:-1]
                if 1 <= len(words) <= 5 and n1 != n2:
                    rels.append(RelationMention(n1, n2, " ".join(words), sent[:200]))
        return Extraction(list(ents.values()), rels)


# ------------------------------------------------------------------------ cache

class ExtractionCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=60)  # shared across processes
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()

    @staticmethod
    def key(spec: str, text: str) -> str:
        return hashlib.sha256(f"{spec}\x00{PROMPT[:200]}\x00{text}".encode()).hexdigest()

    def get(self, k: str) -> Extraction | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return Extraction.from_obj(json.loads(row[0])) if row else None

    def put(self, k: str, ex: Extraction) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (k, ex.to_json()))
            self._db.commit()


def get_extractor(spec: str | None = None):
    spec = spec or os.environ.get("KGRAG_EXTRACTOR")
    if not spec:
        if os.environ.get("ANTHROPIC_API_KEY"):
            spec = "anthropic:claude-haiku-5-5"
        else:
            log.warning("no KGRAG_EXTRACTOR and no ANTHROPIC_API_KEY: using the offline heuristic extractor "
                        "(pattern matching, not an LLM). Set KGRAG_EXTRACTOR for real use.")
            spec = "heuristic"
    if spec == "heuristic":
        return HeuristicExtractor()
    provider, _, model = spec.partition(":")
    if provider not in {"anthropic", "openai"} or not model:
        raise ValueError(f"unknown extractor spec {spec!r}")
    return LLMExtractor(provider, model)


def extract_many(extractor, texts: list[str], cache: ExtractionCache | None,
                 workers: int | None = None) -> tuple[list[Extraction], dict[int, str]]:
    """Extract from many chunks concurrently, using the cache when possible.

    A failed chunk yields an empty extraction (logged) rather than failing the whole
    ingest; it stays searchable densely. Returns (extractions, failed) where failed
    maps chunk index to "transient" (network, timeout: worth retrying later) or
    "output" (the model's answer for this chunk was unusable: retrying gives the same
    answer at temperature 0, so don't)."""
    results: list[Extraction | None] = [None] * len(texts)
    todo: list[int] = []
    for i, t in enumerate(texts):
        hit = cache.get(ExtractionCache.key(extractor.spec, t)) if cache else None
        if hit is not None:
            results[i] = hit
        else:
            todo.append(i)

    failed: dict[int, str] = {}
    workers = workers or int(os.environ.get("KGRAG_WORKERS", "8"))

    def one(i: int) -> Extraction:
        ex = extractor.extract(texts[i])
        if cache:
            cache.put(ExtractionCache.key(extractor.spec, texts[i]), ex)
        return ex

    def run(i: int) -> tuple[int, Extraction]:
        try:
            return i, one(i)
        except Exception as exc:  # already retried; failures are not cached
            failed[i] = "output" if isinstance(exc, LLMOutputError) else "transient"
            log.warning("extraction failed for chunk %d: %s", i, exc)
            return i, Extraction()

    if todo:
        # Preflight: the first call runs alone, and a provider error (bad key, model
        # name, endpoint) aborts the ingest instead of producing an empty graph. A bad
        # answer for that one chunk is not a provider error, so it does not abort.
        try:
            results[todo[0]] = one(todo[0])
        except LLMOutputError as exc:
            failed[todo[0]] = "output"
            log.warning("extraction failed for chunk %d: %s", todo[0], exc)
        rest = todo[1:]
        if isinstance(extractor, HeuristicExtractor) or workers <= 1:
            for i in rest:
                results[i] = run(i)[1]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for i, ex in pool.map(run, rest):
                    results[i] = ex
        if len(failed) > max(3, len(todo) // 4):
            raise LLMError(f"{len(failed)} of {len(todo)} extractions failed; aborting ingest (nothing was written)")
        if failed:
            n_out = sum(v == "output" for v in failed.values())
            log.warning("%d/%d chunks failed extraction (%d transient, retried on the next add; %d unusable model "
                        "output, not retried). All stay searchable densely.",
                        len(failed), len(todo), len(failed) - n_out, n_out)
    return [r or Extraction() for r in results], failed
