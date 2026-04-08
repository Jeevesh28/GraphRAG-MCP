"""Text utilities: sentence splitting, chunking, and name normalization."""

from __future__ import annotations

import re
import unicodedata

# Abbreviations that end with a period but do not end a sentence.
_ABBREV = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "mt", "inc", "corp", "co",
    "ltd", "llc", "plc", "vs", "etc", "e.g", "i.e", "no", "fig", "u.s", "u.k", "jan",
    "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

_SENT_END = re.compile(r"([.!?])(\s+)(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping common abbreviations intact."""
    text = re.sub(r"[ \t]+", " ", text.strip())
    if not text:
        return []
    out: list[str] = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.end(1)
        prev = text[start:end]
        last_word = re.findall(r"([A-Za-z.]+)\.$", prev)
        if last_word:
            w = last_word[0].lower().rstrip(".")
            # single capital initial like "J." is an abbreviation too
            if w in _ABBREV or (len(w) == 1 and w.isalpha()):
                continue
        out.append(prev.strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def chunk_text(text: str, max_chars: int = 900, overlap_sentences: int = 1) -> list[str]:
    """Pack sentences into chunks of roughly max_chars, respecting paragraph breaks.

    Paragraphs are packed greedily; a paragraph longer than max_chars is split on
    sentence boundaries with a small sentence overlap so facts that straddle a
    boundary still appear whole in one chunk.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append(" ".join(buf).strip())
        buf, buf_len = [], 0

    for para in paragraphs:
        sents = split_sentences(para) or [para]
        para_len = sum(len(s) + 1 for s in sents)
        if buf_len + para_len <= max_chars:
            buf.extend(sents)
            buf_len += para_len
            continue
        flush()
        if para_len <= max_chars:
            buf, buf_len = list(sents), para_len
            continue
        # long paragraph: sentence windows with overlap
        cur: list[str] = []
        cur_len = 0
        for s in sents:
            if cur and cur_len + len(s) + 1 > max_chars:
                chunks.append(" ".join(cur))
                cur = cur[-overlap_sentences:] if overlap_sentences else []
                cur_len = sum(len(x) + 1 for x in cur)
            cur.append(s)
            cur_len += len(s) + 1
        if cur:
            buf, buf_len = cur, cur_len
    flush()
    return chunks


# ---------------------------------------------------------------- normalization

HONORIFICS = {"dr", "mr", "mrs", "ms", "miss", "prof", "professor", "sir", "dame", "the"}
ORG_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "llc", "plc", "gmbh", "ag", "sa", "group", "holdings",
}
_STOP = {"of", "and", "the", "for", "&", "a", "an", "in", "on"}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def tokens(name: str) -> list[str]:
    s = strip_accents(unicodedata.normalize("NFKC", name)).lower()
    s = s.replace("&", " and ")
    return re.findall(r"[a-z0-9]+", s)


def normalize_name(name: str) -> str:
    """Canonical key: lowercase, accents and punctuation removed, honorifics and
    corporate suffixes dropped. "Dr. Mira Okonkwo" -> "mira okonkwo",
    "Velantis Corp." -> "velantis"."""
    toks = tokens(name)
    while toks and toks[0] in HONORIFICS:
        toks = toks[1:]
    while len(toks) > 1 and toks[-1] in ORG_SUFFIXES:
        toks = toks[:-1]
    return " ".join(toks)


def acronym(name: str) -> str:
    """Initials of the significant words: "Blue Harbor Systems" -> "bhs"."""
    toks = [t for t in normalize_name(name).split() if t not in _STOP]
    return "".join(t[0] for t in toks) if len(toks) >= 2 else ""


def normalize_relation(label: str) -> str:
    """Collapse surface variation in relation labels so duplicate edges merge.

    "Was founded by", "founded-by" and "FOUNDED BY" all map to "founded by".
    """
    s = re.sub(r"[^a-z0-9 ]+", " ", strip_accents(label).lower())
    words = [w for w in s.split() if w not in {"is", "was", "were", "are", "has", "had", "have", "been", "the", "a", "an"}]
    return " ".join(words) or s.strip()
