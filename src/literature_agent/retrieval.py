"""Lexical BM25 and optional semantic retrieval with reciprocal rank fusion."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+", re.I)
_STOP_WORDS = frozenset(
    "a an and are as at be by did do does for from how in into is it its of on or "
    "that the their these this to was were what when where which who why with".split()
)


def tokens(text: str) -> list[str]:
    result = []
    for match in _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower()):
        if "\u3400" <= match[0] <= "\u9fff":
            result.extend(match)
            result.extend(match[i:i + 2] for i in range(len(match) - 1))
        elif match not in _STOP_WORDS:
            # A small, deterministic English suffix normalization; not a full stemmer.
            if len(match) > 5 and match.endswith("ing"):
                match = match[:-3]
            elif len(match) > 4 and match.endswith("ed"):
                match = match[:-2]
            elif len(match) > 4 and match.endswith("s"):
                match = match[:-1]
            result.append(match)
    return result


def lexical_scores(chunks: list[dict], query: str) -> list[float]:
    terms = set(tokens(query))
    corpus = [tokens(c["text"]) for c in chunks]
    if not corpus or not terms:
        return [0.0] * len(corpus)
    avg = sum(map(len, corpus)) / len(corpus) or 1
    df = Counter(t for doc in corpus for t in set(doc) if t in terms)
    scores = []
    for doc in corpus:
        counts = Counter(doc)
        score = 0.0
        for term in terms:
            freq = counts[term]
            if freq:
                idf = math.log(1 + (len(corpus) - df[term] + .5) / (df[term] + .5))
                score += idf * freq * 2.5 / (freq + 1.5 * (.25 + .75 * len(doc) / avg))
        scores.append(score)
    return scores


def _weak_cross_language_lexical_signal(chunks: list[dict], query: str) -> bool:
    """Identify CJK queries with no CJK match and only very common Latin terms."""
    terms = set(tokens(query))
    cjk = {t for t in terms if len(t) > 1 and "\u3400" <= t[0] <= "\u9fff"}
    if not cjk or not chunks:
        return False
    documents = [set(tokens(chunk["text"])) for chunk in chunks]
    if any(cjk & document for document in documents):
        return False
    latin = {t for t in terms if t and t[0].isascii()}
    for term in latin:
        df = sum(term in document for document in documents)
        if 0 < df <= max(1, len(documents) // 4):
            return False  # Keep a rare exact anchor, such as a dataset name.
    return True


def search(index: dict, query: str, top_k: int = 5, query_vector: list[float] | None = None) -> list[dict]:
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50")
    chunks = index["chunks"]
    docs = {d["id"]: d for d in index["documents"]}
    lexical = lexical_scores(chunks, query)
    lexical_ranking = [i for i in sorted(range(len(chunks)), key=lambda i: (-lexical[i], i)) if lexical[i] > 0]
    rankings = [lexical_ranking]
    if query_vector is not None:
        norms = math.sqrt(sum(x * x for x in query_vector))
        dense = []
        for i, chunk in enumerate(chunks):
            vector = chunk.get("embedding")
            if not vector or len(vector) != len(query_vector):
                continue
            den = norms * math.sqrt(sum(x*x for x in vector))
            sim = sum(x*y for x, y in zip(query_vector, vector)) / den if den else 0
            if sim >= 0.2:
                dense.append((i, sim))
        dense_ranking = [i for i, _ in sorted(dense, key=lambda pair: (-pair[1], pair[0]))]
        # RRF gives a common matched acronym the same influence as strong
        # lexical evidence. Prefer dense retrieval when the query's CJK terms
        # do not occur in the corpus and no rare exact anchor is available.
        if dense_ranking and _weak_cross_language_lexical_signal(chunks, query):
            rankings = []
        rankings.append(dense_ranking)
    # Fuse at page level: lexical and semantic retrieval can match different
    # chunks on the same page, and both are valid evidence for that page.
    fused: dict[tuple[str, int], float] = {}
    candidates: dict[tuple[str, int], dict[int, float]] = {}
    first_index: dict[tuple[str, int], int] = {}
    for ranking in rankings:
        seen = set()
        for i in ranking:
            chunk = chunks[i]
            page_key = (chunk["document_id"], chunk["page"])
            if page_key in seen:
                continue
            seen.add(page_key)
            weight = 1 / (60 + len(seen))
            fused[page_key] = fused.get(page_key, 0.0) + weight
            choices = candidates.setdefault(page_key, {})
            choices[i] = choices.get(i, 0.0) + weight
            first_index[page_key] = min(first_index.get(page_key, i), i)
    result = []
    for page_key in sorted(fused, key=lambda key: (-fused[key], first_index[key])):
        choices = candidates[page_key]
        selected = sorted(choices, key=lambda i: (-choices[i], i))
        # Preserve one adjacent text chunk when only one ranking selected a
        # passage. Page splits can otherwise leave out the sentence it follows.
        if len(selected) == 1 and chunks[selected[0]]["kind"] == "text":
            for neighbor in (selected[0] - 1, selected[0] + 1):
                if 0 <= neighbor < len(chunks):
                    item = chunks[neighbor]
                    if (item["document_id"], item["page"]) == page_key and item["kind"] == "text":
                        selected.append(neighbor)
                        break
        parts = [{"id": chunks[i]["id"], "kind": chunks[i]["kind"], "text": chunks[i]["text"]}
                 for i in selected]
        chunk = chunks[selected[0]]
        kinds = {part["kind"] for part in parts}
        result.append({**chunk, "kind": next(iter(kinds)) if len(kinds) == 1 else "mixed",
                       "text": "\n...\n".join(part["text"] for part in parts), "parts": parts,
                       "document": docs[chunk["document_id"]], "score": fused[page_key]})
        if len(result) == top_k:
            break
    return result
