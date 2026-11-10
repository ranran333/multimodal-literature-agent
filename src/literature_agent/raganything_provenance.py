"""Convert retrieved LightRAG chunks into verifiable, page-level evidence.

The indexing integration supplies the project's document ID as LightRAG's
``full_doc_id``. A retrieved chunk without unambiguous stored page metadata is
not suitable for a clickable PDF citation and is deliberately left out.
"""
from __future__ import annotations

from typing import Any


def page_evidence(retrieved: list[dict], stored: list[dict | None], documents: list[dict],
                  *, top_k: int = 4) -> tuple[list[dict], dict[str, int]]:
    """Map ranked chunks to distinct PDF pages without guessing missing pages.

    ``stored`` must correspond positionally to the chunk IDs in ``retrieved``,
    as returned by LightRAG's ``text_chunks.get_by_ids``. Cross-page chunks are
    excluded until their text can be attributed to individual PDF pages.
    """
    if len(retrieved) != len(stored):
        raise ValueError("Retrieved and stored chunk counts do not match")
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50")
    docs = {str(document["id"]): document for document in documents}
    if len(docs) != len(documents):
        raise ValueError("Document IDs must be unique")
    stats = {"retrieved_chunks": len(retrieved), "missing_record": 0,
             "unknown_document": 0, "missing_or_invalid_page": 0,
             "cross_page": 0, "empty_content": 0, "mapped_chunks": 0,
             "mapped_pages": 0}
    by_page: dict[tuple[str, int], dict[str, Any]] = {}
    for chunk, record in zip(retrieved, stored):
        if not isinstance(record, dict) or not chunk.get("chunk_id"):
            stats["missing_record"] += 1
            continue
        doc_id = str(record.get("full_doc_id", ""))
        doc = docs.get(doc_id)
        if doc is None:
            stats["unknown_document"] += 1
            continue
        start = record.get("page_idx")
        end = record.get("page_idx_end", start)
        if end is None:
            end = start
        page_count = doc.get("pages")
        if (isinstance(start, bool) or not isinstance(start, int) or
                isinstance(end, bool) or not isinstance(end, int) or
                isinstance(page_count, bool) or not isinstance(page_count, int) or
                start < 0 or end < start or end >= page_count):
            stats["missing_or_invalid_page"] += 1
            continue
        if end != start:
            stats["cross_page"] += 1
            continue
        content = record.get("content")
        if not isinstance(content, str) or not content.strip():
            stats["empty_content"] += 1
            continue
        stats["mapped_chunks"] += 1
        page = start + 1  # MinerU uses zero-based page_idx; PDF viewers use one-based pages.
        kind = "visual" if record.get("is_multimodal") else "text"
        part = {"id": chunk["chunk_id"], "kind": kind, "text": content.strip()}
        key = (doc_id, page)
        if key not in by_page:
            by_page[key] = {"id": chunk["chunk_id"], "document_id": doc_id, "page": page,
                            "kind": kind, "text": part["text"], "parts": [part],
                            "document": doc, "score": 0.0}
        else:
            evidence = by_page[key]
            if all(p["id"] != part["id"] for p in evidence["parts"]):
                evidence["parts"].append(part)
                evidence["text"] += "\n...\n" + part["text"]
                if evidence["kind"] != kind:
                    evidence["kind"] = "mixed"
    stats["mapped_pages"] = len(by_page)
    return list(by_page.values())[:top_k], stats
