"""Small inspectable evaluator; no claims of semantic correctness from string matching."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from .index import verify_sources
from .retrieval import search


def evaluate(index: dict, dataset: Path, *, top_k: int = 5, model=None, encoder=None,
             document_scoped: bool = False, progress=None) -> dict:
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    verify_sources(index["documents"])
    details = []
    for number, row in enumerate(rows, 1):
        if progress is not None:
            progress(number, len(rows))
        query = row["question"]
        pages = row.get("pages", [row.get("page")])
        if (not isinstance(pages, list) or not pages or
                any(isinstance(p, bool) or not isinstance(p, int) or p < 1 for p in pages)):
            raise ValueError("Each question must have at least one positive PDF page number")
        gold = set(pages)
        question_index = index
        if document_scoped:
            documents = [doc for doc in index["documents"] if doc["name"] == row["document"]]
            if len(documents) != 1:
                raise ValueError(f"Expected one indexed document named {row['document']!r}")
            document_id = documents[0]["id"]
            question_index = {**index, "documents": documents,
                              "chunks": [c for c in index["chunks"] if c["document_id"] == document_id]}
        start = perf_counter()
        vec = encoder([query])[0] if encoder else None
        hits = search(question_index, query, top_k, vec)
        retrieval_seconds = perf_counter() - start
        def is_gold(hit):
            return hit["document"]["name"] == row["document"] and hit["page"] in gold
        hit = any(is_gold(h) for h in hits)
        item = {"question": query, "gold": {"document": row["document"], "page": pages[0], "pages": pages},
                "retrieval_hit_at_1": bool(hits and is_gold(hits[0])),
                "retrieval_hit": hit,
                "retrieval_seconds": round(retrieval_seconds, 4),
                "retrieved": [{"document": h["document"]["name"], "page": h["page"]} for h in hits]}
        for field in ("id", "type"):
            if field in row:
                item[field] = row[field]
        if model is not None:
            from .agent import ask
            start = perf_counter()
            response = ask(question_index, model, query, encoder=encoder, top_k=top_k)
            item["qa_seconds"] = round(perf_counter() - start, 4)
            item["response"] = response
            if row.get("answer_contains"):
                item["answer_string_match"] = any(s.casefold() in response["answer"].casefold()
                                                   for s in row["answer_contains"])
            if row.get("route"):
                item["route_correct"] = response["route"] == row["route"]
            item["gold_citation_present"] = any(c["document"] == row["document"] and c["page"] in gold
                                                for c in response["citations"])
        details.append(item)
    def average(key, subset=details):
        values = [x[key] for x in subset if key in x]
        return {"value": sum(values) / len(values), "n": len(values)} if values else None
    categories = sorted({x["type"] for x in details if x.get("type")})
    per_type = {}
    for category in categories:
        subset = [x for x in details if x.get("type") == category]
        per_type[category] = {"n": len(subset), "retrieval_hit_at_1": average("retrieval_hit_at_1", subset),
                              "retrieval_hit_at_k": average("retrieval_hit", subset),
                              "answer_string_match": average("answer_string_match", subset),
                              "gold_citation_present": average("gold_citation_present", subset)}
    return {"dataset": str(dataset), "n_questions": len(rows), "top_k": top_k,
            "document_scope": "selected_document" if document_scoped else "all_documents",
            "retrieval_hit_at_1": average("retrieval_hit_at_1"),
            "retrieval_hit_at_k": average("retrieval_hit"),
            "answer_string_match": average("answer_string_match"),
            "route_accuracy": average("route_correct"),
            "gold_citation_present": average("gold_citation_present"),
            "per_type": per_type,
            "note": "String matches and cited-page presence are proxies, not semantic correctness or verified citation support.",
            "details": details}
