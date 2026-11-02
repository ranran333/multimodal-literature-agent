"""Small inspectable evaluator; no claims of semantic correctness from string matching."""
from __future__ import annotations

import json
from pathlib import Path

from .retrieval import search


def evaluate(index: dict, dataset: Path, *, top_k: int = 5, model=None, encoder=None) -> dict:
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    details = []
    for row in rows:
        query = row["question"]
        vec = encoder([query])[0] if encoder else None
        hits = search(index, query, top_k, vec)
        hit = any(h["document"]["name"] == row["document"] and h["page"] == row["page"] for h in hits)
        item = {"question": query, "gold": {"document": row["document"], "page": row["page"]},
                "retrieval_hit": hit,
                "retrieved": [{"document": h["document"]["name"], "page": h["page"]} for h in hits]}
        if model is not None:
            from .agent import ask
            response = ask(index, model, query, encoder=encoder, top_k=top_k)
            item["response"] = response
            if row.get("answer_contains"):
                item["answer_string_match"] = any(s.casefold() in response["answer"].casefold()
                                                   for s in row["answer_contains"])
            if row.get("route"):
                item["route_correct"] = response["route"] == row["route"]
            item["gold_citation_present"] = any(c["document"] == row["document"] and c["page"] == row["page"]
                                                for c in response["citations"])
        details.append(item)
    def average(key):
        values = [x[key] for x in details if key in x]
        return {"value": sum(values) / len(values), "n": len(values)} if values else None
    return {"dataset": str(dataset), "n_questions": len(rows), "top_k": top_k,
            "retrieval_hit_at_k": average("retrieval_hit"),
            "answer_string_match": average("answer_string_match"),
            "route_accuracy": average("route_correct"),
            "gold_citation_present": average("gold_citation_present"),
            "note": "String matches and cited-page presence are proxies, not semantic correctness or verified citation support.",
            "details": details}
