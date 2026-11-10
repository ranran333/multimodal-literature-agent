#!/usr/bin/env python3
"""Run an optional, local RAG-Anything experiment in isolated workspaces."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from literature_agent.agent import ask
from literature_agent.index import load_index, verify_sources
from literature_agent.local_rag_models import LocalRAGModels
from literature_agent.models import DenseEncoder, QwenVisionModel
from literature_agent.raganything_backend import RAGAnythingBackend


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="Existing page index containing the source documents")
    parser.add_argument("--workspace", default="data/raganything/workspaces")
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--embedding-model", default="models/multilingual-MiniLM")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--parser-device", default="cuda:0")
    sub = parser.add_subparsers(dest="action", required=True)
    ingest = sub.add_parser("ingest", help="Parse and index one PDF")
    ingest.add_argument("--document", required=True, help="Exact source file name in the existing index")
    query = sub.add_parser("ask", help="Retrieve pages and answer one question")
    query.add_argument("--document", required=True)
    query.add_argument("question")
    query.add_argument("--mode", choices=("auto", "text", "vision"), default="auto")
    evaluate = sub.add_parser("evaluate", help="Evaluate the same labelled questions as the baseline")
    evaluate.add_argument("dataset")
    evaluate.add_argument("--output", required=True, help="New JSON output path")
    evaluate.add_argument("--qa", action="store_true", help="Also run the existing Agent on retrieved evidence")
    return parser.parse_args()


def write_report(path: Path, report: dict) -> None:
    """Replace only the output file created by this run, keeping a readable checkpoint."""
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def summary(rows: list[dict], expected: int) -> dict:
    def count(key, subset):
        return {"n": len(subset), "count": sum(bool(row.get(key)) for row in subset)}
    by_type = {}
    for category in sorted({row.get("type") for row in rows if row.get("type")}):
        subset = [row for row in rows if row.get("type") == category]
        by_type[category] = {"hit_at_1": count("hit_at_1", subset),
                             "hit_at_4": count("hit_at_4", subset)}
    return {"completed": len(rows), "expected": expected, "errors": sum("error" in r for r in rows),
            "hit_at_1": count("hit_at_1", rows), "hit_at_4": count("hit_at_4", rows),
            "by_type": by_type}


async def run(args):
    # MinerU is a subprocess; without activating the environment its executable
    # would otherwise not be on PATH even when this Python is the correct one.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    root = Path(args.workspace).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(root.parent / "tiktoken-cache"))
    index = load_index(Path(args.index))
    verify_sources(index["documents"])
    by_name = {}
    for document in index["documents"]:
        if document["name"] in by_name:
            raise ValueError("Experimental single-paper workspaces require unique document names")
        by_name[document["name"]] = document
    model = QwenVisionModel(args.model, args.device)
    encoder = DenseEncoder(args.embedding_model)
    callbacks = LocalRAGModels(model, encoder)
    backends: dict[str, RAGAnythingBackend] = {}

    def backend(name):
        if name not in by_name:
            raise ValueError(f"Document is not present in the source index: {name}")
        doc = by_name[name]
        if name not in backends:
            backends[name] = RAGAnythingBackend(root / doc["id"], doc, callbacks,
                                                parser_device=args.parser_device)
        return backends[name]

    try:
        if args.action == "ingest":
            result = await backend(args.document).ingest()
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        elif args.action == "ask":
            document = by_name[args.document]
            selected = {**index, "documents": [document],
                        "chunks": [c for c in index["chunks"] if c["document_id"] == document["id"]]}
            evidence, provenance = await backend(args.document).search(args.question)
            response = await asyncio.to_thread(ask, selected, model, args.question, mode=args.mode,
                                               retrieved_evidence=evidence, retrieval_backend="raganything")
            print(json.dumps({"response": response, "provenance": provenance}, ensure_ascii=False, indent=2))
        else:
            dataset = Path(args.dataset)
            questions = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not questions:
                raise ValueError("Evaluation dataset is empty")
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            # Fail before the experiment if the requested output already exists.
            with output.open("x", encoding="utf-8") as stream:
                stream.write("{}\n")
            rows: list[dict] = []
            report = {"created_utc": datetime.now(timezone.utc).isoformat(),
                      "backend": "raganything", "document_scope": "selected_document", "top_k": 4,
                      "dataset": str(dataset), "model": args.model, "embedding_model": args.embedding_model,
                      "note": "Exploratory set; matching a page or answer string does not prove citation support.",
                      "summary": summary(rows, len(questions)), "details": rows}
            write_report(output, report)
            for number, item in enumerate(questions, 1):
                print(f"Evaluating {number}/{len(questions)}: {item.get('id', number)}", file=sys.stderr, flush=True)
                doc = by_name[item["document"]]
                pages = item.get("pages", [item.get("page")])
                row = {"id": item.get("id"), "type": item.get("type"), "question": item["question"],
                       "gold": {"document": doc["name"], "pages": pages}, "hit_at_1": False, "hit_at_4": False}
                start = perf_counter()
                try:
                    evidence, provenance = await backend(doc["name"]).search(item["question"])
                    row["provenance"] = provenance
                    row["retrieved"] = [{"document": hit["document"]["name"], "page": hit["page"]} for hit in evidence]
                    row["hit_at_1"] = bool(evidence and evidence[0]["page"] in pages)
                    row["hit_at_4"] = any(hit["page"] in pages for hit in evidence[:4])
                    if args.qa:
                        selected = {**index, "documents": [doc],
                                    "chunks": [c for c in index["chunks"] if c["document_id"] == doc["id"]]}
                        response = await asyncio.to_thread(ask, selected, model, item["question"],
                                                           retrieved_evidence=evidence,
                                                           retrieval_backend="raganything")
                        row["response"] = response
                        row["answer_string_match"] = any(x.casefold() in response["answer"].casefold()
                                                         for x in item.get("answer_contains", []))
                        row["gold_citation_present"] = any(c["document"] == doc["name"] and c["page"] in pages
                                                           for c in response["citations"])
                except (ValueError, RuntimeError, OSError) as exc:
                    row["error"] = str(exc)
                row["seconds"] = round(perf_counter() - start, 3)
                rows.append(row)
                report["summary"] = summary(rows, len(questions))
                write_report(output, report)
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    finally:
        for item in backends.values():
            await item.close()


def main():
    try:
        asyncio.run(run(arguments()))
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
