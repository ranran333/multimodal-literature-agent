"""Command-line interface for ingestion, retrieval, answering and evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .index import build_index, load_index, save_index
from .models import DEFAULT_EMBEDDING, DEFAULT_VLM
from .retrieval import search


def _files(paths: list[str]) -> list[Path]:
    found = []
    for entry in paths:
        path = Path(entry).expanduser()
        if path.is_dir():
            found.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
        else:
            found.append(path)
    return found


def _encoder(index: dict):
    if index.get("embedding_model"):
        from .models import DenseEncoder
        return DenseEncoder(index["embedding_model"])
    return None


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="paper-agent", description="Local multimodal literature QA")
    sub = p.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Create an index from PDF files or directories")
    ingest.add_argument("paths", nargs="+", help="PDF files or directories")
    ingest.add_argument("--index", default="data/processed/index.json")
    ingest.add_argument("--describe-pages", action="store_true", help="Index model descriptions of each page")
    ingest.add_argument("--dense", action="store_true", help="Add multilingual dense embeddings")
    ingest.add_argument("--model", default=DEFAULT_VLM)
    ingest.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    ingest.add_argument("--device", default="auto", help="auto, cpu, or cuda:0 (after CUDA_VISIBLE_DEVICES)")

    retrieve = sub.add_parser("search", help="Inspect retrieved pages")
    retrieve.add_argument("query")
    retrieve.add_argument("--index", default="data/processed/index.json")
    retrieve.add_argument("--top-k", type=int, default=5)

    query = sub.add_parser("ask", help="Answer using retrieved text and optional page images")
    query.add_argument("question")
    query.add_argument("--index", default="data/processed/index.json")
    query.add_argument("--model", default=DEFAULT_VLM)
    query.add_argument("--device", default="auto")
    query.add_argument("--mode", choices=("auto", "text", "vision"), default="auto")
    query.add_argument("--top-k", type=int, default=4)
    query.add_argument("--max-visual-pages", type=int, default=2)
    query.add_argument("--output", help="Write the full answer and trace as JSON")

    chat = sub.add_parser("chat", help="Interactive questions with one loaded model")
    chat.add_argument("--index", default="data/processed/index.json")
    chat.add_argument("--model", default=DEFAULT_VLM)
    chat.add_argument("--device", default="auto")
    chat.add_argument("--mode", choices=("auto", "text", "vision"), default="auto")
    chat.add_argument("--top-k", type=int, default=4)
    chat.add_argument("--max-visual-pages", type=int, default=2)

    evaluate = sub.add_parser("evaluate", help="Evaluate annotated page retrieval; optional model QA")
    evaluate.add_argument("dataset", help="JSONL with question, document, page or pages and optional answer_contains/route")
    evaluate.add_argument("--index", default="data/processed/index.json")
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--qa", action="store_true", help="Also generate answers and score string matches")
    evaluate.add_argument("--scope-document", action="store_true", help="Restrict each question to its labelled document, as when selecting one paper in the web UI")
    evaluate.add_argument("--model", default=DEFAULT_VLM)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--output", help="Write report to JSON")

    web = sub.add_parser("web", help="Launch a local web interface")
    web.add_argument("--socket", help="Private Unix socket (default: DATA_DIR/agent.sock)")
    web.add_argument("--data-dir", default="data/web", help="Private uploads and index directory")
    web.add_argument("--model", default=DEFAULT_VLM)
    web.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    web.add_argument("--device", default="auto", help="auto, cpu, or cuda:0 (after CUDA_VISIBLE_DEVICES)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "ingest":
            from .models import DenseEncoder, QwenVisionModel
            model = QwenVisionModel(args.model, args.device) if args.describe_pages else None
            encoder = DenseEncoder(args.embedding_model) if args.dense else None
            index = build_index(_files(args.paths), describe_page=model.describe if model else None,
                                embed=encoder, progress=lambda message: print(message, file=sys.stderr, flush=True))
            save_index(index, Path(args.index))
            _print({"index": args.index, "documents": len(index["documents"]), "chunks": len(index["chunks"]),
                    "visual_chunks": sum(c["kind"] == "visual" for c in index["chunks"]),
                    "embedding_model": index.get("embedding_model")})
        elif args.command == "search":
            index = load_index(Path(args.index))
            encoder = _encoder(index)
            vector = encoder([args.query])[0] if encoder else None
            hits = search(index, args.query, args.top_k, vector)
            _print([{"document": h["document"]["name"], "document_id": h["document"]["id"],
                     "page": h["page"], "kind": h["kind"],
                     "score": round(h["score"], 6), "excerpt": h["text"][:500]} for h in hits])
        elif args.command in ("ask", "chat"):
            from .agent import ask
            from .models import QwenVisionModel
            index = load_index(Path(args.index))
            model, encoder = QwenVisionModel(args.model, args.device), _encoder(index)
            if args.command == "chat":
                print("Enter a question; type 'exit' to quit. Each question is answered independently.")
                while True:
                    try:
                        question = input("\nQuestion> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                    if question.lower() in {"exit", "quit"}:
                        break
                    if question:
                        _print(ask(index, model, question, mode=args.mode, top_k=args.top_k,
                                   encoder=encoder, max_visual_pages=args.max_visual_pages))
            else:
                result = ask(index, model, args.question, mode=args.mode, top_k=args.top_k,
                             encoder=encoder, max_visual_pages=args.max_visual_pages)
                if args.output:
                    destination = Path(args.output)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                _print(result)
        elif args.command == "web":
            from .web import serve
            serve(socket_path=Path(args.socket) if args.socket else None,
                  data_dir=Path(args.data_dir), model_name=args.model,
                  embedding_model=args.embedding_model, device=args.device)
        else:
            from .evaluation import evaluate
            index = load_index(Path(args.index))
            model = None
            if args.qa:
                from .models import QwenVisionModel
                model = QwenVisionModel(args.model, args.device)
            report = evaluate(index, Path(args.dataset), top_k=args.top_k, model=model, encoder=_encoder(index),
                              document_scoped=args.scope_document,
                              progress=lambda number, total: print(f"Evaluating {number}/{total}",
                                                                   file=sys.stderr, flush=True))
            if args.output:
                destination = Path(args.output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            _print(report)
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
