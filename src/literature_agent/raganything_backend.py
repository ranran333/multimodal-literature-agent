"""Optional RAG-Anything backend using local Qwen and page-grounded sources.

This module imports RAG-Anything only when instantiated, so the existing web
and CLI continue to work without the optional experimental environment.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter

from .index import verify_sources
from .raganything_provenance import page_evidence

PAGE_BREAK = "[LITERATURE_AGENT_PAGE_BOUNDARY_73B2F8A9]"


def wrap_callbacks(callbacks):
    """Keep model objects out of LightRAG's dataclass deep-copy operation."""
    async def text(*args, **kwargs):
        return await callbacks.text(*args, **kwargs)

    async def vision(*args, **kwargs):
        return await callbacks.vision(*args, **kwargs)

    async def embed(*args, **kwargs):
        return await callbacks.embed(*args, **kwargs)

    return text, vision, embed


def validate_parsed_content(content: list[dict], pages: int, output_dir: Path) -> None:
    """Reject ambiguous pages or images outside the parser output."""
    if not isinstance(content, list) or not content:
        raise ValueError("The document parser returned no content")
    allowed_root = output_dir.resolve()
    for item in content:
        if not isinstance(item, dict):
            raise ValueError("The document parser returned an invalid content item")
        if item.get("type") == "text":
            if not isinstance(item.get("text"), str):
                raise ValueError("The document parser returned invalid text")
            if not item["text"].strip():
                continue
        page = item.get("page_idx")
        if isinstance(page, bool) or not isinstance(page, int) or not 0 <= page < pages:
            raise ValueError("A content item has a missing or invalid source page")
        for field in ("img_path", "table_img_path"):
            value = item.get(field)
            if value:
                if not isinstance(value, str):
                    raise ValueError("A parsed image path is invalid")
                image = Path(value)
                if not image.is_absolute() or not image.resolve().is_relative_to(allowed_root):
                    raise ValueError("A parsed image points outside the parser output directory")
                if not image.is_file():
                    raise ValueError("A parsed image file is missing")


def prepare_content(content: list[dict]) -> list[dict]:
    """Split text at page boundaries and route MinerU charts to image reading."""
    prepared = []
    previous_text_page = None
    for item in content:
        block = dict(item)
        if block.get("type") == "chart":
            block["type"] = "image"
            block["image_caption"] = block.get("chart_caption", [])
            block["image_footnote"] = block.get("chart_footnote", [])
        if block.get("type") == "text" and block["text"].strip():
            if PAGE_BREAK in block["text"]:
                raise ValueError("The source text contains the reserved page boundary")
            page = block["page_idx"]
            if previous_text_page is not None and page != previous_text_page:
                prepared.append({"type": "text", "text": PAGE_BREAK, "page_idx": page})
            previous_text_page = page
        prepared.append(block)
    return prepared


class RAGAnythingBackend:
    """A single-document experimental workspace with unambiguous page IDs."""

    def __init__(self, workspace: Path, document: dict, callbacks, *, parser_device: str = "cuda:0"):
        from lightrag.utils import EmbeddingFunc
        from raganything import RAGAnything, RAGAnythingConfig

        if not isinstance(document.get("id"), str) or not re.fullmatch(r"[0-9a-f]{16}", document["id"]):
            raise ValueError("Invalid document ID in source index")
        verify_sources([document])
        self.document = document
        self.workspace = Path(workspace).expanduser().resolve()
        self.parsed = self.workspace / "parsed"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.parsed.mkdir(parents=True, exist_ok=True)
        self.parser_device = parser_device
        manifest = self.workspace / "document.json"
        identity = {key: document[key] for key in ("id", "name", "path", "sha256", "pages")}
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != identity:
                raise ValueError("The RAG-Anything workspace belongs to a different PDF")
        else:
            with manifest.open("x", encoding="utf-8") as stream:
                json.dump(identity, stream, ensure_ascii=False, indent=2)
        config = RAGAnythingConfig(working_dir=str(self.workspace / "storage"),
                                   parser_output_dir=str(self.parsed), parser="mineru", parse_method="auto",
                                   max_concurrent_files=1, enable_image_processing=True,
                                   enable_table_processing=True, enable_equation_processing=True)
        dimension = int(callbacks.encoder.model.config.hidden_size)
        text, vision, embed = wrap_callbacks(callbacks)
        embedding = EmbeddingFunc(embedding_dim=dimension, max_token_size=256, func=embed)
        self.rag = RAGAnything(config=config, llm_model_func=text,
                               vision_model_func=vision, embedding_func=embedding,
                               lightrag_kwargs={"workspace": document["id"],
                                                "llm_model_max_async": 1, "embedding_func_max_async": 1,
                                                "max_parallel_insert": 1, "chunk_token_size": 600,
                                                "chunk_overlap_token_size": 100})

    async def ingest(self) -> dict:
        verify_sources([self.document])
        start = perf_counter()
        content, _ = await self.rag.parse_document(
            self.document["path"], output_dir=str(self.parsed), parse_method="auto",
            display_stats=False, backend="pipeline", device=self.parser_device,
        )
        validate_parsed_content(content, self.document["pages"], self.parsed)
        prepared = prepare_content(content)
        await self.rag.insert_content_list(prepared, file_path=self.document["id"], doc_id=self.document["id"],
                                           split_by_character=PAGE_BREAK)
        counts: dict[str, int] = {}
        for item in content:
            kind = str(item.get("type", "unknown"))
            counts[kind] = counts.get(kind, 0) + 1
        return {"document_id": self.document["id"], "items": len(content), "types": counts,
                "seconds": round(perf_counter() - start, 2)}

    async def search(self, question: str, *, top_k: int = 4, mode: str = "mix") -> tuple[list[dict], dict]:
        from lightrag import QueryParam

        if not question.strip():
            raise ValueError("Question cannot be empty")
        if mode not in {"local", "global", "hybrid", "mix", "naive"}:
            raise ValueError("Unsupported RAG-Anything retrieval mode")
        verify_sources([self.document])
        initialized = await self.rag._ensure_lightrag_initialized()
        if not initialized or not initialized.get("success"):
            raise RuntimeError(f"RAG-Anything initialization failed: {(initialized or {}).get('error', 'unknown')}")
        start = perf_counter()
        param = QueryParam(mode=mode, top_k=max(20, top_k * 4), chunk_top_k=max(20, top_k * 4),
                           enable_rerank=False)
        result = await self.rag.lightrag.aquery_data(question, param=param)
        if result.get("status") != "success":
            raise RuntimeError(f"RAG-Anything retrieval failed: {result.get('message', 'unknown error')}")
        chunks = result.get("data", {}).get("chunks", [])
        if not isinstance(chunks, list):
            raise RuntimeError("RAG-Anything returned invalid chunks")
        ids = [item.get("chunk_id", "") for item in chunks]
        stored = await self.rag.lightrag.text_chunks.get_by_ids(ids) if ids else []
        evidence, stats = page_evidence(chunks, stored, [self.document], top_k=top_k)
        stats.update({"seconds": round(perf_counter() - start, 3), "mode": mode})
        return evidence, stats

    async def close(self) -> None:
        if self.rag.lightrag is not None:
            await self.rag.finalize_storages()
