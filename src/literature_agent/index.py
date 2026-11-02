"""PDF ingestion and page-level index. Indexes are local and reproducible."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Callable

from pypdf import PdfReader


class IndexError_(ValueError):
    """A malformed index or unsupported PDF."""


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_text(text: str, size: int = 1200, overlap: int = 180) -> list[str]:
    """Split one PDF page while preserving its page number for every chunk."""
    if size < 64 or overlap < 0 or overlap >= size:
        raise ValueError("Require size >= 64 and 0 <= overlap < size")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start + size // 2, end), text.rfind(" ", start + size // 2, end))
            if boundary > start:
                end = boundary
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def render_page(path: str | Path, page_number: int, scale: float = 1.5):
    """Render a one-based PDF page to an independent Pillow image."""
    import pypdfium2 as pdfium
    document = pdfium.PdfDocument(str(path))
    try:
        if not 1 <= page_number <= len(document):
            raise IndexError_(f"PDF page {page_number} is out of bounds")
        page = document[page_number - 1]
        try:
            bitmap = page.render(scale=scale)
            try:
                image = bitmap.to_pil().convert("RGB").copy()
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        document.close()
    return image


def build_index(paths: list[Path], *, describe_page: Callable | None = None,
                embed: Callable | None = None, chunk_size: int = 1200,
                overlap: int = 180, progress: Callable[[str], None] | None = None) -> dict:
    """Create a fresh index; visual descriptions and dense embeddings are optional."""
    if not paths:
        raise IndexError_("No PDF files found")
    documents, chunks = [], []
    seen = set()
    for candidate in sorted(paths, key=lambda p: str(p)):
        path = candidate.expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise IndexError_(f"Not a readable PDF: {path}")
        fingerprint = _digest(path)
        # Document name distinguishes two differently named copies; path is local-only.
        doc_id = hashlib.sha256((path.name + ":" + fingerprint).encode()).hexdigest()[:16]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise IndexError_(f"Password-protected PDF: {path.name}")
        meta = reader.metadata
        title = str(meta.title).strip() if meta and meta.title else path.stem
        documents.append({"id": doc_id, "name": path.name, "title": title,
                          "path": str(path), "sha256": fingerprint, "pages": len(reader.pages)})
        for page_number, page in enumerate(reader.pages, 1):
            if progress is not None:
                progress(f"{path.name}: page {page_number}/{len(reader.pages)}")
            text = page.extract_text() or ""
            for number, chunk in enumerate(split_text(text, chunk_size, overlap)):
                chunks.append({"id": f"{doc_id}:p{page_number}:t{number}", "document_id": doc_id,
                               "page": page_number, "kind": "text", "text": chunk})
            if describe_page is not None:
                description = describe_page(render_page(path, page_number))
                if description and description.strip():
                    chunks.append({"id": f"{doc_id}:p{page_number}:visual", "document_id": doc_id,
                                   "page": page_number, "kind": "visual", "text": description.strip()})
    if not chunks:
        raise IndexError_("No extractable text. Use a text-based PDF or --describe-pages.")
    index = {"schema_version": 1, "documents": documents, "chunks": chunks}
    if embed is not None:
        if progress is not None:
            progress(f"Computing embeddings for {len(chunks)} chunks")
        vectors = embed([chunk["text"] for chunk in chunks])
        if len(vectors) != len(chunks):
            raise IndexError_("Embedding count does not match chunk count")
        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector
        index["embedding_model"] = getattr(embed, "model_name", None)
    return index


def save_index(index: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".index-", suffix=".json", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(index, stream, ensure_ascii=False, indent=2)
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_index(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema_version") != 1 or not isinstance(data.get("chunks"), list):
        raise IndexError_("Unsupported or invalid index")
    return data


def verify_sources(documents: list[dict]) -> None:
    """Refuse to cite or render a PDF that differs from the indexed source."""
    for document in documents:
        path = Path(document["path"])
        if not path.is_file():
            raise IndexError_(f"Indexed PDF is missing: {document['name']}. Rebuild the index.")
        if _digest(path) != document["sha256"]:
            raise IndexError_(f"Indexed PDF has changed: {document['name']}. Rebuild the index.")
