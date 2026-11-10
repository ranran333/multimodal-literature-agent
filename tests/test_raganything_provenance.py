import pytest

from literature_agent.raganything_provenance import page_evidence


def test_page_evidence_preserves_document_and_one_based_page():
    docs = [{"id": "a", "name": "paper-a.pdf", "pages": 3},
            {"id": "b", "name": "paper-b.pdf", "pages": 3}]
    retrieved = [{"chunk_id": "x"}, {"chunk_id": "y"}, {"chunk_id": "z"}]
    stored = [{"full_doc_id": "a", "page_idx": 0, "content": "first"},
              {"full_doc_id": "a", "page_idx": 0, "content": "figure", "is_multimodal": True},
              {"full_doc_id": "b", "page_idx": 0, "content": "other paper"}]
    evidence, stats = page_evidence(retrieved, stored, docs)
    assert [(e["document_id"], e["page"]) for e in evidence] == [("a", 1), ("b", 1)]
    assert evidence[0]["kind"] == "mixed" and len(evidence[0]["parts"]) == 2
    assert stats["mapped_chunks"] == 3 and stats["mapped_pages"] == 2


def test_page_evidence_does_not_guess_ambiguous_or_invalid_pages():
    docs = [{"id": "a", "name": "paper.pdf", "pages": 3}]
    retrieved = [{"chunk_id": str(i)} for i in range(6)]
    stored = [None, {"full_doc_id": "a", "content": "no page"},
              {"full_doc_id": "a", "page_idx": 1, "page_idx_end": 2, "content": "two pages"},
              {"full_doc_id": "other", "page_idx": 0, "content": "wrong document"},
              {"full_doc_id": "a", "page_idx": 3, "content": "out of range"},
              {"full_doc_id": "a", "page_idx": True, "content": "boolean page"}]
    evidence, stats = page_evidence(retrieved, stored, docs)
    assert evidence == []
    assert stats["missing_record"] == 1 and stats["unknown_document"] == 1
    assert stats["cross_page"] == 1 and stats["missing_or_invalid_page"] == 3


def test_page_evidence_rejects_misaligned_chunk_records():
    with pytest.raises(ValueError, match="counts do not match"):
        page_evidence([{"chunk_id": "x"}], [], [])
