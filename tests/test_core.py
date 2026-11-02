import json
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from literature_agent.agent import _parse_route, ask
from literature_agent.evaluation import evaluate
from literature_agent.index import IndexError_, build_index, load_index, render_page, save_index, split_text
from literature_agent.retrieval import search, tokens


@pytest.fixture
def index(tmp_path):
    path = tmp_path / "paper.pdf"
    c = canvas.Canvas(str(path))
    c.setTitle("Test Document")
    c.drawString(40, 750, "The laboratory measured 120 molecules in the experiment.")
    c.showPage()
    c.drawString(40, 750, "Figure 1 compares a blue triangle and a red square.")
    c.showPage()
    c.save()
    return build_index([path])


def test_page_provenance_and_persistence(index, tmp_path):
    target = tmp_path / "index.json"
    save_index(index, target)
    loaded = load_index(target)
    assert loaded["documents"][0]["pages"] == 2
    assert search(loaded, "120 molecules", top_k=1)[0]["page"] == 1
    assert search(loaded, "blue triangle", top_k=1)[0]["page"] == 2
    assert search(loaded, "completelyunmatchedterm") == []


def test_render_page_and_boundaries(index):
    path = index["documents"][0]["path"]
    image = render_page(path, 1)
    assert image.width > 100 and image.height > 100
    with pytest.raises(IndexError_):
        render_page(path, 3)


def test_splitting_and_chinese_tokens():
    parts = split_text("a" * 600, size=100, overlap=20)
    assert len(parts) > 1
    assert all(len(p) <= 100 for p in parts)
    assert "文献" in tokens("多模态文献检索")
    assert "fine" in tokens("ﬁne-tuning")
    assert set(tokens("fine-tuned")) == set(tokens("ﬁne-tuning"))
    with pytest.raises(ValueError):
        split_text("text", size=100, overlap=100)


def test_visual_route_trace_and_valid_citation(index):
    class FakeModel:
        def generate(self, prompt, images=None, max_new_tokens=0):
            if "Return ONLY JSON" in prompt:
                return '{"needs_vision": true}'
            if images:
                return "The blue triangle is visible."
            return "A blue triangle is visible [E1]."
    result = ask(index, FakeModel(), "In Figure 1, what is blue?", top_k=1)
    assert result["route"] == "vision"
    assert result["citations"][0]["page"] == 2
    assert result["citations"][0]["document_id"] == index["documents"][0]["id"]
    assert [t["tool"] for t in result["trace"]] == ["retrieve", "route", "read_page_image", "answer"]


def test_invalid_citations_and_no_evidence_are_not_presented_as_grounded(index):
    class FakeModel:
        def generate(self, prompt, images=None, max_new_tokens=0):
            return "This claim has a fabricated citation [E9]."
    result = ask(index, FakeModel(), "120 molecules", mode="text", top_k=1)
    assert result["citations"] == []
    assert result["trace"][-1]["status"] == "invalid_citations"
    result = ask(index, FakeModel(), "unmatchedtermxyz", mode="text")
    assert result["citations"] == []
    assert result["route"] == "none"


def test_structured_model_output_adds_only_valid_source_labels(index):
    class StructuredModel:
        def generate(self, prompt, images=None, max_new_tokens=0):
            return '```json\n{"answer": "There are 120 molecules.", "citations": ["E1"]}\n```'
    result = ask(index, StructuredModel(), "120 molecules", mode="text", top_k=1)
    assert result["answer"] == "There are 120 molecules. [E1]"
    assert result["citations"][0]["page"] == 1


def test_evaluation_reports_counts_and_page_hits(index, tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text(json.dumps({"question": "120 molecules", "document": "paper.pdf", "page": 1}) + "\n")
    report = evaluate(index, data, top_k=1)
    assert report["retrieval_hit_at_k"] == {"value": 1.0, "n": 1}
    assert report["answer_string_match"] is None


def test_visual_description_becomes_retrievable(index, tmp_path):
    path = Path(index["documents"][0]["path"])
    enriched = build_index([path], describe_page=lambda image: "histogram with turquoise dragonfruit")
    hits = search(enriched, "turquoise dragonfruit", top_k=2)
    assert hits and all(hit["kind"] == "visual" for hit in hits)


def test_hybrid_fuses_different_chunks_on_the_same_page():
    documents = [{"id": "d", "name": "paper.pdf"}]
    chunks = [
        {"id": "a", "document_id": "d", "page": 1, "kind": "text", "text": "alpha", "embedding": [0, 1]},
        {"id": "b", "document_id": "d", "page": 1, "kind": "text", "text": "semantic evidence", "embedding": [1, 0]},
        {"id": "c", "document_id": "d", "page": 2, "kind": "text", "text": "alpha", "embedding": [.6, .8]},
    ]
    hits = search({"documents": documents, "chunks": chunks}, "alpha", top_k=2, query_vector=[1, 0])
    assert hits[0]["page"] == 1
    assert {part["id"] for part in hits[0]["parts"]} == {"a", "b"}


def test_cross_language_query_prefers_dense_without_a_rare_exact_anchor():
    documents = [{"id": "d", "name": "paper.pdf"}]
    chunks = [
        {"id": "a", "document_id": "d", "page": 1, "kind": "text", "text": "Wikipedia knowledge source", "embedding": [0, 1]},
        {"id": "a2", "document_id": "d", "page": 1, "kind": "text", "text": "retrieval model", "embedding": [1, 0]},
        {"id": "b", "document_id": "d", "page": 2, "kind": "text", "text": "RAG RAG NQ", "embedding": [.5, .866]},
        {"id": "c", "document_id": "d", "page": 3, "kind": "text", "text": "RAG results", "embedding": [.4, .9165]},
        {"id": "d", "document_id": "d", "page": 4, "kind": "text", "text": "RAG model", "embedding": [.3, .954]},
    ]
    hits = search({"documents": documents, "chunks": chunks}, "RAG 模型的记忆来源", top_k=1, query_vector=[1, 0])
    assert hits[0]["page"] == 1
    assert "Wikipedia" in hits[0]["text"]
    anchored = search({"documents": documents, "chunks": chunks}, "NQ 数据集分数", top_k=1, query_vector=[1, 0])
    assert anchored[0]["page"] == 2


def test_routing_parser_and_invalid_query(index):
    assert _parse_route("unparseable", "Which curve is higher in the figure?") == ("vision", "keyword_fallback")
    assert _parse_route('{"needs_vision": false}', "figure") == ("text", "model")
    with pytest.raises(ValueError):
        search(index, " ")


def test_agent_rejects_changed_source_pdf(index):
    path = Path(index["documents"][0]["path"])
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(IndexError_, match="changed"):
        ask(index, object(), "120 molecules", mode="text", top_k=1)


def test_chat_loads_model_once_for_multiple_questions(index, tmp_path, monkeypatch, capsys):
    from literature_agent import cli, models
    target = tmp_path / "index.json"
    save_index(index, target)
    loaded = []
    class FakeModel:
        def __init__(self, *args):
            loaded.append(True)
        def generate(self, prompt, images=None, max_new_tokens=0):
            return "The study measured 120 molecules [E1]."
    monkeypatch.setattr(models, "QwenVisionModel", FakeModel)
    inputs = iter(["120 molecules", "120 molecules", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert cli.main(["chat", "--index", str(target), "--mode", "text", "--top-k", "1"]) == 0
    assert len(loaded) == 1
    assert capsys.readouterr().out.count('"citations"') == 2
