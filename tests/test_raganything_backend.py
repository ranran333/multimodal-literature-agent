from copy import deepcopy
from threading import Lock

import pytest

from literature_agent.raganything_backend import PAGE_BREAK, prepare_content, validate_parsed_content, wrap_callbacks


def test_light_rag_can_copy_callbacks_without_copying_model_locks():
    class Callbacks:
        lock = Lock()

        async def text(self, *_args, **_kwargs):
            return "answer"

        vision = text
        embed = text

    callbacks = wrap_callbacks(Callbacks())
    assert all(deepcopy(callback) is callback for callback in callbacks)


def test_parser_page_metadata_and_image_paths_are_checked(tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"test")
    validate_parsed_content([{"type": "image", "page_idx": 0, "img_path": str(image)}], 2, tmp_path)
    with pytest.raises(ValueError, match="source page"):
        validate_parsed_content([{"type": "image", "img_path": str(image)}], 2, tmp_path)
    with pytest.raises(ValueError, match="outside"):
        validate_parsed_content([{"type": "image", "page_idx": 0, "img_path": "/tmp/outside.png"}],
                                2, tmp_path)


def test_parser_empty_content_and_invalid_page_rejected(tmp_path):
    with pytest.raises(ValueError, match="no content"):
        validate_parsed_content([], 2, tmp_path)
    with pytest.raises(ValueError, match="source page"):
        validate_parsed_content([{"type": "table", "page_idx": 2}], 2, tmp_path)


def test_chart_becomes_image_and_text_is_split_between_pages(tmp_path):
    image = tmp_path / "chart.png"
    image.write_bytes(b"test")
    original = [{"type": "text", "text": "first page", "page_idx": 0},
                {"type": "text", "text": "second page", "page_idx": 1},
                {"type": "chart", "img_path": str(image), "page_idx": 1,
                 "chart_caption": ["Figure 1"]}]
    validate_parsed_content(original, 2, tmp_path)
    prepared = prepare_content(original)
    assert prepared[1] == {"type": "text", "text": PAGE_BREAK, "page_idx": 1}
    assert prepared[-1]["type"] == "image"
    assert prepared[-1]["image_caption"] == ["Figure 1"]
    assert original[-1]["type"] == "chart"


def test_parser_rejects_unattributed_chart_or_text(tmp_path):
    for block in [{"type": "chart", "img_path": "/tmp/example"},
                  {"type": "text", "text": "No source page"}]:
        with pytest.raises(ValueError, match="source page"):
            validate_parsed_content([block], 2, tmp_path)
    with pytest.raises(ValueError, match="invalid text"):
        validate_parsed_content([{"type": "text", "text": None, "page_idx": 0}], 2, tmp_path)
