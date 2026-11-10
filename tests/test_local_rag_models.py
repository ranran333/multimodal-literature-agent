import asyncio
import base64
import io

import pytest
from PIL import Image

from literature_agent.local_rag_models import LocalRAGModels, prepare_messages


def test_local_callbacks_preserve_system_history_and_vision():
    class Model:
        def generate_messages(self, messages, max_new_tokens):
            self.messages = messages
            self.limit = max_new_tokens
            return "local response"
    model = Model()
    callbacks = LocalRAGModels(model, lambda texts: [[1.0, 0.0] for _ in texts])
    image = io.BytesIO()
    Image.new("RGB", (2, 2), "green").save(image, format="PNG")
    data = base64.b64encode(image.getvalue()).decode()
    result = asyncio.run(callbacks.vision("question", system_prompt="system", image_data=data))
    assert result == "local response"
    assert [m["role"] for m in model.messages] == ["system", "user"]
    assert model.messages[-1]["content"][0]["image"].size == (2, 2)
    assert model.messages[-1]["content"][1]["text"] == "question"
    vectors = asyncio.run(callbacks.embed(["one", "two"]))
    assert vectors.shape == (2, 2)


def test_local_callbacks_reject_remote_images():
    with pytest.raises(ValueError, match="embedded image"):
        prepare_messages("", messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
        ]}])
    with pytest.raises(ValueError, match="Unsupported message part"):
        prepare_messages("", messages=[{"role": "user", "content": [
            {"type": "image", "image": "https://example.com/image.png"}
        ]}])
