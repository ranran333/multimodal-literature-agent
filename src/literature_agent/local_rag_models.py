"""Adapt local Qwen and MiniLM models to RAG-Anything's async callbacks."""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import threading


def _decode_image(value: str):
    from PIL import Image

    if not isinstance(value, str):
        raise ValueError("Image data must be a base64 string")
    if value.startswith("data:"):
        header, separator, value = value.partition(",")
        if not separator or ";base64" not in header or not header.startswith("data:image/"):
            raise ValueError("Unsupported image data URL")
    if len(value) > 40 * 1024 * 1024:
        raise ValueError("Image data is too large")
    try:
        raw = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            return image.convert("RGB").copy()
    except (ValueError, OSError, binascii.Error) as exc:
        raise ValueError("Invalid base64 image") from exc


def _content(value) -> list[dict]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if not isinstance(value, list):
        raise ValueError("Message content must be text or a list of parts")
    result = []
    for part in value:
        if not isinstance(part, dict):
            raise ValueError("Invalid message part")
        if part.get("type") in {"text", "input_text"} and isinstance(part.get("text"), str):
            result.append({"type": "text", "text": part["text"]})
        elif part.get("type") in {"image_url", "input_image"}:
            url = part.get("image_url", "")
            if isinstance(url, dict):
                url = url.get("url", "")
            # Local models must not fetch arbitrary URLs supplied by document content.
            if not isinstance(url, str) or not url.startswith("data:image/"):
                raise ValueError("Only embedded image data is supported")
            result.append({"type": "image", "image": _decode_image(url)})
        else:
            raise ValueError("Unsupported message part")
    return result


def prepare_messages(prompt: str, *, system_prompt: str | None = None,
                     history_messages: list | None = None, image_data: str | None = None,
                     messages: list | None = None) -> list[dict]:
    """Convert RAG-Anything callbacks to the Qwen processor's message format."""
    raw = list(messages) if messages is not None else list(history_messages or [])
    if messages is None:
        parts = []
        if image_data is not None:
            parts.append({"type": "image", "image": _decode_image(image_data)})
        parts.append({"type": "text", "text": str(prompt)})
        raw.append({"role": "user", "content": parts})
    result = []
    if system_prompt:
        result.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    for position, item in enumerate(raw):
        if not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("Unsupported conversation role")
        content = item.get("content", "")
        # Only our own final message can contain pre-decoded Pillow images.
        if messages is None and position == len(raw) - 1:
            parts = content
        else:
            parts = _content(content)
        result.append({"role": item["role"], "content": parts})
    if not result or result[-1]["role"] != "user":
        raise ValueError("The final message must be from the user")
    return result


class LocalRAGModels:
    """Share one local vision model and one local encoder between callbacks."""

    def __init__(self, model, encoder):
        self.model, self.encoder = model, encoder
        self._model_lock, self._encoder_lock = threading.Lock(), threading.Lock()

    def _generate(self, messages, limit):
        with self._model_lock:
            return self.model.generate_messages(messages, max_new_tokens=limit)

    async def text(self, prompt, system_prompt=None, history_messages=None, **kwargs):
        if kwargs.get("stream"):
            raise ValueError("Streaming is not supported by the local model adapter")
        messages = prepare_messages(prompt, system_prompt=system_prompt, history_messages=history_messages)
        requested = kwargs.get("max_tokens", 1024)
        limit = min(max(requested, 1), 1024) if isinstance(requested, int) else 1024
        return await asyncio.to_thread(self._generate, messages, limit)

    async def vision(self, prompt, system_prompt=None, history_messages=None,
                     image_data=None, messages=None, **kwargs):
        if kwargs.get("stream"):
            raise ValueError("Streaming is not supported by the local model adapter")
        converted = prepare_messages(prompt, system_prompt=system_prompt, history_messages=history_messages,
                                     image_data=image_data, messages=messages)
        requested = kwargs.get("max_tokens", 768)
        limit = min(max(requested, 1), 1024) if isinstance(requested, int) else 768
        return await asyncio.to_thread(self._generate, converted, limit)

    def _embed(self, texts):
        import numpy as np
        with self._encoder_lock:
            return np.asarray(self.encoder(texts), dtype=np.float32)

    async def embed(self, texts, **_kwargs):
        return await asyncio.to_thread(self._embed, texts)
