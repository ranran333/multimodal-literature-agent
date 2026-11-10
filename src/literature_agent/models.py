"""Local Hugging Face models; no network inference service is required."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_VLM = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def project_cache() -> None:
    # Honor an explicit user setting, otherwise keep downloads inside the project.
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class QwenVisionModel:
    def __init__(self, model_name: str = DEFAULT_VLM, device: str = "auto"):
        project_cache()
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.model_name = model_name
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=256 * 28 * 28,
                                                       max_pixels=1024 * 28 * 28)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, dtype=dtype, low_cpu_mem_usage=True
        ).to(device).eval()

    def generate(self, prompt: str, images: list | None = None, max_new_tokens: int = 320) -> str:
        parts = [{"type": "image", "image": image} for image in (images or [])]
        parts.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": parts}]
        return self.generate_messages(messages, max_new_tokens=max_new_tokens)

    def generate_messages(self, messages: list[dict], max_new_tokens: int = 320) -> str:
        """Generate from system/history/vision messages in the processor's native format."""
        batch = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt")
        batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        prefix = batch["input_ids"].shape[1]
        return self.processor.batch_decode(generated[:, prefix:], skip_special_tokens=True)[0].strip()

    def describe(self, image) -> str:
        return self.generate(
            "Describe the figures, charts and tables on this scientific paper page. "
            "Include visible captions, labels, axes, trends and numbers. "
            "Do not guess values that are not legible. If none are visible, say so.",
            [image], max_new_tokens=400,
        )


class DenseEncoder:
    """Mean-pooled multilingual text embeddings, computed on CPU by default."""
    def __init__(self, model_name: str = DEFAULT_EMBEDDING):
        project_cache()
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.model_name = torch, model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()

    def __call__(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        outputs = []
        for offset in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[offset:offset + batch_size], padding=True, truncation=True,
                                   max_length=256, return_tensors="pt")
            with self.torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.extend(pooled.tolist())
        return outputs
