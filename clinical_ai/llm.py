"""MedGemma loader and streaming SOAP generator (MLX backend, no Streamlit knowledge)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from clinical_ai.prompts import format_soap_messages

DEFAULT_MODEL_ID = "mlx-community/medgemma-27b-text-it-4bit"
DEFAULT_MAX_TOKENS = 2048


def load_medgemma(model_id: str = DEFAULT_MODEL_ID) -> tuple[Any, Any]:
    result: tuple[Any, Any] = load(model_id)[:2]
    return result


def stream_soap(
    model: Any,
    tokenizer: Any,
    transcript: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.2,
    meta: dict[str, object] | None = None,
) -> Iterator[str]:
    if not transcript or not transcript.strip():
        raise ValueError("transcript must be a non-empty string")
    messages = format_soap_messages(transcript)
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    sampler = make_sampler(temp=temperature)
    last: Any = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        last = response
        yield response.text
    if meta is not None and last is not None:
        meta["finish_reason"] = getattr(last, "finish_reason", None)
