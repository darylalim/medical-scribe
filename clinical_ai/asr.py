"""MedASR loader and transcription helpers (no Streamlit knowledge)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import pipeline


def load_asr_pipeline(model_id: str, device: str) -> Any:
    return pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        device=device,
    )


def transcribe(
    pipe: Any,
    audio: bytes | str | Path,
    chunk_s: float = 20.0,
    stride_s: float = 2.0,
) -> str:
    if isinstance(audio, Path):
        audio = str(audio)
    result = pipe(audio, chunk_length_s=chunk_s, stride_length_s=stride_s)
    return result["text"] if isinstance(result, dict) else str(result)
