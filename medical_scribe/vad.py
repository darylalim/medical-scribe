"""Silero VAD silence-trimming helpers (no Streamlit knowledge).

Operates purely on bytes-in / bytes-out: decode once via librosa, run VAD,
concatenate the speech-only segments, re-encode as 16 kHz mono WAV. Never
raises — any failure (decode, VAD, encode) returns a TrimResult with
status="error" and the original input bytes, so callers can dispatch on
result.status without a try/except wrap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TrimResult:
    audio_bytes: bytes
    original_seconds: float
    trimmed_seconds: float
    status: Literal["trimmed", "no_speech", "error"]
    error: str | None = None
