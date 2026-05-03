"""Silero VAD silence-trimming helpers (no Streamlit knowledge).

Operates purely on bytes-in / bytes-out: decode once via librosa, run VAD,
concatenate the speech-only segments, re-encode as 16 kHz mono WAV. Never
raises — any failure (decode, VAD, encode) returns a TrimResult with
status="error" and the original input bytes, so callers can dispatch on
result.status without a try/except wrap.
"""

from __future__ import annotations

import io
import sys
import traceback
from dataclasses import dataclass
from typing import Literal

import librosa
import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

VAD_SR = 16000


@dataclass(frozen=True)
class TrimResult:
    audio_bytes: bytes
    original_seconds: float
    trimmed_seconds: float
    status: Literal["trimmed", "no_speech", "error"]
    error: str | None = None


def load_vad() -> torch.nn.Module:
    return load_silero_vad()


def trim_silence(
    audio_bytes: bytes,
    model: torch.nn.Module,
    *,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 300,
    speech_pad_ms: int = 100,
) -> TrimResult:
    try:
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=VAD_SR, mono=True)
        original_seconds = len(audio) / VAD_SR

        ranges = get_speech_timestamps(
            torch.from_numpy(audio),
            model,
            threshold=threshold,
            sampling_rate=VAD_SR,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

        speech = np.concatenate([audio[r["start"] : r["end"]] for r in ranges])
        buf = io.BytesIO()
        sf.write(buf, speech, VAD_SR, format="WAV")
        return TrimResult(
            audio_bytes=buf.getvalue(),
            original_seconds=original_seconds,
            trimmed_seconds=len(speech) / VAD_SR,
            status="trimmed",
        )
    except Exception as exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        return TrimResult(
            audio_bytes=audio_bytes,
            original_seconds=0.0,
            trimmed_seconds=0.0,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
