"""Tests for medical_scribe.vad — Silero VAD silence-trimming helpers."""

from __future__ import annotations

import dataclasses
import io

import numpy as np
import pytest
import soundfile as sf

from medical_scribe.vad import TrimResult

VAD_SR = 16000


def _wav_bytes(audio: np.ndarray, sr: int = VAD_SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def test_trim_result_is_frozen_dataclass_with_expected_fields():
    """TrimResult is frozen (callers can't mutate it) and carries the five
    fields format_trim_caption needs."""
    assert dataclasses.is_dataclass(TrimResult)
    assert TrimResult.__dataclass_params__.frozen is True

    field_names = {f.name for f in dataclasses.fields(TrimResult)}
    assert field_names == {
        "audio_bytes",
        "original_seconds",
        "trimmed_seconds",
        "status",
        "error",
    }


def test_trim_result_error_defaults_to_none():
    result = TrimResult(
        audio_bytes=b"x",
        original_seconds=1.0,
        trimmed_seconds=1.0,
        status="trimmed",
    )
    assert result.error is None


def test_trim_result_is_immutable():
    result = TrimResult(
        audio_bytes=b"x",
        original_seconds=1.0,
        trimmed_seconds=1.0,
        status="trimmed",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = "error"  # type: ignore[misc]


def test_load_vad_calls_load_silero_vad(mocker):
    fake_model = object()
    load_mock = mocker.patch(
        "medical_scribe.vad.load_silero_vad",
        return_value=fake_model,
    )

    from medical_scribe.vad import load_vad

    result = load_vad()

    assert result is fake_model
    load_mock.assert_called_once_with()


def test_trim_silence_returns_status_trimmed_with_speech_only_audio(mocker):
    """Happy path: VAD returns one speech range covering 1s of a 3s clip;
    the result has status='trimmed', a shorter audio_bytes blob, and the
    durations reflect the trim."""
    audio = np.zeros(VAD_SR * 3, dtype=np.float32)
    audio[VAD_SR : 2 * VAD_SR] = 0.1  # 1 s of "speech" in the middle
    input_bytes = _wav_bytes(audio)

    mocker.patch(
        "medical_scribe.vad.get_speech_timestamps",
        return_value=[{"start": VAD_SR, "end": 2 * VAD_SR}],
    )

    from medical_scribe.vad import trim_silence

    result = trim_silence(input_bytes, model=mocker.MagicMock())

    assert result.status == "trimmed"
    assert result.error is None
    assert abs(result.original_seconds - 3.0) < 0.01
    assert abs(result.trimmed_seconds - 1.0) < 0.01

    # The returned bytes are a WAV blob shorter than the input.
    assert len(result.audio_bytes) < len(input_bytes)
    decoded, sr = sf.read(io.BytesIO(result.audio_bytes))
    assert sr == VAD_SR
    assert abs(len(decoded) - VAD_SR) < 100  # ~1 s of audio
