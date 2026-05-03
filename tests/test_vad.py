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


def test_trim_silence_returns_status_no_speech_when_vad_finds_nothing(mocker):
    """When VAD returns an empty range list, trim_silence falls back to the
    original bytes (passed through by identity) and reports status='no_speech'.
    trimmed_seconds must equal original_seconds in this case."""
    audio = np.zeros(VAD_SR * 2, dtype=np.float32)
    input_bytes = _wav_bytes(audio)

    mocker.patch("medical_scribe.vad.get_speech_timestamps", return_value=[])

    from medical_scribe.vad import trim_silence

    result = trim_silence(input_bytes, model=mocker.MagicMock())

    assert result.status == "no_speech"
    assert result.error is None
    assert result.audio_bytes is input_bytes
    assert abs(result.original_seconds - 2.0) < 0.01
    assert result.trimmed_seconds == result.original_seconds


def test_trim_silence_returns_status_error_on_decode_failure(mocker):
    """Garbage input bytes that librosa can't decode → status='error',
    audio_bytes passed through by identity, error populated, no exception."""
    from medical_scribe.vad import trim_silence

    garbage = b"this is not a wav file"
    result = trim_silence(garbage, model=mocker.MagicMock())

    assert result.status == "error"
    assert result.audio_bytes is garbage
    assert result.error is not None and len(result.error) > 0


def test_trim_silence_returns_status_error_on_vad_exception(mocker):
    """VAD itself raises → same fallback shape (no exception escapes)."""
    audio = np.zeros(VAD_SR, dtype=np.float32)
    input_bytes = _wav_bytes(audio)

    mocker.patch(
        "medical_scribe.vad.get_speech_timestamps",
        side_effect=RuntimeError("model OOM"),
    )

    from medical_scribe.vad import trim_silence

    result = trim_silence(input_bytes, model=mocker.MagicMock())

    assert result.status == "error"
    assert result.audio_bytes is input_bytes
    assert "RuntimeError" in (result.error or "")
    assert "model OOM" in (result.error or "")


@pytest.mark.parametrize(
    "failure_point",
    ["decode", "vad", "encode"],
    ids=["decode_failure", "vad_failure", "encode_failure"],
)
def test_trim_silence_never_raises(mocker, failure_point):
    """Defensive contract: regardless of which step fails, trim_silence
    returns a TrimResult — never raises. _render_transcript_pane in app.py
    relies on this; do not weaken it."""
    from medical_scribe.vad import trim_silence

    audio = np.zeros(VAD_SR, dtype=np.float32)
    input_bytes = _wav_bytes(audio)

    if failure_point == "decode":
        bytes_arg = b"garbage"
        mocker.patch("medical_scribe.vad.get_speech_timestamps", return_value=[])
    elif failure_point == "vad":
        bytes_arg = input_bytes
        mocker.patch(
            "medical_scribe.vad.get_speech_timestamps",
            side_effect=RuntimeError("vad bang"),
        )
    else:  # encode
        bytes_arg = input_bytes
        mocker.patch(
            "medical_scribe.vad.get_speech_timestamps",
            return_value=[{"start": 0, "end": VAD_SR}],
        )
        mocker.patch(
            "medical_scribe.vad.sf.write",
            side_effect=RuntimeError("encode bang"),
        )

    # The assertion is that this does not raise.
    result = trim_silence(bytes_arg, model=mocker.MagicMock())
    assert isinstance(result, TrimResult)
