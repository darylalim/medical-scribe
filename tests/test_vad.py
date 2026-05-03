"""Tests for medical_scribe.vad — Silero VAD silence-trimming helpers."""

from __future__ import annotations

import dataclasses

import pytest

from medical_scribe.vad import TrimResult


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
