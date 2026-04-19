"""Tests for transcribe.py CLI helpers. Most CLI logic is exercised by manual smoke;
these cover require_hf_token's env-gate and fetch_sample's local-first resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

import transcribe
from transcribe import fetch_sample, require_hf_token


def test_require_hf_token_passes_when_token_set(monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_test_value")

    require_hf_token()  # must not raise

    captured = capsys.readouterr()
    assert captured.err == ""


def test_require_hf_token_exits_when_token_missing(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        require_hf_token()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "HF_TOKEN not set" in captured.err
    assert "huggingface.co/settings/tokens" in captured.err


def test_require_hf_token_exits_when_token_empty_string(monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "")

    with pytest.raises(SystemExit) as excinfo:
        require_hf_token()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "HF_TOKEN not set" in captured.err


def test_fetch_sample_prefers_local_file_when_present(tmp_path, monkeypatch, mocker):
    local = tmp_path / "test_audio.wav"
    local.touch()
    monkeypatch.setattr(transcribe, "LOCAL_SAMPLE", local)
    hf_mock = mocker.patch("huggingface_hub.hf_hub_download")

    result = fetch_sample("google/medasr")

    assert result == local
    hf_mock.assert_not_called()


def test_fetch_sample_falls_back_to_hf_when_local_absent(tmp_path, monkeypatch, mocker):
    missing = tmp_path / "nope.wav"
    monkeypatch.setattr(transcribe, "LOCAL_SAMPLE", missing)
    hf_mock = mocker.patch("huggingface_hub.hf_hub_download", return_value="/fake/hf/cached.wav")

    result = fetch_sample("google/medasr")

    assert result == Path("/fake/hf/cached.wav")
    hf_mock.assert_called_once_with(repo_id="google/medasr", filename="test_audio.wav")
