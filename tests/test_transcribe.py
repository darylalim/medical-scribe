"""Tests for transcribe.py CLI helpers. Most CLI logic is exercised by manual smoke; these
cover require_hf_token's env-gate behavior directly."""

from __future__ import annotations

import pytest

from transcribe import require_hf_token


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
