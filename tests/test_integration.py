"""Integration tests that hit real models / network. Excluded from default `uv run pytest`.

Run with: `uv run pytest -m integration`
"""

from pathlib import Path

import pytest

from medical_scribe.asr import load_asr_pipeline, transcribe
from medical_scribe.device import pick_device


@pytest.mark.integration
def test_medasr_transcribes_sample_audio_end_to_end():
    from huggingface_hub import hf_hub_download

    audio_path = Path(hf_hub_download(repo_id="google/medasr", filename="test_audio.wav"))
    pipe = load_asr_pipeline("google/medasr", pick_device())

    text = transcribe(pipe, audio_path)

    assert isinstance(text, str)
    assert len(text.strip()) > 0
