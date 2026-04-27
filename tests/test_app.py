"""Tests for app.py.

Logic-level tests use plain dicts to exercise the pure helpers and the
state-machine invariants without booting Streamlit. A single AppTest
smoke verifies the UI boots into State A with the expected layout
markers.
"""

from __future__ import annotations

import pytest


def _fresh_state() -> dict:
    return {
        "audio_bytes": b"abc",
        "audio_name": "a.wav",
        "audio_hash": "deadbeef",
        "tx": "some transcript",
        "tx_edit": "edited transcript",
        "soap": "generated soap",
        "soap_edit": "edited soap",
        "expanded_pane": "left",
        "soap_truncated": True,
    }


def test_clear_downstream_state_after_audio_wipes_transcript_and_soap():
    from app import clear_downstream_state

    state = _fresh_state()
    clear_downstream_state(state, after="audio")

    # Preserved (audio identity + orthogonal UI focus).
    assert state["audio_bytes"] == b"abc"
    assert state["audio_name"] == "a.wav"
    assert state["audio_hash"] == "deadbeef"
    assert state["expanded_pane"] == "left"
    # Cleared (downstream of new audio).
    assert state["tx"] is None
    assert state["tx_edit"] == ""
    assert state["soap"] is None
    assert state["soap_edit"] == ""
    assert state["soap_truncated"] is False


def test_clear_downstream_state_after_tx_wipes_only_soap():
    from app import clear_downstream_state

    state = _fresh_state()
    clear_downstream_state(state, after="tx")

    # Preserved (audio identity + transcript + orthogonal UI focus).
    assert state["audio_bytes"] == b"abc"
    assert state["audio_name"] == "a.wav"
    assert state["audio_hash"] == "deadbeef"
    assert state["tx"] == "some transcript"
    assert state["tx_edit"] == "edited transcript"
    assert state["expanded_pane"] == "left"
    # Cleared (downstream of transcript edits).
    assert state["soap"] is None
    assert state["soap_edit"] == ""
    assert state["soap_truncated"] is False


def test_clear_downstream_state_unknown_stage_is_noop():
    from app import clear_downstream_state

    state = _fresh_state()
    before = dict(state)
    clear_downstream_state(state, after="nonsense")

    assert state == before


def test_clear_downstream_state_preserves_expanded_pane():
    from app import clear_downstream_state

    for after in ("audio", "tx"):
        state = _fresh_state()
        state["expanded_pane"] = "right"
        clear_downstream_state(state, after=after)
        assert state["expanded_pane"] == "right", (
            f"clear_downstream_state(after={after!r}) clobbered expanded_pane"
        )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("visit.wav", "audio/wav"),
        ("visit.mp3", "audio/mpeg"),
        ("visit.flac", "audio/flac"),
        ("visit.m4a", "audio/mp4"),
        ("VISIT.WAV", "audio/wav"),
        (None, None),
        ("visit", None),
        ("visit.aiff", None),
    ],
    ids=[
        "wav",
        "mp3",
        "flac",
        "m4a",
        "uppercase_extension",
        "none_input",
        "no_extension",
        "unknown_extension",
    ],
)
def test_audio_mime_from_name(name, expected):
    from app import audio_mime_from_name

    assert audio_mime_from_name(name) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"audio_bytes": None, "tx": None, "soap": None}, "No audio loaded"),
        ({"audio_bytes": b"abc", "tx": None, "soap": None}, "Transcribing…"),
        ({"audio_bytes": b"abc", "tx": "some text", "soap": None}, "Transcript ready"),
        (
            {"audio_bytes": b"abc", "tx": "some text", "soap": None, "_streaming": True},
            "Generating SOAP…",
        ),
        ({"audio_bytes": b"abc", "tx": "some text", "soap": "soap text"}, "SOAP ready"),
    ],
    ids=[
        "no_audio",
        "transcribing",
        "transcript_ready",
        "streaming_overrides_transcript_ready",
        "soap_ready",
    ],
)
def test_derive_stage_label(state, expected):
    from app import derive_stage_label

    assert derive_stage_label(state) == expected


@pytest.mark.parametrize(
    ("soap", "expected"),
    [
        (None, "Generate SOAP note"),
        ("", "Generate SOAP note"),
        ("S: patient reports headache…", "Regenerate SOAP"),
    ],
    ids=[
        "no_soap_says_generate",
        "empty_string_says_generate",
        "with_soap_says_regenerate",
    ],
)
def test_primary_action_label(soap, expected):
    from app import primary_action_label

    assert primary_action_label(soap) == expected


@pytest.mark.parametrize(
    ("initial", "meta", "expected"),
    [
        (False, {"finish_reason": "length"}, True),
        (True, {"finish_reason": "stop"}, False),
        (True, {}, False),
    ],
    ids=[
        "finish_reason_length_sets_true",
        "finish_reason_stop_sets_false",
        "missing_finish_reason_sets_false",
    ],
)
def test_update_truncation_flag(initial, meta, expected):
    from app import update_truncation_flag

    state: dict = {"soap_truncated": initial}
    update_truncation_flag(state, meta)
    assert state["soap_truncated"] is expected


def test_expanded_pane_toggle_round_trip():
    from app import INITIAL_STATE

    state = dict(INITIAL_STATE)
    assert state["expanded_pane"] is None

    state["expanded_pane"] = "left"
    assert state["expanded_pane"] == "left"

    state["expanded_pane"] = "right"
    assert state["expanded_pane"] == "right"

    state["expanded_pane"] = None
    assert state["expanded_pane"] is None


def test_app_boots_to_state_a_with_models_mocked(mocker, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_value")

    mocker.patch("app.load_asr_pipeline", return_value=mocker.MagicMock())
    mocker.patch(
        "app.load_medgemma",
        return_value=(mocker.MagicMock(), mocker.MagicMock()),
    )

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py")
    at.run(timeout=30)

    assert not at.exception, f"app raised: {at.exception}"
    assert len(at.file_uploader) == 1
    assert at.file_uploader[0].label == "Upload a patient visit recording"

    rendered_markdown = " ".join(md.value for md in at.markdown)
    assert "Medical Scribe" in rendered_markdown
    assert "Upload audio to begin" in rendered_markdown
