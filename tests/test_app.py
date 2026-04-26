"""Tests for app.py helpers. UI rendering is covered by a light AppTest smoke check;
state-machine invariants are covered by driving clear_downstream_state with a plain dict
so we don't depend on Streamlit's ScriptRunner for logic-level tests."""

from __future__ import annotations


def _fresh_state() -> dict:
    return {
        "audio_bytes": b"abc",
        "audio_name": "a.wav",
        "audio_hash": "deadbeef",
        "tx": "some transcript",
        "tx_edit": "edited transcript",
        "soap": "generated soap",
        "soap_edit": "edited soap",
    }


def test_clear_downstream_state_after_audio_wipes_transcript_and_soap():
    from app import clear_downstream_state

    state = _fresh_state()
    clear_downstream_state(state, after="audio")

    assert state["audio_bytes"] == b"abc"
    assert state["audio_name"] == "a.wav"
    assert state["audio_hash"] == "deadbeef"
    assert state["tx"] is None
    assert state["tx_edit"] == ""
    assert state["soap"] is None
    assert state["soap_edit"] == ""


def test_clear_downstream_state_after_tx_wipes_only_soap():
    from app import clear_downstream_state

    state = _fresh_state()
    clear_downstream_state(state, after="tx")

    assert state["tx"] == "some transcript"
    assert state["tx_edit"] == "edited transcript"
    assert state["soap"] is None
    assert state["soap_edit"] == ""
    assert state["audio_bytes"] == b"abc"


def test_clear_downstream_state_unknown_stage_is_noop():
    from app import clear_downstream_state

    state = _fresh_state()
    before = dict(state)
    clear_downstream_state(state, after="nonsense")

    assert state == before


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


def test_audio_mime_from_name_wav():
    from app import audio_mime_from_name

    assert audio_mime_from_name("visit.wav") == "audio/wav"


def test_audio_mime_from_name_mp3():
    from app import audio_mime_from_name

    assert audio_mime_from_name("visit.mp3") == "audio/mpeg"


def test_audio_mime_from_name_flac():
    from app import audio_mime_from_name

    assert audio_mime_from_name("visit.flac") == "audio/flac"


def test_audio_mime_from_name_m4a():
    from app import audio_mime_from_name

    assert audio_mime_from_name("visit.m4a") == "audio/mp4"


def test_audio_mime_from_name_uppercase_extension():
    from app import audio_mime_from_name

    assert audio_mime_from_name("VISIT.WAV") == "audio/wav"


def test_audio_mime_from_name_none_input():
    from app import audio_mime_from_name

    assert audio_mime_from_name(None) is None


def test_audio_mime_from_name_no_extension():
    from app import audio_mime_from_name

    assert audio_mime_from_name("visit") is None


def test_audio_mime_from_name_unknown_extension():
    from app import audio_mime_from_name

    assert audio_mime_from_name("visit.aiff") is None
