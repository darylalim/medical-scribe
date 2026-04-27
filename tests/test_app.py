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
        "expanded_pane": "left",
        "soap_truncated": True,
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
    assert state["soap_truncated"] is False
    assert state["expanded_pane"] == "left"


def test_clear_downstream_state_after_tx_wipes_only_soap():
    from app import clear_downstream_state

    state = _fresh_state()
    clear_downstream_state(state, after="tx")

    assert state["tx"] == "some transcript"
    assert state["tx_edit"] == "edited transcript"
    assert state["soap"] is None
    assert state["soap_edit"] == ""
    assert state["audio_bytes"] == b"abc"
    assert state["soap_truncated"] is False
    assert state["expanded_pane"] == "left"


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


def test_derive_stage_label_no_audio():
    from app import derive_stage_label

    state = {"audio_bytes": None, "tx": None, "soap": None}
    assert derive_stage_label(state) == "No audio loaded"


def test_derive_stage_label_transcribing():
    from app import derive_stage_label

    state = {"audio_bytes": b"abc", "tx": None, "soap": None}
    assert derive_stage_label(state) == "Transcribing…"


def test_derive_stage_label_transcript_ready():
    from app import derive_stage_label

    state = {"audio_bytes": b"abc", "tx": "some text", "soap": None}
    assert derive_stage_label(state) == "Transcript ready"


def test_derive_stage_label_soap_ready():
    from app import derive_stage_label

    state = {"audio_bytes": b"abc", "tx": "some text", "soap": "soap text"}
    assert derive_stage_label(state) == "SOAP ready"


def test_primary_action_label_no_soap_says_generate():
    from app import primary_action_label

    assert primary_action_label(None) == "Generate SOAP note"


def test_primary_action_label_empty_string_says_generate():
    from app import primary_action_label

    assert primary_action_label("") == "Generate SOAP note"


def test_primary_action_label_with_soap_says_regenerate():
    from app import primary_action_label

    assert primary_action_label("S: patient reports headache…") == "Regenerate SOAP"


def test_update_truncation_flag_finish_reason_length_sets_true():
    from app import update_truncation_flag

    state: dict = {"soap_truncated": False}
    update_truncation_flag(state, {"finish_reason": "length"})
    assert state["soap_truncated"] is True


def test_update_truncation_flag_finish_reason_stop_sets_false():
    from app import update_truncation_flag

    state: dict = {"soap_truncated": True}
    update_truncation_flag(state, {"finish_reason": "stop"})
    assert state["soap_truncated"] is False


def test_update_truncation_flag_missing_finish_reason_sets_false():
    from app import update_truncation_flag

    state: dict = {"soap_truncated": True}
    update_truncation_flag(state, {})
    assert state["soap_truncated"] is False
