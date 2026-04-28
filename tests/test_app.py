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
        "soap_truncated": True,
        "active_tab": "notes",
        "is_editing": True,
        "subjective_edit": "edited s",
        "objective_edit": "edited o",
        "assessment_edit": "edited a",
        "plan_edit": "edited p",
    }


def test_clear_downstream_state_after_audio_wipes_transcript_and_soap():
    from app import clear_downstream_state

    state = _fresh_state()
    clear_downstream_state(state, after="audio")

    # Preserved (audio identity + orthogonal UI focus).
    assert state["audio_bytes"] == b"abc"
    assert state["audio_name"] == "a.wav"
    assert state["audio_hash"] == "deadbeef"
    assert state["active_tab"] == "notes"
    # Cleared (downstream of new audio).
    assert state["tx"] is None
    assert state["tx_edit"] == ""
    assert state["soap"] is None
    assert state["soap_truncated"] is False
    assert state["is_editing"] is False
    assert state["subjective_edit"] == ""
    assert state["objective_edit"] == ""
    assert state["assessment_edit"] == ""
    assert state["plan_edit"] == ""


def test_clear_downstream_state_after_tx_wipes_only_soap():
    from app import clear_downstream_state

    state = _fresh_state()
    clear_downstream_state(state, after="tx")

    # Preserved.
    assert state["audio_bytes"] == b"abc"
    assert state["tx"] == "some transcript"
    assert state["tx_edit"] == "edited transcript"
    assert state["active_tab"] == "notes"
    # Cleared (downstream of transcript edits).
    assert state["soap"] is None
    assert state["soap_truncated"] is False
    assert state["is_editing"] is False
    assert state["subjective_edit"] == ""
    assert state["objective_edit"] == ""
    assert state["assessment_edit"] == ""
    assert state["plan_edit"] == ""


def test_clear_downstream_state_unknown_stage_is_noop():
    from app import clear_downstream_state

    state = _fresh_state()
    before = dict(state)
    clear_downstream_state(state, after="nonsense")

    assert state == before


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


def test_app_boots_to_state_a_with_models_mocked(mocker, monkeypatch):
    """Smoke test for the new shell: header + sidebar + controlled tabs."""
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

    # Header markdown contains "Medical Scribe".
    rendered_md = " ".join(md.value for md in at.markdown)
    assert "Medical Scribe" in rendered_md

    # Sidebar has a `+ New session` button.
    sidebar_button_labels = [b.label for b in at.sidebar.button]
    assert any("New session" in label for label in sidebar_button_labels)

    # Tab bar buttons render in the main area.
    main_button_labels = [b.label for b in at.button]
    assert "Transcript" in main_button_labels
    assert "Notes" in main_button_labels

    # Active tab defaults to Transcript — Notes-tab placeholder copy
    # is NOT visible.
    assert "No SOAP note yet" not in rendered_md


def test_initial_state_includes_new_keys():
    """Locks the new session-state shape introduced by the
    live-capture-and-tabs redesign."""
    from app import INITIAL_STATE

    assert INITIAL_STATE["active_tab"] == "transcript"
    assert INITIAL_STATE["is_editing"] is False
    assert INITIAL_STATE["subjective_edit"] == ""
    assert INITIAL_STATE["objective_edit"] == ""
    assert INITIAL_STATE["assessment_edit"] == ""
    assert INITIAL_STATE["plan_edit"] == ""
    assert INITIAL_STATE["_streaming"] is False


def test_active_tab_round_trips():
    """Direct state assignment is the toggle mechanism — verify the
    values we use round-trip cleanly through INITIAL_STATE iteration."""
    from app import INITIAL_STATE

    state = dict(INITIAL_STATE)
    assert state["active_tab"] == "transcript"

    state["active_tab"] = "notes"
    assert state["active_tab"] == "notes"

    state["active_tab"] = "transcript"
    assert state["active_tab"] == "transcript"


def test_is_editing_round_trips():
    from app import INITIAL_STATE

    state = dict(INITIAL_STATE)
    assert state["is_editing"] is False

    state["is_editing"] = True
    assert state["is_editing"] is True

    state["is_editing"] = False
    assert state["is_editing"] is False


@pytest.mark.parametrize(
    ("soap", "parsed", "expected"),
    [
        (None, {}, ""),
        ("", {}, ""),
        (
            "Some unstructured prose with no headers.",
            {},
            "Some unstructured prose with no headers.",
        ),
        (
            "## Subjective\nfoo\n## Plan\np\n",
            {"Subjective": "foo", "Plan": "p"},
            "",
        ),
        (
            "preamble text\n\n## Subjective\nbody\n",
            {"Subjective": "body"},
            "preamble text",
        ),
        (
            "\n\n  preamble  \n\n## Subjective\nbody\n",
            {"Subjective": "body"},
            "preamble",
        ),
        (
            "## Plan\np body\n",
            {"Plan": "p body"},
            "",
        ),
    ],
    ids=[
        "none_soap_returns_empty",
        "empty_soap_returns_empty",
        "no_parsed_returns_full_soap_stripped",
        "all_sections_no_preamble_returns_empty",
        "preamble_before_first_header_returned",
        "preamble_with_surrounding_whitespace_stripped",
        "single_section_no_preamble_returns_empty",
    ],
)
def test_compute_unparsed_remainder(soap, parsed, expected):
    from app import compute_unparsed_remainder

    assert compute_unparsed_remainder(soap, parsed) == expected


def test_escape_text_for_inline_script_basic():
    from app import escape_text_for_inline_script

    assert escape_text_for_inline_script("hello world") == '"hello world"'


def test_escape_text_for_inline_script_quotes_and_newlines():
    from app import escape_text_for_inline_script

    # JSON encoding handles JS string escaping for quotes and newlines.
    assert (
        escape_text_for_inline_script('Hello "world"\nNext line')
        == '"Hello \\"world\\"\\nNext line"'
    )


def test_escape_text_for_inline_script_escapes_script_close_tag():
    from app import escape_text_for_inline_script

    # `</script>` substring must be escaped to `<\/script>` so it cannot
    # prematurely close an enclosing <script> tag in the rendered HTML.
    result = escape_text_for_inline_script("body with </script> tag")
    assert "<\\/script>" in result
    assert "</script>" not in result


def test_escape_text_for_inline_script_empty_string():
    from app import escape_text_for_inline_script

    assert escape_text_for_inline_script("") == '""'


def test_section_key_map_covers_all_soap_sections():
    """Drift guard: SECTION_KEY_MAP must have one entry per SOAP_SECTIONS
    name. Catches the case where SOAP_SECTIONS gains a section but the
    derivation breaks. Currently SECTION_KEY_MAP is built via comprehension
    over SOAP_SECTIONS, so this is also a smoke check on the derivation."""
    from app import SECTION_KEY_MAP, SOAP_SECTIONS

    assert set(SECTION_KEY_MAP.keys()) == set(SOAP_SECTIONS)


def test_initial_state_keys_match_section_key_map():
    """Every value in SECTION_KEY_MAP must be a key in INITIAL_STATE,
    or reset_state() / clear_downstream_state() will silently miss them.
    Catches the case where SOAP_SECTIONS / SECTION_KEY_MAP grows but the
    INITIAL_STATE entries weren't added."""
    from app import INITIAL_STATE, SECTION_KEY_MAP

    for buffer_key in SECTION_KEY_MAP.values():
        assert buffer_key in INITIAL_STATE, (
            f"SECTION_KEY_MAP value {buffer_key!r} is missing from INITIAL_STATE"
        )
