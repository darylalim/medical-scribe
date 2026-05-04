"""Tests for app.py.

Logic-level tests use plain dicts to exercise the pure helpers and the
state-machine invariants without booting Streamlit. A small set of
AppTest-driven tests (built on the `booted_app` fixture) cover the
boot smoke and the `+ New session` dialog-gating paths.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def booted_app(mocker, monkeypatch):
    """A booted AppTest with both model loaders mocked. Lands in State A
    after the initial run. Tests that need a different starting state
    (e.g., audio loaded) should mutate `at.session_state` and call
    `at.run()` again before asserting on widgets."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_value")
    mocker.patch("app.load_asr_pipeline", return_value=mocker.MagicMock())
    mocker.patch(
        "app.load_medgemma",
        return_value=(mocker.MagicMock(), mocker.MagicMock()),
    )
    mocker.patch("app.load_vad", return_value=mocker.MagicMock())

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert not at.exception, f"app boot raised: {at.exception}"
    return at


# Minimal valid four-section SOAP markdown — used by tests that need a
# parseable draft to exercise State-E behavior. Mirrors the SAMPLE_FULL
# pattern in test_soap_sections.py.
_MINIMAL_SOAP = "## Subjective\nfoo\n## Objective\no\n## Assessment\na\n## Plan\np\n"

_FAKE_AUDIO_STATE = {
    "audio_bytes": b"fake-audio-bytes",
    "audio_name": "fake.wav",
    "audio_hash": "fake-hash",
}


def _seed_state_c(at) -> None:
    """Mutate `at.session_state` so the next `at.run()` lands in State C
    (transcript ready, no SOAP). Caller is responsible for the rerun."""
    for key, value in _FAKE_AUDIO_STATE.items():
        at.session_state[key] = value
    at.session_state["tx"] = "fake transcript"
    at.session_state["tx_edit"] = "fake transcript"


def _seed_state_e(at) -> None:
    """Mutate `at.session_state` so the next `at.run()` lands in State E
    (SOAP ready). Calls `populate_section_edit_buffers` so the four card
    buffers stay consistent with `_MINIMAL_SOAP` — if the constant changes
    the buffers track it automatically."""
    from collections.abc import MutableMapping
    from typing import cast

    from app import populate_section_edit_buffers

    _seed_state_c(at)
    at.session_state["soap"] = _MINIMAL_SOAP
    populate_section_edit_buffers(
        cast(MutableMapping[str, object], at.session_state), _MINIMAL_SOAP
    )


def _fresh_state() -> dict:
    """All session-state keys populated to non-default values, so
    `clear_downstream_state` tests can verify exactly which keys it
    touches and which it leaves alone."""
    return {
        "audio_bytes": b"abc",
        "audio_name": "a.wav",
        "audio_hash": "deadbeef",
        "tx": "some transcript",
        "tx_edit": "edited transcript",
        "tx_trim": "fake_trim_sentinel",
        "soap": "generated soap",
        "soap_truncated": True,
        "subjective_edit": "edited s",
        "objective_edit": "edited o",
        "assessment_edit": "edited a",
        "plan_edit": "edited p",
        "subjective_editing": True,
        "objective_editing": True,
        "assessment_editing": True,
        "plan_editing": True,
        "subjective_edit_snapshot": "snap s",
        "objective_edit_snapshot": "snap o",
        "assessment_edit_snapshot": "snap a",
        "plan_edit_snapshot": "snap p",
        "_streaming": True,
        "_show_reset_dialog": True,
    }


def test_clear_downstream_state_after_audio_wipes_transcript_and_soap():
    from app import clear_downstream_state

    state = _fresh_state()
    clear_downstream_state(state, after="audio")

    # Preserved (audio identity + workflow flags).
    assert state["audio_bytes"] == b"abc"
    assert state["audio_name"] == "a.wav"
    assert state["audio_hash"] == "deadbeef"
    assert state["_streaming"] is True
    assert state["_show_reset_dialog"] is True
    # Cleared (downstream of new audio).
    assert state["tx"] is None
    assert state["tx_edit"] == ""
    assert state["tx_trim"] is None
    assert state["soap"] is None
    assert state["soap_truncated"] is False
    assert state["subjective_edit"] == ""
    assert state["objective_edit"] == ""
    assert state["assessment_edit"] == ""
    assert state["plan_edit"] == ""


def test_clear_downstream_state_after_tx_wipes_only_soap():
    from app import clear_downstream_state

    state = _fresh_state()
    clear_downstream_state(state, after="tx")

    # Preserved (audio + transcript + workflow flags).
    assert state["audio_bytes"] == b"abc"
    assert state["tx"] == "some transcript"
    assert state["tx_edit"] == "edited transcript"
    assert state["tx_trim"] == "fake_trim_sentinel"
    assert state["_streaming"] is True
    assert state["_show_reset_dialog"] is True
    # Cleared (downstream of transcript edits).
    assert state["soap"] is None
    assert state["soap_truncated"] is False
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


def test_app_boots_to_state_a_with_models_mocked(booted_app):
    """Smoke test for the split-view shell: top bar + State-A chooser."""
    at = booted_app

    # Top bar markdown contains "Medical Scribe".
    rendered_md = " ".join(md.value for md in at.markdown)
    assert "Medical Scribe" in rendered_md

    # New session button is in the main body (keyed), not the sidebar.
    main_button_keys = [b.key for b in at.button]
    assert "new_session_btn" in main_button_keys

    # State-A chooser renders both affordances. Tab buttons no longer exist.
    main_button_labels = [b.label for b in at.button]
    assert "Transcript" not in main_button_labels
    assert "Notes" not in main_button_labels

    # Notes-tab placeholder copy is gone (never reached State C).
    assert "No SOAP note yet" not in rendered_md


def test_new_session_in_state_a_bypasses_dialog(booted_app):
    """Regression guard for the top-bar New session State-A bypass:
    when no audio is loaded there's nothing to lose, so clicking
    + New session must NOT open the confirmation dialog. Catches the
    case where the gating on `audio_bytes` is removed or inverted."""
    at = booted_app

    # Find the New session button by its key in the main column.
    new_session = next(b for b in at.button if b.key == "new_session_btn")
    new_session.click()
    at.run()
    assert not at.exception, f"click raised: {at.exception}"

    # Bypass path took effect: dialog flag stayed False, no dialog opens.
    assert at.session_state["_show_reset_dialog"] is False


def test_new_session_with_audio_opens_dialog(booted_app):
    """Regression guard for the destructive-confirm half of the top-bar
    New session button: when audio is loaded the click must arm the
    dialog (set `_show_reset_dialog=True`) and must NOT wipe state.
    Catches the case where the bypass branch fires in both directions."""
    at = booted_app

    # Inject a loaded-audio state and re-render so the top bar sees it.
    at.session_state["audio_bytes"] = b"fake-audio-bytes"
    at.session_state["audio_name"] = "fake.wav"
    at.session_state["audio_hash"] = "fake-hash"
    at.session_state["tx"] = "fake transcript"
    at.session_state["tx_edit"] = "fake transcript"
    at.run(timeout=30)
    assert not at.exception

    # Find the New session button by its key in the main column.
    new_session = next(b for b in at.button if b.key == "new_session_btn")
    new_session.click()
    at.run(timeout=30)
    assert not at.exception, f"click raised: {at.exception}"

    # Dialog armed; underlying state preserved (the dialog gates the wipe).
    assert at.session_state["_show_reset_dialog"] is True
    assert at.session_state["audio_bytes"] == b"fake-audio-bytes"
    assert at.session_state["tx"] == "fake transcript"


def test_initial_state_includes_new_keys():
    """Locks the session-state shape of the split-view redesign: per-section
    edit buffers and the streaming flag must all be present with their
    correct default values."""
    from app import INITIAL_STATE

    assert INITIAL_STATE["subjective_edit"] == ""
    assert INITIAL_STATE["objective_edit"] == ""
    assert INITIAL_STATE["assessment_edit"] == ""
    assert INITIAL_STATE["plan_edit"] == ""
    assert INITIAL_STATE["_streaming"] is False


def test_initial_state_includes_show_reset_dialog():
    """Locks the new session-state key introduced by the +New session
    confirmation dialog. Must default to False so reset_state() also
    closes any open dialog as a side effect."""
    from app import INITIAL_STATE

    assert INITIAL_STATE["_show_reset_dialog"] is False


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
    derivation breaks. SECTION_KEY_MAP is built via comprehension over
    SOAP_SECTIONS, so this is also a smoke check on the derivation."""
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


def test_section_color_and_initial_maps_cover_all_soap_sections():
    """Drift guard: SECTION_COLORS / SECTION_INITIALS must have one entry
    per SOAP_SECTIONS name. Catches the case where SOAP_SECTIONS gains
    a section but the visual maps weren't updated — which would crash
    _render_section_header at runtime."""
    from app import SECTION_COLORS, SECTION_INITIALS, SOAP_SECTIONS

    assert set(SECTION_COLORS.keys()) == set(SOAP_SECTIONS)
    assert set(SECTION_INITIALS.keys()) == set(SOAP_SECTIONS)


@pytest.mark.parametrize(
    ("section_names", "expected"),
    [
        ([], "Generating…"),
        (["Subjective"], "Drafting Subjective…"),
        (["Subjective", "Objective"], "Drafting Objective…"),
        (["Subjective", "Objective", "Assessment"], "Drafting Assessment…"),
        (["Subjective", "Objective", "Assessment", "Plan"], "Drafting Plan…"),
    ],
    ids=[
        "no_headers_yet",
        "first_section_streaming",
        "second_section_streaming",
        "third_section_streaming",
        "final_section_streaming",
    ],
)
def test_streaming_status_label(section_names, expected):
    """Status label is derived from the last detected section name —
    that's the one currently being drafted. Empty list means we haven't
    seen any section header yet (preamble / very early stream)."""
    from app import streaming_status_label

    assert streaming_status_label(section_names) == expected


@pytest.mark.parametrize(
    ("soap", "expected"),
    [
        (None, "Generate SOAP note"),
        ("", "Generate SOAP note"),
        ("## Subjective\nbody", "Regenerate SOAP"),
        ("anything truthy", "Regenerate SOAP"),
    ],
    ids=[
        "none_returns_generate",
        "empty_string_returns_generate",
        "soap_with_section_returns_regenerate",
        "non_empty_string_returns_regenerate",
    ],
)
def test_primary_action_label(soap, expected):
    """Label flips on truthiness of `soap` — falsy (None / empty) → Generate;
    truthy → Regenerate. Click handler is the same either way."""
    from app import primary_action_label

    assert primary_action_label(soap) == expected


def test_populate_section_edit_buffers_full_four_sections():
    """All four section bodies land in their matching *_edit buffers."""
    from app import SECTION_KEY_MAP, populate_section_edit_buffers

    state: dict = dict.fromkeys(SECTION_KEY_MAP.values(), "")
    soap = (
        "## Subjective\ns body\n\n"
        "## Objective\no body\n\n"
        "## Assessment\na body\n\n"
        "## Plan\np body\n"
    )

    populate_section_edit_buffers(state, soap)

    assert state["subjective_edit"] == "s body"
    assert state["objective_edit"] == "o body"
    assert state["assessment_edit"] == "a body"
    assert state["plan_edit"] == "p body"


def test_populate_section_edit_buffers_missing_section_leaves_empty():
    """Sections absent from the SOAP blob get empty-string buffers, not
    KeyError. Defensive against partial / truncated model output."""
    from app import SECTION_KEY_MAP, populate_section_edit_buffers

    state: dict = dict.fromkeys(SECTION_KEY_MAP.values(), "preexisting value")
    soap = "## Subjective\ns body\n\n## Plan\np body\n"

    populate_section_edit_buffers(state, soap)

    assert state["subjective_edit"] == "s body"
    assert state["objective_edit"] == ""
    assert state["assessment_edit"] == ""
    assert state["plan_edit"] == "p body"


def test_populate_section_edit_buffers_empty_soap_clears_all():
    """Empty SOAP wipes every buffer to ''. Same semantics as the regenerate
    click path (which calls this helper after resetting `soap`)."""
    from app import SECTION_KEY_MAP, populate_section_edit_buffers

    state: dict = dict.fromkeys(SECTION_KEY_MAP.values(), "preexisting")

    populate_section_edit_buffers(state, "")

    for key in SECTION_KEY_MAP.values():
        assert state[key] == ""


def test_state_a_renders_mic_and_upload_chooser_cards(booted_app):
    """State A renders both chooser cards (Record + Upload) inline as
    side-by-side cards in the split-view shell.

    Assertions are pinned to the wrapper-tag forms (e.g.,
    `<div class="ms-chooser-card">`) rather than bare class-name substrings —
    bare substrings would also match the design-tokens CSS block, so a
    regression that dropped the chooser's markdown emission would pass
    silently."""
    at = booted_app

    markdown_blocks = [m.value for m in at.markdown]
    joined = "\n".join(markdown_blocks)
    assert "Record this visit" in joined
    assert "Upload a recording" in joined
    assert '<div class="ms-chooser-card">' in joined
    assert '<div class="ms-mic-circle">' in joined

    # File uploader is exposed by AppTest at Streamlit 1.39 — assert key
    # presence unconditionally so a regression dropping the widget fails loudly.
    file_uploader_keys = [w.key for w in at.file_uploader]
    assert "file_uploader_widget" in file_uploader_keys

    # st.audio_input may or may not be exposed by AppTest at the pinned
    # Streamlit version. Conditionally assert when the accessor is available.
    if hasattr(at, "audio_input"):
        audio_input_keys = [w.key for w in at.audio_input]
        assert "audio_input_widget" in audio_input_keys


def test_state_c_renders_transcript_and_soap_panes_simultaneously(booted_app):
    """When transcript and SOAP both exist (State E / SOAP-ready), the split
    view renders the transcript text_area AND the SOAP read-mode cards
    (pencil buttons) in a single render pass. Regression guard for the
    redesign's primary win — verifying SOAP claims against the transcript
    without tab-switching.

    State E now defaults cards to read mode (st.markdown + pencil button),
    so the pre-Task-13 assertion of `>= 2 text_areas` is replaced by checking
    for the transcript text_area plus the four pencil-edit buttons."""
    at = booted_app
    _seed_state_e(at)
    at.run(timeout=30)
    assert not at.exception, f"render raised: {at.exception}"

    # Transcript text_area is always present in State E.
    assert len(at.text_area) >= 1, f"transcript text_area missing; saw {len(at.text_area)}"

    # All four pencil buttons present means the SOAP pane rendered in read
    # mode — both panes are visible simultaneously in the same render pass.
    edit_btn_keys = {
        f"edit_{n.lower()}_btn" for n in ("Subjective", "Objective", "Assessment", "Plan")
    }
    actual_keys = {b.key for b in at.button if b.key}
    assert edit_btn_keys.issubset(actual_keys), (
        f"SOAP pencil buttons missing from split view; have {actual_keys}"
    )


def test_state_c_renders_trim_caption_when_present(booted_app):
    """When tx_trim is populated with a 'trimmed' status above the 5%
    threshold, the caption renders above the transcript text area."""
    from medical_scribe import TrimResult

    at = booted_app
    _seed_state_c(at)
    at.session_state["tx_trim"] = TrimResult(
        audio_bytes=b"fake-trimmed",
        original_seconds=536.0,
        trimmed_seconds=284.0,
        status="trimmed",
    )
    at.run(timeout=30)
    assert not at.exception, f"render raised: {at.exception}"

    caption_texts = [c.value for c in at.caption]
    assert any("Trimmed 4m 12s of silence" in text for text in caption_texts), (
        f"expected trim caption above transcript; saw captions: {caption_texts!r}"
    )


def test_state_c_suppresses_trim_caption_below_5_percent(booted_app):
    """When the trim ratio is below the 5% noise floor, format_trim_caption
    returns None and the caption block doesn't render — locks the
    declutter behavior in spec §5.5."""
    from medical_scribe import TrimResult

    at = booted_app
    _seed_state_c(at)
    at.session_state["tx_trim"] = TrimResult(
        audio_bytes=b"fake",
        original_seconds=100.0,
        trimmed_seconds=98.0,  # 2% trim
        status="trimmed",
    )
    at.run(timeout=30)
    assert not at.exception

    caption_texts = [c.value for c in at.caption]
    assert not any("Trimmed" in text for text in caption_texts), (
        f"caption should be suppressed below 5% trim; saw captions: {caption_texts!r}"
    )


def test_state_c_renders_error_caption_when_status_error(booted_app):
    """When tx_trim status is 'error', the fallback caption renders so the
    user knows VAD failed but transcription still proceeded."""
    from medical_scribe import TrimResult

    at = booted_app
    _seed_state_c(at)
    at.session_state["tx_trim"] = TrimResult(
        audio_bytes=b"fake",
        original_seconds=0.0,
        trimmed_seconds=0.0,
        status="error",
        error="RuntimeError: boom",
    )
    at.run(timeout=30)
    assert not at.exception

    caption_texts = [c.value for c in at.caption]
    assert any("Couldn't trim silence" in text for text in caption_texts), (
        f"expected error caption; saw: {caption_texts!r}"
    )


def test_initial_state_excludes_active_tab_and_is_editing():
    """Drift guard against accidental re-introduction of pre-redesign keys.
    `active_tab` was removed when tabs were dropped in favor of the split
    view; the singular `is_editing` was replaced by four per-section
    `*_editing` keys (see SECTION_EDITING_KEY_MAP). Re-introducing either
    singular form would silently conflict with the current schema."""
    from app import INITIAL_STATE

    assert "active_tab" not in INITIAL_STATE
    assert "is_editing" not in INITIAL_STATE


def test_card_edit_buffer_persists_across_reruns(booted_app):
    """Per-section edit buffer behavior: writing to a *_edit buffer and
    re-rendering preserves the value across reruns. The CLAUDE.md
    invariant about value= + manual sync (not key=) protects this in
    _render_section_card's conditionally-rendered edit branch; the test
    exercises the basic persistence path."""
    at = booted_app
    _seed_state_e(at)
    # Override one buffer to a non-default value so the rerun's persistence
    # claim is testable.
    at.session_state["subjective_edit"] = "edited subjective"
    at.run(timeout=30)
    assert not at.exception

    # Re-run: the value must survive — text_areas use value= + manual sync,
    # not key=, so they don't get GC'd by Streamlit's widget-state cleanup.
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["subjective_edit"] == "edited subjective"


@pytest.mark.parametrize(
    ("seeder", "expected_label", "absent_label"),
    [
        (_seed_state_c, "Generate SOAP note", "Regenerate SOAP"),
        (_seed_state_e, "Regenerate SOAP", "Generate SOAP note"),
    ],
    ids=["pre_soap_shows_generate", "post_soap_shows_regenerate"],
)
def test_primary_action_button_label_flips_with_soap_state(
    booted_app, seeder, expected_label, absent_label
):
    """The button label is sourced from primary_action_label and reflects
    the current `soap` state in the rendered UI. Catches a regression where
    someone hardcodes either label string in the render code, bypassing the
    helper."""
    at = booted_app
    seeder(at)
    at.run(timeout=30)
    assert not at.exception, f"render raised: {at.exception}"

    button_labels = [b.label for b in at.button]
    assert expected_label in button_labels, (
        f"expected primary-action button label {expected_label!r}; saw {button_labels!r}"
    )
    assert absent_label not in button_labels, (
        f"unexpected label {absent_label!r} present alongside {expected_label!r}; "
        f"saw {button_labels!r}"
    )


def test_state_b_right_pane_shows_awaiting_transcript_placeholder(booted_app, mocker):
    """In State B (audio captured, transcribing in progress), the right pane
    shows 'Awaiting transcript…' while the left pane's spinner blocks. Locks
    the column-order swap behavior in _render_split_view: SOAP pane renders
    first so the right column isn't blank during the synchronous transcribe.

    Stubs out _render_transcript_pane so the test can observe State B without
    the synchronous transcribe call advancing immediately to State C."""
    at = booted_app
    mocker.patch("app._render_transcript_pane", lambda asr_pipe, vad_model: None)
    for key, value in _FAKE_AUDIO_STATE.items():
        at.session_state[key] = value
    # tx stays None — State B
    at.run(timeout=30)
    assert not at.exception, f"render raised: {at.exception}"

    rendered_md = " ".join(md.value for md in at.markdown)
    assert "Awaiting transcript" in rendered_md, (
        f"State B right pane should show 'Awaiting transcript…' placeholder; "
        f"rendered markdown: {rendered_md!r}"
    )


def test_state_c_right_pane_shows_click_generate_placeholder(booted_app):
    """In State C (transcript ready, no SOAP, not streaming), the right pane
    shows the enhanced 'what to expect' placeholder with model name and timing
    context. Locks the 'no blank pane' UX guarantee — without this, the user
    would see only the transcript on the left and an empty right column until
    they click Generate."""
    at = booted_app
    _seed_state_c(at)
    at.run(timeout=30)
    assert not at.exception, f"render raised: {at.exception}"

    rendered_md = " ".join(md.value for md in at.markdown)
    assert "Ready to draft a SOAP note" in rendered_md, (
        f"State C right pane should show 'Ready to draft a SOAP note' heading; "
        f"rendered markdown: {rendered_md!r}"
    )
    # Model display name and token count are both derived from medical_scribe
    # constants so the displayed values can't drift from the real model id and
    # max-tokens budget passed to stream_soap.
    from medical_scribe import DEFAULT_MAX_TOKENS, MODEL_DISPLAY_NAME

    assert MODEL_DISPLAY_NAME in rendered_md, (
        f"State C right pane should name the model ({MODEL_DISPLAY_NAME}); "
        f"rendered markdown: {rendered_md!r}"
    )
    assert f"~{DEFAULT_MAX_TOKENS} tokens" in rendered_md
    assert "Click <b>Generate SOAP note</b>" in rendered_md, (
        f"State C right pane should show 'Click <b>Generate SOAP note</b>' CTA; "
        f"rendered markdown: {rendered_md!r}"
    )


def _iframe_srcdocs(at) -> list[str]:
    """Walk the AppTest tree and collect every iframe element's `srcdoc`.

    `st.iframe` renders as an `UnknownElement` with `type="iframe"`; its
    `proto.srcdoc` holds the embedded HTML. AppTest doesn't expose a typed
    accessor for iframes (Streamlit 1.56), so we walk the tree manually."""
    found: list[str] = []

    def walk(node):
        if getattr(node, "type", None) == "iframe":
            proto = getattr(node, "proto", None)
            if proto is not None and proto.srcdoc:
                found.append(proto.srcdoc)
        children = getattr(node, "children", {})
        if isinstance(children, dict):
            for child in children.values():
                walk(child)

    walk(at._tree)
    return found


def test_copy_button_renders_iframe_with_clipboard_payload(booted_app):
    """Locks the clipboard button's iframe render path post-st.iframe migration.

    `st.iframe` auto-detects HTML-string input via a fallback heuristic
    (Path → URL → existing file → /-prefixed → else srcdoc). This test
    asserts the button id, `clipboard.writeText` JS, and a unique marker
    from the SOAP payload all land in the iframe's `srcdoc` — catches
    regressions where the heuristic mis-routes the payload to `src` (URL),
    which would silently produce a blank iframe."""
    at = booted_app
    _seed_state_e(at)
    # Override one buffer with a unique marker to confirm the SOAP payload is
    # threaded into the JS clipboard call.
    at.session_state["subjective_edit"] = "TEST_S_BODY_MARKER"
    at.run(timeout=30)
    assert not at.exception, f"render raised: {at.exception}"

    srcdocs = _iframe_srcdocs(at)
    copy_iframe = next((s for s in srcdocs if 'id="copy_btn"' in s), None)
    assert copy_iframe is not None, (
        f"no iframe with id='copy_btn' rendered; iframe srcdocs: {[s[:80] for s in srcdocs]}"
    )
    assert "navigator.clipboard.writeText" in copy_iframe, (
        "clipboard JS missing from copy-button iframe"
    )
    assert "TEST_S_BODY_MARKER" in copy_iframe, (
        "SOAP payload not embedded in copy-button iframe srcdoc"
    )


def test_truncation_warning_renders_when_flagged(booted_app):
    """When soap_truncated is True (set on stream completion when
    finish_reason=='length'), a persistent warning renders at the top of the
    right pane. The warning is a clinical-safety message — the SOAP may be
    incomplete and the user must verify all four sections before copying."""
    at = booted_app
    _seed_state_e(at)
    at.session_state["soap_truncated"] = True
    at.run(timeout=30)
    assert not at.exception, f"render raised: {at.exception}"

    warning_texts = [w.value for w in at.warning]
    assert any("output token limit" in text for text in warning_texts), (
        f"truncation warning should render when soap_truncated=True; "
        f"saw warnings: {warning_texts!r}"
    )


def test_format_trim_caption_returns_none_for_no_result():
    from app import format_trim_caption

    assert format_trim_caption(None) is None


@pytest.mark.parametrize(
    ("original", "trimmed", "expected"),
    [
        # 100s → 95s = 5s removed (5%) — exactly at the 5% suppression threshold
        (100.0, 95.0, "Trimmed 0m 5s of silence (5% of recording)."),
        # 8m 56s → 4m 44s = 4m 12s removed (47%)
        (536.0, 284.0, "Trimmed 4m 12s of silence (47% of recording)."),
        # 1m → 30s = 30s removed (50%)
        (60.0, 30.0, "Trimmed 0m 30s of silence (50% of recording)."),
        # 13m → 1m = 12m removed (92%)
        (780.0, 60.0, "Trimmed 12m 0s of silence (92% of recording)."),
        # 2h 10m → 1h 5m = 1h 5m removed (50%)
        (7800.0, 3900.0, "Trimmed 1h 5m of silence (50% of recording)."),
    ],
    ids=[
        "five_percent_boundary",
        "typical_47pct",
        "short_30s",
        "twelve_min_zero_sec",
        "over_one_hour",
    ],
)
def test_format_trim_caption_status_trimmed_above_threshold(original, trimmed, expected):
    from app import format_trim_caption
    from medical_scribe import TrimResult

    result = TrimResult(
        audio_bytes=b"x",
        original_seconds=original,
        trimmed_seconds=trimmed,
        status="trimmed",
    )
    assert format_trim_caption(result) == expected


@pytest.mark.parametrize(
    ("original", "trimmed"),
    [
        (100.0, 98.0),  # 2% trim
        (100.0, 96.0),  # 4% trim
        (100.0, 100.0),  # 0% trim
    ],
    ids=["two_percent", "four_percent", "zero_percent"],
)
def test_format_trim_caption_suppresses_below_5_percent(original, trimmed):
    """When trim ratio < 5%, the caption is suppressed (returns None) to
    avoid visual clutter for recordings that were already mostly speech."""
    from app import format_trim_caption
    from medical_scribe import TrimResult

    result = TrimResult(
        audio_bytes=b"x",
        original_seconds=original,
        trimmed_seconds=trimmed,
        status="trimmed",
    )
    assert format_trim_caption(result) is None


def test_format_trim_caption_no_speech_returns_fallback_message():
    from app import format_trim_caption
    from medical_scribe import TrimResult

    result = TrimResult(
        audio_bytes=b"x",
        original_seconds=10.0,
        trimmed_seconds=10.0,
        status="no_speech",
    )
    assert format_trim_caption(result) == (
        "Couldn't detect speech; transcribing the full recording instead."
    )


def test_format_trim_caption_error_returns_fallback_message():
    from app import format_trim_caption
    from medical_scribe import TrimResult

    result = TrimResult(
        audio_bytes=b"x",
        original_seconds=0.0,
        trimmed_seconds=0.0,
        status="error",
        error="RuntimeError: boom",
    )
    assert format_trim_caption(result) == (
        "Couldn't trim silence; transcribing the full recording instead."
    )


def test_format_trim_caption_min_ratio_override():
    """The `min_ratio` parameter overrides the default 5% suppression
    threshold. Pinning this prevents a regression where the parameter
    becomes inert (silently ignored)."""
    from app import format_trim_caption
    from medical_scribe import TrimResult

    result = TrimResult(
        audio_bytes=b"x",
        original_seconds=100.0,
        trimmed_seconds=90.0,  # 10% trim
        status="trimmed",
    )
    # Default min_ratio=0.05: 10% > 5%, renders.
    assert format_trim_caption(result) is not None
    # Explicit min_ratio=0.20: 10% < 20%, suppressed.
    assert format_trim_caption(result, min_ratio=0.20) is None


def test_initial_state_includes_tx_trim_default_none():
    """Drift guard: the new tx_trim key must default to None (matches
    Streamlit's session_state.get(...) None branch in _render_transcript_pane).
    Catches a regression where someone defaults it to an empty TrimResult."""
    from app import INITIAL_STATE

    assert "tx_trim" in INITIAL_STATE
    assert INITIAL_STATE["tx_trim"] is None


def test_section_colors_have_no_duplicates():
    """All four SOAP sections must use distinct colors. The indigo-700 swap
    moves S out of the blue/green collision zone with O — accidentally
    re-aligning S to a near-O hue would silently regress that fix."""
    from app import SECTION_COLORS, SOAP_SECTIONS

    values = [SECTION_COLORS[name] for name in SOAP_SECTIONS]
    assert len(set(values)) == 4, f"SECTION_COLORS has duplicates: {values}"


def test_section_colors_are_all_aa_grade():
    """Every chip color, paired with white (#fff) text, must clear WCAG AA
    contrast (4.5:1). The all-700 family was chosen so this test passes
    without per-chip caveats."""
    from app import SECTION_COLORS, SOAP_SECTIONS

    def relative_luminance(hex_color: str) -> float:
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))

        def channel(c: float) -> float:
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)

    for name in SOAP_SECTIONS:
        bg = relative_luminance(SECTION_COLORS[name])
        # White luminance is 1.0; ratio formula = (lighter + 0.05) / (darker + 0.05)
        ratio = (1.0 + 0.05) / (bg + 0.05)
        assert ratio >= 4.5, f"{name} chip {SECTION_COLORS[name]} contrast {ratio:.2f}:1 < 4.5:1"


def test_design_tokens_css_contains_required_variables():
    """The design tokens CSS block must declare every CSS variable that
    component styles and inline styles consume."""
    from app import _design_tokens_css

    css = _design_tokens_css()
    for var in [
        "--color-surface",
        "--color-canvas",
        "--color-border",
        "--color-text",
        "--color-text-muted",
        "--color-text-subtle",
    ]:
        assert var in css, f"missing CSS variable {var}"


def test_design_tokens_css_is_wrapped_in_style_tag():
    from app import _design_tokens_css

    css = _design_tokens_css()
    assert css.startswith("<style>")
    assert css.endswith("</style>") or css.endswith("</style>\n")


def test_components_css_contains_required_classes():
    """The components CSS block must declare every class name that other
    render functions reference."""
    from app import _components_css

    css = _components_css()
    for cls in [
        ".ms-topbar",
        ".ms-stage-chip",
        ".ms-stage-static",
        ".ms-stage-active",
        ".soap-section-header",
        ".soap-chip",
        ".ms-skel-line",
        ".ms-streaming-cursor",
        ".ms-chooser-card",
        ".ms-chooser-title",
        ".ms-chooser-caption",
        ".ms-mic-circle",
        # Newly extracted classes from inline-style cleanup:
        ".ms-state-c-block",
        ".ms-state-c-heading",
        ".ms-state-c-detail",
        ".ms-state-c-cta",
        ".ms-streaming-card-body",
        ".ms-copy-bar-divider",
        ".ms-copy-bar-caption",
    ]:
        assert cls in css, f"missing CSS class {cls}"

    for anim in ["@keyframes ms-pulse", "@keyframes ms-shimmer", "@keyframes ms-cursor-blink"]:
        assert anim in css, f"missing animation {anim}"


@pytest.mark.parametrize(
    "label,expected_variant",
    [
        ("No audio loaded", "static"),
        ("Transcribing…", "active"),
        ("Transcript ready", "static"),
        ("Generating SOAP…", "active"),
        ("SOAP ready", "static"),
    ],
)
def test_stage_chip_html_picks_correct_variant(label, expected_variant):
    from app import _stage_chip_html

    markup = _stage_chip_html(label)
    assert f"ms-stage-{expected_variant}" in markup
    assert label in markup
    assert 'class="ms-dot"' in markup


def test_stage_chip_variants_exact_match_with_derive_stage_label():
    """Drift-guard. Every label `derive_stage_label` produces must have a
    `STAGE_CHIP_VARIANTS` entry, AND `STAGE_CHIP_VARIANTS` must not have
    orphan keys. The helper's `.get(label, "static")` fallback would
    silently mis-render unknown labels — this test makes that case CI-
    visible by asserting set equality, not just inclusion."""
    from app import STAGE_CHIP_VARIANTS, derive_stage_label

    sample_states = [
        {"audio_bytes": None},  # State A
        {"audio_bytes": b"x", "tx": None},  # State B
        {"audio_bytes": b"x", "tx": "t", "_streaming": True},  # State D
        {"audio_bytes": b"x", "tx": "t", "soap": None},  # State C
        {"audio_bytes": b"x", "tx": "t", "soap": "s"},  # State E
    ]
    expected_labels = {derive_stage_label(state) for state in sample_states}
    assert set(STAGE_CHIP_VARIANTS.keys()) == expected_labels, (
        f"STAGE_CHIP_VARIANTS keys {set(STAGE_CHIP_VARIANTS.keys())} "
        f"don't match derive_stage_label outputs {expected_labels}"
    )


def test_stage_chip_html_escapes_label_text():
    """Stage labels are static strings, but if a future state label embeds
    HTML metacharacters, the chip must escape them. Asserts both that the
    raw form is absent AND the escaped form is present so a half-escape
    regression fails loudly."""
    from app import _stage_chip_html

    markup = _stage_chip_html("<script>")
    assert "<script>" not in markup, "raw <script> tag leaked into chip output"
    assert "&lt;script&gt;" in markup, "escaped form missing from chip output"


@pytest.mark.parametrize(
    "state,expected",
    [
        # State A — no audio
        ({"audio_bytes": None}, ""),
        # State B — audio present, no transcript yet, no trim result yet
        (
            {"audio_bytes": b"x", "audio_name": "v.wav", "tx": None, "tx_trim": None},
            "recording · v.wav",
        ),
        # State C onward — transcript ready, trim result available
        # (TrimResult is a dataclass, but the helper only reads .original_seconds
        # and .trimmed_seconds, so a duck-typed namespace is enough for the test)
        (
            {
                "audio_bytes": b"x",
                "audio_name": "v.wav",
                "tx": "transcript",
                "tx_trim": type("TR", (), {"original_seconds": 600.0, "trimmed_seconds": 318.0})(),
            },
            "session · 10m 0s · trimmed 47%",
        ),
        # Trim result with status="no_speech" — trimmed_seconds == original_seconds
        (
            {
                "audio_bytes": b"x",
                "audio_name": "v.wav",
                "tx": "transcript",
                "tx_trim": type("TR", (), {"original_seconds": 60.0, "trimmed_seconds": 60.0})(),
            },
            "session · 1m 0s",
        ),
        # status="error" TrimResult — VAD failed, audio_bytes is original input,
        # trimmed_seconds=0.0. Without the trimmed<=0 clamp, the helper would
        # compute 1.0 - 0/300 = 100% and output a misleading "trimmed 100%".
        (
            {
                "audio_bytes": b"x",
                "audio_name": "v.wav",
                "tx": "transcript",
                "tx_trim": type("TR", (), {"original_seconds": 300.0, "trimmed_seconds": 0.0})(),
            },
            "session · 5m 0s",
        ),
        # State C+ degenerate — tx is set but tx_trim is missing (defensive
        # path). Helper returns plain "session" rather than reusing State B's
        # "session · {filename}" form.
        (
            {
                "audio_bytes": b"x",
                "audio_name": "v.wav",
                "tx": "transcript",
                "tx_trim": None,
            },
            "session",
        ),
    ],
)
def test_format_session_meta(state, expected):
    from app import _format_session_meta

    assert _format_session_meta(state) == expected


# ---------------------------------------------------------------------------
# _topbar_html unit tests
# ---------------------------------------------------------------------------


def test_topbar_html_includes_title_and_chip_and_meta():
    from app import _topbar_html

    html_out = _topbar_html(stage_label="SOAP ready", meta="session · 9m 04s · trimmed 47%")
    assert 'class="ms-topbar"' in html_out
    assert "Medical Scribe" in html_out
    assert "ms-stage-chip" in html_out
    assert "SOAP ready" in html_out
    assert "session · 9m 04s · trimmed 47%" in html_out


def test_topbar_html_omits_meta_div_when_empty():
    """State A has no meta — the helper should not render an empty
    .ms-topbar-meta span (would create a stray clickable element)."""
    from app import _topbar_html

    html_out = _topbar_html(stage_label="No audio loaded", meta="")
    assert "ms-topbar-meta" not in html_out


def test_topbar_html_escapes_meta_string():
    from app import _topbar_html

    html_out = _topbar_html(stage_label="SOAP ready", meta="<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;" in html_out


# ---------------------------------------------------------------------------
# AppTest scenarios for topbar and new-session button
# ---------------------------------------------------------------------------


def test_topbar_renders_in_state_a(booted_app):
    """In State A, the top bar renders with title, "No audio loaded" chip,
    and the New session button."""
    at = booted_app

    # Title and chip live inside markdown; verify the inline HTML.
    markdown_blocks = [m.value for m in at.markdown]
    joined = "\n".join(markdown_blocks)
    assert "Medical Scribe" in joined
    assert "No audio loaded" in joined
    assert "ms-stage-static" in joined

    # New session button is keyed:
    assert any(b.key == "new_session_btn" for b in at.button), "New session button missing"


def test_topbar_meta_renders_in_state_e(booted_app):
    """Once a SOAP draft exists, the top bar's mono-text meta should appear."""
    at = booted_app
    _seed_state_e(at)
    at.session_state["tx_trim"] = type(
        "TR", (), {"original_seconds": 540.0, "trimmed_seconds": 254.0, "status": "trimmed"}
    )()
    at.run(timeout=30)

    markdown_blocks = [m.value for m in at.markdown]
    joined = "\n".join(markdown_blocks)
    assert "session · 9m 0s · trimmed 53%" in joined
    assert "SOAP ready" in joined


def test_compute_section_states_empty_buffer():
    """No section headers seen yet — every section is pending."""
    from app import SOAP_SECTIONS, compute_section_states

    states = compute_section_states("", list(SOAP_SECTIONS))
    assert len(states) == len(SOAP_SECTIONS)
    for (name, status, body), expected_name in zip(states, SOAP_SECTIONS, strict=True):
        assert name == expected_name
        assert status == "pending"
        assert body == ""


def test_compute_section_states_first_section_active():
    """Subjective header seen, body partial, no later headers — Subjective
    active, others pending."""
    from app import compute_section_states

    buf = "## Subjective\n62 y/o M with"
    states = compute_section_states(buf, ["Subjective", "Objective", "Assessment", "Plan"])
    assert states[0] == ("Subjective", "active", "62 y/o M with")
    for tup in states[1:]:
        assert tup[1] == "pending"


def test_compute_section_states_first_completed_second_active():
    """Both headers seen — Subjective is completed (later header appeared),
    Objective is active."""
    from app import compute_section_states

    buf = "## Subjective\n62 y/o M with chest pain.\n## Objective\nBP 142/88"
    states = compute_section_states(buf, ["Subjective", "Objective", "Assessment", "Plan"])
    assert states[0] == ("Subjective", "completed", "62 y/o M with chest pain.")
    assert states[1] == ("Objective", "active", "BP 142/88")
    assert states[2][1] == "pending"
    assert states[3][1] == "pending"


def test_compute_section_states_all_four_complete():
    """All four headers seen with a trailing line break — last is active
    (still streaming, by convention) until the LLM emits a stop token,
    which compute_section_states cannot infer; the calling code flips it
    to completed on stream end."""
    from app import compute_section_states

    buf = "## Subjective\nA\n## Objective\nB\n## Assessment\nC\n## Plan\nD\n"
    states = compute_section_states(buf, ["Subjective", "Objective", "Assessment", "Plan"])
    assert states[0] == ("Subjective", "completed", "A")
    assert states[1] == ("Objective", "completed", "B")
    assert states[2] == ("Assessment", "completed", "C")
    assert states[3] == ("Plan", "active", "D")


def test_compute_section_states_preserves_section_order():
    """Even if the LLM emits sections out of order, the helper must return
    them in the canonical SOAP_SECTIONS order — the calling code's
    skeleton-card placement depends on it."""
    from app import compute_section_states

    buf = "## Plan\nfirst.\n## Subjective\nlater."
    states = compute_section_states(buf, ["Subjective", "Objective", "Assessment", "Plan"])
    assert [s[0] for s in states] == ["Subjective", "Objective", "Assessment", "Plan"]
    # Plan is the most recent header → active. Subjective was followed by Plan
    # → completed.
    assert states[0][1] == "completed"
    assert states[3][1] == "active"


def test_section_editing_key_map_covers_all_soap_sections():
    from app import SECTION_EDITING_KEY_MAP, SOAP_SECTIONS

    for name in SOAP_SECTIONS:
        assert name in SECTION_EDITING_KEY_MAP
    assert set(SECTION_EDITING_KEY_MAP.keys()) == set(SOAP_SECTIONS)


def test_section_snapshot_key_map_covers_all_soap_sections():
    from app import SECTION_SNAPSHOT_KEY_MAP, SOAP_SECTIONS

    for name in SOAP_SECTIONS:
        assert name in SECTION_SNAPSHOT_KEY_MAP
    assert set(SECTION_SNAPSHOT_KEY_MAP.keys()) == set(SOAP_SECTIONS)


def test_initial_state_includes_editing_flags():
    from app import INITIAL_STATE, SECTION_EDITING_KEY_MAP

    for key in SECTION_EDITING_KEY_MAP.values():
        assert key in INITIAL_STATE, f"INITIAL_STATE missing {key}"
        assert INITIAL_STATE[key] is False


def test_initial_state_includes_edit_snapshots():
    from app import INITIAL_STATE, SECTION_SNAPSHOT_KEY_MAP

    for key in SECTION_SNAPSHOT_KEY_MAP.values():
        assert key in INITIAL_STATE, f"INITIAL_STATE missing {key}"
        assert INITIAL_STATE[key] is None


def test_clear_downstream_state_after_audio_resets_editing_flags_and_snapshots():
    from app import clear_downstream_state

    state = _fresh_state()
    clear_downstream_state(state, after="audio")

    for key in (
        "subjective_editing",
        "objective_editing",
        "assessment_editing",
        "plan_editing",
    ):
        assert state[key] is False, f"{key} should be reset to False"
    for key in (
        "subjective_edit_snapshot",
        "objective_edit_snapshot",
        "assessment_edit_snapshot",
        "plan_edit_snapshot",
    ):
        assert state[key] is None, f"{key} should be reset to None"


def test_clear_downstream_state_after_tx_resets_editing_flags_and_snapshots():
    from app import clear_downstream_state

    state = _fresh_state()
    clear_downstream_state(state, after="tx")

    for key in (
        "subjective_editing",
        "objective_editing",
        "assessment_editing",
        "plan_editing",
    ):
        assert state[key] is False
    for key in (
        "subjective_edit_snapshot",
        "objective_edit_snapshot",
        "assessment_edit_snapshot",
        "plan_edit_snapshot",
    ):
        assert state[key] is None


def test_populate_section_edit_buffers_does_not_touch_editing_flags():
    """Cards must always land in read mode after a stream completes —
    populate_section_edit_buffers fills *_edit but not *_editing."""
    from collections.abc import MutableMapping

    from app import populate_section_edit_buffers

    state: MutableMapping[str, object] = {
        "subjective_editing": True,
        "objective_editing": True,
        "assessment_editing": True,
        "plan_editing": True,
        "subjective_edit": "",
        "objective_edit": "",
        "assessment_edit": "",
        "plan_edit": "",
    }
    populate_section_edit_buffers(
        state, "## Subjective\nA\n## Objective\nB\n## Assessment\nC\n## Plan\nD"
    )
    # *_editing should be unchanged.
    for key in (
        "subjective_editing",
        "objective_editing",
        "assessment_editing",
        "plan_editing",
    ):
        assert state[key] is True, f"populate_section_edit_buffers must not touch {key}"


def test_toggle_section_edit_save_clears_snapshot_and_keeps_buffer():
    from collections.abc import MutableMapping

    from app import toggle_section_edit

    state: MutableMapping[str, object] = {
        "subjective_editing": True,
        "subjective_edit": "user-typed text",
        "subjective_edit_snapshot": "original text",
    }
    toggle_section_edit(state, "Subjective", save=True)

    assert state["subjective_editing"] is False
    assert state["subjective_edit"] == "user-typed text"
    assert state["subjective_edit_snapshot"] is None


def test_toggle_section_edit_cancel_restores_buffer_from_snapshot():
    from collections.abc import MutableMapping

    from app import toggle_section_edit

    state: MutableMapping[str, object] = {
        "objective_editing": True,
        "objective_edit": "user-typed text we don't want",
        "objective_edit_snapshot": "original text",
    }
    toggle_section_edit(state, "Objective", save=False)

    assert state["objective_editing"] is False
    assert state["objective_edit"] == "original text"
    assert state["objective_edit_snapshot"] is None


def test_toggle_section_edit_cancel_with_no_snapshot_keeps_buffer():
    """Defensive: if a Cancel fires without a snapshot in place, don't crash
    or wipe the buffer — just flip the flag."""
    from collections.abc import MutableMapping

    from app import toggle_section_edit

    state: MutableMapping[str, object] = {
        "plan_editing": True,
        "plan_edit": "current text",
        "plan_edit_snapshot": None,
    }
    toggle_section_edit(state, "Plan", save=False)

    assert state["plan_editing"] is False
    assert state["plan_edit"] == "current text"
    assert state["plan_edit_snapshot"] is None


def test_state_e_lands_in_read_mode_for_all_sections(booted_app):
    """Cards must render with *_editing all False after a stream — the
    `populate_section_edit_buffers` invariant in spec §6.5."""
    at = booted_app
    _seed_state_e(at)
    at.run(timeout=30)

    for key in (
        "subjective_editing",
        "objective_editing",
        "assessment_editing",
        "plan_editing",
    ):
        assert at.session_state[key] is False, f"{key} should default to False"

    # Pencil edit buttons should be present (read mode).
    edit_btn_keys = {
        f"edit_{n.lower()}_btn" for n in ("Subjective", "Objective", "Assessment", "Plan")
    }
    actual_keys = {b.key for b in at.button if b.key}
    assert edit_btn_keys.issubset(actual_keys), f"missing pencil buttons; have {actual_keys}"


def test_state_e_pencil_click_enters_edit_mode(booted_app):
    """Clicking the Subjective pencil flips its *_editing flag and exposes
    Save/Cancel buttons."""
    at = booted_app
    _seed_state_e(at)
    at.run(timeout=30)

    pencil = next(b for b in at.button if b.key == "edit_subjective_btn")
    pencil.click()
    at.run(timeout=30)

    assert at.session_state["subjective_editing"] is True
    assert at.session_state["subjective_edit_snapshot"] is not None

    actual_keys = {b.key for b in at.button if b.key}
    assert "save_subjective_btn" in actual_keys
    assert "cancel_subjective_btn" in actual_keys


def test_state_e_cancel_reverts_buffer(booted_app):
    """Editing then canceling restores the buffer from the snapshot."""
    at = booted_app
    _seed_state_e(at)
    at.run(timeout=30)

    next(b for b in at.button if b.key == "edit_subjective_btn").click()
    at.run(timeout=30)

    # Mutate the buffer (simulating typed edits).
    at.session_state["subjective_edit"] = "TYPED OVER"
    at.run(timeout=30)

    next(b for b in at.button if b.key == "cancel_subjective_btn").click()
    at.run(timeout=30)

    assert at.session_state["subjective_editing"] is False
    # _seed_state_e populated subjective_edit from _MINIMAL_SOAP — body "foo".
    assert at.session_state["subjective_edit"] == "foo"
    assert at.session_state["subjective_edit_snapshot"] is None


def test_regenerate_while_editing_clears_editing_flags_and_snapshots(booted_app):
    """Regenerate while a section is in edit mode must clear the
    edit-mode meta-state. Otherwise a subsequent Cancel could restore
    the stale pre-regenerate body via the snapshot, silently
    overwriting the freshly streamed SOAP.

    The fix lives in _render_transcript_pane's primary-action click
    handler — alongside the *_edit buffer clear, it must also clear
    *_editing and *_edit_snapshot."""
    at = booted_app
    _seed_state_e(at)
    at.run(timeout=30)

    # Enter edit mode on Subjective so a snapshot exists.
    next(b for b in at.button if b.key == "edit_subjective_btn").click()
    at.run(timeout=30)

    assert at.session_state["subjective_editing"] is True
    assert at.session_state["subjective_edit_snapshot"] is not None

    # Click the primary action (Regenerate SOAP).
    regen = next(b for b in at.button if b.key == "generate_btn")
    regen.click()
    at.run(timeout=30)

    # The click handler should have cleared the editing meta-state.
    # The streaming flag is now True, so don't run further; just verify
    # the session_state shape that the click handler produced.
    for key in (
        "subjective_editing",
        "objective_editing",
        "assessment_editing",
        "plan_editing",
    ):
        assert at.session_state[key] is False, f"{key} should be False after Regenerate"
    for key in (
        "subjective_edit_snapshot",
        "objective_edit_snapshot",
        "assessment_edit_snapshot",
        "plan_edit_snapshot",
    ):
        assert at.session_state[key] is None, f"{key} should be None after Regenerate"


def test_full_user_flow_record_to_copy(booted_app):
    """End-to-end smoke test of the five-state machine, exercising the
    transitions between states. Each step synthesizes state changes
    (rather than driving real ASR / streaming) and verifies the wiring
    that connects per-state renders.

    The per-helper unit tests cover their slice; this test catches
    regressions in the seams between slices — e.g., an `_streaming` flag
    that doesn't get cleared on stream end, or a `populate_section_edit_buffers`
    call that doesn't run after the stream loop.
    """
    at = booted_app

    # ---- State A: no audio loaded ----
    assert at.session_state["audio_bytes"] is None
    actual_keys = {b.key for b in at.button if b.key}
    assert "new_session_btn" in actual_keys, "+ New session button missing in State A"

    # ---- Transition A → B: seed audio ----
    at.session_state["audio_bytes"] = b"fake-audio"
    at.session_state["audio_name"] = "visit.wav"
    at.session_state["audio_hash"] = "deadbeef"
    at.run(timeout=30)

    # State B: top bar shows "Transcribing…" (active chip), right pane
    # shows the awaiting placeholder.
    markdown_blocks = " ".join(m.value for m in at.markdown)
    assert "Transcribing" in markdown_blocks, "State B chip should show Transcribing label"
    assert "ms-stage-active" in markdown_blocks, "State B chip should be active variant"
    assert "Awaiting transcript" in markdown_blocks, "State B right pane should show placeholder"

    # ---- Transition B → C: seed transcript + trim metadata ----
    # Re-seed audio_bytes because the State-B run above triggers _render_transcript_pane's
    # transcription branch (tx is None), which calls reset_state() on ASR failure and
    # clears audio_bytes.  The test synthesises the post-transcription state here.
    at.session_state["audio_bytes"] = b"fake-audio"
    at.session_state["audio_name"] = "visit.wav"
    at.session_state["audio_hash"] = "deadbeef"
    at.session_state["tx"] = "Patient reports chest pain."
    at.session_state["tx_edit"] = "Patient reports chest pain."
    at.session_state["tx_trim"] = type(
        "TR", (), {"original_seconds": 540.0, "trimmed_seconds": 254.0, "status": "trimmed"}
    )()
    at.run(timeout=30)

    # State C: top bar shows "Transcript ready" (static chip), right pane
    # shows the model+timing placeholder, primary button reads "Generate SOAP note".
    markdown_blocks = " ".join(m.value for m in at.markdown)
    assert "Transcript ready" in markdown_blocks, "State C chip should show Transcript ready"
    assert "ms-stage-static" in markdown_blocks, "State C chip should be static variant"
    assert "Ready to draft a SOAP note" in markdown_blocks, "State C placeholder should be visible"

    from medical_scribe import MODEL_DISPLAY_NAME

    assert MODEL_DISPLAY_NAME in markdown_blocks, "State C placeholder should name the model"

    button_labels = {b.label for b in at.button if b.label}
    assert any("Generate SOAP note" in label for label in button_labels), (
        f"Primary button should read 'Generate SOAP note' in State C; saw: {button_labels}"
    )

    # ---- Transition C → D: simulate Generate click (streaming begins) ----
    # Set the streaming flag directly; AppTest can't freeze mid-stream, so
    # we verify State D by inspecting derive_stage_label against the current
    # session state rather than via a rendered rerun (the mocked LLM would
    # complete streaming in the same rerun, jumping straight to State E).
    from app import derive_stage_label

    at.session_state["_streaming"] = True
    at.session_state["soap"] = None  # Clear any leftover soap
    state_snapshot = {k: at.session_state[k] for k in ["audio_bytes", "tx", "_streaming", "soap"]}
    d_label = derive_stage_label(state_snapshot)
    assert d_label == "Generating SOAP…", (
        f"State D label should be 'Generating SOAP…'; got {d_label!r}"
    )
    from app import STAGE_CHIP_VARIANTS

    assert STAGE_CHIP_VARIANTS[d_label] == "active", "State D chip variant should be active"

    # ---- Transition D → E: simulate stream completion ----
    from collections.abc import MutableMapping
    from typing import cast

    from app import populate_section_edit_buffers

    soap_text = "## Subjective\nfoo\n## Objective\no\n## Assessment\na\n## Plan\np\n"
    at.session_state["soap"] = soap_text
    populate_section_edit_buffers(cast(MutableMapping[str, object], at.session_state), soap_text)
    at.session_state["_streaming"] = False
    at.run(timeout=30)

    # State E: chip flips to "SOAP ready" (static), all four cards land in
    # read mode (*_editing all False), pencil buttons exposed, Copy button
    # present in the bottom bar.
    markdown_blocks = " ".join(m.value for m in at.markdown)
    assert "SOAP ready" in markdown_blocks, "State E chip should show SOAP ready"
    assert "ms-stage-static" in markdown_blocks, "State E chip should be static variant"

    for key in (
        "subjective_editing",
        "objective_editing",
        "assessment_editing",
        "plan_editing",
    ):
        assert at.session_state[key] is False, (
            f"{key} should be False in State E (read mode default)"
        )

    actual_keys = {b.key for b in at.button if b.key}
    for section in ("subjective", "objective", "assessment", "plan"):
        assert f"edit_{section}_btn" in actual_keys, (
            f"pencil button for {section} missing in State E"
        )

    # ---- State E round-trip: edit Subjective, then cancel ----
    pencil = next(b for b in at.button if b.key == "edit_subjective_btn")
    pencil.click()
    at.run(timeout=30)

    assert at.session_state["subjective_editing"] is True, "pencil click should flip editing flag"
    assert at.session_state["subjective_edit_snapshot"] is not None, (
        "pencil click should capture a snapshot"
    )

    actual_keys = {b.key for b in at.button if b.key}
    assert "save_subjective_btn" in actual_keys, "Save button should appear in edit mode"
    assert "cancel_subjective_btn" in actual_keys, "Cancel button should appear in edit mode"

    # Mutate the buffer (simulating typed edits), then cancel.
    at.session_state["subjective_edit"] = "TYPED OVER"
    at.run(timeout=30)
    cancel = next(b for b in at.button if b.key == "cancel_subjective_btn")
    cancel.click()
    at.run(timeout=30)

    assert at.session_state["subjective_editing"] is False, "cancel should flip editing flag back"
    # _MINIMAL_SOAP's Subjective body was "foo" — that's what populate_section_edit_buffers
    # wrote, and what the snapshot captured on pencil click.
    assert at.session_state["subjective_edit"] == "foo", (
        "cancel should restore subjective_edit from the snapshot"
    )
    assert at.session_state["subjective_edit_snapshot"] is None, "cancel should clear the snapshot"
