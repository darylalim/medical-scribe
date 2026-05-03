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
    touches and which it leaves alone (workflow flags are
    stage-orthogonal and must survive both clear paths)."""
    return {
        "audio_bytes": b"abc",
        "audio_name": "a.wav",
        "audio_hash": "deadbeef",
        "tx": "some transcript",
        "tx_edit": "edited transcript",
        "soap": "generated soap",
        "soap_truncated": True,
        "subjective_edit": "edited s",
        "objective_edit": "edited o",
        "assessment_edit": "edited a",
        "plan_edit": "edited p",
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
    """Smoke test for the split-view shell: header + sidebar + State-A chooser."""
    at = booted_app

    # Header markdown contains "Medical Scribe".
    rendered_md = " ".join(md.value for md in at.markdown)
    assert "Medical Scribe" in rendered_md

    # Sidebar has a `+ New session` button.
    sidebar_button_labels = [b.label for b in at.sidebar.button]
    assert any("New session" in label for label in sidebar_button_labels)

    # State-A chooser renders both affordances. Tab buttons no longer exist.
    main_button_labels = [b.label for b in at.button]
    assert "Transcript" not in main_button_labels
    assert "Notes" not in main_button_labels

    # Notes-tab placeholder copy is gone (never reached State C).
    assert "No SOAP note yet" not in rendered_md


def test_new_session_in_state_a_bypasses_dialog(booted_app):
    """Regression guard for `_render_sidebar`'s State-A bypass:
    when no audio is loaded there's nothing to lose, so clicking
    + New session must NOT open the confirmation dialog. Catches the
    case where the gating on `audio_bytes` is removed or inverted."""
    at = booted_app

    new_session_btn = next(b for b in at.sidebar.button if "New session" in b.label)
    new_session_btn.click().run(timeout=30)
    assert not at.exception, f"click raised: {at.exception}"

    # Bypass path took effect: dialog flag stayed False, no dialog opens.
    assert at.session_state["_show_reset_dialog"] is False


def test_new_session_with_audio_opens_dialog(booted_app):
    """Regression guard for the destructive-confirm half of
    `_render_sidebar`: when audio is loaded the click must arm the
    dialog (set `_show_reset_dialog=True`) and must NOT wipe state.
    Catches the case where the bypass branch fires in both directions."""
    at = booted_app

    # Inject a loaded-audio state and re-render so the sidebar sees it.
    at.session_state["audio_bytes"] = b"fake-audio-bytes"
    at.session_state["audio_name"] = "fake.wav"
    at.session_state["audio_hash"] = "fake-hash"
    at.session_state["tx"] = "fake transcript"
    at.session_state["tx_edit"] = "fake transcript"
    at.run(timeout=30)
    assert not at.exception

    new_session_btn = next(b for b in at.sidebar.button if "New session" in b.label)
    new_session_btn.click().run(timeout=30)
    assert not at.exception, f"click raised: {at.exception}"

    # Dialog armed; underlying state preserved (the dialog gates the wipe).
    assert at.session_state["_show_reset_dialog"] is True
    assert at.session_state["audio_bytes"] == b"fake-audio-bytes"
    assert at.session_state["tx"] == "fake transcript"


def test_initial_state_includes_new_keys():
    """Locks the session-state shape introduced by the live-capture-and-tabs
    redesign and amended by the split-view redesign."""
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
    click path in Task 3 (which calls this helper after resetting `soap`)."""
    from app import SECTION_KEY_MAP, populate_section_edit_buffers

    state: dict = dict.fromkeys(SECTION_KEY_MAP.values(), "preexisting")

    populate_section_edit_buffers(state, "")

    for key in SECTION_KEY_MAP.values():
        assert state[key] == ""


def test_state_a_renders_mic_and_upload_without_expander(booted_app):
    """State A landing exposes both mic and upload affordances without
    an expander click. Regression guard for the redesign goal of
    'discoverability for first-time clinicians testing with a sample
    recording'. The old layout hid upload behind a 'Or upload an existing
    recording' expander."""
    at = booted_app

    rendered_md = " ".join(md.value for md in at.markdown)

    # Both affordance labels render at the top level.
    assert "Record this visit" in rendered_md, "mic affordance label missing"
    assert "Upload a recording" in rendered_md, "upload affordance label missing"

    # The old expander label must not appear anywhere — its presence would
    # mean upload is still hidden behind a click.
    assert "Or upload an existing recording" not in rendered_md, (
        "upload should not be inside an expander"
    )

    # Structural check: no expanders rendered (resilient to label changes
    # — a future implementation that uses a different expander label
    # would otherwise silently pass).
    assert len(at.expander) == 0, "upload should not be inside an expander"

    # File-uploader widget directly present in the DOM. (audio_input has
    # no AppTest accessor at Streamlit 1.39, so we don't assert its
    # presence structurally — markdown label assertion above is the
    # best we can do for that widget.)
    assert len(at.file_uploader) == 1, "file_uploader_widget not rendered"


def test_state_c_renders_transcript_and_soap_panes_simultaneously(booted_app):
    """When transcript and SOAP both exist (State C / SOAP-ready), the split
    view renders the transcript text_area AND at least one SOAP card body
    text_area in a single render pass. Regression guard for the redesign's
    primary win — verifying SOAP claims against the transcript without
    tab-switching."""
    at = booted_app
    _seed_state_e(at)
    at.run(timeout=30)
    assert not at.exception, f"render raised: {at.exception}"

    # At least 2 text_areas: one for the transcript, at least one for a SOAP card.
    # Today's render pre-redesign would only show ONE (whichever tab is active).
    assert len(at.text_area) >= 2, (
        f"split view should render transcript + at least one SOAP card "
        f"text_area in the same pass; saw {len(at.text_area)}"
    )


def test_initial_state_excludes_active_tab_and_is_editing():
    """Drift guard against accidental re-introduction. Both keys were
    removed in the split-view redesign — `active_tab` because tabs are gone,
    `is_editing` because cards are always-editable post-stream."""
    from app import INITIAL_STATE

    assert "active_tab" not in INITIAL_STATE
    assert "is_editing" not in INITIAL_STATE


def test_card_edit_buffer_persists_across_reruns(booted_app):
    """Always-editable card behavior: writing to a *_edit buffer and
    re-rendering preserves the value across reruns. The CLAUDE.md
    invariant about value= + manual sync (not key=) protects this in
    conditionally-rendered branches; the test exercises the basic
    persistence path."""
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
    mocker.patch("app._render_transcript_pane", lambda asr_pipe: None)
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
    shows the 'Click Generate SOAP note on the left…' placeholder. Locks the
    'no blank pane' UX guarantee — without this, the user would see only the
    transcript on the left and an empty right column until they click Generate."""
    at = booted_app
    _seed_state_c(at)
    at.run(timeout=30)
    assert not at.exception, f"render raised: {at.exception}"

    rendered_md = " ".join(md.value for md in at.markdown)
    assert "Click **Generate SOAP note**" in rendered_md, (
        f"State C right pane should show the 'Click Generate SOAP note…' "
        f"placeholder; rendered markdown: {rendered_md!r}"
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
        # 8m 56s → 4m 44s = 4m 12s removed (47%)
        (536.0, 284.0, "Trimmed 4m 12s of silence (47% of recording)."),
        # 1m → 30s = 30s removed (50%)
        (60.0, 30.0, "Trimmed 0m 30s of silence (50% of recording)."),
        # 13m → 1m = 12m removed (92%)
        (780.0, 60.0, "Trimmed 12m 0s of silence (92% of recording)."),
        # 2h 10m → 1h 5m = 1h 5m removed (50%)
        (7800.0, 3900.0, "Trimmed 1h 5m of silence (50% of recording)."),
    ],
    ids=["typical_47pct", "short_30s", "twelve_min_zero_sec", "over_one_hour"],
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
