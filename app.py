"""Streamlit UI for the MedASR -> MedGemma SOAP pipeline.

Sidebar + Transcript/Notes tabs. Live capture via st.audio_input; file
upload as a secondary affordance. Notes tab renders the SOAP draft as
four cards (Subjective / Objective / Assessment / Plan) with an
explicit Edit mode. Copy to clipboard is the only export — nothing is
written to disk."""

from __future__ import annotations

import hashlib
import html
import json
import os
import sys
import traceback
from collections.abc import Mapping, MutableMapping
from typing import cast

from dotenv import load_dotenv

# Must run before any HF/MLX import — they read HF_TOKEN at import time.
load_dotenv()

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from medical_scribe import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_ID,
    SECTION_HEADER_RE,
    SOAP_SECTIONS,
    assemble_soap,
    format_for_clipboard,
    load_asr_pipeline,
    load_medgemma,
    parse_soap_sections,
    pick_device,
    stream_soap,
    transcribe,
)

ASR_MODEL = "google/medasr"
MAX_UPLOAD_MB = 100

# Derived from SOAP_SECTIONS so adding a section in soap_sections.py
# auto-propagates the per-section edit-buffer key. The matching INITIAL_STATE
# entries below still need a manual addition; tests/test_app.py's
# test_initial_state_keys_match_section_key_map catches that drift.
SECTION_KEY_MAP: dict[str, str] = {name: f"{name.lower()}_edit" for name in SOAP_SECTIONS}

# Visual vocabulary for the SOAP cards. Both color *and* letter chip
# are used together — color alone fails for ~8% of male users, and
# clinicians scan fast. test_section_color_and_initial_maps_cover_all_soap_sections
# catches drift if SOAP_SECTIONS grows a fifth section.
# White-on-color contrast must clear WCAG AA (4.5:1) for the chip letter
# to remain readable. The amber/green Tailwind-500 shades fail badly
# (~2:1); the -700 shades clear AA. Blue and purple at -500 only just
# fail AA but the section name is the primary label, so the chip is
# decorative — keeping them as-is preserves the visual identity that's
# already been browser-verified.
SECTION_COLORS: dict[str, str] = {
    "Subjective": "#3b82f6",  # blue-500
    "Objective": "#047857",  # emerald-700 (darkened for contrast)
    "Assessment": "#b45309",  # amber-700 (darkened for contrast)
    "Plan": "#8b5cf6",  # violet-500
}

SECTION_INITIALS: dict[str, str] = {
    "Subjective": "S",
    "Objective": "O",
    "Assessment": "A",
    "Plan": "P",
}

INITIAL_STATE = {
    # Audio
    "audio_bytes": None,
    "audio_name": None,
    "audio_hash": None,
    # Transcript
    "tx": None,
    "tx_edit": "",
    # SOAP — full markdown blob, source of truth
    "soap": None,
    "soap_truncated": False,
    # UI mode
    "active_tab": "transcript",
    "is_editing": False,
    # Per-section edit buffers (populated on entry to edit mode)
    "subjective_edit": "",
    "objective_edit": "",
    "assessment_edit": "",
    "plan_edit": "",
    # Streaming flag — set when SOAP generation is in progress.
    # Read via st.session_state.get(...) elsewhere; including the default
    # here ensures reset_state() clears it on `+ New session`.
    "_streaming": False,
    # Confirmation-dialog flag for `+ New session`. Persisting in
    # session_state (rather than relying on @st.dialog's internal state)
    # keeps the dialog open across the rerun that follows the button
    # click, and makes the bypass-in-State-A path explicit.
    "_show_reset_dialog": False,
}


def init_state() -> None:
    for k, v in INITIAL_STATE.items():
        st.session_state.setdefault(k, v)


def reset_state() -> None:
    for k, v in INITIAL_STATE.items():
        st.session_state[k] = v


def clear_downstream_state(state: MutableMapping[str, object], after: str) -> None:
    """Enforce the spec's state invariants. `after` names the last valid stage.

    `active_tab` is intentionally not touched — it's a UI focus concern,
    orthogonal to workflow stage. `+ New session` is the only path that
    resets it (via reset_state).
    """
    if after == "audio":
        state["tx"] = None
        state["tx_edit"] = ""
        state["soap"] = None
        state["soap_truncated"] = False
        state["is_editing"] = False
        state["subjective_edit"] = ""
        state["objective_edit"] = ""
        state["assessment_edit"] = ""
        state["plan_edit"] = ""
    elif after == "tx":
        state["soap"] = None
        state["soap_truncated"] = False
        state["is_editing"] = False
        state["subjective_edit"] = ""
        state["objective_edit"] = ""
        state["assessment_edit"] = ""
        state["plan_edit"] = ""


EXT_TO_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
}


def audio_mime_from_name(name: object) -> str | None:
    """Map filename extension to MIME for st.audio.

    Returns None for missing or unrecognized inputs; the file_uploader's
    type filter constrains live uploads to the four keys in EXT_TO_MIME,
    so the None branch is defensive.
    """
    if not isinstance(name, str) or "." not in name:
        return None
    return EXT_TO_MIME.get(name.rsplit(".", 1)[-1].lower())


def derive_stage_label(state: Mapping[str, object]) -> str:
    """Header stage label, derived from session state.

    The `_streaming` flag is set by the Generate click handler before
    streaming begins (via st.rerun), so it is available in session state
    when the header renders on the streaming pass.
    """
    if state.get("audio_bytes") is None:
        return "No audio loaded"
    if state.get("tx") is None:
        return "Transcribing…"
    if state.get("_streaming"):
        return "Generating SOAP…"
    if state.get("soap") is None:
        return "Transcript ready"
    return "SOAP ready"


def update_truncation_flag(state: MutableMapping[str, object], meta: Mapping[str, object]) -> None:
    """Set state['soap_truncated'] based on streaming meta's finish_reason."""
    state["soap_truncated"] = meta.get("finish_reason") == "length"


def primary_action_label(soap: object) -> str:
    """Label for the primary action button.

    Returns 'Regenerate SOAP' when a SOAP draft already exists (truthy
    `soap`), 'Generate SOAP note' otherwise. The click handler is the same
    either way (idempotent — discards section edits and re-runs against the
    current transcript); the label flip is purely to surface the destructive
    nature post-SOAP."""
    return "Regenerate SOAP" if soap else "Generate SOAP note"


def populate_section_edit_buffers(state: MutableMapping[str, object], soap: str) -> None:
    """Populate the four `*_edit` session-state buffers from a SOAP markdown blob.

    Called once on stream completion so the post-stream rerun lands with
    cards already in always-editable mode and their bodies pre-filled.
    Sections missing from `soap` get '' (defensive against partial / truncated
    model output)."""
    parsed = parse_soap_sections(soap)
    for name in SOAP_SECTIONS:
        state[SECTION_KEY_MAP[name]] = parsed.get(name, "")


def streaming_status_label(section_names: list[str]) -> str:
    """Label for the streaming-status placeholder.

    The last detected section is the one currently being drafted — any
    earlier section is "complete" the moment a successor header appears.
    Empty list means we haven't seen any section header yet (preamble or
    very early stream).
    """
    if not section_names:
        return "Generating…"
    return f"Drafting {section_names[-1]}…"


def compute_unparsed_remainder(soap: str | None, parsed: dict[str, str]) -> str:
    """Return text in `soap` that isn't part of any recognised SOAP section.

    Two cases produce a non-empty remainder:
    - `parsed` is empty (no `## Subjective`/`## Objective`/`## Assessment`/
      `## Plan` headers found in `soap`): the entire `soap` (stripped).
    - `parsed` is non-empty: any preamble before the first recognised H2
      header (stripped). Per `parse_soap_sections`'s greedy header-to-header
      semantics, mid-buffer content is always absorbed by the preceding
      section, so preamble is the only place stray text can land.

    Empty/None input returns "".
    """
    if not soap:
        return ""
    if not parsed:
        return soap.strip()
    match = SECTION_HEADER_RE.search(soap)
    return soap[: match.start()].strip() if match else ""


def escape_text_for_inline_script(text: str) -> str:
    """JSON-encode `text` for safe inline JavaScript, plus replace `</`
    with `<\\/` so a `</script>` substring inside `text` cannot prematurely
    close an enclosing `<script>` tag.

    Used by `copy_to_clipboard_button` to embed user-controlled SOAP content
    in a `<script>` block via `streamlit.components.v1.html`.
    """
    return json.dumps(text).replace("</", "<\\/")


def copy_to_clipboard_button(text: str, *, label: str = "Copy to clipboard", key: str) -> None:
    """Render a Copy-to-clipboard button using the JavaScript Clipboard API.

    Uses `streamlit.components.v1.html` (iframe-based component) rather
    than `st.html` because the latter strips/sanitizes inline event
    handlers in current Streamlit versions, causing silent click
    failures. The component iframe runs JavaScript reliably and Streamlit
    grants it `clipboard-write` permission by default.

    Click writes `text` to the clipboard via `navigator.clipboard.writeText`,
    flashes a "✓ Copied" toast for 1.5s on success, and surfaces a "✗ Copy
    failed" toast plus a console.error on failure (so future regressions
    are visible).
    """
    # JSON encoding handles JS string escaping (quotes, newlines).
    # Replace "</" with "<\/" so a `</script>` substring in `text` cannot
    # prematurely terminate the inline <script> block.
    payload = escape_text_for_inline_script(text)
    safe_label = html.escape(label)
    btn_style = (
        "padding:0.45em 1.1em; border-radius:6px;"
        " border:1px solid rgba(49,51,63,0.2);"
        " background:#ff4b4b; color:white; cursor:pointer;"
        " font-weight:500; font-size:14px;"
    )
    components.html(
        f"""
<button id="{key}" style="{btn_style}">{safe_label}</button>
<script>
  (function() {{
    const btn = document.getElementById("{key}");
    btn.addEventListener("click", function() {{
      navigator.clipboard.writeText({payload}).then(function() {{
        const original = btn.textContent;
        btn.textContent = "\u2713 Copied";
        setTimeout(function() {{ btn.textContent = original; }}, 1500);
      }}).catch(function(err) {{
        console.error("Copy to clipboard failed:", err);
        const original = btn.textContent;
        btn.textContent = "\u2717 Copy failed";
        setTimeout(function() {{ btn.textContent = original; }}, 2000);
      }});
    }});
  }})();
</script>
""",
        height=50,
    )


def require_hf_token() -> None:
    if not os.environ.get("HF_TOKEN"):
        st.error(
            "HF_TOKEN not set — copy .env.example to .env and add your "
            "Hugging Face token (https://huggingface.co/settings/tokens)."
        )
        st.stop()


@st.cache_resource
def _asr():
    return load_asr_pipeline(ASR_MODEL, pick_device())


@st.cache_resource
def _llm():
    return load_medgemma(DEFAULT_MODEL_ID)


def show_error(label: str, exc: BaseException) -> None:
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    st.error(f"{label}: {type(exc).__name__}: {exc}")


def _soap_chip_styles_html() -> str:
    """One-shot CSS for the SOAP card chips. Injected once per page
    render via main(). Per-section background color is read from
    SECTION_COLORS so adding a section in soap_sections.py also
    auto-propagates here (paired with the SECTION_COLORS drift guard
    in tests/test_app.py)."""
    color_rules = "\n".join(
        f".soap-chip-{name.lower()} {{ background: {color}; }}"
        for name, color in SECTION_COLORS.items()
    )
    return f"""
<style>
.soap-section-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}}
.soap-chip {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 6px;
    color: white;
    font-weight: 600;
    font-size: 13px;
    flex-shrink: 0;
}}
.soap-section-name {{
    font-weight: 600;
    letter-spacing: 0.04em;
}}
{color_rules}
</style>
"""


def _render_section_header(name: str) -> None:
    """Colored letter-chip + section name. Reused by all three card render
    paths (streaming, read, edit). Visual styles live in
    _soap_chip_styles_html(), injected once per page render via main()."""
    initial = html.escape(SECTION_INITIALS[name])
    label = html.escape(name.upper())
    st.markdown(
        f'<div class="soap-section-header">'
        f'<span class="soap-chip soap-chip-{name.lower()}">{initial}</span>'
        f'<span class="soap-section-name">{label}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    """Top header bar: app title (left) + stage label (right)."""
    cols = st.columns([3, 5])
    with cols[0]:
        st.markdown("### Medical Scribe")
    with cols[1]:
        stage_label = derive_stage_label(cast(Mapping[str, object], st.session_state))
        st.markdown(
            f"<div style='text-align:right; padding-top:8px; color:#666'>"
            f"<em>{html.escape(stage_label)}</em></div>",
            unsafe_allow_html=True,
        )
    st.divider()


@st.dialog("Discard this session?")
def _confirm_new_session_dialog() -> None:
    """Modal confirmation for `+ New session` when there's work to lose.

    Driven by the `_show_reset_dialog` session-state flag rather than by
    @st.dialog's internal lifecycle, so the dialog persists across the
    button-click rerun and either button explicitly closes it.
    """
    st.write(
        "Starting a new session will discard the current audio, "
        "transcript, and SOAP draft. This cannot be undone."
    )
    cols = st.columns(2)
    with cols[0]:
        if st.button("Cancel", key="dlg_cancel_btn", use_container_width=True):
            st.session_state["_show_reset_dialog"] = False
            st.rerun()
    with cols[1]:
        if st.button(
            "Discard",
            key="dlg_confirm_btn",
            type="primary",
            use_container_width=True,
        ):
            reset_state()
            st.rerun()


def _render_sidebar() -> None:
    """Sidebar: `+ New session` button. In State A (no audio loaded) the
    click is direct; otherwise it opens a confirmation dialog so a stray
    click can't wipe an in-progress draft."""
    with st.sidebar:
        if st.button(
            "+ New session", key="new_session_btn", type="primary", use_container_width=True
        ):
            if st.session_state.get("audio_bytes") is None:
                reset_state()
                st.rerun()
            else:
                st.session_state["_show_reset_dialog"] = True
                st.rerun()

    if st.session_state.get("_show_reset_dialog"):
        _confirm_new_session_dialog()


def _render_tab_bar() -> None:
    """Two buttons styled as tabs. `active_tab` session-state key
    decides which tab body the dispatcher renders.

    Native st.tabs() is client-side only — the active tab cannot be
    set from Python — so the Generate handler couldn't auto-switch
    to Notes. Hand-rolled tabs solve that.
    """
    active = st.session_state["active_tab"]
    cols = st.columns([1, 1, 6])
    with cols[0]:
        if st.button(
            "Transcript",
            key="tab_transcript_btn",
            type="primary" if active == "transcript" else "secondary",
            use_container_width=True,
        ):
            st.session_state["active_tab"] = "transcript"
            st.rerun()
    with cols[1]:
        if st.button(
            "Notes",
            key="tab_notes_btn",
            type="primary" if active == "notes" else "secondary",
            use_container_width=True,
        ):
            st.session_state["active_tab"] = "notes"
            st.rerun()
    st.divider()


def _handle_upload(upload) -> bool:
    """Validate and persist a new audio capture (recorded or uploaded)
    into session state.

    Accepts the UploadedFile-compatible object returned by either
    `st.audio_input` or `st.file_uploader`. SHA-256 hash detects "is
    this actually new audio?" so unrelated reruns don't re-transcribe.
    Returns True if new audio was persisted, False otherwise.
    """
    incoming_bytes = upload.getvalue()
    if len(incoming_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        size_mb = len(incoming_bytes) // (1024 * 1024)
        st.error(
            f"File too large ({size_mb} MB). Maximum upload size is "
            f"{MAX_UPLOAD_MB} MB — please split long recordings."
        )
        st.stop()
    incoming_name = getattr(upload, "name", "audio.wav")
    incoming_hash = hashlib.sha256(incoming_bytes).hexdigest()
    is_new_upload = (
        incoming_name != st.session_state["audio_name"]
        or incoming_hash != st.session_state["audio_hash"]
    )
    if is_new_upload:
        st.session_state["audio_bytes"] = incoming_bytes
        st.session_state["audio_name"] = incoming_name
        st.session_state["audio_hash"] = incoming_hash
        clear_downstream_state(cast(MutableMapping[str, object], st.session_state), after="audio")
        return True
    return False


def _render_state_a_chooser() -> None:
    """State A landing — mic + upload affordances side by side.

    Called only when `audio_bytes is None` (State A). Caller is
    responsible for the gating; this function does not check.

    Replaces the prior centered single-column with the upload widget hidden
    behind 'Or upload an existing recording' expander. The new layout makes
    both paths first-class so a first-time clinician testing with a sample
    recording sees upload immediately."""
    cols = st.columns([1, 1])
    with cols[0]:  # noqa: SIM117
        with st.container(border=True):
            st.markdown("**Record this visit**")
            st.caption("Click the mic to start, click again to stop. Audio stays on this device.")
            recorded = st.audio_input(
                "Record audio",
                label_visibility="collapsed",
                key="audio_input_widget",
            )
            if recorded is not None and _handle_upload(recorded):
                st.rerun()
    with cols[1]:  # noqa: SIM117
        with st.container(border=True):
            st.markdown("**Upload a recording**")
            st.caption("WAV, MP3, FLAC, or M4A. Max 100 MB.")
            uploaded = st.file_uploader(
                "Audio file",
                type=["wav", "mp3", "flac", "m4a"],
                label_visibility="collapsed",
                key="file_uploader_widget",
            )
            if uploaded is not None and _handle_upload(uploaded):
                st.rerun()


def _render_transcript_tab(asr_pipe) -> None:
    """Transcript tab: audio capture/upload + editable transcript +
    Generate SOAP Note button. State-aware."""
    audio_bytes = st.session_state["audio_bytes"]
    tx = st.session_state["tx"]
    is_streaming = bool(st.session_state.get("_streaming"))

    # State A: no audio yet.
    if audio_bytes is None:
        _render_state_a_chooser()
        return

    # States B-E: audio captured. Audio player at top.
    mime = audio_mime_from_name(st.session_state["audio_name"])
    st.audio(audio_bytes, format=mime if mime is not None else "audio/wav")

    # State B: transcribing.
    if tx is None:
        with st.spinner("Transcribing audio…"):
            try:
                text = transcribe(asr_pipe, audio_bytes)
            except Exception as exc:
                show_error("Could not transcribe audio", exc)
                reset_state()
                st.stop()
        st.session_state["tx"] = text
        st.session_state["tx_edit"] = text
        st.rerun()

    # States C / D / E: editable transcript (disabled during streaming).
    # Use value= + manual sync rather than key= so the value persists
    # across tab switches. Streamlit clears widget-managed keys when the
    # widget unmounts (e.g. when the user navigates to the Notes tab),
    # which would silently wipe the transcript on return.
    new_tx_edit = st.text_area(
        "Transcript",
        value=st.session_state["tx_edit"],
        height=400,
        label_visibility="collapsed",
        disabled=is_streaming,
    )
    st.session_state["tx_edit"] = new_tx_edit

    # Generate button (idempotent — re-clicking re-runs the LLM).
    cols = st.columns([4, 1])
    with cols[1]:
        if st.button(
            "Generate SOAP Note →",
            type="primary",
            disabled=is_streaming,
            key="generate_btn",
            use_container_width=True,
        ):
            if not st.session_state["tx_edit"].strip():
                st.warning(
                    "Transcript is empty — please provide or correct the "
                    "transcription before generating a SOAP note."
                )
                return
            # New generation: discard any in-progress edits.
            st.session_state["is_editing"] = False
            st.session_state["soap_truncated"] = False
            for key in SECTION_KEY_MAP.values():
                st.session_state[key] = ""
            st.session_state["active_tab"] = "notes"
            st.session_state["_streaming"] = True
            st.rerun()


def _render_notes_tab(model, tokenizer) -> None:
    """Notes tab: stage-aware. Placeholder → streaming cards →
    read-mode cards → edit-mode textareas."""
    audio_bytes = st.session_state["audio_bytes"]
    tx = st.session_state["tx"]
    soap = st.session_state["soap"]
    is_streaming = bool(st.session_state.get("_streaming"))

    # State A: no audio yet.
    if audio_bytes is None:
        st.markdown("_No SOAP note yet — record or upload audio in the Transcript tab._")
        return

    # State B: transcribing.
    if tx is None:
        st.markdown("_Awaiting transcript…_")
        return

    # State C: SOAP idle.
    if soap is None and not is_streaming:
        st.markdown("_Click Generate from the Transcript tab to draft a SOAP note._")
        return

    # State D: streaming.
    if is_streaming:
        cards_placeholder = st.empty()
        status_placeholder = st.empty()
        status_placeholder.markdown(f"_{streaming_status_label([])}_")
        buf = ""
        meta: dict[str, object] = {}
        last_complete_count = 0
        # Guard the status DOM update — without this we'd re-render the
        # markdown every chunk (dozens per section), which is visibly stuttery.
        last_status: str | None = None
        try:
            for chunk in stream_soap(
                model,
                tokenizer,
                st.session_state["tx_edit"],
                max_tokens=DEFAULT_MAX_TOKENS,
                meta=meta,
            ):
                buf += chunk
                sections = parse_soap_sections(buf)
                # A section is "complete" once a successor header appears.
                # During streaming, all detected sections except the last are complete.
                section_names = [n for n in SOAP_SECTIONS if n in sections]

                status_text = streaming_status_label(section_names)
                if status_text != last_status:
                    status_placeholder.markdown(f"_{status_text}_")
                    last_status = status_text

                complete_count = max(0, len(section_names) - 1)
                if complete_count > last_complete_count:
                    with cards_placeholder.container():
                        for name in section_names[:complete_count]:
                            with st.container(border=True):
                                _render_section_header(name)
                                st.markdown(sections[name])
                    last_complete_count = complete_count
        except Exception as exc:
            cards_placeholder.empty()
            status_placeholder.empty()
            show_error("SOAP generation failed", exc)
            st.session_state["soap"] = None
            st.session_state["soap_truncated"] = False
            st.session_state["_streaming"] = False
            return

        # Stream completed successfully. Persist and rerun to land in State E.
        # Don't call placeholder.empty() — Streamlit garbage-collects them on
        # the rerun, and explicit clearing causes a visible flicker before
        # State E's cards render in their place.
        st.session_state["soap"] = buf
        update_truncation_flag(cast(MutableMapping[str, object], st.session_state), meta)
        st.session_state["_streaming"] = False
        st.rerun()

    # States E and E-edit: SOAP exists.
    parsed = parse_soap_sections(soap or "")

    # Persistent truncation warning.
    if st.session_state.get("soap_truncated", False):
        st.warning(
            "The SOAP note reached the output token limit and may be incomplete. "
            "Verify all four sections (Subjective, Objective, Assessment, Plan) "
            "are present before copying."
        )

    # Defensive fallback: model output didn't parse into all four sections.
    expected = set(SOAP_SECTIONS)
    if expected - set(parsed):
        st.warning(
            "The SOAP note couldn't be fully parsed into sections — "
            "review carefully before copying."
        )

    if st.session_state["is_editing"]:
        # State E-edit: textareas in cards.
        # Use value= + manual sync rather than key= so the buffer values
        # persist across tab switches and edit-mode toggles. Same fix as
        # the transcript text area in _render_transcript_tab — Streamlit
        # clears widget-managed keys on unmount, which would silently
        # wipe in-progress edits if the user switches tabs mid-edit.
        for name in SOAP_SECTIONS:
            with st.container(border=True):
                _render_section_header(name)
                buffer_key = SECTION_KEY_MAP[name]
                new_value = st.text_area(
                    f"{name} edit",
                    value=st.session_state.get(buffer_key, ""),
                    height=120,
                    label_visibility="collapsed",
                )
                st.session_state[buffer_key] = new_value

        # Action row: Done · Copy to clipboard. `edits` is read once per
        # render and used by both buttons.
        edits = {name: st.session_state[SECTION_KEY_MAP[name]] for name in SOAP_SECTIONS}
        cols = st.columns([1, 2, 5])
        with cols[0]:
            if st.button("Done", key="done_btn", use_container_width=True):
                st.session_state["soap"] = assemble_soap(edits)
                st.session_state["is_editing"] = False
                st.rerun()
        with cols[1]:
            copy_to_clipboard_button(
                format_for_clipboard(edits),
                key="copy_btn_edit",
            )
    else:
        # State E: read-mode cards.
        for name in SOAP_SECTIONS:
            if name in parsed:
                with st.container(border=True):
                    _render_section_header(name)
                    st.markdown(parsed[name])

        # "Other" card surfaces text the parser didn't recognise as a SOAP section.
        remainder = compute_unparsed_remainder(soap, parsed)
        if remainder:
            with st.container(border=True):
                st.markdown("**OTHER**")
                st.markdown(remainder)

        # Action row: Edit · Copy to clipboard.
        cols = st.columns([1, 2, 5])
        with cols[0]:
            if st.button("Edit", key="edit_btn", use_container_width=True):
                # Populate edit buffers from current parsed soap.
                for name in SOAP_SECTIONS:
                    st.session_state[SECTION_KEY_MAP[name]] = parsed.get(name, "")
                st.session_state["is_editing"] = True
                st.rerun()
        with cols[1]:
            copy_to_clipboard_button(
                format_for_clipboard(parsed),
                key="copy_btn_read",
            )


def main() -> None:
    st.set_page_config(page_title="Medical Scribe — SOAP", layout="wide")
    init_state()
    # CSS for SOAP chips. Injected once per render so _render_section_header
    # can emit class-based markup at all three card render sites without
    # repeating the inline styles.
    st.markdown(_soap_chip_styles_html(), unsafe_allow_html=True)
    require_hf_token()

    # Eager model load — surface any error before the user uploads.
    # @st.cache_resource on _asr/_llm makes subsequent reruns instant.
    with st.spinner("Loading models (first run downloads ~14 GB; subsequent runs are instant)…"):
        try:
            asr_pipe = _asr()
        except Exception as exc:
            show_error("Failed to load MedASR", exc)
            st.stop()
        try:
            model, tokenizer = _llm()
        except Exception as exc:
            show_error("Failed to load MedGemma", exc)
            st.stop()

    _render_header()
    _render_sidebar()
    _render_tab_bar()

    if st.session_state["active_tab"] == "transcript":
        _render_transcript_tab(asr_pipe)
    else:
        _render_notes_tab(model, tokenizer)


if __name__ == "__main__":
    main()
