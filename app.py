"""Streamlit UI for the MedASR -> MedGemma SOAP pipeline.

Persistent split view (transcript left, SOAP right) from State C
onward. Live capture via st.audio_input; file upload as a secondary
affordance. SOAP cards are always-editable post-stream. Copy to
clipboard is the only export — nothing is written to disk."""

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

from medical_scribe import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_ID,
    SECTION_HEADER_RE,
    SOAP_SECTIONS,
    TrimResult,
    format_for_clipboard,
    load_asr_pipeline,
    load_medgemma,
    load_vad,
    parse_soap_sections,
    pick_device,
    stream_soap,
    transcribe,
    trim_silence,
)

ASR_MODEL = "google/medasr"
MAX_UPLOAD_MB = 100

# Derived from SOAP_SECTIONS so adding a section in soap_sections.py
# auto-propagates the per-section edit-buffer key. The matching INITIAL_STATE
# entries below still need a manual addition; tests/test_app.py's
# test_initial_state_keys_match_section_key_map catches that drift.
SECTION_KEY_MAP: dict[str, str] = {name: f"{name.lower()}_edit" for name in SOAP_SECTIONS}

# Per-section edit-mode flag keys. Toggled by `toggle_section_edit`. Read
# by `_render_section_card` to branch read-mode vs edit-mode rendering.
# Drift guard: tests/test_app.py::test_section_editing_key_map_covers_all_soap_sections.
SECTION_EDITING_KEY_MAP: dict[str, str] = {
    name: f"{name.lower()}_editing" for name in SOAP_SECTIONS
}

# Per-section edit-cancel snapshot keys. Populated when entering edit mode;
# cleared on Save (commit) or Cancel (revert + clear).
SECTION_SNAPSHOT_KEY_MAP: dict[str, str] = {
    name: f"{name.lower()}_edit_snapshot" for name in SOAP_SECTIONS
}

# SOAP cards combine color + letter chip; color alone fails for ~8% of
# male readers. White-on-color chip contrast must clear WCAG AA (4.5:1).
# All four chips use the -700 family for uniform visual depth and AA-grade
# contrast. Subjective uses indigo (not blue) so it stays distinguishable
# from emerald Objective under deuteranopia (most common color-blindness).
# Drift guards: tests/test_app.py::test_section_colors_have_no_duplicates,
# test_section_colors_are_all_aa_grade, and
# test_section_color_and_initial_maps_cover_all_soap_sections.
SECTION_COLORS: dict[str, str] = {
    "Subjective": "#4338ca",  # indigo-700
    "Objective": "#047857",  # emerald-700
    "Assessment": "#b45309",  # amber-700
    "Plan": "#6d28d9",  # violet-700
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
    # VAD trim metadata. Populated alongside `tx` in State B's _render_transcript_pane;
    # consumed by format_trim_caption to render the status line above the transcript.
    "tx_trim": None,
    # SOAP — full markdown blob, source of truth
    "soap": None,
    "soap_truncated": False,
    # Per-section edit buffers (now the canonical SOAP body post-stream;
    # populated from the streaming buffer via populate_section_edit_buffers).
    "subjective_edit": "",
    "objective_edit": "",
    "assessment_edit": "",
    "plan_edit": "",
    # Per-section edit-mode flags (False = read mode, True = edit mode).
    # Driven by the pencil-icon click in _render_section_card.
    "subjective_editing": False,
    "objective_editing": False,
    "assessment_editing": False,
    "plan_editing": False,
    # Per-section snapshots for cancel-revert. Populated on edit-enter;
    # cleared on Save or Cancel.
    "subjective_edit_snapshot": None,
    "objective_edit_snapshot": None,
    "assessment_edit_snapshot": None,
    "plan_edit_snapshot": None,
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

    Stage-orthogonal flags (e.g., `_streaming`, `_show_reset_dialog`)
    survive both clear paths — they don't pertain to the audio/transcript/
    SOAP pipeline. `tx_trim` survives the `after="tx"` path: editing the
    transcript doesn't change what VAD saw."""
    if after == "audio":
        state["tx"] = None
        state["tx_edit"] = ""
        state["tx_trim"] = None
        state["soap"] = None
        state["soap_truncated"] = False
        for key in SECTION_KEY_MAP.values():
            state[key] = ""
        for key in SECTION_EDITING_KEY_MAP.values():
            state[key] = False
        for key in SECTION_SNAPSHOT_KEY_MAP.values():
            state[key] = None
    elif after == "tx":
        state["soap"] = None
        state["soap_truncated"] = False
        for key in SECTION_KEY_MAP.values():
            state[key] = ""
        for key in SECTION_EDITING_KEY_MAP.values():
            state[key] = False
        for key in SECTION_SNAPSHOT_KEY_MAP.values():
            state[key] = None


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


# Maps every label `derive_stage_label` can return to "static" (idle states
# A, C, E) or "active" (system-working states B, D). _stage_chip_html
# selects the visual variant from this map. Drift guard:
# tests/test_app.py::test_stage_chip_variants_cover_every_derive_stage_label_output.
STAGE_CHIP_VARIANTS: dict[str, str] = {
    "No audio loaded": "static",
    "Transcribing…": "active",
    "Transcript ready": "static",
    "Generating SOAP…": "active",
    "SOAP ready": "static",
}


def _stage_chip_html(label: str) -> str:
    """Return the stage chip span — variant chosen by STAGE_CHIP_VARIANTS,
    label HTML-escaped (defensive against future dynamic labels)."""
    variant = STAGE_CHIP_VARIANTS.get(label, "static")
    safe_label = html.escape(label)
    return (
        f'<span class="ms-stage-chip ms-stage-{variant}">'
        f'<span class="ms-dot"></span>{safe_label}'
        f"</span>"
    )


def _topbar_html(*, stage_label: str, meta: str) -> str:
    """Title + stage chip + (optional) right-aligned meta.

    Renders inline; the New session button is rendered separately by
    `_render_topbar` as a real `st.button` in the right column of an
    `st.columns([8, 2])` split, so the click handler can drive the
    `_show_reset_dialog` flag.
    """
    chip_html = _stage_chip_html(stage_label)
    parts = [
        '<div class="ms-topbar">',
        '<span class="ms-topbar-title">Medical Scribe</span>',
        chip_html,
    ]
    if meta:
        parts.append(f'<span class="ms-topbar-meta">{html.escape(meta)}</span>')
    parts.append("</div>")
    return "".join(parts)


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


def _format_duration(seconds: float) -> str:
    """Format a non-negative duration in seconds as 'Xm Ys' (under 1 hour)
    or 'Xh Ym' (1 hour or longer)."""
    total = round(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m {s}s"


def format_trim_caption(
    result: TrimResult | None,
    *,
    min_ratio: float = 0.05,
) -> str | None:
    """Caption above the transcript text area. Returns None when nothing
    should render (no result yet, or trim ratio below the noise floor).

    Three rendered branches:
      - status="trimmed", ratio >= min_ratio:
          "Trimmed 4m 12s of silence (47% of recording)."
      - status="no_speech":
          "Couldn't detect speech; transcribing the full recording instead."
      - status="error":
          "Couldn't trim silence; transcribing the full recording instead."
    """
    if result is None:
        return None
    if result.status == "no_speech":
        return "Couldn't detect speech; transcribing the full recording instead."
    if result.status == "error":
        return "Couldn't trim silence; transcribing the full recording instead."
    # status == "trimmed"
    if result.original_seconds <= 0:
        return None
    ratio = 1.0 - (result.trimmed_seconds / result.original_seconds)
    if ratio < min_ratio:
        return None
    removed = result.original_seconds - result.trimmed_seconds
    return f"Trimmed {_format_duration(removed)} of silence ({round(ratio * 100)}% of recording)."


def _format_session_meta(state: Mapping[str, object]) -> str:
    """Right-aligned mono-text for the top bar.

    - State A: empty string (no chrome metadata before audio is loaded).
    - State B (audio present, no transcript): "recording · {filename}".
    - State C onward: "session · {Xm Ys} · trimmed {N}%" if VAD trimmed
      anything, otherwise "session · {Xm Ys}".

    The trim-percentage branch is suppressed in three degenerate cases:
    `trimmed_seconds <= 0` (VAD error path returns 0 — claiming "100%
    trimmed" would be misleading), `trimmed_seconds >= original_seconds`
    (VAD no_speech path or no-op trim), and `original_seconds <= 0`
    (defensive — should never happen for a successfully decoded clip).
    """
    if state.get("audio_bytes") is None:
        return ""
    audio_name = state.get("audio_name") or "audio"
    if state.get("tx") is None:
        return f"recording · {audio_name}"
    trim = state.get("tx_trim")
    if trim is None:
        return f"session · {audio_name}"
    original = getattr(trim, "original_seconds", 0)
    trimmed = getattr(trim, "trimmed_seconds", 0)
    if original <= 0:
        return f"session · {audio_name}"
    duration = _format_duration(original)
    if trimmed <= 0 or trimmed >= original:
        return f"session · {duration}"
    pct = round((1.0 - trimmed / original) * 100)
    return f"session · {duration} · trimmed {pct}%"


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


def compute_section_states(
    buf: str,
    section_order: list[str],
) -> list[tuple[str, str, str]]:
    """Drive the State-D skeleton-card streaming render.

    Returns one (name, status, body) tuple per section in `section_order`.
    `status` is one of "pending", "active", "completed". A section is
    `completed` when a later section's header has appeared in `buf`;
    the most-recently-seen section is `active`; sections not yet seen are
    `pending` with body=''.

    Order in the output always matches `section_order` (not buffer order),
    so skeleton-card placement in the right pane stays stable even if the
    model emits sections out of canonical order.
    """
    parsed = parse_soap_sections(buf)
    seen = [name for name in section_order if name in parsed]
    last_seen = seen[-1] if seen else None

    out: list[tuple[str, str, str]] = []
    for name in section_order:
        if name not in parsed:
            out.append((name, "pending", ""))
        elif name == last_seen:
            out.append((name, "active", parsed[name]))
        else:
            out.append((name, "completed", parsed[name]))
    return out


def escape_text_for_inline_script(text: str) -> str:
    """JSON-encode `text` for safe inline JavaScript, plus replace `</`
    with `<\\/` so a `</script>` substring inside `text` cannot prematurely
    close an enclosing `<script>` tag.

    Used by `copy_to_clipboard_button` to embed user-controlled SOAP content
    in a `<script>` block via `st.iframe`.
    """
    return json.dumps(text).replace("</", "<\\/")


def copy_to_clipboard_button(text: str, *, label: str = "Copy to clipboard", key: str) -> None:
    """Render a Copy-to-clipboard button using the JavaScript Clipboard API.

    Uses `st.iframe` rather than `st.html` because the latter
    strips/sanitizes inline event handlers in current Streamlit versions,
    causing silent click failures. The iframe runs JavaScript reliably
    and Streamlit grants it `clipboard-write` permission by default.

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
    # f-string starts with `<` so `st.iframe`'s input-type heuristic
    # (Path → URL → existing file → /-prefixed → else srcdoc) routes
    # unambiguously to srcdoc. Don't reintroduce a leading newline or
    # whitespace.
    st.iframe(
        f"""<button id="{key}" style="{btn_style}">{safe_label}</button>
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


@st.cache_resource
def _vad():
    return load_vad()


def show_error(label: str, exc: BaseException) -> None:
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    st.error(f"{label}: {type(exc).__name__}: {exc}")


def _design_tokens_css() -> str:
    """Compact Modern design tokens — typography, color, spacing, chip and
    chrome styles, animations. Injected once per page render in main().

    Combines what the previous `_soap_chip_styles_html` covered (SOAP card
    chips + section headers) with the rest of the token system (top bar,
    stage chips, skeleton-card shimmer, streaming cursor). Per-section
    background colors are read from SECTION_COLORS so adding a section in
    soap_sections.py auto-propagates the chip rule. The drift guards in
    tests/test_app.py catch SECTION_COLORS / SOAP_SECTIONS divergence.
    """
    soap_chip_rules = "\n".join(
        f".soap-chip-{name.lower()} {{ background: {color}; }}"
        for name, color in SECTION_COLORS.items()
    )
    return f"""<style>
:root {{
  --color-surface: #ffffff;
  --color-surface-2: #fafafa;
  --color-canvas: #f9fafb;
  --color-border: #e5e7eb;
  --color-text: #111111;
  --color-text-muted: #6b7280;
  --color-text-subtle: #9ca3af;

  --s-1: 4px;
  --s-2: 8px;
  --s-3: 12px;
  --s-4: 16px;
  --s-6: 24px;

  --font-mono: "SF Mono", "Menlo", monospace;
}}

/* Top bar */
.ms-topbar {{
  display: flex;
  align-items: center;
  gap: var(--s-3);
  padding: var(--s-2) var(--s-4);
  background: var(--color-surface-2);
  border-bottom: 1px solid var(--color-border);
  font-size: 13px;
}}
.ms-topbar-title {{ font-weight: 600; color: var(--color-text); }}
.ms-topbar-meta {{
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--color-text-subtle);
  margin-left: auto;
}}

/* Stage chips — monochrome with active-state pulse */
.ms-stage-chip {{
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  border: 1px solid;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}
.ms-stage-static {{
  background: #f3f4f6;
  color: var(--color-text-muted);
  border-color: var(--color-border);
}}
.ms-stage-active {{
  background: var(--color-text);
  color: var(--color-surface);
  border-color: var(--color-text);
}}
.ms-stage-static .ms-dot {{
  width: 6px; height: 6px; border-radius: 50%;
  border: 1px solid var(--color-text-subtle);
}}
.ms-stage-active .ms-dot {{
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--color-surface);
  animation: ms-pulse 1.5s ease-in-out infinite;
}}
@keyframes ms-pulse {{
  0%, 100% {{ opacity: 1; transform: scale(1); }}
  50%      {{ opacity: 0.4; transform: scale(0.85); }}
}}

/* Skeleton lines (state D streaming placeholders) */
.ms-skel-line {{
  background: linear-gradient(90deg, #f3f4f6 0%, #e5e7eb 50%, #f3f4f6 100%);
  background-size: 200% 100%;
  animation: ms-shimmer 1.5s linear infinite;
  height: 10px;
  border-radius: 2px;
  margin-bottom: 6px;
}}
@keyframes ms-shimmer {{
  0%   {{ background-position: 100% 0; }}
  100% {{ background-position: -100% 0; }}
}}

/* Streaming text cursor */
.ms-streaming-cursor {{
  display: inline-block;
  width: 1px; height: 13px;
  background: var(--color-text);
  animation: ms-cursor-blink 1s infinite;
  vertical-align: middle;
  margin-left: 1px;
}}
@keyframes ms-cursor-blink {{
  50% {{ opacity: 0; }}
}}

/* SOAP section header (chip + name) — shared by streaming + post-stream */
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
  width: 22px;
  height: 22px;
  border-radius: 4px;
  color: white;
  font-weight: 700;
  font-size: 12px;
  flex-shrink: 0;
}}
.soap-section-name {{
  font-weight: 600;
  letter-spacing: 0.04em;
  color: var(--color-text);
}}
{soap_chip_rules}

/* State A — chooser cards */
.ms-chooser-card {{
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: var(--s-6) var(--s-4);
  text-align: center;
  background: var(--color-surface);
  transition: border-color 0.15s;
}}
.ms-chooser-card:hover {{ border-color: var(--color-text); }}
.ms-chooser-title {{
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text);
  margin-bottom: 4px;
}}
.ms-chooser-caption {{
  font-size: 12px;
  color: var(--color-text-muted);
  line-height: 1.5;
  margin-bottom: 12px;
}}
.ms-mic-circle {{
  background: var(--color-text);
  color: var(--color-surface);
  width: 56px;
  height: 56px;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  margin: 4px auto 0;
}}

/* Streamlit file uploader dropzone — dashed-border treatment */
[data-testid="stFileUploaderDropzone"] {{
  border: 2px dashed #d1d5db !important;
  background: var(--color-surface) !important;
}}
</style>"""


def _render_section_header(name: str, *, muted: bool = False) -> None:
    """Colored letter-chip + section name. Reused by both card render
    paths (streaming markdown during generation, always-editable text_areas
    post-stream). Visual styles live in _design_tokens_css(), injected
    once per page render via main().

    `muted=True` (used by the streaming-pane skeleton state) renders the
    chip at 0.4 opacity and the section name in --color-text-subtle, so
    pending cards visually recede vs active/completed cards.
    """
    initial = html.escape(SECTION_INITIALS[name])
    label = html.escape(name.upper())
    chip_opacity = "0.4" if muted else "1.0"
    label_color = "var(--color-text-subtle)" if muted else "var(--color-text)"
    st.markdown(
        f'<div class="soap-section-header">'
        f'<span class="soap-chip soap-chip-{name.lower()}" '
        f'style="opacity:{chip_opacity}">{initial}</span>'
        f'<span class="soap-section-name" style="color:{label_color}">{label}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_topbar() -> None:
    """Top bar — replaces both the previous header and sidebar.

    The title + stage chip + meta block is rendered as a single markdown
    HTML span (left column of an 8/2 split). The `+ New session` button is
    a real st.button in the right column so the click drives the
    `_show_reset_dialog` flag and confirm dialog (unchanged behavior; only
    the placement moved).
    """
    state = cast(Mapping[str, object], st.session_state)
    stage_label = derive_stage_label(state)
    meta = _format_session_meta(state)

    cols = st.columns([8, 2])
    with cols[0]:
        st.markdown(
            _topbar_html(stage_label=stage_label, meta=meta),
            unsafe_allow_html=True,
        )
    with cols[1]:
        if st.button(
            "+ New session",
            key="new_session_btn",
            use_container_width=True,
        ):
            if st.session_state.get("audio_bytes") is None:
                reset_state()
                st.rerun()
            else:
                st.session_state["_show_reset_dialog"] = True
                st.rerun()

    if st.session_state.get("_show_reset_dialog"):
        _confirm_new_session_dialog()


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
    """State A landing — Record / Upload affordances side by side.

    Each side is a tightly-styled card (.ms-chooser-card) with a heading,
    caption, and the actual Streamlit input widget below. The audio_input
    widget gets a decorative 56px black mic affordance above it; the
    file_uploader's native dropzone is styled into a dashed-border drop
    zone via global CSS targeting [data-testid='stFileUploaderDropzone'].

    Caller is responsible for the gating; this function does not check
    `audio_bytes is None`.
    """
    cols = st.columns([1, 1])
    with cols[0]:
        st.markdown(
            '<div class="ms-chooser-card">'
            '<div class="ms-chooser-title">Record this visit</div>'
            '<div class="ms-chooser-caption">'
            "Click the mic to start, click again to stop. Audio stays on this device."
            "</div>"
            '<div class="ms-mic-circle">●</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        recorded = st.audio_input(
            "Record audio",
            label_visibility="collapsed",
            key="audio_input_widget",
        )
        if recorded is not None and _handle_upload(recorded):
            st.rerun()
    with cols[1]:
        st.markdown(
            '<div class="ms-chooser-card">'
            '<div class="ms-chooser-title">Upload a recording</div>'
            '<div class="ms-chooser-caption">'
            "WAV, MP3, FLAC, or M4A. Max 100 MB. Drag-and-drop supported."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Audio file",
            type=["wav", "mp3", "flac", "m4a"],
            label_visibility="collapsed",
            key="file_uploader_widget",
        )
        if uploaded is not None and _handle_upload(uploaded):
            st.rerun()


def _render_transcript_pane(asr_pipe, vad_model) -> None:
    """Left pane of State C split view: audio player + editable transcript +
    Generate / Regenerate button.

    Caller dispatches to this only when `audio_bytes is not None` (States B
    and onward). State B (transcribing) renders inside this pane via the
    spinner branch; States C / streaming / SOAP-ready all render the
    audio + textarea + button. The textarea is `disabled=is_streaming` to
    prevent a re-run race during generation.

    Uses `value=` + manual `st.session_state` sync (not `key=`) per the
    CLAUDE.md invariant: text_areas in conditionally-rendered branches lose
    widget-managed values when their parent unmounts."""
    audio_bytes = st.session_state["audio_bytes"]
    tx = st.session_state["tx"]
    is_streaming = bool(st.session_state.get("_streaming"))

    # Audio player at top.
    mime = audio_mime_from_name(st.session_state["audio_name"])
    st.audio(audio_bytes, format=mime if mime is not None else "audio/wav")

    # Trim status caption. format_trim_caption returns None when there's
    # nothing to render (no result, or trim < 5%) so the layout stays clean.
    caption = format_trim_caption(st.session_state.get("tx_trim"))
    if caption is not None:
        st.caption(caption)

    # State B: transcribing. trim_silence runs first to drop non-speech
    # segments before MedASR sees the audio. trim_silence never raises
    # (medical_scribe.vad invariant); failures return status="error" and
    # the original bytes pass through unchanged.
    #
    # Phased progress: st.status containers expose the two-step pipeline
    # (trim → transcribe). The trim phase reports concrete numbers from
    # the TrimResult; the transcribe phase reports model + audio length.
    if tx is None:
        with st.status("Preparing transcription…", expanded=True) as status:
            status.update(label="Trimming silence…", state="running")
            trim_result = trim_silence(audio_bytes, vad_model)
            if trim_result.status == "trimmed" and trim_result.original_seconds > 0:
                pct = round(
                    (1.0 - trim_result.trimmed_seconds / trim_result.original_seconds) * 100
                )
                removed = trim_result.original_seconds - trim_result.trimmed_seconds
                st.write(f"✓ Trimmed {_format_duration(removed)} of silence ({pct}%)")
            elif trim_result.status == "no_speech":
                st.write("⚠ No speech detected — transcribing the full recording")
            elif trim_result.status == "error":
                st.write("⚠ Couldn't trim silence — transcribing the full recording")

            status.update(label="Transcribing audio…", state="running")
            try:
                text = transcribe(asr_pipe, trim_result.audio_bytes)
            except Exception as exc:
                status.update(label="Transcription failed", state="error")
                show_error("Could not transcribe audio", exc)
                reset_state()
                st.stop()
            status.update(label="Transcription complete", state="complete")

        st.session_state["tx_trim"] = trim_result
        st.session_state["tx"] = text
        st.session_state["tx_edit"] = text
        st.rerun()

    # State C onward: editable transcript (disabled during streaming).
    new_tx_edit = st.text_area(
        "Transcript",
        value=st.session_state["tx_edit"],
        height=400,
        label_visibility="collapsed",
        disabled=is_streaming,
    )
    st.session_state["tx_edit"] = new_tx_edit

    # Generate / Regenerate button (idempotent — re-clicking discards section
    # edits and re-runs against the current transcript).
    cols = st.columns([4, 2])
    with cols[1]:
        if st.button(
            primary_action_label(st.session_state["soap"]),
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
            # New generation: discard any in-progress section edits.
            st.session_state["soap_truncated"] = False
            for key in SECTION_KEY_MAP.values():
                st.session_state[key] = ""
            st.session_state["_streaming"] = True
            st.rerun()


def _render_streaming_card(name: str, status: str, body: str) -> None:
    """Render one streaming SOAP card based on its status.

    `pending` → muted chip + 3 shimmering skeleton lines.
    `active` → body markdown + a blinking cursor span.
    `completed` → plain body markdown.

    Reuses `_render_section_header` (with `muted=True` for pending) so
    both the streaming and post-stream paths share a single chip+name
    helper, per the CLAUDE.md invariant.
    """
    with st.container(border=True):
        _render_section_header(name, muted=(status == "pending"))
        if status == "pending":
            st.markdown(
                '<div style="padding-left:30px;">'
                '<div class="ms-skel-line" style="width:80%"></div>'
                '<div class="ms-skel-line" style="width:60%"></div>'
                '<div class="ms-skel-line" style="width:70%"></div>'
                "</div>",
                unsafe_allow_html=True,
            )
        elif status == "active":
            st.markdown(body)
            st.markdown('<span class="ms-streaming-cursor"></span>', unsafe_allow_html=True)
        else:  # completed
            st.markdown(body)


def _render_soap_pane(model, tokenizer) -> None:
    """Right pane of State C split view: streaming SOAP cards while
    `_streaming` is True; always-editable cards once a SOAP draft exists;
    placeholder copy when transcript is ready but Generate has not been
    clicked yet.

    The four `*_edit` session-state buffers are the canonical SOAP body
    post-stream. They are populated once via `populate_section_edit_buffers`
    on stream completion; from that point on they're the source of truth
    for the Copy button (`format_for_clipboard`) and survive across reruns.

    Uses `value=` + manual `st.session_state` sync for each card body
    text_area (not `key=`) per the CLAUDE.md invariant — same reasoning as
    the transcript pane."""
    tx = st.session_state["tx"]
    soap = st.session_state["soap"]
    is_streaming = bool(st.session_state.get("_streaming"))

    # State B: transcribing. Right pane renders BEFORE the transcript pane
    # (see _render_split_view's column order) so this placeholder is visible
    # while the synchronous transcribe call blocks the left pane's spinner.
    if tx is None:
        st.markdown("_Awaiting transcript…_")
        return

    # State C (transcript ready, no SOAP, not streaming): "what to expect"
    # placeholder. Names the model, rough token budget, and typical
    # wall-clock so the clinician sees the work that's about to happen.
    if soap is None and not is_streaming:
        st.markdown(
            "<div style='text-align:center; padding-top:48px; "
            "color: var(--color-text-muted); font-size:13px;'>"
            "<div style='color: var(--color-text); font-weight:500; margin-bottom:6px;'>"
            "Ready to draft a SOAP note"
            "</div>"
            "<div style='color: var(--color-text-subtle); line-height:1.6;'>"
            # "MedGemma-27B (4-bit MLX)" is hand-typed; sync with DEFAULT_MODEL_ID
            # in medical_scribe/llm.py if that ever changes.
            f"MedGemma-27B (4-bit MLX), ~{DEFAULT_MAX_TOKENS} tokens<br>"
            "Streamed live; typically 20–40 seconds."  # noqa: RUF001
            "</div>"
            "<div style='margin-top:16px; color: var(--color-text-muted);'>"
            "Click <b>Generate SOAP note</b> on the left to start."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # State D: streaming with per-card skeleton placeholders.
    #
    # Render all four cards from t=0 with their own st.empty() placeholders.
    # On each chunk, compute_section_states yields the new per-card states;
    # we update only the placeholders whose (name, status, body) tuple
    # actually changed. Pending/completed cards stay stable across chunks
    # (no DOM churn); only the active card re-renders as its body grows.
    if is_streaming:
        status_placeholder = st.empty()
        status_placeholder.markdown(f"_{streaming_status_label([])}_")

        # Pre-create one placeholder per card so we can update them
        # independently inside the streaming loop.
        card_placeholders: list = []
        for _ in SOAP_SECTIONS:
            card_placeholders.append(st.empty())

        # Initial render: every card pending.
        initial_states: list[tuple[str, str, str]] = [
            (name, "pending", "") for name in SOAP_SECTIONS
        ]
        for placeholder, (name, status, body) in zip(
            card_placeholders, initial_states, strict=True
        ):
            with placeholder.container():
                _render_streaming_card(name, status, body)
        last_card_states = list(initial_states)

        buf = ""
        meta: dict[str, object] = {}
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
                section_states = compute_section_states(buf, list(SOAP_SECTIONS))
                seen_names = [name for name, status, _ in section_states if status != "pending"]

                status_text = streaming_status_label(seen_names)
                if status_text != last_status:
                    status_placeholder.markdown(f"_{status_text}_")
                    last_status = status_text

                # Per-card throttle: re-render only the cards whose
                # (name, status, body) actually changed since last chunk.
                for i, (state, last_state) in enumerate(
                    zip(section_states, last_card_states, strict=True)
                ):
                    if state != last_state:
                        with card_placeholders[i].container():
                            _render_streaming_card(*state)
                        last_card_states[i] = state
        except Exception as exc:
            for placeholder in card_placeholders:
                placeholder.empty()
            status_placeholder.empty()
            show_error("SOAP generation failed", exc)
            st.session_state["soap"] = None
            st.session_state["soap_truncated"] = False
            st.session_state["_streaming"] = False
            return

        # Stream completed. Persist, populate edit buffers, flip flag, rerun.
        st.session_state["soap"] = buf
        update_truncation_flag(cast(MutableMapping[str, object], st.session_state), meta)
        populate_section_edit_buffers(cast(MutableMapping[str, object], st.session_state), buf)
        st.session_state["_streaming"] = False
        st.rerun()

    # States E (merged): SOAP exists, cards always editable.
    parsed = parse_soap_sections(soap or "")

    # Persistent truncation warning.
    if st.session_state.get("soap_truncated", False):
        st.warning(
            "The SOAP note reached the output token limit and may be incomplete. "
            "Verify all four sections (Subjective, Objective, Assessment, Plan) "
            "are present before copying."
        )

    # Defensive parse-failure warning.
    expected = set(SOAP_SECTIONS)
    if expected - set(parsed):
        st.warning(
            "The SOAP note couldn't be fully parsed into sections — "
            "review carefully before copying."
        )

    # Always-editable cards. value= + manual sync (CLAUDE.md invariant).
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

    # OTHER card surfaces text the parser didn't recognise as a SOAP section.
    remainder = compute_unparsed_remainder(soap, parsed)
    if remainder:
        with st.container(border=True):
            st.markdown("**OTHER**")
            st.markdown(remainder)

    # Copy button reads the current edit buffers (the post-stream canonical
    # source of truth).
    edits = {name: st.session_state[SECTION_KEY_MAP[name]] for name in SOAP_SECTIONS}
    cols = st.columns([4, 2])
    with cols[1]:
        copy_to_clipboard_button(
            format_for_clipboard(edits),
            key="copy_btn",
        )


def _render_split_view(asr_pipe, vad_model, model, tokenizer) -> None:
    """States B and beyond: persistent vertical split (transcript left,
    SOAP right). Each pane handles its own state-aware sub-render.

    The SOAP pane is rendered before the transcript pane because State B's
    transcript branch blocks on a synchronous `transcribe()` call inside
    a spinner. Rendering SOAP first means the right column shows its
    "Awaiting transcript…" placeholder during transcription instead of
    going blank. Streamlit positions columns by `cols[i]` index, not by
    `with`-block order, so the visual layout is unaffected."""
    cols = st.columns([1, 1])
    with cols[1]:
        _render_soap_pane(model, tokenizer)
    with cols[0]:
        _render_transcript_pane(asr_pipe, vad_model)


def main() -> None:
    st.set_page_config(page_title="Medical Scribe — SOAP", layout="wide")
    init_state()
    # Compact Modern design tokens — typography, color, chrome styles,
    # animations. Injected once per page render so every helper that emits
    # class-based markup (top bar, stage chip, SOAP cards, streaming
    # placeholders) finds its styles already declared.
    st.markdown(_design_tokens_css(), unsafe_allow_html=True)
    require_hf_token()

    # Eager model load — surface any error before the user uploads.
    # @st.cache_resource on _asr/_llm/_vad makes subsequent reruns instant.
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
        try:
            vad_model = _vad()
        except Exception as exc:
            show_error("Failed to load Silero VAD", exc)
            st.stop()

    _render_topbar()

    # Dispatch by state. A → chooser; B/C → split view (transcript pane
    # handles the spinner-during-transcribe sub-state internally).
    if st.session_state["audio_bytes"] is None:
        _render_state_a_chooser()
    else:
        _render_split_view(asr_pipe, vad_model, model, tokenizer)


if __name__ == "__main__":
    main()
