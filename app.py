"""Streamlit UI for the MedASR -> MedGemma SOAP pipeline."""

from __future__ import annotations

import hashlib
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
    load_asr_pipeline,
    load_medgemma,
    pick_device,
    stream_soap,
    transcribe,
)

ASR_MODEL = "google/medasr"
MAX_UPLOAD_MB = 100
INITIAL_STATE = {
    "audio_bytes": None,
    "audio_name": None,
    "audio_hash": None,
    "tx": None,
    "tx_edit": "",
    "soap": None,
    "soap_edit": "",
    "expanded_pane": None,
    "soap_truncated": False,
}


def init_state() -> None:
    for k, v in INITIAL_STATE.items():
        st.session_state.setdefault(k, v)


def reset_state() -> None:
    for k, v in INITIAL_STATE.items():
        st.session_state[k] = v


def clear_downstream_state(state: MutableMapping[str, object], after: str) -> None:
    """Enforce the spec's state invariants. `after` names the last valid stage.

    `expanded_pane` is intentionally not touched — it's a UI focus concern,
    orthogonal to workflow stage.
    """
    if after == "audio":
        state["tx"] = None
        state["tx_edit"] = ""
        state["soap"] = None
        state["soap_edit"] = ""
        state["soap_truncated"] = False
    elif after == "tx":
        state["soap"] = None
        state["soap_edit"] = ""
        state["soap_truncated"] = False


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

    State D's 'Generating SOAP…' is set inline by the streaming branch
    and is not derivable from state alone.
    """
    if state.get("audio_bytes") is None:
        return "No audio loaded"
    if state.get("tx") is None:
        return "Transcribing…"
    if state.get("soap") is None:
        return "Transcript ready"
    return "SOAP ready"


def primary_action_label(soap: object) -> str:
    """Right-pane primary button label.

    Truthy soap value -> regenerate; falsy -> generate (covers None and "").
    """
    return "Regenerate SOAP" if soap else "Generate SOAP note"


def update_truncation_flag(state: MutableMapping[str, object], meta: Mapping[str, object]) -> None:
    """Set state['soap_truncated'] based on streaming meta's finish_reason."""
    state["soap_truncated"] = meta.get("finish_reason") == "length"


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


def _render_header() -> None:
    """Top header bar: app title (left) + filename + stage label (right)."""
    cols = st.columns([3, 5])
    with cols[0]:
        st.markdown("### Medical Scribe")
    with cols[1]:
        stage_label = derive_stage_label(cast(Mapping[str, object], st.session_state))
        filename = st.session_state["audio_name"] or "no audio uploaded"
        st.markdown(
            f"<div style='text-align:right; padding-top:8px; color:#666'>"
            f"{filename} · <em>{stage_label}</em></div>",
            unsafe_allow_html=True,
        )
    st.divider()


def _handle_upload(upload) -> None:
    """Validate and persist a new audio upload into session state.

    Uses hash-based 'is this actually new?' detection so reruns triggered
    by other widgets don't re-transcribe the same audio.
    """
    incoming_bytes = upload.getvalue()
    if len(incoming_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        size_mb = len(incoming_bytes) // (1024 * 1024)
        st.error(
            f"File too large ({size_mb} MB). Maximum upload size is "
            f"{MAX_UPLOAD_MB} MB — please split long recordings."
        )
        st.stop()
    incoming_hash = hashlib.sha256(incoming_bytes).hexdigest()
    is_new_upload = (
        upload.name != st.session_state["audio_name"]
        or incoming_hash != st.session_state["audio_hash"]
    )
    if is_new_upload:
        st.session_state["audio_bytes"] = incoming_bytes
        st.session_state["audio_name"] = upload.name
        st.session_state["audio_hash"] = incoming_hash
        clear_downstream_state(cast(MutableMapping[str, object], st.session_state), after="audio")


def _render_left_pane(asr_pipe) -> None:
    """Left pane: audio uploader/player + transcript area, stage-aware."""
    audio_bytes = st.session_state["audio_bytes"]
    tx = st.session_state["tx"]
    is_streaming = bool(st.session_state.get("_streaming"))

    st.markdown("**Transcript**")

    # State A: no audio yet — uploader fills the pane.
    if audio_bytes is None:
        upload = st.file_uploader(
            "Upload a patient visit recording",
            type=["wav", "mp3", "flac", "m4a"],
        )
        if upload is not None:
            _handle_upload(upload)
            st.rerun()
        return

    # States B-E: audio player at top.
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
    st.text_area(
        "Transcript",
        key="tx_edit",
        height=400,
        label_visibility="collapsed",
        disabled=is_streaming,
    )


def _render_right_pane(model, tokenizer) -> None:
    """Right pane: SOAP placeholder/streaming/editable area, stage-aware."""
    audio_bytes = st.session_state["audio_bytes"]
    tx = st.session_state["tx"]
    soap = st.session_state["soap"]

    st.markdown("**SOAP note**")

    # State A: no audio yet.
    if audio_bytes is None:
        st.markdown("_Upload audio to begin._")
        return

    # State B: transcribing.
    if tx is None:
        st.markdown("_Awaiting transcript…_")
        return

    # Persistent truncation warning (visible whenever the flag is set).
    if st.session_state.get("soap_truncated", False):
        st.warning(
            "The SOAP note reached the output token limit and may be incomplete. "
            "Verify all four sections (Subjective, Objective, Assessment, Plan) "
            "are present before signing or downloading."
        )

    # State C: SOAP idle.
    if soap is None and not st.session_state.get("_streaming"):
        st.markdown("_Generate the SOAP note from the reviewed transcript._")
        if st.button(primary_action_label(soap), type="primary"):
            if not st.session_state["tx_edit"].strip():
                st.warning(
                    "Transcript is empty — please provide or correct the transcription "
                    "before generating a SOAP note."
                )
                return
            st.session_state["_streaming"] = True
            st.rerun()
        return

    # State D: streaming.
    if st.session_state.get("_streaming"):
        placeholder = st.empty()
        buf = ""
        meta: dict[str, object] = {}
        try:
            for chunk in stream_soap(
                model,
                tokenizer,
                st.session_state["tx_edit"],
                max_tokens=DEFAULT_MAX_TOKENS,
                meta=meta,
            ):
                buf += chunk
                placeholder.markdown(buf)
        except Exception as exc:
            placeholder.empty()
            show_error("SOAP generation failed", exc)
            st.session_state["soap"] = None
            st.session_state["soap_edit"] = ""
            st.session_state["_streaming"] = False
            return
        st.session_state["soap"] = buf
        st.session_state["soap_edit"] = buf
        update_truncation_flag(cast(MutableMapping[str, object], st.session_state), meta)
        st.session_state["_streaming"] = False
        st.rerun()

    # State E: SOAP ready, editable.
    st.text_area(
        "SOAP note",
        key="soap_edit",
        height=400,
        label_visibility="collapsed",
    )
    action_cols = st.columns([2, 2, 1])
    with action_cols[0]:
        if st.button(primary_action_label(soap), type="primary", key="regen_btn"):
            st.session_state["_streaming"] = True
            st.rerun()
    with action_cols[1]:
        st.download_button(
            "Download .md",
            data=st.session_state["soap_edit"],
            file_name="soap_note.md",
            mime="text/markdown",
        )
    with action_cols[2]:
        if st.button("Start over", key="reset_btn"):
            reset_state()
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Medical Scribe — SOAP", layout="wide")
    init_state()
    require_hf_token()

    # Eager model load so warmup happens once and any error surfaces before the user uploads.
    # `@st.cache_resource` on _asr/_llm already makes subsequent reruns instant.
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

    cols = st.columns([1, 1])
    with cols[0]:
        _render_left_pane(asr_pipe)
    with cols[1]:
        _render_right_pane(model, tokenizer)


if __name__ == "__main__":
    main()
