"""Streamlit UI for the MedASR -> MedGemma SOAP pipeline."""

from __future__ import annotations

import hashlib
import os
import sys
import traceback

from dotenv import load_dotenv

# Must run before any HF/MLX import — they read HF_TOKEN at import time.
load_dotenv()

import streamlit as st  # noqa: E402

from clinical_ai.asr import load_asr_pipeline, transcribe  # noqa: E402
from clinical_ai.device import pick_device  # noqa: E402
from clinical_ai.llm import load_medgemma, stream_soap  # noqa: E402

ASR_MODEL = "google/medasr"
LLM_MODEL = "mlx-community/medgemma-27b-text-it-4bit"


def init_state() -> None:
    defaults = {
        "audio_bytes": None,
        "audio_name": None,
        "audio_hash": None,
        "tx": None,
        "tx_edit": "",
        "soap": None,
        "soap_edit": "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def clear_downstream_state(after: str) -> None:
    """Enforce the spec's state invariants. `after` names the last valid stage."""
    if after == "audio":
        st.session_state["tx"] = None
        st.session_state["tx_edit"] = ""
        st.session_state["soap"] = None
        st.session_state["soap_edit"] = ""
    elif after == "tx":
        st.session_state["soap"] = None
        st.session_state["soap_edit"] = ""


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
    return load_medgemma(LLM_MODEL)


def show_error(label: str, exc: BaseException) -> None:
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    st.error(f"{label}: {type(exc).__name__}: {exc}")


def main() -> None:
    st.set_page_config(page_title="Clinical AI — SOAP", layout="wide")
    st.title("Clinical AI — visit transcription & SOAP draft")
    init_state()
    require_hf_token()

    # Eager model load so warmup happens once and any error surfaces before the user uploads.
    if not st.session_state.get("_models_loaded"):
        with st.spinner(
            "Loading models (first run downloads ~14 GB; subsequent runs are instant)…"
        ):
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
        st.session_state["_models_loaded"] = True
    else:
        try:
            asr_pipe = _asr()
            model, tokenizer = _llm()
        except Exception as exc:
            show_error("Failed to load models", exc)
            st.stop()

    # State A: audio upload
    upload = st.file_uploader(
        "Upload a patient visit recording",
        type=["wav", "mp3", "flac", "m4a"],
    )
    if upload is not None:
        incoming_bytes = upload.getvalue()
        incoming_hash = hashlib.sha256(incoming_bytes).hexdigest()
        is_new_upload = (
            upload.name != st.session_state["audio_name"]
            or incoming_hash != st.session_state["audio_hash"]
        )
        if is_new_upload:
            st.session_state["audio_bytes"] = incoming_bytes
            st.session_state["audio_name"] = upload.name
            st.session_state["audio_hash"] = incoming_hash
            clear_downstream_state(after="audio")

            # State B: transcribe
            with st.spinner("Transcribing audio…"):
                try:
                    text = transcribe(asr_pipe, st.session_state["audio_bytes"])
                except Exception as exc:
                    show_error("Could not transcribe audio", exc)
                    st.session_state["audio_bytes"] = None
                    st.session_state["audio_name"] = None
                    st.session_state["audio_hash"] = None
                    st.stop()
            st.session_state["tx"] = text
            st.session_state["tx_edit"] = text

    if st.session_state["tx"] is None:
        return  # Stay in State A.

    # State C: transcript ready
    st.subheader("Transcript")
    st.caption("Review the transcript before generating — fix any misheard terms.")
    st.text_area(
        "Transcript",
        key="tx_edit",
        height=300,
        label_visibility="collapsed",
    )

    if st.button("Generate SOAP note", type="primary"):
        if not st.session_state["tx_edit"].strip():
            st.warning(
                "Transcript is empty — please provide or correct the transcription "
                "before generating a SOAP note."
            )
            return
        # State D: streaming SOAP
        st.subheader("SOAP note")
        placeholder = st.empty()
        buf = ""
        meta: dict[str, object] = {}
        try:
            for chunk in stream_soap(model, tokenizer, st.session_state["tx_edit"], meta=meta):
                buf += chunk
                placeholder.markdown(buf)
        except Exception as exc:
            placeholder.empty()
            show_error("SOAP generation failed", exc)
            st.session_state["soap"] = None
            st.session_state["soap_edit"] = ""
            return
        st.session_state["soap"] = buf
        st.session_state["soap_edit"] = buf
        if meta.get("finish_reason") == "length":
            st.warning(
                "The SOAP note reached the output token limit and may be incomplete. "
                "Verify all four sections (Subjective, Objective, Assessment, Plan) "
                "are present before signing or downloading."
            )

    if st.session_state["soap"] is None:
        return  # Stay in State C.

    # State E: SOAP ready
    st.subheader("SOAP note (editable)")
    st.text_area(
        "SOAP note",
        key="soap_edit",
        height=500,
        label_visibility="collapsed",
    )
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download .md",
            data=st.session_state["soap_edit"],
            file_name="soap_note.md",
            mime="text/markdown",
        )
    with col2:
        if st.button("Start over"):
            st.session_state["audio_bytes"] = None
            st.session_state["audio_name"] = None
            st.session_state["audio_hash"] = None
            st.session_state["tx"] = None
            st.session_state["tx_edit"] = ""
            st.session_state["soap"] = None
            st.session_state["soap_edit"] = ""
            st.rerun()


if __name__ == "__main__":
    main()
