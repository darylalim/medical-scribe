# Streamlit SOAP Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-file `transcribe.py` CLI with a `clinical_ai/` package and a Streamlit UI that chains MedASR transcription with MedGemma-27B-text-it-4bit (MLX) SOAP-note generation, including an editable transcript review step.

**Architecture:** A `clinical_ai/` Python package exposes four small modules (`device`, `asr`, `llm`, `prompts`) with no Streamlit knowledge. A new `app.py` is the only Streamlit entry point and uses `@st.cache_resource` to load both models exactly once. The existing `transcribe.py` is reduced to a thin CLI shim that imports from the new package and continues to support `--sample` and `--device auto`.

**Tech Stack:** Python 3.10+, `transformers` (MedASR on MPS), `mlx-lm` (MedGemma 4-bit), `streamlit`, `python-dotenv`, `pytest` + `pytest-mock` for tests, `ruff` for lint/format, `ty` for typing, `uv` for env management.

**Spec:** `docs/superpowers/specs/2026-04-18-streamlit-soap-pipeline-design.md`

---

## File Structure

After this plan executes, the repository will look like:

```
clinical_ai/
  __init__.py        # empty package marker
  device.py          # pick_device()
  asr.py             # load_asr_pipeline(), transcribe()
  llm.py             # load_medgemma(), stream_soap()
  prompts.py         # SOAP_SYSTEM_PROMPT, format_soap_messages()
app.py               # Streamlit UI (only file that imports streamlit)
transcribe.py        # thin CLI shim — imports from clinical_ai
pyproject.toml       # adds streamlit + mlx-lm; coverage source switches to clinical_ai
tests/
  test_device.py
  test_asr.py
  test_llm.py
  test_prompts.py
  test_integration.py    # placeholder, gated by @pytest.mark.integration
docs/superpowers/
  specs/2026-04-18-streamlit-soap-pipeline-design.md
  plans/2026-04-18-streamlit-soap-pipeline.md
```

The existing `tests/test_transcribe.py` is removed (its three tests move to `test_device.py` with updated patch targets).

---

## Task 1: Add Streamlit and mlx-lm dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `streamlit` and `mlx-lm` to project dependencies**

Edit `pyproject.toml`. In the `[project]` table's `dependencies` array, add two new lines after `python-dotenv`:

```toml
dependencies = [
    "transformers @ git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5",
    "torch>=2.1.0",
    "librosa>=0.10.0",
    "soundfile>=0.12.0",
    "huggingface-hub>=0.24.0",
    "numpy>=1.24.0",
    "accelerate>=0.30.0",
    "python-dotenv>=1.0.0",
    "streamlit>=1.32",
    "mlx-lm>=0.20",
]
```

- [ ] **Step 2: Resolve and install the new dependencies**

Run: `uv sync`

Expected: `uv` updates `uv.lock` and installs `streamlit`, `mlx-lm`, and `mlx`. No errors. (If you're not on Apple Silicon, `mlx` will fail to install — this project targets Apple Silicon by design.)

- [ ] **Step 3: Sanity-check the new packages import**

Run: `uv run python -c "import streamlit, mlx_lm; print(streamlit.__version__, mlx_lm.__version__)"`

Expected: two version numbers print, no traceback.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add streamlit and mlx-lm dependencies"
```

---

## Task 2: Create `clinical_ai` package and move `pick_device`

**Files:**
- Create: `clinical_ai/__init__.py`
- Create: `clinical_ai/device.py`
- Create: `tests/test_device.py`
- Modify: `transcribe.py` (import `pick_device` from new location)
- Delete: `tests/test_transcribe.py`

- [ ] **Step 1: Write the failing tests for `pick_device`**

Create `tests/test_device.py`:

```python
from clinical_ai.device import pick_device


def test_pick_device_prefers_cuda(mocker):
    mocker.patch("clinical_ai.device.torch.cuda.is_available", return_value=True)
    mocker.patch("clinical_ai.device.torch.backends.mps.is_available", return_value=False)
    assert pick_device() == "cuda"


def test_pick_device_falls_back_to_mps(mocker):
    mocker.patch("clinical_ai.device.torch.cuda.is_available", return_value=False)
    mocker.patch("clinical_ai.device.torch.backends.mps.is_available", return_value=True)
    assert pick_device() == "mps"


def test_pick_device_falls_back_to_cpu(mocker):
    mocker.patch("clinical_ai.device.torch.cuda.is_available", return_value=False)
    mocker.patch("clinical_ai.device.torch.backends.mps.is_available", return_value=False)
    assert pick_device() == "cpu"
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest tests/test_device.py -v`

Expected: `ModuleNotFoundError: No module named 'clinical_ai'` (or all three tests collected as ERROR for the same reason).

- [ ] **Step 3: Create the empty package marker**

Create `clinical_ai/__init__.py` with no content (empty file).

- [ ] **Step 4: Implement `pick_device`**

Create `clinical_ai/device.py`:

```python
"""Device selection for PyTorch-backed models (MedASR runs on MPS/CUDA/CPU)."""

from __future__ import annotations

from typing import Literal

import torch


def pick_device() -> Literal["cpu", "cuda", "mps"]:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

- [ ] **Step 5: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_device.py -v`

Expected: 3 passed.

- [ ] **Step 6: Update `transcribe.py` to import `pick_device` from the new location**

In `transcribe.py`, delete the `pick_device` function definition (lines defining `def pick_device() -> str:` through `return "cpu"`) and add this import alongside the other imports:

```python
from clinical_ai.device import pick_device  # noqa: E402
```

Place it under the `from transformers import pipeline` line (also noqa-ed for the same reason).

- [ ] **Step 7: Verify the CLI still works**

Run: `uv run transcribe.py --help`

Expected: argparse help text prints, no traceback.

- [ ] **Step 8: Delete the old test file**

Delete `tests/test_transcribe.py` (its 3 tests now live in `test_device.py`).

```bash
git rm tests/test_transcribe.py
```

- [ ] **Step 9: Run the full test suite to confirm nothing else broke**

Run: `uv run pytest -v`

Expected: 3 passed (the migrated `test_device.py` tests).

- [ ] **Step 10: Commit**

```bash
git add clinical_ai/__init__.py clinical_ai/device.py tests/test_device.py transcribe.py
git commit -m "Move pick_device into clinical_ai package"
```

---

## Task 3: Build `clinical_ai/asr.py`

**Files:**
- Create: `clinical_ai/asr.py`
- Create: `tests/test_asr.py`

- [ ] **Step 1: Write the failing tests for the ASR module**

Create `tests/test_asr.py`:

```python
from pathlib import Path

from clinical_ai.asr import load_asr_pipeline, transcribe


def test_load_asr_pipeline_calls_transformers_with_right_args(mocker):
    fake_pipe = object()
    pipeline_mock = mocker.patch("clinical_ai.asr.pipeline", return_value=fake_pipe)

    result = load_asr_pipeline("google/medasr", "mps")

    assert result is fake_pipe
    pipeline_mock.assert_called_once_with(
        task="automatic-speech-recognition",
        model="google/medasr",
        device="mps",
    )


def test_transcribe_forwards_chunk_and_stride_and_unwraps_text(mocker):
    pipe = mocker.MagicMock(return_value={"text": "hello world"})

    result = transcribe(pipe, "/tmp/audio.wav", chunk_s=15.0, stride_s=1.5)

    assert result == "hello world"
    pipe.assert_called_once_with("/tmp/audio.wav", chunk_length_s=15.0, stride_length_s=1.5)


def test_transcribe_stringifies_path(mocker):
    pipe = mocker.MagicMock(return_value={"text": "ok"})

    transcribe(pipe, Path("/tmp/audio.wav"))

    args, _ = pipe.call_args
    assert args[0] == "/tmp/audio.wav"
    assert isinstance(args[0], str)


def test_transcribe_passes_bytes_unchanged(mocker):
    pipe = mocker.MagicMock(return_value={"text": "ok"})
    audio_bytes = b"\x00\x01\x02"

    transcribe(pipe, audio_bytes)

    args, _ = pipe.call_args
    assert args[0] is audio_bytes


def test_transcribe_falls_back_to_str_for_non_dict_results(mocker):
    pipe = mocker.MagicMock(return_value="raw string result")

    result = transcribe(pipe, "/tmp/audio.wav")

    assert result == "raw string result"
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest tests/test_asr.py -v`

Expected: `ModuleNotFoundError: No module named 'clinical_ai.asr'`.

- [ ] **Step 3: Implement `clinical_ai/asr.py`**

Create `clinical_ai/asr.py`:

```python
"""MedASR loader and transcription helpers (no Streamlit knowledge)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import pipeline


def load_asr_pipeline(model_id: str, device: str) -> Any:
    return pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        device=device,
    )


def transcribe(
    pipe: Any,
    audio: bytes | str | Path,
    chunk_s: float = 20.0,
    stride_s: float = 2.0,
) -> str:
    if isinstance(audio, Path):
        audio = str(audio)
    result = pipe(audio, chunk_length_s=chunk_s, stride_length_s=stride_s)
    return result["text"] if isinstance(result, dict) else str(result)
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_asr.py -v`

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add clinical_ai/asr.py tests/test_asr.py
git commit -m "Add clinical_ai.asr with load and transcribe helpers"
```

---

## Task 4: Build `clinical_ai/prompts.py`

**Files:**
- Create: `clinical_ai/prompts.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing tests for the prompt module**

Create `tests/test_prompts.py`:

```python
from clinical_ai.prompts import SOAP_SYSTEM_PROMPT, format_soap_messages


def test_system_prompt_mentions_soap_and_all_four_sections():
    assert "SOAP" in SOAP_SYSTEM_PROMPT
    for label in ("Subjective", "Objective", "Assessment", "Plan"):
        assert label in SOAP_SYSTEM_PROMPT


def test_format_soap_messages_returns_two_role_messages():
    msgs = format_soap_messages("patient reports headache")

    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_format_soap_messages_uses_system_prompt_verbatim():
    msgs = format_soap_messages("anything")

    assert msgs[0]["content"] == SOAP_SYSTEM_PROMPT


def test_format_soap_messages_embeds_transcript_in_user_message():
    transcript = "patient reports a sharp left-sided chest pain for two days"

    msgs = format_soap_messages(transcript)

    assert transcript in msgs[1]["content"]
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest tests/test_prompts.py -v`

Expected: `ModuleNotFoundError: No module named 'clinical_ai.prompts'`.

- [ ] **Step 3: Implement `clinical_ai/prompts.py`**

Create `clinical_ai/prompts.py`:

```python
"""SOAP-note prompt template — kept separate so wording can iterate without touching the LLM wrapper."""

from __future__ import annotations

SOAP_SYSTEM_PROMPT = """You are a medical scribe assisting a clinician. Given a transcript of a patient visit, produce a concise SOAP note in Markdown with exactly four sections, in this order:

# Subjective
The patient's reported symptoms, history of present illness, and relevant background in their own words.

# Objective
Examination findings, vital signs, and any test results explicitly mentioned in the transcript.

# Assessment
Your differential or working diagnosis based only on what was discussed. State uncertainty when present.

# Plan
The agreed next steps: investigations, treatments, follow-up, and patient education.

Rules:
- Use only information present in the transcript. Do NOT invent symptoms, doses, or findings.
- If a section has no supporting information, write "Not discussed."
- Keep the note suitable for a clinician to review and edit before signing — this is a draft, not a final record.
"""


def format_soap_messages(transcript: str) -> list[dict]:
    return [
        {"role": "system", "content": SOAP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Transcript of the patient visit:\n\n{transcript}\n\nProduce the SOAP note now.",
        },
    ]
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_prompts.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add clinical_ai/prompts.py tests/test_prompts.py
git commit -m "Add clinical_ai.prompts with SOAP system prompt"
```

---

## Task 5: Build `clinical_ai/llm.py`

**Files:**
- Create: `clinical_ai/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write the failing tests for the LLM module**

Create `tests/test_llm.py`:

```python
from types import SimpleNamespace

from clinical_ai.llm import load_medgemma, stream_soap


def test_load_medgemma_calls_mlx_load_with_default_model_id(mocker):
    fake_model = object()
    fake_tokenizer = object()
    load_mock = mocker.patch("clinical_ai.llm.load", return_value=(fake_model, fake_tokenizer))

    model, tokenizer = load_medgemma()

    assert model is fake_model
    assert tokenizer is fake_tokenizer
    load_mock.assert_called_once_with("mlx-community/medgemma-27b-text-it-4bit")


def test_load_medgemma_accepts_custom_model_id(mocker):
    load_mock = mocker.patch("clinical_ai.llm.load", return_value=(object(), object()))

    load_medgemma("mlx-community/some-other-checkpoint")

    load_mock.assert_called_once_with("mlx-community/some-other-checkpoint")


def test_stream_soap_yields_chunks_in_order(mocker):
    tokenizer = mocker.MagicMock()
    tokenizer.apply_chat_template.return_value = "PROMPT"
    chunks = [SimpleNamespace(text="S"), SimpleNamespace(text="OAP"), SimpleNamespace(text=" note")]
    mocker.patch("clinical_ai.llm.stream_generate", return_value=iter(chunks))
    mocker.patch("clinical_ai.llm.make_sampler", return_value="SAMPLER")

    result = list(stream_soap(object(), tokenizer, "transcript text"))

    assert result == ["S", "OAP", " note"]


def test_stream_soap_applies_chat_template_with_messages(mocker):
    tokenizer = mocker.MagicMock()
    tokenizer.apply_chat_template.return_value = "PROMPT"
    mocker.patch("clinical_ai.llm.stream_generate", return_value=iter([]))
    mocker.patch("clinical_ai.llm.make_sampler", return_value="SAMPLER")

    list(stream_soap(object(), tokenizer, "patient reports headache"))

    args, kwargs = tokenizer.apply_chat_template.call_args
    messages = args[0] if args else kwargs["conversation"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "patient reports headache" in messages[1]["content"]
    assert kwargs.get("add_generation_prompt") is True


def test_stream_soap_threads_max_tokens_and_temperature(mocker):
    tokenizer = mocker.MagicMock()
    tokenizer.apply_chat_template.return_value = "PROMPT"
    sampler_mock = mocker.patch("clinical_ai.llm.make_sampler", return_value="SAMPLER")
    stream_mock = mocker.patch("clinical_ai.llm.stream_generate", return_value=iter([]))

    list(stream_soap(object(), tokenizer, "x", max_tokens=512, temperature=0.7))

    sampler_mock.assert_called_once_with(temp=0.7)
    _, kwargs = stream_mock.call_args
    assert kwargs["max_tokens"] == 512
    assert kwargs["sampler"] == "SAMPLER"
    assert kwargs["prompt"] == "PROMPT"
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest tests/test_llm.py -v`

Expected: `ModuleNotFoundError: No module named 'clinical_ai.llm'`.

- [ ] **Step 3: Implement `clinical_ai/llm.py`**

Create `clinical_ai/llm.py`:

```python
"""MedGemma loader and streaming SOAP generator (MLX backend, no Streamlit knowledge)."""

from __future__ import annotations

from typing import Any, Iterator

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from clinical_ai.prompts import format_soap_messages

DEFAULT_MODEL_ID = "mlx-community/medgemma-27b-text-it-4bit"


def load_medgemma(model_id: str = DEFAULT_MODEL_ID) -> tuple[Any, Any]:
    return load(model_id)


def stream_soap(
    model: Any,
    tokenizer: Any,
    transcript: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> Iterator[str]:
    messages = format_soap_messages(transcript)
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    sampler = make_sampler(temp=temperature)
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        yield response.text
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_llm.py -v`

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add clinical_ai/llm.py tests/test_llm.py
git commit -m "Add clinical_ai.llm with MedGemma loader and streaming SOAP generator"
```

---

## Task 6: Refactor `transcribe.py` to a thin CLI shim

**Files:**
- Modify: `transcribe.py` (full rewrite)

- [ ] **Step 1: Replace `transcribe.py` with a CLI shim**

Replace the entire contents of `transcribe.py` with:

```python
#!/usr/bin/env python3
"""CLI shim for MedASR transcription. The Streamlit app uses the same clinical_ai.asr module."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing anything that touches HF — they read HF_TOKEN at import time.
load_dotenv()

from clinical_ai.asr import load_asr_pipeline, transcribe  # noqa: E402
from clinical_ai.device import pick_device  # noqa: E402


def fetch_sample(model_id: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=model_id, filename="test_audio.wav"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Transcribe medical audio with Google MedASR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("audio", type=Path, nargs="?", help="Path to an audio file (wav/mp3/flac/m4a).")
    ap.add_argument("--model", default="google/medasr", help="HF model id.")
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device.",
    )
    ap.add_argument("--chunk-s", type=float, default=20.0, help="Chunk length in seconds.")
    ap.add_argument("--stride-s", type=float, default=2.0, help="Chunk overlap in seconds.")
    ap.add_argument(
        "--sample",
        action="store_true",
        help="Download and transcribe the sample audio bundled with the model repo.",
    )
    args = ap.parse_args()

    if args.sample:
        audio_path = fetch_sample(args.model)
    elif args.audio is None:
        ap.error("audio path is required (or pass --sample)")
    else:
        audio_path = args.audio
        if not audio_path.exists():
            print(f"error: audio file not found: {audio_path}", file=sys.stderr)
            return 1

    device = pick_device() if args.device == "auto" else args.device
    print(f"[medasr] model={args.model} device={device} audio={audio_path}", file=sys.stderr)

    pipe = load_asr_pipeline(args.model, device)
    text = transcribe(pipe, audio_path, chunk_s=args.chunk_s, stride_s=args.stride_s)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify `--help` still works**

Run: `uv run transcribe.py --help`

Expected: argparse help text prints with all 6 flags (`audio`, `--model`, `--device`, `--chunk-s`, `--stride-s`, `--sample`) and defaults visible.

- [ ] **Step 3: Run the full test suite to confirm nothing broke**

Run: `uv run pytest -v`

Expected: 17 passed (3 device + 5 asr + 4 prompts + 5 llm).

- [ ] **Step 4: Commit**

```bash
git add transcribe.py
git commit -m "Refactor transcribe.py to thin CLI shim over clinical_ai package"
```

---

## Task 7: Switch coverage scope to the `clinical_ai` package

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update the coverage source**

In `pyproject.toml`, find the `[tool.coverage.run]` section and change:

```toml
[tool.coverage.run]
source = ["transcribe"]
```

to:

```toml
[tool.coverage.run]
source = ["clinical_ai"]
```

- [ ] **Step 2: Run the test suite with coverage to confirm the new scope works**

Run: `uv run pytest --cov=clinical_ai --cov-report=term-missing`

Expected: 17 tests pass; coverage report shows lines for `clinical_ai/device.py`, `clinical_ai/asr.py`, `clinical_ai/llm.py`, `clinical_ai/prompts.py` (no `transcribe` row, since the CLI shim is intentionally excluded from the package scope).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "Point coverage scope at clinical_ai package"
```

---

## Task 8: Build the Streamlit app (`app.py`)

**Files:**
- Create: `app.py`

This task has no automated tests — the spec marks the smoke test as optional and worth skipping if flaky. Verification is manual via `streamlit run app.py`. Each section of the file maps to a state in the design (A: empty, B: transcribing, C: transcript ready, D: streaming SOAP, E: SOAP ready).

- [ ] **Step 1: Create `app.py`**

Create `app.py`:

```python
"""Streamlit UI for the MedASR -> MedGemma SOAP pipeline."""

from __future__ import annotations

import os
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
    st.error(f"{label}: {type(exc).__name__}: {exc}")
    with st.expander("Details"):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def main() -> None:
    st.set_page_config(page_title="Clinical AI — SOAP", layout="wide")
    st.title("Clinical AI — visit transcription & SOAP draft")
    init_state()
    require_hf_token()

    # Eager model load so warmup happens once and any error surfaces before the user uploads.
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

    # State A: audio upload
    upload = st.file_uploader(
        "Upload a patient visit recording",
        type=["wav", "mp3", "flac", "m4a"],
    )
    if upload is not None and upload.name != st.session_state["audio_name"]:
        st.session_state["audio_bytes"] = upload.getvalue()
        st.session_state["audio_name"] = upload.name
        clear_downstream_state(after="audio")

        # State B: transcribe
        with st.spinner("Transcribing audio…"):
            try:
                text = transcribe(asr_pipe, st.session_state["audio_bytes"])
            except Exception as exc:
                show_error("Could not transcribe audio", exc)
                st.session_state["audio_bytes"] = None
                st.session_state["audio_name"] = None
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
        # State D: streaming SOAP
        st.subheader("SOAP note")
        placeholder = st.empty()
        buf = ""
        try:
            for chunk in stream_soap(model, tokenizer, st.session_state["tx_edit"]):
                buf += chunk
                placeholder.markdown(buf)
        except Exception as exc:
            show_error("SOAP generation failed", exc)
            st.session_state["soap"] = None
            st.session_state["soap_edit"] = ""
            return
        st.session_state["soap"] = buf
        st.session_state["soap_edit"] = buf

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
            st.session_state["tx"] = None
            st.session_state["tx_edit"] = ""
            st.session_state["soap"] = None
            st.session_state["soap_edit"] = ""
            st.rerun()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Boot the Streamlit app and confirm it starts**

Run: `uv run streamlit run app.py`

Expected:
1. Browser opens to `http://localhost:8501`.
2. The page shows the title "Clinical AI — visit transcription & SOAP draft" and a spinner that reads "Loading models…".
3. After model load (first run: minutes; subsequent: seconds), the page shows the file uploader.

If `HF_TOKEN` is missing, you should instead see the red `HF_TOKEN not set` error message and the app halts there — that's correct behavior.

Stop the server with `Ctrl+C` once the boot is confirmed.

- [ ] **Step 3: Manual smoke test of the full pipeline**

Run: `uv run streamlit run app.py`, then in the browser:

1. Upload the MedASR sample audio (you can grab it via `uv run python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='google/medasr', filename='test_audio.wav'))"` and then drag the printed path into the uploader). Confirm the transcript appears (State C).
2. Click **Generate SOAP note**. Confirm the SOAP note streams in token-by-token (State D → E).
3. Edit a word in the SOAP text area. Click **Download .md** and confirm a `soap_note.md` file is downloaded with your edited content.
4. Click **Start over** and confirm the page returns to the empty uploader state.

Stop the server.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "Add Streamlit UI chaining MedASR transcription with MedGemma SOAP generation"
```

---

## Task 9: Add the integration-test placeholder

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Create the placeholder integration test**

Create `tests/test_integration.py`:

```python
"""Integration tests that hit real models / network. Excluded from default `uv run pytest`.

Run with: `uv run pytest -m integration`
"""

from pathlib import Path

import pytest

from clinical_ai.asr import load_asr_pipeline, transcribe
from clinical_ai.device import pick_device


@pytest.mark.integration
def test_medasr_transcribes_sample_audio_end_to_end():
    from huggingface_hub import hf_hub_download

    audio_path = Path(hf_hub_download(repo_id="google/medasr", filename="test_audio.wav"))
    pipe = load_asr_pipeline("google/medasr", pick_device())

    text = transcribe(pipe, audio_path)

    assert isinstance(text, str)
    assert len(text.strip()) > 0
```

- [ ] **Step 2: Confirm the default test run still excludes it**

Run: `uv run pytest -v`

Expected: 17 tests pass; no test from `test_integration.py` is collected (pytest reports `1 deselected` in the summary).

- [ ] **Step 3: Confirm the marker actually selects it (collection only — do NOT run)**

Run: `uv run pytest -m integration --collect-only`

Expected: `tests/test_integration.py::test_medasr_transcribes_sample_audio_end_to_end` is listed as collected. Do not run it without `-m integration` since it downloads the model.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "Add MedASR integration-test placeholder gated by integration marker"
```

---

## Task 10: Final lint, format, type-check, and test pass

**Files:** none (verification only — fix any issues inline if they appear).

- [ ] **Step 1: Format the codebase**

Run: `uv run ruff format .`

Expected: a list of files formatted (or "X files left unchanged"). No errors.

- [ ] **Step 2: Lint with auto-fix**

Run: `uv run ruff check --fix .`

Expected: "All checks passed!" If anything is reported, fix the underlying issue (do not blanket-add `noqa` unless the issue is the deliberate late-imports pattern in `app.py` and `transcribe.py`, which already carry `# noqa: E402`).

- [ ] **Step 3: Type-check**

Run: `uv run ty check`

Expected: type check passes. If `ty` complains about untyped `Any` returns from `mlx_lm.load` or the `transformers.pipeline` factory, that's expected for third-party code without stubs — leave the `Any` annotations in place.

- [ ] **Step 4: Run the full test suite one more time**

Run: `uv run pytest -v`

Expected: 17 passed, 1 deselected.

- [ ] **Step 5: Commit any lint/format fixups (if there are any)**

```bash
git add -A
git status   # double-check what's staged before committing
git commit -m "Apply final ruff format and lint fixes" || echo "nothing to commit"
```

If `git status` shows nothing staged after Steps 1–3 ran cleanly, skip the commit.

---

## Done

After Task 10, the project will have:
- A `clinical_ai/` package with four focused modules and 17 unit tests (~100% covered for the three new modules).
- A Streamlit app at `app.py` that chains MedASR transcription with MedGemma SOAP generation, with editable transcript and SOAP, streaming output, and a download button.
- A `transcribe.py` CLI shim preserved for offline debugging.
- An integration-test placeholder gated behind `@pytest.mark.integration`.
- New runtime deps `streamlit` and `mlx-lm` in `pyproject.toml` and locked in `uv.lock`.

Run `uv run streamlit run app.py` to start the UI.
