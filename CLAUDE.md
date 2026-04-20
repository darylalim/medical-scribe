# Clinical Documentation

Local-first clinical documentation pipeline for Apple Silicon. Transcribes physician-patient audio with Google MedASR, then drafts a SOAP note with Google MedGemma-27B (MLX 4-bit). Runs entirely locally; audio and notes stay in process memory.

## Architecture

- `clinical_documentation/` — backend package. No Streamlit knowledge.
  - `device.py` — `pick_device()` for PyTorch (CUDA → MPS → CPU).
  - `asr.py` — MedASR pipeline loader + `transcribe()` helper.
  - `llm.py` — MedGemma MLX loader + `stream_soap()` generator.
  - `prompts.py` — SOAP system prompt and `format_soap_messages()`.
- `app.py` — Streamlit UI. The only file that imports `streamlit`.
- `transcribe.py` — thin CLI shim over `clinical_documentation.asr`.
- `.streamlit/config.toml` — server config (caps upload at `maxUploadSize = 100` MB).
- `tests/` — 29 unit tests + 1 gated integration test.

## Commands

| Task | Command |
|---|---|
| Launch UI | `uv run streamlit run app.py` |
| CLI transcribe (sample) | `uv run transcribe.py --sample` |
| CLI transcribe (file) | `uv run transcribe.py path/to/audio.wav` |
| Unit tests | `uv run pytest` |
| Integration test | `uv run pytest -m integration` |
| Format | `uv run ruff format .` |
| Lint | `uv run ruff check .` |
| Type-check | `uv run ty check` |
| Coverage | `uv run pytest --cov=clinical_documentation --cov-report=term-missing` |

## Load-bearing invariants

- `load_dotenv()` runs **before** any import of `transformers`, `torch`, or `mlx_lm`. They read `HF_TOKEN` at import time. Late imports in `app.py` and `transcribe.py` carry `# noqa: E402`.
- `clinical_documentation/` modules never import `streamlit`. `app.py` is the only Streamlit entry point.
- No PHI on disk. Audio bytes live in `st.session_state`; the only artifact is a user-initiated `.md` download.
- Upload cap is enforced in **two** places: `.streamlit/config.toml` (`maxUploadSize = 100`) *and* a soft guard in `app.py`. Keep them in sync.
- `clinical_documentation/prompts.py` is the single source of truth for the SOAP system prompt. Iterate there, not inside `llm.py`.
- `load_medgemma()` must register `<end_of_turn>` as a stop token via `tokenizer.add_eos_token("<end_of_turn>")`. MLX-community Gemma quants default stop tokens to `{<eos>}` only; without this, `stream_generate` runs to `max_tokens` and the model loops on post-hoc "thought" scaffolding.

## Gated models

Both require `HF_TOKEN` in `.env` and acceptance of the Health AI Developer Foundations terms:

- `google/medasr` — ASR, ~400 MB, runs on MPS.
- `mlx-community/medgemma-27b-text-it-4bit` — SOAP generator, ~14 GB, runs on MLX (unified memory). First run downloads weights into `~/.cache/huggingface`.
