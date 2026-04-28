# Medical Scribe

Local-first pipeline for Apple Silicon. Live-capture or upload a physician-patient visit, transcribe with Google MedASR, then draft a SOAP note with Google MedGemma-27B (MLX 4-bit). Runs entirely locally; audio and notes stay in process memory and **nothing is written to disk**.

## Architecture

- `medical_scribe/` — backend package. No Streamlit knowledge.
  - `__init__.py` — re-exports the public API; `__all__` is the canonical surface.
  - `device.py` — `pick_device()` for PyTorch (CUDA → MPS → CPU).
  - `asr.py` — MedASR pipeline loader + `transcribe()` helper.
  - `llm.py` — MedGemma MLX loader + `stream_soap()` generator.
  - `prompts.py` — SOAP system prompt and `format_soap_messages()`.
  - `soap_sections.py` — `parse_soap_sections`, `assemble_soap`, `format_for_clipboard` (pure string utilities for the four-section SOAP format).
- `app.py` — Streamlit UI: sidebar (with `+ New session`) + main area with Transcript / Notes tabs. The only file that imports `streamlit`. Six-state state machine (A: empty → B: transcribing → C: transcript ready → D: streaming → E: SOAP ready → E-edit: editing).
- `.streamlit/config.toml` — server config (caps upload at `maxUploadSize = 100` MB).
- `tests/` — ~64 unit tests + 1 gated integration test.

## Commands

| Task | Command |
|---|---|
| Launch UI | `uv run streamlit run app.py` |
| Unit tests | `uv run pytest` |
| Integration test | `uv run pytest -m integration` |
| Format | `uv run ruff format .` |
| Lint | `uv run ruff check .` |
| Type-check | `uv run ty check` |
| Coverage | `uv run pytest --cov=medical_scribe --cov-report=term-missing` |

## Load-bearing invariants

- `load_dotenv()` runs **before** any import of `transformers`, `torch`, or `mlx_lm`. They read `HF_TOKEN` at import time. Late imports in `app.py` carry `# noqa: E402`.
- `medical_scribe/` modules never import `streamlit`. `app.py` is the only Streamlit entry point.
- **Nothing is written to disk.** Audio bytes live in `st.session_state`; the SOAP draft lives there too. `Copy to clipboard` is the only export path; there is no Markdown download.
- Upload cap is enforced in **two** places: `.streamlit/config.toml` (`maxUploadSize = 100`) *and* a soft guard in `app.py`. Keep them in sync.
- `medical_scribe/prompts.py` is the single source of truth for the SOAP system prompt. The prompt mandates exact H2 section headers (`## Subjective`, `## Objective`, `## Assessment`, `## Plan`) which `medical_scribe/soap_sections.parse_soap_sections` splits on. Iterate the prompt there, not inside `llm.py`.
- `load_medgemma()` must register `<end_of_turn>` as a stop token via `tokenizer.add_eos_token("<end_of_turn>")`. MLX-community Gemma quants default stop tokens to `{<eos>}` only; without this, `stream_generate` runs to `max_tokens` and the model loops on post-hoc "thought" scaffolding.
- `medical_scribe/__init__.py` re-exports the backend's public API; `__all__` is the canonical surface and `tests/test_init.py` keeps it in sync with the defining modules.
- The `Generate SOAP Note` button on the Transcript tab is **idempotent** — clicking it post-SOAP discards in-progress edits and re-runs against the current transcript. There is no separate Regenerate button.

## Gated models

Both require `HF_TOKEN` in `.env` and acceptance of the Health AI Developer Foundations terms:

- `google/medasr` — ASR, ~400 MB, runs on MPS.
- `mlx-community/medgemma-27b-text-it-4bit` — SOAP generator, ~14 GB, runs on MLX (unified memory). First run downloads weights into `~/.cache/huggingface`.
