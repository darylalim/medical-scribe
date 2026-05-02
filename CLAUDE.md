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
- `app.py` — Streamlit UI: sidebar (with `+ New session`) + persistent vertical split view (transcript pane left, SOAP pane right) starting from State C. The only file that imports `streamlit`. Three-state state machine (A: empty → B: transcribing → C: working). State C absorbs streaming, SOAP-ready, and editable cards as sub-renders; the SOAP pane handles its own state-aware branching internally.
- `.streamlit/config.toml` — server config (caps upload at `maxUploadSize = 100` MB).
- `tests/` — ~95 unit tests + 1 gated integration test.

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
- The primary action button (label flips between `Generate SOAP note` and `Regenerate SOAP` via `primary_action_label` based on whether `soap` is truthy) is **idempotent** — clicking it post-SOAP discards in-progress section edits and re-runs against the current transcript. The label flip surfaces the destructive nature; the click handler is the same in both states.
- Editable `st.text_area` widgets in conditionally-rendered branches (the transcript text_area in `_render_transcript_pane`; the four per-section buffers in `_render_soap_pane`'s always-editable cards) use the `value=` + manual `st.session_state` sync pattern, **not** `key=`. Streamlit cleans up widget-managed session-state keys on unmount, so a `key=`-bound text area loses its value the moment its parent branch stops rendering — silently wiping `tx_edit` and the four `<section>_edit` buffers when the user transitions between states.
- The Copy-to-clipboard button uses `streamlit.components.v1.html` (iframe-rendered, JS executes reliably, has `clipboard-write` permission), **not** `st.html` — the latter strips inline event handlers in current Streamlit versions, causing silent click failures.
- The `+ New session` confirm dialog is gated on the `_show_reset_dialog` session-state flag, **not** on `@st.dialog`'s implicit open/close lifecycle. The button click sets the flag and triggers `st.rerun()`; the dialog's Cancel/Discard buttons clear the flag and rerun again. This keeps the modal open across the click-rerun and makes the bypass-in-State-A path explicit. `_show_reset_dialog` lives in `INITIAL_STATE` so `reset_state()` always closes a stale dialog as a side effect.
- Both SOAP card render paths (streaming markdown during generation, always-editable text_areas post-stream) call the single `_render_section_header(name)` helper for the chip + name header. Iterate the visual there. `SECTION_COLORS` and `SECTION_INITIALS` must stay in sync with `SOAP_SECTIONS`; `tests/test_app.py::test_section_color_and_initial_maps_cover_all_soap_sections` catches drift.

## Gated models

Both require `HF_TOKEN` in `.env` and acceptance of the Health AI Developer Foundations terms:

- `google/medasr` — ASR, ~400 MB, runs on MPS.
- `mlx-community/medgemma-27b-text-it-4bit` — SOAP generator, ~14 GB, runs on MLX (unified memory). First run downloads weights into `~/.cache/huggingface`.
