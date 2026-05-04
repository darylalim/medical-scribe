# Medical Scribe

Local-first pipeline for Apple Silicon. Capture a physician-patient visit (live recording or uploaded file), transcribe with Google MedASR, then draft a SOAP note with Google MedGemma-27B (MLX 4-bit). Runs entirely on-device; audio and notes stay in process memory — **nothing is written to disk**.

## Architecture

- `medical_scribe/` — backend package; never imports Streamlit.
  - `__init__.py` — re-exports the public API. `__all__` is the canonical surface.
  - `device.py` — `pick_device()` for PyTorch (CUDA → MPS → CPU).
  - `asr.py` — MedASR pipeline loader + `transcribe()` helper.
  - `vad.py` — Silero VAD loader + `trim_silence()` returning a `TrimResult` (pre-ASR silence trim; never raises).
  - `llm.py` — MedGemma MLX loader + `stream_soap()` generator.
  - `prompts.py` — SOAP system prompt + `format_soap_messages()`.
  - `soap_sections.py` — `parse_soap_sections`, `assemble_soap`, `format_for_clipboard` (pure string utilities for the four-section SOAP format).
- `app.py` — Streamlit UI; the only file that imports `streamlit`. Five-state machine (A: no audio → B: transcribing → C: transcript ready → D: streaming SOAP → E: SOAP-ready/editable) with a single top bar (`Medical Scribe` title + stage chip + session metadata + `+ New session` button). From State B onward the UI shell is a persistent vertical split (transcript left, SOAP right); the SOAP pane has four sub-renders — "Awaiting transcript" (B), model+timing placeholder (C), skeleton-card streaming (D), and read-mode + click-to-edit cards (E).
- `.streamlit/config.toml` — server config; caps upload at `maxUploadSize = 100` MB.
- `tests/` — ~170 unit tests + 1 gated integration test.

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

### Import order & cross-module boundaries

- `load_dotenv()` runs **before** any `transformers` / `torch` / `mlx_lm` import — these libraries read `HF_TOKEN` at import time. Late imports in `app.py` carry `# noqa: E402`.
- `medical_scribe/` modules **never** import `streamlit`. `app.py` is the only Streamlit entry point.
- `medical_scribe/__init__.py` re-exports the backend's public API. `__all__` is the canonical surface; `tests/test_init.py` enforces it stays in sync with the defining modules.

### Privacy & data lifetime

- **Nothing is written to disk.** Audio bytes and the SOAP draft live in `st.session_state`; `Copy to clipboard` is the only export path (no Markdown download).
- The 100 MB upload cap is enforced in **two** places: `.streamlit/config.toml` (`maxUploadSize = 100`) and a soft guard in `app.py` (`MAX_UPLOAD_MB`). Keep them in sync.
- The audio player in `_render_transcript_pane` plays the **original** uploaded audio (`st.session_state["audio_bytes"]`), **not** the VAD-trimmed bytes from `tx_trim`. Clinician verification depends on hearing what was actually said and where; replacing it with the speech-only concatenation is a silent UX regression.

### Model & pipeline contracts

- `medical_scribe/prompts.py` is the single source of truth for the SOAP system prompt — iterate it there, not in `llm.py`. The prompt mandates exact H2 section headers (`## Subjective`, `## Objective`, `## Assessment`, `## Plan`) which `parse_soap_sections` splits on; capitalization drift breaks parsing silently.
- `load_medgemma()` **must** register `<end_of_turn>` as a stop token via `tokenizer.add_eos_token("<end_of_turn>")`. MLX-community Gemma quants ship with `{<eos>}` only as the default stop set; without this, `stream_generate` runs to `max_tokens` and the model loops on post-hoc "thought" scaffolding.
- `trim_silence()` in `medical_scribe/vad.py` **never raises** — VAD failure paths return a `TrimResult` with `status="error"` and the original input bytes. `_render_transcript_pane`'s State-B branch in `app.py` relies on this contract; it has no try/except wrapping the call. Tests in `tests/test_vad.py::test_trim_silence_never_raises` enforce it across decode, VAD, and encode failures.
- The primary action button is **idempotent** — clicking it post-SOAP discards in-progress section edits and re-runs against the current transcript. Its label flips between `Generate SOAP note` and `Regenerate SOAP` via `primary_action_label` (based on truthiness of `soap`); the click handler is the same in both states. The label flip surfaces the destructive nature.

### UI patterns

- Editable `st.text_area` widgets in conditionally-rendered branches use the `value=` + manual `st.session_state` sync pattern, **not** `key=`. Streamlit cleans up widget-managed session-state keys on unmount, so a `key=`-bound text area in a branch that stops rendering silently loses its value. Affected widgets: the transcript editor in `_render_transcript_pane`, the four per-section buffers in the edit-mode branch of `_render_section_card` (entered when `{name}_editing` is True — the conditionally-rendered branch is exactly the case the invariant exists for).
- The Copy-to-clipboard button uses `st.iframe`, **not** `st.html`. `st.html` strips inline event handlers in current Streamlit versions, causing silent click failures; `st.iframe` renders in a sandboxed iframe with `clipboard-write` permission, so the JS Clipboard API works.
- The `+ New session` confirm dialog is gated on the `_show_reset_dialog` session-state flag, **not** on `@st.dialog`'s implicit open/close lifecycle. The button click sets the flag and reruns; Cancel/Discard clear it and rerun again — this keeps the modal open across the click-rerun and makes the State-A bypass explicit. `_show_reset_dialog` lives in `INITIAL_STATE`, so `reset_state()` also closes a stale dialog as a side effect.
- Both SOAP card render paths (streaming markdown during generation; read-mode and edit-mode branches of `_render_section_card` post-stream) share the single `_render_section_header(name)` helper for the chip + name header — iterate the visual there.
- `SECTION_COLORS` and `SECTION_INITIALS` must stay in sync with `SOAP_SECTIONS`; `tests/test_app.py::test_section_color_and_initial_maps_cover_all_soap_sections` catches drift.
- All four chips use the -700 family (Subjective indigo-700, Objective emerald-700, Assessment amber-700, Plan violet-700) for uniform visual depth and clean WCAG AA contrast on white text. Subjective uses indigo (not blue) so the chip stays distinguishable from the green Objective chip under deuteranopia (most common color-blindness, ~5% of males). The contrast and color-blindness properties are enforced by `tests/test_app.py::test_section_colors_are_all_aa_grade` and `::test_section_colors_have_no_duplicates`.

## Gated models

Both require `HF_TOKEN` in `.env` and acceptance of the Health AI Developer Foundations terms:

- `google/medasr` — ASR, ~400 MB, runs on MPS.
- `mlx-community/medgemma-27b-text-it-4bit` — SOAP generator, ~14 GB, runs on MLX (unified memory). First run downloads weights into `~/.cache/huggingface`.
