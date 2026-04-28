# medical-scribe

Local-first pipeline for Apple Silicon:

1. **Transcribe** a patient-visit recording with [Google MedASR](https://huggingface.co/google/medasr) (Conformer-CTC, 105M parameters).
2. **Review** the transcript and fix any misheard terms.
3. **Generate** a SOAP note draft with [MedGemma-27B](https://huggingface.co/mlx-community/medgemma-27b-text-it-4bit) (MLX 4-bit).

Everything runs locally. Audio and generated notes stay in process memory — the only artifact written to disk is a Markdown file you download from the UI.

## Safety

Outputs are **preliminary drafts**. A clinician must review and edit every SOAP note before it enters the clinical record or is signed. See each model card for speaker, language, and vocabulary caveats.

## One-time setup

1. **Accept the licenses.** Both models are gated under the Health AI Developer Foundations terms. Visit each page, sign in, and click *Agree and access repository*:
   - https://huggingface.co/google/medasr
   - https://huggingface.co/google/medgemma-27b-text-it

2. **Install dependencies** (creates `.venv/` automatically):
   ```bash
   uv sync
   ```
   The first install pulls a pinned build of `transformers` from source (MedASR needs v5.0+), plus `streamlit`, `mlx-lm`, and `mlx`. Expect a few minutes.

   > **iCloud users:** `uv sync` creates `.venv/` inside the repo. If this repo lives on iCloud Drive, iCloud can replace symlinks with alias files during sync — silently breaking the virtualenv on another device. If you plan to use this on multiple machines, either keep the repo outside iCloud Drive or exclude `.venv/` from iCloud sync.

3. **Set your Hugging Face token.** Create a read-access token at https://huggingface.co/settings/tokens, then:
   ```bash
   cp .env.example .env
   # edit .env and set HF_TOKEN=hf_...
   ```
   `.env` is gitignored. The Streamlit app calls `load_dotenv()` before any HF import.

## Streamlit app

```bash
uv run streamlit run app.py
```

On first run, MedGemma weights (~14 GB) download into `~/.cache/huggingface` — be patient. Subsequent launches load from cache in seconds.

Uploads are capped at **100 MB** (`.streamlit/config.toml`). Split longer recordings before uploading.

The UI has a sidebar plus two main-area tabs:

- **Sidebar** — `+ New session` button to reset state and start fresh.
- **Transcript tab** — record audio in-browser via `st.audio_input`, or expand "Or upload an existing recording" to choose a `.wav` / `.mp3` / `.flac` / `.m4a` file. Once audio exists, the audio player and the editable transcript text area appear; the **Generate SOAP Note →** button at the bottom-right kicks off generation and auto-switches to the Notes tab.
- **Notes tab** — the SOAP draft renders as four cards (Subjective / Objective / Assessment / Plan), one card per section. While the model streams, cards appear one at a time as each section completes; a "Generating…" indicator shows below until the last card lands. The action row at the bottom is `[Edit · Copy to clipboard]`.
- **Edit mode** — clicking **Edit** turns each card into an editable text area in place; the action row becomes `[Done · Copy to clipboard]`. Done commits the edits back to the SOAP source. Copy can be clicked at any time, including mid-edit.

Workflow:

1. **Record** the visit live with the mic widget — or expand the upload section to attach an existing `.wav` / `.mp3` / `.flac` / `.m4a` recording.
2. **Review** the auto-generated transcript — replay the audio to verify ambiguous segments. Edit if needed.
3. **Generate** the SOAP note. The view auto-switches to Notes; cards stream in section-by-section.
4. **Edit** any card if the draft needs correction, then click Done.
5. **Copy** the note to clipboard for paste into the EHR or chart.

To retry the same visit with a different draft, edit the transcript and click **Generate SOAP Note** again — the button is idempotent and re-runs against the current transcript. To start completely over, click **+ New session** in the sidebar.

Nothing is written to disk. Audio bytes and the SOAP draft live in process memory only.

## Notes

- Input audio is resampled to 16 kHz mono automatically.
- MedGemma-27B at 4-bit needs ~14 GB of unified memory. 32 GB+ Apple Silicon is comfortable; 16 GB may thrash.

## Project layout

```
medical_scribe/            # backend package (no streamlit)
  __init__.py              # public API re-exports (__all__ is canonical)
  asr.py                   # MedASR loader + transcribe
  device.py                # pick_device
  llm.py                   # MedGemma loader + stream_soap
  prompts.py               # SOAP system prompt + message formatter
app.py                     # Streamlit UI (single entry point)
.streamlit/config.toml     # server config (maxUploadSize = 100 MB)
tests/
  test_app.py              # app.py helpers + AppTest smoke
  test_asr.py              # load_asr_pipeline + transcribe
  test_device.py           # pick_device precedence
  test_init.py             # public API surface + re-export identity
  test_integration.py      # gated by @pytest.mark.integration
  test_llm.py              # load_medgemma + stream_soap
  test_prompts.py          # SOAP prompt + format_soap_messages
```

## Development

[Ruff](https://docs.astral.sh/ruff/) handles lint and format:
```bash
uv run ruff format .
uv run ruff check .
uv run ruff check --fix .
```

### Type checking

[ty](https://docs.astral.sh/ty/) is Astral's Python type checker and LSP:
```bash
uv run ty check
uv run ty server   # LSP over stdio
```

Editor integration: VS Code uses the [ty extension](https://marketplace.visualstudio.com/items?itemName=astral-sh.ty); Zed/Neovim/Helix point their LSP client at `uv run ty server`.

### Testing

[pytest](https://docs.pytest.org/) with [pytest-cov](https://pytest-cov.readthedocs.io/) and [pytest-mock](https://pytest-mock.readthedocs.io/):

```bash
uv run pytest                                             # 46 unit tests
uv run pytest --cov=medical_scribe --cov-report=term-missing # with coverage
uv run pytest -m integration                              # real-model integration test
```

Tests marked `@pytest.mark.integration` (currently one, covering MedASR) are excluded from the default run. They download real model weights and hit the network. Opt in with `-m integration`.

Config lives under `[tool.ruff]`, `[tool.ty.environment]`, and `[tool.pytest.ini_options]` in `pyproject.toml`.
