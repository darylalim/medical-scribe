# medical-scribe

Local Apple Silicon pipeline for clinical documentation:

1. **Transcribe** a patient-visit recording with [Google MedASR](https://huggingface.co/google/medasr) (Conformer-CTC, 105M parameters).
2. **Review** the transcript and fix any misheard terms.
3. **Generate** a SOAP note draft with [MedGemma-27B](https://huggingface.co/mlx-community/medgemma-27b-text-it-4bit) (MLX 4-bit).

Everything runs locally. Audio and generated notes stay in process memory — the only artifact written to disk is a Markdown file you download from the UI.

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

The UI is a single page with five states:

1. **Upload** a `.wav` / `.mp3` / `.flac` / `.m4a` recording.
2. **Transcribe** (automatic on upload). Transcript appears in an editable text area.
3. **Review** the transcript and fix any ASR errors before generating.
4. **Generate SOAP note.** Output streams token-by-token into a Markdown view.
5. **Edit and download** the SOAP note as `soap_note.md`, or start over.

Outputs are **drafts**. A clinician must review and edit before signing.

## Notes

- Input audio is resampled to 16 kHz mono automatically.
- MedGemma-27B at 4-bit needs ~14 GB of unified memory. 32 GB+ Apple Silicon is comfortable; 16 GB may thrash.
- Outputs are **preliminary** and must be verified before any clinical use. See each model card for speaker, language, and vocabulary caveats.

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
  test_app.py              # state-machine helpers + AppTest smoke
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
uv run pytest                                             # 25 unit tests
uv run pytest --cov=medical_scribe --cov-report=term-missing # with coverage
uv run pytest -m integration                              # real-model integration test
```

Tests marked `@pytest.mark.integration` (currently one, covering MedASR) are excluded from the default run. They download real model weights and hit the network. Opt in with `-m integration`.

Config lives under `[tool.ruff]`, `[tool.ty.environment]`, and `[tool.pytest.ini_options]` in `pyproject.toml`.
