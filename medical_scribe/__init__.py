"""Public API for the medical-scribe backend package.

Importing from this package transitively loads `transformers` (via `asr`) and
`mlx_lm` (via `llm`), which read `HF_TOKEN` at import time. Callers that need
`.env` values must run `load_dotenv()` before the first import.
"""

from medical_scribe.asr import load_asr_pipeline, transcribe
from medical_scribe.device import pick_device
from medical_scribe.llm import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_ID,
    load_medgemma,
    stream_soap,
)

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL_ID",
    "load_asr_pipeline",
    "load_medgemma",
    "pick_device",
    "stream_soap",
    "transcribe",
]
