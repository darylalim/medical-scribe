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
from medical_scribe.soap_sections import (
    SECTION_HEADER_RE,
    SOAP_SECTIONS,
    assemble_soap,
    format_for_clipboard,
    parse_soap_sections,
)

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL_ID",
    "SECTION_HEADER_RE",
    "SOAP_SECTIONS",
    "assemble_soap",
    "format_for_clipboard",
    "load_asr_pipeline",
    "load_medgemma",
    "parse_soap_sections",
    "pick_device",
    "stream_soap",
    "transcribe",
]
