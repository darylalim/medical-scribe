"""Public API surface for the medical_scribe package.

Asserts `__all__` stays in sync with its defining modules: every name advertised
is importable at the package level and is the same object as its module-level
definition. Catches re-export drift (renames, removals, accidental wrappers)."""

from __future__ import annotations

import medical_scribe
from medical_scribe import asr, device, llm, soap_sections


def test_package_reexports_match_module_level_definitions():
    expected = {
        "DEFAULT_MAX_TOKENS": llm.DEFAULT_MAX_TOKENS,
        "DEFAULT_MODEL_ID": llm.DEFAULT_MODEL_ID,
        "assemble_soap": soap_sections.assemble_soap,
        "format_for_clipboard": soap_sections.format_for_clipboard,
        "load_asr_pipeline": asr.load_asr_pipeline,
        "load_medgemma": llm.load_medgemma,
        "parse_soap_sections": soap_sections.parse_soap_sections,
        "pick_device": device.pick_device,
        "stream_soap": llm.stream_soap,
        "transcribe": asr.transcribe,
    }
    for name, canonical in expected.items():
        assert getattr(medical_scribe, name) is canonical, (
            f"medical_scribe.{name} is not the same object as its module-level definition"
        )
    assert set(medical_scribe.__all__) == set(expected)
