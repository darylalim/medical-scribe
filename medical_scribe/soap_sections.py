"""SOAP-output parsing, assembly, and clipboard formatting.

Pure string utilities for the four-section SOAP format produced by
medical_scribe.prompts.SOAP_SYSTEM_PROMPT. Kept separate from the LLM
wrapper so the output contract can iterate independently.
"""

from __future__ import annotations

import re

SOAP_SECTIONS: tuple[str, ...] = ("Subjective", "Objective", "Assessment", "Plan")

SECTION_HEADER_RE = re.compile(
    r"^## (Subjective|Objective|Assessment|Plan)\s*$",
    re.MULTILINE,
)


def parse_soap_sections(text: str) -> dict[str, str]:
    """Split SOAP markdown on `## <Section>` H2 headers.

    Returns a dict of {section_name: body}. Tolerant: ignores text
    before the first recognised header; ignores sections appearing in
    unexpected order; returns whatever it finds. Empty dict if no
    recognised headers are present.
    """
    matches = list(SECTION_HEADER_RE.finditer(text))
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1)
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[body_start:body_end].strip()
    return sections


def assemble_soap(sections: dict[str, str]) -> str:
    """Inverse of `parse_soap_sections`. Produces canonical
    `## Subjective\\n<body>\\n\\n## Objective\\n…` markdown.

    Sections missing from the input dict are skipped (no empty
    `## Foo\\n` blocks). Order follows SOAP_SECTIONS regardless of
    insertion order in the input dict. Bodies are preserved verbatim.
    """
    blocks: list[str] = []
    for name in SOAP_SECTIONS:
        if name in sections:
            blocks.append(f"## {name}\n{sections[name]}\n")
    return "\n".join(blocks)


def format_for_clipboard(sections: dict[str, str]) -> str:
    """Plain-text-with-caps export format for the clipboard (and
    eventually EHR-paste) action.

    Real EHRs render the clinical-notes field as plain text — markdown
    headers like `## Subjective` would appear verbatim. Capitalised
    section labels read clearly on paste and match longstanding
    chart-note conventions.

    Sections missing from the input dict are skipped. Order follows
    SOAP_SECTIONS regardless of insertion order in the input dict.
    """
    blocks: list[str] = []
    for name in SOAP_SECTIONS:
        if name in sections:
            blocks.append(f"{name.upper()}\n{sections[name]}\n")
    return "\n".join(blocks)
