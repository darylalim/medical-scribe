"""Tests for medical_scribe.soap_sections — SOAP-output parser, assembler,
and clipboard formatter."""

from medical_scribe.soap_sections import parse_soap_sections

SAMPLE_FULL = """\
## Subjective
54M reports L-sided chest pain x 2 days, worse with exertion.

## Objective
BP 132/84, HR 78. Cardiac exam unremarkable.

## Assessment
Atypical chest pain, low-risk presentation. Differential includes MSK strain.

## Plan
ECG, troponin. Trial PPI. Follow-up 1 week.
"""


def test_parse_soap_sections_all_four_in_order():
    sections = parse_soap_sections(SAMPLE_FULL)
    assert set(sections) == {"Subjective", "Objective", "Assessment", "Plan"}
    assert sections["Subjective"].startswith("54M reports")
    assert sections["Objective"].startswith("BP 132/84")
    assert sections["Assessment"].startswith("Atypical")
    assert sections["Plan"].startswith("ECG, troponin")


def test_parse_soap_sections_strips_section_bodies():
    text = "## Subjective\n  hello  \n\n## Objective\n  world  \n"
    sections = parse_soap_sections(text)
    assert sections["Subjective"] == "hello"
    assert sections["Objective"] == "world"


def test_parse_soap_sections_preamble_ignored():
    text = "Some preamble.\n\n## Subjective\nbody\n"
    sections = parse_soap_sections(text)
    assert sections == {"Subjective": "body"}


def test_parse_soap_sections_partial_only_subjective():
    sections = parse_soap_sections("## Subjective\nbody\n")
    assert sections == {"Subjective": "body"}


def test_parse_soap_sections_partial_subjective_and_objective():
    sections = parse_soap_sections("## Subjective\nfoo\n\n## Objective\nbar\n")
    assert sections == {"Subjective": "foo", "Objective": "bar"}


def test_parse_soap_sections_empty_input_returns_empty_dict():
    assert parse_soap_sections("") == {}


def test_parse_soap_sections_no_recognised_headers_returns_empty_dict():
    assert parse_soap_sections("This is just prose with no SOAP headers.") == {}


def test_parse_soap_sections_unexpected_order_still_parses():
    text = "## Plan\np\n\n## Subjective\ns\n"
    sections = parse_soap_sections(text)
    assert sections == {"Plan": "p", "Subjective": "s"}


def test_parse_soap_sections_blank_lines_within_body_preserved():
    text = "## Subjective\nline 1\n\nline 2\n\n## Objective\nbody\n"
    sections = parse_soap_sections(text)
    assert sections["Subjective"] == "line 1\n\nline 2"
