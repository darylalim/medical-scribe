from medical_scribe.prompts import SOAP_SYSTEM_PROMPT, format_soap_messages


def test_system_prompt_mentions_soap_and_all_four_sections():
    assert "SOAP" in SOAP_SYSTEM_PROMPT
    for label in ("Subjective", "Objective", "Assessment", "Plan"):
        assert label in SOAP_SYSTEM_PROMPT


def test_format_soap_messages_returns_two_role_messages():
    msgs = format_soap_messages("patient reports headache")

    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_format_soap_messages_uses_system_prompt_verbatim():
    msgs = format_soap_messages("anything")

    assert msgs[0]["content"] == SOAP_SYSTEM_PROMPT


def test_format_soap_messages_embeds_transcript_in_user_message():
    transcript = "patient reports a sharp left-sided chest pain for two days"

    msgs = format_soap_messages(transcript)

    assert transcript in msgs[1]["content"]


def test_system_prompt_mandates_exact_section_headers():
    """Catches removal AND capitalisation drift on the H2 markers
    that medical_scribe.soap_sections.parse_soap_sections splits on."""
    for header in ("## Subjective", "## Objective", "## Assessment", "## Plan"):
        assert header in SOAP_SYSTEM_PROMPT, f"missing literal {header!r}"
