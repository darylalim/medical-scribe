from types import SimpleNamespace

import pytest

from clinical_ai.llm import load_medgemma, stream_soap


def test_stream_soap_populates_meta_finish_reason(mocker):
    tokenizer = mocker.MagicMock()
    tokenizer.apply_chat_template.return_value = "PROMPT"
    chunks = [
        SimpleNamespace(text="part1", finish_reason=None),
        SimpleNamespace(text="part2", finish_reason="length"),
    ]
    mocker.patch("clinical_ai.llm.stream_generate", return_value=iter(chunks))
    mocker.patch("clinical_ai.llm.make_sampler", return_value="SAMPLER")

    meta: dict = {}
    list(stream_soap(object(), tokenizer, "x", meta=meta))

    assert meta["finish_reason"] == "length"


def test_load_medgemma_calls_mlx_load_with_default_model_id(mocker):
    fake_model = object()
    fake_tokenizer = object()
    load_mock = mocker.patch("clinical_ai.llm.load", return_value=(fake_model, fake_tokenizer))

    model, tokenizer = load_medgemma()

    assert model is fake_model
    assert tokenizer is fake_tokenizer
    load_mock.assert_called_once_with("mlx-community/medgemma-27b-text-it-4bit")


def test_load_medgemma_accepts_custom_model_id(mocker):
    load_mock = mocker.patch("clinical_ai.llm.load", return_value=(object(), object()))

    load_medgemma("mlx-community/some-other-checkpoint")

    load_mock.assert_called_once_with("mlx-community/some-other-checkpoint")


def test_stream_soap_yields_chunks_in_order(mocker):
    tokenizer = mocker.MagicMock()
    tokenizer.apply_chat_template.return_value = "PROMPT"
    chunks = [SimpleNamespace(text="S"), SimpleNamespace(text="OAP"), SimpleNamespace(text=" note")]
    mocker.patch("clinical_ai.llm.stream_generate", return_value=iter(chunks))
    mocker.patch("clinical_ai.llm.make_sampler", return_value="SAMPLER")

    result = list(stream_soap(object(), tokenizer, "transcript text"))

    assert result == ["S", "OAP", " note"]


def test_stream_soap_applies_chat_template_with_messages(mocker):
    tokenizer = mocker.MagicMock()
    tokenizer.apply_chat_template.return_value = "PROMPT"
    mocker.patch("clinical_ai.llm.stream_generate", return_value=iter([]))
    mocker.patch("clinical_ai.llm.make_sampler", return_value="SAMPLER")

    list(stream_soap(object(), tokenizer, "patient reports headache"))

    args, kwargs = tokenizer.apply_chat_template.call_args
    messages = args[0] if args else kwargs["conversation"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "patient reports headache" in messages[1]["content"]
    assert kwargs.get("add_generation_prompt") is True
    assert kwargs.get("tokenize") is False


def test_stream_soap_threads_max_tokens_and_temperature(mocker):
    tokenizer = mocker.MagicMock()
    tokenizer.apply_chat_template.return_value = "PROMPT"
    sampler_mock = mocker.patch("clinical_ai.llm.make_sampler", return_value="SAMPLER")
    stream_mock = mocker.patch("clinical_ai.llm.stream_generate", return_value=iter([]))

    list(stream_soap(object(), tokenizer, "x", max_tokens=512, temperature=0.7))

    sampler_mock.assert_called_once_with(temp=0.7)
    _, kwargs = stream_mock.call_args
    assert kwargs["max_tokens"] == 512
    assert kwargs["sampler"] == "SAMPLER"
    assert kwargs["prompt"] == "PROMPT"


def test_stream_soap_rejects_empty_transcript(mocker):
    mocker.patch("clinical_ai.llm.make_sampler", return_value="SAMPLER")
    mocker.patch("clinical_ai.llm.stream_generate", return_value=iter([]))

    with pytest.raises(ValueError, match="transcript must be a non-empty string"):
        list(stream_soap(object(), mocker.MagicMock(), ""))

    with pytest.raises(ValueError, match="transcript must be a non-empty string"):
        list(stream_soap(object(), mocker.MagicMock(), "   \n\t  "))
