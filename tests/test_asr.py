from pathlib import Path

from clinical_ai.asr import load_asr_pipeline, transcribe


def test_load_asr_pipeline_calls_transformers_with_right_args(mocker):
    fake_pipe = object()
    pipeline_mock = mocker.patch("clinical_ai.asr.pipeline", return_value=fake_pipe)

    result = load_asr_pipeline("google/medasr", "mps")

    assert result is fake_pipe
    pipeline_mock.assert_called_once_with(
        task="automatic-speech-recognition",
        model="google/medasr",
        device="mps",
    )


def test_transcribe_forwards_chunk_and_stride_and_unwraps_text(mocker):
    pipe = mocker.MagicMock(return_value={"text": "hello world"})

    result = transcribe(pipe, "/tmp/audio.wav", chunk_s=15.0, stride_s=1.5)

    assert result == "hello world"
    pipe.assert_called_once_with("/tmp/audio.wav", chunk_length_s=15.0, stride_length_s=1.5)


def test_transcribe_stringifies_path(mocker):
    pipe = mocker.MagicMock(return_value={"text": "ok"})

    transcribe(pipe, Path("/tmp/audio.wav"))

    args, _ = pipe.call_args
    assert args[0] == "/tmp/audio.wav"
    assert isinstance(args[0], str)


def test_transcribe_passes_bytes_unchanged(mocker):
    pipe = mocker.MagicMock(return_value={"text": "ok"})
    audio_bytes = b"\x00\x01\x02"

    transcribe(pipe, audio_bytes)

    args, _ = pipe.call_args
    assert args[0] is audio_bytes


def test_transcribe_falls_back_to_str_for_non_dict_results(mocker):
    pipe = mocker.MagicMock(return_value="raw string result")

    result = transcribe(pipe, "/tmp/audio.wav")

    assert result == "raw string result"
