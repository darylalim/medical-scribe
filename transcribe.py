#!/usr/bin/env python3
"""CLI shim for MedASR transcription. The Streamlit app uses the same clinical_ai.asr module."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing anything that touches HF — they read HF_TOKEN at import time.
load_dotenv()

from clinical_ai.asr import load_asr_pipeline, transcribe  # noqa: E402
from clinical_ai.device import pick_device  # noqa: E402


def require_hf_token() -> None:
    if not os.environ.get("HF_TOKEN"):
        print(
            "error: HF_TOKEN not set — copy .env.example to .env and add your "
            "Hugging Face token (https://huggingface.co/settings/tokens).",
            file=sys.stderr,
        )
        raise SystemExit(1)


def fetch_sample(model_id: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=model_id, filename="test_audio.wav"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Transcribe medical audio with Google MedASR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("audio", type=Path, nargs="?", help="Path to an audio file (wav/mp3/flac/m4a).")
    ap.add_argument("--model", default="google/medasr", help="HF model id.")
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device.",
    )
    ap.add_argument("--chunk-s", type=float, default=20.0, help="Chunk length in seconds.")
    ap.add_argument("--stride-s", type=float, default=2.0, help="Chunk overlap in seconds.")
    ap.add_argument(
        "--sample",
        action="store_true",
        help="Download and transcribe the sample audio bundled with the model repo.",
    )
    args = ap.parse_args()
    require_hf_token()

    if args.sample:
        audio_path = fetch_sample(args.model)
    elif args.audio is None:
        ap.error("audio path is required (or pass --sample)")
    else:
        audio_path = args.audio
        if not audio_path.exists():
            print(f"error: audio file not found: {audio_path}", file=sys.stderr)
            return 1

    device = pick_device() if args.device == "auto" else args.device
    print(f"[medasr] model={args.model} device={device} audio={audio_path}", file=sys.stderr)

    pipe = load_asr_pipeline(args.model, device)
    text = transcribe(pipe, audio_path, chunk_s=args.chunk_s, stride_s=args.stride_s)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
