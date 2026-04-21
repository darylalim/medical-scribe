"""Device selection for PyTorch-backed models (MedASR runs on MPS/CUDA/CPU)."""

from __future__ import annotations

from typing import Literal

import torch


def pick_device() -> Literal["cpu", "cuda", "mps"]:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
