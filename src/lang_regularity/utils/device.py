from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    selected: str
    fallback_reason: str | None = None


def resolve_torch_device(requested: str) -> DeviceInfo:
    requested_norm = requested.lower().strip()

    import torch

    if requested_norm == "auto":
        if torch.cuda.is_available():
            return DeviceInfo(requested=requested_norm, selected="cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return DeviceInfo(requested=requested_norm, selected="mps")
        return DeviceInfo(requested=requested_norm, selected="cpu")

    if requested_norm == "cuda":
        if torch.cuda.is_available():
            return DeviceInfo(requested=requested_norm, selected="cuda")
        return DeviceInfo(
            requested=requested_norm,
            selected="cpu",
            fallback_reason="CUDA requested but no CUDA device is available.",
        )

    if requested_norm == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return DeviceInfo(requested=requested_norm, selected="mps")
        return DeviceInfo(
            requested=requested_norm,
            selected="cpu",
            fallback_reason="MPS requested but Apple Metal backend is unavailable.",
        )

    if requested_norm == "cpu":
        return DeviceInfo(requested=requested_norm, selected="cpu")

    raise ValueError("train.device must be one of auto|cuda|mps|cpu.")

