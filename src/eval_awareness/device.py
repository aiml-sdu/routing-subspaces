"""Device detection utility — works on CUDA, MPS, and CPU."""

import torch


def get_device() -> tuple[str, torch.dtype]:
    """Return (device_string, dtype) for the best available backend.

    Priority: CUDA > MPS > CPU.
    """
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def empty_cache():
    """Clear GPU cache for whichever backend is active."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def seed_everything(seed: int = 42):
    """Set seeds for all backends."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
