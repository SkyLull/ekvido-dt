import os

import torch

_KNOWN = {"cuda", "mps", "cpu"}


def pick_device(prefer: str | None = None) -> torch.device:
    """cuda → mps → cpu, unless `prefer` or $EKVIDO_DEVICE names one."""
    choice = prefer or os.environ.get("EKVIDO_DEVICE")
    if choice:
        if choice not in _KNOWN:
            raise ValueError(
                f"unknown device {choice!r}; expected one of {sorted(_KNOWN)}"
            )
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Block until queued GPU work finishes (for honest timings)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
