import hashlib
import math

import torch

REC2020_Y = (0.2627, 0.6780, 0.0593)
_EPS = 1e-8


def positive(img: torch.Tensor) -> torch.Tensor:
    """NaN/±Inf → 0, negatives → 0. Every glow computation starts here."""
    return torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def luminance(img: torch.Tensor) -> torch.Tensor:
    w = torch.tensor(REC2020_Y, dtype=img.dtype, device=img.device).view(3, 1, 1)
    return (img * w).sum(0)


def smoothstep(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    if hi <= lo:
        return (x >= lo).to(x.dtype)
    t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def extract_energy(
    img: torch.Tensor, threshold: float, knee: float, referred: str
) -> torch.Tensor:
    """Soft-knee highlight extraction; keeps each source's colour."""
    img = positive(img)
    y = luminance(img)
    if referred == "scene":
        x = torch.log2(y.clamp_min(_EPS))
    elif referred == "display":
        x = y
    else:
        raise ValueError(f"referred must be 'scene' or 'display', got {referred!r}")
    return img * smoothstep(x, threshold - knee, threshold + knee)


def mask_energy(
    energy: torch.Tensor, mask: torch.Tensor, invert: bool = False
) -> torch.Tensor:
    """Kazuki source mask (spec §5.7): E' = M · E with M clamped to 0…1 (NaN/±Inf → 0),
    then 1 − M when inverted. Only masked areas emit; the glow still spreads freely."""
    m = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    if invert:
        m = 1.0 - m
    return energy * m.to(energy.dtype)[None]


def mask_key(mask: torch.Tensor) -> str:
    """Content key for a source mask (caches must see a changed mask)."""
    data = mask.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha1(data, usedforsecurity=False).hexdigest()


def estimate_clip_level(img: torch.Tensor) -> float:
    """Clip level from the pile-up at the top of the max-channel histogram (full-res input)."""
    m = positive(img).amax(0)
    top = float(m.max())
    if top <= 0:
        return math.inf
    # A real clip is a plateau: clipped pixels share (almost) exactly the max value,
    # while a smooth distribution has almost nothing that close to its top.
    if float((m >= top * (1 - 1e-5)).float().mean()) > 1e-4:
        return 0.999 * top
    return math.inf
