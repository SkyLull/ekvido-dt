import colorsys

import torch

from ekvido.common.energy import REC2020_Y, positive, smoothstep


def chromaticity(img: torch.Tensor) -> torch.Tensor:
    img = positive(img)
    s = img.sum(0, keepdim=True)
    neutral = torch.full_like(img, 1.0 / 3.0)
    return torch.where(s > 1e-12, img / s.clamp_min(1e-12), neutral)


def warmth(chroma: torch.Tensor, neutral_point: float, softness: float) -> torch.Tensor:
    """0 = cool, 1 = warm, from (c_r − c_b) relative to `neutral_point`."""
    d = chroma[0] - chroma[2] - neutral_point
    return smoothstep(d, -softness, softness)


def unit_luminance(rgb: tuple[float, float, float]) -> torch.Tensor:
    c = torch.tensor(rgb, dtype=torch.float32)
    y = float((c * torch.tensor(REC2020_Y)).sum())
    return c / max(y, 1e-8)


def hsv_to_rgb(hue_deg: float, sat: float) -> tuple[float, float, float]:
    """Colour from hue (degrees) and saturation at V = 1, as the darktable port's colour
    sliders store it (spec §5.2); C mirror: `ekvido_hsv_to_rgb`."""
    return colorsys.hsv_to_rgb((hue_deg % 360.0) / 360.0, sat, 1.0)
