"""How kazuki lays its glow onto the image (spec §5.6).

`additive` and `conserve` work on scene-linear light directly. The other modes are
Clip Studio Paint's layer modes (plus `overlay_glow`, overlay with the glow as a layer on a
transparent background): they work on sRGB-encoded values in 0…1, so
`scene_blend` maps a chosen white (2^white_ev) to 1, encodes, blends, decodes, and
passes light above white (and negative values) through untouched. Each function here
has a 1:1 counterpart in the darktable port (`src/common/ekvido/blend.c`).
"""

import torch

from ekvido.common.display import srgb_eotf, srgb_oetf

BLEND_MODES = (
    "additive",
    "conserve",
    "screen",
    "overlay",
    "color_dodge",
    "glow_dodge",
    "overlay_glow",
)  # append only: the index is stored in darktable edit histories
CSP_MODES = BLEND_MODES[2:]
GLOW_EPS = 1e-6  # overlay_glow: layer opacity m at or below this counts as no glow


def dodge(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """CSP colour dodge D(A, C): 0 if A = 0, else 1 if C ≥ 1, else min(1, A / (1 − C))."""
    one = torch.ones_like(a)
    q = torch.where(c >= 1.0, one, torch.minimum(one, a / (1.0 - c).clamp_min(1e-12)))
    return torch.where(a <= 0.0, torch.zeros_like(a), q)


def _overlay(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.where(a <= 0.5, 2.0 * a * c, 1.0 - 2.0 * (1.0 - a) * (1.0 - c))


def overlay_glow(a: torch.Tensor, c: torch.Tensor, alpha) -> torch.Tensor:
    """Overlay of a layer on a transparent background (spec §5.6 amendment): per pixel
    m = max(C_r, C_g, C_b) is the layer's opacity and C/m its colour; channels on dim 0.
    result = A + α·m·(O(A, C/m) − A), and exactly A where m ≤ GLOW_EPS."""
    m = c.amax(0, keepdim=True)
    colour = (c / m.clamp_min(GLOW_EPS)).clamp_max(1.0)
    return torch.where(m > GLOW_EPS, a + alpha * m * (_overlay(a, colour) - a), a)


def csp_blend(
    mode: str, a: torch.Tensor, c: torch.Tensor, alpha: float | torch.Tensor
) -> torch.Tensor:
    """Spec §5.6 table, on display-encoded values in 0…1 (A background, C layer;
    overlay_glow needs the colour channels on dim 0)."""
    if mode == "screen":
        return a + alpha * ((1.0 - (1.0 - a) * (1.0 - c)) - a)
    if mode == "overlay":
        return a + alpha * (_overlay(a, c) - a)
    if mode == "color_dodge":
        return a + alpha * (dodge(a, c) - a)
    if mode == "glow_dodge":
        return dodge(a, alpha * c)
    if mode == "overlay_glow":
        return overlay_glow(a, c, alpha)
    raise ValueError(f"not a CSP blend mode: {mode!r}")


def scene_blend(
    img: torch.Tensor, glow: torch.Tensor, mode: str, strength: float, white_ev: float
) -> torch.Tensor:
    """A CSP mode on scene-linear light, white = 2^white_ev (spec §5.6 steps 1–4)."""
    alpha = min(strength, 1.0)
    if alpha == 0.0:
        return img
    w = 2.0**white_ev
    a = img / w
    a_c = a.clamp(0.0, 1.0)
    excess = a - a_c  # light above white and negative values pass through untouched
    big_a = srgb_oetf(a_c)
    big_c = srgb_oetf((glow / w).clamp_max(1.0))  # glow above white saturates
    out = w * (srgb_eotf(csp_blend(mode, big_a, big_c, alpha)) + excess)
    if mode == "overlay_glow":  # glow-free pixels stay exactly as they were
        out = torch.where(big_c.amax(0, keepdim=True) > GLOW_EPS, out, img)
    return out


def blend_glow(
    img: torch.Tensor,
    glow: torch.Tensor,
    energy: torch.Tensor,
    mode: str,
    strength: float,
    white_ev: float,
) -> torch.Tensor:
    """Kazuki's output: `img` with the glow (before strength) laid on by `mode`."""
    if mode == "additive":
        return img + strength * glow
    if mode == "conserve":
        return img + min(strength, 1.0) * (glow - energy)
    if mode in CSP_MODES:
        return scene_blend(img, glow, mode, strength, white_ev)
    raise ValueError(f"unknown blend mode {mode!r}; known: {BLEND_MODES}")
