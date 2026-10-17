import torch

from ekvido.common.energy import positive

# Linear Rec.2020 → linear sRGB/Rec.709 (D65).
REC2020_TO_SRGB = torch.tensor(
    [
        [1.6605, -0.5876, -0.0728],
        [-0.1246, 1.1329, -0.0083],
        [-0.0182, -0.1006, 1.1187],
    ]
)
_WHITE = 16.0


def tone_curve(x: torch.Tensor) -> torch.Tensor:
    """Extended Reinhard: smooth highlight roll-off, 16.0 maps to 1.0."""
    x = x.clamp_min(0.0)
    return (x * (1.0 + x / (_WHITE * _WHITE)) / (1.0 + x)).clamp(0.0, 1.0)


def srgb_oetf(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.pow(1.0 / 2.4) - 0.055)


def srgb_eotf(v: torch.Tensor) -> torch.Tensor:
    """Inverse of `srgb_oetf`: sRGB-encoded 0…1 → linear 0…1."""
    v = v.clamp(0.0, 1.0)
    return torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055).pow(2.4))


def to_display(
    img: torch.Tensor, referred: str, exposure_ev: float = 0.0
) -> torch.Tensor:
    if referred not in ("scene", "display"):
        raise ValueError(f"referred must be 'scene' or 'display', got {referred!r}")
    x = positive(img) * (2.0**exposure_ev)
    m = REC2020_TO_SRGB.to(x.device, x.dtype)
    x = torch.einsum("ij,jhw->ihw", m, x).clamp_min(0.0)
    x = tone_curve(x) if referred == "scene" else x.clamp(0.0, 1.0)
    return (srgb_oetf(x) * 255.0 + 0.5).to(torch.uint8).permute(1, 2, 0).contiguous()


_COOL = torch.tensor([40.0, 90.0, 255.0])
_MID = torch.tensor([128.0, 128.0, 128.0])
_WARM = torch.tensor([255.0, 140.0, 30.0])


def colormap_warmth(t: torch.Tensor) -> torch.Tensor:
    t = t.clamp(0.0, 1.0)[..., None]
    cool, mid, warm = (c.to(t.device) for c in (_COOL, _MID, _WARM))
    lo = cool + (mid - cool) * (t * 2.0)
    hi = mid + (warm - mid) * (t * 2.0 - 1.0)
    return torch.where(t <= 0.5, lo, hi).round().to(torch.uint8)


def mask_overlay(base: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = base.clone()
    out[mask] = torch.tensor([255, 0, 0], dtype=torch.uint8, device=base.device)
    return out


def kernel_image(kernel: torch.Tensor) -> torch.Tensor:
    k = kernel.clamp_min(0.0)
    peak = k.max().clamp_min(1e-30)
    v = ((torch.log10((k / peak).clamp_min(1e-6)) + 6.0) / 6.0).clamp(0.0, 1.0)
    g = (v * 255.0 + 0.5).to(torch.uint8)
    return g[..., None].expand(*g.shape, 3).contiguous()
