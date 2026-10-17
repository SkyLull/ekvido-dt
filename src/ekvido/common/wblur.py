import math

import torch
import torch.nn.functional as F

_DIRECT_MAX_SIGMA = 8.0


def _gauss1d(sigma: float, device, dtype) -> torch.Tensor:
    r = max(1, math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=dtype, device=device)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    return g / g.sum()


def gaussian_blur(x: torch.Tensor, sigma_px: float) -> torch.Tensor:
    if sigma_px <= 0:
        return x
    c, h, w = x.shape
    if sigma_px > _DIRECT_MAX_SIGMA:
        # Box-downsample by an integer factor (replicate-padded to a multiple of f, so
        # avg_pool works on every device incl. MPS), blur small, upsample back.
        f = int(sigma_px // 4)
        ph, pw = -h % f, -w % f
        padded = F.pad(x[None], (0, pw, 0, ph), mode="replicate")
        small = F.avg_pool2d(padded, f)[0]
        blurred = gaussian_blur(small, sigma_px / f)
        up = F.interpolate(
            blurred[None], size=(h + ph, w + pw), mode="bilinear", align_corners=False
        )
        return up[0, :, :h, :w]
    g = _gauss1d(sigma_px, x.device, x.dtype)
    r = (g.numel() - 1) // 2
    y = F.pad(x[None], (r, r, 0, 0), mode="replicate")
    y = F.conv2d(y, g.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    y = F.pad(y, (0, 0, r, r), mode="replicate")
    return F.conv2d(y, g.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)[0]


def weighted_blur(
    x: torch.Tensor, w: torch.Tensor, sigma_px: float
) -> tuple[torch.Tensor, torch.Tensor]:
    num = gaussian_blur(x * w[None], sigma_px)
    den = gaussian_blur(w[None], sigma_px)[0]
    return num / den.clamp_min(1e-12)[None], den
