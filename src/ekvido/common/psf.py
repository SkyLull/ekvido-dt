import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ekvido.common.energy import smoothstep

MAX_KERNEL_FRAC = 0.25  # kernel half-size cap, as a fraction of the diagonal
_FLOOR = 1e-6  # parts are cut where they fall below this fraction of their peak


@dataclass(frozen=True)
class KernelSpec:
    halo_radius: float = 0.004
    halo_beta: float = 1.5
    halo_stretch_x: float = 1.0
    halo_stretch_y: float = 1.0
    veil_radius: float = 0.15
    veil_beta: float = 1.0
    arms: int = 0
    arm_angle: float = 0.0
    arm_radius: float = 0.001
    arm_beta: float = 1.5
    arm_sharpness: float = 0.4
    arm_stretch_x: float = 1.0
    arm_stretch_y: float = 1.0
    w_halo: float = 1.0
    w_veil: float = 0.0
    w_arms: float = 0.0


def kernel_half_cap(diag_px: float) -> int:
    return max(1, math.ceil(MAX_KERNEL_FRAC * diag_px))


def _moffat_extent(r0: float, beta: float) -> float:
    return r0 * math.sqrt(_FLOOR ** (-1.0 / beta) - 1.0)


def kernel_half_size(spec: KernelSpec, diag_px: float) -> int:
    ext = 0.0
    if spec.w_halo > 0:
        stretch = max(spec.halo_stretch_x, spec.halo_stretch_y)
        ext = max(ext, _moffat_extent(spec.halo_radius, spec.halo_beta) * stretch)
    if spec.w_veil > 0:
        ext = max(ext, _moffat_extent(spec.veil_radius, spec.veil_beta))
    if spec.w_arms > 0 and spec.arms > 0:
        stretch = max(spec.arm_stretch_x, spec.arm_stretch_y)
        ext = max(ext, _moffat_extent(spec.arm_radius, spec.arm_beta) * stretch)
    return int(min(kernel_half_cap(diag_px), max(1, math.ceil(ext * diag_px))))


def _grid(r: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    ax = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(ax, ax, indexing="ij")
    return x, y


def _normalise(k: torch.Tensor) -> torch.Tensor:
    s = k.sum()
    return k / s if s > 0 else k


def _moffat(
    x, y, r0_px: float, beta: float, sx: float = 1.0, sy: float = 1.0
) -> torch.Tensor:
    r0 = max(r0_px, 0.5)  # below half a pixel the halo is a point anyway
    r2 = (x / sx) ** 2 + (y / sy) ** 2
    return (1.0 + r2 / (r0 * r0)) ** (-beta)


def _star(
    x,
    y,
    n: int,
    angle_deg: float,
    r0_px: float,
    beta: float,
    sharpness: float,
    sx: float,
    sy: float,
) -> torch.Tensor:
    """Moffat falloff over an L_p distance: p = 2 round, 1 diamond, < 1 an n-armed star.

    d = (|u|^p + |v|^p)^(1/p) with the angle folded by n/4 (exact L_p for n = 4), and
    k = (1 + (d/r0)^2)^-beta: a power-law tail, so arms fade without an end point and a
    brighter source shows longer arms.
    """
    r0, p = max(r0_px, 0.5), max(sharpness, 0.05)
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    xs, ys = (x * c + y * s) / sx, (-x * s + y * c) / sy
    rad = torch.sqrt(xs * xs + ys * ys) / r0
    fold = torch.atan2(ys, xs) * (n / 4.0)
    # (d/r0)^2 from p-th powers of O(R/r0) numbers: no huge intermediate d.
    sum_p = (rad * torch.cos(fold).abs()) ** p + (rad * torch.sin(fold).abs()) ** p
    k = (1.0 + sum_p ** (2.0 / p)) ** (-beta)
    # Half-pixel blur: needle-thin arms (small p) would otherwise alias on the grid.
    g = torch.exp(
        -0.5 * (torch.arange(-2, 3, dtype=k.dtype, device=k.device) / 0.5) ** 2
    )
    g = g / g.sum()
    k = F.conv2d(k[None, None], g.view(1, 1, 1, 5), padding=(0, 2))
    return F.conv2d(k, g.view(1, 1, 5, 1), padding=(2, 0))[0, 0]


def build_kernel(spec: KernelSpec, diag_px: float, device=None) -> torch.Tensor:
    r = kernel_half_size(spec, diag_px)
    x, y = _grid(r, device)
    # Fade every part to zero over the outer 20% of the support, inside a disc: the
    # half-size cap would otherwise cut long tails as a visible hard-edged square.
    taper = 1.0 - smoothstep(torch.sqrt(x * x + y * y), 0.8 * r, float(r))
    parts: list[tuple[float, torch.Tensor]] = []
    if spec.w_halo > 0:
        halo = _moffat(
            x,
            y,
            spec.halo_radius * diag_px,
            spec.halo_beta,
            spec.halo_stretch_x,
            spec.halo_stretch_y,
        )
        parts.append((spec.w_halo, _normalise(halo * taper)))
    if spec.w_veil > 0:
        parts.append(
            (
                spec.w_veil,
                _normalise(
                    _moffat(x, y, spec.veil_radius * diag_px, spec.veil_beta) * taper
                ),
            )
        )
    if spec.w_arms > 0 and spec.arms > 0:
        arms = _star(
            x,
            y,
            spec.arms,
            spec.arm_angle,
            spec.arm_radius * diag_px,
            spec.arm_beta,
            spec.arm_sharpness,
            spec.arm_stretch_x,
            spec.arm_stretch_y,
        )
        parts.append((spec.w_arms, _normalise(arms * taper)))
    if not parts:
        k = torch.zeros_like(x)
        k[r, r] = 1.0
        return k
    total = sum(w for w, _ in parts)
    return sum(w * p for w, p in parts) / total


def split_bands(
    kernel: torch.Tensor, r_split_px: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inner/outer radial bands with a soft ±20% edge; inner + outer == kernel."""
    r = (kernel.shape[0] - 1) // 2
    x, y = _grid(r, kernel.device)
    rad = torch.sqrt(x * x + y * y)
    if r_split_px <= 0:
        inner = torch.zeros_like(kernel)
    else:
        inner = kernel * (1.0 - smoothstep(rad, 0.8 * r_split_px, 1.2 * r_split_px))
    return inner, kernel - inner
