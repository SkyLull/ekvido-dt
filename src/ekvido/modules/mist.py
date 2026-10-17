from dataclasses import dataclass

import torch

from ekvido.common.convolve import Convolver
from ekvido.common.energy import positive
from ekvido.common.psf import build_kernel, kernel_half_cap
from ekvido.modules.base import Context, Module, kernel_spec, param, register


@dataclass(frozen=True)
class MistParams:
    strength: float = param(
        0.15, min=0.0, max=1.0, desc="fraction of all light that scatters"
    )
    halo_radius: float = param(
        0.004, min=0.0005, max=0.05, desc="halo core radius (diagonal fraction)"
    )
    halo_beta: float = param(
        1.5, min=0.6, max=4.0, desc="halo tail steepness (low = long misty tail)"
    )
    halo_stretch_x: float = param(
        1.0, min=0.25, max=5.0, desc="halo horizontal stretch"
    )
    halo_stretch_y: float = param(1.0, min=0.25, max=5.0, desc="halo vertical stretch")
    veil_radius: float = param(
        0.15, min=0.02, max=0.25, desc="veil radius (diagonal fraction)"
    )
    veil_beta: float = param(1.0, min=0.6, max=3.0, desc="veil tail steepness")
    w_halo: float = param(0.5, min=0.0, max=1.0, desc="halo share (black mist: high)")
    w_veil: float = param(0.5, min=0.0, max=1.0, desc="veil share (white mist: high)")
    arms: int = param(
        0, kind="int", min=0, max=8, desc="number of star arms (0 = none)"
    )
    arm_angle: float = param(0.0, min=0.0, max=180.0, desc="first arm angle (degrees)")
    arm_radius: float = param(
        0.001, min=0.0001, max=0.02, desc="arm core size (diagonal fraction)"
    )
    arm_beta: float = param(
        1.3, min=0.6, max=4.0, desc="arm fade (low: long arms on bright sources)"
    )
    arm_sharpness: float = param(
        0.3, min=0.2, max=2.0, desc="star shape: 2 round, 1 diamond, <1 thin arms"
    )
    arm_stretch_x: float = param(1.0, min=0.25, max=5.0, desc="star horizontal stretch")
    arm_stretch_y: float = param(1.0, min=0.25, max=5.0, desc="star vertical stretch")
    w_arms: float = param(0.0, min=0.0, max=1.0, desc="arms share")


def _arms_v1_to_v2(old: dict) -> dict:
    """v2 replaced stick arms (length/width/falloff) with an L_p star (see psf._star).

    The core size comes from the old arm width; length and falloff have no
    counterpart (a power-law star has no end point), so they take the new defaults.
    """
    new = {k: v for k, v in old.items() if k not in ("arm_length", "arm_falloff")}
    if "arm_width" in new:
        new["arm_radius"] = 2.0 * float(new.pop("arm_width"))
    return new


@register
class Mist(Module):
    NAME = "mist"
    VERSION = 2

    @staticmethod
    def legacy_params(old_version: int, old: dict) -> dict:
        if old_version == 1:
            return _arms_v1_to_v2(old)
        raise ValueError(f"no migration from version {old_version}")

    Params = MistParams

    def process(self, img: torch.Tensor, p: MistParams, ctx: Context) -> torch.Tensor:
        spec = kernel_spec(p)
        pad = kernel_half_cap(ctx.diag_px)
        conv = ctx.memo(
            "source",
            (ctx.input_key, pad),
            lambda: Convolver(positive(img), pad, ctx.keep_spectra),
        )
        kernel = ctx.memo(
            "kernel",
            (spec, ctx.diag_px),
            lambda: build_kernel(spec, ctx.diag_px, img.device),
        )
        glow = ctx.memo(
            "glow", (ctx.input_key, pad, spec, ctx.diag_px), lambda: conv.apply(kernel)
        )
        # Normalised convolution: divide by how much of the kernel fell inside the
        # frame, so edges aren't darkened by light scattering out with none scattering in.
        _, h, w = img.shape

        def coverage():
            ones = torch.ones(1, h, w, dtype=img.dtype, device=img.device)
            return Convolver(ones, pad, keep_spectra=False).apply(kernel)[0]

        cover = ctx.memo("coverage", (h, w, pad, spec, ctx.diag_px), coverage)
        glow = glow / cover.clamp_min(1e-6)
        if ctx.stash is not None:
            ctx.stash["glow"] = glow
            ctx.stash["kernel"] = kernel
        k = p.strength
        if k == 0:
            return img
        return (1.0 - k) * img + k * glow
