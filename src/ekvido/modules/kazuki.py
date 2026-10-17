from dataclasses import dataclass

import torch

from ekvido.common.blend import BLEND_MODES, blend_glow
from ekvido.common.color import chromaticity, unit_luminance, warmth
from ekvido.common.convolve import Convolver
from ekvido.common.energy import (
    extract_energy,
    luminance,
    mask_energy,
    mask_key,
    positive,
    smoothstep,
)
from ekvido.common.psf import build_kernel, kernel_half_cap, split_bands
from ekvido.common.wblur import gaussian_blur, weighted_blur
from ekvido.modules.base import Context, Module, kernel_spec, param, register

Color = tuple[float, float, float]


@dataclass(frozen=True)
class KazukiParams:
    strength: float = param(0.5, min=0.0, max=4.0, desc="glow amount")
    blend: str = param(
        "additive",
        kind="choice",
        choices=BLEND_MODES,
        desc="additive adds light; conserve redistributes it; screen, overlay, "
        "color_dodge, glow_dodge: Clip Studio Paint layer modes; overlay_glow: "
        "overlay that leaves glow-free areas alone",
    )
    blend_white_ev: float = param(
        0.0,
        min=-4.0,
        max=8.0,
        desc="white for the CSP modes (EV, 0 = 1.0); light above it passes through",
    )
    threshold_ev: float = param(
        0.0, min=-6.0, max=6.0, mode="scene", desc="glow threshold (EV, 0 = 1.0)"
    )
    knee_ev: float = param(
        1.0, min=0.0, max=4.0, mode="scene", desc="threshold softness (EV)"
    )
    threshold_lin: float = param(
        0.85, min=0.0, max=1.0, mode="display", desc="glow threshold (0–1)"
    )
    knee_lin: float = param(
        0.1, min=0.0, max=0.5, mode="display", desc="threshold softness (0–1)"
    )
    hue_radius: float = param(
        0.01,
        min=0.001,
        max=0.1,
        desc="area a source's colour is read from (diagonal fraction)",
    )
    neutral_point: float = param(
        0.0, min=-0.3, max=0.3, desc="chromaticity (r−b) that counts as neutral"
    )
    warmth_softness: float = param(
        0.05, min=0.005, max=0.3, desc="how gradually sources turn warm/cool"
    )
    neutral_bias: float = param(
        0.8,
        min=0.0,
        max=1.0,
        desc="warmth for clipped sources with no fringe (1 = warm)",
    )
    push_warm: float = param(
        0.7, min=0.0, max=1.0, desc="how far warm glows move to the warm ramp"
    )
    push_cool: float = param(
        0.7, min=0.0, max=1.0, desc="how far cool glows move to the cool ramp"
    )
    warm_core: Color = param(
        (1.0, 0.85, 0.55), kind="color", desc="warm ramp, inner (pale gold)"
    )
    warm_edge: Color = param(
        (1.0, 0.45, 0.12), kind="color", desc="warm ramp, outer (orange)"
    )
    cool_core: Color = param(
        (0.35, 0.85, 1.0), kind="color", desc="cool ramp, inner (cyan)"
    )
    cool_edge: Color = param(
        (0.08, 0.2, 1.0), kind="color", desc="cool ramp, outer (deep blue)"
    )
    band_split: float = param(
        0.006,
        min=0.0,
        max=0.05,
        desc="inner/outer colour band radius (diagonal fraction)",
    )
    halo_radius: float = param(
        0.006, min=0.0005, max=0.05, desc="halo core radius (diagonal fraction)"
    )
    halo_beta: float = param(1.3, min=0.6, max=4.0, desc="halo tail steepness")
    halo_stretch_x: float = param(
        1.85, min=0.25, max=5.0, desc="halo horizontal stretch"
    )
    halo_stretch_y: float = param(1.0, min=0.25, max=5.0, desc="halo vertical stretch")
    veil_radius: float = param(
        0.1, min=0.02, max=0.25, desc="veil radius (diagonal fraction)"
    )
    veil_beta: float = param(1.2, min=0.6, max=3.0, desc="veil tail steepness")
    w_halo: float = param(1.0, min=0.0, max=1.0, desc="halo share")
    w_veil: float = param(0.0, min=0.0, max=1.0, desc="veil share")
    arms: int = param(
        4, kind="int", min=0, max=8, desc="number of glint arms (4 = cross)"
    )
    arm_angle: float = param(0.0, min=0.0, max=180.0, desc="first arm angle (degrees)")
    arm_radius: float = param(
        0.001, min=0.0001, max=0.02, desc="arm core size (diagonal fraction)"
    )
    arm_beta: float = param(
        0.96, min=0.6, max=4.0, desc="arm fade (low: long arms on bright sources)"
    )
    arm_sharpness: float = param(
        0.32, min=0.2, max=2.0, desc="star shape: 2 round, 1 diamond, <1 thin arms"
    )
    arm_stretch_x: float = param(
        2.93, min=0.25, max=5.0, desc="star horizontal stretch"
    )
    arm_stretch_y: float = param(0.58, min=0.25, max=5.0, desc="star vertical stretch")
    w_arms: float = param(
        0.2, min=0.0, max=1.0, desc="arms share (low: only hot points show arms)"
    )


def _warmth_map(
    src: torch.Tensor, p: KazukiParams, ctx: Context, y_floor: float
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma = p.hue_radius * ctx.diag_px
    y = luminance(src)
    clipped = src.amax(0) >= ctx.clip_level
    w = y * (~clipped)
    hue, den = weighted_blur(chromaticity(src), w, sigma)
    # Confidence = trustworthy (unclipped) light vs clipped area, each clipped pixel
    # counted at the glow threshold's luminance — not at its (unknown, possibly huge)
    # clipped value, which would drown a dim but real fringe.
    clipped_mass = gaussian_blur(clipped[None].to(src.dtype), sigma)[0] * y_floor
    conf = den / (den + clipped_mass + 1e-12)
    a = smoothstep(conf, 0.15, 0.4)
    t = a * warmth(hue, p.neutral_point, p.warmth_softness) + (1.0 - a) * p.neutral_bias
    return t, clipped


def _mask_key(ctx: Context):
    if ctx.source_mask is None:
        return None
    key = ctx.source_mask_key or mask_key(ctx.source_mask)
    return (key, ctx.source_mask_invert)


def _basis(e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    y = luminance(e)
    return torch.cat([e, t * e, y[None], (t * y)[None], (t * t * y)[None]])


def _combine(
    g: torch.Tensor, warm: Color, cool: Color, p: KazukiParams
) -> torch.Tensor:
    cu = unit_luminance(cool).to(g.device).view(3, 1, 1)
    du = unit_luminance(warm).to(g.device).view(3, 1, 1) - cu
    pc, dp = p.push_cool, p.push_warm - p.push_cool
    g_e, g_te, g_y, g_ty, g_t2y = g[0:3], g[3:6], g[6:7], g[7:8], g[8:9]
    return (
        (1 - pc) * g_e
        - dp * g_te
        + cu * (pc * g_y + dp * g_ty)
        + du * (pc * g_ty + dp * g_t2y)
    )


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
class Kazuki(Module):
    NAME = "kazuki"
    VERSION = 2

    @staticmethod
    def legacy_params(old_version: int, old: dict) -> dict:
        if old_version == 1:
            return _arms_v1_to_v2(old)
        raise ValueError(f"no migration from version {old_version}")

    Params = KazukiParams

    def process(self, img: torch.Tensor, p: KazukiParams, ctx: Context) -> torch.Tensor:
        src = positive(img)
        if ctx.referred == "scene":
            thr, knee = p.threshold_ev, p.knee_ev
        else:
            thr, knee = p.threshold_lin, p.knee_lin
        y_floor = 2.0**thr if ctx.referred == "scene" else thr  # threshold luminance
        mask = ctx.source_mask
        if mask is not None and tuple(mask.shape) != tuple(img.shape[1:]):
            raise ValueError(
                f"source mask is {tuple(mask.shape)}, image is {tuple(img.shape[1:])}"
            )
        e_key = (ctx.input_key, ctx.referred, thr, knee, _mask_key(ctx))

        def energy_map():
            e = extract_energy(src, thr, knee, ctx.referred)
            return e if mask is None else mask_energy(e, mask, ctx.source_mask_invert)

        energy = ctx.memo("energy", e_key, energy_map)
        h_key = (
            ctx.input_key,
            p.hue_radius,
            p.neutral_point,
            p.warmth_softness,
            p.neutral_bias,
            ctx.clip_level,
            ctx.diag_px,
            y_floor,
        )
        t, clipped = ctx.memo(
            "warmth", h_key, lambda: _warmth_map(src, p, ctx, y_floor)
        )
        b_key = e_key + h_key
        basis = ctx.memo("basis", b_key, lambda: _basis(energy, t))
        pad = kernel_half_cap(ctx.diag_px)
        conv = ctx.memo(
            "source", b_key + (pad,), lambda: Convolver(basis, pad, ctx.keep_spectra)
        )
        spec = kernel_spec(p)
        k_key = (spec, p.band_split, ctx.diag_px)

        def bands():
            kernel = build_kernel(spec, ctx.diag_px, img.device)
            return kernel, *split_bands(kernel, p.band_split * ctx.diag_px)

        kernel, inner, outer = ctx.memo("kernel", k_key, bands)
        if ctx.cache is None:  # full render: never hold both 9-channel band results
            glow = _combine(conv.apply(inner), p.warm_core, p.cool_core, p)
            glow += _combine(conv.apply(outer), p.warm_edge, p.cool_edge, p)
        else:
            g_in, g_out = ctx.memo(
                "glow_basis",
                b_key + k_key,
                lambda: (conv.apply(inner), conv.apply(outer)),
            )
            glow = _combine(g_in, p.warm_core, p.cool_core, p) + _combine(
                g_out, p.warm_edge, p.cool_edge, p
            )
        if ctx.stash is not None:
            ctx.stash.update(
                energy=energy, warmth=t, clip=clipped, glow=glow, kernel=kernel
            )
        return blend_glow(img, glow, energy, p.blend, p.strength, p.blend_white_ev)
