"""Write the C-core oracle fixtures for the darktable port.

Usage: uv run python tools/make_c_fixtures.py darktable/src/common/ekvido/tests/fixtures
"""

import colorsys
import dataclasses
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from ekvido.common.blend import BLEND_MODES, blend_glow
from ekvido.common.color import hsv_to_rgb
from ekvido.common.convolve import fft_convolve
from ekvido.common.display import kernel_image
from ekvido.common.energy import estimate_clip_level, extract_energy, mask_energy
from ekvido.common.psf import KernelSpec, build_kernel, split_bands
from ekvido.common.wblur import gaussian_blur, weighted_blur
from ekvido.modules.base import Context, kernel_spec
from ekvido.modules.kazuki import Kazuki, KazukiParams, _warmth_map
from ekvido.modules.mist import Mist, MistParams
from ekvido.presets import load_preset, resolve_preset

SPEC_KEYS = [f.name for f in dataclasses.fields(KernelSpec)]

PSF_CASES = {
    "halo_default": (400, KernelSpec()),
    "halo_stretch": (
        400,
        KernelSpec(
            halo_radius=0.01,
            halo_beta=1.2,
            halo_stretch_x=2.5,
            halo_stretch_y=0.7,
        ),
    ),
    "halo_veil": (
        400,
        KernelSpec(w_halo=0.5, w_veil=0.5, veil_radius=0.15, veil_beta=1.0),
    ),
    "star_cross": (
        400,
        KernelSpec(
            w_halo=0.0,
            w_arms=1.0,
            arms=4,
            arm_radius=0.002,
            arm_beta=1.3,
            arm_sharpness=0.3,
        ),
    ),
    "star_three_rot": (
        400,
        KernelSpec(
            w_halo=0.0,
            w_arms=1.0,
            arms=3,
            arm_angle=25.0,
            arm_radius=0.003,
            arm_beta=1.5,
            arm_sharpness=0.5,
        ),
    ),
    "star_round_stretch": (
        400,
        KernelSpec(
            w_halo=0.0,
            w_arms=1.0,
            arms=4,
            arm_radius=0.004,
            arm_beta=1.5,
            arm_sharpness=2.0,
            arm_stretch_x=2.0,
            arm_stretch_y=0.6,
        ),
    ),
    "kazuki_shape": (
        400,
        KernelSpec(
            halo_radius=0.006,
            halo_beta=1.3,
            halo_stretch_x=1.85,
            w_halo=1.0,
            w_veil=0.0,
            arms=4,
            arm_radius=0.001,
            arm_beta=0.96,
            arm_sharpness=0.32,
            arm_stretch_x=2.93,
            arm_stretch_y=0.58,
            w_arms=0.2,
        ),
    ),
    "nothing": (400, KernelSpec(w_halo=0.0)),
    "tiny": (
        50,
        KernelSpec(
            halo_radius=1e-6,
            arms=3,
            w_arms=1.0,
            arm_radius=1e-9,
            arm_sharpness=0.2,
        ),
    ),
}

MIST_CASES = {
    "mist_default": MistParams(),
    "mist_black_star": MistParams(
        strength=0.3,
        w_halo=0.85,
        w_veil=0.15,
        halo_beta=1.8,
        arms=4,
        arm_radius=0.004,
        w_arms=0.3,
        arm_stretch_x=1.5,
    ),
}


# Kazuki params as the darktable module stores them: colours as hue/saturation pairs.
COLOURS = ("warm_core", "warm_edge", "cool_core", "cool_edge")


def _rgb_to_hue_sat(rgb: tuple[float, float, float]) -> tuple[float, float]:
    h, s, _ = colorsys.rgb_to_hsv(*rgb)  # the default colours all have V = 1
    return h * 360.0, s


# Derived from the Python defaults, so there is one source of truth.
KAZUKI_HSV_DEFAULTS = {c: _rgb_to_hue_sat(getattr(KazukiParams(), c)) for c in COLOURS}
KAZUKI_KEYS = (
    "strength",
    "blend_white_ev",
    "threshold_ev",
    "knee_ev",
    "hue_radius",
    "neutral_point",
    "warmth_softness",
    "neutral_bias",
    "push_warm",
    "push_cool",
    "band_split",
)
# The shipped preset's full parameter set (colours are overridden per case via HSV).
KAZUKI_WIDE = dataclasses.asdict(load_preset(resolve_preset("kazuki-wide")).params)
# name: (param overrides, hsv overrides, clip level or None = estimate from the image)
KAZUKI_CASES = {
    "kazuki_default": ({}, {}, None),
    "kazuki_wide": (KAZUKI_WIDE, {}, math.inf),
    "kazuki_screen": (
        {
            "blend": "screen",
            "strength": 0.8,
            "blend_white_ev": 1.0,
            "hue_radius": 0.08,
            "push_warm": 0.5,
            "neutral_point": 0.05,
            "band_split": 0.02,
            "arms": 6,
            "arm_angle": 15.0,
        },
        {"warm_edge": (15.0, 0.9)},
        12.0,
    ),
    "kazuki_glow_dodge": (
        {
            "blend": "glow_dodge",
            "strength": 1.7,
            "blend_white_ev": -1.0,
            "threshold_ev": 1.0,
            "knee_ev": 0.5,
        },
        {},
        None,
    ),
    "kazuki_conserve": ({"blend": "conserve", "strength": 0.6}, {}, None),
    "kazuki_overlay_glow": ({"blend": "overlay_glow", "strength": 0.7}, {}, None),
}
BLUR_SIGMAS = {
    "blur_tiny": 0.7,
    "blur_mid": 3.0,
    "blur_edge8": 8.0,
    "blur_down3": 12.5,
    "blur_down10": 40.0,
}
# name: (mode, strength, white_ev)
BLEND_CASES = {
    "blend_additive": ("additive", 1.5, 0.0),
    "blend_conserve": ("conserve", 1.5, 0.0),
    "blend_screen": ("screen", 0.6, 0.0),
    "blend_screen_white": ("screen", 0.6, 2.5),
    "blend_overlay": ("overlay", 0.6, 0.0),
    "blend_overlay_zero": ("overlay", 0.0, 0.0),
    "blend_color_dodge": ("color_dodge", 0.6, -1.5),
    "blend_glow_dodge": ("glow_dodge", 1.7, 0.0),
    "blend_overlay_glow": ("overlay_glow", 0.6, 0.0),
    "blend_overlay_glow_white": ("overlay_glow", 1.4, 1.5),
}


def write_ekv(path: Path, arr) -> None:
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f4"))
    with open(path, "wb") as f:
        f.write(b"EKV1")
        f.write(struct.pack("<i", a.ndim))
        f.write(struct.pack(f"<{a.ndim}i", *a.shape))
        f.write(a.tobytes())


def read_ekv(path: Path) -> np.ndarray:
    b = Path(path).read_bytes()
    if b[:4] != b"EKV1":
        raise ValueError(f"{path}: not an EKV1 file")
    nd = struct.unpack_from("<i", b, 4)[0]
    shape = struct.unpack_from(f"<{nd}i", b, 8)
    return np.frombuffer(b, dtype="<f4", offset=8 + 4 * nd).reshape(shape)


def _spec_kv(spec: KernelSpec) -> str:
    return " ".join(f"{k}={getattr(spec, k)!r}" for k in SPEC_KEYS)


def _mist_image(h: int = 48, w: int = 64) -> torch.Tensor:
    rng = np.random.default_rng(7)
    img = 0.05 + 0.02 * rng.random((3, h, w), dtype=np.float32)
    img[:, 12, 20] = (20.0, 12.0, 5.0)
    img[:, 30, 45] = (4.0, 8.0, 16.0)
    img[:, 40:43, 8:11] = 30.0
    img[0, 5, 5] = np.nan
    img[1, 6, 60] = -0.5
    img[2, 44, 30] = np.inf
    return torch.from_numpy(img)


def _ctx(diag: float, clip: float, stash: dict | None = None) -> Context:
    return Context(
        device=torch.device("cpu"),
        referred="scene",
        diag_px=diag,
        clip_level=clip,
        input_key="fixture",
        cache=None,
        keep_spectra=False,
        stash=stash,
    )


def _kazuki_image(h: int = 71, w: int = 97) -> torch.Tensor:
    """Coloured sources, a clipped plateau with a warm fringe, and bad pixels."""
    rng = np.random.default_rng(11)
    img = 0.05 + 0.02 * rng.random((3, h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]

    def disc(y, x, r, rgb):
        m = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        img[:, m] = np.asarray(rgb, dtype=np.float32)[:, None]

    disc(35, 45, 5, (2.0, 1.0, 0.4))  # warm fringe ...
    disc(35, 45, 3, (30.0, 30.0, 30.0))  # ... around a clipped plateau
    disc(20, 25, 2, (8.0, 4.0, 1.2))  # orange
    disc(50, 70, 2, (1.5, 4.0, 10.0))  # blue
    disc(60, 15, 2, (5.0, 5.0, 5.0))  # neutral
    img[0, 5, 5] = np.nan
    img[1, 6, 60] = -0.5
    img[2, 65, 90] = np.inf
    return torch.from_numpy(img)


def _hsv_kv(hsv: dict) -> str:
    return " ".join(f"{c}_hue={hsv[c][0]!r} {c}_sat={hsv[c][1]!r}" for c in COLOURS)


def kazuki_params(over: dict, hsv_over: dict) -> tuple[KazukiParams, dict]:
    hsv = {**KAZUKI_HSV_DEFAULTS, **hsv_over}
    colours = {c: hsv_to_rgb(*hsv[c]) for c in COLOURS}
    return KazukiParams(**{**over, **colours}), hsv


# Source mask (spec §5.7): name: (param overrides, invert)
KAZUKI_MASKED_CASES = {
    "kazuki_masked": ({}, False),
    "kazuki_masked_inv_conserve": ({"blend": "conserve", "strength": 0.6}, True),
    "kazuki_masked_overlay_glow": ({"blend": "overlay_glow", "strength": 0.7}, False),
}


def _kazuki_mask(h: int = 71, w: int = 97) -> torch.Tensor:
    """1 left of x = 30, 0 right of x = 60, a ramp between (the plateau source sits at
    0.5, the blue one outside); one NaN and one 1.5 to check the clamp."""
    x = np.arange(w, dtype=np.float32)
    m = np.clip((60.0 - x) / 30.0, 0.0, 1.0)[None].repeat(h, 0)
    m[2, 2] = np.nan
    m[2, 3] = 1.5
    return torch.from_numpy(np.ascontiguousarray(m))


def _write_kazuki(out: Path, lines: list[str]) -> None:
    img = _kazuki_image()
    _, h, w = img.shape
    diag = math.hypot(h, w)
    for name, (over, hsv_over, clip) in KAZUKI_CASES.items():
        p, hsv = kazuki_params(over, hsv_over)
        level = estimate_clip_level(img) if clip is None else clip
        stash: dict = {}
        result = Kazuki().process(img, p, _ctx(diag, level, stash))
        write_ekv(out / f"{name}.out.ekv", result)
        write_ekv(out / f"{name}.glow.ekv", stash["glow"])
        write_ekv(out / f"{name}.energy.ekv", stash["energy"])
        keys = " ".join(f"{k}={getattr(p, k)!r}" for k in KAZUKI_KEYS)
        lines.append(
            f"kazuki {name} input=kazuki_input diag={diag!r} clip={level!r} "
            f"mode={BLEND_MODES.index(p.blend)} {keys} {_hsv_kv(hsv)} "
            f"{_spec_kv(kernel_spec(p))}"
        )
    mask = _kazuki_mask(h, w)
    write_ekv(out / "kazuki_mask.mask.ekv", mask)
    level = estimate_clip_level(img)
    for name, (over, invert) in KAZUKI_MASKED_CASES.items():
        p, hsv = kazuki_params(over, {})
        stash = {}
        ctx = _ctx(diag, level, stash)
        ctx.source_mask, ctx.source_mask_invert = mask, invert
        result = Kazuki().process(img, p, ctx)
        write_ekv(out / f"{name}.out.ekv", result)
        write_ekv(out / f"{name}.glow.ekv", stash["glow"])
        write_ekv(out / f"{name}.energy.ekv", stash["energy"])
        keys = " ".join(f"{k}={getattr(p, k)!r}" for k in KAZUKI_KEYS)
        lines.append(
            f"kazuki {name} input=kazuki_input mask=kazuki_mask invert={int(invert)} "
            f"diag={diag!r} clip={level!r} mode={BLEND_MODES.index(p.blend)} {keys} "
            f"{_hsv_kv(hsv)} {_spec_kv(kernel_spec(p))}"
        )
    src = img.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    for name, invert in (("maskenergy_plain", False), ("maskenergy_inverted", True)):
        e = mask_energy(extract_energy(src, 0.0, 1.0, "scene"), mask, invert)
        write_ekv(out / f"{name}.energy.ekv", e)
        lines.append(
            f"maskenergy {name} input=kazuki_input mask=kazuki_mask "
            f"invert={int(invert)} thr=0.0 knee=1.0"
        )


def _write_stages(out: Path, lines: list[str]) -> None:
    """One fixture per kazuki stage; shared inputs are written once (`input=` key)."""
    img = _kazuki_image()
    src = img.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    _, h, w = img.shape
    diag = math.hypot(h, w)
    write_ekv(out / "kazuki_input.in.ekv", img)

    for name, thr, knee in (("energy_default", 0.0, 1.0), ("energy_hard", 1.5, 0.0)):
        write_ekv(out / f"{name}.energy.ekv", extract_energy(src, thr, knee, "scene"))
        lines.append(f"energy {name} input=kazuki_input thr={thr!r} knee={knee!r}")

    rng = np.random.default_rng(5)
    smooth = torch.from_numpy(rng.random((3, 120, 160), dtype=np.float32))
    write_ekv(out / "clip_smooth.in.ekv", smooth)
    write_ekv(out / "clip_black.in.ekv", torch.zeros(3, 8, 8))
    for name, base, arr in (
        ("clip_plateau", "kazuki_input", img),
        ("clip_smooth", "clip_smooth", smooth),
        ("clip_black", "clip_black", torch.zeros(3, 8, 8)),
    ):
        lines.append(f"clip {name} input={base} clip={estimate_clip_level(arr)!r}")

    rng = np.random.default_rng(9)
    plane = torch.from_numpy(rng.random((45, 67), dtype=np.float32))
    plane[10, 20] = 25.0
    plane[40:45, 60:67] = 9.0
    write_ekv(out / "blur_input.in.ekv", plane)
    for name, sigma in BLUR_SIGMAS.items():
        write_ekv(out / f"{name}.out.ekv", gaussian_blur(plane[None], sigma)[0])
        lines.append(f"blur {name} input=blur_input sigma={sigma!r}")

    x = torch.from_numpy(rng.random((2, 45, 67), dtype=np.float32))
    wt = plane * (plane < 5.0)  # zero weight on the hot pixels
    write_ekv(out / "wblur_input.x.ekv", x)
    write_ekv(out / "wblur_input.w.ekv", wt)
    for name, sigma in (("wblur_direct", 2.0), ("wblur_down", 12.5)):
        num, den = weighted_blur(x, wt, sigma)
        write_ekv(out / f"{name}.out.ekv", num)
        write_ekv(out / f"{name}.den.ekv", den)
        lines.append(f"wblur {name} input=wblur_input sigma={sigma!r}")

    level = estimate_clip_level(img)
    for name, over in (
        ("warmth_default", {}),
        (
            "warmth_wide",
            {"hue_radius": 0.08, "neutral_point": 0.05, "neutral_bias": 0.2},
        ),
    ):
        p = KazukiParams(**over)
        y_floor = 2.0**p.threshold_ev
        t, clipped = _warmth_map(src, p, _ctx(diag, level), y_floor)
        write_ekv(out / f"{name}.t.ekv", t)
        write_ekv(out / f"{name}.clipped.ekv", clipped.float())
        lines.append(
            f"warmth {name} input=kazuki_input diag={diag!r} clip={level!r} "
            f"y_floor={y_floor!r} hue_radius={p.hue_radius!r} "
            f"neutral_point={p.neutral_point!r} warmth_softness={p.warmth_softness!r} "
            f"neutral_bias={p.neutral_bias!r}"
        )

    k = build_kernel(PSF_CASES["kazuki_shape"][1], 200)
    write_ekv(out / "bands_input.kernel.ekv", k)
    for name, split in (("bands_default", 0.006 * 200), ("bands_zero", 0.0)):
        inner, outer = split_bands(k, split)
        write_ekv(out / f"{name}.inner.ekv", inner)
        write_ekv(out / f"{name}.outer.ekv", outer)
        lines.append(f"bands {name} input=bands_input split={split!r}")

    for name, hue, sat in (
        ("hsv_warm_core", 40.0, 0.45),
        ("hsv_cool_core", 193.846154, 0.65),
        ("hsv_wrap", 360.0, 0.3),
        ("hsv_grey", 123.0, 0.0),
    ):
        r, g, b = hsv_to_rgb(hue, sat)
        lines.append(f"hsv {name} hue={hue!r} sat={sat!r} r={r!r} g={g!r} b={b!r}")

    glow = torch.from_numpy(np.abs(rng.normal(0.0, 0.6, (3, h, w))).astype(np.float32))
    glow[:, 30:40, 40:50] = 3.0  # glow above white
    glow[:, 0:5, 0:5] = 0.0  # no glow (overlay_glow must leave these pixels alone)
    energy = extract_energy(src, 0.0, 1.0, "scene")
    write_ekv(out / "blend_input.glow.ekv", glow)
    write_ekv(out / "blend_input.energy.ekv", energy)
    for name, (mode, strength, white_ev) in BLEND_CASES.items():
        result = blend_glow(img, glow, energy, mode, strength, white_ev)
        write_ekv(out / f"{name}.out.ekv", result)
        lines.append(
            f"blend {name} input=kazuki_input mode={BLEND_MODES.index(mode)} "
            f"strength={strength!r} white_ev={white_ev!r}"
        )


def main(argv: list[str]) -> int:
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, (diag, spec) in PSF_CASES.items():
        write_ekv(out / f"{name}.kernel.ekv", build_kernel(spec, diag))
        lines.append(f"psf {name} diag={diag!r} {_spec_kv(spec)}")

    rng = np.random.default_rng(3)
    src = torch.from_numpy(rng.random((3, 37, 53), dtype=np.float32))
    kernel = build_kernel(KernelSpec(halo_radius=0.02, halo_stretch_x=1.7), 60)
    write_ekv(out / "conv_small.src.ekv", src)
    write_ekv(out / "conv_small.kernel.ekv", kernel)
    write_ekv(out / "conv_small.out.ekv", fft_convolve(src, kernel))
    lines.append("conv conv_small")

    img = _mist_image()
    _, h, w = img.shape
    diag = math.hypot(h, w)
    for name, p in MIST_CASES.items():
        stash: dict = {}
        ctx = _ctx(diag, math.inf, stash)
        result = Mist().process(img, p, ctx)
        write_ekv(out / f"{name}.in.ekv", img)
        write_ekv(out / f"{name}.out.ekv", result)
        write_ekv(out / f"{name}.glow.ekv", stash["glow"])
        lines.append(
            f"mist {name} diag={diag!r} strength={p.strength!r} {_spec_kv(kernel_spec(p))}"
        )

    k = build_kernel(PSF_CASES["kazuki_shape"][1], 400)
    write_ekv(out / "kview_kazuki.kernel.ekv", k)
    write_ekv(out / "kview_kazuki.u8.ekv", kernel_image(k)[..., 0].float())
    lines.append("kview kview_kazuki")

    _write_stages(out, lines)
    _write_kazuki(out, lines)

    (out / "cases.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} cases to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
