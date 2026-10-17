"""Measure a preview-only kazuki glow approximation against the exact glow (spec §11).

Each band kernel K (inner, outer) is split at r_near with a smooth cross-fade
w(ρ) = 1 − smoothstep(ρ, 0.8·r_near, r_near): K_near = w·K is convolved at full
resolution, K_far = K − K_near on the basis planes box-downsampled by f (kernel
sum-pooled by f, total weight kept), then bilinearly upsampled (the wblur
`gaussian_blur` downsample convention). Both finals go through Rec.2020 → linear sRGB →
clip → sRGB OETF → 8 bit and are compared. Measurement only: nothing in the package is
changed, the exact result is the module's own `Kazuki.process`.

Usage:
  uv run python tools/approx_glow_probe.py --input TIF [--max-side N] [--out CSV]
      [--preset NAME ...] [--f 2 4 8] [--r-near 16 32 64 128]
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dt_compare import (  # noqa: E402
    KAZUKI_DEFAULTS,
    KAZUKI_PRESETS,
    kazuki_c_to_py,
    kazuki_preset,
    pack_kazuki,
    unpack_kazuki,
)

from ekvido.common.blend import blend_glow
from ekvido.common.convolve import Convolver
from ekvido.common.display import REC2020_TO_SRGB, srgb_oetf
from ekvido.common.energy import (
    estimate_clip_level,
    extract_energy,
    positive,
    smoothstep,
)
from ekvido.common.psf import build_kernel, kernel_half_cap, split_bands
from ekvido.io import read_tiff
from ekvido.modules.base import Context, kernel_spec
from ekvido.modules.kazuki import Kazuki, _basis, _combine, _warmth_map

PRESETS = ("default", "wide", "screen", "glowdodge", "overlayglow", "star-heavy")
COLUMNS = (
    "preset",
    "f",
    "r_near",
    "max_diff_8bit",
    "pct_gt1",
    "pct_gt2",
    "t_exact_s",
    "t_approx_s",
    "fft_points_ratio",
)


def preset_c(name: str) -> dict:
    if name == "star-heavy":
        return unpack_kazuki(pack_kazuki({**KAZUKI_DEFAULTS, "w_arms": 1.0}))
    return kazuki_preset(name)


def load(path: str, max_side: int | None) -> torch.Tensor:
    img = read_tiff(path).data
    _, h, w = img.shape
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        img = F.interpolate(img[None], size=(round(h * s), round(w * s)), mode="area")[
            0
        ].contiguous()
    return img


def to_8bit(img: torch.Tensor) -> torch.Tensor:
    m = REC2020_TO_SRGB.to(img.dtype)
    x = torch.einsum("ij,jhw->ihw", m, img).clamp(0.0, 1.0)
    return torch.round(srgb_oetf(x) * 255.0).to(torch.int16)


def _rad(r: int) -> torch.Tensor:
    ax = torch.arange(-r, r + 1, dtype=torch.float32)
    y, x = torch.meshgrid(ax, ax, indexing="ij")
    return torch.sqrt(x * x + y * y)


def split_near_far(k: torch.Tensor, r_near: float):
    """(K_near cropped to its support, K_far or None if it is all zero)."""
    r = (k.shape[0] - 1) // 2
    near = k * (1.0 - smoothstep(_rad(r), 0.8 * r_near, r_near))
    far = k - near
    rn = min(r, math.ceil(r_near))
    near_c = near[r - rn : r + rn + 1, r - rn : r + rn + 1]
    return near_c, (far if bool((far != 0).any()) else None)


def _box(f: int) -> torch.Tensor:
    """Box of width f centred on a tap: odd f plain, even f with half-weight ends."""
    if f % 2:
        return torch.ones(f)
    b = torch.ones(f + 1)
    b[0] = b[-1] = 0.5
    return b


def sum_pool(k: torch.Tensor, f: int) -> torch.Tensor:
    """Kernel on the f-times coarser grid: K_s[d] = Σ_o K[o]·b(o − d·f), a partition of
    unity in o, so K_s.sum() == K.sum()."""
    r = (k.shape[0] - 1) // 2
    rs = math.ceil((r + f / 2) / f)
    half = rs * f + f
    kp = F.pad(k, (half - r,) * 4)
    b = _box(f)
    pb = (b.numel() - 1) // 2
    kb = F.conv2d(kp[None, None], b.view(1, 1, 1, -1), padding=(0, pb))
    kb = F.conv2d(kb, b.view(1, 1, -1, 1), padding=(pb, 0))[0, 0]
    idx = half + f * torch.arange(-rs, rs + 1)
    return kb[idx][:, idx].contiguous()


def downsample(x: torch.Tensor, f: int) -> torch.Tensor:
    _, h, w = x.shape
    padded = F.pad(x[None], (0, -w % f, 0, -h % f), mode="replicate")
    return F.avg_pool2d(padded, f)[0]


def upsample(x: torch.Tensor, f: int, h: int, w: int) -> torch.Tensor:
    up = F.interpolate(
        x[None], size=(h + (-h % f), w + (-w % f)), mode="bilinear", align_corners=False
    )
    return up[0, :, :h, :w]


def exact_glow(basis, inner, outer, p, pad):
    conv = Convolver(basis, pad, True)
    glow = _combine(conv.apply(inner), p.warm_core, p.cool_core, p)
    glow += _combine(conv.apply(outer), p.warm_edge, p.cool_edge, p)
    return glow, conv.fh * conv.fw


def approx_glow(basis, inner, outer, p, f: int, r_near: float):
    """Near part at full resolution, far part on the f-downsampled basis (f = 1: both
    full resolution, the sanity row). Returns (glow, FFT grid points used)."""
    _, h, w = basis.shape
    (ni, fi), (no, fo) = split_near_far(inner, r_near), split_near_far(outer, r_near)
    near = Convolver(basis, max((ni.shape[0] - 1) // 2, (no.shape[0] - 1) // 2), True)
    points = near.fh * near.fw
    fars = [fi, fo]
    g = [near.apply(ni), near.apply(no)]
    if any(k is not None for k in fars):
        small = downsample(basis, f) if f > 1 else basis
        pooled = [None if k is None else (sum_pool(k, f) if f > 1 else k) for k in fars]
        pad = max((k.shape[0] - 1) // 2 for k in pooled if k is not None)
        far = Convolver(small, pad, True)
        points += far.fh * far.fw
        for i, k in enumerate(pooled):
            if k is not None:
                gs = far.apply(k)
                g[i] += upsample(gs, f, h, w) if f > 1 else gs
    glow = _combine(g[0], p.warm_core, p.cool_core, p)
    glow += _combine(g[1], p.warm_edge, p.cool_edge, p)
    return glow, points


def run_preset(img, name, grid, sanity, rows, writer):
    c = preset_c(name)
    p, manual = kazuki_c_to_py(c)
    _, h, w = img.shape
    clip = estimate_clip_level(img) if manual is None else manual
    ctx = Context(
        device=torch.device("cpu"),
        referred="scene",
        diag_px=math.hypot(h, w),
        clip_level=clip,
        input_key="probe",
        cache=None,
        keep_spectra=True,
    )
    exact_final = Kazuki().process(img, p, ctx)  # the module's own result
    exact8 = to_8bit(exact_final)

    # Shared inputs, built exactly as Kazuki.process builds them.
    src = positive(img)
    thr, knee = p.threshold_ev, p.knee_ev
    energy = extract_energy(src, thr, knee, "scene")
    t, _ = _warmth_map(src, p, ctx, 2.0**thr)
    basis = _basis(energy, t)
    pad = kernel_half_cap(ctx.diag_px)
    kernel = build_kernel(kernel_spec(p), ctx.diag_px)
    inner, outer = split_bands(kernel, p.band_split * ctx.diag_px)

    t0 = time.perf_counter()
    glow, exact_points = exact_glow(basis, inner, outer, p, pad)
    t_exact = time.perf_counter() - t0
    mine8 = to_8bit(
        blend_glow(img, glow, energy, p.blend, p.strength, p.blend_white_ev)
    )
    if not torch.equal(mine8, exact8):
        print(f"WARNING {name}: re-implemented exact path differs from Kazuki.process")
    del glow, mine8

    settings = ([(1, grid[1][0])] if sanity else []) + [
        (f, r) for f in grid[0] for r in grid[1]
    ]
    for f, r_near in settings:
        t0 = time.perf_counter()
        glow, points = approx_glow(basis, inner, outer, p, f, r_near)
        t_approx = time.perf_counter() - t0
        final = blend_glow(img, glow, energy, p.blend, p.strength, p.blend_white_ev)
        d = (to_8bit(final) - exact8).abs()
        if f == 1:  # sanity: the split itself, all at full resolution
            print(
                f"{name:12s} sanity f=1: max |final - exact| (linear) "
                f"{float((final - exact_final).abs().max()):.3g}, "
                f"pixels with |Δ8| = 1: {int((d == 1).sum())} of {d.numel()}",
                flush=True,
            )
        del glow, final
        n = d.numel()
        row = {
            "preset": name,
            "f": f,
            "r_near": r_near,
            "max_diff_8bit": int(d.max()),
            "pct_gt1": 100.0 * int((d > 1).sum()) / n,
            "pct_gt2": 100.0 * int((d > 2).sum()) / n,
            "t_exact_s": round(t_exact, 3),
            "t_approx_s": round(t_approx, 3),
            "fft_points_ratio": round(points / exact_points, 4),
        }
        rows.append(row)
        if writer:
            writer.writerow(row)
        print(
            f"{name:12s} f={f} r_near={r_near:4g} max={row['max_diff_8bit']:3d} "
            f">1={row['pct_gt1']:.5f}% >2={row['pct_gt2']:.5f}% "
            f"t_exact={t_exact:.2f}s t_approx={t_approx:.2f}s "
            f"fft_ratio={row['fft_points_ratio']:.3f}",
            flush=True,
        )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--max-side", type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--preset", nargs="+", choices=PRESETS, default=list(PRESETS))
    ap.add_argument("--f", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--r-near", nargs="+", type=float, default=[16, 32, 64, 128])
    ap.add_argument("--no-sanity", action="store_true", help="skip the f = 1 row")
    args = ap.parse_args(argv)
    assert set(PRESETS) - {"star-heavy"} <= set(KAZUKI_PRESETS)
    torch.set_grad_enabled(False)
    img = load(args.input, args.max_side)
    _, h, w = img.shape
    print(f"{args.input}: {w}x{h}, torch threads {torch.get_num_threads()}", flush=True)
    rows: list[dict] = []
    fh = None
    writer = None
    if args.out:
        new = not args.out.exists()
        fh = args.out.open("a", newline="")
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
    try:
        for name in args.preset:
            run_preset(
                img, name, (args.f, args.r_near), not args.no_sanity, rows, writer
            )
            if fh:
                fh.flush()
    finally:
        if fh:
            fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
