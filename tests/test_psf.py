import math

import pytest
import torch

from ekvido.common.psf import (
    KernelSpec,
    build_kernel,
    kernel_half_cap,
    kernel_half_size,
    split_bands,
)

SPECS = [
    KernelSpec(),
    KernelSpec(w_halo=0.5, w_veil=0.5),
    KernelSpec(arms=4, w_arms=0.3),
    KernelSpec(halo_stretch_x=3.0, arms=2, w_arms=1.0, w_halo=0.0),
]


@pytest.mark.parametrize("spec", SPECS)
def test_kernel_normalised_square_odd(spec):
    k = build_kernel(spec, diag_px=1000)
    assert k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1
    assert k.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert torch.isfinite(k).all() and (k >= 0).all()


def test_halo_symmetric_and_monotone():
    k = build_kernel(KernelSpec(halo_radius=0.01), diag_px=1000)
    assert torch.allclose(k, k.flip(0)) and torch.allclose(k, k.flip(1))
    r = (k.shape[0] - 1) // 2
    row = k[r, r:]
    assert (row[1:] <= row[:-1]).all()


def _axis_energy(k, skip=5):
    """Light on the horizontal and vertical axes, outside the central core."""
    r = (k.shape[0] - 1) // 2
    h = k[r, : r - skip].sum() + k[r, r + skip + 1 :].sum()
    v = k[: r - skip, r].sum() + k[r + skip + 1 :, r].sum()
    return h.item(), v.item()


def test_streak_follows_angle():
    base = dict(arms=2, w_arms=1.0, w_halo=0.0, arm_radius=0.001, arm_sharpness=0.3)
    h0, v0 = _axis_energy(build_kernel(KernelSpec(arm_angle=0.0, **base), 1000))
    h90, v90 = _axis_energy(build_kernel(KernelSpec(arm_angle=90.0, **base), 1000))
    assert h0 > 3 * v0 and v90 > 3 * h90


def test_cross_has_both_axes():
    h, v = _axis_energy(
        build_kernel(KernelSpec(arms=4, w_arms=1.0, w_halo=0.0, arm_radius=0.002), 1000)
    )
    assert h == pytest.approx(v, rel=0.05)


def test_half_size_capped():
    spec = KernelSpec(w_halo=0.0, w_veil=1.0, veil_radius=0.2, veil_beta=0.6)
    assert kernel_half_size(spec, 1000) == kernel_half_cap(1000) == 250


def test_resolution_consistency():
    spec = KernelSpec(halo_radius=0.01)
    k1, k2 = build_kernel(spec, 1000), build_kernel(spec, 2000)
    r1, r2 = (k1.shape[0] - 1) // 2, (k2.shape[0] - 1) // 2
    for i in (0, 5, 10, 20):
        assert k1[r1, r1 + i].item() == pytest.approx(
            4 * k2[r2, r2 + 2 * i].item(), rel=0.05
        )


def test_no_parts_gives_delta():
    k = build_kernel(KernelSpec(w_halo=0.0), 1000)
    assert k.numel() >= 1 and k.max().item() == 1.0


def test_tiny_sizes_stay_finite():
    k = build_kernel(
        KernelSpec(
            halo_radius=1e-6, arms=3, w_arms=1.0, arm_radius=1e-9, arm_sharpness=0.2
        ),
        50,
    )
    assert torch.isfinite(k).all() and k.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_split_bands():
    k = build_kernel(KernelSpec(halo_radius=0.01), 1000)
    inner, outer = split_bands(k, r_split_px=15.0)
    assert torch.allclose(inner + outer, k)
    r = (k.shape[0] - 1) // 2
    assert inner[r, r] == k[r, r] and inner[r, r + 40] == 0
    zi, zo = split_bands(k, 0.0)
    assert zi.abs().sum() == 0 and torch.allclose(zo, k)


def test_capped_kernel_tapers_to_zero_at_its_edge():
    # Review I1: a hard square cut of the veil printed a visible square around suns.
    k = build_kernel(KernelSpec(w_halo=0.5, w_veil=0.5), 1000)
    r = (k.shape[0] - 1) // 2
    assert k[r, 0] == 0 and k[0, 0] == 0  # edge and corner of the square
    peak = k[r, r]
    assert k[r, 5] < 1e-3 * peak  # nearly zero just inside the cut


STAR = dict(w_halo=0.0, w_arms=1.0, arms=4, arm_radius=0.002, arm_beta=1.5)


def _star(**kw):
    k = build_kernel(KernelSpec(**{**STAR, **kw}), 1000)
    return k, (k.shape[0] - 1) // 2


def test_arms_fade_smoothly_with_no_end():
    # The old arms dropped five decades in their last few pixels and stopped dead.
    k, r = _star()
    row = k[r, r:]
    stop = int(0.75 * r)  # inside the edge taper
    assert (row[1 : stop + 1] < row[:stop]).all()
    for i in range(4, int(stop / 1.25)):  # never steeper than the 1/r^(2*beta) tail
        assert row[round(1.25 * i)] > 0.4 * row[i], i


def test_brighter_source_gets_longer_arms():
    k, r = _star()
    row = k[r, r:] / k[r, r]

    def visible(brightness, threshold=1.0):
        return int((brightness * row > threshold).sum())

    assert visible(1e4) > 1.5 * visible(1e3) > 1.5 * 1.5 * visible(1e2)


def test_arms_carry_more_light_than_the_diagonals():
    k, r = _star(arm_sharpness=0.4)
    a = 20
    assert k[r, r + a] > 5 * k[r + round(a / 2**0.5), r + round(a / 2**0.5)]


def test_sharpness_two_is_round_and_one_is_a_diamond():
    k, r = _star(arm_sharpness=2.0)
    assert k[r + 12, r + 16].item() == pytest.approx(k[r, r + 20].item(), rel=0.02)
    k, r = _star(arm_sharpness=1.0)
    assert k[r + 10, r + 10].item() == pytest.approx(k[r, r + 20].item(), rel=0.05)


def test_arm_stretch_widens_the_star():
    h, v = _axis_energy(_star(arm_stretch_x=2.0)[0])
    assert h > 1.5 * v
    h, v = _axis_energy(_star()[0])
    assert h == pytest.approx(v, rel=0.02)


def _ring(k, r, radius, deg):
    """Bilinear sample of k at polar (radius, deg) around the centre."""
    x = r + radius * math.cos(math.radians(deg))
    y = r + radius * math.sin(math.radians(deg))
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    return (
        k[y0, x0] * (1 - fx) * (1 - fy)
        + k[y0, x0 + 1] * fx * (1 - fy)
        + k[y0 + 1, x0] * (1 - fx) * fy
        + k[y0 + 1, x0 + 1] * fx * fy
    ).item()


@pytest.mark.parametrize("n", [3, 6, 8])
def test_arm_count(n):
    k, r = _star(arms=n, arm_sharpness=0.4, arm_angle=10.0)
    step = 360.0 / n
    for i in range(n):
        on = _ring(k, r, 25, 10.0 + i * step)
        between = _ring(k, r, 25, 10.0 + (i + 0.5) * step)
        assert on > 3 * between, (i, on, between)
