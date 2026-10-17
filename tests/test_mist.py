import math

import pytest
import torch

import ekvido.common.convolve as convolve
from ekvido.modules import base
from ekvido.modules.mist import Mist, MistParams


def _ctx(img, cache=None, stash=None):
    _, h, w = img.shape
    return base.Context(
        device=img.device,
        referred="scene",
        diag_px=math.hypot(h, w),
        clip_level=math.inf,
        input_key="img",
        cache=cache,
        stash=stash,
    )


def _point(h=200, w=200, v=10.0):
    # Black background: light scattered past the frame edge is (correctly) lost,
    # which would muddy the conservation check.
    img = torch.zeros(3, h, w)
    img[:, h // 2, w // 2] = v
    return img


def test_registered():
    import ekvido.modules  # noqa: F401

    assert base.get_module("mist").NAME == "mist"


def test_strength_zero_is_identity():
    img = _point()
    out = Mist().process(img, MistParams(strength=0.0), _ctx(img))
    assert torch.equal(out, img)


def test_conserves_energy():
    img = _point()
    out = Mist().process(img, MistParams(strength=0.4), _ctx(img))
    # Edge normalisation (no vignette) gives up strict conservation near the frame
    # edge; away from it the total is still preserved to well under 1%.
    assert out.sum().item() == pytest.approx(img.sum().item(), rel=1e-2)
    assert out[:, 100, 100].mean() < img[:, 100, 100].mean()  # core lost light
    assert out[:, 100, 110].mean() > img[:, 100, 110].mean()  # halo gained it


def test_glow_scales_with_intensity():
    p = MistParams(strength=0.3)
    a, b = _point(v=10.0), _point(v=20.0)
    ga = Mist().process(a, p, _ctx(a)) - 0.7 * a
    gb = Mist().process(b, p, _ctx(b)) - 0.7 * b
    assert torch.allclose(gb, 2 * ga, atol=1e-5)


@pytest.mark.parametrize("shape", [(16, 16), (8, 400), (400, 8)])
def test_small_and_extreme_shapes(shape):
    img = torch.rand(3, *shape)
    out = Mist().process(img, MistParams(arms=4, w_arms=0.3), _ctx(img))
    assert out.shape == img.shape and torch.isfinite(out).all()


def test_nan_pixel_stays_local():
    img = _point()
    img[0, 5, 5] = float("nan")
    out = Mist().process(img, MistParams(), _ctx(img))
    assert torch.isnan(out[0, 5, 5])
    out[0, 5, 5] = 0
    assert torch.isfinite(out).all()


def test_strength_change_reuses_glow(monkeypatch):
    calls = []
    orig = convolve.Convolver.apply
    monkeypatch.setattr(
        convolve.Convolver, "apply", lambda s, k: calls.append(1) or orig(s, k)
    )
    img, cache = _point(), {}
    Mist().process(img, MistParams(strength=0.1), _ctx(img, cache))
    n = len(calls)  # glow + frame-coverage map
    Mist().process(img, MistParams(strength=0.5), _ctx(img, cache))
    assert len(calls) == n
    Mist().process(img, MistParams(halo_radius=0.01), _ctx(img, cache))
    assert len(calls) == 2 * n


def test_stash():
    img, stash = _point(), {}
    Mist().process(img, MistParams(), _ctx(img, stash=stash))
    assert set(stash) == {"glow", "kernel"}


@pytest.mark.gpu
def test_device_parity(gpu):
    img = _point()
    ref = Mist().process(img, MistParams(), _ctx(img))
    g = img.to(gpu)
    out = Mist().process(g, MistParams(), _ctx(g)).cpu()
    assert torch.allclose(out, ref, atol=1e-4)


def test_uniform_field_stays_uniform():
    # Review C1: zero-padding made mist vignette every frame.
    img = torch.full((3, 400, 600), 0.5)
    out = Mist().process(
        img, MistParams(strength=0.3, w_halo=0.4, w_veil=0.6), _ctx(img)
    )
    assert torch.allclose(out, img, rtol=1e-3)
