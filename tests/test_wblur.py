import pytest
import torch

from ekvido.common.wblur import gaussian_blur, weighted_blur


@pytest.mark.parametrize("sigma", [0.0, 1.5, 40.0])
def test_constant_image_preserved(sigma):
    x = torch.full((3, 120, 90), 0.7)
    assert torch.allclose(gaussian_blur(x, sigma), x, atol=1e-5)


def test_small_sigma_variance():
    x = torch.zeros(1, 41, 41)
    x[0, 20, 20] = 1.0
    y = gaussian_blur(x, 2.0)[0]
    ax = torch.arange(-20, 21, dtype=torch.float32)
    assert y.sum().item() == pytest.approx(1.0, abs=1e-4)
    assert (y.sum(0) * ax**2).sum().item() == pytest.approx(4.0, rel=0.05)


def test_large_sigma_spreads():
    x = torch.zeros(1, 200, 200)
    x[0, 90:110, 90:110] = 1.0
    y = gaussian_blur(x, 30.0)
    assert y[0, 100, 100] < 0.5 and y[0, 100, 160] > 0.005  # analytic ≈ 0.0096


def test_tiny_image_large_sigma():
    assert gaussian_blur(torch.rand(3, 3, 5), 50.0).shape == (3, 3, 5)


def test_clipped_core_inherits_fringe_hue():
    yy, xx = torch.meshgrid(torch.arange(101.0), torch.arange(101.0), indexing="ij")
    r = torch.sqrt((yy - 50) ** 2 + (xx - 50) ** 2)
    ring = (r >= 10) & (r < 16)  # core (r < 10) is clipped: weight 0
    chroma = torch.full((3, 101, 101), 1 / 3)
    orange = torch.tensor([0.55, 0.3, 0.15]).view(3, 1)
    chroma[:, ring] = orange
    w = ring.float()  # clipped core and dark surround carry no weight
    hue, den = weighted_blur(chroma, w, sigma_px=8.0)
    assert torch.allclose(hue[:, 50, 50], orange[:, 0], atol=1e-3)
    assert den[50, 50] > 0


def test_no_weight_anywhere_gives_zero_density():
    _, den = weighted_blur(torch.rand(3, 20, 20), torch.zeros(20, 20), 3.0)
    assert den.abs().max() == 0


@pytest.mark.gpu
def test_large_sigma_on_gpu_with_odd_size(gpu):
    x = torch.rand(3, 203, 307, generator=torch.Generator().manual_seed(0))
    ref = gaussian_blur(x, 30.0)
    out = gaussian_blur(x.to(gpu), 30.0).cpu()
    assert torch.allclose(out, ref, atol=1e-4)
