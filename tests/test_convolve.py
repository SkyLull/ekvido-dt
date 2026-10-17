import pytest
import torch
import torch.nn.functional as F

from ekvido.common.convolve import Convolver, fft_convolve, fft_size


def _direct(src, k):
    r = (k.shape[0] - 1) // 2
    w = k.flip(0, 1).view(1, 1, *k.shape).expand(src.shape[0], 1, *k.shape)
    return F.conv2d(src[None], w, padding=r, groups=src.shape[0])[0]


def test_fft_size():
    assert [fft_size(n) for n in (1, 7, 11, 97, 128)] == [1, 8, 12, 100, 128]


def test_matches_direct_conv_with_asymmetric_kernel():
    g = torch.Generator().manual_seed(0)
    src = torch.rand(3, 20, 30, generator=g)
    k = torch.rand(7, 7, generator=g)
    assert torch.allclose(fft_convolve(src, k), _direct(src, k), atol=1e-4)


def test_edge_impulse_does_not_wrap():
    src = torch.zeros(1, 16, 16)
    src[0, 0, 15] = 1.0
    out = fft_convolve(src, torch.full((5, 5), 1 / 25))
    assert out[0, :, :3].abs().max() < 1e-6
    assert out[0, 0, 13:].sum() > 0


def test_kernel_larger_than_image():
    g = torch.Generator().manual_seed(1)
    src, k = torch.rand(3, 4, 4, generator=g), torch.rand(21, 21, generator=g)
    assert torch.allclose(fft_convolve(src, k), _direct(src, k), atol=1e-4)


def test_extreme_aspect_runs():
    out = fft_convolve(torch.rand(3, 8, 400), torch.ones(101, 101) / 101**2)
    assert out.shape == (3, 8, 400) and torch.isfinite(out).all()


def test_kernel_too_big_for_pad_rejected():
    with pytest.raises(ValueError):
        Convolver(torch.rand(1, 8, 8), pad=2).apply(torch.ones(7, 7))


def test_keep_spectra_false_same_result():
    g = torch.Generator().manual_seed(2)
    src, k = torch.rand(3, 12, 10, generator=g), torch.rand(5, 5, generator=g)
    a = Convolver(src, pad=4).apply(k)
    b = Convolver(src, pad=4, keep_spectra=False).apply(k)
    assert torch.allclose(a, b, atol=1e-6)


def test_reuses_source_spectra_across_kernels():
    conv = Convolver(torch.rand(3, 12, 10), pad=4)
    conv.apply(torch.ones(3, 3))
    first = conv._spectra
    conv.apply(torch.ones(5, 5))
    assert conv._spectra is first


@pytest.mark.gpu
def test_device_parity(gpu):
    g = torch.Generator().manual_seed(3)
    src, k = torch.rand(3, 64, 80, generator=g), torch.rand(15, 15, generator=g)
    ref = fft_convolve(src, k)
    out = fft_convolve(src.to(gpu), k.to(gpu)).cpu()
    assert torch.allclose(out, ref, atol=1e-3)
