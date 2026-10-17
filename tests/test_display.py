import pytest
import torch

from ekvido.common.display import (
    REC2020_TO_SRGB,
    colormap_warmth,
    kernel_image,
    mask_overlay,
    srgb_oetf,
    to_display,
    tone_curve,
)


def test_matrix_keeps_white():
    assert torch.allclose(REC2020_TO_SRGB @ torch.ones(3), torch.ones(3), atol=1e-3)


def test_tone_curve_monotone_bounded():
    x = torch.linspace(0, 64, 1000)
    y = tone_curve(x)
    assert (y[1:] >= y[:-1]).all() and y.max() <= 1.0 and y[0] == 0
    assert tone_curve(torch.tensor([16.0])).item() == pytest.approx(1.0, abs=1e-6)


def test_srgb_oetf_endpoints():
    assert srgb_oetf(torch.tensor([0.0, 1.0, 2.0, -1.0])).tolist() == pytest.approx(
        [0, 1, 1, 0], abs=1e-6
    )


def test_to_display_shape_dtype_and_white():
    img = torch.ones(3, 4, 5)
    out = to_display(img, "display")
    assert out.shape == (4, 5, 3) and out.dtype == torch.uint8
    assert out.min().item() >= 254


def test_display_mode_clamps_headroom():
    assert to_display(torch.full((3, 1, 1), 5.0), "display").min().item() == 255


def test_scene_mode_keeps_highlight_detail():
    a = to_display(torch.full((3, 1, 1), 2.0), "scene")
    b = to_display(torch.full((3, 1, 1), 4.0), "scene")
    assert a.max() < b.max() < 255


def test_exposure_brightens():
    img = torch.full((3, 1, 1), 0.1)
    assert to_display(img, "scene", 1.0).max() > to_display(img, "scene", 0.0).max()


def test_nan_and_negative_safe():
    img = torch.tensor([float("nan"), -1.0, 0.5]).view(3, 1, 1)
    out = to_display(img, "scene")
    assert out.dtype == torch.uint8


def test_bad_mode():
    with pytest.raises(ValueError):
        to_display(torch.ones(3, 1, 1), "linear")


def test_colormap_warmth_endpoints():
    c = colormap_warmth(torch.tensor([[0.0, 0.5, 1.0]]))
    assert c[0, 0].tolist() == [40, 90, 255]
    assert c[0, 1].tolist() == [128, 128, 128]
    assert c[0, 2].tolist() == [255, 140, 30]


def test_mask_overlay():
    base = torch.zeros(2, 2, 3, dtype=torch.uint8)
    mask = torch.tensor([[True, False], [False, False]])
    out = mask_overlay(base, mask)
    assert out[0, 0].tolist() == [255, 0, 0] and out[1, 1].tolist() == [0, 0, 0]


def test_kernel_image_log_scale():
    k = torch.tensor([[1e-7, 1e-3], [1.0, 0.0]])
    img = kernel_image(k)
    assert img.shape == (2, 2, 3) and img[1, 0, 0] == 255 and img[0, 0, 0] == 0
    assert img[0, 1, 0] == pytest.approx(
        128, abs=1
    )  # 1e-3 → halfway on a 6-decade scale
