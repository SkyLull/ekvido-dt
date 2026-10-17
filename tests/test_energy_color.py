import math

import pytest
import torch

from ekvido.common.color import chromaticity, unit_luminance, warmth
from ekvido.common.energy import (
    estimate_clip_level,
    extract_energy,
    luminance,
    positive,
    smoothstep,
)


def test_positive_sanitises_nan_inf_and_negatives():
    x = torch.tensor([float("nan"), float("inf"), -float("inf"), -2.0, 3.0]).view(
        1, 1, 5
    )
    assert positive(x).flatten().tolist() == [0.0, 0.0, 0.0, 0.0, 3.0]


def test_luminance_of_white_is_one():
    assert torch.allclose(luminance(torch.ones(3, 2, 2)), torch.ones(2, 2), atol=1e-6)


def test_smoothstep_bounds_and_hard_step():
    x = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
    assert smoothstep(x, 0.0, 1.0).tolist() == [0.0, 0.0, 0.5, 1.0, 1.0]
    assert smoothstep(x, 0.5, 0.5).tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_scene_energy_threshold_in_ev():
    img = torch.tensor([0.25, 1.0, 4.0]).view(1, 1, 3).expand(3, 1, 3).clone()
    e = extract_energy(img, threshold=0.0, knee=1.0, referred="scene")  # knee −1..+1 EV
    assert e[:, 0, 0].abs().max() == 0  # −2 EV: below
    assert torch.allclose(e[:, 0, 1], img[:, 0, 1] * 0.5)  # at threshold: half
    assert torch.allclose(e[:, 0, 2], img[:, 0, 2])  # +2 EV: full


def test_display_energy_linear():
    img = torch.tensor([0.1, 0.9]).view(1, 1, 2).expand(3, 1, 2).clone()
    e = extract_energy(img, threshold=0.5, knee=0.1, referred="display")
    assert e[:, 0, 0].abs().max() == 0 and torch.allclose(e[:, 0, 1], img[:, 0, 1])


def test_energy_keeps_source_colour():
    img = torch.tensor([8.0, 4.0, 1.0]).view(3, 1, 1)
    e = extract_energy(img, threshold=0.0, knee=0.5, referred="scene")
    assert torch.allclose(e, img)


def test_energy_bad_mode():
    with pytest.raises(ValueError):
        extract_energy(torch.ones(3, 1, 1), 0.0, 1.0, "linear")


def test_clip_level_detects_plateau():
    img = torch.rand(3, 100, 100)
    img[:, :10, :10] = 4.0
    assert estimate_clip_level(img) == pytest.approx(4.0 * 0.999)


def test_clip_level_none_without_plateau():
    img = torch.rand(3, 100, 100, generator=torch.Generator().manual_seed(0)) * 5
    assert math.isinf(estimate_clip_level(img))


def test_chromaticity_sums_to_one_and_black_is_neutral():
    img = torch.tensor([[[2.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]])
    c = chromaticity(img)
    assert torch.allclose(c.sum(0), torch.ones(1, 2))
    assert torch.allclose(c[:, 0, 1], torch.full((3,), 1 / 3))


def test_warmth_orange_warm_blue_cool_white_middle():
    img = torch.tensor([[[1.0, 0.25, 1.0]], [[0.55, 0.5, 1.0]], [[0.2, 1.0, 1.0]]])
    t = warmth(chromaticity(img), neutral_point=0.0, softness=0.05)[0]
    assert t[0] > 0.99 and t[1] < 0.01 and t[2] == pytest.approx(0.5)


def test_unit_luminance():
    u = unit_luminance((1.0, 0.5, 0.2))
    assert luminance(u.view(3, 1, 1)).item() == pytest.approx(1.0)


def test_clip_level_detects_small_plateau():
    # A few small blown glints: ~5e-4 of the frame, all at the exact clip value.
    img = torch.rand(3, 200, 300, generator=torch.Generator().manual_seed(1))
    img[:, 10:15, 10:16] = 2.0
    assert estimate_clip_level(img) == pytest.approx(2.0 * 0.999)


def test_hsv_pairs_reproduce_the_kazuki_default_colours():
    from ekvido.common.color import hsv_to_rgb
    from ekvido.modules.kazuki import KazukiParams

    p = KazukiParams()
    pairs = {
        "warm_core": (40.0, 0.45),
        "warm_edge": (22.5, 0.88),
        "cool_core": (193.846154, 0.65),
        "cool_edge": (232.173913, 0.92),
    }
    for name, (hue, sat) in pairs.items():
        assert hsv_to_rgb(hue, sat) == pytest.approx(getattr(p, name), abs=1e-6), name
    assert hsv_to_rgb(360.0, 0.5) == pytest.approx(hsv_to_rgb(0.0, 0.5))
