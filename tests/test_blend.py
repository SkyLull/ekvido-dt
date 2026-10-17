import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ekvido.common.blend import (
    BLEND_MODES,
    CSP_MODES,
    blend_glow,
    csp_blend,
    dodge,
    scene_blend,
)
from ekvido.common.display import srgb_eotf, srgb_oetf

CSP_DIR = Path(__file__).parent / "data" / "csp-blend"
# Clip Studio Paint exports: layer FF7F00 over a flat grey, opacity ramp left→right.
# One row near the top of each band (text labels sit in the band middles).
BAND_ROWS = {
    "normal": 15,
    "glow_dodge": 305,
    "color_dodge": 600,
    "screen": 895,
    "overlay": 1190,
}
LAYER = torch.tensor([255.0, 127.0, 0.0]) / 255.0


def t(*v):
    return torch.tensor(v, dtype=torch.float32)


def test_mode_list_appends_csp_modes_after_the_originals():
    assert BLEND_MODES[:2] == ("additive", "conserve")
    assert CSP_MODES == (
        "screen",
        "overlay",
        "color_dodge",
        "glow_dodge",
        "overlay_glow",
    )


def test_dodge_cases():
    a = t(0.0, 0.3, 0.2, 0.5, 0.9)
    c = t(0.9, 1.0, 0.5, 0.5, 0.2)
    assert dodge(a, c).tolist() == pytest.approx([0.0, 1.0, 0.4, 1.0, 1.0])


@pytest.mark.parametrize(
    ("mode", "a", "c", "alpha", "want"),
    [
        ("screen", 0.5, 0.5, 1.0, 0.75),
        ("screen", 0.5, 0.5, 0.5, 0.625),
        ("overlay", 0.25, 0.5, 1.0, 0.25),
        ("overlay", 0.75, 0.5, 1.0, 0.75),
        ("overlay", 0.75, 1.0, 0.5, 0.875),
        ("color_dodge", 0.2, 0.5, 1.0, 0.4),
        ("color_dodge", 0.2, 0.5, 0.5, 0.3),
        ("glow_dodge", 0.2, 0.5, 0.5, 0.2 / 0.75),
        ("glow_dodge", 0.0, 1.0, 1.0, 0.0),
    ],
)
def test_csp_formulas(mode, a, c, alpha, want):
    assert float(csp_blend(mode, t(a), t(c), alpha)) == pytest.approx(want, abs=1e-6)


def test_csp_blend_rejects_unknown_mode():
    with pytest.raises(ValueError):
        csp_blend("multiply", t(0.5), t(0.5), 1.0)


@pytest.mark.parametrize("png", sorted(CSP_DIR.glob("*.png")), ids=lambda p: p.stem)
def test_reproduces_clip_studio_paint(png):
    px = torch.from_numpy(np.asarray(Image.open(png).convert("RGB")).astype(np.float32))
    bg = px[0, 0] / 255.0
    normal = px[BAND_ROWS["normal"]] / 255.0  # (W, 3)
    k = int((LAYER - bg).abs().argmax())  # channel where the layer differs most
    alpha = ((normal[:, k] - bg[k]) / (LAYER[k] - bg[k])).clamp(0.0, 1.0)[:, None]
    a, c = bg.expand_as(normal), LAYER.expand_as(normal)
    for mode in (
        "screen",
        "overlay",
        "color_dodge",
        "glow_dodge",
    ):  # CSP has no overlay_glow
        out = (csp_blend(mode, a, c, alpha) * 255.0).round()
        err = float((out - px[BAND_ROWS[mode]]).abs().max())
        assert err <= 3.0, f"{png.stem} {mode}: max error {err}/255"


def test_srgb_round_trip():
    x = torch.linspace(0.0, 1.0, 1001)
    assert torch.allclose(srgb_eotf(srgb_oetf(x)), x, atol=2e-6)


def _scene(img, glow, mode, strength=0.6, white_ev=0.0):
    return scene_blend(t(*img), t(*glow), mode, strength, white_ev)


@pytest.mark.parametrize("mode", CSP_MODES)
def test_scene_blend_is_the_encoded_formula_below_white(mode):
    img, glow = t(0.02, 0.18, 0.7), t(0.3, 0.05, 2.0)
    want = srgb_eotf(
        csp_blend(mode, srgb_oetf(img), srgb_oetf(glow.clamp_max(1.0)), 0.6)
    )
    assert torch.allclose(scene_blend(img, glow, mode, 0.6, 0.0), want, atol=1e-7)


@pytest.mark.parametrize("mode", CSP_MODES)
def test_light_above_white_passes_through(mode):
    out = _scene((3.0, 1.5, 40.0), (0.5, 2.0, 9.0), mode)
    assert out.tolist() == pytest.approx([3.0, 1.5, 40.0], rel=1e-6)


@pytest.mark.parametrize("mode", CSP_MODES)
def test_white_scales_the_whole_blend(mode):
    img, glow, w_ev = t(0.02, 0.4, 3.0), t(0.1, 0.9, 0.5), 2.5
    w = 2.0**w_ev
    scaled = scene_blend(img * w, glow * w, mode, 0.7, w_ev)
    assert torch.allclose(scaled, w * scene_blend(img, glow, mode, 0.7, 0.0), rtol=1e-5)


@pytest.mark.parametrize("mode", CSP_MODES)
def test_strength_zero_is_identity_and_above_one_clamps(mode):
    img, glow = t(0.02, -0.1, 7.0), t(0.5, 0.5, 0.5)
    assert scene_blend(img, glow, mode, 0.0, 0.0) is img
    assert torch.equal(
        scene_blend(img, glow, mode, 3.0, 0.0), scene_blend(img, glow, mode, 1.0, 0.0)
    )


@pytest.mark.parametrize("mode", CSP_MODES)
def test_negative_and_non_finite_pixels_stay_local(mode):
    img = t(-0.2, math.nan, math.inf)
    out = scene_blend(img, t(0.5, 0.5, 0.5), mode, 0.5, 0.0)
    assert float(out[0]) == pytest.approx(
        -0.2 + float(srgb_eotf(csp_blend(mode, t(0.0), srgb_oetf(t(0.5)), 0.5))),
        abs=1e-6,
    )
    assert math.isnan(out[1]) and out[2] == math.inf


def test_glow_dodge_brightens_more_than_screen_in_the_mids():
    assert _scene((0.18,), (0.3,), "glow_dodge") > _scene((0.18,), (0.3,), "screen")


def test_blend_glow_keeps_additive_and_conserve():
    img, glow, energy = t(0.1, 2.0), t(0.5, 0.25), t(0.0, 1.0)
    assert torch.equal(
        blend_glow(img, glow, energy, "additive", 2.0, 0.0), img + 2.0 * glow
    )
    assert torch.equal(
        blend_glow(img, glow, energy, "conserve", 2.0, 0.0), img + 1.0 * (glow - energy)
    )
    with pytest.raises(ValueError):
        blend_glow(img, glow, energy, "multiply", 1.0, 0.0)


def test_overlay_glow_hand_cases():
    # A < 0.5: m = 0.4, C/m = (1, 0.5, 0), O = 2·A·C/m = (0.5, 0.25, 0)
    out = csp_blend("overlay_glow", t(0.25, 0.25, 0.25), t(0.4, 0.2, 0.0), 1.0)
    assert out.tolist() == pytest.approx([0.35, 0.25, 0.15], abs=1e-6)
    # A > 0.5: m = 0.5, C/m = (1, 1, 0.5), O = 1 − 2(1−A)(1−C/m) = (1, 1, 0.75)
    out = csp_blend("overlay_glow", t(0.75, 0.75, 0.75), t(0.5, 0.5, 0.25), 0.5)
    assert out.tolist() == pytest.approx([0.8125, 0.8125, 0.75], abs=1e-6)


def test_overlay_glow_at_full_opacity_is_overlay():
    a, c = t(0.2, 0.6, 0.9), t(1.0, 0.3, 0.6)  # m = 1
    assert torch.allclose(
        csp_blend("overlay_glow", a, c, 0.7), csp_blend("overlay", a, c, 0.7), atol=1e-7
    )


def test_overlay_glow_leaves_glow_free_pixels_exactly():
    a = t(0.2, 0.6, 0.9)
    for c in (t(0.0, 0.0, 0.0), t(1e-9, 0.0, 5e-10)):  # none, and below GLOW_EPS
        assert torch.equal(csp_blend("overlay_glow", a, c, 1.0), a)
    tiny = csp_blend("overlay_glow", a, t(2e-6, 1e-6, 0.0), 1.0)
    assert torch.isfinite(tiny).all() and torch.allclose(tiny, a, atol=1e-5)
    img = t(0.02, -0.1, 7.0)[:, None]  # (3, 1): scene-linear pixel, no glow
    out = scene_blend(img, torch.zeros(3, 1), "overlay_glow", 0.8, 0.0)
    assert torch.equal(out, img)


def test_overlay_glow_differs_from_overlay_only_where_glow_is_partial():
    img = torch.tensor([[0.05, 0.05], [0.05, 0.05], [0.05, 0.05]])  # 2 pixels
    glow = torch.tensor([[0.0, 0.3], [0.0, 0.2], [0.0, 0.1]])  # none / some
    lit = scene_blend(img, glow, "overlay", 0.5, 0.0)
    soft = scene_blend(img, glow, "overlay_glow", 0.5, 0.0)
    assert (lit[:, 0] < img[:, 0]).all()  # literal overlay darkens the glow-free pixel
    assert torch.equal(soft[:, 0], img[:, 0])
    assert not torch.allclose(soft[:, 1], img[:, 1])
