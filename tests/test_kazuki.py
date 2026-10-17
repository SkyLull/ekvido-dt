import math

import pytest
import torch

import ekvido.common.convolve as convolve
import ekvido.modules  # noqa: F401  (registers modules)
from ekvido.common.color import unit_luminance
from ekvido.common.convolve import fft_convolve
from ekvido.common.energy import extract_energy, luminance, positive
from ekvido.common.psf import build_kernel, split_bands
from ekvido.modules.base import REGISTRY, Context, kernel_spec, param_specs
from ekvido.modules.kazuki import Kazuki, KazukiParams

ORANGE, BLUE = (8.0, 4.0, 1.2), (1.5, 4.0, 10.0)


def _ctx(img, referred="scene", clip=math.inf, cache=None, stash=None):
    _, h, w = img.shape
    return Context(
        device=img.device,
        referred=referred,
        diag_px=math.hypot(h, w),
        clip_level=clip,
        input_key="img",
        cache=cache,
        stash=stash,
    )


def _img(points, h=201, w=201, r=2):
    img = torch.zeros(3, h, w)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    for y, x, rgb in points:
        disc = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        img[:, disc] = torch.tensor(rgb).view(3, 1)
    return img


def _glow(img, p, **kw):
    out = Kazuki().process(img, p, _ctx(img, **kw))
    return (out - img) / p.strength


def test_registered():
    assert REGISTRY["kazuki"].NAME == "kazuki"


def test_push_zero_keeps_source_colour():
    img = _img([(100, 100, ORANGE)])
    p = KazukiParams(push_warm=0.0, push_cool=0.0, strength=1.0)
    diag = math.hypot(201, 201)
    e = extract_energy(positive(img), p.threshold_ev, p.knee_ev, "scene")
    ref = fft_convolve(e, build_kernel(kernel_spec(p), diag))
    assert torch.allclose(_glow(img, p), ref, atol=1e-4)


def test_decomposition_matches_direct_recolour():
    img = _img([(60, 60, ORANGE), (140, 140, BLUE), (60, 140, (5.0, 5.0, 5.0))])
    p = KazukiParams(
        push_warm=0.8,
        push_cool=0.3,
        warm_core=(1.0, 0.8, 0.3),
        warm_edge=(1.0, 0.3, 0.05),
        cool_core=(0.2, 0.9, 1.0),
        cool_edge=(0.05, 0.1, 1.0),
        hue_radius=0.02,
        strength=1.0,
    )
    stash = {}
    glow = _glow(img, p, stash=stash)
    t = stash["warmth"]
    diag = math.hypot(201, 201)
    e = extract_energy(positive(img), p.threshold_ev, p.knee_ev, "scene")
    y = luminance(e)
    push = p.push_cool + t * (p.push_warm - p.push_cool)
    inner, outer = split_bands(build_kernel(kernel_spec(p), diag), p.band_split * diag)
    ref = torch.zeros_like(img)
    for band, warm, cool in (
        (inner, p.warm_core, p.cool_core),
        (outer, p.warm_edge, p.cool_edge),
    ):
        cu, wu = unit_luminance(cool).view(3, 1, 1), unit_luminance(warm).view(3, 1, 1)
        eb = (1 - push) * e + push * y * (cu + t * (wu - cu))
        ref += fft_convolve(eb, band)
    assert torch.allclose(glow, ref, atol=1e-4)


def test_warm_and_cool_sources_glow_their_own_way():
    img = _img([(100, 40, ORANGE), (100, 160, BLUE)])
    g = _glow(img, KazukiParams(push_warm=1.0, push_cool=1.0))
    left, right = g[:, 100, 48], g[:, 100, 152]
    assert left[0] > left[2] and right[2] > right[0]


def test_clipped_disc_takes_fringe_colour_else_bias():
    ring = _img([(100, 100, (2.0, 1.0, 0.4))], r=9)
    disc = _img([(100, 100, (4.0, 4.0, 4.0))], r=6)
    with_fringe = torch.where(disc > 0, disc, ring)
    p = KazukiParams(push_warm=1.0, push_cool=1.0, neutral_bias=0.0, hue_radius=0.05)
    g = _glow(with_fringe, p, clip=3.9)
    assert g[0, 100, 115] > g[2, 100, 115]  # fringe says warm, beats the cool bias
    bare = _glow(disc, p, clip=3.9)
    assert bare[2, 100, 115] > bare[0, 100, 115]  # no fringe → bias (cool)
    warm_bias = _glow(
        disc, KazukiParams(push_warm=1.0, push_cool=1.0, neutral_bias=1.0), clip=3.9
    )
    assert warm_bias[0, 100, 115] > warm_bias[2, 100, 115]


def test_below_threshold_adds_nothing():
    img = torch.full((3, 50, 50), 0.1)  # −3.3 EV
    out = Kazuki().process(img, KazukiParams(), _ctx(img))
    assert torch.allclose(out, img, atol=1e-7)


def test_glow_scales_with_intensity():
    p = KazukiParams()
    a = _glow(_img([(100, 100, ORANGE)]), p)
    b = _glow(_img([(100, 100, tuple(2 * c for c in ORANGE))]), p)
    assert torch.allclose(b, 2 * a, atol=1e-4)


def test_display_mode_uses_linear_threshold():
    img = _img([(100, 100, (0.9, 0.9, 0.9))])
    on = _glow(
        img,
        KazukiParams(threshold_lin=0.85, knee_lin=0.02),
        referred="display",
        clip=1.0,
    )
    off = _glow(
        img,
        KazukiParams(threshold_lin=0.97, knee_lin=0.01),
        referred="display",
        clip=1.0,
    )
    assert on.abs().sum() > 0 and off.abs().sum() == 0


def test_conserve_blend():
    img = _img([(100, 100, ORANGE)])
    add = _glow(img, KazukiParams(strength=0.5))
    p = KazukiParams(strength=0.5, blend="conserve")
    e = extract_energy(positive(img), p.threshold_ev, p.knee_ev, "scene")
    out = Kazuki().process(img, p, _ctx(img))
    assert torch.allclose(out, img + 0.5 * (add - e), atol=1e-5)


@pytest.mark.parametrize("shape", [(16, 16), (8, 400)])
def test_small_and_extreme_shapes(shape):
    img = torch.rand(3, *shape) * 4
    out = Kazuki().process(img, KazukiParams(), _ctx(img))
    assert out.shape == img.shape and torch.isfinite(out).all()


def test_nan_pixel_stays_local():
    img = _img([(100, 100, ORANGE)])
    img[1, 3, 3] = float("nan")
    out = Kazuki().process(img, KazukiParams(), _ctx(img))
    assert torch.isnan(out[1, 3, 3])
    out[1, 3, 3] = 0
    assert torch.isfinite(out).all()


def test_colour_and_push_changes_do_no_fft(monkeypatch):
    calls = []
    orig = convolve.Convolver.apply
    monkeypatch.setattr(
        convolve.Convolver, "apply", lambda s, k: calls.append(1) or orig(s, k)
    )
    img, cache = _img([(100, 100, ORANGE)]), {}
    Kazuki().process(img, KazukiParams(), _ctx(img, cache=cache))
    n = len(calls)
    Kazuki().process(
        img,
        KazukiParams(warm_core=(1.0, 1.0, 0.0), push_warm=0.2, strength=2.0),
        _ctx(img, cache=cache),
    )
    assert len(calls) == n
    Kazuki().process(img, KazukiParams(arms=6), _ctx(img, cache=cache))
    assert len(calls) == 2 * n


def test_stash_keys():
    img, stash = _img([(100, 100, ORANGE)]), {}
    Kazuki().process(img, KazukiParams(), _ctx(img, stash=stash))
    assert set(stash) == {"energy", "warmth", "clip", "glow", "kernel"}


@pytest.mark.parametrize("name", ["mist", "kazuki"])
def test_param_metadata_complete(name):
    for s in param_specs(REGISTRY[name].Params):
        assert s["desc"], s["name"]
        if s["kind"] in ("float", "int"):
            assert s["min"] <= s["default"] <= s["max"], s["name"]
        if s["kind"] == "color":
            assert len(s["default"]) == 3
        if s["kind"] == "choice":
            assert s["default"] in s["choices"]


@pytest.mark.gpu
def test_device_parity(gpu):
    img = _img([(60, 60, ORANGE), (140, 140, BLUE)])
    ref = Kazuki().process(img, KazukiParams(), _ctx(img))
    g = img.to(gpu)
    out = Kazuki().process(g, KazukiParams(), _ctx(g)).cpu()
    assert torch.allclose(out, ref, atol=1e-3)


def test_chart_clipped_discs_take_fringe_colour():
    # Review I2: a 256-bright clipped core drowned its 2-bright fringe's colour.
    from ekvido.chart import make_chart
    from ekvido.common.energy import estimate_clip_level

    img = make_chart(900, 600)
    stash = {}
    Kazuki().process(
        img,
        KazukiParams(neutral_bias=0.8),
        _ctx(img, clip=estimate_clip_level(img), stash=stash),
    )
    t = stash["warmth"]
    y = int(0.65 * 600)
    orange, blue, bare = (t[y, int(900 * i / 4)].item() for i in (1, 2, 3))
    assert orange > 0.9 and blue < 0.1 and bare == pytest.approx(0.8, abs=0.05)


def test_conserve_leaves_non_highlights_alone():
    # Review I7: conserve halved every pixel instead of moving highlight energy.
    img = _img([(100, 100, ORANGE)])
    img += 0.05
    out = Kazuki().process(img, KazukiParams(strength=0.5, blend="conserve"), _ctx(img))
    assert out[:, 5, 5].tolist() == pytest.approx(img[:, 5, 5].tolist(), abs=1e-6)


def test_default_shape_is_hand_tune_2():
    # Shape values chosen by eye on target-1 (2026-10-14); image-dependent values
    # (thresholds, neutral point, strength, colours) keep their own defaults.
    p = KazukiParams()
    shape = {
        "halo_radius": 0.006,
        "halo_beta": 1.3,
        "halo_stretch_x": 1.85,
        "halo_stretch_y": 1.0,
        "w_halo": 1.0,
        "w_veil": 0.0,
        "band_split": 0.006,
        "arms": 4,
        "arm_angle": 0.0,
        "arm_radius": 0.001,
        "arm_beta": 0.96,
        "arm_sharpness": 0.32,
        "arm_stretch_x": 2.93,
        "arm_stretch_y": 0.58,
        "w_arms": 0.2,
    }
    assert {k: getattr(p, k) for k in shape} == shape


@pytest.mark.parametrize("mode", ["screen", "overlay", "color_dodge", "glow_dodge"])
def test_csp_blend_modes_lay_the_glow_on(mode):
    from ekvido.common.blend import scene_blend

    img = _img([(100, 100, ORANGE)])
    img += 0.05
    p = KazukiParams(strength=0.7, blend=mode, blend_white_ev=1.5)
    stash = {}
    out = Kazuki().process(img, p, _ctx(img, stash=stash))
    want = scene_blend(img, stash["glow"], mode, 0.7, 1.5)
    assert torch.allclose(out, want, atol=1e-6)
    if mode != "overlay":  # overlay darkens where the glow is below mid-grey
        assert (out[:, 100, 112] > img[:, 100, 112]).all()  # the halo brightens


def test_overlay_with_no_glow_darkens_as_the_spec_says():
    # Spec §5.6 applies overlay with the glow as an opaque layer, so where there is no
    # glow (C = 0) a background below mid-grey goes towards black by `strength`.
    from ekvido.common.display import srgb_eotf, srgb_oetf

    img = torch.full((3, 40, 40), 0.05)
    out = Kazuki().process(img, KazukiParams(strength=0.5, blend="overlay"), _ctx(img))
    half = srgb_eotf(0.5 * srgb_oetf(torch.tensor(0.05)))  # encoded value halved
    assert torch.allclose(out, half.expand_as(img), atol=1e-6)


def test_blend_choices_and_white_reach_the_ui():
    k = {s["name"]: s for s in param_specs(KazukiParams)}
    assert k["blend"]["choices"] == [
        "additive",
        "conserve",
        "screen",
        "overlay",
        "color_dodge",
        "glow_dodge",
        "overlay_glow",
    ]
    w = k["blend_white_ev"]
    assert (w["kind"], w["min"], w["max"], w["default"]) == ("float", -4.0, 8.0, 0.0)


def test_old_sidecar_without_blend_white_loads_with_default():
    from ekvido.modules.base import get_module, params_from_dict

    p = params_from_dict(get_module("kazuki"), 2, {"blend": "conserve"})
    assert p.blend == "conserve" and p.blend_white_ev == 0.0


# ---- source mask (spec §5.7) ----


def _masked(img, p, mask, invert=False, **kw):
    ctx = _ctx(img, **kw)
    ctx.source_mask, ctx.source_mask_invert = mask, invert
    return Kazuki().process(img, p, ctx)


def test_source_mask_of_ones_changes_nothing():
    img = _img([(60, 60, ORANGE), (140, 140, BLUE)])
    p = KazukiParams(strength=0.8)
    plain = Kazuki().process(img, p, _ctx(img))
    assert torch.equal(_masked(img, p, torch.ones(201, 201)), plain)


@pytest.mark.parametrize("blend", ["additive", "conserve", "screen", "overlay_glow"])
def test_source_mask_of_zeros_emits_no_glow(blend):
    img = _img([(60, 60, ORANGE), (140, 140, BLUE)]) + 0.05
    out = _masked(img, KazukiParams(strength=0.8, blend=blend), torch.zeros(201, 201))
    if blend == "screen":  # sRGB round trip of a zero glow: equal to float precision
        assert torch.allclose(out, img, rtol=1e-6, atol=1e-7)
    else:
        assert torch.equal(out, img)


def test_masked_out_source_is_dark_but_glow_crosses_the_mask_edge():
    both = _img([(100, 90, ORANGE), (100, 150, BLUE)])
    orange = _img([(100, 90, ORANGE)])
    mask = torch.zeros(201, 201)
    mask[:, :120] = 1.0  # only the left part emits; the blue source sits outside
    p = KazukiParams(strength=1.0)
    glow = _masked(both, p, mask) - both
    ref = Kazuki().process(orange, p, _ctx(orange)) - orange
    assert torch.allclose(glow, ref, atol=1e-6)  # blue emits nothing
    assert (glow[:, 100, 125] > 1e-4).all()  # orange's glow spreads past x = 120


def test_inverted_mask_is_one_minus_mask():
    img = _img([(60, 60, ORANGE), (140, 140, BLUE)])
    mask = torch.linspace(0.0, 1.0, 201).expand(201, 201).contiguous()
    p = KazukiParams(strength=0.6)
    assert torch.allclose(
        _masked(img, p, mask, invert=True), _masked(img, p, 1.0 - mask), atol=1e-6
    )


def test_masked_energy_feeds_conserve_and_the_energy_preview():
    from ekvido.common.energy import mask_energy

    img = _img([(60, 60, ORANGE), (140, 140, BLUE)])
    mask = torch.zeros(201, 201)
    mask[:100] = 0.5
    stash = {}
    p = KazukiParams(strength=0.5, blend="conserve")
    _masked(img, p, mask, stash=stash)
    e = extract_energy(positive(img), p.threshold_ev, p.knee_ev, "scene")
    assert torch.equal(stash["energy"], mask_energy(e, mask))
    assert stash["energy"][:, 140, 140].abs().sum() == 0  # blue source masked out


def test_mask_energy_clamps_and_cleans_the_mask():
    from ekvido.common.energy import mask_energy

    e = torch.ones(3, 1, 5)
    m = torch.tensor([[float("nan"), -1.0, 0.25, 2.0, float("inf")]])
    assert mask_energy(e, m)[0].tolist() == [[0.0, 0.0, 0.25, 1.0, 0.0]]
    assert mask_energy(e, m, invert=True)[0].tolist() == [[1.0, 1.0, 0.75, 0.0, 1.0]]


def test_changed_mask_recomputes_with_a_cache():
    img = _img([(60, 60, ORANGE), (140, 140, BLUE)])
    p, cache = KazukiParams(), {}
    a, b = torch.ones(201, 201), torch.zeros(201, 201)
    b[:, :100] = 1.0
    _masked(img, p, a, cache=cache)
    assert torch.equal(_masked(img, p, b, cache=cache), _masked(img, p, b))


def test_source_mask_must_match_the_image():
    img = _img([(60, 60, ORANGE)])
    with pytest.raises(ValueError, match="source mask"):
        _masked(img, KazukiParams(), torch.ones(10, 10))
