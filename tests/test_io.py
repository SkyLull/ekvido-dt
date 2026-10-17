import warnings

import numpy as np
import pytest
import tifffile
import torch

from ekvido.io import (
    is_linear_rec2020,
    make_proxy,
    parse_focal_mm,
    read_tiff,
    write_tiff,
)

ICC_LINEAR = b"\x00" * 40 + b"d\x00e\x00s\x00c Linear Rec2020 RGB" + b"\x00" * 20
XMP = (
    b'<x:xmpmeta><rdf:Description exif:FocalLength="50/1" '
    b'darktable:xmp_version="5"/></x:xmpmeta>'
)


def _write_raw(path, arr, **kw):
    tifffile.imwrite(path, arr, photometric="rgb" if arr.ndim == 3 else None, **kw)


def test_float_roundtrip_keeps_headroom_and_negatives(tmp_path):
    arr = np.random.default_rng(0).normal(1.0, 2.0, (8, 12, 3)).astype(np.float32)
    src = tmp_path / "a.tif"
    _write_raw(src, arr, iccprofile=ICC_LINEAR)
    img = read_tiff(src)
    assert img.data.shape == (3, 8, 12) and img.data.dtype == torch.float32
    out = tmp_path / "b.tif"
    write_tiff(out, img.data, img)
    back = read_tiff(out)
    np.testing.assert_array_equal(back.data.numpy(), img.data.numpy())
    assert back.data.max() > 1 and back.data.min() < 0


def test_metadata_tags_copied(tmp_path):
    arr = np.ones((4, 4, 3), np.float32)
    src = tmp_path / "a.tif"
    _write_raw(
        src,
        arr,
        iccprofile=ICC_LINEAR,
        extratags=[
            (700, 1, len(XMP), XMP, True),
            (33723, 7, 8, b"\x01\x02\x03\x04" * 2, True),
        ],
    )
    img = read_tiff(src)
    assert img.focal_mm == 50.0
    out = tmp_path / "b.tif"
    write_tiff(out, img.data, img)
    with tifffile.TiffFile(out) as t:
        tags = t.pages[0].tags
        assert bytes(tags[700].value) == XMP
        assert bytes(tags[34675].value) == ICC_LINEAR
        assert 33723 in tags


def test_reads_deflate_float_predictor(tmp_path):
    arr = np.linspace(0, 3, 4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
    src = tmp_path / "p.tif"
    _write_raw(src, arr, compression="zlib", predictor=3, iccprofile=ICC_LINEAR)
    np.testing.assert_allclose(read_tiff(src).data.permute(1, 2, 0).numpy(), arr)


def test_warns_on_missing_or_foreign_icc(tmp_path):
    src = tmp_path / "n.tif"
    _write_raw(src, np.zeros((2, 2, 3), np.float32))
    with pytest.warns(UserWarning, match="linear Rec.2020"):
        read_tiff(src)


def test_no_warning_for_linear_rec2020(tmp_path):
    src = tmp_path / "l.tif"
    _write_raw(src, np.zeros((2, 2, 3), np.float32), iccprofile=ICC_LINEAR)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        read_tiff(src)


def test_uint16_scaled_with_warning(tmp_path):
    src = tmp_path / "u.tif"
    _write_raw(src, np.full((2, 2, 3), 65535, np.uint16), iccprofile=ICC_LINEAR)
    with pytest.warns(UserWarning, match="not float"):
        img = read_tiff(src)
    assert torch.allclose(img.data, torch.ones(3, 2, 2))


def test_rgba_drops_alpha_with_warning(tmp_path):
    src = tmp_path / "rgba.tif"
    tifffile.imwrite(
        src,
        np.ones((2, 2, 4), np.float32),
        photometric="rgb",
        extrasamples=["unassalpha"],
        iccprofile=ICC_LINEAR,
    )
    with pytest.warns(UserWarning, match="alpha"):
        img = read_tiff(src)
    assert img.data.shape == (3, 2, 2)


def test_greyscale_rejected(tmp_path):
    src = tmp_path / "g.tif"
    tifffile.imwrite(src, np.ones((2, 2), np.float32))
    with pytest.raises(ValueError, match="RGB"):
        read_tiff(src)


def test_is_linear_rec2020():
    assert is_linear_rec2020(ICC_LINEAR)
    assert not is_linear_rec2020(b"sRGB IEC61966-2.1")
    assert not is_linear_rec2020(None)


def test_parse_focal_variants():
    assert parse_focal_mm(XMP) == 50.0
    assert parse_focal_mm(b"<exif:FocalLength>35/2</exif:FocalLength>") == 17.5
    assert parse_focal_mm(b"<x/>") is None


def test_make_proxy_area_and_no_upsample():
    x = torch.ones(3, 400, 600)
    p = make_proxy(x, 0.06)  # 0.06 MP = 60000 px → 200x300
    assert p.shape == (3, 200, 300)
    assert torch.allclose(p, torch.ones_like(p))
    assert make_proxy(x, 10.0) is x


def test_write_without_source_image(tmp_path):
    out = tmp_path / "c.tif"
    write_tiff(out, torch.zeros(3, 2, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert read_tiff(out).data.shape == (3, 2, 2)


@pytest.mark.gpu
def test_make_proxy_on_gpu_with_odd_size(gpu):
    x = torch.rand(3, 403, 601, generator=torch.Generator().manual_seed(0))
    ref = make_proxy(x, 0.05)
    out = make_proxy(x.to(gpu), 0.05)
    assert out.device.type == gpu.type
    assert torch.allclose(out.cpu(), ref, atol=1e-5)
