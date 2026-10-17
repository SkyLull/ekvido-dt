import json
import warnings

import numpy as np
import pytest
import tifffile

from ekvido.cli import main
from ekvido.io import read_tiff
from ekvido.presets import Session, save_sidecar, sidecar_path

ICC = b"\x00" * 40 + b"Linear Rec2020 RGB"
XMP = b"<x:xmpmeta>hello</x:xmpmeta>"


@pytest.fixture
def scene(tmp_path):
    arr = np.full((90, 120, 3), 0.05, np.float32)
    arr[45, 60] = (8.0, 4.0, 1.2)
    path = tmp_path / "in.tif"
    tifffile.imwrite(
        path,
        arr,
        photometric="rgb",
        iccprofile=ICC,
        extratags=[(700, 1, len(XMP), XMP, True)],
    )
    return path


def _read(p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_tiff(p)


def test_chart_command(tmp_path):
    out = tmp_path / "chart.tif"
    assert main(["chart", "-o", str(out), "--width", "300", "--height", "200"]) == 0
    assert _read(out).data.shape == (3, 200, 300)


def test_render_with_preset_changes_image_and_keeps_metadata(scene, tmp_path):
    out = tmp_path / "out.tif"
    assert (
        main(
            [
                "render",
                str(scene),
                "-o",
                str(out),
                "--preset",
                "kazuki-default",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    a, b = _read(scene), _read(out)
    assert b.data.shape == a.data.shape and not np.allclose(b.data, a.data)
    assert b.icc == a.icc and b.tags == a.tags


def test_render_uses_sidecar_next_to_input(scene, tmp_path):
    save_sidecar(sidecar_path(scene), Session(referred="display"))
    out = tmp_path / "out.tif"
    assert main(["render", str(scene), "-o", str(out), "--device", "cpu"]) == 0


def test_render_explicit_sidecar_and_overrides(scene, tmp_path, capsys):
    side = tmp_path / "s.json"
    save_sidecar(side, Session())
    out = tmp_path / "out.tif"
    assert (
        main(
            [
                "render",
                str(scene),
                "-o",
                str(out),
                "--sidecar",
                str(side),
                "--referred",
                "display",
                "--clip-level",
                "0.9",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert "display" in capsys.readouterr().out


def test_render_needs_a_session(scene, tmp_path, capsys):
    assert main(["render", str(scene), "-o", str(tmp_path / "o.tif")]) == 2
    assert "sidecar" in capsys.readouterr().err


def test_render_refuses_to_overwrite_input(scene, capsys):
    assert (
        main(["render", str(scene), "-o", str(scene), "--preset", "kazuki-default"])
        == 2
    )
    assert "overwrite" in capsys.readouterr().err


def test_render_bad_preset_reports_cleanly(scene, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"module": "kazuki", "version": 99, "params": {}}))
    assert (
        main(
            ["render", str(scene), "-o", str(tmp_path / "o.tif"), "--preset", str(bad)]
        )
        == 2
    )
    assert "newer" in capsys.readouterr().err


def test_render_user_errors_are_clean(scene, tmp_path, capsys):
    # Review I5: these used to end in tracebacks.
    grey = tmp_path / "grey.tif"
    tifffile.imwrite(grey, np.ones((4, 4), np.float32))
    side_list = tmp_path / "list.json"
    side_list.write_text("[]")
    cases = [
        [
            "render",
            str(grey),
            "-o",
            str(tmp_path / "a.tif"),
            "--preset",
            "kazuki-default",
        ],
        [
            "render",
            str(tmp_path / "missing.tif"),
            "-o",
            str(tmp_path / "b.tif"),
            "--preset",
            "kazuki-default",
        ],
        [
            "render",
            str(scene),
            "-o",
            str(tmp_path / "c.tif"),
            "--sidecar",
            str(tmp_path / "nope.json"),
        ],
        [
            "render",
            str(scene),
            "-o",
            str(tmp_path / "d.tif"),
            "--sidecar",
            str(side_list),
        ],
        [
            "render",
            str(scene),
            "-o",
            str(tmp_path / "no/dir/e.tif"),
            "--preset",
            "kazuki-default",
        ],
    ]
    for argv in cases:
        assert main(argv) == 2, argv
        assert capsys.readouterr().err.startswith("ekvido: ")


def _mask_png(path, value):
    from PIL import Image as PILImage

    PILImage.fromarray(np.full((90, 120), value, np.uint8)).save(path)
    return path


def test_render_source_mask_of_zeros_leaves_the_image(scene, tmp_path):
    out = tmp_path / "out.tif"
    mask = _mask_png(tmp_path / "m.png", 0)
    args = ["render", str(scene), "-o", str(out), "--preset", "kazuki-default"]
    assert main([*args, "--device", "cpu", "--source-mask", str(mask)]) == 0
    assert np.array_equal(_read(out).data, _read(scene).data)
    assert (
        main(
            [
                *args,
                "--device",
                "cpu",
                "--source-mask",
                str(mask),
                "--invert-source-mask",
            ]
        )
        == 0
    )
    assert not np.allclose(_read(out).data, _read(scene).data)


def test_render_source_mask_errors_are_clean(scene, tmp_path, capsys):
    out = tmp_path / "out.tif"
    args = ["render", str(scene), "-o", str(out), "--preset", "kazuki-default"]
    assert main([*args, "--invert-source-mask"]) == 2
    small = tmp_path / "small.png"
    from PIL import Image as PILImage

    PILImage.fromarray(np.zeros((5, 5), np.uint8)).save(small)
    assert main([*args, "--source-mask", str(small)]) == 2
    assert "mask is 5x5" in capsys.readouterr().err
