import importlib.util
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from ekvido.modules.kazuki import KazukiParams
from ekvido.modules.mist import MistParams
from ekvido.presets import load_preset, resolve_preset

TOOL = Path(__file__).resolve().parents[1] / "tools" / "dt_compare.py"
spec = importlib.util.spec_from_file_location("dt_compare", TOOL)
dtc = importlib.util.module_from_spec(spec)
sys.modules["dt_compare"] = dtc
spec.loader.exec_module(dtc)

FORK_MODULE = (
    Path(__file__).resolve().parents[2] / "darktable" / "src" / "iop" / "ekvidokazuki.c"
)


def test_mist_params_layout_matches_the_c_struct():
    p = MistParams(strength=0.3, arms=4, w_arms=0.25)
    blob = dtc.pack_mist_params(p)
    assert len(blob) == 68
    vals = struct.unpack("<9fi7f", blob)
    assert vals[0] == np.float32(0.3) and vals[9] == 4 and vals[16] == np.float32(0.25)


def test_kazuki_blob_is_184_bytes_and_round_trips():
    c = dtc.kazuki_preset("screen")
    blob = dtc.pack_kazuki(c)
    assert len(blob) == 184 == struct.calcsize(dtc.KAZUKI_FORMAT)
    assert dtc.unpack_kazuki(blob) == c
    assert len(dtc.pack_kazuki_v1(c)) == 152 and blob[:152] == dtc.pack_kazuki_v1(c)
    named = {**c, "source_mask_op": b"exposure", "source_mask_instance": 1}
    assert dtc.unpack_kazuki(dtc.pack_kazuki(named)) == named
    assert (c["blend"], c["clip_mode"], c["arms"]) == (2, 1, 6)


def test_kazuki_c_defaults_are_the_python_defaults():
    p, clip = dtc.kazuki_c_to_py(dtc.kazuki_preset("default"))
    ref = KazukiParams()
    assert clip is None
    for name in KazukiParams.__dataclass_fields__:
        if name in ("threshold_lin", "knee_lin"):
            continue  # display mode is not ported
        got, want = getattr(p, name), getattr(ref, name)
        assert got == pytest.approx(want, rel=1e-6, abs=1e-6), name


def test_kazuki_wide_matches_the_shipped_preset():
    shipped = load_preset(resolve_preset("kazuki-wide")).params
    p, _ = dtc.kazuki_c_to_py(dtc.kazuki_preset("wide"))
    for name in dtc.KAZUKI_PRESETS["wide"]:
        assert getattr(p, name) == pytest.approx(getattr(shipped, name)), name


def test_manual_clip_and_blend_mode_map_across():
    p, clip = dtc.kazuki_c_to_py(dtc.kazuki_preset("screen"))
    assert p.blend == "screen" and clip == pytest.approx(4.0)
    assert (
        dtc.kazuki_c_to_py(dtc.kazuki_preset("overlayglow"))[0].blend == "overlay_glow"
    )
    assert p.warm_edge != KazukiParams().warm_edge


def test_xmp_carries_the_history_entry():
    xmp = dtc.module_xmp("kazuki", "default")
    assert 'darktable:operation="ekvidokazuki"' in xmp
    assert 'darktable:modversion="2"' in xmp
    assert dtc.pack_kazuki(dtc.kazuki_preset("default")).hex() in xmp
    assert 'darktable:auto_presets_applied="1"' in xmp
    assert 'darktable:operation="ekvidomist"' in dtc.module_xmp("mist", "black")
    assert 'darktable:modversion="1"' in dtc.module_xmp("mist", "black")
    legacy = dtc.module_xmp("kazuki", "default", legacy_v1=True)
    assert 'darktable:modversion="1"' in legacy
    assert dtc.pack_kazuki_v1(dtc.kazuki_preset("default")).hex() in legacy


def test_read_pfm_flips_rows(tmp_path):
    h, w = 2, 3
    rows = np.arange(h * w * 3, dtype="<f4").reshape(h, w, 3)  # file: bottom row first
    path = tmp_path / "a.pfm"
    path.write_bytes(f"PF\n{w} {h}\n-1.0\n".encode() + rows.tobytes())
    img = dtc.read_pfm(path)
    assert img.shape == (3, h, w)
    assert img[0, 0, 0] == rows[1, 0, 0] and img[0, 1, 0] == rows[0, 0, 0]


def test_compare_reports_mismatched_non_finite():
    want = np.array([[[1.0, np.nan, np.inf]]], dtype=np.float32)
    ok = np.array([[[1.00001, np.nan, np.inf]]], dtype=np.float32)
    bad = np.array([[[1.0, 0.0, np.inf]]], dtype=np.float32)
    worst, _, peak, same = dtc.compare(ok, want)
    assert same and worst == pytest.approx(1e-5, rel=0.1) and peak == 1.0
    assert not dtc.compare(bad, want)[3]


def test_dump_dir_mode_compares_existing_dumps(tmp_path, monkeypatch):
    src = np.random.default_rng(1).random((3, 6, 8), dtype=np.float32) * 3
    out = src * 1.5
    exp = tmp_path / "export"
    exp.mkdir()
    for kind, arr in (("in", src), ("out", out)):
        rows = arr.transpose(1, 2, 0)[::-1].astype("<f4")
        (exp / f"0003_ekvidokazuki_cpu_{kind}_C.pfm").write_bytes(
            b"PF\n8 6\n-1.0\n" + rows.tobytes()
        )
    monkeypatch.setattr(dtc, "python_reference", lambda m, p, s, *a: s * 1.5)
    assert dtc.main(["--module", "kazuki", "--dump-dir", str(tmp_path)]) == 0
    monkeypatch.setattr(dtc, "python_reference", lambda m, p, s, *a: s * 1.6)
    assert dtc.main(["--module", "kazuki", "--dump-dir", str(tmp_path)]) == 1


def test_python_reference_runs_on_a_small_frame():
    src = np.full((3, 20, 30), 0.05, np.float32)
    src[:, 10, 15] = (8.0, 4.0, 1.2)
    for preset in dtc.KAZUKI_PRESETS:
        out = dtc.python_reference("kazuki", preset, src)
        assert out.shape == src.shape and np.isfinite(out).all()


def test_fork_module_struct_matches_fields_and_defaults():
    if not FORK_MODULE.exists():
        pytest.skip("ekvidokazuki.c not written yet (Plan B Task 9)")
    src = FORK_MODULE.read_text()
    body = src[src.index("typedef struct dt_iop_ekvidokazuki_params_t") :]
    body = body[: body.index("} dt_iop_ekvidokazuki_params_t;")]
    enums = {"EKVIDOKAZUKI_BLEND_ADDITIVE": 0, "EKVIDOKAZUKI_CLIP_AUTO": 0, "FALSE": 0}
    found = re.findall(r"^\s*\w+\s+(\w+);\s*//.*?\$DEFAULT:\s*([-\w.]+)", body, re.M)
    assert re.search(r"^\s*char source_mask_op\[20\];", body, re.M)  # no $DEFAULT: ""
    assert [n for n, _ in found] == [
        f for f in dtc.KAZUKI_FIELDS if f != "source_mask_op"
    ]
    for name, value in found:
        want = dtc.KAZUKI_DEFAULTS[name]
        got = enums[value] if value in enums else float(value)
        assert got == pytest.approx(want, rel=1e-6, abs=1e-6), name


C_ENUMS = {
    **{f"EKVIDOKAZUKI_BLEND_{m.upper()}": i for i, m in enumerate(dtc.BLEND_MODES)},
    "EKVIDOKAZUKI_CLIP_AUTO": 0,
    "EKVIDOKAZUKI_CLIP_MANUAL": 1,
    "FALSE": 0,
    "TRUE": 1,
}
C_ASSIGN = r"\b(\w+)\s*=\s*([-\w.]+?)f?\s*[,;}]"


def _c_value(token: str) -> float:
    return C_ENUMS[token] if token in C_ENUMS else float(token)


def _fork_c_defaults_and_wide() -> tuple[dict, dict]:
    """The fork's `_defaults()` and its *kazuki wide* preset (defaults + overrides)."""
    if not FORK_MODULE.exists():
        pytest.skip("ekvidokazuki.c not written yet (Plan B Task 9)")
    src = FORK_MODULE.read_text()
    body = src[src.index("static dt_iop_ekvidokazuki_params_t _defaults(void)") :]
    body = body[body.index("{") : body.index("};") + 1]
    defaults = {n: _c_value(v) for n, v in re.findall(r"\." + C_ASSIGN, body)}
    presets = src[src.index('_("kazuki default")') : src.index('_("kazuki wide")')]
    overrides = {n: _c_value(v) for n, v in re.findall(r"\bp\." + C_ASSIGN, presets)}
    return defaults, {**defaults, **overrides}


def test_fork_c_defaults_function_matches_the_defaults():
    defaults, _ = _fork_c_defaults_and_wide()
    assert list(defaults) == list(dtc.KAZUKI_FIELDS[:38])  # source mask: zero = none
    for name, want in dtc.KAZUKI_DEFAULTS.items():
        if name.startswith("source_mask"):
            continue
        assert defaults[name] == pytest.approx(want, rel=1e-6, abs=1e-6), name


def test_fork_c_wide_preset_matches_the_shipped_preset():
    _, wide = _fork_c_defaults_and_wide()
    want = dtc.KAZUKI_WIDE  # kazuki-wide.json over the Python defaults
    assert wide["blend"] == dtc.BLEND_MODES.index(want["blend"])
    assert (wide["clip_mode"], wide["clip_ev"]) == (0, 0.0)  # auto, as Python
    for name, value in wide.items():
        if name in want and name != "blend":
            assert value == pytest.approx(want[name], rel=1e-6, abs=1e-6), name
    for colour in ("warm_core", "warm_edge", "cool_core", "cool_edge"):
        rgb = dtc.hsv_to_rgb(wide[f"{colour}_hue"], wide[f"{colour}_sat"])
        assert rgb == pytest.approx(want[colour], rel=1e-6, abs=1e-6), colour


def test_pass_rule_enforces_max_mean_and_non_finite():
    assert dtc.passes(5e-5, 1e-7, 1.0, True)
    assert not dtc.passes(5e-4, 1e-7, 1.0, True)  # max too large
    assert not dtc.passes(5e-5, 5e-6, 1.0, True)  # small max, high mean
    assert not dtc.passes(5e-5, 1e-7, 1.0, False)  # non-finite mismatch


def test_main_fails_on_uniformly_small_offset(tmp_path, monkeypatch):
    src = np.ones((3, 6, 8), np.float32)
    exp = tmp_path / "export"
    exp.mkdir()
    for kind, arr in (("in", src), ("out", src + 5e-5)):  # max and mean both 5e-5
        rows = arr.transpose(1, 2, 0)[::-1].astype("<f4")
        (exp / f"0003_ekvidomist_cpu_{kind}_C.pfm").write_bytes(
            b"PF\n8 6\n-1.0\n" + rows.tobytes()
        )
    monkeypatch.setattr(dtc, "python_reference", lambda m, p, s, *a: s)
    assert dtc.main(["--module", "mist", "--dump-dir", str(tmp_path)]) == 1


def test_read_pfm_reads_grey_mask_dumps(tmp_path):
    h, w = 2, 3
    rows = np.arange(h * w, dtype="<f4").reshape(h, w)  # file: bottom row first
    path = tmp_path / "m.pfm"
    path.write_bytes(f"Pf\n{w} {h}\n-1.0\n".encode() + rows.tobytes())
    m = dtc.read_pfm(path)
    assert m.shape == (h, w) and m[0, 0] == rows[1, 0] and m[1, 2] == rows[0, 2]


def _xmp_with(entries: str, history_end: int) -> str:
    return (
        f'<x:xmpmeta><rdf:RDF><rdf:Description darktable:history_end="{history_end}">'
        f"<darktable:history><rdf:Seq>{entries}</rdf:Seq></darktable:history>"
        "</rdf:Description></rdf:RDF></x:xmpmeta>"
    )


def test_kazuki_params_from_a_darktable_xmp(tmp_path):
    import base64
    import zlib

    c = {**dtc.kazuki_preset("screen"), "source_mask_op": b"exposure"}
    later = {**c, "strength": 2.0}
    gz = base64.b64encode(zlib.compress(dtc.pack_kazuki(c))).decode()
    entries = (
        '<rdf:li darktable:num="0" darktable:operation="exposure" darktable:modversion="6" darktable:params="00"/>'
        f'<rdf:li darktable:num="1" darktable:operation="ekvidokazuki" darktable:modversion="2" darktable:params="gz12{gz}"/>'
        f'<rdf:li darktable:num="2" darktable:operation="ekvidokazuki" darktable:modversion="2" darktable:params="{dtc.pack_kazuki(later).hex()}"/>'
    )
    path = tmp_path / "e.xmp"
    path.write_text(_xmp_with(entries, 2))  # entry 2 is past history_end
    assert dtc.kazuki_params_from_xmp(path) == c
    path.write_text(_xmp_with(entries, 3))
    assert dtc.kazuki_params_from_xmp(path)["strength"] == 2.0
    path.write_text(_xmp_with(entries.replace('modversion="2"', 'modversion="1"'), 3))
    with pytest.raises(ValueError, match="version 1"):
        dtc.kazuki_params_from_xmp(path)


def test_python_reference_applies_the_source_mask():
    src = np.full((3, 20, 30), 0.05, np.float32)
    src[:, 10, 5] = (8.0, 4.0, 1.2)
    c = {**dtc.kazuki_preset("default"), "source_mask_op": b"exposure"}
    zeros = np.zeros((20, 30), np.float32)
    assert np.array_equal(dtc.python_reference("kazuki", "default", src, c, zeros), src)
    ones = np.ones((20, 30), np.float32)
    plain = dtc.python_reference("kazuki", "default", src)
    assert np.array_equal(
        dtc.python_reference("kazuki", "default", src, c, ones), plain
    )
    inverted = {**c, "source_mask_invert": 1}
    assert np.array_equal(
        dtc.python_reference("kazuki", "default", src, inverted, ones), src
    )


def test_missing_source_means_no_glow():
    src = np.full((3, 20, 30), 0.05, np.float32)
    src[:, 10, 5] = (8.0, 4.0, 1.2)
    assert dtc.kazuki_preset("missingsource")["source_mask_op"] == b"exposure"
    assert np.array_equal(dtc.python_reference("kazuki", "missingsource", src), src)


def test_xmp_mode_needs_the_mask_it_promises(tmp_path, monkeypatch, capsys):
    c = {**dtc.kazuki_preset("default"), "source_mask_op": b"exposure"}
    entry = f'<rdf:li darktable:num="0" darktable:operation="ekvidokazuki" darktable:modversion="2" darktable:params="{dtc.pack_kazuki(c).hex()}"/>'
    xmp = tmp_path / "e.xmp"
    xmp.write_text(_xmp_with(entry, 1))
    exp = tmp_path / "dump" / "export"
    exp.mkdir(parents=True)
    src = np.ones((3, 6, 8), np.float32)
    for kind in ("in", "out"):
        rows = src.transpose(1, 2, 0)[::-1].astype("<f4")
        (exp / f"0003_ekvidokazuki_cpu_{kind}_C.pfm").write_bytes(
            b"PF\n8 6\n-1.0\n" + rows.tobytes()
        )
    args = [
        "--module",
        "kazuki",
        "--xmp",
        str(xmp),
        "--dump-dir",
        str(tmp_path / "dump"),
    ]
    assert dtc.main(args) == 2  # no source_mask dump
    assert "received none" in capsys.readouterr().out
    md = tmp_path / "dump" / "ekvidokazuki"
    md.mkdir()
    (md / "0004_source_mask_cpu_M.pfm").write_bytes(
        b"Pf\n8 6\n-1.0\n" + np.zeros((6, 8), "<f4").tobytes()
    )
    assert dtc.main(args) == 0  # zero mask: no glow, output = input


def _write_pfm(path: Path, w: int = 2, h: int = 1) -> None:
    data = np.arange(w * h * 3, dtype="<f4")
    path.write_bytes(f"PF\n{w} {h}\n-1.0\n".encode() + data.tobytes())


def test_load_dump_reads_gpu_dumps_only_with_opencl(tmp_path):
    exp = tmp_path / "export"
    exp.mkdir()
    for kind in ("in", "out"):
        _write_pfm(exp / f"0001_ekvidokazuki_GPU_{kind}_C.pfm")
    src, out = dtc.load_dump(tmp_path, "kazuki", opencl=True)
    assert src.shape == out.shape == (3, 1, 2)
    with pytest.raises(FileNotFoundError):
        dtc.load_dump(tmp_path, "kazuki")  # CPU dumps expected: none there


def test_load_dump_gpu_missing_says_fallback(tmp_path):
    exp = tmp_path / "export"
    exp.mkdir()
    for kind in ("in", "out"):
        _write_pfm(exp / f"0001_ekvidokazuki_cpu_{kind}_C.pfm")
    with pytest.raises(FileNotFoundError, match="GPU"):
        dtc.load_dump(tmp_path, "kazuki", opencl=True)


def test_run_darktable_opencl_flags(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd

        class R:
            returncode = 0
            stdout = stderr = ""

        return R()

    monkeypatch.setattr(dtc.subprocess, "run", fake_run)
    dtc.run_darktable("cli", "in.tif", "mist", "default", tmp_path, opencl=True)
    assert "--disable-opencl" not in seen["cmd"] and "opencl=TRUE" in seen["cmd"]
    dtc.run_darktable("cli", "in.tif", "mist", "default", tmp_path)
    assert "--disable-opencl" in seen["cmd"]
