import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

TOOL = Path(__file__).resolve().parents[1] / "tools" / "make_c_fixtures.py"
spec = importlib.util.spec_from_file_location("make_c_fixtures", TOOL)
mcf = importlib.util.module_from_spec(spec)
sys.modules["make_c_fixtures"] = mcf
spec.loader.exec_module(mcf)


def test_ekv_round_trip(tmp_path):
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    mcf.write_ekv(tmp_path / "a.ekv", a)
    assert (mcf.read_ekv(tmp_path / "a.ekv") == a).all()


def test_generates_every_kind(tmp_path):
    assert mcf.main([str(tmp_path)]) == 0
    lines = (tmp_path / "cases.txt").read_text().splitlines()
    kinds = {ln.split()[0] for ln in lines}
    assert kinds == {
        "psf",
        "conv",
        "mist",
        "kview",
        "energy",
        "clip",
        "blur",
        "wblur",
        "warmth",
        "bands",
        "hsv",
        "blend",
        "kazuki",
        "maskenergy",
    }
    for ln in lines:
        kind, name = ln.split()[:2]
        keys = dict(kv.split("=", 1) for kv in ln.split()[2:])
        if "input" in keys:
            assert list(tmp_path.glob(f"{keys['input']}.*.ekv")), ln
        if "mask" in keys:
            assert (tmp_path / f"{keys['mask']}.mask.ekv").exists(), ln
            assert keys["invert"] in ("0", "1"), ln
        if kind == "kazuki":
            assert np.isfinite(mcf.read_ekv(tmp_path / f"{name}.glow.ekv")).all()
            assert not np.isfinite(mcf.read_ekv(tmp_path / f"{name}.out.ekv")).all()
            assert 0 <= int(keys["mode"]) <= 6
        if kind == "psf":
            k = mcf.read_ekv(tmp_path / f"{name}.kernel.ekv")
            assert k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1
            assert math.isclose(float(k.sum()), 1.0, abs_tol=1e-4)
        if kind == "mist":
            assert np.isfinite(mcf.read_ekv(tmp_path / f"{name}.glow.ekv")).all()
            assert not np.isfinite(mcf.read_ekv(tmp_path / f"{name}.out.ekv")).all()
