import json

import pytest

from ekvido.modules.kazuki import KazukiParams
from ekvido.modules.mist import MistParams
from ekvido.pipeline import StackEntry
from ekvido.presets import (
    SIDECAR_VERSION,
    Session,
    default_stack,
    load_preset,
    load_sidecar,
    resolve_preset,
    save_preset,
    save_sidecar,
    session_from_dict,
    session_to_dict,
    shipped_presets,
    sidecar_path,
)


def test_sidecar_path(tmp_path):
    assert sidecar_path(tmp_path / "IMG_0001.tif").name == "IMG_0001.ekvido.json"


def test_default_stack():
    s = default_stack()
    assert [(e.module, e.enabled) for e in s] == [("mist", False), ("kazuki", True)]


def test_sidecar_roundtrip(tmp_path):
    s = Session(
        referred="display",
        clip_level=0.98,
        stack=[
            StackEntry("mist", MistParams(strength=0.3), enabled=False),
            StackEntry("kazuki", KazukiParams(warm_edge=(1.0, 0.4, 0.1))),
        ],
    )
    path = tmp_path / "x.ekvido.json"
    save_sidecar(path, s)
    assert json.loads(path.read_text())["ekvido_version"] == SIDECAR_VERSION
    assert load_sidecar(path) == s


def test_newer_sidecar_rejected():
    d = session_to_dict(Session())
    d["ekvido_version"] = SIDECAR_VERSION + 1
    with pytest.raises(ValueError, match="newer"):
        session_from_dict(d)


def test_unknown_module_in_sidecar_rejected():
    d = session_to_dict(Session())
    d["stack"][0]["module"] = "sparkle"
    with pytest.raises(ValueError, match="unknown module"):
        session_from_dict(d)


def test_newer_module_params_rejected():
    d = session_to_dict(Session())
    d["stack"][1]["version"] = 99
    with pytest.raises(ValueError, match="newer"):
        session_from_dict(d)


def test_bad_referred_rejected():
    d = session_to_dict(Session())
    d["referred"] = "linear"
    with pytest.raises(ValueError, match="referred"):
        session_from_dict(d)


def test_preset_roundtrip(tmp_path):
    e = StackEntry("mist", MistParams(halo_beta=2.0))
    save_preset(tmp_path / "p.json", e)
    assert load_preset(tmp_path / "p.json") == e


def test_shipped_presets_load():
    names = shipped_presets()
    assert {"mist-white", "mist-black", "kazuki-default"} <= set(names)
    for path in names.values():
        load_preset(path)
    black = load_preset(names["mist-black"]).params
    assert black.w_halo > black.w_veil


def test_resolve_preset(tmp_path):
    assert resolve_preset("kazuki-default") == shipped_presets()["kazuki-default"]
    p = tmp_path / "mine.json"
    p.write_text('{"module": "kazuki", "version": 1, "params": {}}')
    assert resolve_preset(str(p)) == p
    with pytest.raises(ValueError, match="no preset"):
        resolve_preset("nonexistent")


@pytest.mark.parametrize("module", ["kazuki", "mist"])
def test_v1_arm_params_migrate(module, recwarn):
    from ekvido.modules.base import get_module, params_from_dict

    old = {"arms": 4, "arm_length": 0.05, "arm_width": 0.001, "arm_falloff": 3.0}
    p = params_from_dict(get_module(module), 1, old)
    assert p.arms == 4 and p.arm_radius == pytest.approx(0.002)
    assert not [w for w in recwarn if "unknown params" in str(w.message)]


def test_slider_ranges_widened():
    from ekvido.modules.base import param_specs

    k = {s["name"]: s for s in param_specs(KazukiParams)}
    m = {s["name"]: s for s in param_specs(MistParams)}
    assert k["strength"]["max"] == 4.0
    for specs in (k, m):
        for name in (
            "halo_stretch_x",
            "halo_stretch_y",
            "arm_stretch_x",
            "arm_stretch_y",
        ):
            assert specs[name]["max"] == 5.0, name


def test_kazuki_wide_ships_hand_tune_1_shape():
    entry = load_preset(resolve_preset("kazuki-wide"))
    p = entry.params
    assert entry.module == "kazuki"
    assert (p.w_halo, p.w_veil, p.w_arms) == (0.0, 0.0, 1.0)
    assert (p.arm_radius, p.arm_beta, p.arm_sharpness) == (0.02, 4.0, 1.23)
    assert (p.arm_stretch_x, p.arm_stretch_y) == (4.0, 1.71)
    assert p.threshold_ev == KazukiParams().threshold_ev  # image-dependent: default
