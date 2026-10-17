import pytest

from ekvido.modules.kazuki import KazukiParams
from ekvido.pipeline import StackEntry
from ekvido.presets import (
    list_presets,
    load_preset,
    resolve_preset,
    save_user_preset,
    shipped_presets,
    user_preset_dir,
)


@pytest.fixture(autouse=True)
def preset_dir(tmp_path, monkeypatch):
    d = tmp_path / "presets"
    monkeypatch.setenv("EKVIDO_PRESET_DIR", str(d))
    return d


def test_dir_from_env(preset_dir):
    assert user_preset_dir() == preset_dir


def test_save_and_list_and_resolve(preset_dir):
    e = StackEntry("kazuki", KazukiParams(push_warm=0.9))
    path = save_user_preset("golden-hour", e)
    assert path.parent == preset_dir and path.exists()
    assert list_presets()["golden-hour"] == path
    assert resolve_preset("golden-hour") == path
    assert load_preset(path) == e


def test_user_shadows_shipped():
    path = save_user_preset(
        "kazuki-default", StackEntry("kazuki", KazukiParams(strength=1.0))
    )
    assert list_presets()["kazuki-default"] == path
    assert set(shipped_presets()) <= set(list_presets())


@pytest.mark.parametrize("bad", ["", "../escape", "a/b", "x" * 65, "sp ace"])
def test_bad_names_rejected(bad):
    with pytest.raises(ValueError, match="preset name"):
        save_user_preset(bad, StackEntry("kazuki", KazukiParams()))


def test_missing_dir_lists_shipped_only():
    assert list_presets() == shipped_presets()
