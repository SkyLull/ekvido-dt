import io as pyio
import json
import math
import threading

import numpy as np
import pytest
import tifffile
from PIL import Image as PILImage

from ekvido.presets import sidecar_path
from ekvido.server import session as session_mod
from ekvido.server.session import QUICK_SKIP_S, VIEWS, UiSession, ViewRequest

ICC = b"\x00" * 40 + b"Linear Rec2020 RGB"


@pytest.fixture
def tif(tmp_path, monkeypatch):
    monkeypatch.setenv("EKVIDO_PRESET_DIR", str(tmp_path / "presets"))
    arr = np.full((200, 300, 3), 0.05, np.float32)
    arr[100, 150] = (8.0, 4.0, 1.2)
    arr[40:46, 40:46] = 16.0  # clipped plateau
    path = tmp_path / "shot.tif"
    tifffile.imwrite(path, arr, photometric="rgb", iccprofile=ICC)
    return path


def _size(webp: bytes):
    return PILImage.open(pyio.BytesIO(webp)).size


def test_opens_with_defaults_and_json_state(tif):
    s = UiSession(tif, device="cpu")
    st = s.state()
    json.dumps(st)
    assert st["size"] == [300, 200] and {"mist", "kazuki"} <= set(st["modules"])
    assert st["session"]["referred"] == "scene" and st["views"] == list(VIEWS)
    assert s.clip_level == pytest.approx(16.0 * 0.999)


def test_referred_override_and_display_clip(tif):
    s = UiSession(tif, referred="display", device="cpu")
    assert s.session.referred == "display" and s.clip_level == 1.0


def test_render_fits_viewport_preserving_aspect(tif):
    s = UiSession(tif, device="cpu")
    assert _size(s.render("proxy", ViewRequest(width=150, height=150))) == (150, 100)


@pytest.mark.parametrize("view", VIEWS)
def test_every_view_renders(tif, view):
    s = UiSession(tif, device="cpu")
    assert (
        _size(s.render("quick", ViewRequest(view=view, width=120, height=80)))[0] <= 120
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"crop": [0.9, 0.9, 2.0, 2.0]},
        {"crop": [0.5, 0.5, 0.5, 0.5]},
        {"crop": [-1, -1, 0.0, 0.0]},
        {"width": 0, "height": -5},
        {"width": 10**6, "height": 10**6, "exposure": float("nan")},
    ],
)
def test_hostile_view_requests_are_clamped(tif, raw):
    s = UiSession(tif, device="cpu")
    req = ViewRequest.from_dict(raw)
    x0, y0, x1, y1 = req.crop
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 and 1 <= req.width <= 8192
    assert math.isfinite(req.exposure)
    s.render(
        "quick",
        ViewRequest(
            **{
                **req.__dict__,
                "width": min(req.width, 64),
                "height": min(req.height, 64),
            }
        ),
    )


def test_unknown_view_rejected():
    with pytest.raises(ValueError, match="view"):
        ViewRequest.from_dict({"view": "x-ray"})


def test_apply_state_autosaves_and_restores(tif):
    s = UiSession(tif, device="cpu")
    st = s.state()["session"]
    st["stack"][1]["params"]["push_warm"] = 0.25
    st["referred"] = "display"
    s.apply_state(st)
    assert json.loads(sidecar_path(tif).read_text())["referred"] == "display"
    again = UiSession(tif, device="cpu")
    assert again.session.stack[1].params.push_warm == 0.25


def test_bad_state_keeps_previous(tif):
    s = UiSession(tif, device="cpu")
    before = s.session
    with pytest.raises(ValueError):
        s.apply_state({"stack": [{"module": "nope", "params": {}}]})
    assert s.session is before and not sidecar_path(tif).exists()


def test_broken_sidecar_warns_and_is_not_overwritten(tif):
    side = sidecar_path(tif)
    side.write_text('{"ekvido_version": 99}')
    with pytest.warns(UserWarning, match=side.name):
        s = UiSession(tif, device="cpu")
    assert s.session.stack[1].module == "kazuki"
    assert side.read_text() == '{"ekvido_version": 99}'


def test_quick_pass_skipped_once_proxy_is_fast(tif):
    s = UiSession(tif, device="cpu")
    assert s.should_quick()
    s.render("proxy", ViewRequest(width=64, height=64))
    assert s.last_proxy_s is not None
    assert s.should_quick() == (s.last_proxy_s >= QUICK_SKIP_S)


def test_render_full_writes_next_to_input_under_lock(tif, monkeypatch):
    s = UiSession(tif, device="cpu")
    seen = []
    orig = session_mod.Pipeline.run

    def run(self, *a, **kw):
        seen.append(s.lock.locked())
        return orig(self, *a, **kw)

    monkeypatch.setattr(session_mod.Pipeline, "run", run)
    out, secs = s.render_full()
    assert out == tif.with_name("shot.ekvido.tif") and out.exists() and secs >= 0
    assert seen and all(seen)
    with tifffile.TiffFile(out) as t:
        assert bytes(t.pages[0].tags[34675].value) == ICC


def test_render_waits_for_lock(tif):
    s = UiSession(tif, device="cpu")
    s.lock.acquire()
    done = threading.Event()
    threading.Thread(
        target=lambda: (s.render("quick", ViewRequest(width=32, height=32)), done.set())
    ).start()
    assert not done.wait(0.3)
    s.lock.release()
    assert done.wait(10)


def test_presets_save_and_load(tif):
    s = UiSession(tif, device="cpu")
    path = s.save_preset("mine", s.state()["session"]["stack"][1])
    assert path.exists() and s.state()["presets"]["mine"] == "kazuki"
    entry = s.preset_entry("mine")
    assert entry["module"] == "kazuki" and "push_warm" in entry["params"]
    with pytest.raises(KeyError):
        s.preset_entry("nope")


def test_unchanged_state_does_not_touch_broken_sidecar(tif):
    side = sidecar_path(tif)
    side.write_text('{"ekvido_version": 99}')
    with pytest.warns(UserWarning):
        s = UiSession(tif, device="cpu")
    s.apply_state(s.state()["session"])  # what the page sends on connect
    assert side.read_text() == '{"ekvido_version": 99}'


def test_unchanged_state_writes_nothing_changed_state_writes(tif):
    s = UiSession(tif, device="cpu")
    st = s.state()["session"]
    s.apply_state(st)
    assert not sidecar_path(tif).exists()
    st["stack"][1]["params"]["push_warm"] = 0.2
    s.apply_state(st)
    assert sidecar_path(tif).exists()


def test_instance_id_is_per_session(tif):
    a, b = UiSession(tif, device="cpu"), UiSession(tif, device="cpu")
    assert a.state()["instance"] != b.state()["instance"]
