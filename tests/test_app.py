import asyncio
import io as pyio
import json
import struct

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from ekvido.server.app import LatestBox, create_app
from ekvido.server.session import UiSession

ICC = b"\x00" * 40 + b"Linear Rec2020 RGB"
TEST_HOSTS = ("127.0.0.1", "localhost", "testserver")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EKVIDO_PRESET_DIR", str(tmp_path / "presets"))
    arr = np.full((120, 160, 3), 0.05, np.float32)
    arr[60, 80] = (8.0, 4.0, 1.2)
    path = tmp_path / "shot.tif"
    tifffile.imwrite(path, arr, photometric="rgb", iccprofile=ICC)
    app = create_app(UiSession(path, device="cpu"), allowed_hosts=TEST_HOSTS)
    return TestClient(app)


def _collect(ws):
    """Frames for one request: (binary frames, final timing dict)."""
    frames = []
    while True:
        msg = ws.receive()
        if msg.get("bytes") is not None:
            frames.append(msg["bytes"])
            continue
        data = json.loads(msg["text"])
        if data["type"] == "timing":
            return frames, data
        if data["type"] in ("error", "reload"):
            return frames, data


def test_latest_box_keeps_only_newest():
    async def go():
        box = LatestBox()
        for i in range(5):
            box.put(i)
        seq, item = await box.get()
        assert (seq, item) == (5, 4) and not box.is_stale(seq)
        box.put(99)
        assert box.is_stale(seq)

    asyncio.run(go())


def test_index_and_static(client):
    assert "ekvido" in client.get("/").text
    assert client.get("/api/state").json()["image"] == "shot.tif"


def test_ws_renders_two_levels_then_one(client):
    state = client.get("/api/state").json()["session"]
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"id": 7, "session": state, "view": {"width": 80, "height": 80}})
        frames, timing = _collect(ws)
        assert timing["type"] == "timing" and timing["id"] == 7
        rid, level = struct.unpack("<IB", frames[-1][:5])
        assert rid == 7 and level == 1
        assert PILImage.open(pyio.BytesIO(frames[-1][5:])).size == (80, 60)
        assert len(frames) == 2  # first request: quick + proxy
        ws.send_json({"id": 8, "session": state, "view": {"width": 80, "height": 80}})
        frames, _ = _collect(ws)
        assert len(frames) == 1  # tiny image: proxy pass is fast, quick skipped


def test_ws_bad_state_reports_and_keeps_going(client):
    good = client.get("/api/state").json()["session"]
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"id": 1, "session": {"stack": [{"module": "nope"}]}, "view": {}})
        frames, err = _collect(ws)
        assert err["type"] == "error" and "nope" in err["message"] and not frames
        ws.send_json({"id": 2, "session": good, "view": {"view": "x-ray"}})
        _, err = _collect(ws)
        assert err["type"] == "error" and "view" in err["message"]
        ws.send_json({"id": 3, "session": good, "view": {"width": 40, "height": 40}})
        frames, timing = _collect(ws)
        assert timing["type"] == "timing" and frames


def test_render_endpoint(client, tmp_path):
    r = client.post("/api/render")
    assert r.status_code == 200 and r.json()["path"].endswith("shot.ekvido.tif")


def test_preset_endpoints(client):
    entry = client.get("/api/state").json()["session"]["stack"][1]
    r = client.post("/api/presets", json={"name": "warm-one", "entry": entry})
    assert r.status_code == 200
    assert client.get("/api/presets/warm-one").json()["module"] == "kazuki"
    bad_name = {"name": "../x", "entry": entry}
    assert client.post("/api/presets", json=bad_name).status_code == 400
    bad_entry = {"name": "ok", "entry": {"module": "nope"}}
    assert client.post("/api/presets", json=bad_entry).status_code == 400
    assert client.get("/api/presets/missing").status_code == 404


@pytest.mark.parametrize(
    "bad", [[1, 2], "hi", {"id": "abc", "view": {}}, {"id": 1, "view": [3]}]
)
def test_ws_malformed_message_reports_and_keeps_serving(client, bad):
    good = client.get("/api/state").json()["session"]
    with client.websocket_connect("/ws") as ws:
        ws.send_json(bad)
        _, err = _collect(ws)
        assert err["type"] == "error"
        ws.send_json({"id": 3, "session": good, "view": {"width": 40, "height": 40}})
        frames, timing = _collect(ws)
        assert timing["type"] == "timing" and frames


def test_preset_saves_the_posted_entry_not_server_state(client):
    entry = client.get("/api/state").json()["session"]["stack"][1]
    entry["params"]["push_warm"] = 0.123
    client.post("/api/presets", json={"name": "posted", "entry": entry})
    assert client.get("/api/presets/posted").json()["params"]["push_warm"] == 0.123


@pytest.mark.parametrize(
    "frame",
    [
        "not json",
        '{"id": Infinity, "view": {}}',
        '{"id": 1, "session": {"clip_level": 1' + "0" * 400 + "}}",
        b"\x00\x01binary",
    ],
    ids=["not-json", "infinite-id", "huge-number", "binary"],
)
def test_ws_hostile_frames_report_and_keep_serving(client, frame):
    good = client.get("/api/state").json()["session"]
    with client.websocket_connect("/ws") as ws:
        if isinstance(frame, bytes):
            ws.send_bytes(frame)
        else:
            ws.send_text(frame)
        _, err = _collect(ws)
        assert err["type"] == "error"
        ws.send_json({"id": 3, "session": good, "view": {"width": 40, "height": 40}})
        frames, timing = _collect(ws)
        assert timing["type"] == "timing" and frames


def test_ws_stale_instance_asks_for_reload(client):
    state = client.get("/api/state").json()
    assert state["instance"]
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {"id": 1, "instance": "old", "session": state["session"], "view": {}}
        )
        frames, msg = _collect(ws)
        assert msg["type"] == "reload" and not frames
        ws.send_json(
            {
                "id": 2,
                "instance": state["instance"],
                "session": state["session"],
                "view": {"width": 40, "height": 40},
            }
        )
        frames, timing = _collect(ws)
        assert timing["type"] == "timing" and frames


def test_foreign_host_rejected(client):
    assert client.get("/api/state", headers={"host": "evil.example"}).status_code == 400


def test_foreign_origin_rejected_on_http_and_ws(client):
    evil = {"origin": "https://evil.example"}
    assert client.post("/api/render", headers=evil).status_code == 403
    assert client.get("/api/state", headers=evil).status_code == 403
    with pytest.raises(Exception):  # noqa: B017 - handshake refused
        with client.websocket_connect("/ws", headers=evil) as ws:
            ws.receive()


def test_own_origin_allowed(client):
    ok = {"origin": "http://127.0.0.1:8765", "host": "127.0.0.1:8765"}
    assert client.get("/api/state", headers=ok).status_code == 200
    with client.websocket_connect("/ws", headers=ok) as ws:
        ws.send_json({"id": 1, "view": {"width": 40, "height": 40}})
        frames, timing = _collect(ws)
        assert frames
