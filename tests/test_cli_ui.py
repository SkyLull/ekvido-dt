import numpy as np
import pytest
import tifffile
from fastapi import FastAPI

from ekvido.cli import main

ICC = b"\x00" * 40 + b"Linear Rec2020 RGB"


@pytest.fixture
def tif(tmp_path):
    path = tmp_path / "shot.tif"
    tifffile.imwrite(
        path, np.full((40, 60, 3), 0.1, np.float32), photometric="rgb", iccprofile=ICC
    )
    return path


@pytest.fixture
def served(monkeypatch):
    calls = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.update(app=app, **kw))
    return calls


def test_ui_builds_app_and_serves_locally(tif, served, capsys):
    assert (
        main(["ui", str(tif), "--no-browser", "--device", "cpu", "--port", "9999"]) == 0
    )
    assert isinstance(served["app"], FastAPI)
    assert served["host"] == "127.0.0.1" and served["port"] == 9999
    assert "http://127.0.0.1:9999/" in capsys.readouterr().out


def test_ui_warns_on_non_loopback_host(tif, served, capsys):
    assert (
        main(["ui", str(tif), "--no-browser", "--device", "cpu", "--host", "0.0.0.0"])
        == 0
    )
    assert "without authentication" in capsys.readouterr().err


def test_ui_missing_file(tmp_path, served, capsys):
    assert main(["ui", str(tmp_path / "nope.tif"), "--no-browser"]) == 2
    assert capsys.readouterr().err.startswith("ekvido: ")


def test_ui_allows_its_own_host_only(tif, served, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("EKVIDO_PRESET_DIR", str(tmp_path / "presets"))
    main(["ui", str(tif), "--no-browser", "--device", "cpu", "--host", "192.168.1.5"])
    client = TestClient(served["app"])
    assert (
        client.get("/api/state", headers={"host": "192.168.1.5:8765"}).status_code
        == 200
    )
    assert (
        client.get("/api/state", headers={"host": "localhost:8765"}).status_code == 200
    )
    assert client.get("/api/state", headers={"host": "evil.example"}).status_code == 400
