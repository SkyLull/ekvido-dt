import pytest
import torch

from ekvido.device import pick_device, sync


def test_prefer_cpu():
    assert pick_device("cpu") == torch.device("cpu")


def test_env_override(monkeypatch):
    monkeypatch.setenv("EKVIDO_DEVICE", "cpu")
    assert pick_device() == torch.device("cpu")


def test_auto_returns_available_device():
    d = pick_device()
    assert d.type in {"cuda", "mps", "cpu"}
    if d.type == "cuda":
        assert torch.cuda.is_available()
    if d.type == "mps":
        assert torch.backends.mps.is_available()


def test_unknown_device_rejected():
    with pytest.raises(ValueError):
        pick_device("tpu")


def test_sync_cpu_is_noop(cpu):
    sync(cpu)
