import dataclasses

import pytest
import torch

from ekvido.common.psf import KernelSpec
from ekvido.modules.base import (
    REGISTRY,
    Context,
    Module,
    get_module,
    kernel_spec,
    param,
    param_specs,
    params_from_dict,
    params_to_dict,
    register,
)


@dataclasses.dataclass(frozen=True)
class DemoParams:
    size: float = param(0.5, min=0.1, max=1.0, desc="size")
    count: int = param(3, kind="int", min=0, max=8, desc="count")
    on: bool = param(True, kind="bool", desc="on")
    shape: str = param("round", kind="choice", choices=("round", "cross"), desc="shape")
    tint: tuple = param((1.0, 0.5, 0.2), kind="color", desc="tint")
    halo_radius: float = param(0.01, min=0.001, max=0.05, desc="halo")


class Demo(Module):
    NAME = "demo"
    VERSION = 2
    Params = DemoParams

    @staticmethod
    def legacy_params(old_version, old):
        if old_version == 1:  # v1 called `size` `radius`
            old["size"] = old.pop("radius")
            return old
        raise ValueError(old_version)

    def process(self, img, p, ctx):
        return img * p.size


@pytest.fixture
def demo():
    register(Demo)
    yield REGISTRY["demo"]
    REGISTRY.pop("demo")


def test_param_specs_schema():
    specs = {s["name"]: s for s in param_specs(DemoParams)}
    assert specs["size"] == {
        "name": "size",
        "kind": "float",
        "default": 0.5,
        "min": 0.1,
        "max": 1.0,
        "desc": "size",
        "choices": [],
        "mode": None,
    }
    assert specs["shape"]["choices"] == ["round", "cross"]


def test_registry_and_unknown(demo):
    assert get_module("demo") is demo
    with pytest.raises(ValueError, match="unknown module"):
        get_module("nope")


def test_roundtrip(demo):
    p = DemoParams(size=0.3, tint=(0.1, 0.2, 0.3))
    assert params_from_dict(demo, 2, params_to_dict(p)) == p


def test_migration_from_v1(demo):
    assert params_from_dict(demo, 1, {"radius": 0.7}).size == 0.7


def test_newer_version_rejected(demo):
    with pytest.raises(ValueError, match="newer"):
        params_from_dict(demo, 3, {})


def test_missing_keys_take_defaults_unknown_dropped(demo):
    with pytest.warns(UserWarning, match="bogus"):
        p = params_from_dict(demo, 2, {"bogus": 1, "count": 5})
    assert p.count == 5 and p.size == 0.5


def test_out_of_range_clamped_with_warning(demo):
    with pytest.warns(UserWarning, match="clamped"):
        p = params_from_dict(demo, 2, {"size": 0, "count": -1, "halo_radius": 0})
    assert (p.size, p.count, p.halo_radius) == (0.1, 0, 0.001)


def test_coercion_and_bad_choice(demo):
    with pytest.warns(UserWarning, match="shape"):
        p = params_from_dict(
            demo, 2, {"count": 2.6, "on": 0, "shape": "star", "tint": [1, 1, 1]}
        )
    assert (
        p.count == 3
        and p.on is False
        and p.shape == "round"
        and p.tint == (1.0, 1.0, 1.0)
    )


def test_kernel_spec_copies_matching_fields():
    spec = kernel_spec(DemoParams(halo_radius=0.02))
    assert isinstance(spec, KernelSpec) and spec.halo_radius == 0.02
    assert spec.halo_beta == KernelSpec().halo_beta


def _ctx(**kw):
    return Context(
        device=torch.device("cpu"),
        referred="scene",
        diag_px=100.0,
        clip_level=float("inf"),
        input_key="k",
        **kw,
    )


def test_memo_caches_and_replaces():
    calls = []
    ctx = _ctx(cache={})
    f = lambda: calls.append(1) or len(calls)  # noqa: E731
    assert ctx.memo("s", 1, f) == 1 and ctx.memo("s", 1, f) == 1
    assert ctx.memo("s", 2, f) == 2 and len(ctx.cache) == 1


def test_memo_without_cache_always_computes():
    calls = []
    ctx = _ctx()
    ctx.memo("s", 1, lambda: calls.append(1))
    ctx.memo("s", 1, lambda: calls.append(1))
    assert len(calls) == 2


def test_memo_records_timings():
    ctx = _ctx(timings={})
    ctx.memo("s", 1, lambda: 0)
    assert ctx.timings["s"] >= 0


def test_non_finite_param_falls_back_to_default(demo):
    # Review I4: json accepts NaN; it slipped past the clamp and blanked the image.
    with pytest.warns(UserWarning, match="not finite"):
        p = params_from_dict(demo, 2, {"size": float("nan"), "count": float("inf")})
    assert (p.size, p.count) == (0.5, 3)
