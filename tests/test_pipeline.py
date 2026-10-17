import math

import pytest
import torch

import ekvido.modules  # noqa: F401
from ekvido.modules.base import REGISTRY, Context
from ekvido.modules.kazuki import KazukiParams
from ekvido.modules.mist import MistParams
from ekvido.pipeline import Pipeline, StackEntry


def _img():
    img = torch.zeros(3, 120, 160)
    img[:, 60, 80] = torch.tensor([8.0, 4.0, 1.2])
    return img


def _run(pipe, img, stack, **kw):
    return pipe.run(
        img, stack, input_key="img", referred="scene", clip_level=math.inf, **kw
    )


@pytest.fixture
def counts(monkeypatch):
    n = {"mist": 0, "kazuki": 0}
    for name in n:
        mod = REGISTRY[name]
        orig = type(mod).process

        def wrapped(self, img, p, ctx, _orig=orig, _name=name):
            n[_name] += 1
            return _orig(self, img, p, ctx)

        monkeypatch.setattr(type(mod), "process", wrapped)
    return n


STACK = [
    StackEntry("mist", MistParams(strength=0.2)),
    StackEntry("kazuki", KazukiParams()),
]


def test_disabled_and_empty_are_identity():
    img = _img()
    assert torch.equal(_run(Pipeline(), img, []), img)
    off = [StackEntry("kazuki", KazukiParams(), enabled=False)]
    assert torch.equal(_run(Pipeline(), img, off), img)


def test_stack_equals_sequential_application():
    img = _img()
    diag = math.hypot(120, 160)
    mid = REGISTRY["mist"].process(
        img, STACK[0].params, Context(img.device, "scene", diag, math.inf, "a")
    )
    ref = REGISTRY["kazuki"].process(
        mid, STACK[1].params, Context(img.device, "scene", diag, math.inf, "b")
    )
    assert torch.allclose(_run(Pipeline(), img, STACK), ref, atol=1e-6)


def test_output_cache_and_downstream_invalidation(counts):
    pipe, img = Pipeline(), _img()
    _run(pipe, img, STACK, scope="proxy")
    _run(pipe, img, STACK, scope="proxy")
    assert counts == {"mist": 1, "kazuki": 1}
    changed = [STACK[0], StackEntry("kazuki", KazukiParams(push_warm=0.1))]
    _run(pipe, img, changed, scope="proxy")
    assert counts == {"mist": 1, "kazuki": 2}
    upstream = [StackEntry("mist", MistParams(strength=0.3)), changed[1]]
    _run(pipe, img, upstream, scope="proxy")
    assert counts == {"mist": 2, "kazuki": 3}


def test_scopes_are_independent(counts):
    pipe, img = Pipeline(), _img()
    _run(pipe, img, STACK, scope="quick")
    _run(pipe, img, STACK, scope="proxy")
    assert counts == {"mist": 2, "kazuki": 2}


def test_no_scope_never_caches(counts):
    pipe, img = Pipeline(), _img()
    _run(pipe, img, STACK)
    _run(pipe, img, STACK)
    assert counts == {"mist": 2, "kazuki": 2}


def test_stash_per_module_bypasses_output_cache(counts):
    pipe, img = Pipeline(), _img()
    _run(pipe, img, STACK, scope="proxy")
    stash = {}
    _run(pipe, img, STACK, scope="proxy", stash=stash)
    assert set(stash) == {"0:mist", "1:kazuki"} and "warmth" in stash["1:kazuki"]
    assert counts == {"mist": 2, "kazuki": 2}


def test_unknown_module():
    with pytest.raises(ValueError, match="unknown module"):
        _run(Pipeline(), _img(), [StackEntry("nope", None)])
