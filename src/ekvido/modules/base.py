import dataclasses
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from ekvido.common.psf import KernelSpec
from ekvido.device import sync


@dataclass(frozen=True)
class ParamMeta:
    kind: str = "float"  # float | int | bool | choice | color
    min: float | None = None
    max: float | None = None
    desc: str = ""
    choices: tuple[str, ...] = ()
    mode: str | None = None  # "scene" / "display": only meaningful in that input mode


def param(default, *, kind="float", min=None, max=None, desc="", choices=(), mode=None):
    meta = ParamMeta(kind, min, max, desc, tuple(choices), mode)
    return field(default=default, metadata={"param": meta})


def param_specs(params_cls) -> list[dict]:
    out = []
    for f in dataclasses.fields(params_cls):
        m: ParamMeta = f.metadata["param"]
        out.append(
            {
                "name": f.name,
                "kind": m.kind,
                "default": f.default,
                "min": m.min,
                "max": m.max,
                "desc": m.desc,
                "choices": list(m.choices),
                "mode": m.mode,
            }
        )
    return out


@dataclass
class Context:
    device: torch.device
    referred: str
    diag_px: float
    clip_level: float
    input_key: str
    cache: dict | None = None
    keep_spectra: bool = True
    stash: dict | None = None
    timings: dict | None = None
    # kazuki source mask (spec §5.7): (H, W) in 0…1 or None; only kazuki reads it
    source_mask: torch.Tensor | None = None
    source_mask_invert: bool = False
    source_mask_key: str | None = None  # identifies the mask for caches; None = hash it

    def memo(self, stage: str, key: Any, fn):
        if self.cache is not None:
            hit = self.cache.get(stage)
            if hit is not None and hit[0] == key:
                return hit[1]
        t0 = time.perf_counter()
        value = fn()
        if self.timings is not None:
            sync(self.device)
            self.timings[stage] = (
                self.timings.get(stage, 0.0) + time.perf_counter() - t0
            )
        if self.cache is not None:
            self.cache[stage] = (key, value)
        return value


class Module:
    NAME: ClassVar[str]
    VERSION: ClassVar[int]
    Params: ClassVar[type]

    @staticmethod
    def legacy_params(old_version: int, old: dict) -> dict:
        raise ValueError(f"no migration from version {old_version}")

    def process(self, img: torch.Tensor, p, ctx: Context) -> torch.Tensor:
        raise NotImplementedError


REGISTRY: dict[str, Module] = {}


def register(cls):
    REGISTRY[cls.NAME] = cls()
    return cls


def get_module(name: str) -> Module:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown module {name!r}; known: {sorted(REGISTRY)}"
        ) from None


def _clamp(name: str, m: ParamMeta, v):
    lo = m.min if m.min is not None else v
    hi = m.max if m.max is not None else v
    c = min(max(v, lo), hi)
    if c != v:
        warnings.warn(
            f"{name}={v!r} out of range, clamped to {c!r}", UserWarning, stacklevel=4
        )
    return c


def _coerce(f: dataclasses.Field, v):
    m: ParamMeta = f.metadata["param"]
    if m.kind in ("float", "int") and not math.isfinite(float(v)):
        warnings.warn(
            f"{f.name}={v!r} is not finite; using {f.default!r}",
            UserWarning,
            stacklevel=3,
        )
        return f.default
    if m.kind == "float":
        return _clamp(f.name, m, float(v))
    if m.kind == "int":
        return _clamp(f.name, m, int(round(float(v))))
    if m.kind == "bool":
        return bool(v)
    if m.kind == "choice":
        if v in m.choices:
            return v
        warnings.warn(
            f"{f.name}={v!r} not one of {m.choices}; using {f.default!r}",
            UserWarning,
            stacklevel=3,
        )
        return f.default
    if m.kind == "color":
        rgb = tuple(max(0.0, float(c)) for c in v)
        if len(rgb) != 3:
            raise ValueError(f"{f.name} must have 3 components, got {v!r}")
        return rgb
    raise ValueError(f"unknown param kind {m.kind!r}")


def params_from_dict(module: Module, version: int, data: dict):
    if version > module.VERSION:
        raise ValueError(
            f"{module.NAME} params are version {version}, newer than this ekvido "
            f"({module.VERSION}); upgrade ekvido"
        )
    data = dict(data)
    while version < module.VERSION:
        data = module.legacy_params(version, data)
        version += 1
    fields = {f.name: f for f in dataclasses.fields(module.Params)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        warnings.warn(
            f"{module.NAME}: ignoring unknown params {unknown}",
            UserWarning,
            stacklevel=2,
        )
    return module.Params(
        **{k: _coerce(fields[k], v) for k, v in data.items() if k in fields}
    )


def params_to_dict(p) -> dict:
    return {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in dataclasses.asdict(p).items()
    }


def kernel_spec(p) -> KernelSpec:
    names = {f.name for f in dataclasses.fields(KernelSpec)}
    return KernelSpec(**{n: getattr(p, n) for n in names if hasattr(p, n)})
