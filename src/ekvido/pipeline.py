import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch

import ekvido.modules  # noqa: F401  (registers modules)
from ekvido.common.energy import mask_key
from ekvido.modules.base import Context, get_module


@dataclass
class StackEntry:
    module: str
    params: Any
    enabled: bool = True


def _key(*parts) -> str:
    return hashlib.sha1(repr(parts).encode(), usedforsecurity=False).hexdigest()


class Pipeline:
    def __init__(self):
        self._outputs: dict[tuple[str, int], tuple[str, torch.Tensor]] = {}
        self._caches: dict[tuple[str, int, str], dict] = {}

    def run(
        self,
        img: torch.Tensor,
        stack: list[StackEntry],
        *,
        input_key: str,
        referred: str,
        clip_level: float,
        scope: str | None = None,
        stash: dict | None = None,
        timings: dict | None = None,
        source_mask: torch.Tensor | None = None,
        source_mask_invert: bool = False,
    ) -> torch.Tensor:
        _, h, w = img.shape
        m_key = None if source_mask is None else mask_key(source_mask)
        diag = math.hypot(h, w)
        upstream, x = input_key, img
        for i, entry in enumerate(stack):
            if not entry.enabled:
                continue
            mod = get_module(entry.module)
            key = _key(
                upstream,
                mod.NAME,
                mod.VERSION,
                entry.params,
                referred,
                clip_level,
                m_key,
                source_mask_invert,
            )
            if scope is not None and stash is None:
                hit = self._outputs.get((scope, i))
                if hit is not None and hit[0] == key:
                    upstream, x = key, hit[1]
                    continue
            cache = (
                None
                if scope is None
                else self._caches.setdefault((scope, i, mod.NAME), {})
            )
            sub = None if stash is None else stash.setdefault(f"{i}:{mod.NAME}", {})
            ctx = Context(
                device=img.device,
                referred=referred,
                diag_px=diag,
                clip_level=clip_level,
                input_key=upstream,
                cache=cache,
                keep_spectra=scope is not None,
                stash=sub,
                timings=timings,
                source_mask=source_mask,
                source_mask_invert=source_mask_invert,
                source_mask_key=m_key,
            )
            x = mod.process(x, entry.params, ctx)
            if scope is not None:
                self._outputs[(scope, i)] = (key, x)
            upstream = key
        return x
