import io
import json
import math
import secrets
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image as PILImage

from ekvido.common.display import (
    colormap_warmth,
    kernel_image,
    mask_overlay,
    to_display,
)
from ekvido.common.energy import estimate_clip_level
from ekvido.device import pick_device, sync
from ekvido.io import make_proxy, read_tiff, write_tiff
from ekvido.modules.base import REGISTRY, param_specs
from ekvido.pipeline import Pipeline
from ekvido.presets import (
    SIDECAR_VERSION,
    Session,
    list_presets,
    load_preset,
    load_sidecar,
    save_sidecar,
    save_user_preset,
    session_from_dict,
    session_to_dict,
    sidecar_path,
)

LEVELS = {"quick": 2.5, "proxy": 10.0}
VIEWS = ("final", "before", "energy", "warmth", "clip", "glow", "kernel")
QUICK_SKIP_S = 0.150
_SIDECAR_ERRORS = (ValueError, OSError, KeyError, TypeError, AttributeError)


def _num(v, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


@dataclass
class ViewRequest:
    view: str = "final"
    exposure: float = 0.0
    crop: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    width: int = 1600
    height: int = 1000

    @classmethod
    def from_dict(cls, d: dict) -> "ViewRequest":
        view = d.get("view", "final")
        if view not in VIEWS:
            raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
        raw = d.get("crop") or (0, 0, 1, 1)
        x0, y0, x1, y1 = (min(max(_num(c, 0.0), 0.0), 1.0) for c in list(raw)[:4])
        x1, y1 = max(x1, min(x0 + 1e-4, 1.0)), max(y1, min(y0 + 1e-4, 1.0))
        x0, y0 = min(x0, x1 - 1e-4), min(y0, y1 - 1e-4)
        exposure = min(max(_num(d.get("exposure", 0.0), 0.0), -10.0), 10.0)

        def size(v):
            return int(min(max(_num(v, 1.0), 1.0), 8192.0))

        return cls(
            view,
            exposure,
            (x0, y0, x1, y1),
            size(d.get("width", 1600)),
            size(d.get("height", 1000)),
        )


def _fit(region_w: int, region_h: int, width: int, height: int) -> tuple[int, int]:
    scale = min(width / region_w, height / region_h)
    return max(1, round(region_w * scale)), max(1, round(region_h * scale))


def _encode(rgb: torch.Tensor, width: int, height: int) -> bytes:
    arr = rgb.to("cpu").numpy()
    img = PILImage.fromarray(arr, "RGB")
    size = _fit(img.width, img.height, width, height)
    if size != img.size:
        method = (
            PILImage.Resampling.BOX
            if size[0] < img.width
            else PILImage.Resampling.BILINEAR
        )
        img = img.resize(size, method)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=85, method=0)
    return buf.getvalue()


def _crop(x: torch.Tensor, crop) -> torch.Tensor:
    h, w = x.shape[-2:]
    x0, y0, x1, y1 = crop
    c0, r0 = min(int(x0 * w), w - 1), min(int(y0 * h), h - 1)
    c1, r1 = max(c0 + 1, math.ceil(x1 * w)), max(r0 + 1, math.ceil(y1 * h))
    return x[..., r0:r1, c0:c1]


def _find(stash: dict, key: str):
    for name in sorted(stash, key=lambda s: int(s.split(":")[0]), reverse=True):
        if key in stash[name]:
            return stash[name][key]
    return None


class UiSession:
    def __init__(self, path, *, referred: str | None = None, device: str | None = None):
        self.path = Path(path)
        self.image = read_tiff(self.path)
        self.device = pick_device(device)
        self.sidecar = sidecar_path(self.path)
        session = Session()
        if self.sidecar.exists():
            try:
                session = load_sidecar(self.sidecar)
            except _SIDECAR_ERRORS as exc:
                warnings.warn(
                    f"ignoring sidecar {self.sidecar.name}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
        if referred:
            session.referred = referred
        self.session = session
        self.auto_clip = estimate_clip_level(self.image.data)
        self.proxies = {
            lvl: make_proxy(self.image.data, mp).to(self.device)
            for lvl, mp in LEVELS.items()
        }
        self.pipe = Pipeline()
        self.instance = secrets.token_hex(8)
        self.lock = threading.Lock()
        self.last_proxy_s: float | None = None
        self.timings: dict[str, float] = {}

    @property
    def clip_level(self) -> float:
        if self.session.clip_level is not None:
            return self.session.clip_level
        return 1.0 if self.session.referred == "display" else self.auto_clip

    def state(self) -> dict:
        _, h, w = self.image.data.shape
        presets = {}
        for name, path in list_presets().items():
            try:
                presets[name] = json.loads(path.read_text())["module"]
            except _SIDECAR_ERRORS:
                continue
        return {
            "image": self.path.name,
            "instance": self.instance,
            "size": [w, h],
            "device": self.device.type,
            "referred": self.session.referred,
            "auto_clip": None if math.isinf(self.auto_clip) else self.auto_clip,
            "session": session_to_dict(self.session),
            "modules": {
                name: {"version": m.VERSION, "params": param_specs(m.Params)}
                for name, m in REGISTRY.items()
            },
            "presets": presets,
            "views": list(VIEWS),
            "levels": {
                k: [int(v.shape[2]), int(v.shape[1])] for k, v in self.proxies.items()
            },
        }

    def apply_state(self, d: dict) -> None:
        new = session_from_dict({"ekvido_version": SIDECAR_VERSION, **d})
        changed = session_to_dict(new) != session_to_dict(self.session)
        self.session = new
        if changed:  # an unchanged state (e.g. a page load) never touches the sidecar
            save_sidecar(self.sidecar, new)

    def should_quick(self) -> bool:
        return self.last_proxy_s is None or self.last_proxy_s >= QUICK_SKIP_S

    def _view(self, req: ViewRequest, src, out, stash) -> torch.Tensor:
        referred, ev = self.session.referred, req.exposure
        if req.view == "before":
            return to_display(_crop(src, req.crop), referred, ev)
        if req.view == "final" or stash is None:
            return to_display(_crop(out, req.crop), referred, ev)
        key = req.view
        found = _find(stash, key)
        if found is None:  # no enabled module produced this map
            return to_display(_crop(out, req.crop), referred, ev)
        if key == "kernel":
            return kernel_image(found)
        if key == "warmth":
            return colormap_warmth(_crop(found, req.crop))
        if key == "clip":
            return mask_overlay(
                to_display(_crop(src, req.crop), referred, ev), _crop(found, req.crop)
            )
        return to_display(_crop(found, req.crop), referred, ev)  # energy, glow

    def render(self, level: str, req: ViewRequest) -> bytes:
        with self.lock:
            src = self.proxies[level]
            t0 = time.perf_counter()
            stash = None if req.view in ("final", "before") else {}
            out = src
            if req.view != "before":
                out = self.pipe.run(
                    src,
                    self.session.stack,
                    input_key=f"{self.path}:{level}",
                    referred=self.session.referred,
                    clip_level=self.clip_level,
                    scope=level,
                    stash=stash,
                )
            rgb = self._view(req, src, out, stash)
            sync(self.device)
            elapsed = time.perf_counter() - t0
            self.timings[level] = elapsed
            if level == "proxy" and req.view == "final":
                self.last_proxy_s = elapsed
        return _encode(rgb, req.width, req.height)

    def render_full(self) -> tuple[Path, float]:
        with self.lock:
            t0 = time.perf_counter()
            result = Pipeline().run(
                self.image.data.to(self.device),
                self.session.stack,
                input_key=str(self.path),
                referred=self.session.referred,
                clip_level=self.clip_level,
                scope=None,
            )
            out = self.path.with_name(self.path.stem + ".ekvido.tif")
            write_tiff(out, result, self.image)
            return out, time.perf_counter() - t0

    def save_preset(self, name: str, entry: dict) -> Path:
        """Save the entry the client shows (not the server's possibly older copy)."""
        parsed = session_from_dict({"stack": [{**entry, "enabled": True}]}).stack[0]
        return save_user_preset(name, parsed)

    def preset_entry(self, name: str) -> dict:
        entry = load_preset(list_presets()[name])
        return session_to_dict(Session(stack=[entry]))["stack"][0]
