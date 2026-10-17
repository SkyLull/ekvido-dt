import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ekvido.modules.base import get_module, params_from_dict, params_to_dict
from ekvido.modules.kazuki import KazukiParams
from ekvido.modules.mist import MistParams
from ekvido.pipeline import StackEntry

SIDECAR_VERSION = 1
_PRESET_DIR = Path(__file__).with_name("preset_data")
_REFERRED = ("scene", "display")


def default_stack() -> list[StackEntry]:
    return [
        StackEntry("mist", MistParams(), enabled=False),
        StackEntry("kazuki", KazukiParams()),
    ]


@dataclass
class Session:
    referred: str = "scene"
    clip_level: float | None = None
    stack: list[StackEntry] = field(default_factory=default_stack)


def sidecar_path(tif: Path) -> Path:
    tif = Path(tif)
    return tif.with_name(tif.stem + ".ekvido.json")


def _entry_to_dict(e: StackEntry) -> dict:
    mod = get_module(e.module)
    return {
        "module": e.module,
        "version": mod.VERSION,
        "enabled": e.enabled,
        "params": params_to_dict(e.params),
    }


def _entry_from_dict(d: dict) -> StackEntry:
    mod = get_module(d["module"])
    params = params_from_dict(
        mod, int(d.get("version", mod.VERSION)), d.get("params", {})
    )
    return StackEntry(d["module"], params, bool(d.get("enabled", True)))


def session_to_dict(s: Session) -> dict:
    return {
        "ekvido_version": SIDECAR_VERSION,
        "referred": s.referred,
        "clip_level": s.clip_level,
        "stack": [_entry_to_dict(e) for e in s.stack],
    }


def session_from_dict(d: dict) -> Session:
    version = int(d.get("ekvido_version", SIDECAR_VERSION))
    if version > SIDECAR_VERSION:
        raise ValueError(
            f"sidecar format {version} is newer than this ekvido ({SIDECAR_VERSION})"
        )
    referred = d.get("referred", "scene")
    if referred not in _REFERRED:
        raise ValueError(f"referred must be one of {_REFERRED}, got {referred!r}")
    clip = d.get("clip_level")
    return Session(
        referred=referred,
        clip_level=None if clip is None else float(clip),
        stack=[_entry_from_dict(e) for e in d.get("stack", [])],
    )


def save_sidecar(path: Path, s: Session) -> None:
    Path(path).write_text(json.dumps(session_to_dict(s), indent=2) + "\n")


def load_sidecar(path: Path) -> Session:
    return session_from_dict(json.loads(Path(path).read_text()))


def save_preset(path: Path, entry: StackEntry) -> None:
    d = _entry_to_dict(entry)
    d.pop("enabled")
    Path(path).write_text(json.dumps(d, indent=2) + "\n")


def load_preset(path: Path) -> StackEntry:
    return _entry_from_dict({**json.loads(Path(path).read_text()), "enabled": True})


def shipped_presets() -> dict[str, Path]:
    return {p.stem: p for p in sorted(_PRESET_DIR.glob("*.json"))}


_NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")


def user_preset_dir() -> Path:
    env = os.environ.get("EKVIDO_PRESET_DIR")
    return Path(env) if env else Path.home() / ".config" / "ekvido" / "presets"


def list_presets() -> dict[str, Path]:
    found = dict(shipped_presets())
    d = user_preset_dir()
    if d.is_dir():
        found.update({p.stem: p for p in sorted(d.glob("*.json"))})
    return found


def save_user_preset(name: str, entry: StackEntry) -> Path:
    if not _NAME.fullmatch(name) or name.startswith("."):
        raise ValueError(
            f"invalid preset name {name!r}: use letters, digits, . _ - (max 64)"
        )
    d = user_preset_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    save_preset(path, entry)
    return path


def resolve_preset(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.is_file():
        return p
    shipped = list_presets()
    if name_or_path in shipped:
        return shipped[name_or_path]
    raise ValueError(
        f"no preset file or shipped preset named {name_or_path!r}; "
        f"known: {sorted(shipped)}"
    )
