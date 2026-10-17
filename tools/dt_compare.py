"""Check the darktable ekvido modules against the Python reference.

Runs darktable-cli with an XMP holding only one ekvido module, dumps the module's exact
input and output (--dump-pipe), runs the Python module on that input, and compares.

Usage:
  uv run python tools/dt_compare.py --darktable-cli PATH --input FLOAT_TIFF \\
      --module mist|kazuki --preset NAME [--legacy-v1] [--diff-image OUT.tif]
  uv run python tools/dt_compare.py --darktable-cli PATH --input FLOAT_TIFF \\
      --module kazuki --xmp EDIT.xmp
      (an edit saved in darktable, e.g. with a source mask: kazuki's params are read from
      the XMP and the mask kazuki received is dumped with --dump-pfm)
  uv run python tools/dt_compare.py --module kazuki --preset NAME --write-xmp OUT.xmp
  uv run python tools/dt_compare.py --module kazuki --preset NAME --dump-dir DIR
      (compare a dump made elsewhere, e.g. on Windows, from DIR/export/*.pfm)
"""

import argparse
import base64
import math
import re
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_c_fixtures import KAZUKI_HSV_DEFAULTS, KAZUKI_WIDE  # noqa: E402

from ekvido.common.blend import BLEND_MODES
from ekvido.common.color import hsv_to_rgb
from ekvido.common.energy import estimate_clip_level
from ekvido.modules.base import Context
from ekvido.modules.kazuki import Kazuki, KazukiParams
from ekvido.modules.mist import Mist, MistParams

TOL = 1e-4  # allowed max |darktable − Python|, relative to the largest finite output
MEAN_TOL = (
    1e-6  # allowed mean |darktable − Python|, same scale (a higher mean is a defect)
)

MIST_FIELDS = (
    "strength",
    "halo_radius",
    "halo_beta",
    "halo_stretch_x",
    "halo_stretch_y",
    "veil_radius",
    "veil_beta",
    "w_halo",
    "w_veil",
    "arms",
    "arm_angle",
    "arm_radius",
    "arm_beta",
    "arm_sharpness",
    "arm_stretch_x",
    "arm_stretch_y",
    "w_arms",
)
MIST_FORMAT = "<9fi7f"
MIST_PRESETS = {
    "default": MistParams(),
    "black": MistParams(strength=0.15, w_halo=0.85, w_veil=0.15, halo_beta=1.8),
    "star": MistParams(
        strength=0.3, arms=4, arm_radius=0.002, w_arms=0.3, arm_stretch_x=2.0
    ),
}

# dt_iop_ekvidokazuki_params_t, in struct order (src/iop/ekvidokazuki.c in the fork).
KAZUKI_FIELDS = (
    "strength",
    "blend",
    "blend_white_ev",
    "threshold_ev",
    "knee_ev",
    "clip_mode",
    "clip_ev",
    "hue_radius",
    "neutral_point",
    "warmth_softness",
    "neutral_bias",
    "push_warm",
    "push_cool",
    "warm_core_hue",
    "warm_core_sat",
    "warm_edge_hue",
    "warm_edge_sat",
    "cool_core_hue",
    "cool_core_sat",
    "cool_edge_hue",
    "cool_edge_sat",
    "band_split",
    "halo_radius",
    "halo_beta",
    "halo_stretch_x",
    "halo_stretch_y",
    "veil_radius",
    "veil_beta",
    "w_halo",
    "w_veil",
    "arms",
    "arm_angle",
    "arm_radius",
    "arm_beta",
    "arm_sharpness",
    "arm_stretch_x",
    "arm_stretch_y",
    "w_arms",
    # version 2: source mask (spec §5.7)
    "source_mask_op",
    "source_mask_instance",
    "source_mask_id",
    "source_mask_invert",
)
KAZUKI_FORMAT = "<fi3fif6f8f9fi7f20s3i"
KAZUKI_VERSION = 2
KAZUKI_V1_FIELDS = KAZUKI_FIELDS[:38]
KAZUKI_V1_FORMAT = "<fi3fif6f8f9fi7f"
KAZUKI_DEFAULTS = {
    **{
        f: getattr(KazukiParams(), f)
        for f in KAZUKI_FIELDS
        if f not in ("blend", "clip_mode", "clip_ev")
        and not f.endswith(("_hue", "_sat"))
        and not f.startswith("source_mask")
    },
    "blend": 0,
    "clip_mode": 0,
    "clip_ev": 0.0,
    "source_mask_op": b"",
    "source_mask_instance": 0,
    "source_mask_id": 0,
    "source_mask_invert": 0,
    # darktable $DEFAULT hues are 6-decimal roundings of the Python colours' hue
    **{
        f"{c}_{part}": round(value, 6) if part == "hue" else value
        for c, hs in KAZUKI_HSV_DEFAULTS.items()
        for part, value in zip(("hue", "sat"), hs, strict=True)
    },
}
KAZUKI_PRESETS = {
    "default": {},
    # shipped preset kazuki-wide (shape only), taken from the one Python-side source
    "wide": {
        f: KAZUKI_WIDE[f]
        for f in KAZUKI_FIELDS
        if f in KAZUKI_WIDE and f != "blend" and KAZUKI_WIDE[f] != KAZUKI_DEFAULTS[f]
    },
    "screen": {  # non-default: CSP screen, manual clip, wide colour sampling, 6 arms
        "blend": BLEND_MODES.index("screen"),
        "strength": 0.8,
        "blend_white_ev": 1.0,
        "clip_mode": 1,
        "clip_ev": 2.0,
        "hue_radius": 0.03,
        "push_warm": 0.5,
        "warm_edge_hue": 15.0,
        "warm_edge_sat": 0.9,
        "arms": 6,
        "arm_angle": 15.0,
    },
    "glowdodge": {
        "blend": BLEND_MODES.index("glow_dodge"),
        "strength": 0.6,
        "blend_white_ev": -1.0,
    },
    "overlayglow": {
        "blend": BLEND_MODES.index("overlay_glow"),
        "strength": 0.7,
        "blend_white_ev": 0.5,
    },
    # source mask from a module that is not in the edit: spec §5.7 says no glow
    "missingsource": {"source_mask_op": b"exposure"},
}
COLOURS = ("warm_core", "warm_edge", "cool_core", "cool_edge")


def pack_mist_params(p: MistParams) -> bytes:
    return struct.pack(MIST_FORMAT, *(getattr(p, f) for f in MIST_FIELDS))


def pack_kazuki(c: dict) -> bytes:
    return struct.pack(KAZUKI_FORMAT, *(c[f] for f in KAZUKI_FIELDS))


def unpack_kazuki(blob: bytes) -> dict:
    c = dict(zip(KAZUKI_FIELDS, struct.unpack(KAZUKI_FORMAT, blob), strict=True))
    c["source_mask_op"] = c["source_mask_op"].split(b"\0", 1)[0]
    return c


def pack_kazuki_v1(c: dict) -> bytes:
    """The version-1 blob (no source mask), for checking darktable's legacy_params."""
    return struct.pack(KAZUKI_V1_FORMAT, *(c[f] for f in KAZUKI_V1_FIELDS))


def kazuki_preset(name: str) -> dict:
    """C-side params for a preset, rounded through the 184-byte blob (as darktable sees them)."""
    return unpack_kazuki(pack_kazuki({**KAZUKI_DEFAULTS, **KAZUKI_PRESETS[name]}))


def kazuki_c_to_py(c: dict) -> tuple[KazukiParams, float | None]:
    """Python params for C params, and the manual clip level (None = auto)."""
    names = set(KazukiParams.__dataclass_fields__)
    kw = {k: v for k, v in c.items() if k in names and k != "blend"}
    kw["blend"] = BLEND_MODES[c["blend"]]
    for col in COLOURS:
        kw[col] = hsv_to_rgb(c[f"{col}_hue"], c[f"{col}_sat"])
    clip = 2.0 ** c["clip_ev"] if c["clip_mode"] == 1 else None
    return KazukiParams(**kw), clip


def history_xmp(op: str, blob: bytes, modversion: int = 1) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 4.4.0-Exiv2">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:darktable="http://darktable.sf.net/"
   xmp:Rating="0" darktable:xmp_version="5" darktable:raw_params="0" darktable:auto_presets_applied="1"
   darktable:history_end="1" darktable:iop_order_version="5">
   <darktable:masks_history><rdf:Seq/></darktable:masks_history>
   <darktable:history><rdf:Seq>
     <rdf:li darktable:num="0" darktable:operation="{op}" darktable:enabled="1"
       darktable:modversion="{modversion}" darktable:params="{blob.hex()}"
       darktable:multi_name="" darktable:multi_priority="0"/>
   </rdf:Seq></darktable:history>
  </rdf:Description></rdf:RDF></x:xmpmeta>
"""


def module_xmp(module: str, preset: str, legacy_v1: bool = False) -> str:
    if module == "mist":
        return history_xmp("ekvidomist", pack_mist_params(MIST_PRESETS[preset]))
    c = kazuki_preset(preset)
    if legacy_v1:
        return history_xmp("ekvidokazuki", pack_kazuki_v1(c), 1)
    return history_xmp("ekvidokazuki", pack_kazuki(c), KAZUKI_VERSION)


def decode_xmp_params(text: str) -> bytes:
    """darktable's params encoding: plain hex, or "gz" + 2 digits + base64(zlib)."""
    if text.startswith("gz"):
        return zlib.decompress(base64.b64decode(text[4:]))
    return bytes.fromhex(text)


def kazuki_params_from_xmp(path: Path) -> dict:
    """C-side params of the last applied ekvidokazuki history entry in a darktable XMP."""
    text = Path(path).read_text()
    end = re.search(r'darktable:history_end="(\d+)"', text)
    limit = int(end.group(1)) if end else 1 << 30
    found = None
    for li in re.findall(r"<rdf:li\b[^>]*?/?>", text, re.S):
        attrs = dict(re.findall(r'darktable:(\w+)="([^"]*)"', li))
        if (
            attrs.get("operation") != "ekvidokazuki"
            or int(attrs.get("num", 0)) >= limit
        ):
            continue
        if int(attrs.get("modversion", 0)) != KAZUKI_VERSION:
            raise ValueError(
                f"{path}: ekvidokazuki entry is version {attrs.get('modversion')}"
            )
        found = attrs
    if found is None:
        raise ValueError(f"{path}: no applied ekvidokazuki history entry")
    return unpack_kazuki(decode_xmp_params(found["params"]))


def read_pfm(path: Path) -> np.ndarray:
    data = Path(path).read_bytes()
    header, pos = [], 0
    while len(header) < 3:
        end = data.index(b"\n", pos)
        header.append(data[pos:end].decode().strip())
        pos = end + 1
    if header[0] not in ("PF", "Pf"):
        raise ValueError(f"{path}: not a PFM")
    w, h = map(int, header[1].split())
    endian = "<" if float(header[2]) < 0 else ">"
    if header[0] == "Pf":  # grey (darktable's mask dumps): (H, W)
        grey = np.frombuffer(data, dtype=f"{endian}f4", count=w * h, offset=pos)
        return np.ascontiguousarray(grey.reshape(h, w)[::-1])
    img = np.frombuffer(data, dtype=f"{endian}f4", count=w * h * 3, offset=pos).reshape(
        h, w, 3
    )
    return np.ascontiguousarray(img[::-1].transpose(2, 0, 1))


def python_reference(
    module: str,
    preset: str,
    src: np.ndarray,
    c: dict | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Python output for the dumped input; kazuki params from `c` (C-side dict) if given,
    else from the preset; `mask` = the source mask kazuki received (H, W) or None."""
    img = torch.from_numpy(src.copy())
    _, h, w = img.shape
    if module == "mist":
        p, clip = MIST_PRESETS[preset], math.inf
    else:
        c = kazuki_preset(preset) if c is None else c
        if c["source_mask_op"] and mask is None:
            return src.copy()  # chosen source mask unavailable: no glow (spec §5.7)
        p, manual = kazuki_c_to_py(c)
        clip = estimate_clip_level(img) if manual is None else manual
    ctx = Context(
        device=torch.device("cpu"),
        referred="scene",
        diag_px=math.hypot(h, w),
        clip_level=clip,
        input_key="dt",
        cache=None,
        keep_spectra=True,
    )
    if mask is not None:
        ctx.source_mask = torch.from_numpy(mask.copy())
        ctx.source_mask_invert = bool(c["source_mask_invert"])
    mod = Mist() if module == "mist" else Kazuki()
    return mod.process(img, p, ctx).numpy()


def compare(got: np.ndarray, want: np.ndarray) -> tuple[float, float, float, bool]:
    """(max diff, mean diff, peak, non-finite classes agree) over finite expected values."""
    finite = np.isfinite(want)
    same_class = bool(
        np.array_equal(np.isnan(want), np.isnan(got))
        and np.array_equal(np.isposinf(want), np.isposinf(got))
        and np.array_equal(np.isneginf(want), np.isneginf(got))
    )
    diff = np.abs(got[finite].astype(np.float64) - want[finite])
    peak = float(np.abs(want[finite]).max()) if finite.any() else 0.0
    return float(diff.max()), float(diff.mean()), peak, same_class


def passes(worst: float, mean: float, peak: float, same_class: bool) -> bool:
    """Max diff within TOL of the peak, mean diff within MEAN_TOL of it, non-finite agree."""
    return same_class and worst <= TOL * peak and mean <= MEAN_TOL * peak


def write_diff_image(path: Path, got: np.ndarray, want: np.ndarray) -> None:
    diff = np.nan_to_num(np.abs(got - want), nan=0.0, posinf=0.0)
    tifffile.imwrite(
        path, diff.transpose(1, 2, 0).astype(np.float32), photometric="rgb"
    )


def run_darktable(
    cli: str,
    image: str,
    module: str,
    preset: str,
    work: Path,
    legacy_v1: bool = False,
    xmp: Path | None = None,
    opencl: bool = False,
) -> Path:
    """Render through darktable-cli with a dump of the module (and of the source mask it
    received: --dump-pfm); returns the dump dir. With opencl, darktable runs with OpenCL
    on (the GPU path)."""
    op = "ekvido" + module
    if xmp is None:
        (work / "module.xmp").write_text(module_xmp(module, preset, legacy_v1))
        xmp = work / "module.xmp"
    cmd = [
        cli,
        image,
        str(xmp),
        str(work / "out.tif"),
        "--width",
        "0",
        "--height",
        "0",
        "--apply-custom-presets",
        "false",
        "--icc-type",
        "LIN_REC2020",
        "--core",
        "--configdir",
        str(work / "cfg"),
        "--dumpdir",
        str(work / "dump"),
        "--dump-pipe",
        op,
        "--dump-pfm",
        op,
        *(["--conf", "opencl=TRUE"] if opencl else ["--disable-opencl"]),
        "--conf",
        "plugins/imageio/format/tiff/bpp=32",
    ]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"darktable-cli failed:\n{run.stdout}\n{run.stderr}")
    return work / "dump"


def load_dump(
    dump: Path, module: str, opencl: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    op = "ekvido" + module
    dev = "GPU" if opencl else "cpu"
    ins = sorted((dump / "export").glob(f"*_{op}_{dev}_in_C.pfm"))
    outs = sorted((dump / "export").glob(f"*_{op}_{dev}_out_C.pfm"))
    if not ins or not outs:
        hint = (
            "it did not run on the GPU (fallback to the CPU? run with -d opencl)"
            if opencl
            else "the module did not run (check params size/version)"
        )
        raise FileNotFoundError(f"no {op} {dev} dump in {dump}: {hint}")
    return read_pfm(ins[-1]), read_pfm(outs[-1])


def load_mask_dump(dump: Path) -> np.ndarray | None:
    """The source mask kazuki received (written by its dt_dump_pfm call), or None."""
    masks = sorted((dump / "ekvidokazuki").glob("*_source_mask_cpu_M.pfm"))
    return read_pfm(masks[-1]) if masks else None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--module", choices=("mist", "kazuki"), required=True)
    ap.add_argument("--preset", default="default")
    ap.add_argument("--darktable-cli")
    ap.add_argument("--input")
    ap.add_argument("--write-xmp", type=Path)
    ap.add_argument("--dump-dir", type=Path)
    ap.add_argument("--diff-image", type=Path)
    ap.add_argument(
        "--legacy-v1",
        action="store_true",
        help="send kazuki params as a version-1 blob (checks legacy_params)",
    )
    ap.add_argument("--xmp", type=Path, help="kazuki: use this darktable edit")
    ap.add_argument(
        "--opencl",
        action="store_true",
        help="render with OpenCL on and compare darktable's GPU output",
    )
    ap.add_argument(
        "--against-cpu",
        action="store_true",
        help="with --opencl: compare the GPU output with darktable's own CPU output "
        "instead of the Python reference",
    )
    args = ap.parse_args(argv)
    presets = MIST_PRESETS if args.module == "mist" else KAZUKI_PRESETS
    if args.preset not in presets:
        ap.error(f"--preset must be one of {sorted(presets)} for {args.module}")
    if args.xmp and args.module != "kazuki":
        ap.error("--xmp is for kazuki")
    if args.against_cpu and not args.opencl:
        ap.error("--against-cpu needs --opencl")
    if args.against_cpu and not (args.darktable_cli and args.input):
        ap.error("--against-cpu needs --darktable-cli and --input")
    if args.write_xmp:
        args.write_xmp.write_text(module_xmp(args.module, args.preset, args.legacy_v1))
        return 0
    mask = None
    try:
        c = kazuki_params_from_xmp(args.xmp) if args.xmp else None
        if args.dump_dir:
            src, got = load_dump(args.dump_dir, args.module, opencl=args.opencl)
            mask = load_mask_dump(args.dump_dir)
        else:
            if not (args.darktable_cli and args.input):
                ap.error("--darktable-cli and --input are required to render")
            with tempfile.TemporaryDirectory() as tmp:
                dump = run_darktable(
                    args.darktable_cli,
                    args.input,
                    args.module,
                    args.preset,
                    Path(tmp),
                    args.legacy_v1,
                    args.xmp,
                    args.opencl,
                )
                src, got = load_dump(dump, args.module, opencl=args.opencl)
                mask = load_mask_dump(dump)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(exc)
        return 2
    if c is not None and c["source_mask_op"] and mask is None:
        print(
            "the edit uses a source mask but kazuki received none (see the darktable log)"
        )
        return 2
    if args.against_cpu:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cpu = run_darktable(
                    args.darktable_cli,
                    args.input,
                    args.module,
                    args.preset,
                    Path(tmp),
                    args.legacy_v1,
                    args.xmp,
                )
                _, want = load_dump(cpu, args.module)
        except (RuntimeError, FileNotFoundError) as exc:
            print(exc)
            return 2
    else:
        want = python_reference(args.module, args.preset, src, c, mask)
    worst, mean, peak, same_class = compare(got, want)
    if args.diff_image:
        write_diff_image(args.diff_image, got, want)
    _, h, w = src.shape
    label = args.xmp.name if args.xmp else args.preset
    if args.against_cpu:
        label += " vs darktable CPU"
    if mask is not None:
        label += f" (source mask from {c['source_mask_op'].decode()}, mean {mask.mean():.3f})"
    print(
        f"{args.module}/{label}: {w}x{h}, max diff {worst:.3g} "
        f"(allowed {TOL * peak:.3g}), mean {mean:.3g} (allowed {MEAN_TOL * peak:.3g}), non-finite match {same_class}"
    )
    return 0 if passes(worst, mean, peak, same_class) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
