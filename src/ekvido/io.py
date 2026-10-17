import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

XMP_TAG, IPTC_TAG, ICC_TAG = 700, 33723, 34675
_COPY_TAGS = (XMP_TAG, IPTC_TAG)


@dataclass
class Image:
    data: torch.Tensor  # (3, H, W) float32, CPU
    tags: list[tuple[int, int, int, bytes, bool]] = field(default_factory=list)
    icc: bytes | None = None
    focal_mm: float | None = None


def is_linear_rec2020(icc: bytes | None) -> bool:
    if not icc:
        return False
    return b"Linear Rec2020" in icc.replace(b"\x00", b"")


def parse_focal_mm(xmp: bytes) -> float | None:
    text = xmp.decode("utf-8", "replace")
    m = re.search(r'exif:FocalLength(?:="|>)(\d+(?:\.\d+)?)(?:/(\d+))?', text)
    if not m:
        return None
    num, den = float(m.group(1)), float(m.group(2) or 1)
    return num / den if den else None


def _to_float(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32, copy=False)
    warnings.warn(
        f"TIFF is {arr.dtype}, not float: scaled to 0–1, scene-mode headroom is lost",
        UserWarning,
        stacklevel=3,
    )
    return arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)


def read_mask(path: str | Path, height: int, width: int) -> torch.Tensor:
    """A grey source mask (kazuki, spec §5.7) as (H, W) float32: first channel of a TIFF
    or PNG; integer images are scaled to 0…1, floats are taken as they are."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        arr = tifffile.imread(path)
    else:
        from PIL import Image as PILImage

        with PILImage.open(path) as im:
            arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a grey or RGB mask, got shape {arr.shape}")
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr / float(np.iinfo(arr.dtype).max)
    if arr.shape != (height, width):
        raise ValueError(
            f"{path}: mask is {arr.shape[1]}x{arr.shape[0]}, image is {width}x{height}"
        )
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def read_tiff(path: str | Path) -> Image:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        arr = page.asarray()
        tags = []
        for code in _COPY_TAGS:
            if code in page.tags:
                v = page.tags[code].value
                value = v if isinstance(v, bytes) else np.asarray(v).tobytes()
                # XMP stays BYTE; IPTC is re-written as UNDEFINED bytes (darktable
                # writes it as LONG with a raw-bytes payload, which tifffile won't pack).
                tags.append(
                    (code, 1 if code == XMP_TAG else 7, len(value), value, True)
                )
        icc = bytes(page.tags[ICC_TAG].value) if ICC_TAG in page.tags else None
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"{path}: expected an RGB image, got shape {arr.shape}")
    if arr.shape[2] == 4:
        warnings.warn("TIFF has an alpha channel: dropped", UserWarning, stacklevel=2)
        arr = arr[..., :3]
    if not is_linear_rec2020(icc):
        warnings.warn(
            f"{path}: ICC profile is not linear Rec.2020; treating data as linear Rec.2020 anyway",
            UserWarning,
            stacklevel=2,
        )
    data = (
        torch.from_numpy(np.ascontiguousarray(_to_float(arr)))
        .permute(2, 0, 1)
        .contiguous()
    )
    xmp = next((t[3] for t in tags if t[0] == XMP_TAG), b"")
    return Image(data=data, tags=tags, icc=icc, focal_mm=parse_focal_mm(xmp))


def write_tiff(
    path: str | Path, data: torch.Tensor, image: Image | None = None
) -> None:
    arr = data.detach().to("cpu", torch.float32).permute(1, 2, 0).contiguous().numpy()
    tifffile.imwrite(
        path,
        arr,
        photometric="rgb",
        compression="zlib",
        iccprofile=image.icc if image else None,
        extratags=list(image.tags) if image else [],
    )


def make_proxy(data: torch.Tensor, megapixels: float) -> torch.Tensor:
    """Area-resample (3,H,W) down to ~megapixels; returns `data` itself if already small."""
    _, h, w = data.shape
    scale = math.sqrt(megapixels * 1e6 / (h * w))
    if scale >= 1:
        return data
    size = (max(1, round(h * scale)), max(1, round(w * scale)))
    if (
        data.device.type == "mps"
    ):  # MPS area-pooling needs divisible sizes; do it on CPU
        return F.interpolate(data[None].cpu(), size=size, mode="area")[0].to(
            data.device
        )
    return F.interpolate(data[None], size=size, mode="area")[0]
