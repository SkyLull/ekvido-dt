import argparse
import math
import sys
import threading
import time
import webbrowser
from pathlib import Path

import torch

from ekvido.chart import make_chart
from ekvido.common.energy import estimate_clip_level
from ekvido.device import pick_device, sync
from ekvido.io import read_mask, read_tiff, write_tiff
from ekvido.modules.kazuki import KazukiParams
from ekvido.modules.mist import MistParams
from ekvido.pipeline import Pipeline, StackEntry
from ekvido.presets import (
    Session,
    load_preset,
    load_sidecar,
    resolve_preset,
    sidecar_path,
)


def _fail(msg: str) -> int:
    print(f"ekvido: {msg}", file=sys.stderr)
    return 2


def _session(args, inp: Path) -> Session:
    if args.sidecar:
        return load_sidecar(Path(args.sidecar))
    if args.preset:
        return Session(stack=[load_preset(resolve_preset(p)) for p in args.preset])
    if sidecar_path(inp).exists():
        return load_sidecar(sidecar_path(inp))
    raise ValueError(
        f"no --sidecar, --preset, or {sidecar_path(inp).name} next to the input"
    )


def cmd_render(args) -> int:
    inp, out = Path(args.input), Path(args.output)
    if out.resolve() == inp.resolve():
        return _fail("refusing to overwrite the input file")
    if not out.parent.is_dir():
        return _fail(f"output directory {out.parent} does not exist")
    if args.invert_source_mask and not args.source_mask:
        return _fail("--invert-source-mask needs --source-mask")
    try:
        session = _session(args, inp)
        image = read_tiff(inp)
        _, h, w = image.data.shape
        mask = None if not args.source_mask else read_mask(args.source_mask, h, w)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        return _fail(
            f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, ValueError)
            else str(exc)
        )
    referred = args.referred or session.referred
    device = pick_device(args.device)
    data = image.data.to(device)
    clip = args.clip_level or session.clip_level
    if clip is None:
        clip = 1.0 if referred == "display" else estimate_clip_level(data)
    result = Pipeline().run(
        data,
        session.stack,
        input_key=str(inp),
        referred=referred,
        clip_level=clip,
        scope=None,
        source_mask=None if mask is None else mask.to(device),
        source_mask_invert=args.invert_source_mask,
    )
    write_tiff(out, result, image)
    print(f"wrote {out} (referred={referred}, clip={clip:.4g}, device={device.type})")
    return 0


def cmd_chart(args) -> int:
    write_tiff(Path(args.output), make_chart(args.width, args.height))
    print(f"wrote {args.output}")
    return 0


def _timed(pipe, img, stack, clip, device) -> tuple[float, dict]:
    timings: dict = {}
    sync(device)
    t0 = time.perf_counter()
    pipe.run(
        img,
        stack,
        input_key="bench",
        referred="scene",
        clip_level=clip,
        scope="bench",
        timings=timings,
    )
    sync(device)
    return time.perf_counter() - t0, timings


def cmd_bench(args) -> int:
    device = pick_device(args.device)
    print(f"device={device.type}")
    for mp in (float(s) for s in args.sizes.split(",")):
        w = round(math.sqrt(mp * 1e6 * 1.5))
        h = round(w / 1.5)
        try:
            img = make_chart(w, h).to(device)
            clip = estimate_clip_level(img)
            for name, params in (
                ("mist", MistParams(arms=4, w_arms=0.1)),
                ("kazuki", KazukiParams()),
            ):
                pipe = Pipeline()
                total, stages = _timed(
                    pipe, img, [StackEntry(name, params)], clip, device
                )
                detail = "  ".join(f"{k} {v * 1e3:.0f}" for k, v in stages.items())
                print(f"{mp:6.2f} MP  {name:<7} total {total * 1e3:8.0f} ms   {detail}")
                if name == "kazuki":
                    recolour = KazukiParams(push_warm=0.3, warm_edge=(1.0, 0.3, 0.0))
                    total, _ = _timed(
                        pipe, img, [StackEntry(name, recolour)], clip, device
                    )
                    print(f"{mp:6.2f} MP  {name:<7} colour-only {total * 1e3:8.0f} ms")
        except torch.OutOfMemoryError:
            print(f"{mp:6.2f} MP  OOM")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return 0


_LOOPBACK = ("127.0.0.1", "localhost", "::1")


def cmd_ui(args) -> int:
    import uvicorn

    from ekvido.server.app import create_app
    from ekvido.server.session import UiSession

    try:
        session = UiSession(
            Path(args.input), referred=args.referred, device=args.device
        )
    except (ValueError, OSError) as exc:
        return _fail(str(exc))
    if args.host not in _LOOPBACK:
        print(
            f"ekvido: warning: serving on {args.host} without authentication",
            file=sys.stderr,
        )
    wildcard = ("0.0.0.0", "::")  # nosec B104 - compared against --host, not bound
    shown = "127.0.0.1" if args.host in wildcard else args.host
    url = f"http://{shown}:{args.port}/"
    print(f"ekvido ui: {url}  (Ctrl-C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, (url,)).start()
    hosts = ["*"] if args.host in wildcard else [*_LOOPBACK, args.host]
    app = create_app(session, allowed_hosts=hosts)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ekvido", description="scene-linear photo filters"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="filter a float TIFF at full resolution")
    r.add_argument("input")
    r.add_argument("-o", "--output", required=True)
    src = r.add_mutually_exclusive_group()
    src.add_argument("--sidecar", help="session JSON (default: <input>.ekvido.json)")
    src.add_argument(
        "--preset",
        action="append",
        help="preset name or JSON path; repeatable, applied in order",
    )
    r.add_argument("--referred", choices=["scene", "display"])
    r.add_argument("--clip-level", type=float)
    r.add_argument(
        "--source-mask",
        help="grey TIFF/PNG, same size: where kazuki's light may come from (1 = emits)",
    )
    r.add_argument(
        "--invert-source-mask", action="store_true", help="use 1 − the source mask"
    )
    r.add_argument("--device", choices=["cuda", "mps", "cpu"])
    r.set_defaults(fn=cmd_render)

    c = sub.add_parser("chart", help="write the synthetic test chart")
    c.add_argument("-o", "--output", required=True)
    c.add_argument("--width", type=int, default=1800)
    c.add_argument("--height", type=int, default=1200)
    c.set_defaults(fn=cmd_chart)

    b = sub.add_parser("bench", help="time each stage at several sizes")
    b.add_argument("--device", choices=["cuda", "mps", "cpu"])
    b.add_argument("--sizes", default="2.5,10", help="comma-separated megapixels")
    b.set_defaults(fn=cmd_bench)
    u = sub.add_parser("ui", help="open the browser UI for a float TIFF")
    u.add_argument("input")
    u.add_argument("--referred", choices=["scene", "display"])
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--device", choices=["cuda", "mps", "cpu"])
    u.add_argument("--no-browser", action="store_true")
    u.set_defaults(fn=cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
