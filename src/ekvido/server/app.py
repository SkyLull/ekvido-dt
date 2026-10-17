import asyncio
import json
import struct
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import PlainTextResponse

from ekvido.server.session import UiSession, ViewRequest

STATIC = Path(__file__).with_name("static")
LEVEL_CODE = {"quick": 0, "proxy": 1}
_BAD_INPUT = (
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
    IndexError,
    ArithmeticError,
)
LOOPBACK = ("127.0.0.1", "localhost", "::1")


class BadFrame:
    """Placeholder the reader queues for a frame it could not parse."""

    def __init__(self, message: str):
        self.message = message


class SameOriginMiddleware:
    """Refuse browser requests from other sites: an Origin header must name this host.

    WebSockets are not covered by CORS, so without this any page open in the browser
    could read previews or push state. Clients that send no Origin (scripts) pass.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope["headers"])
            origin = headers.get(b"origin")
            host = headers.get(b"host", b"").decode("latin-1")
            if origin is not None and urlsplit(origin.decode("latin-1")).netloc != host:
                if scope["type"] == "websocket":
                    await receive()  # websocket.connect
                    await send({"type": "websocket.close", "code": 1008})
                    return
                await PlainTextResponse("cross-origin request refused", 403)(
                    scope, receive, send
                )
                return
        await self.app(scope, receive, send)


class LatestBox:
    """Holds only the newest item: a slow consumer never works through a backlog."""

    def __init__(self):
        self._item: Any = None
        self._seq = 0
        self._event = asyncio.Event()

    def put(self, item: Any) -> None:
        self._item = item
        self._seq += 1
        self._event.set()

    async def get(self) -> tuple[int, Any]:
        await self._event.wait()
        self._event.clear()
        return self._seq, self._item

    def is_stale(self, seq: int) -> bool:
        return seq != self._seq


class PresetSave(BaseModel):
    name: str
    entry: dict


def create_app(session: UiSession, allowed_hosts=LOOPBACK) -> FastAPI:
    app = FastAPI(title="ekvido")
    app.add_middleware(SameOriginMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    def state():
        return session.state()

    @app.post("/api/render")
    async def render():
        path, seconds = await run_in_threadpool(session.render_full)
        return {"path": str(path), "seconds": seconds}

    @app.post("/api/presets")
    def save_preset(body: PresetSave):
        try:
            path = session.save_preset(body.name, body.entry)
        except _BAD_INPUT as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"name": body.name, "path": str(path)}

    @app.get("/api/presets/{name}")
    def get_preset(name: str):
        try:
            return session.preset_entry(name)
        except KeyError as exc:
            raise HTTPException(404, f"no preset {name!r}") from exc

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        box = LatestBox()

        async def reader():
            while True:
                frame = await socket.receive()
                if frame["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(frame.get("code", 1000))
                text = frame.get("text")
                if text is None:
                    box.put(BadFrame("expected a JSON text frame"))
                    continue
                try:
                    box.put(json.loads(text))
                except ValueError as exc:
                    box.put(BadFrame(f"not JSON: {exc}"))

        async def handle(seq: int, msg: Any):
            rid = 0
            try:
                if isinstance(msg, BadFrame):
                    raise ValueError(msg.message)
                if not isinstance(msg, dict):
                    raise TypeError("message must be a JSON object")
                rid = int(msg.get("id", 0)) & 0xFFFFFFFF
                if msg.get("instance", session.instance) != session.instance:
                    await socket.send_json({"type": "reload", "id": rid})
                    return
                if "session" in msg:
                    session.apply_state(msg["session"])
                req = ViewRequest.from_dict(msg.get("view") or {})
            except _BAD_INPUT as exc:
                await socket.send_json(
                    {"type": "error", "id": rid, "message": str(exc)}
                )
                return
            levels = ["quick", "proxy"] if session.should_quick() else ["proxy"]
            for level in levels:
                if box.is_stale(seq):
                    return
                data = await run_in_threadpool(session.render, level, req)
                if box.is_stale(seq):
                    return
                await socket.send_bytes(
                    struct.pack("<IB", rid, LEVEL_CODE[level]) + data
                )
            await socket.send_json(
                {"type": "timing", "id": rid, "timings": session.timings}
            )

        async def worker():
            while True:
                seq, msg = await box.get()
                try:
                    await handle(seq, msg)
                except Exception as exc:  # last resort: report, keep serving
                    if isinstance(exc, WebSocketDisconnect):
                        raise
                    await socket.send_json(
                        {
                            "type": "error",
                            "id": msg.get("id") if isinstance(msg, dict) else None,
                            "message": repr(exc),
                        }
                    )

        tasks = [asyncio.create_task(reader()), asyncio.create_task(worker())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception()
            if exc is not None and not isinstance(exc, WebSocketDisconnect):
                raise exc

    return app
