from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src/ekvido/server/static"


def test_page_wires_the_protocol():
    html = (STATIC / "index.html").read_text()
    js = (STATIC / "app.js").read_text()
    for el_id in (
        "view",
        "exposure",
        "before",
        "fit",
        "render",
        "status",
        "viewport",
        "frame",
        "referred",
        "stack",
    ):
        assert f'id="{el_id}"' in html, el_id
    assert "/static/app.js" in html and "/static/style.css" in html
    for needle in (
        '"/ws"',
        "/api/state",
        "/api/render",
        "/api/presets",
        "getUint32(0, true)",
        "getUint8(4)",
    ):
        assert needle in js, needle


def test_page_follows_the_hardened_protocol():
    js = (STATIC / "app.js").read_text()
    assert (
        "instance: schema.instance" in js
    )  # every WS message names its server instance
    assert '"reload"' in js and "location.reload()" in js
    assert "JSON.stringify({ name, entry })" in js  # preset saves post the shown entry
    assert "p.module !== entry.module" in js  # never load another module's preset
    assert "index: i" not in js
