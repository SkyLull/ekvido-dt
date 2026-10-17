"use strict";
const $ = (id) => document.getElementById(id);

let schema = null;            // GET /api/state
let session = null;           // sidecar-shaped {referred, clip_level, stack: [{module, version, enabled, params}]}
let crop = [0, 0, 1, 1];      // visible region, normalised x0, y0, x1, y1
let holdBefore = false;
let reqId = 0, shownId = 0, shownLevel = -1;
let ws = null, pending = false, lastUrl = null;

// ---- colour: linear Rec.2020 <-> sRGB hex (for the pickers only) ----
const M2020_TO_709 = [[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]];
const M709_TO_2020 = [[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]];
const mul = (m, v) => m.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
const oetf = (x) => { x = Math.min(1, Math.max(0, x)); return x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055; };
const eotf = (x) => (x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4);
function rec2020ToHex(c) {
  const s = mul(M2020_TO_709, c).map((v) => Math.max(0, v));
  const m = Math.max(1e-9, ...s);   // ramp colours are hues; scale to the brightest channel
  return "#" + s.map((v) => Math.round(oetf(v / m) * 255).toString(16).padStart(2, "0")).join("");
}
function hexToRec2020(h) {
  const s = [1, 3, 5].map((i) => eotf(parseInt(h.slice(i, i + 2), 16) / 255));
  return mul(M709_TO_2020, s).map((v) => Math.max(0, +v.toFixed(5)));
}

// ---- small DOM helpers ----
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
function button(text, onclick, title) { const b = el("button", "small", text); b.onclick = onclick; if (title) b.title = title; return b; }
function status(text, bad = false) { const s = $("status"); s.textContent = text; s.classList.toggle("bad", bad); }

// ---- boot ----
async function init() {
  schema = await (await fetch("/api/state")).json();
  session = schema.session;
  $("image-name").textContent = `${schema.image} · ${schema.size[0]}×${schema.size[1]} · ${schema.device}`;
  for (const v of schema.views) $("view").add(new Option(v, v));
  $("referred").value = session.referred;
  buildStack();
  bindViewport();
  bindToolbar();
  connect();
}

function connect() {
  const url = new URL("/ws", location.href);
  url.protocol = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => send();
  ws.onmessage = onMessage;
  ws.onclose = () => { status("disconnected — retrying", true); setTimeout(connect, 1000); };
}

function onMessage(ev) {
  if (typeof ev.data === "string") {
    const m = JSON.parse(ev.data);
    if (m.type === "reload") { location.reload(); return; }   // server restarted (maybe on another image)
    if (m.type === "error") status(`error: ${m.message}`, true);
    else if (m.type === "timing") status(Object.entries(m.timings).map(([k, v]) => `${k} ${(v * 1000).toFixed(0)} ms`).join(" · "));
    return;
  }
  const dv = new DataView(ev.data);
  const id = dv.getUint32(0, true), level = dv.getUint8(4);
  if (id < shownId || (id === shownId && level <= shownLevel)) return;   // older than what is on screen
  shownId = id; shownLevel = level;
  const url = URL.createObjectURL(new Blob([ev.data.slice(5)], { type: "image/webp" }));
  $("frame").src = url;
  if (lastUrl) URL.revokeObjectURL(lastUrl);
  lastUrl = url;
}

function viewState() {
  const vp = $("viewport"), dpr = window.devicePixelRatio || 1;
  return {
    view: holdBefore ? "before" : $("view").value,
    exposure: +$("exposure").value,
    crop,
    width: Math.round(vp.clientWidth * dpr),
    height: Math.round(vp.clientHeight * dpr),
  };
}

function send() {   // at most one message per animation frame
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    reqId += 1;
    ws.send(JSON.stringify({ id: reqId, instance: schema.instance, session, view: viewState() }));
  });
}

// ---- module stack panel ----
function buildStack() {
  const root = $("stack");
  root.textContent = "";
  session.stack.forEach((entry, i) => root.append(modulePanel(entry, i)));
}

function move(i, d) {
  const j = i + d;
  if (j < 0 || j >= session.stack.length) return;
  [session.stack[i], session.stack[j]] = [session.stack[j], session.stack[i]];
  buildStack();
  send();
}

function modulePanel(entry, i) {
  const spec = schema.modules[entry.module];
  const box = el("section", "module");
  const head = el("div", "module-head");
  const on = el("input");
  on.type = "checkbox";
  on.checked = entry.enabled;
  on.onchange = () => { entry.enabled = on.checked; send(); };
  const title = el("span", "module-title", entry.module);
  title.onclick = () => box.classList.toggle("collapsed");
  head.append(on, title, button("↑", () => move(i, -1), "move up"), button("↓", () => move(i, 1), "move down"), presetControls(entry));
  const body = el("div", "module-body");
  for (const p of spec.params) {
    if (p.mode && p.mode !== session.referred) continue;
    body.append(control(entry, p));
  }
  box.append(head, body);
  return box;
}

function control(entry, p) {
  const row = el("label", "row");
  row.title = p.desc;
  row.append(el("span", "name", p.name));
  const set = (v) => { entry.params[p.name] = v; send(); };
  const value = entry.params[p.name];
  if (p.kind === "float" || p.kind === "int") {
    const step = p.kind === "int" ? 1 : (p.max - p.min) / 1000;
    const range = el("input"), num = el("input");
    for (const inp of [range, num]) { inp.type = inp === range ? "range" : "number"; inp.min = p.min; inp.max = p.max; inp.step = step; inp.value = value; }
    range.oninput = () => { num.value = range.value; set(+range.value); };
    num.onchange = () => { const v = Math.min(p.max, Math.max(p.min, +num.value)); num.value = range.value = v; set(v); };
    range.ondblclick = () => { range.value = num.value = p.default; set(p.default); };
    row.append(range, num);
  } else if (p.kind === "bool") {
    const box = el("input");
    box.type = "checkbox";
    box.checked = value;
    box.onchange = () => set(box.checked);
    row.append(box);
  } else if (p.kind === "choice") {
    const sel = el("select");
    for (const c of p.choices) sel.add(new Option(c, c));
    sel.value = value;
    sel.onchange = () => set(sel.value);
    row.append(sel);
  } else if (p.kind === "color") {
    const pick = el("input");
    pick.type = "color";
    pick.value = rec2020ToHex(value);
    pick.oninput = () => set(hexToRec2020(pick.value));
    pick.ondblclick = () => { pick.value = rec2020ToHex(p.default); set(p.default); };
    row.append(pick);
  }
  return row;
}

function presetControls(entry) {
  const wrap = el("span", "presets");
  const sel = el("select");
  sel.add(new Option("preset…", ""));
  for (const [name, module] of Object.entries(schema.presets)) if (module === entry.module) sel.add(new Option(name, name));
  sel.onchange = async () => {
    if (!sel.value) return;
    const r = await fetch(`/api/presets/${encodeURIComponent(sel.value)}`);
    if (!r.ok) { status(`preset failed: ${(await r.json()).detail}`, true); return; }
    const p = await r.json();
    if (p.module !== entry.module) { status(`preset ${sel.value} is for ${p.module}`, true); return; }
    entry.version = p.version;
    entry.params = { ...entry.params, ...p.params };
    buildStack();
    send();
  };
  const save = button("save", async () => {
    const name = prompt(`Save ${entry.module} settings as preset:`);
    if (!name) return;
    const r = await fetch("/api/presets", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, entry }) });
    const j = await r.json();
    if (!r.ok) { status(`save failed: ${j.detail}`, true); return; }
    schema.presets[name] = entry.module;
    status(`saved preset ${name}`);
    buildStack();
  }, "save these settings as a preset");
  wrap.append(sel, save);
  return wrap;
}

// ---- image: zoom / pan ----
function clampCrop([x0, y0, x1, y1]) {
  const w = Math.min(1, x1 - x0), h = Math.min(1, y1 - y0);
  x0 = Math.min(Math.max(0, x0), 1 - w);
  y0 = Math.min(Math.max(0, y0), 1 - h);
  return [x0, y0, x0 + w, y0 + h];
}

function bindViewport() {
  const vp = $("viewport"), frame = $("frame");
  vp.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = frame.getBoundingClientRect();
    if (!r.width) return;
    const [x0, y0, x1, y1] = crop;
    const fx = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const fy = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
    const px = x0 + fx * (x1 - x0), py = y0 + fy * (y1 - y0);
    const k = Math.exp(e.deltaY * 0.0015);   // > 1 zooms out
    const w = Math.min(1, Math.max(0.002, (x1 - x0) * k)), h = Math.min(1, Math.max(0.002, (y1 - y0) * k));
    crop = clampCrop([px - fx * w, py - fy * h, px - fx * w + w, py - fy * h + h]);
    send();
  }, { passive: false });
  let drag = null;
  vp.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, crop: crop.slice() }; vp.setPointerCapture(e.pointerId); });
  vp.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const r = frame.getBoundingClientRect();
    if (!r.width) return;
    const dx = ((e.clientX - drag.x) / r.width) * (drag.crop[2] - drag.crop[0]);
    const dy = ((e.clientY - drag.y) / r.height) * (drag.crop[3] - drag.crop[1]);
    crop = clampCrop([drag.crop[0] - dx, drag.crop[1] - dy, drag.crop[2] - dx, drag.crop[3] - dy]);
    send();
  });
  vp.addEventListener("pointerup", () => { drag = null; });
  vp.addEventListener("dblclick", () => { crop = [0, 0, 1, 1]; send(); });
  new ResizeObserver(() => send()).observe(vp);
}

// ---- toolbar ----
const isTyping = (e) => ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName) && !["range", "checkbox", "color"].includes(e.target.type);

function bindToolbar() {
  $("view").onchange = send;
  $("exposure").oninput = () => { $("exposure-val").textContent = (+$("exposure").value).toFixed(2); send(); };
  $("exposure").ondblclick = () => { $("exposure").value = 0; $("exposure").oninput(); };
  $("referred").onchange = () => { session.referred = $("referred").value; buildStack(); send(); };
  $("fit").onclick = () => { crop = [0, 0, 1, 1]; send(); };
  const setBefore = (v) => { if (holdBefore !== v) { holdBefore = v; $("before").classList.toggle("on", v); send(); } };
  $("before").onpointerdown = () => setBefore(true);
  $("before").onpointerup = $("before").onpointerleave = () => setBefore(false);
  addEventListener("keydown", (e) => { if (e.key === "b" && !e.repeat && !isTyping(e)) setBefore(true); });
  addEventListener("keyup", (e) => { if (e.key === "b") setBefore(false); });
  $("render").onclick = async () => {
    status("rendering full resolution…");
    $("render").disabled = true;
    try {
      const r = await fetch("/api/render", { method: "POST" });
      const j = await r.json();
      status(r.ok ? `wrote ${j.path} in ${j.seconds.toFixed(1)} s` : `render failed: ${j.detail}`, !r.ok);
    } catch (err) {
      status(`render failed: ${err}`, true);
    } finally {
      $("render").disabled = false;
    }
  };
}

init().catch((err) => status(`failed to start: ${err}`, true));
