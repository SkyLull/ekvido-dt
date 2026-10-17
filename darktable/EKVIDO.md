# ekvido fork of darktable

Branch `ekvido` = upstream release tag + the ekvido modules *ekvido mist* and *ekvido kazuki*. Python reference implementation and fixture generator: the ekvido repo (`..`).

What the fork adds:
- `src/common/ekvido/` — pure C/C++ core (kernel builder, PocketFFT convolution, mist, kazuki: energy, blur, warmth, blend modes incl.
  Clip Studio Paint's and overlay (glow only)), also
  built standalone by `mise run test` against fixtures from `../tools/make_c_fixtures.py`.
- `src/common/ekvido/cl/` + `data/kernels/ekvido.cl` — the GPU path (OpenCL, spec 2026-10-16
  §4.4): the same pipelines as the C core on the device, VkFFT for the FFTs, only the final glow
  cached on the card. darktable-independent except `ekv_cl_dt.c`; all OpenCL calls (ours and
  VkFFT's) go through a function table filled from darktable's `dlopen`ed symbols. Tested
  GPU-vs-CPU by `mise run test-cl` (needs an OpenCL GPU). `ekvido.cl` is built by
  ekvido itself, **not** listed in `data/kernels/programs.conf`.
- `src/external/vkfft/` — vendored VkFFT v1.3.4 (MIT), see its `README.ekvido`.
- `src/iop/ekvidomist.c`, `src/iop/ekvidokazuki.c` — the modules (kazuki then mist, right after
  `blurs`, everywhere). Registration points (check all of them when rebasing, and when adding
  a module):
  - `src/iop/CMakeLists.txt` — `add_iop(...)` x2.
  - `src/common/iop_order.c` — the five order tables plus the `_insert_before` lines for both modules.
  - `src/libs/modulegroups.c` — built-in layouts: `AM("ekvidokazuki")` and `AM("ekvidomist")` after
    `AM("blurs")` in "modules: all" and "workflow: scene-referred". There is no `@@_NEW_MODULE` marker
    there, so this one is easy to miss.
  - `src/CMakeLists.txt` — the core sources in `lib_darktable`'s `SOURCE_FILES`.
  - `src/CMakeLists.txt` also lists `common/ekvido/cl/*.c` and sets `-w` on `ekv_cl_fft.c`
    (VkFFT is not warning-clean; lib_darktable builds with -Werror).
  - `src/develop/imageop.c` — `dt_iop_update_multi_priority()`: only renumbers a raster-mask
    user's blend params that reference that source, because `ekvidokazuki` also registers in
    `raster_mask.source.users` for its source mask (hunks marked `ekvido:`).
  - `src/develop/pixelpipe_hb.c` — not patched, but `_piece_skipped()` in
    `src/iop/ekvidokazuki.c` must mirror its static `_skip_piece_on_tags()`; re-check on rebase.
  - `src/common/ekvido/cl/ekv_cl_dt.c` (not a registration, but darktable internals): it fills the
    OpenCL table from `dt_dlopencl_symbols_t`'s `dt_cl*` names and reads `dt_opencl_device_t`'s
    `context` / `devid` / `cmd_queue`; the modules rely on the `process_cl` error → CPU `process()`
    fallback in `src/develop/pixelpipe_hb.c`. Re-check all three on rebase.
  - `src/libs/history.c` — the history tooltip reads enum labels with `Q_()` instead of `_()`, so
    kazuki's `"kazuki blend|…"` blend-mode labels (context-prefixed, see below) show without the prefix.
  - `po/CMakeLists.txt` — lays `po/ekvido/<lang>.po` over `po/<lang>.po` with `msgcat --use-first`
    before `msgfmt` (hunk marked `ekvido:`).
- `po/ekvido/` — translations of the ekvido strings (`zh_TW.po`) and their template `ekvido.pot`.
  Kept apart from upstream's `po/*.po`, which upstream rewrites on every translation refresh.
- `src/external/pocketfft/` — vendored PocketFFT (BSD-3).
- `mise.toml` + `packaging/ekvido/` — per-box build/install.

Build on each box: `mise run deps` (once), `mise run build`, `mise run test`,
`mise run package` (macOS app / Windows installer), `mise run install`.

`mise run test` builds the core tests twice — plain and with darktable's `-ffast-math
-fno-finite-math-only` — because `lib_darktable` compiles the core with those flags (they once
turned mist's `+Inf` pixels into `NaN`, and caught GCC-only NaN losses in the blend dodge on
Windows).

Speed: `bash packaging/ekvido/bench-core.sh [24|100|all]` times the core alone on a synthetic
frame (export, and one shape / colour / strength change against a warm cache; one process per
case so the peak RSS is that case's). `packaging/ekvido/speed.sh` times real darktable-cli
exports (usage in the script). Kazuki's FFT budget since the Phase 1 restructure: export 12
FFTs (7 basis spectra, 2 band kernels, 3 inverse), shape change 5, colour/pull change 3,
strength/blend/white 0 — the recolouring is folded into the spectra (`ekvido_kazuki_coefficients`).

GPU: both modules have `process_cl`. darktable uses it when OpenCL is on; if the module's need (`tiling_callback`: images + pipeline peak + cached glow) does not fit
darktable's OpenCL budget, the module runs on the CPU, silently. darktable budgets from the card's
*total* memory, so when other processes hold most of the GPU memory our allocations can fail mid-run: the
module then returns an error and darktable runs it on the CPU for that run — correct output,
just slower (spec D8). Observed at 100 MP: with ~5.7 GB free ekvidokazuki itself fell back to
the CPU (no pipe restart); with ~0.5 GB free (24 MP) darktable's own colorin failed first and the
whole pipe restarted on the CPU. Both gave the correct image. For heavy GPU editing, free GPU memory or raise darktable's OpenCL headroom / resource level settings in preferences. If darktable itself cannot
copy the image back, it restarts the whole pipe on the CPU, and after 5 such restarts it
switches OpenCL off for the session. VkFFT plans (0.5–1.3 s each to build) are cached in
`~/.cache/darktable/ekvido-vkfft/` (e.g. `vkfft134-r2c-ut1-NVIDIA_GeForce_RTX_4090-580.178.04-6000x8000.bin`;
`r2c-ut1` tags the VkFFT configuration and is bumped when `_init` in `ekv_cl_fft.c` changes). A
plan file whose size does not match VkFFT's own header is deleted and rebuilt. Files accumulate
per driver version and grid size (and untagged ones from before the tag); the folder is always
safe to delete — plans are simply rebuilt. VRAM while darktable is idle: per device, the
compiled plans (kernels only, at most 2) and the interactive cached glow (3 planes per kazuki /
mist instance); VkFFT's scratch (one grid, 733 MB at 100 MP) is allocated per run and released
after each pipeline's FFT phase. Measured on an RTX 4090 (module time, 100 MP full-res export —
small-frame GPU times are noisy, spec §10.9): kazuki 9.2 → 1.0 s, mist 5.3 → 0.75 s. Speed: `OPENCL=1 packaging/ekvido/speed.sh …`; `bash packaging/ekvido/test-cl.sh
bench W H` times the GPU core alone. Debug: `darktable -d opencl -d perf`. The Mac stays CPU-only:
`ekv_cl_dt_get` returns NULL on Apple, so the modules run `process()` even with OpenCL on (spec D1).

Windows: verified with MSYS2 UCRT64 (`test-cl.sh` passes; darktable runs both modules on the GPU).
VkFFT prints its kernel constants with `sprintf("%.17Le", long double)`, which MinGW's default
MSVCRT printf garbles (every FFT ≈ 0); darktable's build sets `__USE_MINGW_ANSI_STDIO`
(src/CMakeLists.txt) and `ekv_cl_fft.c` sets it too, for the standalone tests. Any new VkFFT
translation unit needs the same. If FFT checks fail totally on Windows, look here first, then
delete stale plan files (they carry the broken kernels).

Source mask: *ekvido kazuki* params version 2 add a source mask (spec §5.7): kazuki registers
in the chosen module's raster-mask users table and carries that mask through the distorting
modules between them to its own input (zoom margin included) — see `_source_mask()` in
`ekvidokazuki.c`; darktable's `dt_dev_get_raster_mask()` only delivers masks at a module's
output size. Both ekvido modules implement `distort_mask()` (a crop), so raster masks used
after them work at 100 % zoom. Renumbering the source module's instances leaves kazuki's
choice stale (the module warning says so): pick the source again.

Translations: after changing any user-visible string in the two modules, run `mise run l10n`. It
extracts the strings from the module sources and their generated introspection (slider and combobox
labels) into `po/ekvido/ekvido.pot`, then merges that into each `po/ekvido/*.po` (new strings show
as untranslated in the printed statistics). Blend-mode labels use darktable's `"context|label"`
convention (`src/common/darktable.h`), turned into a real gettext context `kazuki blend` on
extraction, because upstream zh_TW already translates a bare "overlay" as 套疊. Terms follow
upstream zh_TW (像素, 亮部, 遮罩, 臨界值) and Clip Studio Paint for the blend modes. Adding a
language: copy `zh_TW.po`'s header with that language's `Plural-Forms` (must match upstream's
`po/<lang>.po`, or msgfmt fails), then `mise run l10n`.

Rebase onto a new release: `git fetch upstream --tags && git rebase --onto release-X.Y.Z
release-5.6.2 ekvido`, then re-run the build and tests.

Caveat: XMP sidecars with ekvido steps, if edited in a stock darktable, are rewritten
without those steps.

## Installation notes

**macOS:** `mise run install` refuses to run while darktable is running — quit it first. Once
running, it moves the previous `/Applications/darktable.app` to the Trash (renamed
`darktable-<old version>-<timestamp>.app`) and backs up `~/.config/darktable` to
`~/darktable-config-backups/darktable-config-<timestamp>.tgz`. Deployment target is macOS 13.5.

**Linux (Ubuntu 24.04):** `mise run install` refuses to run while darktable is running — quit it
first. Once running, it installs to the prefix `~/.local/opt/darktable-ekvido` and creates
a desktop entry "darktable (ekvido)". On every install, `~/.config/darktable` is
backed up to `~/darktable-config-backups/darktable-config-<timestamp>.tgz`. If
`~/.config/darktable` has no `library.db` AND the Flatpak config
(`~/.var/app/org.darktable.Darktable/config/darktable`) has one, the existing
`~/.config/darktable` is moved aside to `~/.config/darktable.pre-ekvido-<timestamp>` and
the Flatpak config is copied into `~/.config/darktable`. Otherwise the backup is the only
change. The Flatpak itself is never modified.

**Windows (MSYS2 UCRT64):** needs MSYS2 at `C:\msys64` and Inno Setup 6 (`winget install
MSYS2.MSYS2` / `JRSoftware.InnoSetup`); run the mise tasks from PowerShell or cmd in the
checkout and they run the same scripts in an MSYS2 UCRT64 shell
(`packaging/ekvido/win.cmd`). `build` installs into `C:\msys64\opt\darktable-ekvido`;
`package` turns that into `Output\darktable-<version>-win64.exe` (Inno Setup).
`mise run install` refuses while darktable runs, backs up `%LOCALAPPDATA%\darktable` to
`~/darktable-config-backups/`, silently uninstalls the previous fork build and installs the
new one. Because the version is longer than a release's, the installer names it
**darktable-dev** (`C:\Program Files\darktable-dev`, Start menu "darktable-dev"), so it
sits beside stock darktable. Both share the catalogue in `%LOCALAPPDATA%\darktable`. The fork
takes over "Open with" / App Paths registrations; uninstalling it removes those.

## End-to-end check

From the ekvido repo, run:
```bash
uv run python tools/dt_compare.py --module mist|kazuki --preset <name> \
  --darktable-cli <path to darktable-cli> --input <float linear Rec.2020 TIFF> [--diff-image diff.tif]
```
Presets: mist `default`, `black`, `star`; kazuki `default`, `wide`, `screen` (manual clip, 6
arms), `glowdodge`, `overlayglow`. `--write-xmp` writes the XMP for running darktable
elsewhere and `--dump-dir` compares that run's dump (how a second machine is checked). Also checked (kazuki): the `missingsource` preset (source-mask op not in the pipe, glow
falls back to unmasked); `--legacy-v1` (v1 params blob still loads and matches); and
`--xmp EDIT.xmp` (e.g. `tests/data/dt-source-mask.xmp` with the TIFF it was made from: a real
darktable edit with a source mask).
Results: spec §10.
Phase 2 spikes: packaging/ekvido/spikes/ (VkFFT on OpenCL vs PocketFFT; Linux; fetches VkFFT into $TMPDIR, nothing enters the build). Results: OpenCL design spec §8.
