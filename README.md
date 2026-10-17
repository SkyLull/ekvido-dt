# ekvido

Two **scene-linear** effect modules for [darktable](https://www.darktable.org/), plus the
Python prototype they were tuned with. Esperanto *ekvido*, "a glimpse".

| Module | What it does |
|--------|--------------|
| **ekvido mist** | Physically based diffusion (white / black mist): bloom whose strength scales with true scene intensity, via convolution with a core + halo + veil PSF. |
| **ekvido kazuki** | "Kazuki light": coloured bloom after Higashiji Kazuki's highlight rendering. Warm orange for sunlight reflections, deep blue to cyan for ambient light and moonlight, shaped as a soft cross or streaks rather than a round blob. Clip Studio Paint blend modes (screen, overlay, colour dodge, glow dodge, *overlay (glow only)*) and an optional source mask taken from another module's mask. |

Both share one engine: extract highlight energy from linear data, convolve it with a
composable PSF (core, halo, veil, optional star/streak arms), colour it, blend it back. In
darktable the modules sit right after *blurs* in the pipe and run on the GPU when OpenCL is
available.

## What's here

```
darktable/            the darktable side: whole files at their darktable paths
  UPSTREAM            the upstream tag and commit these files apply to
  src/iop/ekvido*.c   the two modules
  src/common/ekvido/  the C core (CPU), its OpenCL path (cl/) and standalone tests
  data/kernels/ekvido.cl
  src/external/       vendored VkFFT 1.3.4 (MIT) and PocketFFT (BSD-3-Clause)
  packaging/ekvido/   per-OS deps / build / test / package / install scripts
  po/ekvido/          translations (zh_TW)
  EKVIDO.md           how the darktable side fits together
scripts/inject.sh     clones upstream darktable and writes darktable/ over it
src/ tests/ tools/    the Python prototype, its tests, fixture and comparison tools
```

## Build darktable with ekvido

Requirements:
- [mise](https://mise.jdx.dev/) and git, on every OS.
- **macOS:** Homebrew and the Xcode command line tools. `mise run deps` installs the
  Homebrew packages from darktable's `.ci/Brewfile` without upgrading others.
- **Ubuntu 24.04:** `mise run deps` installs the apt packages listed in
  `darktable/packaging/ekvido/ubuntu-24.04.txt` (asks for sudo).
- **Windows:** MSYS2 at `C:\msys64` and Inno Setup 6 (`winget install MSYS2.MSYS2` /
  `JRSoftware.InnoSetup`). Run the mise tasks from PowerShell or cmd; they run the bash
  scripts in an MSYS2 UCRT64 shell. `mise run deps` installs the pacman packages listed in
  `darktable/packaging/ekvido/msys2-ucrt64.txt`. The first run clones darktable with Git for Windows (on
  the Windows `PATH`); later runs can use MSYS2's own git once `deps` has installed it.

Then:
```bash
mise run deps      # once per machine
mise run build     # builds and installs into this OS's prefix (see below)
mise run test-c    # C core vs the committed fixtures (plain and fast-math builds)
mise run package   # macOS: app bundle; Windows: installer (no-op on Linux)
mise run install   # backs up your darktable config, then installs the build for daily use
```

The first darktable task clones darktable `release-5.6.2` (with its submodules) from GitHub
into `.dt/darktable` and writes the files from `darktable/` over it. Later runs only rewrite
files that changed, so builds stay incremental.

Where things go: macOS builds into `.dt/darktable/build/macosx` and `install` replaces
`/Applications/darktable.app` (the old one goes to the Trash). Linux installs to
`~/.local/opt/darktable-ekvido` with a desktop entry "darktable (ekvido)". Windows builds into
`C:\msys64\opt\darktable-ekvido` and installs as **darktable-dev** beside a stock darktable.
`install` refuses to run while darktable is running and always backs up the config to
`~/darktable-config-backups/` first. Details per OS: `darktable/EKVIDO.md`.

XMP sidecars that use the ekvido modules lose those steps if a stock darktable edits them.

## GPU

Both modules have an OpenCL path (VkFFT for the FFTs). If a module does not fit in the GPU
memory darktable allows, or an allocation fails mid-run, darktable runs that module on the
CPU for that run: the output is the same, only slower. Apple GPUs use the CPU path.
`mise run test-cl` checks the GPU path against the C core (needs an OpenCL GPU with 4 GB
free). On an RTX 4090, a 100 MP export takes kazuki from 9.2 s to 1.0 s and mist from
5.3 s to 0.75 s.

## Python prototype

The reference implementation: a CLI and a browser UI with live sliders on a proxy, working
on 32-bit float linear TIFFs exported from darktable.

```bash
mise run test                                               # pytest
uv run ekvido ui in.tif                                     # browser UI on http://127.0.0.1:8765/
uv run ekvido render in.tif -o out.tif --preset kazuki-default
uv run ekvido chart -o chart.tif                            # synthetic test chart
```

In the UI: wheel = zoom, drag = pan, double-click = fit, hold **B** = before. Settings
autosave to `<image>.ekvido.json`; **render full** writes `<image>.ekvido.tif`.
`tools/make_c_fixtures.py` regenerates the C core's fixtures
(`darktable/src/common/ekvido/tests/fixtures`); `tools/dt_compare.py` compares a
darktable-cli export against the Python result.

## Changing the darktable side

Edit files under `darktable/` and run `mise run build`; only changed files are rewritten into
`.dt/darktable`. To change an upstream file, copy it from `.dt/darktable` into the same path
under `darktable/` first. Files under `darktable/` replace upstream's whole, so they only fit
the tag in `darktable/UPSTREAM`.

## Licence

GPL-3.0-or-later (see `LICENSE`), like darktable. VkFFT (MIT) and PocketFFT (BSD-3-Clause)
keep their own licences under `darktable/src/external/`.
