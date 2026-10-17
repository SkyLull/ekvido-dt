#!/usr/bin/env bash
# macOS: build the .app bundle from build/macosx with darktable's own script (ad-hoc signed).
# Windows: build the Inno Setup installer.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
case "$(uname -s)" in
  Darwin) (cd packaging/macosx && ./3_make_hb_darktable_package.sh)
          echo "package: build/macosx/package/darktable.app" ;;
  MINGW64*|MSYS*)
    # Windows: Inno Setup installer from the build.sh prefix, as darktable's nightly does.
    prefix=/opt/darktable-ekvido
    iscc=""
    for c in "$(cygpath -u "${LOCALAPPDATA:-}")/Programs/Inno Setup 6/ISCC.exe" "/c/Program Files (x86)/Inno Setup 6/ISCC.exe"; do
      [ -x "$c" ] && iscc="$c" && break
    done
    [ -n "$iscc" ] || { echo "package: Inno Setup 6 not found (winget install JRSoftware.InnoSetup)" >&2; exit 1; }
    magick data/pixmaps/256x256/darktable.png -define icon:auto-resize="256,128,64,48,32,16" "$prefix/dt_logo_multiresolution.ico"
    cp AUTHORS LICENSE build/darktable.iss "$prefix/"
    rm -rf "$prefix/Output"
    (cd "$prefix" && "$iscc" //Q darktable.iss)
    echo "package: $(cygpath -w "$prefix"/Output/darktable-*.exe)" ;;
  *) echo "package: nothing to do on $(uname -s)" ;;
esac
