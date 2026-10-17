#!/usr/bin/env bash
# Configure, build and install darktable into this OS's prefix (incremental).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
case "$(uname -s)" in
  Darwin)
    prefix="$PWD/build/macosx"
    extra=(-DCMAKE_OSX_DEPLOYMENT_TARGET=13.5 -DCMAKE_CXX_FLAGS=-stdlib=libc++
           -DCMAKE_OBJCXX_FLAGS=-stdlib=libc++ -DBINARY_PACKAGE_BUILD=ON
           -DDONT_USE_INTERNAL_LUA=OFF -DUSE_COLORD=OFF -DUSE_KWALLET=OFF -DBUILD_CMSTEST=OFF)
    ;;
  Linux)
    prefix="$HOME/.local/opt/darktable-ekvido"
    extra=(-DBINARY_PACKAGE_BUILD=OFF)   # this box only: -march=native
    # Installing over a running darktable can crash it; fail now, before the long build.
    if pgrep -x darktable >/dev/null; then
      echo "build: darktable is running - quit darktable first, then re-run." >&2
      exit 1
    fi
    ;;
  MINGW64*|MSYS*)   # MSYS2 UCRT64; /opt maps to C:\msys64\opt, the tree package.sh wraps
    [ "${MSYSTEM:-}" = UCRT64 ] || { echo "build: run from an MSYS2 UCRT64 shell (or mise run build)" >&2; exit 1; }
    prefix=/opt/darktable-ekvido
    extra=(-DBINARY_PACKAGE_BUILD=OFF)   # this box only: -march=native
    # The build's JSON checks run Python, which on a zh-TW Windows reads files as cp950.
    export PYTHONUTF8=1
    ;;
  *) echo "build: unsupported OS $(uname -s)"; exit 1 ;;
esac
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" "${extra[@]}"
cmake --build build
if [ "$(uname -s)" = Linux ] && pgrep -x darktable >/dev/null; then
  echo "build: darktable was started during the build - quit it; build is done, re-run to install." >&2
  exit 1
fi
cmake --install build
echo "build: installed to $prefix"
