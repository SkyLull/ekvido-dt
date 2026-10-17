#!/usr/bin/env bash
# Build and run the standalone ekvido core tests against the committed fixtures, twice:
# with plain Release flags, and with darktable's release math flags (-ffast-math
# -fno-finite-math-only, src/CMakeLists.txt), which is how lib_darktable compiles the core.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
fm="-ffast-math -fno-finite-math-only"
for variant in plain fastmath; do
  dir=build-ekvido-tests
  flags=()
  if [ "$variant" = fastmath ]; then
    dir=build-ekvido-tests-fm
    flags=("-DCMAKE_C_FLAGS=$fm" "-DCMAKE_CXX_FLAGS=$fm")
  fi
  cmake -S src/common/ekvido/tests -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE=Release ${flags[@]+"${flags[@]}"}
  cmake --build "$dir"
  echo "ekvido core tests ($variant):"
  "./$dir/ekvido_core_tests" src/common/ekvido/tests/fixtures
done
