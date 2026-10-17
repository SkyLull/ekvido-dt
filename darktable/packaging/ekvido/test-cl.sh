#!/usr/bin/env bash
# GPU-vs-CPU tests of the ekvido OpenCL path; needs an OpenCL GPU. Builds with
# darktable's release math flags like lib_darktable. usage: test-cl.sh [bench W H]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
# free-VRAM guard (other GPU users may hold memory): 4096 MiB, or 14336 MiB for `bench W H` with W*H >= 50 MP
if command -v nvidia-smi >/dev/null 2>&1; then
  need=4096
  if [ "${1:-}" = bench ] && [ -n "${3:-}" ] && [ $(( $2 * $3 )) -ge 50000000 ]; then need=14336; fi
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
  echo "free VRAM: ${free} MiB (need ${need} MiB)"
  if ! [ "$free" -ge "$need" ] 2>/dev/null; then
    echo "test-cl.sh: not enough free VRAM (${free} MiB < ${need} MiB); free some GPU memory and retry" >&2
    exit 3
  fi
fi
fm="-ffast-math -fno-finite-math-only"
dir=build-ekvido-cl-tests
run=()
[ "$(uname -s)" = Linux ] && run=(nice -n 10)
${run[@]+"${run[@]}"} cmake -S src/common/ekvido/tests/cl -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_C_FLAGS=$fm" "-DCMAKE_CXX_FLAGS=$fm" >/dev/null
${run[@]+"${run[@]}"} cmake --build "$dir" >/dev/null
"./$dir/ekvido_cl_tests" data/kernels/ekvido.cl "$@"
