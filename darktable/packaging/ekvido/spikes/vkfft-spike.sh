#!/usr/bin/env bash
# S1: fetch VkFFT v1.3.4 into a scratch dir, build the spike against the fork's Khronos headers and
# the system libOpenCL, run it on the 24 MP and 100 MP kazuki grids. Linux only.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
work=${TMPDIR:-/tmp}/ekvido-vkfft-spike
mkdir -p "$work" && cd "$work"
[ -d VkFFT-1.3.4 ] || curl -sSL https://github.com/DTolm/VkFFT/archive/refs/tags/v1.3.4.tar.gz | tar xz
g++ -O2 -std=c++17 -I"$root/src" -c "$root/src/common/ekvido/fft.cc" -o fft.o -pthread
gcc -O2 -std=c11 -I"$root/src" -I"$root/src/external/OpenCL" -IVkFFT-1.3.4/vkFFT \
  "$root/packaging/ekvido/spikes/vkfft_spike.c" fft.o -o vkfft_spike -l:libOpenCL.so.1 -lstdc++ -lm -pthread
# grids ekvido_fft_size gives at the 0.25-diagonal cap: 24 MP (6016x4016 -> 5832x8000) and 100 MP (11648x8736 -> 12500x15360)
for g in "5832 8000" "12500 15360"; do ./vkfft_spike $g; done
