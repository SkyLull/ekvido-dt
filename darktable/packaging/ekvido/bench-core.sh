#!/usr/bin/env bash
# Core speed/memory benchmark: builds ekvido_core_bench with darktable's release math flags and
# runs every case once per size. usage: bench-core.sh [24|100|all]   (default all)
# One process per case, so maxrss_mb is that case's peak. Linux runs under nice -n 10.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
fm="-ffast-math -fno-finite-math-only"
dir=build-ekvido-tests-fm
cmake -S src/common/ekvido/tests -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_C_FLAGS=$fm" "-DCMAKE_CXX_FLAGS=$fm" >/dev/null
cmake --build "$dir" --target ekvido_core_bench >/dev/null
which=${1:-all}
sizes=()
case "$which" in
  24) sizes=("6016 4016") ;;
  100) sizes=("11648 8736") ;;
  all) sizes=("6016 4016" "11648 8736") ;;
  *) echo "usage: $0 [24|100|all]" >&2; exit 2 ;;
esac
run=()
[ "$(uname -s)" = Linux ] && run=(nice -n 10)
echo "# $(uname -n) $(git rev-parse --short HEAD) $(date '+%Y-%m-%d %H:%M')"
for s in "${sizes[@]}"; do
  for c in kz-export kz-shape kz-colour kz-strength mist-export mist-shape; do
    # shellcheck disable=SC2086
    ${run[@]+"${run[@]}"} "./$dir/ekvido_core_bench" "$c" $s
  done
done
