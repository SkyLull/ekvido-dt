#!/usr/bin/env bash
# CI: check a built darktable before it is released.
# usage: ci-verify.sh <expected version> <source dir> <library>... -- <darktable-cli command...>
#  - `darktable-cli --version` reports exactly "darktable <expected version>"
#  - every library (libdarktable and the ekvido plugins) exists, and on x86:
#    no AVX-512 (zmm) outside the compiler's unused AVX-512 `declare simd` copies
#    (_ZGVe*: callers built for x86-64-v3 only ever call the AVX2 _ZGVd* copies)
#    and outside target_clones copies (*.avx512f etc.), which run only when the CPU has AVX-512
#  - the source dir (build machine path) does not appear in any library, except in clang's
#    OpenMP location records, which -ffile-prefix-map cannot reach
set -euo pipefail
ver=$1; src=$2; shift 2
libs=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do libs+=("$1"); shift; done
[ "${1:-}" = -- ] && shift
[ ${#libs[@]} -gt 0 ] || { echo "verify: no libraries given"; exit 1; }

out=$("$@" --version 2>/dev/null | head -n1 | tr -d '\r')
if [ "$out" != "darktable $ver" ]; then
  echo "verify: version is '$out', expected 'darktable $ver'"
  exit 1
fi

x86=0
case "$(uname -m)" in x86_64|amd64) x86=1 ;; esac
if [ $x86 = 1 ] && ! command -v objdump >/dev/null; then
  echo "verify: objdump not found, cannot check for AVX-512"
  exit 1
fi

# clang/libomp OpenMP location records (";file;function;line;col;;") ignore -ffile-prefix-map
paths() { strings "$1" | grep -F "$src" | grep -v -E '^;[^;]+;[^;]*;[0-9]+;[0-9]+;;$' || true; }

for lib in "${libs[@]}"; do
  if [ ! -s "$lib" ]; then
    echo "verify: library missing or empty: '$lib'"
    exit 1
  fi
  if [ $x86 = 1 ]; then
    bad=$(objdump -d "$lib" | awk '/^[0-9a-f]+ <.*>:/{fn=$2} /zmm[0-9]/{print fn}' \
          | sort -u | grep -v -E '^<_ZGVe|[.]avx512[a-z0-9_]*>:$' || true)
    if [ -n "$bad" ]; then
      echo "verify: AVX-512 outside _ZGVe/target_clones variants in $lib:"
      echo "$bad"
      exit 1
    fi
  fi
  n=$(paths "$lib" | grep -c . || true)
  if [ "$n" != 0 ]; then
    echo "verify: $n build-path strings ($src) in $lib"
    paths "$lib" | head -5
    exit 1
  fi
done
echo "verify: ok ($out; ${#libs[@]} libraries)"
