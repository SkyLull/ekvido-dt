#!/usr/bin/env bash
# CI: check a built darktable before it is released.
# usage: ci-verify.sh <libdarktable> <expected version> <source dir> -- <darktable-cli command...>
#  - `darktable-cli --version` reports exactly "darktable <expected version>"
#  - on x86: no AVX-512 (zmm) outside the compiler's unused AVX-512 `declare simd` copies
#    (_ZGVe*: callers built for x86-64-v3 only ever call the AVX2 _ZGVd* copies)
#    and outside target_clones copies (*.avx512f etc.), which run only when the CPU has AVX-512
#  - the source dir (build machine path) does not appear in the library, except in clang's
#    OpenMP location records, which -ffile-prefix-map cannot reach
set -euo pipefail
lib=$1; ver=$2; src=$3; shift 3
[ "${1:-}" = -- ] && shift
out=$("$@" --version 2>/dev/null | head -n1 | tr -d '\r')
if [ "$out" != "darktable $ver" ]; then
  echo "verify: version is '$out', expected 'darktable $ver'"
  exit 1
fi
case "$(uname -m)" in
  x86_64|amd64)
    if objdump -h "$lib" >/dev/null 2>&1; then
      bad=$(objdump -d "$lib" | awk '/^[0-9a-f]+ <.*>:/{fn=$2} /zmm[0-9]/{print fn}' \
            | sort -u | grep -v -E '^<_ZGVe|[.]avx512[a-z0-9_]*>:$' || true)
      if [ -n "$bad" ]; then
        echo "verify: AVX-512 outside _ZGVe variants:"
        echo "$bad"
        exit 1
      fi
    fi ;;
esac
# clang/libomp OpenMP location records (";file;function;line;col;;") ignore -ffile-prefix-map
paths() { strings "$lib" | grep -F "$src" | grep -v -E '^;[^;]+;[^;]*;[0-9]+;[0-9]+;;$' || true; }
n=$(paths | grep -c . || true)
if [ "$n" != 0 ]; then
  echo "verify: $n build-path strings ($src) in $lib"
  paths | head -5
  exit 1
fi
echo "verify: ok ($out)"
