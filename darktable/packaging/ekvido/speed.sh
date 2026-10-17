#!/usr/bin/env bash
# Export timings through darktable-cli: module time from -d perf, wall time and peak RSS.
# usage: speed.sh DARKTABLE_CLI TIF RAF XMP_DIR
#   XMP_DIR holds kz.xmp, mist.xmp (from tools/dt_compare.py --write-xmp) and kz-raw.xmp,
#   mist-raw.xmp (same with iop_order_version="4", for raw files).
#   OPENCL=1 runs darktable with OpenCL on (GPU path); default off.
set -euo pipefail
cli=$1; tif=$2; raf=$3; x=$4
case "$(uname -s)" in
  Darwin) timer=(/usr/bin/time -l); rss='maximum resident set size'; unit=1048576 ;;
  *) timer=(nice -n 10 /usr/bin/time -v); rss='Maximum resident set size'; unit=1024 ;;
esac
log=$(mktemp)
cl=(--disable-opencl)
tag=
[ "${OPENCL:-0}" = 1 ] && { cl=(--conf opencl=TRUE); tag="GPU "; }
for img in "$tif" "$raf"; do
  for mod in kz mist; do
    xmp="$x/$mod.xmp"
    case "$img" in *.RAF|*.raf) xmp="$x/$mod-raw.xmp" ;; esac
    for size in "0 0 true" "2560 1440 false"; do
      set -- $size
      start=$(date +%s.%N 2>/dev/null || date +%s)
      "${timer[@]}" "$cli" "$img" "$xmp" "$x/speed.jpg" --width "$1" --height "$2" --hq "$3" \
        --apply-custom-presets false --core --configdir "$(mktemp -d)" "${cl[@]}" -d perf \
        --conf plugins/imageio/storage/disk/overwrite=1 >"$log" 2>&1
      end=$(date +%s.%N 2>/dev/null || date +%s)
      module=$(grep -E "processed .ekvido(kazuki|mist)" "$log" | grep -oE "took [0-9.]+ secs" | head -1)
      peak=$(grep -F "$rss" "$log" | grep -oE "[0-9]+" | head -1)
      printf '%s%s %s %sx%s: module %s, wall %.1f s, peak %.0f MB\n' "$tag" "$(basename "$img")" "$mod" "$1" "$2" \
        "${module:-n/a}" "$(echo "$end - $start" | bc)" "$(echo "${peak:-0} / $unit" | bc)"
    done
  done
done
rm -f "$log"
