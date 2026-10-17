#!/usr/bin/env bash
# Refresh the ekvido translation template and merge it into po/ekvido/*.po.
# Our strings live in po/ekvido/, not in upstream's po/*.po: upstream rewrites those files
# on every translation refresh, so editing them would conflict on each rebase. po/CMakeLists.txt
# lays po/ekvido/<lang>.po over po/<lang>.po (ours win) before msgfmt.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
srcs=(src/iop/ekvidomist.c src/iop/ekvidokazuki.c)
for src in "${srcs[@]}"; do
  out="$tmp/introspection_$(basename "$src")"
  (cd tools/introspection && perl parser.pl ../../src/ "../../$src" "$out")
  # Enum labels carry a "context|label" prefix (read with Q_, see src/common/darktable.h);
  # turn them into real gettext contexts so the .mo key is "context\004label".
  sed -E -i.bak 's/N_\(\(char\*\)"([^"|]+)\|([^"]*)"\)/NC_("\1", "\2")/g' "$out"
  srcs+=("$out")
done
xgettext --from-code=UTF-8 --language=C --no-wrap --sort-by-file \
  --keyword=_ --keyword=N_ --keyword=C_:1c,2 --keyword=NC_:1c,2 --keyword=Q_ \
  --keyword=ngettext:1,2 --add-comments=TRANSLATORS \
  --package-name=darktable-ekvido -o po/ekvido/ekvido.pot "${srcs[@]}"
sed -i.bak -E "s|$tmp/|<introspection>/|" po/ekvido/ekvido.pot && rm -f po/ekvido/ekvido.pot.bak
for po in po/ekvido/*.po; do
  [ -e "$po" ] || continue
  msgmerge --quiet --update --backup=none --no-fuzzy-matching --no-wrap "$po" po/ekvido/ekvido.pot
  echo "l10n: $po: $(msgfmt --statistics -o /dev/null "$po" 2>&1)"
done
