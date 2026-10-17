#!/usr/bin/env bash
# Prepare .dt/darktable: upstream darktable at the tag pinned in darktable/UPSTREAM,
# with every file under darktable/ written over it (only where content differs, so
# CMake stays incremental). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
up=darktable/UPSTREAM
val() { sed -n "s/^$1 = //p" "$up" | tr -d '\r'; }
repo=$(val repo); tag=$(val tag); commit=$(val commit); deleted=$(val deleted)
tree=.dt/darktable

if [ -d "$tree/.git" ] && [ "$(git -C "$tree" rev-parse HEAD)" != "$commit" ]; then
  echo "inject: $tree is not at $tag; cloning again"
  rm -rf "$tree"
fi
if [ ! -d "$tree/.git" ]; then
  rm -rf "$tree"
  mkdir -p .dt
  git clone --depth 1 --branch "$tag" "$repo" "$tree"
fi
# Every run, so an interrupted first init is finished (a no-op once done).
if [ -f "$tree/.gitmodules" ]; then
  subs=$(git -C "$tree" config -f .gitmodules --get-regexp '\.path$' | awk '{print $2}' \
         | grep -v '^src/tests/integration$' || true)
  if [ -n "$subs" ]; then
    # shellcheck disable=SC2086
    git -C "$tree" submodule update --init --depth 1 -- $subs \
      || git -C "$tree" submodule update --init -- $subs
  fi
fi
if [ "$(git -C "$tree" rev-parse HEAD)" != "$commit" ]; then
  echo "inject: $tag is not $commit (upstream tag moved?)" >&2
  exit 1
fi

# Undo files an earlier run wrote that darktable/ no longer has: restore upstream's version,
# or remove a file upstream never had.
manifest=.dt/overlay.list
current=$(cd darktable && find . -type f ! -path ./UPSTREAM | sed 's|^\./||' | sort)
if [ -f "$manifest" ]; then
  comm -23 "$manifest" <(printf '%s\n' "$current") | while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    if git -C "$tree" ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then
      git -C "$tree" checkout -- "$rel"
    else
      rm -f "${tree:?}/$rel"
    fi
  done
fi

n=0
while IFS= read -r -d '' f; do
  rel=${f#darktable/}
  [ "$rel" = UPSTREAM ] && continue
  dst="$tree/$rel"
  if ! cmp -s "$f" "$dst"; then
    mkdir -p "$(dirname "$dst")"
    cp "$f" "$dst"   # fresh mtime: newer than any build output, so CMake rebuilds it
    if [ -x "$f" ]; then chmod +x "$dst"; fi
    n=$((n + 1))
  fi
done < <(find darktable -type f -print0)
for d in $deleted; do rm -rf "${tree:?}/$d"; done
printf '%s\n' "$current" > "$manifest"
echo "inject: $tree ready at $tag ($n file(s) written)"
