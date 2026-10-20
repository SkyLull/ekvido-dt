#!/usr/bin/env bash
# CI: validate the build number, prepare .dt/darktable (inject), and stamp it so darktable's
# tools/get_git_version_string.sh reports <darktable version>+ekvido.NNNN.
# usage: ci-prepare.sh <tag ref name or ""> [<number, used when there is no tag>]
# prints EKV_NUMBER=… and EKV_VERSION=… (append to $GITHUB_ENV)
set -euo pipefail
cd "$(dirname "$0")/.."
ref=${1:-}; num=${2:-}
dt_tag=$(sed -n 's/^tag = //p' darktable/UPSTREAM | tr -d '\r')
dt_ver=${dt_tag#release-}
if [ -n "$ref" ]; then
  if [[ ! "$ref" =~ ^v(.+)-ekvido\.([0-9]{4})$ ]]; then
    echo "ci-prepare: tag '$ref' is not v<darktable version>-ekvido.NNNN (NNNN = 4 digits)" >&2
    exit 1
  fi
  if [ "${BASH_REMATCH[1]}" != "$dt_ver" ]; then
    echo "ci-prepare: tag says darktable ${BASH_REMATCH[1]} but darktable/UPSTREAM pins $dt_tag" >&2
    exit 1
  fi
  num=${BASH_REMATCH[2]}
fi
if [[ ! "$num" =~ ^[0-9]{4}$ ]]; then
  echo "ci-prepare: build number '$num' is not NNNN (4 digits)" >&2
  exit 1
fi
bash scripts/inject.sh >&2
version="$dt_ver+ekvido.$num"
tree=.dt/darktable
g() { git -C "$tree" -c user.name=ekvido-ci -c user.email=ci@ekvido.invalid \
        -c commit.gpgsign=false -c tag.gpgsign=false "$@"; }
g add -A
g commit -q --allow-empty -m "ekvido $num"
for old in $(g tag -l 'release-*+ekvido.*'); do g tag -d "$old" >/dev/null; done
g tag "release-$version"
echo "EKV_NUMBER=$num"
echo "EKV_VERSION=$version"
