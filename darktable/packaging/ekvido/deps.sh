#!/usr/bin/env bash
# Install darktable build dependencies for this OS (never upgrades unrelated packages).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
case "$(uname -s)" in
  Darwin)
    brew bundle install --file .ci/Brewfile --no-upgrade
    ;;
  Linux)
    pkgs=()
    while read -r p; do
      [ -z "$p" ] && continue
      if apt-cache show "$p" >/dev/null 2>&1; then pkgs+=("$p"); else echo "deps: skipping $p (not in this release)"; fi
    done < packaging/ekvido/ubuntu-24.04.txt
    sudo apt-get install -y "${pkgs[@]}"
    ;;
  MINGW64*|MSYS*)   # MSYS2 UCRT64 shell (see win.cmd)
    mapfile -t pkgs < <(grep -v '^$' packaging/ekvido/msys2-ucrt64.txt | tr -d '\r')
    pacman -S --needed --noconfirm "${pkgs[@]}"
    ;;
  *) echo "deps: unsupported OS $(uname -s)"; exit 1 ;;
esac
