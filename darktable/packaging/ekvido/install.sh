#!/usr/bin/env bash
# Back up darktable's config, then install our build for daily use.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
stamp=$(date +%Y%m%d-%H%M%S)
backups="$HOME/darktable-config-backups"
backup_config() {
  if [ -d "$HOME/.config/darktable" ]; then
    mkdir -p "$backups"
    tar czf "$backups/darktable-config-$stamp.tgz" -C "$HOME/.config" darktable
    echo "install: config backed up to $backups/darktable-config-$stamp.tgz"
  fi
}
case "$(uname -s)" in
  Darwin)
    app=build/macosx/package/darktable.app
    [ -d "$app" ] || { echo "install: run 'mise run package' first"; exit 1; }
    if pgrep -x darktable >/dev/null; then echo "install: quit darktable first"; exit 1; fi
    backup_config
    if [ -d /Applications/darktable.app ]; then
      old=$(defaults read /Applications/darktable.app/Contents/Info.plist CFBundleShortVersionString 2>/dev/null || echo old)
      mv /Applications/darktable.app "$HOME/.Trash/darktable-$old-$stamp.app"
      echo "install: previous app moved to the Trash as darktable-$old-$stamp.app"
    fi
    cp -R "$app" /Applications/darktable.app
    xattr -dr com.apple.quarantine /Applications/darktable.app 2>/dev/null || true
    echo "install: /Applications/darktable.app is the ekvido build"
    ;;
  Linux)
    prefix="$HOME/.local/opt/darktable-ekvido"
    [ -x "$prefix/bin/darktable" ] || { echo "install: run 'mise run build' first"; exit 1; }
    flatpak_cfg="$HOME/.var/app/org.darktable.Darktable/config/darktable"
    if pgrep -x darktable >/dev/null; then echo "install: quit darktable first"; exit 1; fi
    backup_config
    if [ ! -e "$HOME/.config/darktable/library.db" ] && [ -e "$flatpak_cfg/library.db" ]; then
      if [ -e "$HOME/.config/darktable" ]; then
        mv "$HOME/.config/darktable" "$HOME/.config/darktable.pre-ekvido-$stamp"
        echo "install: moved the library-less ~/.config/darktable to ~/.config/darktable.pre-ekvido-$stamp"
      fi
      mkdir -p "$HOME/.config"
      cp -a "$flatpak_cfg" "$HOME/.config/darktable"
      echo "install: copied the Flatpak's config into ~/.config/darktable (the Flatpak's copy is untouched)"
    fi
    mkdir -p "$HOME/.local/share/applications"
    cat > "$HOME/.local/share/applications/darktable-ekvido.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=darktable (ekvido)
Comment=darktable with the ekvido modules
Exec=$prefix/bin/darktable %U
Icon=$prefix/share/icons/hicolor/scalable/apps/darktable.svg
Categories=Graphics;Photography;
MimeType=image/x-dcraw;image/tiff;image/jpeg;image/png;
EOF
    echo "install: desktop entry 'darktable (ekvido)' written"
    ;;
  MINGW64*|MSYS*)
    # Installs side by side with stock darktable as "darktable-dev" (C:\Program Files\darktable-dev);
    # both share the catalogue in %LOCALAPPDATA%\darktable, which is backed up first.
    setup=$(ls /opt/darktable-ekvido/Output/darktable-*.exe 2>/dev/null | head -1)
    [ -n "$setup" ] || { echo "install: run 'mise run package' first"; exit 1; }
    if tasklist //FI "IMAGENAME eq darktable.exe" //NH | grep -qi darktable.exe; then
      echo "install: quit darktable first"; exit 1
    fi
    cfg_parent=$(cygpath -u "$LOCALAPPDATA")
    if [ -d "$cfg_parent/darktable" ]; then
      mkdir -p "$backups"
      tar czf "$backups/darktable-config-$stamp.tgz" -C "$cfg_parent" darktable
      echo "install: config backed up to $backups/darktable-config-$stamp.tgz"
    fi
    # Remove our previous build silently; otherwise the installer asks in a dialog.
    for hive in HKLM HKCU; do
      old=$(MSYS2_ARG_CONV_EXCL='*' reg query "$hive\Software\Microsoft\Windows\CurrentVersion\Uninstall\darktable-dev_is1" /v UninstallString 2>/dev/null \
            | tr -d '\r' | sed -n 's/.*REG_SZ *//p' | tr -d '"' || true)
      if [ -n "$old" ] && [ -f "$(cygpath -u "$old")" ]; then
        echo "install: removing the previous darktable-dev"
        MSYS2_ARG_CONV_EXCL='*' "$(cygpath -u "$old")" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
        # The uninstaller re-launches itself from %TEMP% and returns at once; wait for it to finish.
        for _ in $(seq 120); do [ -f "$(cygpath -u "$old")" ] || break; sleep 1; done
      fi
    done
    MSYS2_ARG_CONV_EXCL='*' "$setup" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /ALLUSERS
    echo "install: $(basename "$setup") installed - Start menu: darktable-dev"
    ;;
  *) echo "install: unsupported OS $(uname -s)"; exit 1 ;;
esac
