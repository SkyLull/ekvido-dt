#!/usr/bin/env python3
"""Print an ELF64 section's contents as text, NUL bytes dropped (exit 1 if it is missing).

CI uses it to prove an AppImage's `.upd_info` (AppImageUpdate source) is empty:
    python3 scripts/elf-section.py x.AppImage .upd_info
"""

import struct
import sys


def section(path: str, name: str) -> bytes:
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise SystemExit(f"{path}: not a little-endian ELF64 file")
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)

    def header(i):
        # sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size
        return struct.unpack_from("<IIQQQQ", data, shoff + i * shentsize)

    strtab_off, strtab_size = header(shstrndx)[4:6]
    strtab = data[strtab_off : strtab_off + strtab_size]
    for i in range(shnum):
        sh_name, _, _, _, off, size = header(i)
        end = strtab.find(b"\0", sh_name)
        if strtab[sh_name:end].decode() == name:
            return data[off : off + size]
    raise SystemExit(f"{path}: no section {name}")


if __name__ == "__main__":
    sys.stdout.write(section(sys.argv[1], sys.argv[2]).replace(b"\0", b"").decode())
