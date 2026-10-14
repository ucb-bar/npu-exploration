#!/usr/bin/env python3
"""Rebuild the GPU part of a Radiance VCS run's ELF from the run's objdump (<test>.soc.dump), for cyclotron.

VCS keeps only the .dump of the ELF it ran; the kernel is often rebuilt since (different code / data layout), so
timing comparisons need the exact binary back. The .dump is `objdump -D` of every section: each line of the
.rv32.seg* sections carries its raw bytes (a 32-bit word, or 16-bit groups for long instructions and their
continuation lines; little-endian values), `...` elides zeros. The segment bounds come from cyclotron's copy_elf
lines in the run's .log ("relocating section [a..b] to [c..d]").

objdump cannot print the last bytes of a segment when it tries to decode an instruction running past the end
("Address ... is out of bounds"). Those bytes are taken from a --fill file (e.g. the kernel's regenerated
data/<name>.region<N>.bin, its generator being deterministic); every byte the dump does show must match it.

Output: an ELF32 (RISC-V) with one PROGBITS section per segment at its GPU address (the relocated range), which
cyclotron loads like a *.radiance.elf.

  radiance_elf_from_dump.py <test>.soc.dump <test>.soc.log out.elf [--fill 0x20000000=data/fa.region1.bin ...]
  radiance_elf_from_dump.py --check some.soc.elf      # self-test: objdump it, rebuild, compare the segments
"""
import re
import struct
import subprocess
import sys

LINE = re.compile(r"^\s*([0-9a-f]+):\t([0-9a-f ]+?)\s*(?:\t|$)")
OOB = re.compile(r"^\s*([0-9a-f]+):\tAddress 0x[0-9a-f]+ is out of bounds")
SECTION = re.compile(r"^Disassembly of section (\S+):")
START = re.compile(r"^([0-9a-f]{8,16}) <")
RELOC = re.compile(r"relocating section \[([0-9a-f]+)\.\.([0-9a-f]+)\] to \[([0-9a-f]+)\.\.([0-9a-f]+)\]")


def parse_dump(path):
    """{section start: ({address: byte}, first unprintable address or None)} for the .rv32.seg* sections. `...`
    (elided zeros) and an all-zero section leave no bytes (zeros)."""
    segs, cur, in_seg = {}, None, False
    with open(path) as f:
        for ln in f:
            m = SECTION.match(ln)
            if m:
                in_seg, cur = m.group(1).startswith(".rv32.seg"), None
                continue
            if not in_seg:
                continue
            m = START.match(ln)
            if m and cur is None:
                cur = int(m.group(1), 16)
                segs[cur] = ({}, None)
                continue
            if cur is None:
                continue
            m = OOB.match(ln)
            if m:
                segs[cur] = (segs[cur][0], int(m.group(1), 16))
                continue
            m = LINE.match(ln)
            if not m:
                continue
            addr = int(m.group(1), 16)
            for tok in m.group(2).split():
                n = len(tok) // 2
                v = int(tok, 16)
                for i in range(n):
                    segs[cur][0][addr + i] = (v >> (8 * i)) & 0xFF
                addr += n
    return segs


def segment_bounds(log):
    """[(cpu_start, cpu_end, gpu_start)] from cyclotron's copy_elf lines (first occurrence of each)"""
    out = {}
    with open(log, errors="replace") as f:
        for ln in f:
            m = RELOC.search(ln)
            if m:
                a, b, c, _ = (int(x, 16) for x in m.groups())
                out.setdefault(a, (a, b, c))
    return sorted(out.values())


def build_images(segs, bounds, fills=None):
    """[(gpu_start, bytes)] -- zeros where the dump elided them; an unprintable tail from fills[gpu or cpu start]"""
    fills = fills or {}
    images = []
    for a, b, c in bounds:
        if a not in segs:
            raise SystemExit(f"segment [{a:x}..{b:x}] not in the dump")
        bytes_, oob = segs[a]
        img = bytearray(b - a)
        for addr, byte in bytes_.items():
            if a <= addr < b:
                img[addr - a] = byte
        fill = fills.get(c, fills.get(a))
        if fill is not None:
            if len(fill) < b - a:
                raise SystemExit(f"--fill for [{a:x}..{b:x}] has {len(fill)} bytes, need {b - a}")
            stop = (oob - a) if oob is not None else b - a
            bad = sum(1 for i in range(stop) if img[i] != fill[i])
            if bad:
                raise SystemExit(f"--fill for [{a:x}..{b:x}] differs from the dump in {bad} bytes: not this run's data")
            if oob is not None:
                img[stop:] = fill[stop:b - a]
        elif oob is not None:
            raise SystemExit(f"segment [{a:x}..{b:x}]: bytes from {oob:x} are not in the dump; give --fill "
                             f"0x{c:x}=<file with the segment's bytes>")
        images.append((c, bytes(img)))
    return images


def write_elf32(path, images, entry=0x10000000):
    names = [f".seg{i}" for i in range(len(images))]
    shstr = b"\0" + b"".join(n.encode() + b"\0" for n in names) + b".shstrtab\0"
    off = 52
    data_offs = []
    blob = bytearray()
    for _, img in images:
        data_offs.append(off + len(blob))
        blob += img
        blob += b"\0" * (-len(blob) % 4)
    shstr_off = off + len(blob)
    blob += shstr
    blob += b"\0" * (-len(blob) % 4)
    shoff = off + len(blob)
    shnum = len(images) + 2
    hdr = b"\x7fELF" + bytes([1, 1, 1, 0]) + b"\0" * 8
    hdr += struct.pack("<HHIIIIIHHHHHH", 2, 243, 1, entry, 0, shoff, 0, 52, 0, 0, 40, shnum, shnum - 1)
    sh = struct.pack("<10I", *([0] * 10))
    name_off = 1
    for (addr, img), name, doff in zip(images, names, data_offs):
        sh += struct.pack("<10I", name_off, 1, 0x7, addr, doff, len(img), 0, 0, 4, 0)   # PROGBITS, WAX
        name_off += len(name) + 1
    sh += struct.pack("<10I", name_off, 3, 0, 0, shstr_off, len(shstr), 0, 0, 1, 0)
    with open(path, "wb") as f:
        f.write(hdr + blob + sh)


def check(elf):
    """objdump an ELF the way the VCS flow does, rebuild it (unprintable tails filled from the ELF itself, which
    also checks every printed byte), compare the rv32 segments byte for byte"""
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        dump = os.path.join(d, "x.dump")
        with open(dump, "w") as f:
            subprocess.run([os.environ.get("OBJDUMP", "riscv64-unknown-elf-objdump"), "-D", elf], stdout=f, check=True)
        segs = parse_dump(dump)
        out = subprocess.run(["riscv64-unknown-elf-readelf", "-SW", elf], capture_output=True, text=True).stdout
        bad = 0
        for ln in out.splitlines():
            m = re.search(r"(\.rv32\.seg\d+)\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)", ln)
            if not m:
                continue
            name, addr, offset, size = m.group(1), int(m.group(2), 16), int(m.group(3), 16), int(m.group(4), 16)
            with open(elf, "rb") as f:
                f.seek(offset)
                real = f.read(size)
            img = build_images(segs, [(addr, addr + size, addr)], {addr: real})[0][1]
            same = img == real
            bad += not same
            oob = segs[addr][1]
            tail = f" ({addr + size - oob} tail bytes from --fill)" if oob is not None else ""
            print(f"{name} [{addr:x}..{addr + size:x}]: {'identical' if same else 'DIFFERENT'}{tail}")
        return bad == 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--check":
        sys.exit(0 if check(args[1]) else 1)
    fills = {}
    while "--fill" in args:
        i = args.index("--fill")
        addr, path = args[i + 1].split("=", 1)
        fills[int(addr, 0)] = open(path, "rb").read()
        del args[i:i + 2]
    if len(args) != 3:
        sys.exit(__doc__)
    dump, log, out = args
    images = build_images(parse_dump(dump), segment_bounds(log), fills)
    write_elf32(out, images)
    for addr, img in images:
        print(f"[{addr:x}..{addr + len(img):x}] {len(img)} bytes")
    print(f"wrote {out}")
