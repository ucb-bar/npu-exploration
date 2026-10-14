#!/usr/bin/env python3
"""Muon timing: an RTL VCS trace (sims/vcs/<test>.soc.sqlite, the Trace configs' cyclotron tracer) against
cyclotron's own (--timing, [sim] trace = true), on the same binary. Both are sqlite DBs with
inst(cycle, core_id, warp, pc, rs1_id, rs2_id, ...) and smem / dmem(req_cycle, resp_cycle, ...).

  radiance_trace_compare.py latency <trace.sqlite> <elf> [<trace2.sqlite> ...]
      Per instruction class: the cycles from an instruction to the next instruction of its warp when that one
      reads its result (dependent pair -- architectural registers from the encoding: the RTL trace's ids are physical, after
      renaming -- the result latency as the pipeline exposes it), and when it does not
      (independent pair: issue spacing). Other warps sharing the core inflate some samples; min / p25 / p50 shown.
  radiance_trace_compare.py align <rtl.sqlite> <model.sqlite> <elf>
      Per warp: the k-th barrier (nu.invoke sync) and Gemmini register store (sw.shared to 0x84000-0x84fff) in each
      trace, relative to the kernel's START stamp (printBuf slot 2 + core), and the model's drift.

Instructions are classified from the raw 64-bit Muon encoding in the ELF (cyclotron muon/decode.rs: opcode[6:0],
opext[8:7], rd[16:9], f3[19:17], rs1[27:20], rs2[35:28], rs3[43:36], f7[58:52]).
"""
import sqlite3
import struct
import sys
from collections import defaultdict

import numpy as np


def elf_words(path):
    """{GPU address: 64-bit instruction word} over every allocated section of an ELF32 or ELF64. A CPU-fused (.soc)
    ELF's GPU sections sit at 0x1_0000_0000 + GPU address: relocated as cyclotron's copy_elf does."""
    b = open(path, "rb").read()
    is64 = b[4] == 2
    if is64:
        shoff, = struct.unpack_from("<Q", b, 40)
        shentsize, shnum = struct.unpack_from("<HH", b, 58)
    else:
        shoff, = struct.unpack_from("<I", b, 32)
        shentsize, shnum = struct.unpack_from("<HH", b, 46)
    words = {}
    for i in range(shnum):
        o = shoff + i * shentsize
        if is64:
            _, typ, _, addr, off, size = struct.unpack_from("<IIQQQQ", b, o)
        else:
            _, typ, _, addr, off, size = struct.unpack_from("<6I", b, o)
        if typ != 1 or addr == 0:
            continue
        if 0x100000000 <= addr < 0x200000000:
            addr -= 0x100000000
        for k in range(0, size - 7, 8):
            words[addr + k] = struct.unpack_from("<Q", b, off + k)[0]
    return words


def classify(w):
    """(class, rd or None)"""
    f = lambda hi, lo: (w >> lo) & ((1 << (hi - lo + 1)) - 1)
    op, opext, rd, f3, f7 = f(6, 0), f(8, 7), f(16, 9), f(19, 17), f(58, 52)
    space = "shared" if opext == 1 else "global"
    if op == 0x03:
        return "load." + space, rd
    if op == 0x23:
        return "store." + space, None
    if op in (0x13, 0x17, 0x37):
        return "alu", rd
    if op == 0x33:
        return (("div" if f3 >= 4 else "mul") if f7 == 1 else "alu"), rd
    if op == 0x63:
        return "branch", None
    if op in (0x6f, 0x67):
        return "jump", rd
    if op in (0x53, 0x43, 0x47, 0x4b, 0x4f):
        return ("fp.h" if (f7 & 3) == 2 else "fp.s"), rd
    if op == 0x73:
        return "csr", rd
    if op == 0x0b:
        return "simt", rd          # vx_split / join / pred / tmc / wspawn
    if op == 0x5b:
        return "neutrino", rd      # barriers (vx_bar)
    if op == 0x7b:
        return "sfu", rd           # fexp etc.
    if op == 0x0f:
        return "fence", None
    return "op%02x" % op, rd


def load_insts(db):
    c = sqlite3.connect(db)
    return c.execute("select core_id, warp, cycle, pc, rs1_id, rs2_id, has_rs1, has_rs2 from inst order by id").fetchall()


def latency(dbs, elf):
    words = elf_words(elf)
    for db in dbs:
        rows = load_insts(db)
        dep, ind = defaultdict(list), defaultdict(list)
        last = {}
        for core, warp, cyc, pc, _r1, _r2, h1, h2 in rows:
            key = (core, warp)
            if key in last:
                pcyc, pcls, prd = last[key]
                gap = cyc - pcyc
                # architectural sources from the encoding (the RTL trace's rs ids are physical, after renaming)
                w = words.get(pc, 0)
                reads = {(w >> lo) & 0xFF for lo, h in ((20, h1), (28, h2), (36, True)) if h} - {0}
                if prd and prd in reads:
                    dep[pcls].append(gap)
                else:
                    ind[pcls].append(gap)
            w = words.get(pc, 0)
            cls, rd = classify(w)
            last[key] = (cyc, cls, rd)
        print("==", db)
        print("  %-14s %28s   %28s" % ("class", "dependent: n  min p25 p50", "independent: n  min p25 p50"))
        for cls in sorted(set(dep) | set(ind)):
            def stat(v):
                if not v:
                    return "%28s" % "-"
                a = np.array(v)
                return "%10d %5d %4d %4d" % (len(a), a.min(), np.percentile(a, 25), np.median(a))
            print("  %-14s %28s   %28s" % (cls, stat(dep[cls]), stat(ind[cls])))


def starts(db):
    c = sqlite3.connect(db)
    out = {}
    for core in (0, 1):
        r = c.execute("select data from smem where store=1 and address=? order by id limit 1",
                      (0x80010 + 8 * core,)).fetchone()
        out[core] = r[0] if r else 0
    return out


def landmarks(db, words):
    """{(core, warp): {'bar': [cycles], 'mmio': [cycles]}}"""
    rows = load_insts(db)
    c = sqlite3.connect(db)
    lm = defaultdict(lambda: defaultdict(list))
    for core, warp, cyc, pc, *_ in rows:
        cls, _ = classify(words.get(pc, 0))
        if cls == "neutrino":
            lm[(core, warp)]["bar"].append(cyc)
    for core, req in c.execute("select core_id, req_cycle from smem where store=1 and address between ? and ? "
                               "and lane_id=0 order by id", (0x84000, 0x84fff)):
        lm[(core, 0)]["mmio"].append(req)
    return lm


def align(rtl, model, elf):
    words = elf_words(elf)
    s_r, s_m = starts(rtl), starts(model)
    lr, lm = landmarks(rtl, words), landmarks(model, words)
    print("START: rtl", s_r, "model", s_m)
    for key in sorted(set(lr) | set(lm)):
        core = key[0]
        for kind in ("bar", "mmio"):
            a, b = lr[key][kind], lm[key][kind]
            n = min(len(a), len(b))
            if not n:
                continue
            ra = np.array(a[:n]) - s_r[core]
            mb = np.array(b[:n]) - s_m[core]
            d = mb - ra
            picks = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
            print("core %d warp %d %-4s rtl %4d model %4d | " % (core, key[1], kind, len(a), len(b)) +
                  "  ".join("#%d %d/%d (%+d)" % (i, ra[i], mb[i], d[i]) for i in picks))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "latency":
        latency(sys.argv[2:3] + sys.argv[4:], sys.argv[3])
    elif len(sys.argv) == 5 and sys.argv[1] == "align":
        align(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(__doc__)
