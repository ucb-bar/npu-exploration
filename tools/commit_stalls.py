"""Per-instruction retire gaps from a VCS commit trace (.out): how long each kind of instruction holds the core.
Used to measure the CPU-model penalties (perf_model_plan.md 13.11): an L1 hit retires 1 cycle after its
predecessor, a miss shows up as a cluster of longer gaps (L2 hit vs DRAM).

Usage: python3 tools/commit_stalls.py <test>.out [--from CYCLE] [--to CYCLE]
"""
import argparse
import re
from collections import Counter, defaultdict

LINE = re.compile(r"^C0:\s+(\d+)\s.*inst=\[([0-9a-f]+)\]\s+(\S+)")
KIND = {"ld": "load", "lw": "load", "lwu": "load", "lh": "load", "lhu": "load", "lb": "load", "lbu": "load",
        "c.ld": "load", "c.lw": "load", "c.ldsp": "load", "c.lwsp": "load", "flw": "fload", "fld": "fload",
        "c.fld": "fload", "c.fldsp": "fload",
        "sd": "store", "sw": "store", "sh": "store", "sb": "store", "c.sd": "store", "c.sw": "store",
        "c.sdsp": "store", "c.swsp": "store", "fsw": "fstore", "fsd": "fstore", "c.fsd": "fstore", "c.fsdsp": "fstore"}


def kind(mn):
    if mn in KIND:
        return KIND[mn]
    if mn.startswith(("fdiv", "fsqrt")):
        return "fdiv/sqrt"
    if mn.startswith(("div", "rem")):
        return "div"
    if mn.startswith(("mul",)):
        return "mul"
    if mn.startswith("f"):
        return "fp"
    if mn.startswith(("b", "c.b", "j", "c.j")):
        return "branch/jump"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--from", dest="lo", type=int, default=0)
    ap.add_argument("--to", dest="hi", type=int, default=1 << 62)
    a = ap.parse_args()
    gaps = defaultdict(Counter)
    prev = None
    n = 0
    for ln in open(a.out, errors="replace"):
        m = LINE.match(ln)
        if not m:
            continue
        c = int(m.group(1))
        if prev is not None and a.lo <= c < a.hi and (int(m.group(2), 16) & 0x7F) != 0x7B:   # skip RoCC (custom-3)
            gaps[kind(m.group(3))][c - prev] += 1
            n += 1
        prev = c
    print("instructions %d" % n)
    for k in sorted(gaps, key=lambda k: -sum(gaps[k].values())):
        h = gaps[k]
        tot = sum(h.values())
        extra = sum((g - 1) * v for g, v in h.items() if g > 1)
        big = sorted(((g, v) for g, v in h.items() if g > 3), key=lambda x: -x[1])[:10]
        print("%-12s n=%7d  gap1 %5.1f%%  extra cycles %8d (%.2f/insn)  common long gaps %s" % (
            k, tot, 100.0 * h.get(1, 0) / tot, extra, extra / tot, sorted(big)))


if __name__ == "__main__":
    main()
