"""Result latency per producer kind from a VCS commit trace (.out): for each instruction whose source operand was
written by the instruction right before it, the retire gap is that producer's latency to a dependent consumer
(an L1 miss in between is excluded by needing the consumer to be the very next instruction). Operands come from
the disassembly (the trace's R/W fields name FP registers by their integer number). perf_model_plan.md 13.12.

Usage: python3 tools/commit_latency.py <test>.out
"""
import re
import sys
from collections import Counter, defaultdict

LINE = re.compile(r"^C0:\s+(\d+)\s.*inst=\[([0-9a-f]+)\]\s+(\S+)\s*(.*)$")
STORES = re.compile(r"^(c\.)?f?s[bhwd](sp)?$")


def family(mn):
    m = mn.split(".")[0] if not mn.startswith("c.") else mn
    if m in ("fmadd", "fmsub", "fnmadd", "fnmsub", "fadd", "fsub", "fmul"):
        return "fp-fma"
    if m in ("fdiv",):
        return "fdiv"
    if m in ("fsqrt",):
        return "fsqrt"
    if m in ("fcvt", "fmv", "fsgnj", "fsgnjn", "fsgnjx", "fmin", "fmax", "feq", "flt", "fle", "fclass"):
        return "fp-misc"
    if m in ("flw", "fld", "c.fld", "c.fldsp", "c.flw"):
        return "fp-load"
    if m in ("ld", "lw", "lwu", "lh", "lhu", "lb", "lbu", "c.ld", "c.lw", "c.ldsp", "c.lwsp"):
        return "load"
    if m in ("mul", "mulw", "mulh", "mulhu", "mulhsu"):
        return "mul"
    if m in ("div", "divu", "divw", "divuw", "rem", "remu", "remw", "remuw"):
        return "div"
    return "alu"


def operands(mn, ops):
    regs = re.findall(r"\b([xaftsr][0-9a-z]{1,3}|zero|ra|sp|gp|tp|fp)\b", ops)
    if not regs:
        return None, []
    if STORES.match(mn) or mn.startswith(("b", "c.b")):   # stores / branches: no destination
        return None, regs
    return regs[0], regs[1:]


def main():
    rows = []
    for ln in open(sys.argv[1], errors="replace"):
        m = LINE.match(ln)
        if m:
            rows.append((int(m.group(1)), m.group(3), m.group(4)))
    lat = defaultdict(Counter)
    for i in range(1, len(rows)):
        c0, mn0, op0 = rows[i - 1]
        c1, mn1, op1 = rows[i]
        d0, _ = operands(mn0, op0)
        _, s1 = operands(mn1, op1)
        if d0 and d0 in s1 and d0 not in ("zero", "x0"):
            lat[family(mn0)][c1 - c0] += 1
    for k in sorted(lat, key=lambda k: -sum(lat[k].values())):
        h = lat[k]
        print("%-8s n=%6d  gap hist %s" % (k, sum(h.values()), sorted(h.items(), key=lambda x: -x[1])[:8]))


if __name__ == "__main__":
    main()
