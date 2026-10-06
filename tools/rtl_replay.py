"""Extract the host-side timing of a VCS run for the perf model's replay mode.

The VCS commit trace (sims/vcs/output/<cfg>/<test>.out, one line per retired instruction: `C0: <cycle> ... inst=[hex]`)
gives the exact cycle each Gemmini command, fence and rdcycle retired on the real core. With
GEMMINI_PERF_REPLAY=<this file>, the perf model takes those arrival times instead of its own host estimate, so
the comparison measures the accelerator alone (perf_model_plan.md section 13); each fence's RTL retire cycle is
then a direct measurement of when Gemmini went idle.

Output: one line per event, `<kind> <cycle>`, kind in rocc | fence | rdcycle, in program order.
Usage: python3 tools/rtl_replay.py <test>.out > <test>.replay
"""
import re
import sys

LINE = re.compile(r"^C0:\s+(\d+)\s.*inst=\[([0-9a-f]+)\]")


def main(path):
    out = sys.stdout
    n = {"rocc": 0, "fence": 0, "rdcycle": 0}
    with open(path, errors="replace") as f:
        for ln in f:
            m = LINE.match(ln)
            if not m:
                continue
            cyc, inst = int(m.group(1)), int(m.group(2), 16)
            if inst & 0x3 != 0x3:          # compressed: never one of ours
                continue
            if inst & 0x7F == 0x7B:        # custom-3: Gemmini RoCC
                kind = "rocc"
            elif inst & 0x707F == 0x000F:  # FENCE (not fence.i)
                kind = "fence"
            elif inst & 0xFFFFF07F == 0xC0002073:   # csrrs rd, cycle, x0
                kind = "rdcycle"
            else:
                continue
            n[kind] += 1
            out.write("%s %d\n" % (kind, cyc))
    sys.stderr.write("rtl_replay: %s\n" % n)


if __name__ == "__main__":
    main(sys.argv[1])
