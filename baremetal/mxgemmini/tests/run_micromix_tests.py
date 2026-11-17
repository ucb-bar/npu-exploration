#!/usr/bin/env python3
"""Portable host tests. Device commands are mocked in prepared, layer, profile and decode-row tests.
The optional Spike test executes actual projection commands independently.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from audit_micromix_port import function_body_source

HERE = Path(__file__).resolve().parent
K = HERE.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc", default=os.environ.get("HOST_CC", "cc"))
    p.add_argument("--out", type=Path, default=K / "build/micromix-host")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    canonical = (K / "include/mx_micromix.h").read_text()
    legal = (
        "static inline int mxmm_legal(int m, const mxmm_weight *w, "
        "mxmm_schedule s, int *nc_out) " + function_body_source(canonical, "mxmm_legal")
    )
    shim = (
        """#ifndef MXMM_TEST_BACKEND_H
#define MXMM_TEST_BACKEND_H
#define MX_MICROMIX_H
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include "mxmm_pack2.h"
#include "mx_micromix_formats.h"
#define DIM 16
#define ACC_ROWS 512
#define MXN_HALF_ROWS 8192
typedef struct {int k,n,nb4,nb6,nb8;const uint8_t *b4,*b6,*b8,*scales;}mxmm_weight;
typedef struct {int nc,run,fence_groups;}mxmm_schedule;
static uint64_t mxmm_pack_cycles,mxmm_mesh_cycles,host_ticks;
static unsigned device_calls;
static int mxn_loops;
static int mxmm_round2_pack=0,mxmm_fast_pack=1;
#define mxmm_integer_pack mxmm_round2_pack
static uint64_t mxmm_cycles(void){return ++host_ticks;}
#ifndef MXMM_TEST_FENCE
#define MXMM_TEST_FENCE() ((void)0)
#endif
static void mxmm_device_fence(void) {
  device_calls++;
  MXMM_TEST_FENCE();
}
static size_t mxmm_workspace_bytes(int m,int k){return (size_t)m*(k+k/32);}
#define gemmini_extended3_config_ex(...) (++device_calls)
#define gemmini_extended3_config_ld(...) (++device_calls)
#define gemmini_config_st(...) (++device_calls)
#define gemmini_loop_ws_mx(...) (++device_calls)
#define gemmini_loop_ws_mx_shared(...) (++device_calls)
"""
        + legal
        + "\n#endif\n"
    )
    (a.out / "host_shim.h").write_text(shim)
    results = []
    for mode, flags in [
        ("strict", ["-fno-fast-math", "-ffp-contract=off"]),
        ("prefill", ["-ffast-math"]),
    ]:
        for name in ("formats", "prepared", "profiles", "layer", "decode_rows"):
            binary = a.out / (name + "-" + mode)
            cmd = [
                a.cc,
                "-std=gnu11",
                "-O2",
                *flags,
                "-I" + str(a.out),
                "-I" + str(K / "include"),
                str(HERE / ("test_micromix_" + name + ".c")),
                *([str(HERE / "test_micromix_profiles_other.c")] if name == "profiles" else []),
                "-lm",
                "-o",
                str(binary),
            ]
            subprocess.run(cmd, check=True)
            run = subprocess.run([str(binary)], text=True, stdout=subprocess.PIPE, check=True)
            print(run.stdout, end="")
            results.append(dict(test=name, mode=mode, command=cmd, stdout=run.stdout))
    (a.out / "results.json").write_text(
        json.dumps(
            dict(
                pass_=True,
                native_device_calls_mocked_for=["prepared", "layer", "profiles", "decode_rows"],
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
