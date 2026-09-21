// requant CONTROL: I=J=Kt=4 -- matmul_tiled_fp8_64x64_chain's own shape, through this
// ladder's helpers. MUST pass: if it does not, the fault is in the helpers or the driver and mxl7
// says nothing about the hardware.
//
// Third-tier rung: added after mxl7 FAILED with `0/2048 codes, 36/64 scales` -- the requantizer
// computes the right FP8 values and writes the wrong E8M0 bytes. The chain test PASSES on the same
// bitstream and checks those scales the same way, so what matters is the tile counts, not the
// layout. gen/gen_mx_ladder.py carries the table.
//
// Generated data: data/mx_ladder_mxl15.h. Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl15.h"
#define LADDER_KIND   KIND_REQUANT
#include "mx_ladder.c"
