// mxl4's matmul split into two accumulating K-tiles of 1024 (ex_accumulate = 0 on
// the first, 1 on the second), graded against mxl4's golden. planning/llama_layer_hw_plan.md 10.3
// records that the ex_accumulate fix was made to the FUNCTIONAL MODEL and never verified on RTL.
// This is also the schedule a BANK_ROWS=2048 build picks for every projection, so it decides
// whether such a build can work at all.
//
// Generated data: data/mx_ladder_mxl5.h (gen/gen_mx_ladder.py). Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl5.h"
#define LADDER_KIND   KIND_KTILE
#include "mx_ladder.c"
