// NO MESH AT ALL: mx_host.h's fp32 glue -- RMSNorm, E8M0 block quantize, causal
// softmax (the one expf call), RoPE -- against goldens generated the same way the llama headers
// are. Every one of these feeds the mesh in the real kernel, so if the host's codes are wrong on
// the FPGA then every mesh stage differs for a reason that has nothing to do with the mesh.
//
// Generated data: data/mx_ladder_mxl9.h (gen/gen_mx_ladder.py). Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl9.h"
#define LADDER_KIND   KIND_HOST
#include "mx_ladder.c"
