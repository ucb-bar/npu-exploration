// `llama_mlp.c` at D = 64 -- the smallest hidden size the scratchpad map and the E8M0 block size
// allow (D must be a multiple of 32 for the blocks and of DIM for the mvin tiles).
//
// This is the bisection variant: two E8M0 blocks per row, one K-tile, one output chunk, and a
// 0.2 MB header whose every array can be read by eye. If `llama_mlp_small` fails on hardware and
// this passes, the fault scales with D; if both fail identically, it does not.
//
// Everything `llama_mlp_small.c` says about what the slice costs applies here more strongly.
//
//   cd gen && ../../../.venv/bin/python3 gen_llama_layer.py mlp --d 64 --tag _tiny
#define LLAMA_MLP_HEADER "llama_mlp_tiny.h"
#include "llama_mlp.c"
