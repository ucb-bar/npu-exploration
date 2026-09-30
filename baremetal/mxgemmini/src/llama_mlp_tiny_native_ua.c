// A/B of llama_mlp_tiny_native with the UNALIGNED (pre-fix) header layout: isolates weight alignment
// from the address shift the aligned(64) fix caused.
#define MLP_NATIVE 1
#define LLAMA_MLP_HEADER "llama_mlp_tiny_ua.h"
#include "llama_mlp.c"
