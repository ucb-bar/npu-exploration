// llama_layer_full_native with the host stages replaced by their blob goldens (LLAMA_GOLDEN_HOST):
// an RTL run that is mostly mesh time; every mesh stage is still checked bit-exact.
#define LAYER_NATIVE 1
#define LLAMA_GOLDEN_HOST 1
#include "llama_layer_full.c"
