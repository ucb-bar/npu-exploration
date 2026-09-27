// Isolated single-projection kernel: runs ONLY the Q = Xn @ WQ projection (as a standalone
// FIRST matmul), reusing llama_attention_tiny's setup. If Q/K/V each PASS here but K/V fail when run
// back-to-back (llama_qkv_tiny), the bug is purely cross-matmul state, not the data or the weights.
#define LLAMA_QKV_ONLY
#define LLAMA_ONLY_PROJ 0
#define LLAMA_ATTN_HEADER "llama_attn_tiny.h"
#include "llama_attention.c"
