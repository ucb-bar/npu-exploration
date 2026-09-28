// Minimal scoped repro: TWO back-to-back BF16 LUT matmuls (Q = Xn@WQ, then K = Xn@WK) reusing the
// SAME scratchpad output region (SPAD_QKV). Q comes out bit-exact; K's output row 0 is wrong -- the
// scratchpad/accumulator reuse is not reset between the two matmuls. No scores/softmax/etc, and host
// RMSNorm skipped (golden Xn used directly), so this is the smallest/fastest reproduction.
#define LLAMA_QKV_ONLY
#define LLAMA_NPROJ 2
#define LLAMA_SKIP_RMSNORM
#define LLAMA_ATTN_HEADER "llama_attn_tiny.h"
#include "llama_attention.c"
