// Smaller scoped repro: same as llama_two_mm but on a D=32 slice (half the contraction) -- two
// back-to-back BF16 LUT matmuls Q=Xn@WQ then K=Xn@WK reusing SPAD_QKV, host RMSNorm skipped.
// Smallest matmul the D-slice generator allows (M=32 tokens, H=64 head_dim are model-fixed).
// Expected: Q 0/2048, K row 0 wrong -> FAIL, same bug, fewer cycles / signals to trace.
#define LLAMA_QKV_ONLY
#define LLAMA_NPROJ 2
#define LLAMA_SKIP_RMSNORM
#define LLAMA_ATTN_HEADER "llama_attnmicro.h"
#include "llama_attention.c"
