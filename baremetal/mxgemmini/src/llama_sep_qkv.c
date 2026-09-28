// Region-reuse control test: Q/K/V each drain to their OWN spad region (LLAMA_SEPARATE_QKV) instead
// of reusing SPAD_QKV. If K/V now PASS, the bug is the accumulator/output-region reuse not being
// reset between back-to-back matmuls. Skips host RMSNorm for a fast mesh-only run.
#define LLAMA_QKV_ONLY
#define LLAMA_SEPARATE_QKV
#define LLAMA_SKIP_RMSNORM
#define LLAMA_ATTN_HEADER "llama_attn_tiny.h"
#include "llama_attention.c"
