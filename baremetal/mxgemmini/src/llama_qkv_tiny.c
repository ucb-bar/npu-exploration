// Isolated repro of the QKV-projection regression (K/V each get exactly one output row wrong while
// Q is bit-exact) — the phase-1 Q/K/V projections of `llama_attention_tiny`, with everything from
// RoPE onward compiled out via LLAMA_QKV_ONLY. Three back-to-back BF16 LUT matmuls, same A (Xn),
// different B (WQ/WK/WV), scales reloaded per matmul — the minimal shape that carries the
// cross-matmul residual. Much cheaper than the full head (3 small matmuls, no scores/softmax/P@V/
// o_proj), so it iterates fast in VCS while still failing identically.
//
// Expected on a good build (e.g. commit 6eb1330): Q, K, V all 0/2048 differ -> PASSED.
// Expected on the regressed build (46736b1..HEAD): Q 0/2048, K 64/2048, V 64/2048 -> FAILED.
#define LLAMA_QKV_ONLY
#define LLAMA_ATTN_HEADER "llama_attn_tiny.h"
#include "llama_attention.c"
