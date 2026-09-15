// The same real TinyLlama attention head as `llama_attention.c`, on a hidden size sliced to 256.
//
// WHY THIS EXISTS. `llama_attention` at the captured D = 2048 spends 12.6 M cycles on the host's
// fp32 glue and 15 K on the mesh (planning/llama_layer_hw_plan.md section 8.3). On spike that is
// seconds; on an RTL simulator at ~5 K cycles/s it is hours, and it overruns the +max-cycles=10000000
// that run_mx_vcs.sh passes. The host cost is linear in D, so a 256-wide slice is ~8x less
// simulation while keeping every structural feature of the full kernel: six mesh matmuls, the
// resident P@V -> o_proj seam, both o_proj N-chunks, and all four operand scale layouts.
//
// WHAT THE SLICE COSTS, stated rather than discovered later. The full-D kernel already grades
// PARTIAL SUMS -- one of 32 heads through `Wo`. Cutting D extends that to the input side: RMSNorm
// now normalizes over 256 features instead of 2048, so `xn` is not the real llama normalized
// activation, and Q/K/V are partial sums over 256 of 2048 input features. Every OPERAND is still a
// real llama value at its real index, and `gen_llama_layer.slice_capture` RECOMPUTES the fp32
// reference from the sliced tensors rather than slicing the stored full-D one -- so REF_ATTN grades
// exactly what the device computes here. The mesh goldens are bit-exact either way, which is what
// this kernel is actually gating on.
//
// Regenerate the header with:
//   PATH=../../npu-exploration/.venv/bin:$PATH ../../npu-exploration/.venv/bin/python3 \
//       gen_llama_layer.py attn --d 256 --tag _small
#define LLAMA_ATTN_HEADER "llama_attn_small.h"
#include "llama_attention.c"
