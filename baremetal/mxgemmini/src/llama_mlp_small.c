// The same real TinyLlama MLP as `llama_mlp.c`, on a hidden size sliced to 256.
//
// The MLP had no sliced variant at any size, unlike attention. This is the counterpart of
// `llama_attention_small.c`, and it exists for the same two reasons: an RTL run at the captured
// D = 2048 is dominated by the host's fp32 glue (planning/llama_layer_hw_plan.md 8.3), and a
// header small enough to read is what makes a hardware mismatch tractable.
//
// WHAT THE SLICE COSTS. RMSNorm now normalizes over 256 features instead of 2048, so `xn` is not
// the real llama normalized activation, and `gate`/`up` become partial sums over 256 of 2048 input
// features -- on top of `down` already being a partial sum over 64 of 5632 neurons. Every OPERAND
// is still a real llama value at its real index, and `gen_llama_layer.slice_capture` RECOMPUTES
// the fp32 reference from the sliced tensors rather than slicing the stored full-D one, so the
// grade describes exactly what the device computes. The mesh goldens are bit-exact either way,
// which is what this kernel actually gates on.
//
// Regenerate the header with:
//   cd gen && ../../../.venv/bin/python3 gen_llama_layer.py mlp --d 256 --tag _small
#define LLAMA_MLP_HEADER "llama_mlp_small.h"
#include "llama_mlp.c"
