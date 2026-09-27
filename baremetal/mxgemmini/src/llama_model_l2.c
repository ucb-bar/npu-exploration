// The same stacked-model kernel as `llama_model.c`, over the FIRST TWO layers instead of all 22.
//
// It exists for bring-up. The full model carries ~1 GB of weights and ~20 minutes of golden
// generation; this is 169 MB and ~5 minutes, and it exercises every code path the full one uses --
// the layer loop, the uniform-stride addressing, both residual seams carried across a layer
// boundary, and the lm_head. A mistake in any of those shows up here first and far cheaper.
//
// WHAT THE TRUNCATION COSTS, stated plainly because the numbers look like failures: `lm_head` reads
// the hidden state after ALL 22 layers. Applied to layer 1's output it produces well-formed logits
// of nothing, and the perplexity printed is meaningless -- MEASURED at 35026 against the model's
// 28.5089. That is correct behaviour for a truncated stack, not a bug, and it is why the generator
// prints a warning and the generated header repeats it.
//
// WHAT STILL GATES: the bit-exact comparison of every mesh stage, every host stage and both seams
// of both layers against the goldens. Those are computed for the same truncated chain, so they are
// valid, and they are what this build is actually for.
//
// Regenerate the data with:
//   cd gen && ../../../.venv/bin/python3 gen_llama_model.py --layers 2 --tag _l2
#define LLAMA_MODEL_HEADER "llama_model_l2.h"
#include "llama_model.c"
