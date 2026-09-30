// Native DRAM-loop schedule of llama_mlp_tiny (see MLP_NATIVE in llama_mlp.c): loop-managed scales,
// no manual mvin/mvout/scale loads. Compare against llama_mlp_tiny / _t1 / _db.
#define MLP_NATIVE 1
#include "llama_mlp_tiny.c"
